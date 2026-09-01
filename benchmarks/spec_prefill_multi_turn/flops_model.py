#!/usr/bin/env python3
"""Analytic FLOP model for the combined speculator + target system.

Answers the question wall-clock can't: **is the compute the speculator adds
smaller than the compute the target saves?** Every other metric this
pipeline collects (`turns_per_second`, `ttft_*_ms`, `seconds_per_turn_*` in
`results/all_runs.csv`) is wall-clock under `enforce_eager=True` on one GPU,
which conflates the algorithmic cost with dispatch overhead, the absence of
CUDA graphs, and memory-bandwidth stalls -- exactly the confound
`gpu_vs_host_timing.py` exists to separate. FLOPs separate the *algorithmic*
cost from the *implementation* cost, so optimization work can target the
right one.

Pure Python -- no torch, no vLLM import -- so it stays unit-testable in the
same CPU-only environment `test_vllm_patch.py` already runs in. Lives at the
benchmark top level rather than in `vllm_patch/` because it's a measurement
concern, not part of the Algorithm-1 patch.

Mirrors `sparse_decode_microbench.py::bytes_per_token_kv`'s convention:
everything is derived from `hf_config` at runtime, nothing about a specific
checkpoint is hardcoded.

## What is and isn't counted

Counted: every matmul -- QKV/O projections, the SwiGLU MLP, `lm_head`, and
both attention GEMMs (`QK^T` and `AV`), plus the speculator's own scoring
GEMM (`vllm_patch/scoring.py::compute_attention_score`), which is real GPU
work outside both models' forward passes and is invisible to every other
metric collected here.

NOT counted: RMSNorm, RoPE, softmax, residual adds, elementwise activations.
Together these are well under 1% of the total at these shapes. **This is a
matmul-FLOP model, not a total-instruction count** -- don't compare it
against a hardware instruction counter and expect exact agreement.

A multiply-accumulate is 2 FLOPs throughout.

## GQA does not reduce attention FLOPs

`num_kv_heads < num_heads` shrinks the KV *cache* (which is why
`bytes_per_token_kv` scales with `num_kv_heads`) but not the attention
arithmetic: every one of the `num_heads` query heads still does a full
`head_dim` dot product against its group's K and V. So attention terms here
scale with `num_heads`, and only the QKV projection scales with
`num_kv_heads`. Getting this backwards is the easiest way to silently
under-count attention by 4x on Llama-3.1-8B (32 q heads vs. 8 kv heads).

## FLOPs are not the whole story for the SPARSE row -- read them with
## `bytes_per_token_kv`, not instead of it

Decode is memory-bound, not compute-bound (the premise of
`sparse_decode_microbench.py`'s whole roofline analysis). So the SPARSE
path, which leaves target prefill fully dense and only shrinks decode
attention, can legitimately show MORE total FLOPs than M000 -- it adds the
speculator's cost and saves almost nothing on a compute axis that decode
was never limited by. A turn-0 projection at 77k context and 64 output
tokens puts SPARSE at ~120% of M000's FLOPs at every keep rate.

That is a real finding about where SPARSE's saving lives, not a verdict
against it: its saving is in KV bytes read per decode step, which is
`bytes_per_token_kv`'s axis, and in the growing-context multi-turn regime
this pipeline exists to test. Use this model to see what the SPECULATOR
costs and whether SPECPREFILL's prefill trade pays off; use the byte
accounting for whether the sparse decode pays off. Neither alone answers
both questions.

## lm_head is charged per logits row, not per token

vLLM computes logits only for tokens it needs to sample from: one row per
prefill (the last token) and one per decode step -- not for every prefill
token. Charging `lm_head` per prefill token would over-count a 100k-token
prefill by ~100k x 2 x hidden x vocab, which for Llama-3.1-8B is larger
than the entire rest of the prefill.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ModelFlopConfig:
    """Shape parameters needed to count FLOPs, derived from an `hf_config`."""

    num_layers: int
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int

    # -- per-token, position-independent (dense) work ----------------------

    @property
    def qkv_flops_per_token(self) -> int:
        # Q is [hidden -> num_heads*head_dim]; K and V are each
        # [hidden -> num_kv_heads*head_dim]. This is the ONE place
        # num_kv_heads legitimately shrinks the arithmetic.
        return 2 * self.hidden * (self.num_heads + 2 * self.num_kv_heads) * self.head_dim

    @property
    def o_proj_flops_per_token(self) -> int:
        return 2 * self.num_heads * self.head_dim * self.hidden

    @property
    def mlp_flops_per_token(self) -> int:
        # SwiGLU: gate + up (both hidden -> intermediate) and down
        # (intermediate -> hidden) == 3 GEMMs of hidden*intermediate MACs.
        return 6 * self.hidden * self.intermediate

    @property
    def linear_flops_per_token(self) -> int:
        """All non-attention matmul work for one token through every layer.

        Excludes `lm_head` (charged per logits row, not per token -- see the
        module docstring)."""
        return self.num_layers * (
            self.qkv_flops_per_token + self.o_proj_flops_per_token + self.mlp_flops_per_token
        )

    @property
    def lm_head_flops(self) -> int:
        return 2 * self.hidden * self.vocab

    def describe(self) -> dict:
        """JSON-serialisable summary, mirroring `LayeredFlopConfig.describe`
        so a caller can print either without an isinstance check."""
        return {"kind": "ModelFlopConfig", **self.__dict__,
                "linear_flops_per_token": self.linear_flops_per_token}

    # -- attention ---------------------------------------------------------

    @property
    def _attn_flops_per_query_key_pair(self) -> int:
        # 2 GEMMs (QK^T and AV) x 2 FLOPs/MAC x head_dim, per query head.
        return 4 * self.num_heads * self.head_dim

    def attn_prefill_flops(self, n_new: int, n_cached: int = 0) -> int:
        """Causal attention for `n_new` new tokens on top of `n_cached`
        already-resident ones.

        Query token `i` (0-indexed among the new ones) attends
        `n_cached + i + 1` keys, so the total key count is
        `n_new*n_cached + n_new*(n_new+1)/2`.
        """
        if n_new <= 0:
            return 0
        n_cached = max(n_cached, 0)
        key_visits = n_new * n_cached + n_new * (n_new + 1) // 2
        return self.num_layers * self._attn_flops_per_query_key_pair * key_visits

    def attn_decode_step_flops(self, attended_len: int) -> int:
        """One decode step's attention against `attended_len` resident keys."""
        if attended_len <= 0:
            return 0
        return self.num_layers * self._attn_flops_per_query_key_pair * attended_len

    def scoring_flops(self, look_ahead: int, ctx_len: int) -> int:
        """`vllm_patch/scoring.py::compute_attention_score` -- Algorithm 1
        line 12, `Q @ K^T / sqrt(d)` per layer.

        `QK^T` ONLY, no `AV`: the scorer wants the score matrix itself, not
        an attention output, so it never contracts back against V. That's
        why this is `2 * ...` where `attn_*` above is `4 * ...`.

        Tracked as its own line item because it's real GPU work that no
        other metric in this directory can see -- but **measured, it turns
        out to be negligible**, and that's worth knowing before anyone
        spends effort optimizing it. It scales with `look_ahead` (8), not
        with the number of prefilled tokens, so at 77k context on the 1B
        speculator it's ~0.04 TFLOP against ~538 TFLOP for that same
        speculator's prefill: about 0.007% of the turn. The speculator's
        cost is its PREFILL, essentially in full. Keep the column anyway --
        it's what makes that statement a measurement rather than a guess,
        and it would stop being true if `look_ahead_cnt` were ever swept up
        by orders of magnitude.
        """
        if look_ahead <= 0 or ctx_len <= 0:
            return 0
        return self.num_layers * 2 * self.num_heads * self.head_dim * look_ahead * ctx_len


@dataclass(frozen=True)
class LayerFlopSpec:
    """One decoder layer's shape, for models whose layers are not identical.

    Every field was read out of `vllm/model_executor/models/gemma4.py`, not
    inferred from config field names -- see `layered_flop_config` for the
    line-by-line derivation and for the two places a nearby implementation
    (`evaluation_pipeline/hardware_metrics.py`) disagrees with the model
    code.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int
    has_v_proj: bool = True
    sliding_window: Optional[int] = None
    moe_intermediate: int = 0
    num_experts: int = 0
    moe_top_k: int = 0

    def qkv_flops_per_token(self, hidden: int) -> int:
        """Q, K and (usually) V projections.

        `has_v_proj=False` is Gemma 4's `attention_k_eq_v`: the checkpoint
        has no `v_proj` at all and V is derived from the K projection's
        output, so `QKVParallelLinear` is built with `v_head_size=0` and the
        GEMM is genuinely smaller (gemma4.py's `Gemma4Attention.__init__`).

        A KV-SHARED layer is NOT cheaper here, which is easy to assume and
        wrong: `Gemma4Attention.forward` runs `self.qkv_proj(hidden_states)`
        unconditionally and only skips the cache WRITE, so the projection is
        paid in full.
        """
        kv_multiplier = 2 if self.has_v_proj else 1
        return (
            2 * hidden
            * (self.num_heads + kv_multiplier * self.num_kv_heads)
            * self.head_dim
        )

    def o_proj_flops_per_token(self, hidden: int) -> int:
        return 2 * self.num_heads * self.head_dim * hidden

    def mlp_flops_per_token(self, hidden: int) -> int:
        """Dense SwiGLU MLP, plus the routed experts when this layer has them.

        The dense MLP is charged even on MoE layers. `Gemma4DecoderLayer.
        forward` is explicit -- "MLP runs unconditionally (same inputs for
        MoE and non-MoE)" -- and the routed block runs in ADDITION, on the
        residual. Charging only one of the two would understate every MoE
        layer.

        Only `moe_top_k` of `num_experts` are computed per token; the rest
        are resident but untouched. The router itself is a
        `hidden -> num_experts` GEMM.
        """
        dense = 6 * hidden * self.intermediate
        if not (self.num_experts and self.moe_top_k and self.moe_intermediate):
            return dense
        router = 2 * hidden * self.num_experts
        experts = self.moe_top_k * 6 * hidden * self.moe_intermediate
        return dense + router + experts

    @property
    def attn_flops_per_query_key_pair(self) -> int:
        # QK^T and AV: 2 GEMMs x 2 FLOPs/MAC x head_dim, per query head.
        return 4 * self.num_heads * self.head_dim

    def keys_visited_prefill(self, n_new: int, n_cached: int) -> int:
        """How many (query, key) pairs this layer actually evaluates when
        `n_new` tokens are prefilled on top of `n_cached` resident ones.

        Query `i` sits at absolute position `n_cached + i` and attends
        `n_cached + i + 1` keys causally -- but a sliding-window layer
        attends at most `sliding_window` of them. Ignoring that window is
        the single biggest error a flat FLOP model makes on an interleaved
        model: on Gemma-4-31B, 50 of 60 layers are capped at 512 keys while
        a flat model charges them the full context length.

        Closed form rather than a loop over `n_new`, which reaches ~100k
        here.
        """
        if n_new <= 0:
            return 0
        lo = n_cached + 1                 # keys seen by the first new query
        hi = n_cached + n_new             # keys seen by the last new query
        window = self.sliding_window
        if window is None or hi <= window:
            return (lo + hi) * (hi - lo + 1) // 2
        if lo > window:
            return window * (hi - lo + 1)
        # Split: unclipped up to `window`, then clipped.
        return (lo + window) * (window - lo + 1) // 2 + window * (hi - window)

    def keys_visited_decode(self, attended_len: int) -> int:
        if attended_len <= 0:
            return 0
        if self.sliding_window is None:
            return attended_len
        return min(attended_len, self.sliding_window)


@dataclass(frozen=True)
class LayeredFlopConfig:
    """A model whose layers differ -- interleaved attention, MoE, or both.

    Implements exactly the interface `ModelFlopConfig` exposes to the stage
    functions below (`linear_flops_per_token`, `lm_head_flops`,
    `attn_prefill_flops`, `attn_decode_step_flops`, `scoring_flops`), so
    `speculator_turn_flops` / `target_prefill_flops` / `target_decode_flops`
    take either without change, and the per-stage breakdown stays identical
    to the Llama rows'.

    Kept as a separate type rather than fields bolted onto
    `ModelFlopConfig`: that type is frozen and its scalar `num_heads` /
    `head_dim` are the whole point of its simplicity, and every published
    Llama row was measured through it.
    """

    hidden: int
    vocab: int
    layers: Tuple[LayerFlopSpec, ...]

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def linear_flops_per_token(self) -> int:
        return sum(
            layer.qkv_flops_per_token(self.hidden)
            + layer.o_proj_flops_per_token(self.hidden)
            + layer.mlp_flops_per_token(self.hidden)
            for layer in self.layers
        )

    @property
    def lm_head_flops(self) -> int:
        return 2 * self.hidden * self.vocab

    def attn_prefill_flops(self, n_new: int, n_cached: int = 0) -> int:
        if n_new <= 0:
            return 0
        n_cached = max(n_cached, 0)
        return sum(
            layer.attn_flops_per_query_key_pair
            * layer.keys_visited_prefill(n_new, n_cached)
            for layer in self.layers
        )

    def attn_decode_step_flops(self, attended_len: int) -> int:
        if attended_len <= 0:
            return 0
        return sum(
            layer.attn_flops_per_query_key_pair
            * layer.keys_visited_decode(attended_len)
            for layer in self.layers
        )

    def scoring_flops(self, look_ahead: int, ctx_len: int) -> int:
        """`scoring.py::compute_attention_score` -- `Q @ K^T` only, no AV.

        Charged over EVERY layer and the FULL context, deliberately, even
        for sliding layers and even under `score_layers="global_only"` or
        `mask_sliding_window`: `compute_attention_score` computes the whole
        score tensor and the restriction is applied afterwards, by
        `aggregate_attention_score` dropping rows or by a mask. So the
        scoring modes change what is USED, not what is computed, and this
        column is identical across all three.
        """
        if look_ahead <= 0 or ctx_len <= 0:
            return 0
        return sum(
            2 * layer.num_heads * layer.head_dim * look_ahead * ctx_len
            for layer in self.layers
        )


    def describe(self) -> dict:
        """JSON-serialisable summary, grouped by DISTINCT layer shape.

        A 60-layer Gemma 4 config has 60 `LayerFlopSpec`s but only two
        distinct shapes; printing all 60 buries the two facts a reader
        needs (how many of each, and what each charges). Grouping also makes
        a mis-derived layer obvious -- three groups where there should be
        two, or a count that does not match the interleave ratio.
        """
        from collections import Counter

        counts = Counter(
            (l.num_heads, l.num_kv_heads, l.head_dim, l.intermediate,
             l.has_v_proj, l.sliding_window, l.moe_intermediate,
             l.num_experts, l.moe_top_k)
            for l in self.layers
        )
        groups = []
        for shape, n in counts.most_common():
            (num_heads, num_kv_heads, head_dim, intermediate, has_v_proj,
             window, moe_intermediate, num_experts, moe_top_k) = shape
            groups.append({
                "count": n,
                "kind": "sliding" if window is not None else "full",
                "sliding_window": window,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "has_v_proj": has_v_proj,
                "intermediate": intermediate,
                "moe": ({"experts": num_experts, "top_k": moe_top_k,
                         "intermediate": moe_intermediate}
                        if num_experts and moe_top_k else None),
            })
        return {
            "kind": "LayeredFlopConfig",
            "hidden": self.hidden,
            "vocab": self.vocab,
            "num_layers": self.num_layers,
            "linear_flops_per_token": self.linear_flops_per_token,
            "layer_groups": groups,
        }


def layered_flop_config(hf_config):
    """Build a per-layer FLOP config for an interleaved and/or MoE model.

    Derived by reading `vllm/model_executor/models/gemma4.py`, since the
    config field names alone do not say how they are combined:

      - `layer_types[i] == "full_attention"` decides the layer type
        (`Gemma4DecoderLayer.__init__`).
      - head dim is `global_head_dim` for full-attention layers, `head_dim`
        otherwise -- with `global_head_dim` defaulting to `head_dim`.
      - `attention_k_eq_v` applies ONLY to full-attention layers, and when it
        does the KV head count switches to `num_global_key_value_heads` AND
        the V projection disappears.
      - a sliding layer's window is `config.sliding_window`.
      - the MoE block is enabled by `enable_moe_block` OR `use_second_mlp_block`.

    That last one is a real disagreement with
    `evaluation_pipeline/hardware_metrics.py`, which checks only
    `enable_moe_block` and would therefore charge no expert FLOPs at all for
    a checkpoint configured the other way. It is also the reason this was
    written against the model source rather than adapted from that file.

    Returns None when the model is uniform, so the caller falls back to the
    simpler `ModelFlopConfig` and every published Llama row keeps its exact
    arithmetic.
    """
    cfg = (
        hf_config.get_text_config()
        if hasattr(hf_config, "get_text_config")
        else hf_config
    )

    num_layers = getattr(cfg, "num_hidden_layers", None)
    hidden = getattr(cfg, "hidden_size", None)
    if not num_layers or not hidden:
        return None

    layer_types = getattr(cfg, "layer_types", None) or ["full_attention"] * num_layers
    if len(layer_types) != num_layers:
        return None

    head_dim = getattr(cfg, "head_dim", None) or (
        hidden // cfg.num_attention_heads
    )
    global_head_dim = getattr(cfg, "global_head_dim", None) or head_dim
    kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    global_kv_heads = getattr(cfg, "num_global_key_value_heads", None) or kv_heads
    k_eq_v = bool(getattr(cfg, "attention_k_eq_v", False))
    window = getattr(cfg, "sliding_window", None)

    moe_on = bool(getattr(cfg, "enable_moe_block", False)) or bool(
        getattr(cfg, "use_second_mlp_block", False)
    )
    num_experts = int(getattr(cfg, "num_experts", 0) or 0)
    moe_top_k = int(getattr(cfg, "top_k_experts", 0) or 0)
    moe_intermediate = int(
        getattr(cfg, "moe_intermediate_size", None)
        or getattr(cfg, "expert_intermediate_size", None)
        or 0
    )

    layers = []
    for layer_type in layer_types:
        is_full = layer_type == "full_attention"
        use_k_eq_v = is_full and k_eq_v
        layers.append(
            LayerFlopSpec(
                num_heads=cfg.num_attention_heads,
                num_kv_heads=global_kv_heads if use_k_eq_v else kv_heads,
                head_dim=global_head_dim if is_full else head_dim,
                intermediate=cfg.intermediate_size,
                has_v_proj=not use_k_eq_v,
                sliding_window=None if is_full else window,
                moe_intermediate=moe_intermediate if moe_on else 0,
                num_experts=num_experts if moe_on else 0,
                moe_top_k=moe_top_k if moe_on else 0,
            )
        )

    # The flat `ModelFlopConfig` is exact only when every layer is
    # genuinely identical AND unwindowed AND has no experts. Checking
    # `layer_types` alone is not enough, and both gaps matter:
    #
    #   - all-`full_attention` with `global_head_dim != head_dim` looks
    #     uniform by type, but the flat builder reads `head_dim` and would
    #     silently use the wrong (smaller) one for every layer.
    #   - all-`sliding_attention` also looks uniform, and the flat model has
    #     no window at all, so it would charge full-context attention for
    #     layers that never read past 512 keys.
    #
    # Compare the built specs instead, which is what actually determines
    # whether the two models agree.
    identical = len({
        (l.num_heads, l.num_kv_heads, l.head_dim, l.intermediate,
         l.has_v_proj, l.sliding_window, l.num_experts, l.moe_top_k)
        for l in layers
    }) == 1
    windowed = any(l.sliding_window is not None for l in layers)
    if identical and not windowed and not moe_on:
        # Uniform after all -- but return the FLAT config built from the
        # resolved layer spec rather than None. Returning None would send
        # the caller back to `model_flop_config`'s own flat builder, which
        # reads `head_dim` directly; on an all-`full_attention` model with
        # `global_head_dim != head_dim` that is the wrong (smaller) value
        # for every layer. Building it from the spec cannot disagree with
        # the per-layer derivation above.
        one = layers[0]
        return ModelFlopConfig(
            num_layers=len(layers),
            hidden=hidden,
            num_heads=one.num_heads,
            num_kv_heads=one.num_kv_heads,
            head_dim=one.head_dim,
            intermediate=one.intermediate,
            vocab=cfg.vocab_size,
        )

    return LayeredFlopConfig(
        hidden=hidden, vocab=cfg.vocab_size, layers=tuple(layers)
    )


def model_flop_config(hf_config):
    """Derives a `ModelFlopConfig` from a HuggingFace config, or None when
    this model's shape cannot be expressed by one.

    Same field-extraction fallbacks as `sparse_decode_microbench.py::
    bytes_per_token_kv` (`num_key_value_heads` defaulting to
    `num_attention_heads` for non-GQA models, `head_dim` defaulting to
    `hidden_size // num_attention_heads`) -- kept identical deliberately so
    the two derived quantities can never disagree about a model's shape.

    **Reads the TEXT config.** A natively multimodal checkpoint wraps it:
    Gemma 4's top-level `Gemma4Config` has no `num_attention_heads` at all,
    and reading it raises an AttributeError that never mentions the wrapper.
    `get_text_config()` returns the config itself on a text-only model, so
    this is a no-op for Llama/Qwen.

    **Returns None for a shape this dataclass cannot represent**, rather than
    a number that would be wrong. `ModelFlopConfig` is flat -- one
    `head_dim`, one `num_kv_heads`, one `intermediate` for the whole model --
    and two Gemma-4 properties break that:

      - Per-layer-type attention geometry. Sliding layers use `head_dim` /
        `num_key_value_heads`; full-attention layers use `global_head_dim` /
        `num_global_key_value_heads`, typically 2x the head dim, and may drop
        the V projection entirely (`attention_k_eq_v`). Collapsing those to
        one number misstates every attention FLOP.
      - MoE. A routed block computes only `top_k` of `num_experts`, in
        parallel with a dense MLP, so `intermediate_size` alone does not
        describe the MLP cost either.

    Emitting an approximate number here would put it in the results CSV
    beside genuinely measured ones with nothing marking it as an estimate.
    `benchmarks/evaluation_pipeline/hardware_metrics.py` has correct
    per-layer accounting for Gemma-4-26B-A4B (verified against `gemma4.py`,
    not inferred from config field names) -- but it computes ACTIVE
    PARAMETERS and bytes-moved, not the per-stage prefill/lookahead/scoring/
    decode breakdown this module produces, so it is a starting point for a
    Gemma 4 FLOP model rather than a drop-in.
    """
    cfg = (
        hf_config.get_text_config()
        if hasattr(hf_config, "get_text_config")
        else hf_config
    )

    layer_types = getattr(cfg, "layer_types", None)
    heterogeneous_attention = bool(layer_types) and len(set(layer_types)) > 1
    global_head_dim = getattr(cfg, "global_head_dim", None)
    heterogeneous_head_dim = (
        global_head_dim is not None
        and global_head_dim != getattr(cfg, "head_dim", global_head_dim)
    )
    is_moe = bool(getattr(cfg, "enable_moe_block", False)) or bool(
        getattr(cfg, "num_experts", 0)
    )
    if heterogeneous_attention or heterogeneous_head_dim or is_moe:
        # Not representable by this flat dataclass -- hand it to the
        # per-layer model instead, which charges each layer its own
        # attention geometry, its own sliding window, and its own MoE
        # experts. Still returns None if that cannot describe it either, so
        # a caller that gets None can still omit the columns rather than
        # print an estimate.
        return layered_flop_config(hf_config)

    num_heads = cfg.num_attention_heads
    return ModelFlopConfig(
        num_layers=cfg.num_hidden_layers,
        hidden=cfg.hidden_size,
        num_heads=num_heads,
        num_kv_heads=getattr(cfg, "num_key_value_heads", None) or num_heads,
        head_dim=getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_heads),
        intermediate=cfg.intermediate_size,
        vocab=cfg.vocab_size,
    )


@dataclass
class FlopBreakdown:
    """Per-stage FLOP attribution for one turn (or summed over many).

    The BREAKDOWN is the deliverable, not the total -- the point is knowing
    which stage to optimize. `spec_*` stages are zero for the M000 baseline
    (no speculator at all).
    """

    spec_prefill: int = 0
    spec_lookahead: int = 0
    spec_scoring: int = 0
    target_prefill: int = 0
    target_decode: int = 0

    STAGES = ("spec_prefill", "spec_lookahead", "spec_scoring",
              "target_prefill", "target_decode")

    @property
    def total(self) -> int:
        return sum(getattr(self, s) for s in self.STAGES)

    @property
    def speculator_total(self) -> int:
        return self.spec_prefill + self.spec_lookahead + self.spec_scoring

    @property
    def speculator_fraction(self) -> float:
        total = self.total
        return self.speculator_total / total if total else 0.0

    def __iadd__(self, other: "FlopBreakdown") -> "FlopBreakdown":
        for stage in self.STAGES:
            setattr(self, stage, getattr(self, stage) + getattr(other, stage))
        return self

    def as_dict(self) -> Dict[str, int]:
        d = asdict(self)
        d["total"] = self.total
        return d


def speculator_turn_flops(
    cfg: ModelFlopConfig,
    pool_len: int,
    num_cached: int,
    look_ahead: int,
) -> FlopBreakdown:
    """Everything the speculator costs for one turn.

    Args:
        cfg: the SPECULATOR's shape (Llama-3.2-1B in this pipeline).
        pool_len: total submitted length -- candidate pool + this turn's
            query (`PrunedTurnResult.orig_len`).
        num_cached: prefix-cache hits on the speculator's persistent engine
            (`PrunedTurnResult.num_cached_tokens`). The speculator runs with
            `enable_prefix_caching=True` and a conversation-scoped cache, so
            in a growing multi-turn conversation this is most of `pool_len`
            after turn 0 -- ignoring it would badly over-count.
        look_ahead: lookahead decode steps ACTUALLY taken
            (`PrunedTurnResult.actual_look_ahead_cnt`, which can be less
            than the configured `look_ahead_cnt` if the sample hit EOS).

    The lookahead steps attend a sequence that grows by one each step
    (`pool_len + 1 + j` after `j` prior steps), and each samples a token, so
    each pays `lm_head` as well.
    """
    n_new = max(pool_len - num_cached, 0)
    prefill = n_new * cfg.linear_flops_per_token + cfg.attn_prefill_flops(n_new, num_cached)

    lookahead = 0
    for j in range(max(look_ahead, 0)):
        lookahead += cfg.linear_flops_per_token + cfg.lm_head_flops
        lookahead += cfg.attn_decode_step_flops(pool_len + 1 + j)

    scoring = cfg.scoring_flops(look_ahead, pool_len)
    return FlopBreakdown(spec_prefill=prefill, spec_lookahead=lookahead, spec_scoring=scoring)


def target_prefill_flops(cfg: ModelFlopConfig, prompt_len: int, num_cached: int) -> int:
    """Target prefill for one turn.

    `num_cached` comes from `RequestOutput.num_cached_tokens` -- MEASURED,
    not assumed. It matters in all three modes: baseline never sets
    `enable_prefix_caching` explicitly but gets vLLM's default, sparse runs
    a resumable session, and specprefill's pruned prompts still share the
    constant chat-wrapper prefix.

    One `lm_head` row is charged (the first sampled token), never per-token
    -- see the module docstring.
    """
    n_new = max(prompt_len - num_cached, 0)
    return (
        n_new * cfg.linear_flops_per_token
        + cfg.attn_prefill_flops(n_new, min(num_cached, prompt_len))
        + (cfg.lm_head_flops if n_new > 0 else 0)
    )


def target_sparse_prefill_flops(
    cfg: ModelFlopConfig, prefill_steps: Sequence[Tuple[int, int]]
) -> int:
    """Target prefill for one turn, MEASURED per chunk instead of derived
    from `(prompt_len, num_cached)`.

    Used only by the sparse pipeline's opt-in sparse-prefill scope, where
    `target_prefill_flops`' analytic model no longer applies: that function
    assumes every new token attends every cached token, which is exactly
    the assumption the gather breaks. The replacement input comes from
    `vllm_patch/sparse_target_runner.py::pop_prefill_steps` -- one
    `(num_query_tokens, attended_len)` pair per prefill chunk, with
    `attended_len` the block-PADDED length the kernel was actually handed
    (a chunk the gather declined to restrict reports its full length, so
    dense chunks are charged densely and turn 0 comes out identical to the
    analytic model modulo prefix-cache bookkeeping).

    Each chunk's causal key-visit count is
    `attn_prefill_flops(n_q, attended_len - n_q)`: the chunk's own `n_q`
    tokens sit at the tail of the gathered view by construction (see
    `kv_cache_utils.compute_prefill_gather_view`'s "Why the tail must be
    contiguous"), so they see `attended_len - n_q` keys of history plus the
    usual causal triangle among themselves.

    One `lm_head` row is charged for the turn, not per chunk -- same
    convention as `target_prefill_flops`: only the final chunk produces
    logits, and only the one sampled token's row.
    """
    total = 0
    charged_any = False
    for num_query_tokens, attended_len in prefill_steps:
        if num_query_tokens <= 0:
            continue
        charged_any = True
        total += num_query_tokens * cfg.linear_flops_per_token
        total += cfg.attn_prefill_flops(
            num_query_tokens, max(attended_len - num_query_tokens, 0)
        )
    if charged_any:
        total += cfg.lm_head_flops
    return total


def target_decode_flops(cfg: ModelFlopConfig, attended_lens: Sequence[int]) -> int:
    """Target decode, one entry in `attended_lens` per decode step.

    For the SPARSE path these are the gathered (block-padded) lengths the
    attention kernel was actually told to read -- i.e. **hardware-executed**
    work, not an idealized token-exact figure. That's the right quantity for
    optimization: it's what the GPU does.

    For baseline/specprefill the sequence is simply
    `prompt_len + i + 1 for i in range(out_len)` -- attention is dense
    there, see `dense_decode_attended_lens`.
    """
    total = 0
    for attended in attended_lens:
        total += cfg.linear_flops_per_token + cfg.lm_head_flops
        total += cfg.attn_decode_step_flops(attended)
    return total


def dense_decode_attended_lens(prompt_len: int, num_decode_steps: int) -> List[int]:
    """Attended length per decode step for an UNRESTRICTED (dense) decode.

    Step `i` attends the whole prompt plus the `i` tokens generated before
    it, plus itself. Used for M000 and SPECPREFILL, whose decode attention
    is never gathered, so no runner-side instrumentation is needed at all --
    only the SPARSE path has to measure (see
    `vllm_patch/sparse_target_runner.py::pop_attended_lens`).

    **`num_decode_steps` is one LESS than the number of generated tokens.**
    The first output token is sampled from the prefill's own logits row
    (already charged by `target_prefill_flops`); only tokens 2..N cost a
    decode step. Passing `out_len` here instead would over-count by one
    full step per turn.
    """
    return [prompt_len + i + 1 for i in range(max(num_decode_steps, 0))]
