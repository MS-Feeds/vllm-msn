"""Speculative Prefill scoring math — Algorithm 1, lines 12, 14, 16.

Engine- and architecture-agnostic: pure tensor-in/tensor-out functions, no
vLLM engine dependency. Copied verbatim from `../../spec_prefill_llama/
vllm_patch/scoring.py` — see the approved plan's "Files to create / change"
(copy-unchanged list): this module has no notion of a single vs. multi-turn
caller baked in, it just scores whatever Q/K/context_len it's handed. In the
multi-turn pipeline it's called once per turn by `pruner.py`, and a second
time (against the *target* model's own attention instead of the speculator's)
by the oracle-upper-bound path — see EXPERIMENT_PLAN.md's "Oracle upper
bound" section.

    12: A <- compute_attention_score(Q, K)
    14: A <- aggregate_attention_score(A)
    16: T <- chunk_select_from_smoothed_attention(A)   # T = kept token indices

Lines 1-11 (batch splitting, the lookahead loop, Q/K retrieval) live in
prefill_split.py / proposer.py / kv_cache_utils.py. Lines 18-20 (position-id
restoration, request merging, target forward) are out of scope for this pass
— see EXPERIMENT_PLAN.md's "Implementation status".
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from .config import SpecConfig


# Layer types that attend over the WHOLE sequence, vs. those restricted to a
# local window. Enumerated rather than "anything that isn't sliding" so an
# unrecognised type fails loudly instead of being silently counted as global
# -- the whole point of `global_only` is that a locally-restricted layer's
# score for a distant position is meaningless, and quietly admitting an
# unknown type would defeat it. `mamba`/`linear_attention` are deliberately
# absent: those layers produce no attention scores to aggregate at all.
GLOBAL_LAYER_TYPES = frozenset({"full_attention", "attention"})
LOCAL_LAYER_TYPES = frozenset(
    {"sliding_attention", "chunked_attention", "local_attention"}
)


@dataclass(frozen=True)
class LayerGeometry:
    """Per-layer facts about the SCORING model that the attention math needs
    and cannot infer from the Q/K tensors alone.

    Exists because this module is deliberately vLLM-free (see the module
    docstring): it takes plain lists rather than a model or an `hf_config`,
    so every policy below stays unit-testable on CPU with no checkpoint.
    `speculator_worker.py::layer_geometry_from_attention_layers` is what
    builds one from a live model.

    **Every field defaults to None, and a `geometry=None` (or an all-None
    geometry) is a provable no-op**: the scoring math then behaves exactly as
    it did before this type existed, so every already-published Llama row is
    reproduced bit-identically rather than "probably unchanged". That is the
    same discipline `model_truncation.py` uses for its weight filter.

    Fields:
      layer_types: per-layer `config.layer_types` entry, e.g.
        `"sliding_attention"` / `"full_attention"`. Required by
        `score_layers="global_only"`, ignored otherwise.
      scales: the layer's OWN attention scale -- the multiplier applied to
        `Q @ K^T` before the softmax. Not always `1/sqrt(head_dim)`: Gemma 4
        sets `scaling = 1.0` and lets its learnable Q/K norms carry the
        scaling instead (`gemma4.py`'s `Gemma4Attention.__init__`). Getting
        this wrong is not a constant factor -- it changes the softmax
        TEMPERATURE, and on a model whose head_dim differs between layer
        types it changes it by a DIFFERENT amount per type, which then
        biases the `max`-over-(layer, head) collapse toward whichever type
        got the sharper distribution. When None, the historical
        `1/sqrt(head_dim)` is used.
      sliding_windows: each layer's own attention window, `None` for a
        full-attention layer. Required by `mask_to_window` scoring; carries
        strictly more information than `layer_types`, which is derived from
        it.
      kv_shared: True for a layer that reads another layer's KV cache rather
        than owning one (Gemma 3n/4's `kv_sharing_target_layer_name`). Such
        a layer's K is a DUPLICATE of its target's, so leaving it in gives
        that one K vector a second vote under `max`. Dropped unconditionally
        when supplied; on a model with no sharing the list is all-False and
        nothing is dropped.
      logits_soft_cap: `config.attn_logit_softcapping`, applied as
        `cap * tanh(x / cap)` before the softmax, matching what the attention
        kernel itself does. None disables it.
    """

    layer_types: Optional[List[str]] = None
    scales: Optional[List[float]] = None
    kv_shared: Optional[List[bool]] = None
    sliding_windows: Optional[List[Optional[int]]] = None
    logits_soft_cap: Optional[float] = None

    def is_noop(self) -> bool:
        """Whether this geometry changes nothing -- used to keep the "an
        unsupplied geometry reproduces published rows exactly" claim checkable
        rather than merely asserted."""
        return (
            self.layer_types is None
            and self.scales is None
            and self.kv_shared is None
            and self.sliding_windows is None
            and self.logits_soft_cap is None
        )


def layer_geometry_from_attention_layers(attn_layers) -> LayerGeometry:
    """Build a `LayerGeometry` by reading each layer's OWN `Attention` module.

    Deliberately reads the live modules rather than the `hf_config`. Three
    reasons, all checked against this fork's real source rather than assumed:

    1. `Attention.__init__` is where a model's per-layer decisions actually
       land -- `self.sliding_window` (attention.py, set from
       `per_layer_sliding_window`) and `self.kv_sharing_target_layer_name`.
       A config field is the model's INPUT; these are its output, and for an
       interleaved model the two can diverge (e.g. `CacheConfig.sliding_window`
       is deliberately left unset for interleaved models precisely so it
       cannot override the per-layer values -- see `arg_utils.py`'s
       `is_interleaved` guard).
    2. The attention SCALE is not stored on the `Attention` module at all --
       it is forwarded to the backend impl, which keeps it as
       `self.scale = float(scale)` (confirmed in both `triton_attn.py` and
       `flash_attn.py`). Reading it there gets the number the kernel actually
       uses, whatever the model chose: `1/sqrt(head_dim)` for Llama, `1.0`
       for Gemma 4, `query_pre_attn_scalar**-0.5` for Gemma 2/3.
    3. It works for a truncated scorer (`hf_overrides={"num_hidden_layers":
       n}`) with no extra bookkeeping, because there simply are n modules.

    Takes duck-typed objects (only `getattr` is used, no isinstance and no
    vLLM import), so it is unit-testable on CPU with stand-ins -- the same
    reason `model_truncation.keep_weight_for_layer_range` is a pure function.

    Args:
        attn_layers: the per-layer `Attention` modules, in layer order --
            e.g. `[layer.self_attn.attn for layer in model.model.layers]`.

    Raises:
        ValueError: if a layer's scale cannot be found (better than silently
            falling back to `1/sqrt(head_dim)`, which is the exact bug this
            exists to fix), or if layers disagree on `logits_soft_cap`, which
            this type can only carry as one scalar.
    """
    layer_types: List[str] = []
    scales: List[float] = []
    kv_shared: List[bool] = []
    sliding_windows: List[Optional[int]] = []
    soft_caps = set()

    for idx, attn in enumerate(attn_layers):
        window = getattr(attn, "sliding_window", None)
        layer_types.append("sliding_attention" if window else "full_attention")
        sliding_windows.append(int(window) if window else None)

        kv_shared.append(
            getattr(attn, "kv_sharing_target_layer_name", None) is not None
        )

        impl = getattr(attn, "impl", None)
        scale = getattr(impl, "scale", None)
        if scale is None:
            scale = getattr(attn, "scale", None)
        if scale is None:
            raise ValueError(
                f"could not read the attention scale for layer {idx} "
                f"({type(attn).__name__}) -- expected it on `.impl.scale`. "
                f"Refusing to fall back to 1/sqrt(head_dim): that is right "
                f"for Llama and wrong for Gemma 4, and guessing silently is "
                f"how the scoring softmax ends up at the wrong temperature."
            )
        scales.append(float(scale))

        soft_caps.add(getattr(impl, "logits_soft_cap", None))

    soft_caps.discard(None)
    if len(soft_caps) > 1:
        raise ValueError(
            f"layers disagree on logits_soft_cap ({sorted(soft_caps)}); "
            f"LayerGeometry carries a single scalar."
        )

    return LayerGeometry(
        layer_types=layer_types,
        scales=scales,
        kv_shared=kv_shared,
        sliding_windows=sliding_windows,
        logits_soft_cap=soft_caps.pop() if soft_caps else None,
    )


def compute_attention_score(
    query_buffer: List[torch.Tensor],
    key_buffer: List[List[torch.Tensor]],
    actual_look_ahead_cnts: List[int],
    geometry: Optional[LayerGeometry] = None,
    mask_to_window: bool = False,
) -> List[torch.Tensor]:
    """Algorithm line 12: A <- compute_attention_score(Q, K).

    Q @ K^T, scaled per layer, per sample.

    Args:
        query_buffer: per-layer list of buffered query tensors, each
            [num_prefill_samples, look_ahead_cnt, num_heads * head_dim]
            (the shape produced by stacking one entry per lookahead step —
            see proposer.py's query-capture hook).
        key_buffer: per-layer list of per-sample key tensors, each
            [context_len, num_kv_heads, head_dim] (from kv_cache_utils.py).
        actual_look_ahead_cnts: per-sample count of lookahead steps actually
            used (may be less than the configured look_ahead_cnt if a
            sample hit EOS early).
        geometry: optional per-layer scales and logit softcapping -- see
            `LayerGeometry`. When None (the default) this falls back to
            `1/sqrt(head_dim)` with no softcapping, which is the reference
            implementation's behavior and what every published Llama row was
            measured under.
        mask_to_window: restrict each sliding layer to the positions it could
            actually attend to, by setting everything further than its own
            window to `-inf` before the softmax -- exactly what the attention
            kernel does.

            The alternative to dropping those layers outright
            (`score_layers="global_only"`). Both address the same defect:
            unmasked, a sliding layer scores `Q · K` for pairs the model never
            computes in inference, and those uncalibrated values win the `max`
            aggregation more often than chance. Masking is the more principled
            of the two -- it keeps a sliding layer's real, trained opinion
            about positions inside its window instead of discarding the layer
            -- but note it recovers signal only near the query. At long range
            a sliding layer is silent either way, so for the long-context
            retrieval decisions a keep-rate sweep is made of, both modes leave
            the same handful of full-attention layers deciding.

            Requires `geometry.sliding_windows`. Defaults off, so an
            unconfigured run scores exactly as before.

    Returns:
        Per-sample list of [num_layer, num_head, look_ahead_cnt, context_len]
        attention-score tensors.
    """
    scales = geometry.scales if geometry is not None else None
    soft_cap = geometry.logits_soft_cap if geometry is not None else None
    windows = geometry.sliding_windows if geometry is not None else None
    if mask_to_window and windows is None:
        raise ValueError(
            "mask_to_window needs geometry.sliding_windows -- there is no way "
            "to tell how far a layer can attend from the Q/K tensors alone."
        )
    if scales is not None and len(scales) != len(query_buffer):
        raise ValueError(
            f"LayerGeometry.scales has {len(scales)} entries but the query "
            f"buffer holds {len(query_buffer)} layers -- a geometry built for "
            f"a different model (or before layer truncation) cannot be used "
            f"as-is."
        )
    attn_weights: List[List[torch.Tensor]] = []

    for layer_idx in range(len(query_buffer)):
        attn_weights.append([])

        keys = key_buffer[layer_idx]
        # Unbind along the sample dim: query_buffer[layer_idx] is
        # [num_samples, look_ahead_cnt, num_heads*head_dim] -> per-sample
        # [look_ahead_cnt, num_heads*head_dim].
        queries = torch.unbind(query_buffer[layer_idx], dim=0)

        for q, k, c in zip(queries, keys, actual_look_ahead_cnts):
            look_ahead_cnt, num_heads_times_head_dim = q.shape
            num_kv_heads, head_dim = k.shape[-2], k.shape[-1]
            num_heads = num_heads_times_head_dim // head_dim

            # [look_ahead_cnt, num_heads*head_dim] -> [num_heads, look_ahead_cnt, head_dim]
            query = q.view(look_ahead_cnt, num_heads, head_dim).transpose(0, 1)
            key = k.transpose(0, 1)  # [num_kv_heads, context_len, head_dim]

            if num_heads != num_kv_heads:
                # GQA: repeat KV heads to match query heads (mirrors the
                # reference's repeat_interleave-based reshape).
                assert num_heads % num_kv_heads == 0
                key = key.repeat_interleave(num_heads // num_kv_heads, dim=0)

            query = query[:, :c, :]

            attn = torch.matmul(query, key.transpose(-1, -2))
            if scales is None:
                attn = attn / math.sqrt(head_dim)
            else:
                attn = attn * scales[layer_idx]

            if soft_cap:
                # Exactly what the attention kernel applies before its own
                # softmax (`Attention(..., logits_soft_cap=...)`). Omitting it
                # leaves the scoring softmax reading pre-cap logits, whose
                # tails the model itself never sees.
                attn = soft_cap * torch.tanh(attn / soft_cap)

            if mask_to_window and windows[layer_idx] is not None:
                # Lookahead step j is generated after the whole context, so it
                # sits at absolute position `context_len + j`; context
                # position i is `context_len + j - i` tokens behind it.
                context_len = key.shape[1]
                ctx_pos = torch.arange(context_len, device=attn.device)
                q_pos = context_len + torch.arange(c, device=attn.device)
                out_of_window = (q_pos[:, None] - ctx_pos[None, :]) > windows[layer_idx]
                # -inf, not 0: the softmax below must RENORMALISE over the
                # in-window positions, which is what the kernel does. Zeroing
                # after the softmax would instead leave the layer's mass
                # spread over positions it cannot see.
                #
                # No row can be fully masked: the nearest context position is
                # `j + 1` behind step j, and lookahead counts here are single
                # digits against windows of 512+, so the minimum distance is
                # always well inside the window.
                attn = attn.masked_fill(out_of_window, float("-inf"))

            attn_weights[-1].append(attn)

    num_samples = len(attn_weights[0])
    return [
        torch.stack(
            [attn_weights[layer_idx][sample_idx] for layer_idx in range(len(attn_weights))],
            dim=0,
        )
        for sample_idx in range(num_samples)
    ]


def scoring_layer_indices(
    num_layers: int,
    score_layers: Optional[str],
    layer_types: Optional[List[str]] = None,
    kv_shared: Optional[List[bool]] = None,
    drop_kv_shared: bool = False,
) -> List[int]:
    """Which layer indices get a vote in `aggregate_attention_score`.

    Pure integer/string arithmetic, no tensors and no model, so the
    layer-restriction policy is unit-testable on CPU -- and so the "does this
    selection do what its name says" question is answered in one place rather
    than inside a tensor pipeline.

    `None` means every layer (the reference implementation's behavior and the
    default). `"skip_first2"`/`"second_half"`/`"last_quarter"` each drop EARLY
    layers and keep late ones: layers 0-1 are near-universally
    positional/sink-dominated, and under the default `max` aggregation a
    single peaked early head can set the entire importance vector by itself.
    See ACCURACY_IMPROVEMENTS.md §1.2.

    `"global_only"` is different in kind -- not a fixed slice but a property
    of the architecture. On an interleaved sliding-window model (Gemma 3/3n/4,
    Llama 4) most layers can never attend beyond a 512-1024 token window, so
    their `Q @ K^T` against a distant position is a number the model never
    computes in real inference. Under `max` over (layer, head), ONE such layer
    is enough to decide a token's importance. This selects only the
    full-attention layers, and needs `layer_types` to do it.

    `kv_shared` marks layers that read another layer's KV cache (Gemma 3n/4's
    `kv_sharing_target_layer_name`). It is applied ONLY when `drop_kv_shared`
    is set, and that defaults to False -- see `SpecConfig.
    drop_kv_shared_layers` for why. Short version: such a layer's
    `attn.kv_cache` is aliased to the target's tensor by
    `gpu_model_runner.initialize_kv_cache_tensors`, so its K is the target's
    REAL K, while its Q is its own; the resulting distribution is distinct
    signal, not a duplicate vote. The flag exists for callers reading from a
    hand-built dummy cache that does not reproduce that aliasing, where a
    shared layer's cache is never written.

    Always returns at least one layer -- a selection that would empty the
    list falls back to the last surviving candidate rather than producing a
    NaN score vector further down.
    """
    if score_layers is None:
        indices = list(range(num_layers))
    elif score_layers == "global_only":
        if layer_types is None:
            raise ValueError(
                "score_layers='global_only' needs the model's own layer_types "
                "(pass scoring.LayerGeometry(layer_types=...)); there is no "
                "way to tell a sliding layer from a full-attention one from "
                "the Q/K tensors alone."
            )
        if len(layer_types) != num_layers:
            raise ValueError(
                f"layer_types has {len(layer_types)} entries but the score "
                f"tensor has {num_layers} layers -- a geometry built for a "
                f"different checkpoint (or before layer truncation) cannot be "
                f"reused as-is."
            )
        unknown = sorted(
            set(layer_types) - GLOBAL_LAYER_TYPES - LOCAL_LAYER_TYPES
        )
        if unknown:
            raise ValueError(
                f"unrecognised layer_types {unknown} -- refusing to guess "
                f"whether they attend globally. Known global: "
                f"{sorted(GLOBAL_LAYER_TYPES)}; known local: "
                f"{sorted(LOCAL_LAYER_TYPES)}."
            )
        indices = [i for i, t in enumerate(layer_types) if t in GLOBAL_LAYER_TYPES]
        if not indices:
            raise ValueError(
                "score_layers='global_only' selected no layer -- this model "
                "has no full-attention layer at all, so there is nothing that "
                "can score a long context. Use a different score_layers."
            )
    else:
        if score_layers == "skip_first2":
            start = 2
        elif score_layers == "second_half":
            start = num_layers // 2
        elif score_layers == "last_quarter":
            start = (3 * num_layers) // 4
        else:
            raise ValueError(f"unknown score_layers: {score_layers!r}")
        start = min(start, max(num_layers - 1, 0))
        indices = list(range(start, num_layers))

    if kv_shared is not None and drop_kv_shared:
        if len(kv_shared) != num_layers:
            raise ValueError(
                f"kv_shared has {len(kv_shared)} entries but the score tensor "
                f"has {num_layers} layers."
            )
        surviving = [i for i in indices if not kv_shared[i]]
        # Keep the pre-filter selection if dropping shared layers would empty
        # it: a degenerate all-shared model is not a reason to return nothing.
        if surviving:
            indices = surviving

    if not indices:
        indices = [max(num_layers - 1, 0)]
    return indices


def _collapse_layer_head(attn: torch.Tensor, score_aggregation: str) -> torch.Tensor:
    """Collapse a [layer*head, look_ahead_cnt, context_len] tensor down the
    (layer, head) axis, per `score_aggregation`. See `SpecConfig`'s field
    docstring for what each mode is for.

    `zmean` normalizes each (layer, head)'s distribution over the CONTEXT
    axis before averaging, so heads contribute their *shape* rather than
    their magnitude. The epsilon guards a head whose pooled distribution is
    genuinely flat (zero variance across the context), which is a real case
    for a sink-dominated head after average pooling, not a hypothetical --
    without it that head alone turns the whole score vector into NaN.
    """
    if score_aggregation == "max":
        return attn.max(0)[0]
    if score_aggregation == "mean":
        return attn.mean(0)
    if score_aggregation == "zmean":
        mean = attn.mean(dim=-1, keepdim=True)
        std = attn.std(dim=-1, keepdim=True)
        return ((attn - mean) / (std + 1e-6)).mean(0)
    raise ValueError(f"unknown score_aggregation: {score_aggregation!r}")


def aggregate_attention_score(
    attn_scores: List[torch.Tensor],
    spec_config: SpecConfig,
    geometry: Optional[LayerGeometry] = None,
    winning_layers: Optional[list] = None,
) -> List[torch.Tensor]:
    """Algorithm line 14: A <- aggregate_attention_score(A).

    softmax -> optional smoothing pool -> collapse (layer, head) -> mean over
    lookahead steps, producing one importance score per prompt token.

    The collapse step is `max` by default -- the reference implementation's
    behavior, so an unconfigured run reproduces every already-published row
    exactly -- and `spec_config.score_aggregation` /
    `spec_config.score_layers` select the alternatives (`mean`, `zmean`, and
    late-layer-only voting). Those exist because ORACLE-k20 attributed 17.0
    of `scbench_kv`'s 25.0-point degradation to the speculator's estimation
    error, against 8.0 for the sparse-decode mechanism, while this whole
    function costs ~0.007% of a turn's FLOPs. See ACCURACY_IMPROVEMENTS.md.

    Args:
        attn_scores: per-sample [num_layer, num_head, look_ahead_cnt, context_len]
            tensors, as returned by compute_attention_score.
        spec_config: supplies `pool_kernel_size`, `score_aggregation`,
            `score_layers`, `score_head_set`.
        geometry: optional per-layer `layer_types`/`kv_shared`, consumed by
            `scoring_layer_indices`. None reproduces the reference behavior.
        winning_layers: optional list to APPEND per-sample winner tensors to,
            one `[look_ahead_cnt, context_len]` int tensor per sample giving
            the ORIGINAL layer index whose head won the `max` at each
            (lookahead step, context position).

            Production throws this away -- `attn.max(0)` keeps `.values` and
            discards `.indices`. It is collected here, inside the real
            function, rather than in a diagnostic that reimplements these
            steps: a copy of an aggregation pipeline drifts from the pipeline
            it copied, and a diagnostic that silently measures something
            other than production is worse than no diagnostic. Indices are
            mapped back through `layer_indices`, so a restricted selection
            (`global_only`) still reports true layer numbers.

            Requires `score_aggregation="max"` (there is no argmax to report
            for `mean`/`zmean`) and is incompatible with `score_head_set`,
            which renumbers the head axis this maps through.

    Returns:
        Per-sample 1D [context_len] token-importance tensors.
    """
    layer_types = geometry.layer_types if geometry is not None else None
    kv_shared = geometry.kv_shared if geometry is not None else None
    token_importance: List[torch.Tensor] = []
    collect = winning_layers is not None

    for attn in attn_scores:
        original_dtype = attn.dtype
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(
            original_dtype
        )

        # Layer restriction happens BEFORE the flatten below -- dim 0 is the
        # layer axis only until (layer, head) are folded together.
        layer_indices = scoring_layer_indices(
            attn.shape[0],
            spec_config.score_layers,
            layer_types,
            kv_shared,
            spec_config.drop_kv_shared_layers,
        )
        if len(layer_indices) != attn.shape[0]:
            if spec_config.score_head_set is not None:
                # `SpecConfig.__post_init__` already forbids combining
                # score_head_set with score_layers, but a GEOMETRY can drop
                # layers too (kv_shared) without score_layers being set --
                # and that shifts the flattened layer*head axis the head
                # indices are into. Refuse rather than silently scoring with
                # a different head set than the caller named.
                raise ValueError(
                    f"score_head_set indexes the FULL layer*head axis, but "
                    f"this geometry keeps only {len(layer_indices)} of "
                    f"{attn.shape[0]} layers, which renumbers that axis. "
                    f"Re-derive the head list for the restricted layer set, "
                    f"or drop the geometry's layer restriction."
                )
            attn = attn[layer_indices]

        # Flatten (layer, head) into one axis so pooling/collapse apply
        # uniformly. `num_heads` is captured first: it is what converts a
        # flattened row index back into a layer, for `winning_layers`.
        num_heads = attn.shape[1]
        attn = attn.flatten(0, 1)

        # Retrieval-head filtering, applied here rather than after pooling:
        # `avg_pool1d` is independent per row, so masking first is
        # numerically identical and pools 2 rows instead of 512.
        if spec_config.score_head_set is not None:
            head_rows = torch.tensor(
                [h for h in spec_config.score_head_set if h < attn.shape[0]],
                dtype=torch.long, device=attn.device,
            )
            if head_rows.numel() == 0:
                raise ValueError(
                    f"score_head_set {spec_config.score_head_set} selects no head "
                    f"that exists -- this model's flattened layer*head axis is "
                    f"only {attn.shape[0]} wide. A head list built for a "
                    f"different checkpoint cannot be reused as-is."
                )
            attn = attn[head_rows]

        kernel_size = spec_config.pool_kernel_size
        if kernel_size:
            attn = torch.nn.functional.avg_pool1d(
                attn,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            )

        if collect:
            if spec_config.score_aggregation != "max":
                raise ValueError(
                    f"winning_layers needs score_aggregation='max' (got "
                    f"{spec_config.score_aggregation!r}) -- mean/zmean have no "
                    f"argmax to report."
                )
            if spec_config.score_head_set is not None:
                raise ValueError(
                    "winning_layers is incompatible with score_head_set, "
                    "which renumbers the head axis the winner index maps "
                    "through."
                )
            values, rows = attn.max(0)
            # Row r of the flattened axis is head (r % num_heads) of the
            # r // num_heads'th SELECTED layer -- map back through
            # layer_indices so a restricted selection still reports true
            # layer numbers.
            selected = torch.tensor(
                layer_indices, device=rows.device, dtype=torch.long
            )
            winning_layers.append(selected[rows // num_heads])
            attn = values
        else:
            attn = _collapse_layer_head(attn, spec_config.score_aggregation)
        attn = attn.mean(0)  # mean over lookahead steps

        token_importance.append(attn)

    return token_importance


def phantom_vote_counts(
    winning_layers,
    layer_windows: List[Optional[int]],
    positions: List[int],
    query_positions: List[int],
) -> tuple:
    """How many winning (layer, head) votes were cast by a layer that could
    not actually have attended to the position it won.

    This is the whole measurement behind the sliding-window gate. On an
    interleaved model most layers can only attend within a 512-1024 token
    window, but `compute_attention_score` computes `Q @ K^T` over the ENTIRE
    context for every layer, and `max` over (layer, head) lets any one of
    them decide a token's importance. A vote is "phantom" when the winning
    layer is a sliding layer and the position it won is further from that
    lookahead step's query than that layer's own window.

    Pure integer arithmetic over plain sequences -- no torch, no model -- so
    the accounting is unit-testable on CPU. The caller supplies each layer's
    own window rather than a single global one, because inferring "is this a
    sliding layer" from a proxy fails silently: an earlier version of this
    gate used `head_dim < max(head_dims)`, which reports every layer as
    full-attention on any checkpoint with uniform head dims, making the
    hypothesis look REFUTED when it was never tested.

    Args:
        winning_layers: `[look_ahead_cnt][context_len]` nested sequence of
            ORIGINAL layer indices, from `aggregate_attention_score`'s
            `winning_layers` output.
        layer_windows: per-layer sliding window; `None` for a full-attention
            layer, which can never cast a phantom vote.
        positions: which context positions to score (e.g. the kept set, or a
            random sample of the pruned-away set).
        query_positions: absolute position of each lookahead step's own query
            token, one per step. A lookahead token generated after an
            `orig_len`-token prompt sits at `orig_len + step`.

    Returns:
        `(phantom, total, expected)`.

        `expected` is the phantom count you would get if the winning layer at
        each (step, position) were chosen UNIFORMLY AT RANDOM -- summed, per
        pair, over the fraction of layers whose window is shorter than that
        pair's own query distance. It is the null this measurement has to be
        read against, and without it the headline rate is badly misleading:
        on a model that is 28/35 sliding layers, ~80% of wins land on a
        sliding layer by composition alone, so a raw 86% phantom rate is
        close to what "layer choice carries no signal" looks like, not close
        to "something is broken". What matters is the EXCESS over this null,
        and the difference in that excess between kept and pruned-away
        positions.
    """
    if len(query_positions) != len(winning_layers):
        raise ValueError(
            f"got {len(query_positions)} query positions for "
            f"{len(winning_layers)} lookahead steps -- one per step is needed "
            f"to measure a distance at all."
        )

    import bisect

    num_layers = len(layer_windows)
    # Sorted windows of the sliding layers only; a full-attention layer can
    # never cast a phantom vote, so it contributes to the denominator of the
    # null but never to its numerator.
    sliding_windows = sorted(w for w in layer_windows if w is not None)

    phantom = 0
    total = 0
    expected = 0.0
    for step, row in enumerate(winning_layers):
        query_pos = query_positions[step]
        for p in positions:
            distance = query_pos - p
            window = layer_windows[int(row[p])]
            total += 1
            if window is not None and distance > window:
                phantom += 1
            # How many layers COULD have cast a phantom vote at this
            # distance -- i.e. how many have a window strictly shorter than
            # it. bisect_left over the sorted windows gives that count.
            expected += bisect.bisect_left(sliding_windows, distance) / num_layers
    return phantom, total, expected


def _chunked_topk_indices(
    sample_ti: torch.Tensor, seq_len: int, chunk_size: int, percentage: float
) -> torch.Tensor:
    """Vectorized replacement for a per-chunk Python loop that called
    `.mean()` once per TOTAL chunk and `.item()` once per KEPT chunk --
    O(context_len / chunk_size) separate op dispatches (e.g. ~2,750 chunk
    ops for an 88k-token SCBench context at chunk_size=32), each `.item()`
    additionally a GPU/device sync point under `enforce_eager=True` (no
    CUDA-graph batching to amortize it) on real hardware.

    Pads to a `chunk_size` multiple (zero-padding is safe here: dividing
    each chunk's sum by its own REAL element count, not `chunk_size`,
    means the zero padding never perturbs the mean) so all chunk means can
    be computed in one reshape+sum+divide, then builds the kept token-index
    set for the selected chunks via broadcasting instead of a per-chunk
    `.item()` + `torch.split`-indexing loop. Verified bit-identical output
    against the original per-chunk-loop implementation across context
    lengths both divisible and non-divisible by `chunk_size` -- see
    `test_vllm_patch.py`.
    """
    pad = (-seq_len) % chunk_size
    padded = torch.nn.functional.pad(sample_ti, (0, pad))  # zero-padded
    chunk_cnt = padded.shape[0] // chunk_size
    chunk_ti = padded.view(chunk_cnt, chunk_size)

    real_counts = torch.full(
        (chunk_cnt,), chunk_size, dtype=chunk_ti.dtype, device=chunk_ti.device
    )
    if pad:
        real_counts[-1] = chunk_size - pad
    chunk_means = chunk_ti.sum(-1) / real_counts

    keep_chunk_cnt = math.ceil(chunk_cnt * percentage)
    _, chunk_indices = torch.topk(chunk_means, k=keep_chunk_cnt, dim=-1)

    starts = chunk_indices * chunk_size
    offsets = torch.arange(chunk_size, device=sample_ti.device)
    token_indices = (starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
    return token_indices[token_indices < seq_len]


def chunk_select_from_smoothed_attention(
    token_importance: List[torch.Tensor],
    spec_config: SpecConfig,
) -> List[torch.LongTensor]:
    """Algorithm line 16: T <- chunk_select_from_smoothed_attention(A).

    Groups tokens into chunks of `keep_kwargs.chunk_size`, averages
    importance per chunk, and keeps the top-k% chunks by
    `keep_kwargs.percentage`. Falls back to flat (non-chunked) top-k% if
    `keep_kwargs.chunk` is false.

    Returns:
        Per-sample sorted 1D LongTensor of kept token indices (T).

    Sorting by index (not by score) is load-bearing for the multi-turn
    DISCARD history mode's "monotonic extension" property — see
    `pruner.py`'s `conversation_state.py` docstring and
    EXPERIMENT_PLAN.md's "KEEP vs. DISCARD candidate pools" section: because
    this always returns indices in original-position order, DISCARD's
    survivors are never silently reordered turn to turn, so turn N's final
    pruned prompt is guaranteed to be turn N-1's plus a suffix, with no
    extra bookkeeping needed here to enforce that.
    """
    kept_indices = []

    for sample_ti in token_importance:
        seq_len = len(sample_ti)
        percentage = spec_config.keep_kwargs.get("percentage", 1.0)

        if spec_config.keep_kwargs.get("chunk", False):
            chunk_size = spec_config.keep_kwargs.get("chunk_size", 32)
            indices = _chunked_topk_indices(sample_ti, seq_len, chunk_size, percentage)
        else:
            topk = math.ceil(seq_len * percentage)
            _, indices = torch.topk(sample_ti, k=topk, dim=-1)

        kept_indices.append(torch.sort(indices)[0])

    return kept_indices


def score_and_select_indices(
    query_buffer: List[torch.Tensor],
    key_buffer_per_layer: List[torch.Tensor],
    actual_look_ahead_cnt: int,
    spec_config: SpecConfig,
    geometry: Optional[LayerGeometry] = None,
) -> List[int]:
    """One-sample convenience wrapper chaining lines 12/14/16 above
    (`compute_attention_score` -> `aggregate_attention_score` ->
    `chunk_select_from_smoothed_attention`) for the single-conversation-
    at-a-time case this pipeline always calls with -- factored out so
    every caller that has a `(query_buffer, key_buffer_per_layer)` pair in
    hand and just wants "which local indices to keep" doesn't duplicate
    the 3-call sequence. Two real callers, in two different PROCESSES:
    `pruner.py`'s `_score_and_select` (oracle path, driver-side, scores
    the TARGET's own Q/K) and `speculator_worker.py`'s
    `SpeculatorGPUModelRunner.end_capture_and_score` (speculator path,
    runs IN-PROCESS inside the speculator's own worker so only the
    resulting small index list -- not the Q/K tensors themselves -- ever
    needs to cross `collective_rpc`'s process boundary; see that method's
    own docstring for the real, measured cost this avoids).

    Caller must not call this with `actual_look_ahead_cnt == 0` (aggregating
    over zero lookahead steps produces silent NaN, not an error -- both
    real callers already guard this themselves before calling in)."""
    key_buffer = [[k] for k in key_buffer_per_layer]  # one sample
    attn_scores = compute_attention_score(
        query_buffer, key_buffer, [actual_look_ahead_cnt], geometry,
        spec_config.mask_sliding_window,
    )
    token_importance = aggregate_attention_score(attn_scores, spec_config, geometry)
    kept_local_indices = chunk_select_from_smoothed_attention(token_importance, spec_config)[0]
    return kept_local_indices.tolist()
