"""SpecPrefillProposer — Algorithm 1, lines 3-11.

Loads the speculator (Qwen3-1.7B) as a standalone model with its own
`VllmConfig`, runs the N-step lookahead loop while capturing post-RoPE query
vectors from its own attention layers, gathers those queries across
tensor-parallel ranks if needed, and retrieves the corresponding keys from
the speculator's own KV cache (via kv_cache_utils.py).

Modeled on the model-loading pattern in
`vllm/v1/spec_decode/draft_model.py`'s `DraftModelProposer` (own
`VllmConfig` via `replace(...)`, loaded with `get_model(...)`) and the
per-step forward-call pattern in `vllm/v1/spec_decode/llm_base_proposer.py`'s
`SpecDecodeBaseProposer.propose()` (`set_forward_context(...)` wrapping
`self.model(...)`, called once per lookahead step) -- but *not* a subclass
of either, since those classes carry substantial decode-time
speculative-decoding machinery (cudagraph dispatch, padded-drafter-batch
handling, rejection sampling, parallel drafting, M-RoPE, ...) that doesn't
apply to SpecPrefill's one-shot prefill-time scoring pass.

This is a Qwen3 port of `spec_prefill/vllm_patch/proposer.py` (built and
validated against Gemma-4-E2B-it/Gemma-4-26B-A4B-it). The bulk of this
module -- the lookahead loop mechanics, TP-gather, K-retrieval, and dummy
attention-metadata construction -- is architecture-agnostic and carried over
unchanged; only the query-capturing forward hook and the attention-layer
discovery function were rewritten against Qwen3's (simpler, non-multimodal,
uniform-head_dim) attention implementation -- see their own docstrings below
for what changed and why.

Scope boundary (see EXPERIMENT_PLAN.md's "Implementation status" and the
approved plan): this class implements the forward-call *mechanics* of the
lookahead loop. Building real per-step attention metadata / slot mapping for
a live batch (block-table lookups, cross-step KV-cache-slot advancement --
what vLLM's own EAGLE proposer does via a dedicated Triton kernel,
`eagle_step_update_slot_mapping_and_metadata`) is runner-internal machinery
out of scope for this pass; `run_lookahead_steps` therefore takes
pre-built per-step attention metadata as an explicit parameter rather than
computing it internally. A future runner-integration pass supplies this for
real; the standalone test harness supplies a synthetic version.
"""

import copy
import os
import sys
import tempfile
import types
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig, replace, set_current_vllm_config
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model

from .kv_cache_utils import retrieve_keys_per_sample

_VLLM_REPO_ROOT_ON_SYSPATH = False


def _ensure_vllm_repo_root_on_syspath() -> None:
    """Idempotent: `build_lookahead_metadata` imports `tests.v1.attention.
    utils` (this fork's own test suite, not part of the installed `vllm`
    package -- see that method's docstring for why). That import only
    resolves when the vllm-msn repo ROOT is on `sys.path`, not just
    `benchmarks/spec_prefill_qwen` (which every `vllm_patch` caller script
    adds for its own imports -- confirmed on real hardware for the sibling
    Gemma4 pipeline: `ModuleNotFoundError: No module named 'tests'` without
    this, running from `benchmarks/spec_prefill/`).

    Computed relative to this file's own location
    (`benchmarks/spec_prefill_qwen/vllm_patch/proposer.py`, 3 parents up to
    repo root) rather than relying on every caller script to add it
    themselves.
    """
    global _VLLM_REPO_ROOT_ON_SYSPATH
    if _VLLM_REPO_ROOT_ON_SYSPATH:
        return
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / "tests" / "v1" / "attention" / "utils.py").exists():
        raise RuntimeError(
            f"Expected the vllm-msn repo root at {repo_root} (computed as "
            f"3 parents up from {__file__}) to contain "
            f"tests/v1/attention/utils.py, but it doesn't -- this file may "
            f"have moved, or the repo layout changed. Add the real repo "
            f"root to sys.path manually as a workaround."
        )
    sys.path.insert(0, str(repo_root))
    _VLLM_REPO_ROOT_ON_SYSPATH = True


def _ensure_distributed_environment(device: torch.device) -> None:
    """Idempotent: initializes vLLM's distributed environment (single-process,
    tensor_model_parallel_size=1/pipeline_model_parallel_size=1 -- TP/PP > 1
    is out of scope for this pass, see EXPERIMENT_PLAN.md) if not already
    done in this process.

    Needed because `SpecPrefillProposer` loads its model directly via
    `get_model()`, bypassing the normal `Worker.init_device()` lifecycle that
    a real `LLM(...)`/`Worker` would otherwise run first -- both
    `Qwen3Attention.__init__` (`get_tensor_model_parallel_world_size()`/
    `get_tensor_model_parallel_rank()`) and `tp_gather_qk` (`get_tp_group()`)
    require it, and calling either without it raises
    `AssertionError: tensor model parallel group is not initialized`
    (confirmed on real hardware for the sibling Gemma4 pipeline running
    `validate_proposer.py` standalone -- the same distributed-init
    requirement applies here, since it's a vLLM-wide precondition, not a
    per-model one).

    Safe to call even when a real `LLM(...)` has already initialized this in
    the same process (the case for the real `pruner.py`/`worker.py`
    integration, where a target-model `LLM()` is constructed first, and for
    `validate_runner_integration.py`) -- `torch.distributed.is_initialized()`
    guards the first step, and `ensure_model_parallel_initialized` (true to
    its name) no-ops if the model-parallel groups already exist at the
    expected size.

    Modeled on `tests/conftest.py`'s fixture (lines 225-248), which solves
    the identical "load a model standalone, outside a full engine" problem
    for this fork's own test suite.
    """
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    if not torch.distributed.is_initialized():
        fd, temp_file = tempfile.mkstemp()
        os.close(fd)
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="nccl" if device.type == "cuda" else "gloo",
        )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )


@dataclass
class LookaheadMetadata:
    """Everything `run_lookahead_steps` needs for one single-request
    lookahead run, built by `SpecPrefillProposer.build_lookahead_metadata`.

    `prefill_attn_metadata`/`prefill_slot_mapping` are for the ONE bootstrap
    prefill call only (seeds the KV cache over the full prompt, produces the
    first candidate token -- not itself a scored lookahead step, see
    `run_lookahead_steps`'s docstring). `per_step_attn_metadata`/
    `per_step_slot_mapping` are unambiguously decode-only: exactly
    `look_ahead_cnt` entries, each shaped for a single new token appended to
    a growing context. `slot_mapping` (bare tensor, distinct from
    `prefill_slot_mapping`'s dict form) is for K-retrieval only (see
    `retrieve_qk`) -- always the prompt's own slots, unaffected by any of
    the above, since scoring always attends back to the prompt's keys, never
    the lookahead steps' own generated keys.
    """

    prefill_attn_metadata: Dict[str, Any]
    prefill_slot_mapping: Dict[str, torch.Tensor]
    per_step_attn_metadata: List[Any]
    per_step_slot_mapping: List[dict]
    slot_mapping: torch.Tensor
    block_size: int
    num_kv_heads: int
    head_dim: int


def _query_capturing_qwen3_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    query_buffer: List[List[torch.Tensor]],
    layer_idx: int,
    **kwargs,
) -> torch.Tensor:
    """Instance-level replacement for `Qwen3Attention.forward`
    (vllm/model_executor/models/qwen3.py:145-162) that additionally stashes
    the post-RoPE query tensor into `query_buffer[layer_idx]`.

    This is a faithful copy of that method's body with one addition (the
    `query_buffer[layer_idx].append(...)` line before `self.attn(...)`), not
    a generic wrapper -- capturing `q` requires intercepting mid-forward,
    since it's a local variable, not something the unmodified method
    returns. MUST be kept in sync with qwen3.py's Qwen3Attention.forward if
    that method's body changes.

    Bound via `types.MethodType` to individual `Qwen3Attention` *instances*
    on the speculator model only (see `install_query_capture_hooks` below)
    -- never assigned at the class level, since the target (8B) model uses
    the identical `Qwen3Attention` class and a class-level patch would
    corrupt its forward too.

    This is the file's one real architecture-specific rewrite (ported from
    `spec_prefill/vllm_patch/proposer.py`'s
    `_query_capturing_gemma4_attention_forward`, which mirrored
    `Gemma4Attention.forward` instead): `Qwen3Attention.forward` has no
    `is_kv_shared_layer`/`use_k_eq_v` branching and no separate `v_norm` --
    Qwen3 splits qkv in one shot and applies RMSNorm to q and k only,
    uniformly at every layer, so the branching those Gemma4-only concepts
    required is simply absent here.
    """
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
    q_by_head = self.q_norm(q_by_head)
    q = q_by_head.view(q.shape)
    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
    k_by_head = self.k_norm(k_by_head)
    k = k_by_head.view(k.shape)

    q, k = self.rotary_emb(positions, q, k)

    query_buffer[layer_idx].append(q.detach())

    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


class SpecPrefillProposer:
    def __init__(
        self,
        base_vllm_config: VllmConfig,
        speculator_model_config: ModelConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = self._create_speculator_vllm_config(
            base_vllm_config, speculator_model_config
        )
        self.device = device

        # initialize_model_parallel() (inside _ensure_distributed_environment)
        # reads get_current_vllm_config() internally, so both it and the
        # model load itself must run inside this context -- confirmed on
        # real hardware (AssertionError: Current vLLM config is not set,
        # otherwise).
        with set_current_vllm_config(self.vllm_config):
            _ensure_distributed_environment(device)
            self.model: nn.Module = get_model(
                vllm_config=self.vllm_config, prefix="spec_prefill_speculator"
            )

        self._speculator_layers = self._find_qwen3_attention_layers(self.model)
        self._num_layers = len(self._speculator_layers)
        self._layer_names = [layer.attn.layer_name for layer in self._speculator_layers]
        self._query_buffer: List[List[torch.Tensor]] = []
        self.install_query_capture_hooks()

    @staticmethod
    def _create_speculator_vllm_config(
        base_vllm_config: VllmConfig, speculator_model_config: ModelConfig
    ) -> VllmConfig:
        """Independent VllmConfig for the speculator, modeled on
        DraftModelProposer._create_draft_vllm_config (draft_model.py:54-66)
        -- own model_config, no inherited quantization, backend
        auto-selected independently rather than inherited from the base
        model (kept independent for the same reason DraftModelProposer does
        -- the speculator's own attention shape should drive its own backend
        selection, not the target's).

        Forces `enforce_eager=True` -- the original protocol document's own
        stated requirement ("run with enforce_eager=True,
        enable_chunked_prefill=False"), and confirmed necessary on real
        hardware (for the sibling Gemma4 pipeline) for a different reason
        too: without it, `Qwen3Model`'s `@support_torch_compile`-wrapped
        forward tries to torch.compile/Dynamo-trace through the
        `functools.partial`-wrapped query-capture hook
        (`_query_capturing_qwen3_attention_forward`) installed below, and
        Dynamo can't trace through a `functools.partial` object
        (`torch._dynamo.exc.Unsupported: can't handle functions not
        implemented in python`). Setting `model_config.enforce_eager = True`
        makes `VllmConfig`'s own post-init logic set
        `compilation_config.mode = CompilationMode.NONE` and
        `cudagraph_mode = CUDAGraphMode.NONE` (`vllm/config/vllm.py:1019-1025`),
        avoiding Dynamo entirely rather than trying to make the hook
        Dynamo-traceable. This reasoning is generic to any
        `@support_torch_compile`-wrapped model, not specific to Qwen3 or
        Gemma4.

        Uses a shallow copy + direct attribute set rather than vLLM's own
        `replace()` for this specific field -- confirmed on real hardware
        that `replace(speculator_model_config, enforce_eager=True)` raises
        `ValueError: Field 'model_arch_config' not found in ModelConfig`
        (vLLM's `vllm.config.utils.replace`, not stdlib `dataclasses.replace`,
        chokes on a `model_arch_config` instance attribute that isn't a
        declared dataclass field). `ModelConfig` uses vLLM's own `@config(...)`
        wrapper, not a frozen dataclass, and is mutated in place elsewhere in
        this codebase -- direct attribute assignment on a copy is consistent
        with that convention, not a hack specific to this file."""
        speculator_model_config = copy.copy(speculator_model_config)
        speculator_model_config.enforce_eager = True
        return replace(
            base_vllm_config,
            model_config=speculator_model_config,
            quant_config=None,
        )

    @staticmethod
    def _find_qwen3_attention_layers(model: nn.Module) -> List[nn.Module]:
        """Locate the speculator's Qwen3Attention layers via the known
        Qwen3ForCausalLM/Qwen3Model structure (model.model.layers[i]
        .self_attn), rather than a generic layer registry -- SpecPrefill's
        scope is Qwen3-1.7B specifically (see EXPERIMENT_PLAN.md).

        Ported from `spec_prefill/vllm_patch/proposer.py`'s
        `_find_gemma4_attention_layers`, which had to unwrap a multimodal
        wrapper class (`Gemma4ForConditionalGeneration`) via
        `supports_multimodal()` + `.get_language_model()` before reaching
        the real decoder layers, because Gemma-4-E2B-it always loads as that
        wrapper. Qwen3-1.7B/8B are plain, non-multimodal `Qwen3ForCausalLM`
        (confirmed via `vllm/model_executor/models/qwen3.py:281-292`:
        `Qwen3ForCausalLM.model` is a `Qwen3Model` directly), so that unwrap
        step is dropped entirely here rather than kept as unreachable
        dead code.
        """
        model_name = type(model).__name__
        if "Qwen3" not in model_name:
            raise NotImplementedError(
                f"SpecPrefillProposer currently only supports Qwen3 models "
                f"as the speculator (got {model_name}). The query-capture "
                f"hook (_query_capturing_qwen3_attention_forward) is a "
                f"faithful copy of Qwen3Attention.forward's body and is "
                f"not architecture-generic."
            )

        inner = model.model if hasattr(model, "model") else model
        if not hasattr(inner, "layers"):
            raise NotImplementedError(
                f"Could not locate decoder layers on {model_name} -- "
                f"neither .model.layers nor .layers resolved. This model's "
                f"structure doesn't match the known Qwen3ForCausalLM shape."
            )
        return [layer.self_attn for layer in inner.layers]

    def install_query_capture_hooks(self) -> None:
        """Bind the query-capturing forward per-instance onto the
        speculator's own attention layer objects only (see the forward
        function's docstring for why this must not be class-level)."""
        self._query_buffer = [[] for _ in range(self._num_layers)]
        for layer_idx, self_attn in enumerate(self._speculator_layers):
            self_attn.forward = types.MethodType(
                partial(
                    _query_capturing_qwen3_attention_forward,
                    query_buffer=self._query_buffer,
                    layer_idx=layer_idx,
                ),
                self_attn,
            )

    def reset_query_buffer(self) -> None:
        for buf in self._query_buffer:
            buf.clear()

    def run_lookahead_steps(
        self,
        initial_input_ids: torch.Tensor,
        initial_positions: torch.Tensor,
        look_ahead_cnt: int,
        prefill_attn_metadata: Dict[str, Any],
        prefill_slot_mapping: Dict[str, torch.Tensor],
        per_step_attn_metadata: List[Any],
        per_step_slot_mapping: List[dict],
        next_input_fn: Callable[[torch.Tensor], torch.Tensor],
        next_positions_fn: Callable[[torch.Tensor, int], torch.Tensor],
        eos_token_id: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Algorithm lines 3-7: run the speculator for `look_ahead_cnt`
        steps, buffering its post-RoPE queries via the installed hook.

        **Root cause of a real bug fixed 2026-07-23 in the sibling Gemma4
        pipeline (confirmed via direct tensor-shape simulation, not
        inferred) -- the fix applies identically here, since it's about
        lookahead-step accounting, not model architecture**: this method
        used to treat the initial full-prompt prefill as "lookahead step 0"
        itself, capturing its query alongside the genuine decode steps. That
        query tensor has shape [prompt_len, H*D] (one entry per PROMPT
        token), while every real decode step captures [1, H*D] (one new
        token) -- for look_ahead_cnt>1 the final `torch.stack(dim=1)` below
        raised `RuntimeError: stack expects each tensor to be equal size`
        outright; for look_ahead_cnt==1 it "succeeded" but produced
        [prompt_len, 1, H*D], and scoring.compute_attention_score's
        `zip(queries, keys, actual_look_ahead_cnts)` (lengths prompt_len, 1,
        1) silently truncated to 1 pair -- the FIRST prompt token's own
        query (causally blind to almost the whole prompt) -- discarding
        every other prompt token's captured query with no error. Per
        EXPERIMENT_PLAN.md's own algorithm description ("prefills the full
        prompt, THEN runs a few lookahead decode steps, capturing... during
        THOSE steps"), the prefill was never supposed to be a scored step.
        Fixed by running it as a separate, uncaptured bootstrap call (see
        below) -- every captured step is now uniformly [1, H*D], for any
        look_ahead_cnt.

        Per-step attention metadata/slot mapping are supplied by the caller
        (see module docstring's scope boundary) -- this method only handles
        the forward-call loop and EOS early-stop (line 6), not KV-cache-slot
        bookkeeping between steps.

        Args:
            prefill_attn_metadata/prefill_slot_mapping: for the ONE
                bootstrap prefill call (prompt-shaped) -- NOT one of the
                `look_ahead_cnt` scored steps; its query-buffer contribution
                is discarded immediately after (see body).
            next_input_fn: given the previous step's sampled token ids,
                returns the next step's input_ids (greedy next-token, or a
                caller-supplied sampling strategy).
            next_positions_fn: given the current positions and the decode
                step index (0-indexed, the step ABOUT to run), returns that
                step's positions tensor. Called once per decode iteration,
                starting at step=0 for the first token after the prompt --
                NOT called for the bootstrap prefill.

        Returns:
            Per-layer query_buffer, each entry stacked to
            [1, actual_look_ahead_steps, num_heads*head_dim] -- num_samples
            is always 1 here (one prompt per call) -- the shape
            scoring.compute_attention_score expects. Empty ([1, 0, H*D]) if
            zero decode steps actually ran (look_ahead_cnt=0, or EOS fired
            on the bootstrap's own sampled token) -- callers must check for
            this (see pruner.py's zero-lookahead-steps guard) rather than
            feed it into scoring, since aggregating over an empty step axis
            produces silent NaN/garbage, not an error.
        """
        assert len(per_step_attn_metadata) == look_ahead_cnt
        assert len(per_step_slot_mapping) == look_ahead_cnt

        # Bootstrap prefill: seeds the KV cache over the full prompt,
        # produces the first candidate continuation token. NOT a scored
        # lookahead step -- see docstring above for why conflating this
        # with a scored step was the root cause of the bugs fixed here.
        self.reset_query_buffer()
        with set_forward_context(
            prefill_attn_metadata,
            self.vllm_config,
            num_tokens=initial_input_ids.shape[0],
            slot_mapping=prefill_slot_mapping,
        ):
            hidden_states = self.model(input_ids=initial_input_ids, positions=initial_positions)
        # Slice to the LAST position before compute_logits, not after --
        # confirmed on real hardware for the sibling Gemma4 pipeline
        # (2026-07-24): computing logits over every one of the prompt's
        # teacher-forced positions (the previous
        # `self.model.compute_logits(hidden_states)`, unsliced) projects the
        # full [prompt_len, hidden_size] tensor through the LM head to
        # [prompt_len, vocab_size] -- only the LAST row was ever used (see
        # comment below), but for a real LongBench-v2-scale prompt (tens of
        # thousands of tokens) against a large vocabulary, that wasted
        # projection alone can be tens of GB and OOM on a GPU that has
        # already loaded the speculator. Slicing first reduces the
        # projection to a single position -- identical result (next_input_fn
        # below already re-slices to `[-1:]` regardless, so this changes
        # nothing about correctness, only the wasted compute/memory) -- this
        # reasoning is generic to any model's vocab size, not re-profiled
        # specifically against Qwen3-1.7B's own vocab here.
        next_token_ids = self.model.compute_logits(hidden_states[-1:]).argmax(dim=-1)
        self.reset_query_buffer()  # discard the bootstrap's own capture

        # next_token_ids here has shape [1] now (see slicing above) -- the
        # single real autoregressive continuation; earlier prompt positions
        # were teacher-forced during prefill and their predictions were
        # never meaningful here regardless.
        bootstrap_eos = eos_token_id is not None and bool(
            next_token_ids[-1] == eos_token_id
        )
        if look_ahead_cnt == 0 or bootstrap_eos:
            hd = self._speculator_layers[0].num_heads * self._speculator_layers[0].head_dim
            return [torch.empty(1, 0, hd, device=self.device) for _ in range(self._num_layers)]

        input_ids = next_input_fn(next_token_ids)
        positions = initial_positions

        for step in range(look_ahead_cnt):
            positions = next_positions_fn(positions, step)
            with set_forward_context(
                per_step_attn_metadata[step],
                self.vllm_config,
                num_tokens=input_ids.shape[0],
                slot_mapping=per_step_slot_mapping[step],
            ):
                hidden_states = self.model(input_ids=input_ids, positions=positions)

            next_token_ids = self.model.compute_logits(hidden_states).argmax(dim=-1)

            if eos_token_id is not None and bool(
                torch.all(next_token_ids == eos_token_id)
            ):
                break

            input_ids = next_input_fn(next_token_ids)

        return [torch.stack(layer_steps, dim=1) for layer_steps in self._query_buffer]

    def tp_gather_qk(self, query_buffer: List[torch.Tensor]) -> List[torch.Tensor]:
        """Algorithm lines 8-10: if tensor-parallel, gather Q across ranks.

        Uses `get_tp_group().all_gather(...)` (the idiomatic V1-era call --
        v0's `tensor_model_parallel_gather` has no V1 call sites in this
        fork, see EXPERIMENT_PLAN.md). Gathers along the last dim
        (num_heads*head_dim), where each TP rank holds a disjoint slice of
        heads.
        """
        if get_tensor_model_parallel_world_size() <= 1:
            return query_buffer
        tp_group = get_tp_group()
        return [tp_group.all_gather(q, dim=-1) for q in query_buffer]

    def retrieve_qk(
        self,
        query_buffer: List[torch.Tensor],
        per_sample_slot_mapping: List[torch.Tensor],
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ):
        """Algorithm line 11: Q, K <- retrieve_qk(B_p, C_s).

        Q is already buffered (post tp_gather_qk); K is read back from the
        speculator's own KV cache per layer via kv_cache_utils.py.

        `head_size` is accepted for interface-compatibility with callers
        (`build_lookahead_metadata`'s caller passes one head_dim value), but
        is **not used per-layer** -- reading each layer's own
        `self_attn.head_dim` instead of a single global value was a
        real-hardware-confirmed fix for the sibling Gemma4 pipeline, where
        Gemma-4-E2B-it's head_dim is heterogeneous (256 sliding-attention /
        512 full-attention layers), so K retrieval had to read each layer's
        OWN head_dim to avoid `RuntimeError: stack expects each tensor to be
        equal size` in scoring.compute_attention_score. Qwen3-1.7B's
        head_dim/num_kv_heads are uniform across layers by construction
        (`Qwen3Attention.__init__` computes a single `head_dim`/
        `num_kv_heads` per instance, not derived per-layer from a
        heterogeneous config) -- not independently confirmed on real
        hardware here, verify via `validate_proposer.py`'s Step A before
        relying on it. Reading per-layer here is therefore a no-op
        simplification of the general case for Qwen3, not dead code: it
        still produces the correct result even if that uniformity
        assumption turns out to be wrong. `num_kv_heads`/`block_size` stay
        global, consistent with that.

        Returns:
            (query_buffer, key_buffer) where key_buffer[layer_idx] is a list
            of per-sample [context_len, num_kv_heads, head_size] tensors --
            together, the exact inputs scoring.compute_attention_score
            expects.
        """
        key_buffer = [
            retrieve_keys_per_sample(
                self_attn.attn, block_size, num_kv_heads, self_attn.head_dim, per_sample_slot_mapping
            )
            for self_attn in self._speculator_layers
        ]
        return query_buffer, key_buffer

    def build_lookahead_metadata(
        self, prompt_len: int, look_ahead_cnt: int, head_dim: int
    ) -> LookaheadMetadata:
        """Build real attention metadata plus a dummy KV cache for running
        `run_lookahead_steps`/`retrieve_qk` against a single request of
        length `prompt_len`: one bootstrap-prefill metadata entry (shaped
        for `prompt_len` query tokens) plus `look_ahead_cnt` decode-step
        metadata entries (each shaped for exactly 1 new query token,
        appended to a growing context) -- see `LookaheadMetadata`'s own
        docstring for the field split, and `run_lookahead_steps`'s
        docstring for why the prefill must not be treated as a scored step.

        **Slot-mapping fix (2026-07-23, confirmed against this fork's actual
        source, not assumed -- for the sibling Gemma4 pipeline, but this is
        a property of `create_common_attn_metadata` itself, not of any
        model)**: `create_common_attn_metadata(...,
        arange_block_indices=True)` assigns block ids as literal ascending
        integers (`block_table_tensor = arange(max_blocks)`) fresh per call
        -- this stays physically consistent across the bootstrap + N decode
        calls below ONLY because each call passes the FULL CUMULATIVE
        `seq_lens` (not an incremental delta), so `max_blocks` grows
        monotonically and block `b` always refers to the same physical
        block in the one persistent dummy cache bound below. Its own
        `slot_mapping` (`arange(num_tokens)`) is NOT similarly
        self-correcting -- for a decode call it's always `arange(1) = [0]`
        regardless of context, so it's manually overridden per decode step
        to the true absolute logical position via `.replace(slot_mapping=
        ...)`. This exploits the identity `slot == logical position` that
        holds under this scheme (block id == block number by construction,
        so `slot = block_id*block_size + offset = logical_position`) -- not
        a general-purpose slot-computation utility. The KV-cache *write*
        itself is driven by the `slot_mapping=` dict passed to
        `set_forward_context` in `run_lookahead_steps` (confirmed via
        `Attention.forward` -> `unified_kv_cache_update` ->
        `get_forward_context().slot_mapping[layer_name]`), which is built
        from this same overridden value below -- so this override is what
        actually determines where each decode step's new K/V land.

        Uses this fork's own proven test utilities
        (`tests/v1/attention/utils.py` -- `create_common_attn_metadata`,
        `create_standard_kv_cache_spec`, `try_get_attention_backend`,
        `create_dummy_kv_cache`) rather than hand-rolled metadata, since even
        this fork's own tests avoid constructing `CommonAttentionMetadata`/
        KV-cache tensors from scratch by any other means. See
        `validate_proposer.py`'s module docstring for the same reasoning
        (this method factors out what was previously duplicated inline
        there).

        This allocates and binds a dummy KV cache directly to the
        speculator's own attention layers -- deliberately not real
        dual-model cache-budget management between the speculator and the
        target model, which is out of scope for this pass (see the approved
        plan's "Scope for this pass"). Every call re-allocates a fresh dummy
        cache sized for this request: `prompt_len + look_ahead_cnt` tokens
        (the prompt plus every decode step's own new token) -- not reused
        across requests. Scaling this for arbitrarily long real prompts
        (LongBench v2's up-to-32k-word documents) is a separate, deferred
        concern, same scoping spirit as the target-model chunked-prefill fix
        being scoped to its own mechanism.
        """
        _ensure_vllm_repo_root_on_syspath()
        from tests.v1.attention.utils import (  # proven pattern, see docstring
            BatchSpec,
            create_common_attn_metadata,
            create_dummy_kv_cache,
            create_standard_kv_cache_spec,
            try_get_attention_backend,
        )

        block_size = self.vllm_config.cache_config.block_size
        num_kv_heads = self._speculator_layers[0].num_kv_heads
        # Exact fit: the last decode step's seq_lens == prompt_len +
        # look_ahead_cnt, so this is the true max block count needed --
        # verified arithmetic, no slack/off-by-one.
        num_blocks_needed = -(-(prompt_len + look_ahead_cnt) // block_size)

        kv_cache_spec = create_standard_kv_cache_spec(self.vllm_config)
        for self_attn in self._speculator_layers:
            # Confirmed on real hardware for the sibling Gemma4 pipeline:
            # allocating every layer's dummy cache with a single global
            # head_dim (the `head_dim` parameter, typically caller-supplied
            # from layer 0) was wrong there -- Gemma-4-E2B-it's
            # full-attention layers (head_dim=512) would get a cache tensor
            # sized for its sliding-attention layers' 256, undersized for
            # what their real forward pass writes. Each layer's own
            # self_attn.head_dim is used instead, per layer -- for
            # Qwen3-1.7B (uniform head_dim across layers, not independently
            # confirmed here) this degenerates to every layer getting the
            # same size, so no code change is needed, only this rationale.
            dummy_cache = create_dummy_kv_cache(
                block_size,
                num_kv_heads,
                self_attn.head_dim,
                self.vllm_config.model_config.dtype,
                self.device,
                num_blocks=num_blocks_needed,
            )
            self_attn.attn.kv_cache = dummy_cache

        # self.vllm_config.attention_config.backend is only populated when a
        # model-specific config hook force-sets it (e.g. Gemma4Config.
        # verify_and_update_config's heterogeneous-head_dim TRITON_ATTN
        # forcing, vllm/model_executor/models/config.py:88-99) -- for Qwen3
        # (no such hook, uniform head_dim) it stays None, which
        # try_get_attention_backend can't resolve (`AttributeError:
        # 'NoneType' object has no attribute 'get_class'`, confirmed on real
        # hardware 2026-07-28). This was latent, unexercised code even for
        # the sibling Gemma4 pipeline -- its own validate_proposer.py Step B
        # only runs when head_dim is heterogeneous, which is exactly the
        # condition that happens to force this field via the hook above, so
        # this line was never actually hit there either. Read the
        # ALREADY-RESOLVED backend off a loaded attention layer instead --
        # `Attention.__init__` always sets `self.backend` via
        # get_attn_backend(...) regardless of whether the config override
        # was ever set (vllm/model_executor/layers/attention/attention.py:386)
        # -- robust for any model, not just ones with a forcing hook.
        backend_enum = self._speculator_layers[0].attn.backend
        builder_cls, _ = try_get_attention_backend(backend_enum)
        builder = builder_cls(
            kv_cache_spec=kv_cache_spec,
            layer_names=self._layer_names,
            vllm_config=self.vllm_config,
            device=self.device,
        )

        # Bootstrap prefill metadata: unchanged shape from before this fix --
        # the full prompt, in one chunk.
        prefill_batch_spec = BatchSpec(seq_lens=[prompt_len], query_lens=[prompt_len])
        prefill_common_attn_metadata = create_common_attn_metadata(
            prefill_batch_spec, block_size=block_size, device=self.device,
            arange_block_indices=True,
        )
        prefill_attn_metadata_built = builder.build(
            common_prefix_len=0, common_attn_metadata=prefill_common_attn_metadata
        )

        # Decode-step metadata: query_lens=[1], seq_lens growing by 1 each
        # step, slot_mapping manually corrected -- see method docstring's
        # "Slot-mapping fix" section for why create_common_attn_metadata's
        # own arange(num_tokens)=[0] is wrong for a decode continuation.
        per_step_attn_metadata = []
        per_step_slot_mapping = []
        for step in range(look_ahead_cnt):
            new_token_position = prompt_len + step  # 0-indexed logical position
            decode_batch_spec = BatchSpec(
                seq_lens=[new_token_position + 1], query_lens=[1]
            )
            decode_common_attn_metadata = create_common_attn_metadata(
                decode_batch_spec, block_size=block_size, device=self.device,
                arange_block_indices=True,
            ).replace(
                slot_mapping=torch.tensor(
                    [new_token_position], dtype=torch.int64, device=self.device
                )
            )
            decode_attn_metadata = builder.build(
                common_prefix_len=0, common_attn_metadata=decode_common_attn_metadata
            )
            per_step_attn_metadata.append(
                {name: decode_attn_metadata for name in self._layer_names}
            )
            per_step_slot_mapping.append(
                {name: decode_common_attn_metadata.slot_mapping for name in self._layer_names}
            )

        return LookaheadMetadata(
            prefill_attn_metadata={name: prefill_attn_metadata_built for name in self._layer_names},
            prefill_slot_mapping={
                name: prefill_common_attn_metadata.slot_mapping for name in self._layer_names
            },
            per_step_attn_metadata=per_step_attn_metadata,
            per_step_slot_mapping=per_step_slot_mapping,
            slot_mapping=prefill_common_attn_metadata.slot_mapping,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
