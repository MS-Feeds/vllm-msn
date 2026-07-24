"""SpecPrefillProposer — Algorithm 1, lines 3-11.

Loads the speculator (Gemma-4-E2B-it) as a standalone model with its own
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
    `benchmarks/spec_prefill` (which every `vllm_patch` caller script adds
    for its own imports -- confirmed on real hardware: `ModuleNotFoundError:
    No module named 'tests'` without this, running from
    `benchmarks/spec_prefill/`).

    Computed relative to this file's own location
    (`benchmarks/spec_prefill/vllm_patch/proposer.py`, 3 parents up to repo
    root) rather than relying on every caller script to add it themselves.
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
    `Gemma4Attention.__init__` (`get_tensor_model_parallel_world_size()`/
    `get_tensor_model_parallel_rank()`) and `tp_gather_qk` (`get_tp_group()`)
    require it, and calling either without it raises
    `AssertionError: tensor model parallel group is not initialized`
    (confirmed on real hardware running `validate_proposer.py` standalone).

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


def _query_capturing_gemma4_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    query_buffer: List[List[torch.Tensor]],
    layer_idx: int,
    **kwargs,
) -> torch.Tensor:
    """Instance-level replacement for `Gemma4Attention.forward`
    (vllm/model_executor/models/gemma4.py:510-553) that additionally stashes
    the post-RoPE query tensor into `query_buffer[layer_idx]`.

    This is a faithful copy of that method's body with one addition (the
    `query_buffer[layer_idx].append(...)` line before `self.attn(...)`), not
    a generic wrapper -- capturing `q` requires intercepting mid-forward,
    since it's a local variable, not something the unmodified method
    returns. MUST be kept in sync with gemma4.py's Gemma4Attention.forward
    if that method's body changes.

    Bound via `types.MethodType` to individual `Gemma4Attention` *instances*
    on the speculator model only (see `install_query_capture_hooks` below)
    -- never assigned at the class level, since the target (26B MoE) model
    uses the identical `Gemma4Attention` class and a class-level patch would
    corrupt its forward too.
    """
    qkv, _ = self.qkv_proj(hidden_states)

    q = qkv[..., : self.q_size]
    q = q.unflatten(-1, (self.num_heads, self.head_dim))
    q = self.q_norm(q)
    q = q.flatten(-2, -1)

    if not self.is_kv_shared_layer:
        if self.use_k_eq_v:
            k = qkv[..., self.q_size :]
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(k).flatten(-2, -1)
            k = self.k_norm(k).flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)
        else:
            k = qkv[..., self.q_size : self.q_size + self.kv_size]
            v = qkv[..., self.q_size + self.kv_size :]
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k).flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)
            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v).flatten(-2, -1)
    else:
        k = qkv[..., self.q_size : self.q_size + self.kv_size]
        v = k if self.use_k_eq_v else qkv[..., self.q_size + self.kv_size :]
        q = self.rotary_emb(positions, q, k)[0]

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

        self._speculator_layers = self._find_gemma4_attention_layers(self.model)
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
        model (Gemma4's TRITON_ATTN forcing is head-dim-dependent and must
        be re-derived for the speculator's own config, see gemma4.py's
        `Gemma4Config.verify_and_update_config`).

        Forces `enforce_eager=True` -- the original protocol document's own
        stated requirement ("run with enforce_eager=True,
        enable_chunked_prefill=False"), and confirmed necessary on real
        hardware for a different reason too: without it, `Gemma4Model`'s
        `@support_torch_compile`-wrapped forward tries to torch.compile/
        Dynamo-trace through the `functools.partial`-wrapped query-capture
        hook (`_query_capturing_gemma4_attention_forward`) installed below,
        and Dynamo can't trace through a `functools.partial` object
        (`torch._dynamo.exc.Unsupported: can't handle functions not
        implemented in python`). Setting `model_config.enforce_eager = True`
        makes `VllmConfig`'s own post-init logic set
        `compilation_config.mode = CompilationMode.NONE` and
        `cudagraph_mode = CUDAGraphMode.NONE` (`vllm/config/vllm.py:1019-1025`),
        avoiding Dynamo entirely rather than trying to make the hook
        Dynamo-traceable.

        Uses a shallow copy + direct attribute set rather than vLLM's own
        `replace()` for this specific field -- confirmed on real hardware
        that `replace(speculator_model_config, enforce_eager=True)` raises
        `ValueError: Field 'model_arch_config' not found in ModelConfig`
        (vLLM's `vllm.config.utils.replace`, not stdlib `dataclasses.replace`,
        chokes on a `model_arch_config` instance attribute that isn't a
        declared dataclass field). `ModelConfig` uses vLLM's own `@config(...)`
        wrapper, not a frozen dataclass, and is mutated in place elsewhere in
        this codebase (e.g. `Gemma4Config.verify_and_update_config` mutates
        `vllm_config.attention_config.backend` directly) -- direct attribute
        assignment on a copy is consistent with that convention, not a hack
        specific to this file."""
        speculator_model_config = copy.copy(speculator_model_config)
        speculator_model_config.enforce_eager = True
        return replace(
            base_vllm_config,
            model_config=speculator_model_config,
            quant_config=None,
        )

    @staticmethod
    def _find_gemma4_attention_layers(model: nn.Module) -> List[nn.Module]:
        """Locate the speculator's Gemma4Attention layers via the known
        Gemma4ForCausalLM/Gemma4Model structure (model.model.layers[i]
        .self_attn), rather than a generic layer registry -- SpecPrefill's
        scope is Gemma-4-E2B-it specifically (see EXPERIMENT_PLAN.md).

        **Confirmed on real hardware**: `get_model()` loads Gemma-4-E2B-it as
        `Gemma4ForConditionalGeneration` (the multimodal-capable wrapper
        class -- both the full and the "text-only" checkpoint variants route
        here, per `gemma4_mm.py`'s guardrail comments; there is no
        checkpoint config that yields plain `Gemma4ForCausalLM` directly),
        not `Gemma4ForCausalLM` as originally assumed. Its real text stack
        lives under `.language_model` (itself a `Gemma4ForCausalLM`), not
        `.model` -- unwrap via `supports_multimodal()` +
        `.get_language_model()`, mirroring the exact pattern this fork's own
        `vllm/v1/spec_decode/llm_base_proposer.py:1213-1247` already uses
        for this same class when the MTP/draft-model proposer loads a
        Gemma4 target model."""
        from vllm.model_executor.models import supports_multimodal

        model_name = type(model).__name__
        if "Gemma4" not in model_name:
            raise NotImplementedError(
                f"SpecPrefillProposer currently only supports Gemma4 models "
                f"as the speculator (got {model_name}). The query-capture "
                f"hook (_query_capturing_gemma4_attention_forward) is a "
                f"faithful copy of Gemma4Attention.forward's body and is "
                f"not architecture-generic."
            )

        if supports_multimodal(model):
            # e.g. Gemma4ForConditionalGeneration -- unwrap to the real
            # Gemma4ForCausalLM text stack first.
            text_model = model.get_language_model()
        else:
            text_model = model

        inner = text_model.model if hasattr(text_model, "model") else text_model
        if not hasattr(inner, "layers"):
            raise NotImplementedError(
                f"Could not locate decoder layers on {model_name} -- neither "
                f".get_language_model().model.layers nor .model.layers nor "
                f".layers resolved. This model's structure doesn't match "
                f"either known Gemma4 wrapper shape (Gemma4ForCausalLM or "
                f"Gemma4ForConditionalGeneration)."
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
                    _query_capturing_gemma4_attention_forward,
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

        **Root cause of a real bug fixed 2026-07-23 (confirmed via direct
        tensor-shape simulation, not inferred)**: this method used to treat
        the initial full-prompt prefill as "lookahead step 0" itself,
        capturing its query alongside the genuine decode steps. That query
        tensor has shape [prompt_len, H*D] (one entry per PROMPT token),
        while every real decode step captures [1, H*D] (one new token) --
        for look_ahead_cnt>1 the final `torch.stack(dim=1)` below raised
        `RuntimeError: stack expects each tensor to be equal size` outright;
        for look_ahead_cnt==1 it "succeeded" but produced [prompt_len, 1,
        H*D], and scoring.compute_attention_score's `zip(queries, keys,
        actual_look_ahead_cnts)` (lengths prompt_len, 1, 1) silently
        truncated to 1 pair -- the FIRST prompt token's own query (causally
        blind to almost the whole prompt) -- discarding every other prompt
        token's captured query with no error. Per EXPERIMENT_PLAN.md's own
        algorithm description ("prefills the full prompt, THEN runs a few
        lookahead decode steps, capturing... during THOSE steps"), the
        prefill was never supposed to be a scored step. Fixed by running it
        as a separate, uncaptured bootstrap call (see below) -- every
        captured step is now uniformly [1, H*D], for any look_ahead_cnt.

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
        # confirmed on real hardware (2026-07-24): computing logits over
        # every one of the prompt's teacher-forced positions (the previous
        # `self.model.compute_logits(hidden_states)`, unsliced) projects the
        # full [prompt_len, hidden_size] tensor through the LM head to
        # [prompt_len, vocab_size] -- only the LAST row was ever used (see
        # comment below), but for a real LongBench-v2-scale prompt (tens of
        # thousands of tokens) against Gemma4's ~256k vocab, that wasted
        # projection alone is tens of GB and OOM'd on an 80GB GPU that had
        # already loaded the speculator. Slicing first reduces the
        # projection to a single position -- identical result (next_input_fn
        # below already re-slices to `[-1:]` regardless, so this changes
        # nothing about correctness, only the wasted compute/memory).
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
        is **not used per-layer** -- confirmed on real hardware that using a
        single global head_size here is a real bug: Gemma4's head_dim is
        heterogeneous (256 sliding-attention / 512 full-attention layers),
        so K retrieval must read each layer's OWN `self_attn.head_dim`
        (`RuntimeError: stack expects each tensor to be equal size` in
        scoring.compute_attention_score otherwise -- layer 4's K was
        reinterpreted with layer 0's head_dim). `num_kv_heads`/`block_size`
        are uniform across layers (confirmed: num_kv_heads=1 for every layer
        in validate_proposer.py's inspection) and stay global.

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
        source, not assumed)**: `create_common_attn_metadata(...,
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
            # Confirmed on real hardware: allocating every layer's dummy
            # cache with a single global head_dim (the `head_dim` parameter,
            # typically caller-supplied from layer 0) is wrong for Gemma4 --
            # full-attention layers (head_dim=512) would get a cache tensor
            # sized for the sliding-attention layers' 256, undersized for
            # what their real forward pass writes. Each layer's own
            # self_attn.head_dim is used instead, per layer.
            dummy_cache = create_dummy_kv_cache(
                block_size,
                num_kv_heads,
                self_attn.head_dim,
                self.vllm_config.model_config.dtype,
                self.device,
                num_blocks=num_blocks_needed,
            )
            self_attn.attn.kv_cache = dummy_cache

        backend_enum = self.vllm_config.attention_config.backend
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
