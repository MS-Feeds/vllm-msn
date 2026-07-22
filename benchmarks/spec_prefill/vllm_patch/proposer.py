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
from typing import Any, Callable, List, Optional

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
    """

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
        num_tokens: int,
        look_ahead_cnt: int,
        per_step_attn_metadata: List[Any],
        per_step_slot_mapping: List[dict],
        next_input_fn: Callable[[torch.Tensor], torch.Tensor],
        next_positions_fn: Callable[[torch.Tensor, int], torch.Tensor],
        eos_token_id: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Algorithm lines 3-7: run the speculator for `look_ahead_cnt`
        steps, buffering its post-RoPE queries via the installed hook.

        Per-step attention metadata/slot mapping are supplied by the caller
        (see module docstring's scope boundary) -- this method only handles
        the forward-call loop and EOS early-stop (line 6), not KV-cache-slot
        bookkeeping between steps.

        Args:
            next_input_fn: given the previous step's sampled token ids,
                returns the next step's input_ids (greedy next-token, or a
                caller-supplied sampling strategy).
            next_positions_fn: given the previous positions and the step
                index, returns the next step's positions tensor.

        Returns:
            Per-layer query_buffer, each entry stacked to
            [num_prefill_samples, actual_look_ahead_steps, num_heads*head_dim]
            -- the shape scoring.compute_attention_score expects.
        """
        assert len(per_step_attn_metadata) == look_ahead_cnt
        assert len(per_step_slot_mapping) == look_ahead_cnt

        self.reset_query_buffer()
        input_ids = initial_input_ids
        positions = initial_positions

        for step in range(look_ahead_cnt):
            with set_forward_context(
                per_step_attn_metadata[step],
                self.vllm_config,
                num_tokens=num_tokens,
                slot_mapping=per_step_slot_mapping[step],
            ):
                hidden_states = self.model(input_ids=input_ids, positions=positions)

            next_token_ids = self.model.compute_logits(hidden_states).argmax(dim=-1)

            if eos_token_id is not None and bool(
                torch.all(next_token_ids == eos_token_id)
            ):
                break

            input_ids = next_input_fn(next_token_ids)
            positions = next_positions_fn(positions, step)

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

        Returns:
            (query_buffer, key_buffer) where key_buffer[layer_idx] is a list
            of per-sample [context_len, num_kv_heads, head_size] tensors --
            together, the exact inputs scoring.compute_attention_score
            expects.
        """
        key_buffer = [
            retrieve_keys_per_sample(
                layer_name, block_size, num_kv_heads, head_size, per_sample_slot_mapping
            )
            for layer_name in self._layer_names
        ]
        return query_buffer, key_buffer

    def build_lookahead_metadata(
        self, prompt_len: int, look_ahead_cnt: int, head_dim: int
    ) -> LookaheadMetadata:
        """Build minimal, real per-step attention metadata plus a dummy KV
        cache for running `run_lookahead_steps`/`retrieve_qk` against a
        single request of length `prompt_len`.

        **Known limitation, read before using `look_ahead_cnt > 1`:** every
        step's metadata is currently built identically -- shaped for
        `prompt_len` query tokens, reusing the same `common_attn_metadata`
        (fresh per step via `builder.build(...)`, but always against a
        `BatchSpec(seq_lens=[prompt_len], query_lens=[prompt_len])`). This is
        correct for step 0 (the real prefill). For step 1+, `run_lookahead_steps`
        actually only feeds 1 new token (via its `next_input_fn`), so the
        metadata (still `prompt_len`-shaped) and the actual input (1 token)
        would mismatch -- this is exactly the "cross-step KV-cache-slot
        advancement" complexity this class's module docstring already flags
        as out of scope (what EAGLE's `eagle_step_update_slot_mapping_and_metadata`
        solves for the runner's own drafter). **Only `look_ahead_cnt == 1`
        (a single real prefill forward, no decode continuation) is verified
        consistent by this method as written.** Properly supporting
        `look_ahead_cnt > 1` needs per-step metadata that shrinks to 1 query
        token and grows `seq_lens`/slot mapping consistently against the
        *same* persistent dummy cache across steps -- not yet implemented;
        do not assume correctness for the multi-step case without fixing
        this first.

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
        cache sized for exactly this request; not reused across requests.
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

        kv_cache_spec = create_standard_kv_cache_spec(self.vllm_config)
        for self_attn in self._speculator_layers:
            dummy_cache = create_dummy_kv_cache(
                block_size,
                num_kv_heads,
                head_dim,
                self.vllm_config.model_config.dtype,
                self.device,
            )
            self_attn.attn.kv_cache = dummy_cache

        batch_spec = BatchSpec(seq_lens=[prompt_len], query_lens=[prompt_len])
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, block_size=block_size, device=self.device, arange_block_indices=True
        )

        backend_enum = self.vllm_config.attention_config.backend
        builder_cls, _ = try_get_attention_backend(backend_enum)
        builder = builder_cls(
            kv_cache_spec=kv_cache_spec,
            layer_names=self._layer_names,
            vllm_config=self.vllm_config,
            device=self.device,
        )
        per_step_attn_metadata = []
        per_step_slot_mapping = []
        for _ in range(look_ahead_cnt):
            attn_metadata = builder.build(
                common_prefix_len=0, common_attn_metadata=common_attn_metadata
            )
            per_step_attn_metadata.append({name: attn_metadata for name in self._layer_names})
            per_step_slot_mapping.append(
                {name: common_attn_metadata.slot_mapping for name in self._layer_names}
            )

        return LookaheadMetadata(
            per_step_attn_metadata=per_step_attn_metadata,
            per_step_slot_mapping=per_step_slot_mapping,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
