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

import types
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig, replace
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model

from .kv_cache_utils import retrieve_keys_per_sample


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
        `Gemma4Config.verify_and_update_config`)."""
        return replace(
            base_vllm_config,
            model_config=speculator_model_config,
            quant_config=None,
        )

    @staticmethod
    def _find_gemma4_attention_layers(model: nn.Module) -> List[nn.Module]:
        """Locate the speculator's Gemma4Attention layers directly via the
        known Gemma4ForCausalLM/Gemma4Model structure (model.model.layers[i]
        .self_attn), rather than a generic layer registry -- SpecPrefill's
        scope is Gemma-4-E2B-it specifically (see EXPERIMENT_PLAN.md)."""
        model_name = type(model).__name__
        if "Gemma4" not in model_name:
            raise NotImplementedError(
                f"SpecPrefillProposer currently only supports Gemma4 models "
                f"as the speculator (got {model_name}). The query-capture "
                f"hook (_query_capturing_gemma4_attention_forward) is a "
                f"faithful copy of Gemma4Attention.forward's body and is "
                f"not architecture-generic."
            )
        inner = model.model if hasattr(model, "model") else model
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
