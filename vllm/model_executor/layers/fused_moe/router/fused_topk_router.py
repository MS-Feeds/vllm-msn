# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
import os

import torch
from vllm.triton_utils import tl, triton

import vllm._custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


@triton.jit
def _gemma4_top2_softmax_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    token_expert_indices_ptr,
    num_experts,
    renormalize: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_experts
    logits = tl.load(
        logits_ptr + token * num_experts + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    # Keep both winners in one reduction. This avoids a second argmax pass and
    # avoids the full softmax denominator when selected weights are renormalized.
    best_value = tl.full((), -float("inf"), tl.float32)
    second_value = tl.full((), -float("inf"), tl.float32)
    best_id = tl.full((), 0, tl.int32)
    second_id = tl.full((), 0, tl.int32)
    for offset in tl.static_range(0, BLOCK_SIZE):
        value = tl.load(
            logits_ptr + token * num_experts + offset,
            mask=offset < num_experts,
            other=-float("inf"),
        ).to(tl.float32)
        is_new_best = value > best_value
        is_new_second = value > second_value
        old_best_value = best_value
        old_best_id = best_id
        second_value = tl.where(is_new_best, old_best_value, tl.where(is_new_second, value, second_value))
        second_id = tl.where(is_new_best, old_best_id, tl.where(is_new_second, offset, second_id))
        best_value = tl.where(is_new_best, value, best_value)
        best_id = tl.where(is_new_best, offset, best_id)

    if renormalize:
        selected_max = tl.maximum(best_value, second_value)
        best_exp = tl.exp(best_value - selected_max)
        second_exp = tl.exp(second_value - selected_max)
        selected_sum = best_exp + second_exp
        first_out = best_exp / selected_sum
        second_out = second_exp / selected_sum
    else:
        full_max = tl.max(logits, axis=0)
        full_sum = tl.sum(tl.where(mask, tl.exp(logits - full_max), 0.0), axis=0)
        first_out = tl.exp(best_value - full_max) / full_sum
        second_out = tl.exp(second_value - full_max) / full_sum

    weight_base = token * 2
    tl.store(weights_ptr + weight_base, first_out)
    tl.store(weights_ptr + weight_base + 1, second_out)
    tl.store(ids_ptr + weight_base, best_id)
    tl.store(ids_ptr + weight_base + 1, second_id)
    tl.store(token_expert_indices_ptr + weight_base, token)
    tl.store(token_expert_indices_ptr + weight_base + 1, token)


def _gemma4_top2_softmax(
    gating_output: torch.Tensor,
    renormalize: bool,
    indices_type: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = gating_output.shape
    block_size = 1 << (num_experts - 1).bit_length()
    weights = torch.empty((num_tokens, 2), dtype=torch.float32, device=gating_output.device)
    ids = torch.empty(
        (num_tokens, 2),
        dtype=torch.int32 if indices_type is None else indices_type,
        device=gating_output.device,
    )
    token_expert_indices = torch.empty((num_tokens, 2), dtype=torch.int32, device=gating_output.device)
    _gemma4_top2_softmax_kernel[(num_tokens,)](
        gating_output,
        weights,
        ids,
        token_expert_indices,
        num_experts,
        renormalize=renormalize,
        BLOCK_SIZE=block_size,
    )
    return weights, ids, token_expert_indices


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def dispatch_topk_sigmoid_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_sigmoid
    return vllm_topk_sigmoid


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    # Prototype path for Gemma4's fixed top-2 router. Keep it opt-in until
    # numerical parity and GPU performance are measured against ops.topk_softmax.
    if (
        topk == 2
        and scoring_func == "softmax"
        and os.environ.get("VLLM_GEMMA4_TOP2_ROUTER") == "1"
    ):
        return _gemma4_top2_softmax(gating_output, renormalize, indices_type)

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    elif scoring_func == "sigmoid":
        topk_func = dispatch_topk_sigmoid_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


class FusedTopKRouter(BaseRouter):
    """Default router using standard fused top-k routing."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        eplb_state: EplbLayerState | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using standard fused top-k."""
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=self.scoring_func,
        )

        return topk_weights, topk_ids
