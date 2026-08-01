# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.gemma4 import (
    gemma4_fused_routing_kernel_triton,
    gemma4_fused_routing_topk_softmax,
    gemma4_routing_function_torch,
)


def _gemma4_routing_reference_naive(
    gating_output: torch.Tensor,
    topk: int,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
    router_probabilities = torch.nn.functional.softmax(gating_output, dim=-1)
    indicator = torch.nn.functional.one_hot(
        topk_ids, num_classes=gating_output.size(-1)
    ).sum(dim=-2)
    gate_weights = indicator * router_probabilities
    renorm_factor = torch.sum(gate_weights, dim=-1, keepdim=True)
    renorm_factor = torch.where(renorm_factor > 0.0, renorm_factor, 1.0)
    dispatch_weights = gate_weights / renorm_factor
    topk_weights = dispatch_weights.gather(1, topk_ids)
    topk_weights = topk_weights * per_expert_scale[topk_ids].to(topk_weights.dtype)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def sort_by_id(w, ids):
    order = ids.argsort(dim=-1)
    return w.gather(1, order), ids.gather(1, order)


# Gemma4 MoE Model has context length of 250K
# the minus 1 is to ensure that edge cases are tested
@pytest.mark.parametrize("num_tokens", [1, 2, 2048, 250000])
@pytest.mark.parametrize("num_experts", [128])  # gemma4 moe experts
@pytest.mark.parametrize("topk", [8])  # gemma4 topk
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_gemma4_routing_kernel_triton(
    num_tokens: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
):
    torch.manual_seed(0)

    gating = torch.randn(num_tokens, num_experts, dtype=dtype, device="cuda")
    scales = torch.rand(num_experts, dtype=torch.float32, device="cuda")

    ref_w, ref_ids = gemma4_routing_function_torch(gating, topk, scales)
    tri_w, tri_ids = gemma4_fused_routing_kernel_triton(gating, topk, scales)

    # Sort by expert id — to remove tie-breaking differences
    ref_ws, ref_is = sort_by_id(ref_w, ref_ids)
    tri_ws, tri_is = sort_by_id(tri_w, tri_ids)

    ids_match = (ref_is == tri_is).all().item()
    weights_match = torch.allclose(ref_ws, tri_ws, atol=1e-2, rtol=1e-2)
    all_match = ids_match and weights_match
    max_err = (ref_ws - tri_ws).abs().max().item()
    print(
        f"T={num_tokens:5d} E={num_experts:4d} K={topk} "
        f"{str(dtype).split('.')[-1]:7s} ids={ids_match} max_Δweight={max_err:.2e}"
    )
    if not all_match:
        bad = (ref_is != tri_is).any(dim=-1).nonzero(as_tuple=True)[0]
        if len(bad):
            r = bad[0].item()
            print(
                f"  first bad row {r}: ref_ids={ref_ids[r].tolist()} "
                f"tri_ids={tri_ids[r].tolist()}"
            )
        assert all_match


@pytest.mark.parametrize("num_tokens", [3, 17])
@pytest.mark.parametrize("num_experts", [64, 128])
@pytest.mark.parametrize("topk", [4, 8])
def test_gemma4_routing_torch_matches_naive_reference(
    num_tokens: int,
    num_experts: int,
    topk: int,
):
    torch.manual_seed(1234)
    gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    scales = torch.rand(num_experts, dtype=torch.float32)

    ref_w, ref_ids = _gemma4_routing_reference_naive(gating, topk, scales)
    opt_w, opt_ids = gemma4_routing_function_torch(gating, topk, scales)

    ref_ws, ref_is = sort_by_id(ref_w, ref_ids)
    opt_ws, opt_is = sort_by_id(opt_w, opt_ids)

    assert torch.equal(ref_is, opt_is)
    assert torch.allclose(ref_ws, opt_ws, atol=1e-6, rtol=1e-6)


def test_gemma4_fused_routing_topk_softmax_shape_guard():
    gating = torch.randn(5, 16, dtype=torch.float32)
    scales = torch.rand(16, dtype=torch.float32)
    out_weights = torch.empty(5, 3, dtype=torch.float32)
    out_ids = torch.empty(5, 3, dtype=torch.int32)
    token_expert_indices = torch.empty(5, 3, dtype=torch.int32)

    with pytest.raises(AssertionError):
        gemma4_fused_routing_topk_softmax(
            gating,
            topk=4,
            per_expert_scale=scales,
            out_weights=out_weights,
            out_ids=out_ids,
            token_expert_indices=token_expert_indices,
        )


def test_gemma4_fused_routing_topk_softmax_matches_reference(monkeypatch):
    def fake_topk_softmax(
        out_weights: torch.Tensor,
        out_ids: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool,
    ):
        del token_expert_indices
        topk_values, topk_indices = torch.topk(gating_output, k=out_ids.shape[1], dim=-1)
        out_ids.copy_(topk_indices.to(torch.int32))
        if renormalize:
            out_weights.copy_(torch.softmax(topk_values.to(torch.float32), dim=-1))
        else:
            out_weights.copy_(topk_values.to(torch.float32))
        return out_weights, out_ids

    import vllm.model_executor.models.gemma4 as gemma4_mod

    monkeypatch.setattr(
        gemma4_mod,
        "dispatch_topk_softmax_func",
        lambda: fake_topk_softmax,
    )

    torch.manual_seed(7)
    gating = torch.randn(9, 32, dtype=torch.float32)
    scales = torch.rand(32, dtype=torch.float32)
    out_weights = torch.empty(9, 4, dtype=torch.float32)
    out_ids = torch.empty(9, 4, dtype=torch.int32)
    token_expert_indices = torch.empty(9, 4, dtype=torch.int32)

    w, ids = gemma4_fused_routing_topk_softmax(
        gating,
        topk=4,
        per_expert_scale=scales,
        out_weights=out_weights,
        out_ids=out_ids,
        token_expert_indices=token_expert_indices,
    )
    ref_w, ref_ids = _gemma4_routing_reference_naive(gating, 4, scales)
    rw, ri = sort_by_id(ref_w, ref_ids)
    ww, wi = sort_by_id(w, ids)

    assert torch.equal(ri, wi)
    assert torch.allclose(rw, ww, atol=1e-6, rtol=1e-6)
