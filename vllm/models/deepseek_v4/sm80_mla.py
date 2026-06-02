# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 reference sparse MLA decode/prefill paths for DeepSeek-V4."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.models.deepseek_v4.sm80_sparse_kernels import (
    _dsv4_sm80_sparse_attn_decode_triton,
)
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    gather_dequant_two_scopes_with_mask,
)

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.attention import DeepseekV4MLAAttention


def ref_sparse_attn_decode_gather(
    attn: "DeepseekV4MLAAttention",
    q: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    swa_block_size: int,
    swa_indices: torch.Tensor,
    swa_topk_length: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    extra_kv_cache: torch.Tensor | None,
    extra_block_size: int,
    extra_indices: torch.Tensor | None,
    extra_topk_length: torch.Tensor | None,
) -> torch.Tensor:
    b, s_q, h_q, d_qk = q.shape
    d_v = attn.head_dim
    bs = b * s_q

    swa_indices_2d = swa_indices.reshape(bs, -1)
    extra_indices_2d = extra_indices.reshape(bs, -1) if extra_indices is not None else None

    gathered_kv_flat, invalid_flat = gather_dequant_two_scopes_with_mask(
        swa_kv_cache=swa_kv_cache,
        swa_block_size=swa_block_size,
        swa_indices=swa_indices_2d,
        swa_topk_length=swa_topk_length,
        extra_kv_cache=extra_kv_cache,
        extra_block_size=extra_block_size,
        extra_indices=extra_indices_2d,
        extra_topk_length=extra_topk_length,
        nope_dim=attn.nope_head_dim,
        rope_dim=attn.rope_head_dim,
        head_dim=d_qk,
    )

    q_flat = q.view(bs, h_q, d_qk).to(torch.bfloat16).contiguous()
    out_flat = _dsv4_sm80_sparse_attn_decode_triton(
        q_flat,
        gathered_kv_flat,
        invalid_flat,
        attn_sink,
        attn.scale,
        d_v,
    )
    return out_flat.view(b, h_q, d_v)


def ref_sparse_attn_prefill(
    attn: "DeepseekV4MLAAttention",
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
) -> torch.Tensor:
    indices = indices.clone().squeeze(1)
    s_q, h_q, d_qk = q.shape
    topk = indices.shape[-1]
    s_kv = kv.shape[0]
    if topk_length is not None:
        mask = torch.arange(topk, device=indices.device).unsqueeze(
            0
        ) >= topk_length.unsqueeze(1)
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv)
    indices[invalid_mask] = 0

    qf = q.float()
    gathered_kv = kv.index_select(0, indices.flatten()).reshape(s_q, topk, d_qk).float()
    scores = qf @ gathered_kv.transpose(1, 2)
    scores *= attn.scale
    scores[invalid_mask.unsqueeze(1).expand_as(scores)] = float("-inf")

    orig_lse = torch.logsumexp(scores, dim=-1)
    lse_for_o = orig_lse
    if attn.attn_sink is not None:
        lse_for_o = torch.logsumexp(
            torch.stack(
                [
                    orig_lse,
                    attn.attn_sink[:h_q].view(1, h_q).expand_as(orig_lse),
                ],
                dim=0,
            ),
            dim=0,
        )
    lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")
    probs = torch.exp(scores - lse_for_o.unsqueeze(-1))
    out = probs @ gathered_kv[..., : attn.head_dim]
    lonely_q_mask = orig_lse == float("-inf")
    out[lonely_q_mask.unsqueeze(-1).expand_as(out)] = 0.0
    return out.to(torch.bfloat16)
