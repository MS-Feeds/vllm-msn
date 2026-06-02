# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80/A100 reference fallbacks for DeepSeek-V4 sparse attention."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import LOG2E, tl, triton


def _dsv4_sm80_sparse_attn_split_kernel(
    q_ptr,  # (B*S, H, D) bf16
    kv_ptr,  # (B*S, T, D) bf16 — pre-gathered (zero rows for invalid)
    invalid_mask_ptr,  # (B*S, T) uint8 (1 = invalid)
    acc_split_ptr,  # (B*S, SPLIT_T, H, D_V) fp32
    max_split_ptr,  # (B*S, SPLIT_T, H) fp32
    sum_split_ptr,  # (B*S, SPLIT_T, H) fp32
    n_tokens,
    total_topk,
    sm_scale_log2,  # scale * LOG2E
    H: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_T: tl.constexpr,
):
    """Split-K sparse-attention decode pass 1 for V4 on SM80.

    Each program processes one chunk of `total_topk` (sized `chunk =
    ceil(total_topk / SPLIT_T)`) and emits unnormalised partial outputs:
    `acc = sum_n exp2(qk_n - max_s) * v_n`, plus the per-split max and
    sum. The combine kernel performs the cross-split LSE merge and sink
    correction.

    Splitting the topk axis lifts grid parallelism from
    `(n_tokens, ceil(H/BLOCK_H))` to `(n_tokens, SPLIT_T, ceil(H/BLOCK_H))`,
    which matters at batch=1 single-decode where only 1 SM was active.
    """
    pid_t = tl.program_id(0)
    pid_split = tl.program_id(1)
    pid_h = tl.program_id(2)

    if pid_t >= n_tokens:
        return

    chunk_size = (total_topk + SPLIT_T - 1) // SPLIT_T
    n_start_chunk = pid_split * chunk_size
    n_end_chunk = tl.minimum(n_start_chunk + chunk_size, total_topk)

    head_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_off < H

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    # Hold q in bf16 — only used as bf16 in the inner-loop dot.
    q = tl.load(
        q_ptr + pid_t * H * D + head_off[:, None] * D + d_off[None, :],
        mask=head_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    e_max = tl.zeros((BLOCK_H,), dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)

    n_iter = (chunk_size + BLOCK_N - 1) // BLOCK_N
    for n_block in range(n_iter):
        n_start = n_start_chunk + n_block * BLOCK_N
        n_off = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_off < n_end_chunk

        invalid_u8 = tl.load(
            invalid_mask_ptr + pid_t * total_topk + n_off,
            mask=n_mask,
            other=1,
        )
        valid = (invalid_u8 == 0) & n_mask

        # Load kv directly as bf16; tl.dot accumulates to fp32.
        kv = tl.load(
            kv_ptr + pid_t * total_topk * D + n_off[:, None] * D + d_off[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(kv))
        qk *= sm_scale_log2
        qk = tl.where(head_mask[:, None] & valid[None, :], qk, -1.0e30)

        n_e_max = tl.maximum(tl.max(qk, axis=1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        # V == K for V4 — reuse the loaded kv tile for the pv dot.
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(tl.bfloat16), kv)
        e_sum = e_sum * re_scale + tl.sum(p, axis=1)
        e_max = n_e_max

    # Store partials. Layout: (B*S, SPLIT_T, H, D_V) for acc,
    # (B*S, SPLIT_T, H) for max/sum — keeps the per-split stride contiguous
    # so the combine kernel can issue coalesced loads.
    dv_off = tl.arange(0, BLOCK_DV)
    dv_mask = dv_off < D_V
    base_acc = (
        pid_t * SPLIT_T * H * D_V
        + pid_split * H * D_V
        + head_off[:, None] * D_V
        + dv_off[None, :]
    )
    tl.store(
        acc_split_ptr + base_acc,
        acc,
        mask=head_mask[:, None] & dv_mask[None, :],
    )

    base_ms = pid_t * SPLIT_T * H + pid_split * H + head_off
    tl.store(max_split_ptr + base_ms, e_max, mask=head_mask)
    tl.store(sum_split_ptr + base_ms, e_sum, mask=head_mask)


@triton.jit
def _dsv4_sm80_sparse_attn_combine_kernel(
    acc_split_ptr,  # (B*S, SPLIT_T, H, D_V) fp32
    max_split_ptr,  # (B*S, SPLIT_T, H) fp32
    sum_split_ptr,  # (B*S, SPLIT_T, H) fp32
    attn_sink_ptr,  # (H,) fp32
    out_ptr,  # (B*S, H, D_V) bf16
    n_tokens,
    has_sink: tl.constexpr,
    H: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_T: tl.constexpr,
):
    """Cross-split LSE merge with sink correction.

    Reads `SPLIT_T` partial (acc, max, sum) tuples and emits the final
    softmax-normalised output. Sink contributes `exp2(sink_log2 - max)`
    to the global denominator only — it has no v term.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= n_tokens:
        return

    head_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_off < H

    # Find global max over splits (and sink, if any).
    e_max_global = tl.zeros((BLOCK_H,), dtype=tl.float32) - 1.0e30
    for s in range(SPLIT_T):
        m = tl.load(
            max_split_ptr + pid_t * SPLIT_T * H + s * H + head_off,
            mask=head_mask,
            other=-1.0e30,
        )
        e_max_global = tl.maximum(e_max_global, m)

    sink_log2 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    if has_sink:
        sink = tl.load(attn_sink_ptr + head_off, mask=head_mask, other=0.0)
        # log2(e) inlined: Triton @jit can't read non-constexpr globals.
        sink_log2 = sink * 1.4426950408889634
        e_max_global = tl.maximum(e_max_global, sink_log2)

    # Renormalise and reduce.
    dv_off = tl.arange(0, BLOCK_DV)
    dv_mask = dv_off < D_V
    acc_global = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)
    sum_global = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for s in range(SPLIT_T):
        m_s = tl.load(
            max_split_ptr + pid_t * SPLIT_T * H + s * H + head_off,
            mask=head_mask,
            other=-1.0e30,
        )
        sum_s = tl.load(
            sum_split_ptr + pid_t * SPLIT_T * H + s * H + head_off,
            mask=head_mask,
            other=0.0,
        )
        scale = tl.exp2(m_s - e_max_global)
        sum_global += scale * sum_s

        base_acc = (
            pid_t * SPLIT_T * H * D_V
            + s * H * D_V
            + head_off[:, None] * D_V
            + dv_off[None, :]
        )
        acc_s = tl.load(
            acc_split_ptr + base_acc,
            mask=head_mask[:, None] & dv_mask[None, :],
            other=0.0,
        )
        acc_global += scale[:, None] * acc_s

    if has_sink:
        sum_global += tl.exp2(sink_log2 - e_max_global)

    sum_safe = tl.where(sum_global > 0, sum_global, 1.0)
    out = (acc_global / sum_safe[:, None]).to(tl.bfloat16)
    tl.store(
        out_ptr + pid_t * H * D_V + head_off[:, None] * D_V + dv_off[None, :],
        out,
        mask=head_mask[:, None] & dv_mask[None, :],
    )


def _dsv4_sm80_sparse_attn_decode_triton(
    q: torch.Tensor,  # (B*S, H, D) bf16
    gathered_kv: torch.Tensor,  # (B*S, T, D) bf16
    invalid_mask: torch.Tensor,  # (B*S, T) bool
    attn_sink: torch.Tensor | None,  # (H,) fp32
    sm_scale: float,
    head_dim_v: int,
) -> torch.Tensor:
    """Split-K sparse-attention decode for V4 on SM80.

    Two-kernel pipeline: a split-K pass over the topk dimension followed
    by an LSE-merge combine. SPLIT_T is bounded by the BLOCK_N-tile count
    so each split has real work to do."""
    n_tokens, h, d = q.shape
    _, t, d_kv = gathered_kv.shape
    assert d_kv == d
    assert invalid_mask.shape == (n_tokens, t)

    block_d = triton.next_power_of_2(d)
    block_dv = triton.next_power_of_2(head_dim_v)
    # BLOCK_H capped at 16 (tl.dot's M-min) so the fp32 `acc` tile
    # (BLOCK_H × BLOCK_DV) stays under A100's 100 KB SMEM limit and so we
    # get more per-token head blocks for SM utilisation when h > 16.
    block_h = 16
    block_n = 32  # keeps q/kv/acc tiles within A100's 164KB SM

    # SPLIT_T heuristic: cap at 16 (combine overhead dominates beyond
    # that) but otherwise split as much as we have BLOCK_N tiles. Lifts
    # grid parallelism from (n_tokens, ceil(h/BLOCK_H)) to
    # (n_tokens, SPLIT_T, ceil(h/BLOCK_H)) — at batch=1 single-decode the
    # original kernel used 1 SM out of 108.
    n_tiles = (t + block_n - 1) // block_n
    split_t = max(1, min(16, n_tiles))

    out = torch.empty((n_tokens, h, head_dim_v), dtype=torch.bfloat16, device=q.device)
    invalid_u8 = invalid_mask.to(torch.uint8)

    acc_split = torch.empty(
        (n_tokens, split_t, h, head_dim_v),
        dtype=torch.float32,
        device=q.device,
    )
    max_split = torch.empty(
        (n_tokens, split_t, h), dtype=torch.float32, device=q.device
    )
    sum_split = torch.empty_like(max_split)

    grid_split = (n_tokens, split_t, triton.cdiv(h, block_h))
    _dsv4_sm80_sparse_attn_split_kernel[grid_split](
        q,
        gathered_kv,
        invalid_u8,
        acc_split,
        max_split,
        sum_split,
        n_tokens,
        t,
        sm_scale * LOG2E,
        H=h,
        D=d,
        D_V=head_dim_v,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        SPLIT_T=split_t,
        num_warps=4,
    )

    grid_combine = (n_tokens, triton.cdiv(h, block_h))
    _dsv4_sm80_sparse_attn_combine_kernel[grid_combine](
        acc_split,
        max_split,
        sum_split,
        attn_sink if attn_sink is not None else q.new_zeros(h),
        out,
        n_tokens,
        has_sink=(attn_sink is not None),
        H=h,
        D_V=head_dim_v,
        BLOCK_H=block_h,
        BLOCK_DV=block_dv,
        SPLIT_T=split_t,
        num_warps=4,
    )
    return out
