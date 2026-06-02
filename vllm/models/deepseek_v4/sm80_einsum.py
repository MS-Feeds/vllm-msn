# SPDX-License-Identifier: Apache-2.0
import math
import torch
from vllm.triton_utils import tl, triton

def _inv_rope_bf16_kernel(
    o_ptr,  # (T, H, D) bf16, modified in place
    positions_ptr,  # (T,) int64
    cos_sin_cache_ptr,  # (max_pos, rope_dim) bf16
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope: tl.constexpr,
    nope_dim: tl.constexpr,
):
    """Single-launch inv-RoPE on bf16 for the SM80 reference path.

    One program per (token, head). Replaces the ~10-op PyTorch chain in
    `_apply_inv_rope_to_o` (index_select, clone, slice/stride pairs, mul,
    add, sub, copy_back) with one kernel.

    GPT-J interleaved: even/odd pairs at positions (2r, 2r+1) within the
    rope segment are rotated using the (cos, sin) at index r.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_t >= T:
        return

    pos = tl.load(positions_ptr + pid_t)
    base_cs = cos_sin_cache_ptr + pos * rope_dim
    r = tl.arange(0, half_rope)
    cos_v = tl.load(base_cs + r).to(tl.float32)
    sin_v = tl.load(base_cs + half_rope + r).to(tl.float32)

    base_row = o_ptr + (pid_t * H + pid_h) * D + nope_dim
    even = tl.load(base_row + 2 * r).to(tl.float32)
    odd = tl.load(base_row + 2 * r + 1).to(tl.float32)
    new_even = even * cos_v + odd * sin_v
    new_odd = odd * cos_v - even * sin_v
    tl.store(base_row + 2 * r, new_even.to(tl.bfloat16))
    tl.store(base_row + 2 * r + 1, new_odd.to(tl.bfloat16))


def _apply_inv_rope_to_o(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Apply inverse GPT-J RoPE on the last `rope_dim` dims of each head.
    Used by the SM80/ROCm reference path that skips FP8 quantization.
    Matches the rotation in `_fused_inv_rope_fp8_quant_per_head` numerically."""
    if not o.is_contiguous():
        o = o.contiguous()
    out = o.clone()
    T, H, D = out.shape
    nope_dim = D - rope_dim
    half_rope = rope_dim // 2
    positions_i64 = positions.to(torch.int64).contiguous()
    cs = cos_sin_cache
    # cos_sin_cache is (max_pos, rope_dim) bf16 with the GPT-J layout
    # [cos | sin] along the last dim. We index by position and split inline.
    grid = (T, H)
    _inv_rope_bf16_kernel[grid](
        out,
        positions_i64,
        cs,
        T,
        H=H,
        D=D,
        rope_dim=rope_dim,
        half_rope=half_rope,
        nope_dim=nope_dim,
    )
    return out


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _expand_last_dim_scales(scale: torch.Tensor, last_dim: int) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    block = math.ceil(last_dim / scale.shape[-1])
    return torch.repeat_interleave(scale, block, dim=-1)[..., :last_dim]


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


def _deepseek_v4_fp8_einsum_fallback(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    """SM80/ROCm dequantize-and-einsum fallback for `bhr,hdr->bhd`.

    On SM80 with Marlin-packed FP8 weights the wrapper bypasses this and
    routes through `wo_a` as a regular Linear; this remains for ROCm and
    any non-Marlin SM80 layout."""
    if equation != "bhr,hdr->bhd":
        raise RuntimeError(f"Unsupported fallback equation: {equation}")

    num_groups = a.shape[1]
    hidden_dim = a.shape[2]
    output_dim = b.shape[0] // num_groups

    if b.shape[0] % num_groups != 0:
        raise RuntimeError(
            f"Cannot reshape weight of shape {tuple(b.shape)} into "
            f"({num_groups}, {output_dim}, {hidden_dim})."
        )

    a_deq = (a.to(torch.float32) * _expand_last_dim_scales(a_scale, hidden_dim)).to(
        torch.bfloat16
    )

    b_deq = b.view(num_groups, output_dim, hidden_dim).to(torch.float32)
    b_scale_deq = _expand_2d_block_scales(
        b_scale.view(num_groups, -1, b_scale.shape[-1]),
        output_dim,
        hidden_dim,
    )
    b_deq = (b_deq * b_scale_deq).to(torch.bfloat16)

    out.copy_(torch.einsum(equation, a_deq, b_deq).to(out.dtype))
