"""Relay attention operations for vLLM 0.9.1.

This module provides relay fusion operations for combining system and user attention outputs.
Ported from vLLM 0.2.6 implementation.
"""

import triton
import triton.language as tl

import torch
from torch import Tensor


def relay_fusion(out_sys: Tensor,
                 lse_sys: Tensor,
                 out_usr: Tensor,
                 lse_usr: Tensor,
                 backend='native',
                 trans_lse_sys: bool = False,
                 trans_lse_usr: bool = False) -> Tensor:
    """Fusion operation for relay attention.

    Args:
        out_sys, out_usr: shape = [num_tokens, num_heads, head_size]
        lse_sys, lse_usr: shape = [num_tokens, num_heads], or [num_heads, num_tokens] if trans_lse_x=True
        backend: 'native' or 'triton'
        trans_lse_sys, trans_lse_usr: bool flag to specify if lse1 and lse2 should be transposed
    Returns:
        shape = [num_tokens, num_heads, head_size]
    """
    assert backend in {'native', 'triton'}
    assert out_sys.size() == out_usr.size()
    assert out_sys.ndim == 3
    
    if backend == 'native':
        if trans_lse_sys:
            lse_sys = lse_sys.transpose(0, 1).contiguous()
        if trans_lse_usr:
            lse_usr = lse_usr.transpose(0, 1).contiguous()
        assert lse_sys.size() == out_sys.shape[:2]
        assert lse_usr.size() == out_usr.shape[:2]
        lse_sys = lse_sys.unsqueeze(-1)  # (num_tokens, num_heads, 1)
        lse_usr = lse_usr.unsqueeze(-1)  # (num_tokens, num_heads, 1)
        alpha_sys = 1. / (1. + (lse_usr - lse_sys).exp())  # (num_tokens, num_heads, 1)
        
        # Use fp32 to reduce accumulation error
        out = alpha_sys * out_sys.to(torch.float32) + \
            (1. - alpha_sys) * out_usr.to(torch.float32)  # (num_tokens, nhead, hdim)
        out = out.to(out_sys.dtype)
    else:
        out = _relay_fuse_triton(out_sys, lse_sys, out_usr, lse_usr,
                                trans_lse_sys, trans_lse_usr)
    return out


def _relay_fuse_triton(out_sys: Tensor, lse_sys: Tensor, out_usr: Tensor, lse_usr: Tensor,
                       trans_lse_sys: bool, trans_lse_usr: bool):
    """Triton kernel for relay fusion."""
    # Ensure final dimension is contiguous
    assert out_sys.stride(-1) == 1
    assert out_usr.stride(-1) == 1
    out = torch.empty_like(out_sys)
    
    num_tokens, num_heads, head_size = out_sys.size()
    
    if trans_lse_sys:  # (num_heads, num_tokens)
        lse_sys_stride_h, lse_sys_stride_t = lse_sys.stride()
    else:
        lse_sys_stride_t, lse_sys_stride_h = lse_sys.stride()

    if trans_lse_usr:  # (num_heads, num_tokens)
        lse_usr_stride_h, lse_usr_stride_t = lse_usr.stride()
    else:
        lse_usr_stride_t, lse_usr_stride_h = lse_usr.stride()

    BLOCK_SIZE = triton.next_power_of_2(head_size)
    num_warps = 4
    
    _relay_fuse_kernel[(num_tokens, num_heads)](
        out_fused_ptr=out,
        out_sys_ptr=out_sys,
        lse_sys_ptr=lse_sys,
        out_usr_ptr=out_usr,
        lse_usr_ptr=lse_usr,
        head_size=head_size,
        lse_sys_stride_t=lse_sys_stride_t,
        lse_sys_stride_h=lse_sys_stride_h,
        lse_usr_stride_t=lse_usr_stride_t,
        lse_usr_stride_h=lse_usr_stride_h,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


@triton.jit
def _relay_fuse_kernel(
        out_fused_ptr,  # final output
        out_sys_ptr, lse_sys_ptr,
        out_usr_ptr, lse_usr_ptr,
        head_size,
        lse_sys_stride_t, lse_sys_stride_h,
        lse_usr_stride_t, lse_usr_stride_h,
        BLOCK_SIZE: tl.constexpr):
    """Triton kernel for relay fusion computation."""
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Load LSE values
    lse_sys = tl.load(lse_sys_ptr +
                      token_id * lse_sys_stride_t +
                      head_id * lse_sys_stride_h).to(tl.float32)
    lse_usr = tl.load(lse_usr_ptr +
                      token_id * lse_usr_stride_t +
                      head_id * lse_usr_stride_h).to(tl.float32)
    
    # Calculate fusion weights
    rescale_sys = 1. / (1 + tl.exp(lse_usr - lse_sys))
    rescale_usr = 1. - rescale_sys
    
    # Load attention outputs
    all_head_id = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    head_offs = tl.arange(0, BLOCK_SIZE)
    io_mask = head_offs < head_size
    
    out_sys = tl.load(out_sys_ptr + all_head_id * head_size + head_offs,
                      mask=io_mask, other=0.)
    out_usr = tl.load(out_usr_ptr + all_head_id * head_size + head_offs,
                      mask=io_mask, other=0.)
    
    # Fuse the outputs
    out_fused = rescale_sys * out_sys + rescale_usr * out_usr
    
    # Store the result
    tl.store(out_fused_ptr + all_head_id * head_size + head_offs,
             out_fused, mask=io_mask)


class RelayAttentionBackend:
    """Backend for relay attention operations."""
    
    def __init__(self, use_triton: bool = True):
        self.use_triton = use_triton
    
    def fuse_attention_outputs(self, 
                              out_sys: Tensor,
                              lse_sys: Tensor,
                              out_usr: Tensor,
                              lse_usr: Tensor,
                              trans_lse_sys: bool = False,
                              trans_lse_usr: bool = False) -> Tensor:
        """Fuse system and user attention outputs."""
        backend = 'triton' if self.use_triton else 'native'
        return relay_fusion(out_sys, lse_sys, out_usr, lse_usr,
                           backend=backend,
                           trans_lse_sys=trans_lse_sys,
                           trans_lse_usr=trans_lse_usr) 