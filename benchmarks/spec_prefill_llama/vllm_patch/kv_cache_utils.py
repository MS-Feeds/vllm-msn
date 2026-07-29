"""KV-cache read-back for Speculative Prefill — Algorithm 1, line 11 (K half).

Reads raw Key vectors for a set of token slots out of a model's own paged KV
cache, generically across attention backends (TritonAttention, FlashAttention,
...) and without assuming a uniform head_size across layers. Ported unchanged
from the sibling Gemma4 pipeline, where this per-layer design was load-bearing
because Gemma-4-E2B-it's sliding vs. full-attention layers use different head
dims (see gemma4.py's `head_dim`/`global_head_dim` split); Llama's head_dim is
uniform across layers (not independently confirmed on hardware here, same as
for the sibling Qwen3 pipeline), so this degenerates to every lookup
returning the same head_size, but needs no code change either way.

Verified against this fork's actual source (not assumed):
- `AttentionBackend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads,
  head_size)` and `.get_kv_cache_block_dim(...)` (vllm/v1/attention/
  backend.py:89-116) are backend classmethods reachable via
  `attn_layer.attn_backend`. `get_kv_cache_block_dim` finds the num_blocks
  dim with a sentinel-value trick; we reuse the same trick to find the K/V
  split dim (the dim of size 2) generically, rather than hardcoding e.g.
  TritonAttention's `(num_blocks, 2, block_size, num_kv_heads, head_size)`
  vs. FlashAttention's `(2, num_blocks, block_size, num_kv_heads, head_size)`.
- Q is *not* read from here — it comes from the query-capture forward hook
  in proposer.py, buffered during the lookahead loop rather than read back
  from cache.

**Takes the `Attention` layer module directly (e.g.
`speculator_layers[i].attn`), not a `layer_name` string + `get_attention_
context()`/`get_forward_context()` lookup** (an earlier version did the
latter -- confirmed on real hardware this was actually wrong: `kv_cache`
and `attn_backend` are both plain, persistent attributes on the `Attention`
module itself, not scoped to an active `set_forward_context(...)` block, but
`get_attention_context()` internally calls `get_forward_context()`, which
asserts a context is currently active -- `AssertionError: Forward context is
not set`, since `retrieve_qk`/`retrieve_keys_per_sample` are called *after*
`run_lookahead_steps` has already exited its `set_forward_context` blocks,
by design -- Q/K retrieval is a separate step from the forward passes
themselves, see proposer.py). `SpecPrefillProposer` already holds direct
references to each layer's `Attention` instance (`self._speculator_layers[i]
.attn`), so there's no need to route through the forward-context-scoped
lookup at all -- this module now has no vLLM dependency whatsoever, not even
a lazy one.
"""

from typing import List

import torch


def _find_kv_split_dim(
    attn_backend,
    kv_cache: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> int:
    """Find which dim of `kv_cache` is the K/V-split dim (size 2), using the
    backend's own declared shape rather than hardcoding a layout per backend.
    """
    block_dim = attn_backend.get_kv_cache_block_dim(block_size, num_kv_heads, head_size)
    num_blocks = kv_cache.shape[block_dim]
    expected_shape = attn_backend.get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size
    )
    if tuple(kv_cache.shape) != tuple(expected_shape):
        raise ValueError(
            f"KV cache shape {tuple(kv_cache.shape)} does not match "
            f"{attn_backend.get_name()}'s declared shape {tuple(expected_shape)} "
            f"for block_size={block_size}, num_kv_heads={num_kv_heads}, "
            f"head_size={head_size} -- backend may use a non-generic layout "
            f"not handled by this reader."
        )
    split_dims = [i for i, s in enumerate(expected_shape) if s == 2 and i != block_dim]
    if len(split_dims) != 1:
        raise ValueError(
            f"Could not uniquely locate the K/V split dim in shape "
            f"{expected_shape} (block_dim={block_dim}, candidates={split_dims})."
        )
    return split_dims[0]


def read_layer_keys(
    attn_layer,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> torch.Tensor:
    """Read the full physical K cache for one layer as a flat, slot-indexable
    tensor: [num_blocks * block_size, num_kv_heads, head_size].

    Args:
        attn_layer: the `vllm.model_executor.layers.attention.Attention`
            module instance for this layer (e.g. a speculator layer's
            `self_attn.attn`) -- read directly, not looked up via
            `get_attention_context()`, see module docstring.

    Caller is expected to index the result with a slot_mapping (physical slot
    = block_id * block_size + offset_within_block, vLLM's standard
    convention) to get per-token keys — see `gather_keys_for_slots` below.
    """
    kv_cache = attn_layer.kv_cache
    attn_backend = attn_layer.attn_backend

    split_dim = _find_kv_split_dim(attn_backend, kv_cache, block_size, num_kv_heads, head_size)
    k_cache, _v_cache = kv_cache.unbind(split_dim)

    # Both TritonAttention ((num_blocks, 2, block_size, ...) -> split_dim=1)
    # and FlashAttention ((2, num_blocks, block_size, ...) -> split_dim=0)
    # leave num_blocks and block_size as adjacent leading dims after
    # unbind, so a flatten of the first two dims gives a flat slot axis
    # matching vLLM's `block_id * block_size + offset` slot convention.
    # Assert this rather than silently assuming it for other backends.
    if k_cache.shape[0] * k_cache.shape[1] != k_cache.numel() // (num_kv_heads * head_size):
        raise ValueError(
            f"Unexpected K cache shape {tuple(k_cache.shape)} after unbind -- "
            f"num_blocks/block_size are not the leading two dims, so flat "
            f"slot-index flattening is not valid for this backend layout."
        )
    return k_cache.reshape(-1, num_kv_heads, head_size)


def gather_keys_for_slots(
    flat_keys: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Index a flat [num_slots, num_kv_heads, head_size] cache tensor by a
    per-token slot_mapping, returning [num_tokens, num_kv_heads, head_size].
    """
    return flat_keys[slot_mapping.to(flat_keys.device)]


def retrieve_keys_per_sample(
    attn_layer,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    per_sample_slot_mapping: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Algorithm line 11 (K half) for one layer: read that layer's cache once
    and split it into per-sample key tensors via each sample's own
    slot_mapping (physical slots for that sample's prompt tokens).

    Args:
        attn_layer: see `read_layer_keys`'s docstring.

    Returns a list (one per prefill sample) of [context_len, num_kv_heads,
    head_size] key tensors -- the `key_buffer[layer_idx]` shape expected by
    scoring.compute_attention_score.
    """
    flat_keys = read_layer_keys(attn_layer, block_size, num_kv_heads, head_size)
    return [
        gather_keys_for_slots(flat_keys, slot_mapping)
        for slot_mapping in per_sample_slot_mapping
    ]
