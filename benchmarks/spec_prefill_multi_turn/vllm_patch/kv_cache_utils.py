"""KV-cache read-back for Speculative Prefill — Algorithm 1, line 11 (K half).

Copied verbatim from `../../spec_prefill_llama/vllm_patch/kv_cache_utils.py`
— reads raw Key vectors for a set of token slots out of a model's own paged
KV cache, generically across attention backends (TritonAttention,
FlashAttention, ...) and without assuming a uniform head_size across layers.

Reused for TWO purposes in this multi-turn pipeline (see the approved plan's
"Oracle upper bound" section and `proposer.py`'s module docstring):

1. The speculator's per-turn K read-back (via `proposer.py`'s persistent
   engine), same role as in the single-turn pipeline -- just now reading
   from a request's real, engine-managed KV-cache blocks (obtained through
   the request's own block table) instead of a scratch dummy cache built
   fresh per call.
2. The oracle-upper-bound path, reading the TARGET model's own KV cache
   after its unpruned prefill -- this module has zero engine or
   architecture assumptions baked in, so pointing it at the target's
   attention layers instead of the speculator's needs no code change here,
   only a different `attn_layer` argument from the caller.

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
context()`/`get_forward_context()` lookup** -- `kv_cache` and `attn_backend`
are both plain, persistent attributes on the `Attention` module itself, not
scoped to an active `set_forward_context(...)` block. `SpecPrefillProposer`
(and the oracle path) already hold direct references to each layer's
`Attention` instance, so there's no need to route through the
forward-context-scoped lookup at all -- this module has no vLLM dependency
whatsoever, not even a lazy one.
"""

from typing import List, Optional, Set, Tuple

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


def tensor_to_wire(t: torch.Tensor) -> dict:
    """Plain-Python-native (dict of str/list) encoding for a tensor that
    needs to cross `collective_rpc`'s process boundary.

    **No longer used by this pipeline's own `speculator_worker.py`/
    `proposer.py`** (kept here, not deleted, as a documented, still-tested
    fallback -- see below for why). Original problem this solved, confirmed
    on real hardware: a `torch.Tensor` returned directly as (or nested
    inside) a `collective_rpc` RPC-method's return value does NOT round-trip
    as a `Tensor` in this fork by default -- it comes back as a bare nested
    Python list, silently losing dtype/shape metadata (`AttributeError:
    'list' object has no attribute 'to'` at the call site). Root cause,
    traced through `vllm/v1/serial_utils.py`'s `MsgpackEncoder.enc_hook`:
    utility/collective_rpc results are wrapped in a `UtilityResult`, whose
    encoding only emits the type information needed to reconstruct nested
    tensors on decode when `envs.VLLM_ALLOW_INSECURE_SERIALIZATION` is set
    (`_encode_type_info_recursive(result)`); without it, it encodes as
    `(None, result)` -- no type info, so the decoder has no way to know a
    deeply-nested value was ever a `Tensor` rather than a plain list.

    Encoding explicitly here (dtype string + shape + a flat Python list of
    values) sidesteps that flag entirely -- plain, unambiguous msgpack-
    native data (str/int/float/list), no custom type hook needed on either
    side of the RPC boundary. Reconstruct with `tensor_from_wire`.

    **Why this got superseded**: real-hardware timing showed a single
    `retrieve_keys` call taking 89s for a 720M-element result under this
    encoding -- `.tolist()` (this function) plus msgpack list serialization
    plus `torch.tensor(list)` on the decode side are three unavoidably
    non-vectorized, element-by-element passes. `proposer.py` now sets
    `VLLM_ALLOW_INSECURE_SERIALIZATION=1` instead (see that module's
    docstring for why it's safe for this pipeline's own local-only IPC),
    which lets `UtilityResult` carry real type info so msgspec's OWN
    dedicated, zero-copy `_encode_tensor`/`_decode_tensor` codec handles
    Tensors directly -- no Python-side list conversion at all.

    **Why kept, not deleted, despite being unused now**: this specific
    change (enabling insecure serialization) has not been validated on real
    hardware as of when it was made -- unlike most of this pipeline's other
    mechanisms, which were each confirmed working via a dedicated
    validation script before being trusted (see `validate_proposer.py`,
    `validate_resumable_session.py`, `validate_sparse_attention.py`). If
    that validation surfaces a problem, this function (still covered by
    `test_vllm_patch.py`'s round-trip tests) is the documented, working
    fallback to revert `speculator_worker.py`'s `end_capture`/
    `retrieve_keys` to.
    """
    t = t.detach().to("cpu")
    return {
        "dtype": str(t.dtype).removeprefix("torch."),
        "shape": list(t.shape),
        "data": t.flatten().tolist(),
    }


def tensor_from_wire(d: dict) -> torch.Tensor:
    """Inverse of `tensor_to_wire` -- reconstructs a CPU tensor with the
    original dtype and shape from its plain-Python-native encoding."""
    dtype = getattr(torch, d["dtype"])
    flat = torch.tensor(d["data"], dtype=dtype)
    return flat.reshape(d["shape"])


def stack_decode_only_steps(
    steps: List[torch.Tensor], hidden_dim_fallback: int
) -> torch.Tensor:
    """Pure-tensor core of `speculator_worker.py`'s `SpeculatorGPUModelRunner
    .end_capture` bootstrap-exclusion logic -- lives here (not in
    `speculator_worker.py`, which imports vLLM at module level and so can't
    be imported without vLLM's full runtime installed) specifically so it's
    unit-testable without a live `GPUModelRunner` or a GPU at all -- see
    `end_capture`'s own docstring for the full reasoning behind filtering by
    shape rather than by capture timing (a real race, confirmed on hardware,
    that an earlier "reactively enable capture" design had), and
    `test_vllm_patch.py` for the CPU-only tests this enables.

    Args:
        steps: one request's raw per-forward-call captured query slices for
            ONE layer, in call order -- each `[num_tokens_this_step, H*D]`.
            May include leading multi-token entries (prefill / prefill-
            continuation chunks) before any number of single-token decode
            entries; may also be empty (nothing ever captured) or contain
            only multi-token entries (request ended before any real decode
            step, e.g. bootstrap-only + immediate EOS).
        hidden_dim_fallback: H*D to use for the returned empty tensor's last
            dim when there are no decode-shaped entries to infer it from.

    Returns:
        `[1, num_decode_steps, H*D]` -- only the entries where
        `step.shape[0] == 1` survive, stacked in original order along a new
        dim 1 (matching the shape `scoring.compute_attention_score` expects,
        num_samples=1). `[1, 0, hidden_dim_fallback]` if no decode-shaped
        entries exist.
    """
    decode_steps = [s for s in steps if s.shape[0] == 1]
    if decode_steps:
        return torch.stack(decode_steps, dim=1)
    return torch.empty(1, 0, hidden_dim_fallback)


def block_indices_from_positions(positions: List[int], block_size: int) -> Set[int]:
    """`{p // block_size for p in positions}` -- factored out to its own
    function specifically so `sparse_target_runner.py` can CACHE the result
    per turn (see that module's docstring on why this matters: `positions`
    here is the speculator's registered selection, which can be tens of
    thousands of absolute conversation-ledger positions for a large SCBench
    context, but is only registered ONCE per turn while
    `compute_sparse_gather_view` runs on every DECODE step of that turn --
    up to `max_tokens` times, e.g. 64 -- so recomputing this set from raw
    positions on every single step, as an earlier version of
    `compute_sparse_gather_view` did inline, was a real, measured
    per-conversation slowdown, not a theoretical one)."""
    return {p // block_size for p in positions}


def compute_sparse_gather_view(
    full_block_table_row: torch.Tensor,
    block_size: int,
    base_block_indices: Set[int],
    num_prompt: int,
    num_computed: int,
) -> Optional[Tuple[torch.Tensor, int]]:
    """Pure-tensor core of `sparse_target_runner.py`'s
    `SparseTargetGPUModelRunner._compute_gathered_view` -- lives here (not
    in `sparse_target_runner.py`, which imports vLLM's `GPUModelRunner` at
    module level and so can't be imported without vLLM's full runtime
    installed) specifically so it's unit-testable without a live
    `GPUModelRunner` or a GPU at all -- see that method's own docstring for
    the full reasoning behind block-level (not token-level) gathering and
    the force-keep rule for this turn's own in-progress decode tokens, and
    `test_vllm_patch.py` for the CPU-only tests this enables.

    Args:
        full_block_table_row: this request's own 1D tensor of physical
            block ids, as already built by the stock attention-metadata
            builder (read, not recomputed, from an already-built
            `AttentionMetadata` -- see caller).
        block_size: engine's KV-cache block size (tokens per block).
        base_block_indices: block indices the speculator's selection maps
            to for this turn -- i.e. `block_indices_from_positions(
            selected_positions, block_size)`, but computed and CACHED once
            per turn by the caller (see that function's own docstring),
            not recomputed inline here on every decode step. This function
            only adds the (small, per-step-bounded) in-progress-decode
            force-keep tail on top of it -- never mutates the set passed
            in (copies internally), so the same cached set is safe to pass
            again on the next decode step.
        num_prompt: this request's cumulative prompt length so far
            (`self.input_batch.num_prompt_tokens_cpu_tensor[req_idx]`),
            INCLUDING this turn's own query, per `_update_request_as_
            session`'s real behavior (confirmed: it extends
            `session.prompt_token_ids` with the new turn's tokens before
            decode begins).
        num_computed: this request's tokens actually computed so far
            (`self.input_batch.num_computed_tokens_cpu_tensor[req_idx]`) --
            for a decode step this is `>= num_prompt`, and
            `num_computed - num_prompt` is exactly how many tokens THIS
            turn has generated so far, always force-included regardless of
            `base_block_indices` (see module docstring's "Force-keep"
            reasoning in `sparse_target_runner.py`).

    Returns:
        `(gathered_block_table_row, gathered_seq_len)` -- the block ids to
        actually present to the attention kernel this step (a strict subset
        of `full_block_table_row`'s real, occupied entries, in ascending
        original order) and the corresponding shrunk sequence length
        (accounting for the last included block possibly being only
        partially filled -- see this function's own "last block occupancy"
        handling below). `None` if the selection is degenerate (covers
        every currently-resident block already) -- caller should leave the
        stock metadata untouched in that case rather than gather a no-op
        subset.

    Raises:
        ValueError: if `selected_positions` (plus the always-force-included
            in-progress-decode tail) reference a block index beyond what's
            actually allocated for this request yet -- a stale or
            out-of-range selection must fail loudly here, not silently read
            whatever physical block happens to sit at that index (which
            could belong to a completely different request or conversation
            by then).
    """
    selected_block_indices = set(base_block_indices)  # copy -- never mutate caller's cache
    for pos in range(num_prompt, num_computed):
        selected_block_indices.add(pos // block_size)

    if not selected_block_indices:
        return None

    sorted_blocks = sorted(selected_block_indices)

    num_full_blocks_present = -(-num_computed // block_size)  # ceil div
    if max(sorted_blocks) >= num_full_blocks_present:
        raise ValueError(
            f"selected block index {max(sorted_blocks)} exceeds the number "
            f"of blocks actually allocated so far ({num_full_blocks_present}, "
            f"for num_computed={num_computed} tokens at block_size="
            f"{block_size}) -- the speculator's selection must be scoped to "
            f"positions already resident in this request's OWN cache; a "
            f"stale/out-of-range selection would otherwise silently read "
            f"another request's or another conversation's data at that "
            f"physical block id."
        )
    if len(sorted_blocks) == num_full_blocks_present:
        # Degenerate: selection covers the entire resident cache already --
        # gathering would be a no-op.
        return None

    block_index_tensor = torch.tensor(
        sorted_blocks, dtype=torch.int64, device=full_block_table_row.device
    )
    gathered_block_table_row = full_block_table_row[block_index_tensor].clone()

    last_real_block_index = (num_computed - 1) // block_size
    last_real_block_occupancy = num_computed - last_real_block_index * block_size
    gathered_seq_len = 0
    for block_idx in sorted_blocks:
        if block_idx == last_real_block_index:
            gathered_seq_len += last_real_block_occupancy
        else:
            gathered_seq_len += block_size

    return gathered_block_table_row, gathered_seq_len


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
