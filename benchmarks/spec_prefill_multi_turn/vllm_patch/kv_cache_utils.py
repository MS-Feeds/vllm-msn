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

from dataclasses import dataclass
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


def is_decode_query_slice(start: int, end: int) -> bool:
    """Whether a forward call's `[start, end)` query slice for one request
    is a DECODE step (exactly 1 new token) and therefore worth capturing.

    The same rule `stack_decode_only_steps` below applies AFTER the fact
    (`step.shape[0] == 1`), expressed so `speculator_worker.py`'s capture
    hook can apply it BEFORE retaining anything -- and lives here, beside
    that function, for the same reason it does: `speculator_worker.py`
    imports vLLM at module level and so can't be imported without vLLM's
    full runtime, while this is a two-integer predicate that should be
    unit-testable on CPU.

    Applying it at capture time is what makes the after-the-fact filter
    free. A captured slice is a VIEW into the whole forward call's query
    tensor, so retaining one prefill slice pins that entire tensor, per
    layer, until the turn's scoring pass runs -- 30.3GiB for Llama-3.1-8B
    over a 124k-token context (32 layers x 0.95GiB), which is what OOM'd
    the first real ORACLE-k20 run. See `speculator_worker.py`'s
    `_capture_queries_by_request` docstring for the full accounting.

    `end <= start` (an empty slice -- a request present in the batch with
    no scheduled tokens) is not a decode step either, and returns False.
    """
    return end - start == 1


def prompt_tail_subslice(
    num_computed_before: int, num_scheduled: int, prompt_len: int, tail_n: int
) -> Optional[Tuple[int, int]]:
    """Which LOCAL rows `[lo, hi)` of this forward call's query slice hold the
    PROMPT's last `tail_n` positions -- `None` if this step covers none.

    The counterpart to `is_decode_query_slice` for the query-tail scoring
    source (ACCURACY_IMPROVEMENTS.md section 5a): score with the queries of
    the tokens the USER actually wrote, instead of with queries from tokens
    the scorer generated itself.

    **Why by POSITION and not by shape.** `is_decode_query_slice` selects the
    generated tokens, and does it by shape because every decode step is
    exactly 1 token. The prompt tail is the opposite case -- it arrives inside
    a PREFILL slice, which that predicate exists to reject, and rejects for a
    measured reason (retaining prefill slices pinned 30.3GiB and OOM'd the
    first ORACLE-k20 run; see `speculator_worker.py::_capture_queries_by_
    request`). Shape cannot express "the last N of the prompt" at all:

    - Under chunked prefill the tail may straddle two forward calls, or share
      one with non-tail tokens. Position arithmetic handles both without ever
      asking "is this the final chunk".
    - A prompt that is entirely a prefix-cache hit except its last token
      produces a 1-token PREFILL slice, indistinguishable by shape from a
      decode step -- the ambiguity `end_capture`'s own docstring flags.
      By position it is unambiguous.
    - Generated tokens sit at positions >= `prompt_len` and are excluded by
      the same predicate, with no second rule.

    Valid only where a request's positions are its own contiguous 0..L-1
    numbering. That holds for both engines this pipeline scores with: the
    speculator never prunes its own input (`proposer.py`'s docstring: it needs
    no RoPE-position override at all), and the target's session token stream
    IS the gapless conversation ledger (`sparse_selection_registry.py`'s
    docstring). It is NOT valid for the physically-pruned pipeline's
    gap-containing prompts, which is why this takes plain integers rather
    than reading positions from a runner.

    Retention is bounded by `tail_n` rows per forward call per layer, which is
    what keeps this affordable where capturing the whole prefill is not.

    Args:
        num_computed_before: this request's already-computed token count
            entering this step (vLLM's `num_computed_tokens_cpu[req_idx]`), so
            local row `j` carries absolute position `num_computed_before + j`.
        num_scheduled: tokens scheduled for this request this step (`end -
            start` of its query slice).
        prompt_len: the request's full submitted prompt length.
        tail_n: how many of the prompt's final positions to score with.

    Returns:
        `(lo, hi)` local row offsets within this step's slice, or `None`.
        Concatenating the returned rows over all of a request's forward calls
        yields exactly positions `[prompt_len - tail_n, prompt_len)`, in
        order, once each.
    """
    if num_scheduled <= 0 or tail_n <= 0:
        return None
    lo = max(0, (prompt_len - tail_n) - num_computed_before)
    hi = min(num_scheduled, prompt_len - num_computed_before)
    return (lo, hi) if hi > lo else None


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
    seq_len: int,
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
        seq_len: this request's TOTAL sequence length for THIS step --
            i.e. exactly what the stock metadata builder puts in
            `AttentionMetadata.seq_lens[req_idx]`, which is
            `num_computed_tokens + num_scheduled_tokens` (see
            `gpu_model_runner.py`'s `self.seq_lens[:num_reqs] =
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu`),
            NOT `num_computed_tokens_cpu_tensor[req_idx]` on its own.

            **This parameter used to be named `num_computed` and was
            genuinely fed `num_computed_tokens_cpu_tensor[req_idx]` by
            `sparse_target_runner.py` -- a real, confirmed off-by-one, not
            a naming nit.** `num_computed_tokens_cpu` is the count BEFORE
            this step's own scheduled token, so a decode step's true
            length is one MORE than it: passing it here built every
            gathered view one token short, which meant the token currently
            being decoded could never attend to its OWN key/value (and, at
            a block boundary, that the block physically holding it wasn't
            gathered at all). Positions `[num_prompt, seq_len)` -- this
            turn's own generated tokens INCLUDING the one being decoded
            right now -- are always force-included regardless of
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
    for pos in range(num_prompt, seq_len):
        selected_block_indices.add(pos // block_size)

    if not selected_block_indices:
        return None

    sorted_blocks = sorted(selected_block_indices)

    num_full_blocks_present = -(-seq_len // block_size)  # ceil div
    if max(sorted_blocks) >= num_full_blocks_present:
        raise ValueError(
            f"selected block index {max(sorted_blocks)} exceeds the number "
            f"of blocks actually allocated so far ({num_full_blocks_present}, "
            f"for seq_len={seq_len} tokens at block_size="
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

    last_real_block_index = (seq_len - 1) // block_size
    last_real_block_occupancy = seq_len - last_real_block_index * block_size
    gathered_seq_len = 0
    for block_idx in sorted_blocks:
        if block_idx == last_real_block_index:
            gathered_seq_len += last_real_block_occupancy
        else:
            gathered_seq_len += block_size

    return gathered_block_table_row, gathered_seq_len


@dataclass
class BaseGatherView:
    """Per-turn (not per-decode-step) precomputation for
    `compute_sparse_gather_view_incremental` -- see that function's and
    `compute_base_gather_view`'s docstrings for what problem this solves.
    Built ONCE per turn by `compute_base_gather_view` (cacheable by
    `sparse_target_runner.py` the same way `block_indices_from_positions`'s
    result already is, keyed by `id(selected_positions)`) and reused,
    unmodified, across every decode step of that turn."""

    stable_blocks: List[int]  # sorted, all strictly < boundary_block -- fully historical, occupancy can never change again this turn
    stable_rows: torch.Tensor  # full_block_table_row[stable_blocks], gathered ONCE
    stable_seq_len: int  # len(stable_blocks) * block_size -- every stable block is fully occupied by construction
    boundary_block: int  # num_prompt // block_size -- the one block index whose occupancy can still be ambiguous/growing this turn
    boundary_in_selection: bool  # whether the speculator's own selection includes boundary_block
    overflow_blocks: List[int]  # sorted, all > boundary_block -- should be empty for any legitimate selection (see compute_base_gather_view's docstring), kept only so a stale/out-of-range selection still fails loudly via the per-step range check instead of silently vanishing


def compute_base_gather_view(
    full_block_table_row: torch.Tensor,
    block_size: int,
    base_block_indices: Set[int],
    num_prompt: int,
) -> BaseGatherView:
    """Per-turn half of `compute_sparse_gather_view`'s work, factored out so
    `sparse_target_runner.py` can compute and cache it ONCE per turn --
    the earlier fix that cached `block_indices_from_positions`'s result
    (see that function's own docstring) wasn't enough on its own:
    `compute_sparse_gather_view` was still redoing a full `sorted()` +
    tensor-gather + seq_len loop over the ENTIRE selection on every decode
    step even after that fix -- a real, measured cost (see
    `compute_sparse_gather_view_incremental`'s docstring for numbers).

    Every position in a legitimate `base_block_indices` (the speculator's
    own selection, always scoped to positions < num_prompt -- see
    `sparse_selection_registry.py`'s docstring) maps to a block index
    `<= num_prompt // block_size` (this function's own `boundary_block`):
    a historical position p < num_prompt has
    `p // block_size <= (num_prompt - 1) // block_size <= num_prompt // block_size`.
    So every selected block except possibly `boundary_block` itself is
    provably FULLY occupied and provably STABLE for the rest of the turn
    (num_prompt is fixed once a turn's query is submitted) -- safe to
    gather and cache once. `boundary_block` is the one edge case: it may
    still be receiving newly-generated tokens as decode progresses (when
    `num_prompt` isn't a clean multiple of `block_size`), so its occupancy
    must be re-derived per step by `compute_sparse_gather_view_incremental`,
    same as the force-keep tail already was.

    `overflow_blocks` exists purely for defensive parity with
    `compute_sparse_gather_view`'s out-of-range check: a `base_block_indices`
    entry that (contrary to the invariant above) references a block beyond
    `boundary_block` must still be surfaced so the per-step incremental
    function's own range check catches it, rather than this function
    silently dropping it from `stable_blocks` and letting it vanish.
    """
    boundary_block = num_prompt // block_size
    stable_blocks = sorted(b for b in base_block_indices if b < boundary_block)
    overflow_blocks = sorted(b for b in base_block_indices if b > boundary_block)
    boundary_in_selection = boundary_block in base_block_indices

    if stable_blocks:
        stable_block_tensor = torch.tensor(
            stable_blocks, dtype=torch.int64, device=full_block_table_row.device
        )
        stable_rows = full_block_table_row[stable_block_tensor].clone()
    else:
        stable_rows = full_block_table_row[:0].clone()

    return BaseGatherView(
        stable_blocks=stable_blocks,
        stable_rows=stable_rows,
        stable_seq_len=len(stable_blocks) * block_size,
        boundary_block=boundary_block,
        boundary_in_selection=boundary_in_selection,
        overflow_blocks=overflow_blocks,
    )


def compute_sparse_gather_view_incremental(
    base_view: BaseGatherView,
    full_block_table_row: torch.Tensor,
    block_size: int,
    num_prompt: int,
    seq_len: int,
) -> Optional[Tuple[torch.Tensor, int]]:
    """Per-decode-step counterpart to `compute_base_gather_view` -- same
    contract/return value as `compute_sparse_gather_view` (this function IS
    a drop-in replacement for it, given a cached `base_view`), but does
    O(decode-tokens-so-far-this-turn) work per call instead of
    O(len(selection)): reuses `base_view.stable_rows`/`stable_seq_len`
    (already-gathered, already-summed) as-is and only computes the small,
    per-step-bounded delta -- the force-keep tail plus `boundary_block`'s
    own possibly-still-growing occupancy.

    `seq_len` carries exactly the same contract as
    `compute_sparse_gather_view`'s parameter of the same name -- the
    step's TOTAL sequence length (`num_computed_tokens +
    num_scheduled_tokens`, i.e. what the stock builder puts in
    `AttentionMetadata.seq_lens[req_idx]`), NOT `num_computed_tokens` on
    its own. See that function's own `seq_len` Args entry for the
    confirmed off-by-one that came from feeding it the latter.

    Real, measured cost this avoids (mirrors the reasoning already
    documented for `block_indices_from_positions`'s per-turn caching in
    `sparse_target_runner.py`): a synthetic benchmark reproducing a 64-step
    decode turn at SCBench scale (selection sizes up to tens of thousands
    of positions) showed the OLD per-step `compute_sparse_gather_view` call
    (even with `base_block_indices` already cached) costing 5-32x more
    wall-clock time per turn than this incremental version, because the old
    path still re-sorted and re-gathered the ENTIRE selection's blocks from
    `full_block_table_row` on every one of the 64 steps, when only the
    boundary block's occupancy and the force-keep tail actually change step
    to step.

    Callers MUST use the SAME `base_view` for every decode step of one turn
    (built once via `compute_base_gather_view`, cached the same way
    `sparse_target_runner.py` already caches `block_indices_from_positions`'s
    result -- keyed by `id(selected_positions)`, invalidated exactly once
    per `register_sparse_selection` call).
    """
    boundary_block = base_view.boundary_block

    if seq_len > num_prompt:
        dynamic_end = (seq_len - 1) // block_size + 1
        tail_blocks = set(range(boundary_block, dynamic_end))
    elif base_view.boundary_in_selection:
        tail_blocks = {boundary_block}
    else:
        tail_blocks = set()

    dynamic_blocks = sorted(tail_blocks | set(base_view.overflow_blocks))

    total_selected = len(base_view.stable_blocks) + len(dynamic_blocks)
    if total_selected == 0:
        return None

    num_full_blocks_present = -(-seq_len // block_size)  # ceil div
    dyn_max = dynamic_blocks[-1] if dynamic_blocks else base_view.stable_blocks[-1]
    if dyn_max >= num_full_blocks_present:
        raise ValueError(
            f"selected block index {dyn_max} exceeds the number of blocks "
            f"actually allocated so far ({num_full_blocks_present}, for "
            f"seq_len={seq_len} tokens at block_size={block_size}) "
            f"-- the speculator's selection must be scoped to positions "
            f"already resident in this request's OWN cache; a stale/"
            f"out-of-range selection would otherwise silently read another "
            f"request's or another conversation's data at that physical "
            f"block id."
        )

    if total_selected == num_full_blocks_present:
        # Degenerate: selection covers the entire resident cache already --
        # gathering would be a no-op.
        return None

    if dynamic_blocks:
        dynamic_block_tensor = torch.tensor(
            dynamic_blocks, dtype=torch.int64, device=full_block_table_row.device
        )
        dynamic_rows = full_block_table_row[dynamic_block_tensor].clone()
        gathered_block_table_row = torch.cat([base_view.stable_rows, dynamic_rows])
    else:
        gathered_block_table_row = base_view.stable_rows

    last_real_block_index = (seq_len - 1) // block_size
    last_real_block_occupancy = seq_len - last_real_block_index * block_size
    gathered_seq_len = base_view.stable_seq_len
    for block_idx in dynamic_blocks:
        if block_idx == last_real_block_index:
            gathered_seq_len += last_real_block_occupancy
        else:
            gathered_seq_len += block_size

    return gathered_block_table_row, gathered_seq_len


@dataclass
class PrefillGatherView:
    """Per-turn precomputation for `compute_prefill_gather_view` -- the
    PREFILL counterpart of `BaseGatherView`, and deliberately a separate
    type rather than a reuse of it.

    `BaseGatherView` splits the selection around
    `boundary_block = num_prompt // block_size`, which for a decode step is
    the block currently receiving generated tokens. During a PREFILL chunk
    that same quantity points past the end of the chunk entirely
    (`num_prompt` is the cumulative prompt length INCLUDING the whole
    delta, while `seq_len` has only reached this chunk's end), so its
    stable/boundary/overflow partition is meaningless here: `stable_blocks`
    would happily include blocks that are not yet written, and the
    per-step range check would fire on `boundary_block` itself. Hence a
    separate view whose split point is the TURN START, not `num_prompt`.

    Built ONCE per turn by `compute_prefill_base_view` and reused across
    every prefill chunk of that turn (`sparse_target_runner.py` caches it
    on the same `selection_generation` signal the decode views use)."""

    first_tail_block: int  # turn_start // block_size -- the first block spanned by THIS turn's own new tokens
    historical_blocks: List[int]  # sorted, all strictly < first_tail_block -- selected, already fully written, occupancy fixed
    historical_rows: torch.Tensor  # full_block_table_row[historical_blocks], gathered ONCE
    historical_seq_len: int  # len(historical_blocks) * block_size -- every one is fully occupied by construction


def compute_prefill_base_view(
    full_block_table_row: torch.Tensor,
    block_size: int,
    base_block_indices: Set[int],
    turn_start: int,
) -> PrefillGatherView:
    """Per-turn half of `compute_prefill_gather_view`'s work.

    `turn_start` is the absolute position at which THIS turn's own new
    tokens begin -- i.e. the target session's resident length just before
    the turn's delta was submitted, supplied by the driver at registration
    time (`sparse_selection_registry.register`'s `prefill_turn_start`).
    The runner cannot derive it: it sees only `num_computed` (which has
    already advanced past `turn_start` on every chunk after the first) and
    `num_prompt` (which is the END of the delta, not its start).

    Selection blocks at or above `first_tail_block` are dropped here rather
    than kept: they cover this turn's own delta positions, which
    `compute_prefill_gather_view` force-includes wholesale anyway (see its
    docstring's "Why the tail must be contiguous"). Dropping them is what
    makes the remaining `historical_blocks` provably safe to gather and
    cache -- every one is strictly below
    `first_tail_block <= (seq_len - 1) // block_size`, hence fully written
    and fully occupied, for every chunk of this turn.
    """
    first_tail_block = turn_start // block_size
    historical_blocks = sorted(b for b in base_block_indices if b < first_tail_block)

    if historical_blocks:
        historical_block_tensor = torch.tensor(
            historical_blocks, dtype=torch.int64, device=full_block_table_row.device
        )
        historical_rows = full_block_table_row[historical_block_tensor].clone()
    else:
        historical_rows = full_block_table_row[:0].clone()

    return PrefillGatherView(
        first_tail_block=first_tail_block,
        historical_blocks=historical_blocks,
        historical_rows=historical_rows,
        historical_seq_len=len(historical_blocks) * block_size,
    )


def compute_prefill_gather_view(
    base_view: PrefillGatherView,
    full_block_table_row: torch.Tensor,
    block_size: int,
    seq_len: int,
) -> Optional[Tuple[torch.Tensor, int]]:
    """Restrict a PREFILL chunk's K/V side to the speculator's selection.

    Same contract/return value as `compute_sparse_gather_view_incremental`
    -- `(gathered_block_table_row, gathered_seq_len)`, or `None` for the
    degenerate "selection already covers every resident block" case that
    the caller should leave dense.

    Args:
        base_view: this turn's cached `PrefillGatherView`.
        full_block_table_row: this request's real block-table row.
        block_size: KV block size.
        seq_len: this chunk's TOTAL sequence length (`num_computed_tokens +
            num_scheduled_tokens`), exactly the same contract as
            `compute_sparse_gather_view`'s parameter of the same name.

    ## Why the tail must be contiguous (the causal-mask invariant)

    Every block from `first_tail_block` through the last written block is
    force-included, unconditionally and CONTIGUOUSLY -- not just the ones
    the selection happens to name. That is not over-caution, it is what
    makes the gathered view legal input to a causal kernel at all.

    FlashAttention masks a chunk of `n_q` query tokens against `L` keys
    bottom-right-aligned: query `i` may attend key indices
    `[0, L - n_q + i]`. The gathered view is a COMPACTED renumbering of the
    real K/V stream, so that arithmetic is only correct if the chunk's own
    `n_q` tokens are the final `n_q` entries of the view, in order. They
    are, given a contiguous tail: this chunk occupies real positions
    `[seq_len - n_q, seq_len)`, the tail covers real positions
    `[first_tail_block * block_size, seq_len)` with no holes, and
    `first_tail_block * block_size <= turn_start <= seq_len - n_q`. Punch
    one hole in that tail -- by honoring the selection inside it -- and
    every query in the chunk silently attends the wrong keys.

    The same contiguity is what keeps this turn's earlier prefill chunks
    visible to its later ones, which they must be: a query token that
    cannot see the first half of its own question is not a cheaper
    computation, it is a different one.

    RoPE needs no adjustment here for the same reason it needs none on the
    decode path (`sparse_target_runner.py`'s module docstring): rotation is
    baked into K at write time, and the query's own position comes from
    `self.positions`, which this mechanism never touches.

    ## Turn 0 degenerates to dense, by construction

    At turn 0 `turn_start == 0`, so `first_tail_block == 0` and the tail is
    every block that exists -- `total_selected == num_full_blocks_present`
    and this returns `None`. That is the correct answer rather than a
    special case worth writing: turn 0's prefill is where the context's KV
    is COMPUTED for the first time, and computing it under a restricted
    view would poison the persistent cache that every later turn's
    selection reads from.
    """
    if seq_len <= 0:
        return None

    num_full_blocks_present = -(-seq_len // block_size)  # ceil div
    if base_view.first_tail_block >= num_full_blocks_present:
        raise ValueError(
            f"prefill turn-start block {base_view.first_tail_block} is at or "
            f"beyond the number of blocks actually allocated so far "
            f"({num_full_blocks_present}, for seq_len={seq_len} tokens at "
            f"block_size={block_size}) -- the registered prefill_turn_start "
            f"must lie inside this request's own already-submitted stream. A "
            f"turn_start past it means the driver's resident-length tracking "
            f"has drifted from the engine's, and gathering on it would read "
            f"unwritten (or another request's) physical blocks."
        )

    tail_blocks = list(range(base_view.first_tail_block, num_full_blocks_present))
    total_selected = len(base_view.historical_blocks) + len(tail_blocks)
    if total_selected == num_full_blocks_present:
        # Degenerate: the selection plus the force-kept tail already span
        # every resident block (turn 0 always lands here -- see docstring).
        return None

    tail_block_tensor = torch.tensor(
        tail_blocks, dtype=torch.int64, device=full_block_table_row.device
    )
    tail_rows = full_block_table_row[tail_block_tensor].clone()
    gathered_block_table_row = torch.cat([base_view.historical_rows, tail_rows])

    # The tail is contiguous and ends at the last written block, so exactly
    # one gathered block (the final one) can be partially occupied.
    last_real_block_index = (seq_len - 1) // block_size
    last_real_block_occupancy = seq_len - last_real_block_index * block_size
    gathered_seq_len = (
        base_view.historical_seq_len
        + (len(tail_blocks) - 1) * block_size
        + last_real_block_occupancy
    )

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
