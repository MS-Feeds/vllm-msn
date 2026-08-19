"""Standalone tests for the multi-turn SpecPrefill pieces built in
vllm_patch/ that don't require vLLM's full runtime or a real model
checkpoint -- see EXPERIMENT_PLAN.md for scope.

Covers: scoring.py's math, prefill_split.py, kv_cache_utils.py's
backend-layout logic, and pruning_registry.py's PruneRecord store --
copied over from `../spec_prefill_llama/test_vllm_patch.py` UNCHANGED
(these modules have zero logic changes for the multi-turn port, see their
own docstrings) -- plus NEW tests for `conversation_state.py`, the one
module this port adds that has no single-turn analog and is fully
CPU-testable.

The speculator engine (`speculator_worker.py`/`proposer.py`) and the
target-side conversation-aware pruning driver (`pruner.py`'s
`compute_pruned_turn`/`prune_and_add_turn`) require a running vLLM engine
with a loaded model and are not exercised here -- see `validate_proposer.py`
and each module's own "Known risk areas" notes for what a live run would
need to confirm.

Also covers `predict_scbench.py::LedgerToTargetPositionMap` (a driver-level,
not `vllm_patch/`, class -- imported here anyway since it's pure Python,
CPU-testable, and the sparse-attention experiment path's correctness
depends on its arithmetic being exactly right; see that class's own
docstring for what it's translating between and why).

`sparse_target_runner.py::SparseTargetGPUModelRunner` is a PARTIAL
exception to the "no vLLM runtime" rule above: `_apply_sparse_attention_
overrides` and `_step_seq_len` are exercised here against a hand-built
fake runner, with stub base classes standing in for the two vLLM symbols
that module imports at top level (see `_load_sparse_target_runner`'s
docstring for exactly what that does and does NOT prove). This exists
because both of the bugs that produced repetition-loop SPARSE output --
the gather being fed `num_computed_tokens` instead of the step's real
`seq_len`, and the chat-template scaffolding never reaching the registered
selection -- lived in caller wiring that the pure-`kv_cache_utils` tests
were structurally unable to see: those tests were self-consistent with the
wrong caller. `_build_attention_metadata` is still not exercised (its body
is the `super()` call a stub can't honestly stand in for), and every
field-name/backend assumption remains `validate_sparse_attention.py`'s job
on real hardware.

Run with: python3 test_vllm_patch.py
"""

import math
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from vllm_patch.config import SpecConfig
from vllm_patch.conversation_state import ConversationState
from vllm_patch.kv_cache_utils import (
    _find_kv_split_dim,
    block_indices_from_positions,
    compute_base_gather_view,
    compute_sparse_gather_view,
    compute_sparse_gather_view_incremental,
    gather_keys_for_slots,
    stack_decode_only_steps,
    tensor_from_wire,
    tensor_to_wire,
)
from vllm_patch import sparse_selection_registry
from vllm_patch.prefill_split import split_prefill_decode_requests
from vllm_patch import pruning_registry
from vllm_patch.pruning_registry import PruneRecord
from vllm_patch.scoring import (
    aggregate_attention_score,
    chunk_select_from_smoothed_attention,
    compute_attention_score,
    score_and_select_indices,
)
from vllm_patch.pruner import _positions_from_kept_indices
from predict_scbench import LedgerToTargetPositionMap, build_turn_delta_ids


class _FakeRequest:
    def __init__(self, is_prefill_chunk: bool):
        self.is_prefill_chunk = is_prefill_chunk


def test_split_prefill_decode_requests():
    batch = [_FakeRequest(True), _FakeRequest(False), _FakeRequest(True)]
    prefill, decode = split_prefill_decode_requests(batch)
    assert len(prefill) == 2 and all(r.is_prefill_chunk for r in prefill)
    assert len(decode) == 1 and not decode[0].is_prefill_chunk


def _synthetic_qk(num_layers, num_samples, num_heads, num_kv_heads, head_dim, look_ahead, ctx_len):
    query_buffer = [
        torch.randn(num_samples, look_ahead, num_heads * head_dim) for _ in range(num_layers)
    ]
    key_buffer = [
        [torch.randn(ctx_len, num_kv_heads, head_dim) for _ in range(num_samples)]
        for _ in range(num_layers)
    ]
    return query_buffer, key_buffer


def test_compute_attention_score_shape():
    num_layers, num_samples, num_heads, num_kv_heads, head_dim, look_ahead, ctx_len = (
        2, 2, 4, 4, 8, 3, 16,
    )
    query_buffer, key_buffer = _synthetic_qk(
        num_layers, num_samples, num_heads, num_kv_heads, head_dim, look_ahead, ctx_len
    )
    attn_scores = compute_attention_score(query_buffer, key_buffer, [look_ahead] * num_samples)
    assert len(attn_scores) == num_samples
    assert attn_scores[0].shape == (num_layers, num_heads, look_ahead, ctx_len)


def test_compute_attention_score_gqa():
    num_layers, num_samples, num_heads, num_kv_heads, head_dim, look_ahead, ctx_len = (
        2, 2, 4, 2, 8, 3, 16,
    )
    query_buffer, key_buffer = _synthetic_qk(
        num_layers, num_samples, num_heads, num_kv_heads, head_dim, look_ahead, ctx_len
    )
    attn_scores = compute_attention_score(query_buffer, key_buffer, [look_ahead] * num_samples)
    assert attn_scores[0].shape == (num_layers, num_heads, look_ahead, ctx_len)


def test_aggregate_and_select_pipeline():
    cfg = SpecConfig(
        keep_strategy="percentage",
        keep_kwargs={"chunk": True, "chunk_size": 4, "percentage": 0.5},
        look_ahead_cnt=3,
        pool_kernel_size=None,
    )
    query_buffer, key_buffer = _synthetic_qk(2, 2, 4, 4, 8, 3, 16)
    attn_scores = compute_attention_score(query_buffer, key_buffer, [3, 3])

    token_importance = aggregate_attention_score(attn_scores, cfg)
    assert token_importance[0].shape == (16,)

    kept = chunk_select_from_smoothed_attention(token_importance, cfg)
    assert kept[0].numel() == 8  # 50% of 16, chunk_size=4 divides evenly
    assert torch.equal(kept[0], torch.sort(kept[0])[0])  # sorted, per algorithm

    cfg_keep_all = SpecConfig(
        keep_strategy="percentage", keep_kwargs={"percentage": 1.0}, look_ahead_cnt=3
    )
    kept_all = chunk_select_from_smoothed_attention(token_importance, cfg_keep_all)
    assert kept_all[0].numel() == 16


def _reference_chunked_topk(sample_ti: torch.Tensor, chunk_size: int, percentage: float):
    """Deliberately slow, obviously-correct reference implementation of
    `chunk_select_from_smoothed_attention`'s chunked branch (plain Python,
    no vectorization tricks) -- used to cross-check
    `scoring._chunked_topk_indices`'s vectorized implementation, which
    replaced an equivalent per-chunk Python loop (see that function's
    docstring for why: the original loop's `.item()`-per-kept-chunk call
    is a real GPU-sync cost on real hardware). Random-valued inputs make
    tie-breaking between torch.topk and Python's sorted() irrelevant in
    practice (ties on continuous random floats have probability ~0)."""
    seq_len = sample_ti.shape[0]
    chunks = [sample_ti[i : i + chunk_size] for i in range(0, seq_len, chunk_size)]
    chunk_means = [c.mean().item() for c in chunks]
    chunk_cnt = len(chunks)
    keep_cnt = math.ceil(chunk_cnt * percentage)
    kept_chunk_order = sorted(range(chunk_cnt), key=lambda i: -chunk_means[i])[:keep_cnt]
    indices = []
    for ci in sorted(kept_chunk_order):
        start = ci * chunk_size
        end = min(start + chunk_size, seq_len)
        indices.extend(range(start, end))
    return sorted(indices)


def test_chunk_select_from_smoothed_attention_matches_reference_divisible_and_not():
    """Cross-checks the vectorized `_chunked_topk_indices` (which replaced
    a per-chunk `.mean()`/`.item()` Python loop -- see scoring.py) against
    an independent, unvectorized reference across both a length exactly
    divisible by chunk_size and one that isn't (the padding/remainder-chunk
    path)."""
    torch.manual_seed(42)
    for seq_len in [96, 100, 4000, 4010]:  # 96/4000 divisible by 32, 100/4010 not
        for percentage in [0.3, 0.5, 1.0]:
            sample_ti = torch.rand(seq_len)
            cfg = SpecConfig(
                keep_strategy="percentage",
                keep_kwargs={"chunk": True, "chunk_size": 32, "percentage": percentage},
            )
            got = chunk_select_from_smoothed_attention([sample_ti], cfg)[0]
            expected = _reference_chunked_topk(sample_ti, 32, percentage)
            assert got.tolist() == expected, (
                f"seq_len={seq_len} percentage={percentage}: "
                f"got {got.tolist()} expected {expected}"
            )


def test_chunk_select_from_smoothed_attention_chunk_larger_than_seq_len():
    """A single, partially-empty chunk (seq_len < chunk_size) must still
    keep exactly the real tokens, not the padded tail."""
    torch.manual_seed(7)
    sample_ti = torch.rand(10)
    cfg = SpecConfig(
        keep_strategy="percentage",
        keep_kwargs={"chunk": True, "chunk_size": 32, "percentage": 1.0},
    )
    got = chunk_select_from_smoothed_attention([sample_ti], cfg)[0]
    assert got.tolist() == list(range(10))


def test_score_and_select_indices_returns_sorted_local_indices():
    """`score_and_select_indices` is the single-sample convenience wrapper
    `speculator_worker.py::end_capture_and_score` calls IN-PROCESS (see
    that method's docstring) instead of shipping Q/K back to the driver --
    exercises the exact input shape that in-process call site uses: one
    `torch.Tensor` per layer (no per-sample list nesting, unlike
    `_synthetic_qk`'s shape for `compute_attention_score` directly)."""
    num_layers, num_heads, num_kv_heads, head_dim, look_ahead, ctx_len = 2, 4, 4, 8, 3, 16
    query_buffer = [torch.randn(1, look_ahead, num_heads * head_dim) for _ in range(num_layers)]
    key_buffer_per_layer = [torch.randn(ctx_len, num_kv_heads, head_dim) for _ in range(num_layers)]
    cfg = SpecConfig(keep_strategy="percentage", keep_kwargs={"percentage": 0.5}, look_ahead_cnt=look_ahead)

    kept = score_and_select_indices(query_buffer, key_buffer_per_layer, look_ahead, cfg)

    assert isinstance(kept, list)
    assert all(isinstance(i, int) for i in kept)
    assert kept == sorted(kept)
    assert len(kept) == 8  # ceil(16 * 0.5)
    assert all(0 <= i < ctx_len for i in kept)


def test_score_and_select_indices_keep_all():
    num_layers, num_heads, num_kv_heads, head_dim, look_ahead, ctx_len = 1, 2, 2, 4, 2, 6
    query_buffer = [torch.randn(1, look_ahead, num_heads * head_dim) for _ in range(num_layers)]
    key_buffer_per_layer = [torch.randn(ctx_len, num_kv_heads, head_dim) for _ in range(num_layers)]
    cfg = SpecConfig(keep_strategy="percentage", keep_kwargs={"percentage": 1.0}, look_ahead_cnt=look_ahead)

    kept = score_and_select_indices(query_buffer, key_buffer_per_layer, look_ahead, cfg)
    assert kept == list(range(ctx_len))


def test_positions_from_kept_indices_none_keeps_everything():
    """`kept_local_indices=None` is the "0 lookahead steps, no scoring
    signal" fallback -- both real callers (`compute_pruned_turn`'s
    speculator path, `_score_and_select`'s oracle path) pass this only
    when `actual_look_ahead_cnt == 0`."""
    candidate_pool = [(10, 0), (11, 1), (12, 2)]
    force_keep_query = [(20, 3), (21, 4)]
    pruned_token_ids, kept_positions, orig_len, kept_history_pairs = _positions_from_kept_indices(
        candidate_pool, force_keep_query, None
    )
    assert pruned_token_ids == [10, 11, 12, 20, 21]
    assert kept_positions == [0, 1, 2, 3, 4]
    assert orig_len == 5
    assert kept_history_pairs == candidate_pool


def test_positions_from_kept_indices_selects_subset_and_force_keeps_query():
    candidate_pool = [(10, 0), (11, 1), (12, 2), (13, 3)]
    force_keep_query = [(20, 4), (21, 5)]
    # Scorer selected local indices 0, 2 (within candidate_len=4) plus 4, 5
    # (the force-kept query span's own local indices, since
    # score_and_select_indices scores the WHOLE submitted sequence) -- the
    # query-span indices must be filtered out here (force_keep_query is
    # appended unconditionally below), not double-counted.
    kept_local_indices = [0, 2, 4, 5]

    pruned_token_ids, kept_positions, orig_len, kept_history_pairs = _positions_from_kept_indices(
        candidate_pool, force_keep_query, kept_local_indices
    )
    assert kept_history_pairs == [(10, 0), (12, 2)]
    assert pruned_token_ids == [10, 12, 20, 21]
    assert kept_positions == [0, 2, 4, 5]
    assert orig_len == 6


def test_positions_from_kept_indices_empty_force_keep_query():
    candidate_pool = [(10, 0), (11, 1)]
    pruned_token_ids, kept_positions, orig_len, kept_history_pairs = _positions_from_kept_indices(
        candidate_pool, [], [1]
    )
    assert kept_history_pairs == [(11, 1)]
    assert pruned_token_ids == [11]
    assert kept_positions == [1]
    assert orig_len == 2


def test_kv_split_dim_triton_layout():
    class FakeTritonBackend:
        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
            return (num_blocks, 2, block_size, num_kv_heads, head_size)

        @classmethod
        def get_kv_cache_block_dim(cls, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
            _S = 1234567
            return cls.get_kv_cache_shape(_S, block_size, num_kv_heads, head_size).index(_S)

    block_size, num_kv_heads, head_size, num_blocks = 16, 4, 128, 100
    kv_cache = torch.zeros(num_blocks, 2, block_size, num_kv_heads, head_size)
    split_dim = _find_kv_split_dim(FakeTritonBackend, kv_cache, block_size, num_kv_heads, head_size)
    assert split_dim == 1


def test_tensor_wire_roundtrip_preserves_shape_dtype_values():
    """Regression guard for a real bug found on real hardware: a raw
    torch.Tensor returned through collective_rpc silently degrades to a
    bare nested list (AttributeError: 'list' object has no attribute 'to'
    at the call site) -- see kv_cache_utils.tensor_to_wire's docstring.
    Doesn't exercise collective_rpc itself (no vLLM engine here), just the
    encode/decode round-trip these RPC boundaries are built on."""
    original = torch.randn(3, 4, 5, dtype=torch.bfloat16)
    wire = tensor_to_wire(original)
    assert isinstance(wire, dict)
    assert wire["dtype"] == "bfloat16"
    assert wire["shape"] == [3, 4, 5]
    assert isinstance(wire["data"], list)

    restored = tensor_from_wire(wire)
    assert restored.shape == original.shape
    assert restored.dtype == original.dtype
    assert torch.equal(restored, original)


def test_tensor_wire_roundtrip_empty_tensor():
    original = torch.empty(1, 0, 8, dtype=torch.float32)
    restored = tensor_from_wire(tensor_to_wire(original))
    assert restored.shape == original.shape
    assert restored.dtype == original.dtype


def test_stack_decode_only_steps_drops_single_bootstrap_entry():
    """Regression guard for a real bug found on real hardware: capture used
    to be enabled reactively (in response to observed output progress),
    which raced against EngineCore's own autonomous stepping and silently
    dropped real decode steps (requested look_ahead_cnt=8, actually captured
    6). Fixed by always capturing from before add_request and filtering the
    bootstrap's own entry out by SHAPE here instead of by timing -- this
    test exercises that filter directly (no GPU/engine needed)."""
    HD = 16
    bootstrap = torch.randn(5, HD)  # 5-token prefill chunk -- must be dropped
    decode_steps = [torch.randn(1, HD) for _ in range(8)]
    steps = [bootstrap] + decode_steps

    result = stack_decode_only_steps(steps, hidden_dim_fallback=HD)
    assert result.shape == (1, 8, HD)
    for i, expected in enumerate(decode_steps):
        assert torch.equal(result[0, i], expected[0])


def test_stack_decode_only_steps_drops_multiple_leading_chunks():
    """Chunked prefill: several multi-token entries can precede real decode
    -- not just a single bootstrap entry. Filtering by shape (not "drop
    index 0") must handle this regardless of how many chunks preceded."""
    HD = 8
    chunk1 = torch.randn(10, HD)
    chunk2 = torch.randn(7, HD)
    decode_steps = [torch.randn(1, HD) for _ in range(3)]
    steps = [chunk1, chunk2] + decode_steps

    result = stack_decode_only_steps(steps, hidden_dim_fallback=HD)
    assert result.shape == (1, 3, HD)


def test_stack_decode_only_steps_no_decode_steps_returns_empty():
    """Bootstrap-only (e.g. immediate EOS on the bootstrap's own candidate
    token) -- zero decode steps captured, must return an empty tensor with
    the right hidden dim, not crash on torch.stack([])."""
    HD = 12
    steps = [torch.randn(4, HD)]  # only a prefill chunk, no decode at all
    result = stack_decode_only_steps(steps, hidden_dim_fallback=HD)
    assert result.shape == (1, 0, HD)


def test_stack_decode_only_steps_no_prefill_chunk_all_decode():
    """Degenerate case: the entire prompt was already prefix-cache-hit, so
    even the "bootstrap" forward call is itself a genuine 1-token decode
    step -- no multi-token entry to filter out at all. Must not
    special-case this away."""
    HD = 4
    decode_steps = [torch.randn(1, HD) for _ in range(5)]
    result = stack_decode_only_steps(list(decode_steps), hidden_dim_fallback=HD)
    assert result.shape == (1, 5, HD)


def test_stack_decode_only_steps_empty_input():
    result = stack_decode_only_steps([], hidden_dim_fallback=6)
    assert result.shape == (1, 0, 6)


def test_gather_keys_for_slots():
    num_slots, num_kv_heads, head_size = 10, 3, 4
    flat_keys = torch.arange(num_slots * num_kv_heads * head_size, dtype=torch.float32).reshape(
        num_slots, num_kv_heads, head_size
    )
    slot_mapping = torch.tensor([0, 5, 9])
    gathered = gather_keys_for_slots(flat_keys, slot_mapping)
    assert gathered.shape == (3, num_kv_heads, head_size)
    assert torch.equal(gathered[0], flat_keys[0])
    assert torch.equal(gathered[1], flat_keys[5])
    assert torch.equal(gathered[2], flat_keys[9])


def test_compute_sparse_gather_view_basic_selection():
    """block_size=4, 6 blocks resident, `seq_len == num_prompt == 24` --
    a pure-arithmetic edge case with no in-progress decode tail at all
    (a REAL decode step always has `seq_len >= num_prompt + 1`, see
    `test_gather_view_includes_the_token_being_decoded`). Selecting
    positions in blocks 0 and 3 should gather exactly those two blocks,
    in ascending order."""
    block_size = 4
    full_block_table_row = torch.tensor([100, 101, 102, 103, 104, 105])  # 6 blocks
    result = compute_sparse_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        base_block_indices=block_indices_from_positions([1, 14], block_size),  # blocks 0, 3
        num_prompt=24,
        seq_len=24,
    )
    assert result is not None
    gathered_row, gathered_seq_len = result
    assert torch.equal(gathered_row, torch.tensor([100, 103]))  # blocks 0 and 3
    assert gathered_seq_len == 2 * block_size  # both fully-occupied historical blocks


def test_compute_sparse_gather_view_force_keeps_in_progress_decode_tail():
    """Even with an empty selection, tokens generated so far THIS turn
    (num_prompt..seq_len) must always be force-included -- the model
    needs coherent access to what it just generated within the same turn,
    regardless of what the speculator chose."""
    block_size = 4
    full_block_table_row = torch.tensor([100, 101, 102, 103, 104])  # 5 blocks
    # num_prompt=16 (4 blocks fully resident before this turn's decode
    # started), seq_len=18 -- so positions 16 and 17 belong to this turn's
    # own generated tail (position 17 being the token currently decoding),
    # both landing in block index 4.
    result = compute_sparse_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        base_block_indices=set(),  # nothing selected by the speculator at all
        num_prompt=16,
        seq_len=18,
    )
    assert result is not None
    gathered_row, gathered_seq_len = result
    assert torch.equal(gathered_row, torch.tensor([104]))  # block 4, force-kept
    assert gathered_seq_len == 2  # only 2 of that block's 4 slots are occupied


def test_compute_sparse_gather_view_combines_selection_and_force_keep():
    block_size = 4
    full_block_table_row = torch.tensor([100, 101, 102, 103, 104])
    result = compute_sparse_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        base_block_indices=block_indices_from_positions([0], block_size),  # block 0
        num_prompt=16,
        seq_len=17,  # 1 token generated so far this turn -> block 4
    )
    assert result is not None
    gathered_row, gathered_seq_len = result
    assert torch.equal(gathered_row, torch.tensor([100, 104]))  # block 0 + force-kept block 4
    assert gathered_seq_len == block_size + 1  # block 0 full + block 4's 1 occupied slot


def test_compute_sparse_gather_view_degenerate_full_selection_returns_none():
    """Selecting every currently-resident block is a no-op -- caller should
    leave the stock (already-correct) metadata untouched rather than
    "gather" a subset that's just the whole thing again."""
    block_size = 4
    full_block_table_row = torch.tensor([100, 101])  # 2 blocks, 8 tokens
    result = compute_sparse_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        base_block_indices=block_indices_from_positions([0, 4], block_size),  # blocks 0, 1
        num_prompt=8,
        seq_len=8,
    )
    assert result is None


def test_compute_sparse_gather_view_out_of_range_raises():
    """A selection referencing a block beyond what's actually allocated
    must fail loudly -- silently reading whatever physical block sits at
    that index could be a completely different request's data."""
    block_size = 4
    full_block_table_row = torch.tensor([100, 101])  # only 2 blocks allocated
    try:
        compute_sparse_gather_view(
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            base_block_indices=block_indices_from_positions([100], block_size),  # block 25
            num_prompt=8,
            seq_len=8,
        )
        raise AssertionError("expected ValueError for out-of-range block selection")
    except ValueError:
        pass


def test_compute_sparse_gather_view_preserves_ascending_order():
    """Selected positions arriving out of order must still gather blocks in
    ascending original order, not selection order -- the gathered row's
    order determines the virtual sequence's own (causally-irrelevant for
    single-token decode, but still deterministic) position mapping."""
    block_size = 4
    full_block_table_row = torch.tensor([100, 101, 102, 103])
    result = compute_sparse_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        # blocks 3, 0, 1 -- deliberately unordered input positions
        base_block_indices=block_indices_from_positions([12, 0, 4], block_size),
        num_prompt=16,
        seq_len=16,
    )
    assert result is not None
    gathered_row, _ = result
    assert torch.equal(gathered_row, torch.tensor([100, 101, 103]))  # blocks 0, 1, 3


def test_block_indices_from_positions_dedupes_by_block():
    assert block_indices_from_positions([1, 2, 3, 14, 15], block_size=4) == {0, 3}


def test_block_indices_from_positions_empty():
    assert block_indices_from_positions([], block_size=4) == set()


def test_compute_sparse_gather_view_does_not_mutate_caller_base_set():
    """`sparse_target_runner.py` caches `base_block_indices` per turn and
    reuses the SAME set object across every decode step of that turn (see
    that module's `_get_base_block_indices`) -- `compute_sparse_gather_view`
    must copy it internally, never mutate it in place, or the second decode
    step of a turn would see stale force-keep entries leaked in from the
    first."""
    block_size = 4
    full_block_table_row = torch.tensor([100, 101, 102, 103, 104])
    base_block_indices = block_indices_from_positions([0], block_size)  # {0}
    original = set(base_block_indices)
    compute_sparse_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        base_block_indices=base_block_indices,
        num_prompt=16,
        seq_len=17,  # force-keep tail would add block 4 if it mutated the input
    )
    assert base_block_indices == original


def _run_both_gather_paths(full_block_table_row, block_size, base_block_indices, num_prompt, seq_len, base_view=None):
    """Runs both the old ground-truth `compute_sparse_gather_view` (full
    recompute every call) and the new `compute_base_gather_view` +
    `compute_sparse_gather_view_incremental` (base view cached/reused
    across calls, like `sparse_target_runner.py` now does) for one decode
    step, returning `(old_result, new_result, base_view)` so callers can
    compare and reuse `base_view` across a simulated multi-step turn."""
    if base_view is None:
        base_view = compute_base_gather_view(
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            base_block_indices=base_block_indices,
            num_prompt=num_prompt,
        )

    old_error = None
    try:
        old_result = compute_sparse_gather_view(
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            base_block_indices=set(base_block_indices),
            num_prompt=num_prompt,
            seq_len=seq_len,
        )
    except ValueError as e:
        old_result = None
        old_error = e

    new_error = None
    try:
        new_result = compute_sparse_gather_view_incremental(
            base_view=base_view,
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            num_prompt=num_prompt,
            seq_len=seq_len,
        )
    except ValueError as e:
        new_result = None
        new_error = e

    assert (old_error is None) == (new_error is None), (
        f"error mismatch at seq_len={seq_len}: old={old_error!r} new={new_error!r}"
    )
    if old_error is None:
        if old_result is None or new_result is None:
            assert old_result is None and new_result is None, (
                f"None mismatch at seq_len={seq_len}: old={old_result} new={new_result}"
            )
        else:
            old_row, old_seq_len = old_result
            new_row, new_seq_len = new_result
            assert torch.equal(old_row, new_row), (
                f"row mismatch at seq_len={seq_len}: old={old_row} new={new_row}"
            )
            assert old_seq_len == new_seq_len, (
                f"seq_len mismatch at seq_len={seq_len}: old={old_seq_len} new={new_seq_len}"
            )
    return base_view


def test_gather_view_incremental_matches_full_recompute_on_existing_scenarios():
    """Re-runs every scenario the original `compute_sparse_gather_view`
    unit tests above cover (basic selection, force-keep tail, combined,
    degenerate, out-of-range, ascending-order) through the new cached
    `compute_base_gather_view` + `compute_sparse_gather_view_incremental`
    path and asserts byte-for-byte identical behavior, including the
    ValueError case -- this is the regression guard for
    `sparse_target_runner.py`'s per-turn caching optimization (see that
    module's docstring's "Per-turn caching" section)."""
    block_size = 4

    _run_both_gather_paths(
        torch.tensor([100, 101, 102, 103, 104, 105]), block_size,
        block_indices_from_positions([1, 14], block_size), num_prompt=24, seq_len=24,
    )
    _run_both_gather_paths(
        torch.tensor([100, 101, 102, 103, 104]), block_size,
        set(), num_prompt=16, seq_len=18,
    )
    _run_both_gather_paths(
        torch.tensor([100, 101, 102, 103, 104]), block_size,
        block_indices_from_positions([0], block_size), num_prompt=16, seq_len=17,
    )
    _run_both_gather_paths(
        torch.tensor([100, 101]), block_size,
        block_indices_from_positions([0, 4], block_size), num_prompt=8, seq_len=8,
    )
    _run_both_gather_paths(
        torch.tensor([100, 101]), block_size,
        block_indices_from_positions([100], block_size), num_prompt=8, seq_len=8,
    )
    _run_both_gather_paths(
        torch.tensor([100, 101, 102, 103]), block_size,
        block_indices_from_positions([12, 0, 4], block_size), num_prompt=16, seq_len=16,
    )


def test_gather_view_incremental_matches_full_recompute_across_decode_steps():
    """The property that actually matters for `sparse_target_runner.py`'s
    optimization: build `base_view` ONCE (like the runner's per-turn cache
    does) and reuse it across every decode step of a simulated turn,
    comparing against the old fresh-every-step computation at each step --
    including steps that straddle the boundary block (num_prompt not a
    clean multiple of block_size, so the last historical block is only
    partially occupied at first and fills up as decode progresses) and
    steps that hit the degenerate "selection covers everything" case."""
    random.seed(0)
    for trial in range(8):
        block_size = random.choice([4, 8, 16])
        num_prompt = random.randint(20, 200)
        max_tokens = 20
        # Headroom must cover every block seq_len can reach across the
        # whole simulated turn (num_prompt + max_tokens - 1), plus one for
        # the ceil-division boundary -- an earlier version of this test
        # under-allocated headroom and got an IndexError from the OLD
        # ground-truth function itself (not a new-vs-old mismatch).
        num_full_blocks = -(-(num_prompt + max_tokens) // block_size) + 1
        full_block_table_row = torch.arange(1000, 1000 + num_full_blocks, dtype=torch.int64)
        max_pos = num_prompt - 1
        num_selected = random.randint(0, max_pos) if max_pos > 0 else 0
        selected_positions = random.sample(range(max_pos), num_selected) if max_pos > 0 else []
        base_block_indices = block_indices_from_positions(selected_positions, block_size)

        base_view = None
        for step in range(max_tokens):
            # `num_prompt + 1 + step`, not `num_prompt + step` -- these are
            # REAL decode-step sequence lengths (`num_computed_tokens +
            # num_scheduled_tokens`), so the smallest one a decode step can
            # ever present is `num_prompt + 1`. See
            # `kv_cache_utils.compute_sparse_gather_view`'s `seq_len` Args
            # entry for the off-by-one this distinction encodes.
            seq_len = num_prompt + 1 + step
            base_view = _run_both_gather_paths(
                full_block_table_row, block_size, base_block_indices,
                num_prompt, seq_len, base_view=base_view,
            )


def test_compute_base_gather_view_caches_stable_rows_not_recomputed_per_step():
    """`base_view.stable_rows`/`stable_seq_len` must be the SAME object
    across steps (never rebuilt) -- this is the actual optimization: if a
    caller reused `base_view` the way `sparse_target_runner.py` does, the
    stable portion must not be re-gathered on every call to
    `compute_sparse_gather_view_incremental`."""
    block_size = 4
    # 8 blocks -- enough headroom for seq_len up to 26 (needs block
    # index up to (26-1)//4 == 6, i.e. 7 blocks; 8 leaves margin).
    full_block_table_row = torch.tensor([100, 101, 102, 103, 104, 105, 106, 107])
    base_block_indices = block_indices_from_positions([1, 14], block_size)  # blocks 0, 3
    base_view = compute_base_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        base_block_indices=base_block_indices,
        num_prompt=24,
    )
    stable_rows_id = id(base_view.stable_rows)
    for seq_len in [24, 25, 26]:
        compute_sparse_gather_view_incremental(
            base_view=base_view,
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            num_prompt=24,
            seq_len=seq_len,
        )
        assert id(base_view.stable_rows) == stable_rows_id


def _kernel_visible_positions(
    gathered_row, gathered_seq_len, full_block_table_row, block_size
):
    """Decode a `(gathered_block_table_row, gathered_seq_len)` pair back
    into the set of CONVERSATION positions a paged-attention kernel would
    actually read, by doing exactly what the kernel does: for each logical
    index `i` in `[0, gathered_seq_len)`, look up physical block
    `gathered_row[i // block_size]` and slot `i % block_size`, then map
    that physical block id back to the logical block it originally
    occupied in `full_block_table_row`.

    This is the piece the whole sparse mechanism's correctness actually
    rests on, and the piece no existing check covered: `diagnose_target_
    gather_metadata.py` compares the gather's BLOCK COUNT against an
    independent recomputation, which is blind to a wrong `seq_lens` --
    and a wrong `seq_lens` was exactly the bug (see
    `test_gather_view_exposes_the_token_being_decoded`).

    Requires `full_block_table_row` to hold distinct physical block ids,
    which every caller in this file arranges.
    """
    physical_to_logical = {
        int(block_id): logical
        for logical, block_id in enumerate(full_block_table_row.tolist())
    }
    assert len(physical_to_logical) == len(full_block_table_row), (
        "helper requires distinct physical block ids in full_block_table_row"
    )
    visible = []
    for i in range(gathered_seq_len):
        physical = int(gathered_row[i // block_size])
        visible.append(physical_to_logical[physical] * block_size + (i % block_size))
    return visible


def test_kernel_visible_positions_helper_is_faithful_on_an_ungathered_row():
    """Sanity-check the helper itself before trusting it below: handed a
    row that gathers NOTHING (identity block table, full seq_len), it must
    reproduce `range(seq_len)` exactly -- i.e. it models a plain, dense
    paged-attention read correctly."""
    block_size = 4
    full_block_table_row = torch.tensor([100, 101, 102, 103])
    visible = _kernel_visible_positions(full_block_table_row, 14, full_block_table_row, block_size)
    assert visible == list(range(14))


def test_gather_view_exposes_the_token_being_decoded():
    """**Regression test for the off-by-one that caused the repetition
    loops.** `sparse_target_runner.py` used to pass
    `num_computed_tokens_cpu_tensor[req_idx]` as the gather's length, but
    that counter excludes the step's own scheduled token -- the stock
    runner builds `seq_lens = num_computed_tokens + num_scheduled_tokens`.
    Every gathered view was therefore one token short, so the token
    currently being decoded could never attend to its OWN key/value, on
    every decode step of every turn at every keep rate below the dense
    baseline.

    The invariant, stated in kernel terms: the highest conversation
    position the kernel can reach must be `seq_len - 1` -- the token being
    decoded right now."""
    block_size = 16
    num_prompt = 160  # 10 clean blocks of prompt
    full_block_table_row = torch.arange(1000, 1000 + 32, dtype=torch.int64)
    # Drop block 3 so the gather is non-degenerate (a full selection
    # returns None and patches nothing, which is why keep=1.0-shaped
    # cases never exercised this arithmetic at all).
    base_block_indices = block_indices_from_positions(
        [p for p in range(num_prompt) if p // block_size != 3], block_size
    )
    base_view = compute_base_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        base_block_indices=base_block_indices,
        num_prompt=num_prompt,
    )
    for step in range(6):
        seq_len = num_prompt + 1 + step  # a real decode step's seq_lens value
        result = compute_sparse_gather_view_incremental(
            base_view=base_view,
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            num_prompt=num_prompt,
            seq_len=seq_len,
        )
        assert result is not None, f"unexpected degenerate result at seq_len={seq_len}"
        gathered_row, gathered_seq_len = result
        visible = _kernel_visible_positions(
            gathered_row, gathered_seq_len, full_block_table_row, block_size
        )
        assert seq_len - 1 in visible, (
            f"the token being decoded (position {seq_len - 1}) is not visible "
            f"to the kernel at seq_len={seq_len}; highest visible position is "
            f"{max(visible)} -- this is the off-by-one regression"
        )
        assert max(visible) == seq_len - 1, (
            f"kernel can reach position {max(visible)} at seq_len={seq_len}, "
            f"past the token being decoded ({seq_len - 1})"
        )


def test_gather_view_opens_a_fresh_block_for_the_decoding_token():
    """The sharper half of the same off-by-one: when the decoding token is
    the first occupant of a brand-new block (`seq_len % block_size == 1`),
    the old code didn't merely under-count `seq_lens` -- the block
    physically holding that token wasn't in the gathered block table at
    all, because the force-keep tail was derived from the short length.

    `num_prompt` is a clean multiple of `block_size` here, so the very
    first decode step is exactly that case."""
    block_size = 8
    num_prompt = 64  # 8 clean blocks -> position 64 opens block 8
    full_block_table_row = torch.arange(1000, 1000 + 16, dtype=torch.int64)
    base_block_indices = block_indices_from_positions(
        [p for p in range(num_prompt) if p // block_size != 5], block_size
    )
    base_view = compute_base_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        base_block_indices=base_block_indices,
        num_prompt=num_prompt,
    )
    result = compute_sparse_gather_view_incremental(
        base_view=base_view,
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        num_prompt=num_prompt,
        seq_len=num_prompt + 1,
    )
    assert result is not None
    gathered_row, gathered_seq_len = result
    # Block 8 (physical id 1008) must be gathered, holding exactly 1 token.
    assert 1008 in gathered_row.tolist(), (
        f"block holding the decoding token was not gathered: {gathered_row.tolist()}"
    )
    visible = _kernel_visible_positions(
        gathered_row, gathered_seq_len, full_block_table_row, block_size
    )
    assert visible[-1] == num_prompt, (
        f"expected position {num_prompt} last, got {visible[-1]}"
    )
    assert visible.count(num_prompt) == 1


def test_gather_view_exposes_only_selected_or_this_turns_own_positions():
    """The complement of the two tests above: widening the view must not
    have made it expose blocks nobody asked for. Everything the kernel can
    read has to come from either the speculator's own selection or this
    turn's force-kept tail `[num_prompt, seq_len)`."""
    block_size = 8
    num_prompt = 60  # deliberately NOT a multiple of block_size
    full_block_table_row = torch.arange(1000, 1000 + 16, dtype=torch.int64)
    selected_positions = [1, 2, 30, 31, 59]
    base_block_indices = block_indices_from_positions(selected_positions, block_size)
    base_view = compute_base_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        base_block_indices=base_block_indices,
        num_prompt=num_prompt,
    )
    for step in range(6):
        seq_len = num_prompt + 1 + step
        result = compute_sparse_gather_view_incremental(
            base_view=base_view,
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            num_prompt=num_prompt,
            seq_len=seq_len,
        )
        assert result is not None
        gathered_row, gathered_seq_len = result
        visible = set(
            _kernel_visible_positions(
                gathered_row, gathered_seq_len, full_block_table_row, block_size
            )
        )
        allowed_blocks = set(base_block_indices) | {
            p // block_size for p in range(num_prompt, seq_len)
        }
        for pos in visible:
            assert pos // block_size in allowed_blocks, (
                f"position {pos} (block {pos // block_size}) is visible at "
                f"seq_len={seq_len} but belongs to no selected or force-kept block"
            )
            assert pos < seq_len, (
                f"position {pos} is visible at seq_len={seq_len} but hasn't "
                f"been written yet -- the kernel would read an uninitialized slot"
            )


def _load_sparse_target_runner():
    """Import `vllm_patch.sparse_target_runner` even though vLLM isn't
    installed, by standing in minimal base classes for the two symbols it
    imports at module level (`GPUModelRunner`, `Worker`). Only stubs when
    the real package is genuinely absent, so this never shadows a real
    install.

    **What this does and does NOT prove.** The stub base classes are empty,
    and `SparseTargetGPUModelRunner` never calls up into them from the
    method under test -- `_apply_sparse_attention_overrides` reads only
    attributes the test itself supplies. So these tests exercise this
    module's OWN logic (which length it reads, what it writes into each
    layer's metadata) with no pretence of validating the surrounding vLLM
    contract; that the fields exist and mean what this module assumes is
    still `validate_sparse_attention.py`'s job on real hardware, per the
    module's own "Known risk areas". The `_build_attention_metadata`
    override is deliberately NOT exercised here, since its whole body is
    the `super()` call the stub can't honestly stand in for."""
    import importlib
    import types

    try:
        importlib.import_module("vllm.v1.worker.gpu_model_runner")
    except ImportError:
        stubs = {
            "vllm": {},
            "vllm.v1": {},
            "vllm.v1.worker": {},
            "vllm.v1.worker.gpu_model_runner": {
                "GPUModelRunner": type("GPUModelRunner", (), {})
            },
            "vllm.v1.worker.gpu_worker": {"Worker": type("Worker", (), {})},
        }
        for name, attrs in stubs.items():
            module = sys.modules.setdefault(name, types.ModuleType(name))
            for attr, value in attrs.items():
                setattr(module, attr, value)
            if "." in name:
                parent, _, child = name.rpartition(".")
                setattr(sys.modules[parent], child, module)
    return importlib.import_module("vllm_patch.sparse_target_runner")


class _FakeLayerMetadata:
    """Stands in for one backend's per-layer `AttentionMetadata`, carrying
    exactly the four fields `sparse_target_runner.py` patches."""

    def __init__(self, block_table, seq_lens, max_seq_len, scheduler_metadata="AOT-SCHEDULE"):
        self.block_table = block_table
        self.seq_lens = seq_lens
        self.max_seq_len = max_seq_len
        self.scheduler_metadata = scheduler_metadata


class _FakeInputBatch:
    def __init__(self, req_ids, num_computed_tokens, num_prompt_tokens):
        self.req_ids = list(req_ids)
        self.num_reqs = len(self.req_ids)
        self.num_computed_tokens_cpu_tensor = torch.tensor(
            num_computed_tokens, dtype=torch.int32
        )
        self.num_prompt_tokens_cpu_tensor = torch.tensor(
            num_prompt_tokens, dtype=torch.int32
        )


def _make_fake_runner(
    runner_module,
    block_size,
    req_ids,
    num_computed_tokens,
    num_prompt_tokens,
    step_seq_lens,
    block_table_width=32,
    num_layers=3,
):
    """Builds a `SparseTargetGPUModelRunner` with only the attributes
    `_apply_sparse_attention_overrides` reads, bypassing
    `GPUModelRunner.__init__` entirely (whose real signature this module
    deliberately never depends on -- see its `*args/**kwargs` passthrough).

    `step_seq_lens` is what a REAL `_prepare_inputs` would have put in
    `optimistic_seq_lens_cpu`: `num_computed_tokens + num_scheduled_tokens`.
    Passing it separately from `num_computed_tokens` is the entire point --
    the two differ by exactly the off-by-one these tests guard."""
    runner = object.__new__(runner_module.SparseTargetGPUModelRunner)
    runner.input_batch = _FakeInputBatch(req_ids, num_computed_tokens, num_prompt_tokens)
    runner.optimistic_seq_lens_cpu = torch.tensor(step_seq_lens, dtype=torch.int32)

    cache_config = type("CacheConfig", (), {"block_size": block_size})()
    runner.vllm_config = type("VllmConfig", (), {"cache_config": cache_config})()

    num_reqs = len(req_ids)
    # Distinct physical block ids per row so _kernel_visible_positions can
    # invert the gather unambiguously.
    block_table = torch.stack([
        torch.arange(1000 + r * block_table_width,
                     1000 + (r + 1) * block_table_width, dtype=torch.int64)
        for r in range(num_reqs)
    ])
    seq_lens = torch.tensor(step_seq_lens, dtype=torch.int32)
    stale_max_seq_len = int(seq_lens.max().item())
    attn_metadata = {
        f"layer.{i}": _FakeLayerMetadata(
            block_table=block_table.clone(),
            seq_lens=seq_lens.clone(),
            max_seq_len=stale_max_seq_len,
        )
        for i in range(num_layers)
    }
    return runner, attn_metadata


def test_apply_overrides_uses_the_step_seq_len_not_num_computed_tokens():
    """**The runner-level regression test for the off-by-one.** The bug
    was never in `kv_cache_utils`'s arithmetic -- it was in which number
    this method handed it. `num_computed_tokens_cpu_tensor[req_idx]` is
    the count BEFORE this step's scheduled token; the stock runner builds
    `seq_lens = num_computed_tokens + num_scheduled_tokens`.

    Setup: block_size=16, a 160-token prompt (10 clean blocks), the first
    decode step -- so `num_computed_tokens == 160` while the real
    `seq_lens` for the step is 161. The selection drops block 3.

    Correct: 9 stable blocks (0,1,2,4..9) + block 10 holding the decoding
    token at position 160 -> 10 blocks, `seq_lens = 9*16 + 1 = 145`.
    The old code, fed 160, produced 9 blocks and `seq_lens = 144` -- block
    10 absent entirely, the decoding token invisible to itself."""
    runner_module = _load_sparse_target_runner()
    block_size, num_prompt = 16, 160
    runner, attn_metadata = _make_fake_runner(
        runner_module, block_size,
        req_ids=["conv::sparse-session"],
        num_computed_tokens=[num_prompt],      # first decode step
        num_prompt_tokens=[num_prompt],
        step_seq_lens=[num_prompt + 1],        # what vLLM tells the kernel
    )
    selection = [p for p in range(num_prompt) if p // block_size != 3]

    sparse_selection_registry.clear()
    try:
        sparse_selection_registry.register("conv::sparse-session", selection)
        runner._apply_sparse_attention_overrides(attn_metadata)
    finally:
        sparse_selection_registry.clear()

    for layer_name, layer in attn_metadata.items():
        patched_seq_len = int(layer.seq_lens[0])
        assert patched_seq_len == 145, (
            f"{layer_name}: patched seq_lens={patched_seq_len}, expected 145. "
            f"144 means the old num_computed_tokens value is being used again"
        )
        gathered = layer.block_table[0, :10]
        assert gathered.tolist() == [1000, 1001, 1002, 1004, 1005, 1006, 1007,
                                     1008, 1009, 1010], (
            f"{layer_name}: unexpected gathered blocks {gathered.tolist()} -- "
            f"1010 (holding the decoding token at position 160) must be present"
        )
        # Everything past the gathered prefix is NULL-padded, not stale.
        assert layer.block_table[0, 10:].abs().sum().item() == 0


def test_apply_overrides_result_lets_the_kernel_reach_the_decoding_token():
    """Same setup as above, stated in the terms that actually matter: feed
    the PATCHED metadata through the same paged-read model the kernel uses
    and check the decoding token is reachable. Guards the wiring end to
    end (which length is read, what is written, how it is padded) rather
    than the individual field values."""
    runner_module = _load_sparse_target_runner()
    block_size, num_prompt = 16, 176  # 11 clean blocks
    runner, attn_metadata = _make_fake_runner(
        runner_module, block_size,
        req_ids=["r0"],
        num_computed_tokens=[num_prompt + 3],
        num_prompt_tokens=[num_prompt],
        step_seq_lens=[num_prompt + 4],
    )
    original_row = attn_metadata["layer.0"].block_table[0].clone()
    selection = [p for p in range(num_prompt) if p // block_size not in (2, 7)]

    sparse_selection_registry.clear()
    try:
        sparse_selection_registry.register("r0", selection)
        runner._apply_sparse_attention_overrides(attn_metadata)
    finally:
        sparse_selection_registry.clear()

    layer = attn_metadata["layer.0"]
    patched_seq_len = int(layer.seq_lens[0])
    visible = _kernel_visible_positions(
        layer.block_table[0], patched_seq_len, original_row, block_size
    )
    assert max(visible) == num_prompt + 3, (
        f"kernel's highest reachable position is {max(visible)}, expected "
        f"{num_prompt + 3} (the token being decoded)"
    )
    # The dropped blocks really are dropped -- the fix widened the view by
    # exactly one token, it didn't quietly disable the gather.
    assert not any(pos // block_size in (2, 7) for pos in visible)


def test_apply_overrides_leaves_prefill_steps_untouched():
    """`num_computed_tokens < num_prompt_tokens` means this turn's own
    query is still being prefilled -- deliberately full attention. The
    guard must key on `num_computed_tokens`, NOT on the step seq_len,
    which is larger and would misclassify the last prefill chunk as a
    decode step."""
    runner_module = _load_sparse_target_runner()
    block_size, num_prompt = 16, 160
    runner, attn_metadata = _make_fake_runner(
        runner_module, block_size,
        req_ids=["r0"],
        num_computed_tokens=[96],        # mid-prefill
        num_prompt_tokens=[num_prompt],
        step_seq_lens=[160],             # this chunk finishes the prompt
    )
    before = {name: (layer.block_table.clone(), layer.seq_lens.clone(),
                     layer.max_seq_len, layer.scheduler_metadata)
              for name, layer in attn_metadata.items()}

    sparse_selection_registry.clear()
    try:
        sparse_selection_registry.register("r0", list(range(0, 96, 2)))
        runner._apply_sparse_attention_overrides(attn_metadata)
    finally:
        sparse_selection_registry.clear()

    for name, layer in attn_metadata.items():
        old_bt, old_sl, old_max, old_sched = before[name]
        assert torch.equal(layer.block_table, old_bt)
        assert torch.equal(layer.seq_lens, old_sl)
        assert layer.max_seq_len == old_max
        assert layer.scheduler_metadata == old_sched


def test_apply_overrides_leaves_unregistered_requests_untouched():
    """No registered selection (e.g. a different pipeline's request, or a
    turn whose selection was already discarded) means no patching at all."""
    runner_module = _load_sparse_target_runner()
    runner, attn_metadata = _make_fake_runner(
        runner_module, 16,
        req_ids=["not-registered"],
        num_computed_tokens=[160],
        num_prompt_tokens=[160],
        step_seq_lens=[161],
    )
    before = attn_metadata["layer.0"].seq_lens.clone()

    sparse_selection_registry.clear()
    try:
        runner._apply_sparse_attention_overrides(attn_metadata)
    finally:
        sparse_selection_registry.clear()

    assert torch.equal(attn_metadata["layer.0"].seq_lens, before)
    assert attn_metadata["layer.0"].scheduler_metadata == "AOT-SCHEDULE"


def test_apply_overrides_fixes_max_seq_len_and_nulls_scheduler_metadata():
    """Covers the two previously-confirmed bugs alongside the new one, so a
    later refactor can't quietly drop either: `max_seq_len` must be
    recomputed from the POST-gather seq_lens (it sizes the FA kernel's K/V
    tiling), and FA3's ahead-of-time `scheduler_metadata` must be nulled
    on every layer (it encodes a work schedule built for the original,
    unrestricted context length)."""
    runner_module = _load_sparse_target_runner()
    block_size, num_prompt = 16, 160
    runner, attn_metadata = _make_fake_runner(
        runner_module, block_size,
        req_ids=["r0"],
        num_computed_tokens=[num_prompt],
        num_prompt_tokens=[num_prompt],
        step_seq_lens=[num_prompt + 1],
    )
    assert attn_metadata["layer.0"].max_seq_len == num_prompt + 1  # stale, pre-gather

    sparse_selection_registry.clear()
    try:
        sparse_selection_registry.register(
            "r0", [p for p in range(num_prompt) if p // block_size != 3]
        )
        runner._apply_sparse_attention_overrides(attn_metadata)
    finally:
        sparse_selection_registry.clear()

    for name, layer in attn_metadata.items():
        assert layer.max_seq_len == 145, (
            f"{name}: max_seq_len={layer.max_seq_len}, expected the post-gather "
            f"145 -- a stale value keeps the kernel iterating the full span"
        )
        assert layer.scheduler_metadata is None, f"{name}: scheduler_metadata not nulled"


def test_step_seq_len_raises_a_clear_error_if_the_runner_field_is_missing():
    """`optimistic_seq_lens_cpu` is a vLLM-version-sensitive attribute. If
    it ever disappears, this must fail loudly and point at the right fix --
    the one thing that must NOT happen is a silent fallback to
    `num_computed_tokens_cpu_tensor`, which is the original bug."""
    runner_module = _load_sparse_target_runner()
    runner = object.__new__(runner_module.SparseTargetGPUModelRunner)
    try:
        runner._step_seq_len(0)
        raise AssertionError("expected AttributeError for missing optimistic_seq_lens_cpu")
    except AttributeError as exc:
        assert "optimistic_seq_lens_cpu" in str(exc)
        assert "num_computed_tokens_cpu_tensor" in str(exc)


def test_apply_overrides_per_turn_cache_invalidates_on_a_new_selection():
    """The per-turn caches are keyed on the registry's generation counter.
    Registering a DIFFERENT selection for the same request id (what the
    next turn does) must produce a different gather, not a stale cache
    hit -- the failure mode that made SPARSE output non-deterministic
    before `id(selected_positions)` was replaced."""
    runner_module = _load_sparse_target_runner()
    block_size, num_prompt = 16, 160

    runner, attn_metadata = _make_fake_runner(
        runner_module, block_size,
        req_ids=["same-id"],
        num_computed_tokens=[num_prompt],
        num_prompt_tokens=[num_prompt],
        step_seq_lens=[num_prompt + 1],
    )
    sparse_selection_registry.clear()
    try:
        sparse_selection_registry.register(
            "same-id", [p for p in range(num_prompt) if p // block_size != 3]
        )
        runner._apply_sparse_attention_overrides(attn_metadata)
        first = attn_metadata["layer.0"].block_table[0, :10].tolist()

        # Same request id, same length of selection, different blocks --
        # only the generation counter distinguishes them.
        _, attn_metadata_2 = _make_fake_runner(
            runner_module, block_size,
            req_ids=["same-id"],
            num_computed_tokens=[num_prompt],
            num_prompt_tokens=[num_prompt],
            step_seq_lens=[num_prompt + 1],
        )
        sparse_selection_registry.register(
            "same-id", [p for p in range(num_prompt) if p // block_size != 6]
        )
        runner._apply_sparse_attention_overrides(attn_metadata_2)
        second = attn_metadata_2["layer.0"].block_table[0, :10].tolist()
    finally:
        sparse_selection_registry.clear()

    assert first != second, (
        f"the second turn's gather reused the first turn's cached blocks "
        f"({first}) -- the generation-counter invalidation is broken"
    )
    assert 1003 not in first and 1003 in second
    assert 1006 in first and 1006 not in second


def test_sparse_selection_registry_lifecycle():
    sparse_selection_registry.clear()
    try:
        assert sparse_selection_registry.get("req-1") is None

        sparse_selection_registry.register("req-1", [0, 4, 8])
        assert sparse_selection_registry.get("req-1") == [0, 4, 8]

        # re-registering overwrites, doesn't accumulate (same convention as
        # pruning_registry.py)
        sparse_selection_registry.register("req-1", [12])
        assert sparse_selection_registry.get("req-1") == [12]

        sparse_selection_registry.discard("req-1")
        assert sparse_selection_registry.get("req-1") is None
    finally:
        sparse_selection_registry.clear()


def test_sparse_selection_registry_generation_counter():
    """Regression test for the real bug `get_with_generation()` replaced:
    the runner's per-turn caches used to key on `id(selected_positions)`,
    which is only unique while the object is alive -- once a turn's
    selection list is freed and a later allocation happens to reuse its
    address, a stale cache entry could silently match a different turn's
    selection (confirmed as the cause of non-deterministic SPARSE output
    on real hardware, see sparse_selection_registry.py's "Generation
    counter" section). Deliberately does NOT try to force an actual id()
    collision here (that depends on CPython allocator internals, not this
    module's own logic, and would make the test flaky/environment-
    dependent) -- what's actually guaranteed, and what this asserts, is
    that the generation strictly changes across registrations regardless
    of what id() happens to do, which is the property the old scheme
    lacked and this one provides unconditionally."""
    sparse_selection_registry.clear()
    try:
        assert sparse_selection_registry.get_with_generation("req-1") is None

        sparse_selection_registry.register("req-1", [0, 4, 8])
        gen1, positions1 = sparse_selection_registry.get_with_generation("req-1")
        assert positions1 == [0, 4, 8]

        sparse_selection_registry.register("req-1", [12, 16])
        gen2, positions2 = sparse_selection_registry.get_with_generation("req-1")
        assert positions2 == [12, 16]
        assert gen2 != gen1, "generation must change across registrations"

        # A second request_id's own generation sequence is independent --
        # the counter is global/monotonic, not per-request_id, but that
        # must never cause two DIFFERENT requests' current generations to
        # collide either.
        sparse_selection_registry.register("req-2", [1, 2])
        gen_req2, _ = sparse_selection_registry.get_with_generation("req-2")
        assert gen_req2 not in (gen1, gen2)

        sparse_selection_registry.discard("req-1")
        assert sparse_selection_registry.get_with_generation("req-1") is None
        sparse_selection_registry.discard("req-2")
    finally:
        sparse_selection_registry.clear()


def test_ledger_to_target_position_map_multi_turn_trace():
    """Hand-computed trace matching a real multi-turn shape: chat_before (3
    tokens) + context (10 tokens, ledger 0-9) + turn1 query (2 tokens,
    ledger 10-11) + chat_after (4 tokens, target-only) + turn1 output (3
    tokens, ledger 12-14) + turn_boundary (5 tokens, target-only) + turn2
    query (2 tokens, ledger 15-16). Verifies both that positions translate
    correctly as wrappers accumulate, AND that earlier mappings don't
    retroactively change once later wrappers are recorded (the ledger is
    append-only, so a position's mapping, once past, must stay fixed)."""
    m = LedgerToTargetPositionMap(initial_offset=3)  # len(chat_before_ids)

    # context + turn1 query: ledger [0,12) -> target [3,15), no wrapper yet
    assert m.translate(0) == 3
    assert m.translate(9) == 12  # last context token
    assert m.translate(11) == 14  # last query token

    m.add_wrapper(ledger_position=12, wrapper_len=4)  # chat_after inserted
    # turn1 output: ledger [12,15) -> target [19,22)
    assert m.translate(12) == 19
    assert m.translate(14) == 21

    m.add_wrapper(ledger_position=15, wrapper_len=5)  # turn_boundary inserted
    # turn2 query: ledger [15,17) -> target [27,29)
    assert m.translate(15) == 27
    assert m.translate(16) == 28

    # Earlier mappings unaffected by later wrapper insertions.
    assert m.translate(0) == 3
    assert m.translate(12) == 19


def test_ledger_to_target_position_map_no_wrappers_added_yet():
    """Before any add_wrapper call, every position uses only the initial
    offset -- the single-turn (turn 1 only) case."""
    m = LedgerToTargetPositionMap(initial_offset=5)
    assert m.translate(0) == 5
    assert m.translate(100) == 105


def _simulate_baseline_stream(context_ids, per_turn_query_ids, per_turn_output_ids,
                              chat_before_ids, chat_after_ids, turn_boundary_ids):
    """Reproduces `run_baseline`'s accumulation: it resubmits the whole
    accumulated prompt as a fresh one-shot request each turn, appending
    this turn's own output (minus its last token) before the next turn's
    delta. Returns the list of prompts actually submitted, one per turn."""
    chat_ids: list = []
    submitted = []
    for turn_idx, query_ids in enumerate(per_turn_query_ids):
        chat_ids = chat_ids + build_turn_delta_ids(
            turn_idx=turn_idx, query_ids=query_ids,
            chat_before_ids=chat_before_ids, context_ids=context_ids,
            chat_after_ids=chat_after_ids, turn_boundary_ids=turn_boundary_ids,
        )
        submitted.append(list(chat_ids))
        chat_ids = chat_ids + per_turn_output_ids[turn_idx][:-1]
    return submitted


def _simulate_sparse_session_stream(context_ids, per_turn_query_ids, per_turn_output_ids,
                                    chat_before_ids, chat_after_ids, turn_boundary_ids):
    """Reproduces what `run_sparse_attention`'s persistent target session
    physically holds: it submits only each turn's delta, and vLLM's
    `_update_request_as_session` folds the turn's COMPUTED output tokens
    (all but the final sampled one, whose KV was never computed) into the
    session prompt before the next delta arrives. Returns the resident
    token stream as of each turn's generation start."""
    resident: list = []
    per_turn = []
    for turn_idx, query_ids in enumerate(per_turn_query_ids):
        resident = resident + build_turn_delta_ids(
            turn_idx=turn_idx, query_ids=query_ids,
            chat_before_ids=chat_before_ids, context_ids=context_ids,
            chat_after_ids=chat_after_ids, turn_boundary_ids=turn_boundary_ids,
        )
        per_turn.append(list(resident))
        resident = resident + per_turn_output_ids[turn_idx][:-1]
    return per_turn


def test_baseline_and_sparse_build_the_same_chat_stream():
    """**The property the M000 rendering change exists to establish.**
    M000 and SPARSE reach their prompts by different mechanics -- baseline
    resubmits the whole accumulated prompt as a one-shot request per turn,
    the sparse path submits only a delta into a resumable session -- but
    for the comparison to isolate the attention gather, the tokens the
    model sees must be identical at every turn.

    Before this change M000 flattened the whole conversation into a single
    user message while SPARSE used real `<|eot_id|>` turns, which made
    every turn index except 0 incomparable (M000 decayed 70/74/61/59/51 on
    `scbench_kv` while SPARSE stayed flat and high, i.e. sparse attention
    appeared to beat dense attention -- see `run_baseline`'s docstring)."""
    chat_before_ids = [1, 2, 3]
    chat_after_ids = [80, 81]
    turn_boundary_ids = [90, 91, 92]
    context_ids = [10, 11, 12, 13, 14]
    per_turn_query_ids = [[20, 21], [30, 31, 32], [40]]
    # Last element of each is the "sampled but never computed" token both
    # paths drop; the second turn ends on a length cap rather than EOS.
    per_turn_output_ids = [[50, 51, 999], [60, 61, 62, 999], [70, 999]]

    baseline = _simulate_baseline_stream(
        context_ids, per_turn_query_ids, per_turn_output_ids,
        chat_before_ids, chat_after_ids, turn_boundary_ids)
    sparse = _simulate_sparse_session_stream(
        context_ids, per_turn_query_ids, per_turn_output_ids,
        chat_before_ids, chat_after_ids, turn_boundary_ids)

    assert baseline == sparse, (
        f"M000 and SPARSE diverge.\n  turn-by-turn baseline: {baseline}\n"
        f"  turn-by-turn sparse:   {sparse}"
    )
    # Spelled out for turn 0, the one turn that was already comparable
    # before the change -- it must stay comparable after it.
    assert baseline[0] == [1, 2, 3] + context_ids + [20, 21] + [80, 81]
    # And turn 1 now closes turn 0's assistant message with a real boundary
    # rather than continuing one flattened user message.
    assert baseline[1] == baseline[0] + [50, 51] + turn_boundary_ids + [30, 31, 32] + [80, 81]


def test_turn_delta_drops_the_models_own_end_of_turn_token_exactly_once():
    """The `[:-1]` on each turn's output is what keeps the stream
    well-formed: when a turn stops on the model's own `<|eot_id|>`, that
    token is the one dropped and `turn_boundary_ids` supplies its own
    immediately after. Keeping both would emit a DOUBLE end-of-turn marker
    every turn."""
    eot = 128009  # stand-in for <|eot_id|>, the first token of a boundary
    chat_before_ids, chat_after_ids, turn_boundary_ids = [1], [80], [eot, 91]
    stream = _simulate_baseline_stream(
        context_ids=[10],
        per_turn_query_ids=[[20], [30]],
        per_turn_output_ids=[[50, eot], [60, eot]],
        chat_before_ids=chat_before_ids, chat_after_ids=chat_after_ids,
        turn_boundary_ids=turn_boundary_ids,
    )
    turn1 = stream[1]
    # ... 50, then exactly ONE eot (the boundary's), then 91.
    assert turn1 == [1, 10, 20, 80, 50, eot, 91, 30, 80], turn1
    assert turn1.count(eot) == 1


def test_build_turn_delta_ids_ignores_context_after_turn_zero():
    """Turn 0 opens the conversation; later turns must not re-emit
    `chat_before`/`context`, which are already resident. Passing them
    unconditionally keeps the call sites branch-free."""
    later = build_turn_delta_ids(
        turn_idx=3, query_ids=[7], chat_before_ids=[1, 2], context_ids=[3, 4],
        chat_after_ids=[8], turn_boundary_ids=[9],
    )
    assert later == [9, 7, 8]
    assert 1 not in later and 3 not in later


def test_build_turn_delta_ids_does_not_alias_its_inputs():
    """Both callers accumulate with `chat_ids = chat_ids + delta_ids`, so a
    delta that aliased a caller's list would be corrupted by a later
    append somewhere else."""
    chat_before_ids = [1, 2]
    delta = build_turn_delta_ids(
        turn_idx=0, query_ids=[7], chat_before_ids=chat_before_ids,
        context_ids=[3], chat_after_ids=[8], turn_boundary_ids=[9],
    )
    delta[0] = 999
    assert chat_before_ids == [1, 2]


def test_wrapper_target_spans_cover_the_attention_sink():
    """**Regression test for the second cause of the repetition loops.**
    `translate(p) = p + offset` with `offset >= initial_offset`, so no
    ledger position can EVER map below `len(chat_before_ids)` -- meaning
    target positions `[0, len(chat_before_ids))`, which hold
    `<|begin_of_text|>` and the system header, were unreachable by
    `run_sparse_attention`'s registered selection at every keep rate, and
    so were dropped from decode attention whenever the gather fired.
    Dropping the attention sink is the textbook cause of exactly the
    observed symptom (fluent start, then a repetition loop).

    `wrapper_target_positions()` must therefore always contain target
    position 0."""
    m = LedgerToTargetPositionMap(initial_offset=3)
    assert m.wrapper_target_spans() == [(0, 3)]
    assert m.wrapper_target_positions() == [0, 1, 2]
    assert 0 in m.wrapper_target_positions()

    m.add_wrapper(ledger_position=12, wrapper_len=4)
    m.add_wrapper(ledger_position=15, wrapper_len=5)
    assert 0 in m.wrapper_target_positions(), (
        "the attention sink dropped out of the wrapper set once later "
        "wrappers were recorded"
    )


def test_wrapper_target_spans_match_the_multi_turn_trace():
    """Same hand-computed shape as
    `test_ledger_to_target_position_map_multi_turn_trace`, checked from the
    wrapper side: chat_before occupies target [0,3); chat_after (4 tokens,
    inserted at ledger 12, which sits at target 19 afterwards) occupies
    [15,19); turn_boundary (5 tokens, inserted at ledger 15, at target 27
    afterwards) occupies [22,27)."""
    m = LedgerToTargetPositionMap(initial_offset=3)
    m.add_wrapper(ledger_position=12, wrapper_len=4)
    m.add_wrapper(ledger_position=15, wrapper_len=5)
    assert m.wrapper_target_spans() == [(0, 3), (15, 19), (22, 27)]
    # Each span is exactly wrapper_len wide.
    assert [end - start for start, end in m.wrapper_target_spans()] == [3, 4, 5]


def test_wrapper_target_positions_are_exactly_the_untranslatable_ones():
    """The invariant that makes this correct rather than merely plausible:
    for any target stream, every position is EITHER the image of some
    ledger position under `translate` OR a wrapper position -- never both,
    never neither. Registering the union of the two therefore covers the
    whole resident stream with nothing double-counted or missed.

    Checked exhaustively over a simulated multi-turn stream."""
    initial_offset = 3
    ledger_len_at_end = 17
    m = LedgerToTargetPositionMap(initial_offset=initial_offset)
    m.add_wrapper(ledger_position=12, wrapper_len=4)
    m.add_wrapper(ledger_position=15, wrapper_len=5)

    translated = {m.translate(p) for p in range(ledger_len_at_end)}
    wrappers = set(m.wrapper_target_positions())
    total_len = ledger_len_at_end + initial_offset + 4 + 5

    assert not (translated & wrappers), (
        f"overlap between translated and wrapper positions: "
        f"{sorted(translated & wrappers)}"
    )
    assert translated | wrappers == set(range(total_len)), (
        f"gap in coverage: {sorted(set(range(total_len)) - (translated | wrappers))}"
    )


def test_wrapper_target_spans_ignore_zero_length_wrappers():
    """A tokenizer whose chat template contributes no `chat_after`/
    `turn_boundary` tokens must not produce empty spans (which would put
    meaningless entries in the registered selection)."""
    m = LedgerToTargetPositionMap(initial_offset=0)
    m.add_wrapper(ledger_position=10, wrapper_len=0)
    assert m.wrapper_target_spans() == []
    assert m.wrapper_target_positions() == []

    m2 = LedgerToTargetPositionMap(initial_offset=2)
    m2.add_wrapper(ledger_position=10, wrapper_len=0)
    m2.add_wrapper(ledger_position=10, wrapper_len=3)
    assert m2.wrapper_target_spans() == [(0, 2), (12, 15)]


def test_registered_selection_covers_the_wrapper_scaffolding():
    """End-to-end shape of what `run_sparse_attention` now registers: the
    union of translated ledger positions and every wrapper span. Mirrors
    the driver's own ordering -- this turn's `chat_after` is recorded
    BEFORE the union is built (those tokens are part of the delta being
    submitted, so they're resident during this turn's decode), and
    recording it does not move any translated position, since every kept
    position is `< result.orig_len`.

    Checks the properties the fix exists for: the sink is in, this turn's
    generation header is in, and the speculator's own choices survive."""
    len_chat_before, len_chat_after, len_turn_boundary = 3, 4, 5
    m = LedgerToTargetPositionMap(initial_offset=len_chat_before)

    # --- turn 0: context (ledger 0-9) + query (ledger 10-11) ---
    orig_len = 12
    kept_positions = [0, 3, 10, 11]  # sparse history + force-kept query
    m.add_wrapper(ledger_position=orig_len, wrapper_len=len_chat_after)
    registered = sorted(
        set(m.translate(p) for p in kept_positions) | set(m.wrapper_target_positions())
    )
    assert 0 in registered, "attention sink missing from turn 0's selection"
    assert registered[:3] == [0, 1, 2], "chat_before span missing"
    # chat_after occupies target [15,19) -- this turn's assistant generation
    # header, resident for this turn's own decode.
    assert set(range(15, 19)) <= set(registered)
    # The speculator's own choices are untouched by the union.
    assert {m.translate(p) for p in kept_positions} <= set(registered)
    # Nothing invented past the end of what's resident.
    assert max(registered) < orig_len + len_chat_before + len_chat_after

    # --- turn 1: output (ledger 12-14) then query (ledger 15-16) ---
    orig_len_2 = 17
    m.add_wrapper(ledger_position=15, wrapper_len=len_turn_boundary)
    m.add_wrapper(ledger_position=orig_len_2, wrapper_len=len_chat_after)
    registered_2 = sorted(
        set(m.translate(p) for p in [0, 13, 15, 16])
        | set(m.wrapper_target_positions())
    )
    assert 0 in registered_2, "attention sink missing from turn 1's selection"
    # Both turns' scaffolding is present: chat_before, turn 0's chat_after,
    # the turn boundary, and turn 1's own chat_after.
    for span in m.wrapper_target_spans():
        assert set(range(*span)) <= set(registered_2), f"wrapper span {span} not registered"


def test_prospective_target_len_matches_real_submitted_token_count():
    """Regression test for a real crash: `run_sparse_attention`'s
    pre-submission length guard (`predict_scbench.py`) used to compute
    `prospective_target_len` from a single constant `wrapper_overhead`
    (one chat_before + one chat_after + one turn_boundary), but the REAL
    target session accumulates one chat_after AND one turn_boundary PER
    TURN -- so the old check increasingly undercounted true resident
    length as a conversation progressed, letting a real run overrun
    `--target-max-num-batched-tokens` deep into a conversation instead of
    being cleanly skipped (`ValueError: could not broadcast input array
    from shape (131084,) into shape (131072,)` inside vLLM's own
    `add_request`).

    This builds actual token-id lists (unique sentinel ints per piece) and
    concatenates them in EXACTLY the order `run_sparse_attention` submits
    them (chat_before + context + query0 + chat_after, then per later turn
    turn_boundary + query_i + chat_after, with each turn's own output
    tokens landing in between via `state.complete_turn`), then checks that
    the NEW `position_map`-based formula matches `len()` of the real
    concatenated stream at every turn boundary -- not just a hand-derived
    formula cross-check, but the same token-accounting the real pipeline
    does."""
    chat_before_ids = [-1, -2, -3]
    chat_after_ids = [-4, -5, -6, -7]
    turn_boundary_ids = [-8, -9, -10, -11, -12]
    context_ids = list(range(100, 110))  # 10 tokens

    # Per-turn (query_len, output_len) -- output_len is what gets appended
    # to the ledger via state.complete_turn AFTER that turn's generation,
    # mirroring a real multi-turn conversation's shape.
    turns = [(2, 3), (2, 3), (2, 0), (2, 5)]

    real_stream: list[int] = []
    # ConversationState.total_len == len(context_ids) immediately at
    # construction (context is appended to the ledger in __init__, not on
    # the first begin_turn -- see conversation_state.py) -- state_total_len
    # here mirrors that, NOT reset to 0.
    state_total_len = len(context_ids)

    position_map = LedgerToTargetPositionMap(initial_offset=len(chat_before_ids))
    real_stream.extend(chat_before_ids)

    for turn_idx, (query_len, output_len) in enumerate(turns):
        query_ids = [1000 + turn_idx * 100 + i for i in range(query_len)]

        # The check itself, exactly as run_sparse_attention now computes it
        # (BEFORE any add_wrapper call for this turn).
        prospective_target_len = (
            position_map.translate(state_total_len)
            + (len(turn_boundary_ids) if turn_idx > 0 else 0)
            + len(query_ids)
            + len(chat_after_ids)
        )

        query_start_ledger_pos = state_total_len
        if turn_idx > 0:
            position_map.add_wrapper(query_start_ledger_pos, len(turn_boundary_ids))
            real_stream.extend(turn_boundary_ids)
        elif turn_idx == 0:
            real_stream.extend(context_ids)

        real_stream.extend(query_ids)
        state_total_len += query_len
        position_map.add_wrapper(state_total_len, len(chat_after_ids))
        real_stream.extend(chat_after_ids)

        assert prospective_target_len == len(real_stream), (
            f"turn {turn_idx}: prospective={prospective_target_len} "
            f"real={len(real_stream)}"
        )

        # This turn's own output, appended to the ledger (and, in the real
        # pipeline, to the target's persistent KV cache) before the next
        # turn's check runs.
        output_ids = [2000 + turn_idx * 100 + i for i in range(output_len)]
        real_stream.extend(output_ids)
        state_total_len += output_len


def test_prune_record_validation():
    record = PruneRecord(kept_positions=[0, 3, 5], orig_len=8)
    assert record.num_kept == 3
    assert record.decode_offset == 5

    try:
        PruneRecord(kept_positions=[0, 1, 2, 3], orig_len=3)
        raise AssertionError("expected ValueError for kept_positions > orig_len")
    except ValueError:
        pass


def test_prune_record_positions_for_step_even_multi_chunk():
    record = PruneRecord(kept_positions=list(range(0, 20, 2)), orig_len=20)  # N=10
    assert record.positions_for_step(0, 5) == [0, 2, 4, 6, 8]
    assert record.positions_for_step(5, 5) == [10, 12, 14, 16, 18]
    assert record.positions_for_step(10, 1) is None


def test_pruning_registry_lifecycle():
    pruning_registry.clear()
    try:
        assert pruning_registry.get("req-1") is None

        pruning_registry.register("req-1", kept_positions=[0, 2, 4], orig_len=10)
        record = pruning_registry.get("req-1")
        assert record is not None
        assert record.kept_positions == [0, 2, 4]
        assert record.orig_len == 10

        pruning_registry.discard("req-1")
        assert pruning_registry.get("req-1") is None

        pruning_registry.register("req-a", kept_positions=[0], orig_len=1)
        pruning_registry.register("req-b", kept_positions=[0], orig_len=1)
        pruning_registry.discard_finished(["req-a", "req-nonexistent"])
        assert pruning_registry.get("req-a") is None
        assert pruning_registry.get("req-b") is not None
    finally:
        pruning_registry.clear()


# ---------------------------------------------------------------------------
# conversation_state.py -- new for the multi-turn port, no single-turn
# analog, pure Python (no vLLM/torch dependency in the module itself).
# ---------------------------------------------------------------------------


def test_conversation_state_ledger_grows_monotonically():
    context = [100, 101, 102]  # 3 context tokens
    state = ConversationState("conv-1", context, keep_mode="keep")
    assert state.total_len == 3

    candidate_pool, force_keep_query = state.begin_turn([200, 201])  # turn 1 query, 2 tokens
    assert candidate_pool == [(100, 0), (101, 1), (102, 2)]
    assert force_keep_query == [(200, 3), (201, 4)]
    assert state.query_span == (3, 5)
    assert state.total_len == 5  # query appended, answer not yet

    state.complete_turn(candidate_pool, [300])  # golden answer, 1 token
    assert state.total_len == 6
    assert state.turn_idx == 1

    candidate_pool_2, force_keep_query_2 = state.begin_turn([400])  # turn 2 query
    # KEEP mode: candidate pool is the ENTIRE ledger so far (context + turn1
    # query + turn1 answer), rescored fresh.
    assert candidate_pool_2 == [
        (100, 0), (101, 1), (102, 2), (200, 3), (201, 4), (300, 5),
    ]
    assert force_keep_query_2 == [(400, 6)]


def test_conversation_state_keep_mode_rescoring_can_change_kept_set():
    """KEEP mode: nothing prevents turn N+1 from keeping a DIFFERENT subset
    of history than turn N did -- the pool is always the full ledger."""
    state = ConversationState("conv-2", [10, 11, 12, 13], keep_mode="keep")
    pool_1, _ = state.begin_turn([90])
    # Simulate turn 1 keeping only positions 0 and 2.
    kept_1 = [pool_1[0], pool_1[2]]
    state.complete_turn(kept_1, [91])

    pool_2, _ = state.begin_turn([92])
    # Turn 2's candidate pool still contains EVERY original context token
    # (0,1,2,3) plus turn1's query+answer -- position 1 and 3, dropped by
    # turn 1's own scoring, are still eligible here.
    positions_available = [pos for _, pos in pool_2]
    assert positions_available == [0, 1, 2, 3, 4, 5]


def test_conversation_state_discard_mode_monotonic_extension():
    """DISCARD mode: turn N's candidate pool is exactly turn N-1's kept
    subset + turn N-1's own query -- verifying the "final prompt is always
    turn N-1's plus a suffix" property EXPERIMENT_PLAN.md relies on."""
    state = ConversationState("conv-3", [10, 11, 12, 13, 14], keep_mode="discard")
    pool_1, query_1 = state.begin_turn([90, 91])
    assert pool_1 == [(10, 0), (11, 1), (12, 2), (13, 3), (14, 4)]
    # Turn 1 keeps only 2 of the 5 context tokens.
    kept_1 = [pool_1[1], pool_1[3]]  # (11,1), (13,3)
    state.complete_turn(kept_1, [200])  # golden answer NOT retained in discard pool

    # Ledger so far: context(0-4) + turn1 query(5-6) + turn1 golden answer(7)
    # -- 8 tokens total, so turn 2's own query lands at position 8, not 7.
    pool_2, query_2 = state.begin_turn([92])
    # Turn 2's candidate pool = turn 1's kept history + turn 1's own query
    # (now ordinary, no longer force-kept) -- NOT turn 1's golden answer.
    assert pool_2 == [(11, 1), (13, 3), (90, 5), (91, 6)]
    assert query_2 == [(92, 8)]

    # Turn 2 keeps everything this time.
    state.complete_turn(pool_2, [201])
    # Ledger now: ...+ turn2 query(8) + turn2 golden answer(9) -- 10 tokens,
    # so turn 3's query lands at position 10.
    pool_3, query_3 = state.begin_turn([93])
    assert pool_3 == [(11, 1), (13, 3), (90, 5), (91, 6), (92, 8)]
    assert query_3 == [(93, 10)]

    # Final-prompt monotonic-extension property: turn 3's full candidate
    # sequence (pool_3 + query_3) is exactly turn 2's own
    # (pool_2 + query_2) plus turn 2's own query appended as a suffix, i.e.
    # every token turn 2 actually submitted is still present, in order, at
    # the front of turn 3's submission.
    turn2_full = [tid for tid, _ in pool_2] + [tid for tid, _ in query_2]
    turn3_full = [tid for tid, _ in pool_3] + [tid for tid, _ in query_3]
    assert turn3_full[: len(turn2_full)] == turn2_full


def test_conversation_state_begin_turn_requires_complete_turn_first():
    state = ConversationState("conv-4", [1, 2], keep_mode="keep")
    state.begin_turn([3])
    try:
        state.begin_turn([4])
        raise AssertionError("expected RuntimeError for double begin_turn")
    except RuntimeError:
        pass


def test_conversation_state_complete_turn_requires_pending_begin():
    state = ConversationState("conv-5", [1, 2], keep_mode="keep")
    try:
        state.complete_turn([], [])
        raise AssertionError("expected RuntimeError for complete_turn with no pending turn")
    except RuntimeError:
        pass


def test_conversation_state_query_span_requires_pending_begin():
    state = ConversationState("conv-6", [1], keep_mode="keep")
    try:
        _ = state.query_span
        raise AssertionError("expected RuntimeError for query_span with no pending turn")
    except RuntimeError:
        pass


def test_conversation_state_rejects_invalid_keep_mode():
    try:
        ConversationState("conv-7", [1], keep_mode="bogus")
        raise AssertionError("expected ValueError for invalid keep_mode")
    except ValueError:
        pass


def test_conversation_state_empty_golden_answer_is_noop():
    state = ConversationState("conv-8", [1, 2], keep_mode="keep")
    pool, query = state.begin_turn([3])
    before = state.total_len
    state.complete_turn(pool, [])  # empty golden answer -- must not error
    assert state.total_len == before


def test_spec_config_rejects_invalid_keep_mode():
    try:
        SpecConfig(keep_strategy="percentage", keep_mode="bogus")
        raise AssertionError("expected AssertionError for invalid keep_mode")
    except AssertionError:
        pass


def test_spec_config_keep_mode_defaults_to_keep():
    cfg = SpecConfig(keep_strategy="percentage")
    assert cfg.keep_mode == "keep"


def _run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = []
    for test_fn in tests:
        try:
            test_fn()
            print(f"PASS {test_fn.__name__}")
        except Exception:
            failures.append(test_fn.__name__)
            print(f"FAIL {test_fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("Failed:", failures)
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
