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

Run with: python3 test_vllm_patch.py
"""

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
    compute_sparse_gather_view,
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
)
from predict_scbench import LedgerToTargetPositionMap


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
    """block_size=4, 6 blocks resident (24 tokens computed, decode step at
    the very start so num_prompt == num_computed == 24 -- no in-progress
    decode tail to force-keep yet). Selecting positions in blocks 0 and 3
    should gather exactly those two blocks, in ascending order."""
    block_size = 4
    full_block_table_row = torch.tensor([100, 101, 102, 103, 104, 105])  # 6 blocks
    result = compute_sparse_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        base_block_indices=block_indices_from_positions([1, 14], block_size),  # blocks 0, 3
        num_prompt=24,
        num_computed=24,
    )
    assert result is not None
    gathered_row, gathered_seq_len = result
    assert torch.equal(gathered_row, torch.tensor([100, 103]))  # blocks 0 and 3
    assert gathered_seq_len == 2 * block_size  # both fully-occupied historical blocks


def test_compute_sparse_gather_view_force_keeps_in_progress_decode_tail():
    """Even with an empty selection, tokens generated so far THIS turn
    (num_prompt..num_computed) must always be force-included -- the model
    needs coherent access to what it just generated within the same turn,
    regardless of what the speculator chose."""
    block_size = 4
    full_block_table_row = torch.tensor([100, 101, 102, 103, 104])  # 5 blocks
    # num_prompt=16 (4 blocks fully resident before this turn's decode
    # started), num_computed=18 (2 tokens generated so far this turn, both
    # landing in block index 4 -- positions 16, 17).
    result = compute_sparse_gather_view(
        full_block_table_row=full_block_table_row,
        block_size=block_size,
        base_block_indices=set(),  # nothing selected by the speculator at all
        num_prompt=16,
        num_computed=18,
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
        num_computed=17,  # 1 token generated so far this turn -> block 4
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
        num_computed=8,
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
            num_computed=8,
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
        num_computed=16,
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
        num_computed=17,  # force-keep tail would add block 4 if it mutated the input
    )
    assert base_block_indices == original


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
