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
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from vllm_patch.config import SpecConfig
from vllm_patch.conversation_state import ConversationState
from vllm_patch.kv_cache_utils import (
    prompt_tail_subslice,
    _find_kv_split_dim,
    block_indices_from_positions,
    compute_base_gather_view,
    compute_prefill_base_view,
    compute_prefill_gather_view,
    compute_sparse_gather_view,
    compute_sparse_gather_view_incremental,
    gather_keys_for_slots,
    is_decode_query_slice,
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
    LayerGeometry,
    aggregate_attention_score,
    compute_attention_score,
    layer_geometry_from_attention_layers,
    score_and_select_indices,
    scoring_layer_indices,
)
from vllm_patch.pruner import _positions_from_kept_indices
from vllm_patch.model_truncation import keep_weight_for_layer_range
from predict_scbench import (
    CSV_FIELDS,
    EXPERIMENTS,
    LedgerToTargetPositionMap,
    _flop_summary_fields,
    _num_decode_steps,
    build_turn_delta_ids,
)
from flops_model import (
    FlopBreakdown,
    ModelFlopConfig,
    dense_decode_attended_lens,
    model_flop_config,
    speculator_turn_flops,
    target_decode_flops,
    target_prefill_flops,
    target_sparse_prefill_flops,
)


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


def test_capture_time_and_after_the_fact_decode_filters_agree():
    """`is_decode_query_slice` (applied in the capture hook, before a slice
    is retained) and `stack_decode_only_steps` (applied at end_capture,
    after) implement the SAME rule -- "a decode step is exactly 1 token".
    They have to: if the capture-time predicate were stricter, real decode
    steps would silently vanish from the scoring pass and every keep-rate
    row would be scored on fewer lookahead steps than it reported; if it
    were looser, the prefill queries it exists to drop would still be
    pinned.

    That second half is the 30.3GiB one. A captured slice is a view into
    the whole forward call's query tensor, so one retained prefill slice
    pins that tensor per layer until scoring runs -- 32 layers x 0.95GiB
    for Llama-3.1-8B over a 124k-token context, which OOM'd the first real
    ORACLE-k20 run at 78.00GiB allocated on a 79.25GiB card. The 1B
    speculator's equivalent 7.6GiB had been absorbed by the headroom at
    --speculator-gpu-memory-utilization 0.2 for the whole SPARSE sweep,
    which is why no earlier run ever surfaced it."""
    HD = 8
    # (start, end) pairs a real batch can hand the hook: a full prefill, a
    # chunked-prefill continuation, a decode step, and an empty slice for a
    # request scheduled with no tokens this step.
    cases = [(0, 124009), (0, 32768), (5, 6), (0, 1), (7, 7)]
    for start, end in cases:
        captured = is_decode_query_slice(start, end)
        # What end_capture would decide about the same slice, by shape.
        survives_stacking = (end - start) == 1
        assert captured == survives_stacking, (
            f"slice [{start}, {end}) -- capture-time filter says "
            f"{captured}, end_capture's shape filter says {survives_stacking}"
        )

    assert is_decode_query_slice(5, 6) is True
    assert is_decode_query_slice(0, 32768) is False
    # Empty slice is not a decode step (and must not be retained either).
    assert is_decode_query_slice(7, 7) is False

    # End to end: feed only what the hook would now retain, and confirm
    # end_capture still produces the same [1, num_decode_steps, H*D] stack
    # it produced back when it had to filter prefill entries out itself.
    prefill_then_decodes = [
        torch.randn(32768, HD), torch.randn(32768, HD),  # dropped at capture
        torch.randn(1, HD), torch.randn(1, HD), torch.randn(1, HD),
    ]
    retained = [
        t for t in prefill_then_decodes
        if is_decode_query_slice(0, t.shape[0])
    ]
    assert len(retained) == 3
    assert torch.equal(
        stack_decode_only_steps(retained, HD),
        stack_decode_only_steps(prefill_then_decodes, HD),
    )


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

    # One KV cache group holding every layer -- the uniform-attention case.
    # A real interleaved model has more than one, and the runner resolves the
    # block size from whichever group holds the GATHERED (full-attention)
    # layers; that resolution is tested separately.
    group = type("Group", (), {
        "kv_cache_spec": type("Spec", (), {"block_size": block_size})(),
        "layer_names": [f"layer.{i}" for i in range(num_layers)],
    })()
    runner.kv_cache_config = type(
        "KVCacheConfig", (), {"kv_cache_groups": [group]})()
    runner._gatherable_group_block_size_cache = block_size

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
    # Uniform attention: every layer is gatherable. Seeded rather than looked
    # up because the real lookup goes through vLLM's model registry, which
    # this CPU-only suite has no engine for. The SELECTION RULE itself is
    # tested separately and for real, against
    # `model_structure.gatherable_layer_names`.
    runner._gatherable_layer_names_cache = set(attn_metadata)
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
    query is still being prefilled. Under the DEFAULT (decode-only) scope
    that is deliberately full attention -- and the guard must key on
    `num_computed_tokens`, NOT on the step seq_len, which is larger and
    would misclassify the last prefill chunk as a decode step.

    The opt-in prefill scope is covered separately below; this test pins
    the default, which is what every published SPARSE row was measured
    under and must not change silently."""
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


# ---------------------------------------------------------------------------
# Sparse PREFILL (opt-in scope) -- kv_cache_utils level
# ---------------------------------------------------------------------------
#
# The decode gather only ever has one query token, so "which keys are
# visible" is the whole story. A prefill chunk has many, and the ORDER of
# the compacted view becomes load-bearing: FlashAttention aligns a causal
# mask bottom-right, so query `i` of `n_q` reads keys `[0, L - n_q + i]`.
# That is only the right answer if the chunk's own tokens are the final
# `n_q` entries of the gathered view, in order. These tests check that
# invariant directly, by decoding the gathered view back into real
# positions with `_kernel_visible_positions` -- not by comparing block
# counts, which is exactly the check that was blind to the seq_lens bug on
# the decode side.


def _prefill_gather(block_table_row, block_size, selection, turn_start, seq_len):
    """Build the per-turn view and run one chunk through it, the same two
    calls `sparse_target_runner.py` makes."""
    base_view = compute_prefill_base_view(
        full_block_table_row=block_table_row,
        block_size=block_size,
        base_block_indices=block_indices_from_positions(selection, block_size),
        turn_start=turn_start,
    )
    return compute_prefill_gather_view(
        base_view=base_view,
        full_block_table_row=block_table_row,
        block_size=block_size,
        seq_len=seq_len,
    )


def test_prefill_gather_keeps_selected_history_and_drops_the_rest():
    block_size, turn_start, seq_len = 16, 160, 200
    row = torch.arange(1000, 1000 + 20, dtype=torch.int64)
    # History blocks 0,1,5,9 selected; 2,3,4,6,7,8 not.
    selection = [b * block_size for b in (0, 1, 5, 9)]

    gathered_row, gathered_seq_len = _prefill_gather(
        row, block_size, selection, turn_start, seq_len
    )

    # 4 selected history blocks + the contiguous tail 10,11,12.
    assert gathered_row.tolist() == [1000, 1001, 1005, 1009, 1010, 1011, 1012]
    # 6 full blocks + block 12's 8 real tokens (200 - 192).
    assert gathered_seq_len == 6 * block_size + 8

    visible = _kernel_visible_positions(gathered_row, gathered_seq_len, row, block_size)
    assert not any(pos // block_size in (2, 3, 4, 6, 7, 8) for pos in visible)
    assert max(visible) == seq_len - 1


def test_prefill_gather_puts_this_chunks_own_tokens_at_the_tail_of_the_view():
    """**The causal-mask invariant, checked directly.** Bottom-right
    alignment means the last `n_q` entries of the gathered view must be
    exactly this chunk's own `n_q` positions, in ascending order. If they
    are not, every query in the chunk attends the wrong keys -- silently,
    with no crash and no shape error to catch it."""
    block_size, turn_start = 16, 160
    num_computed, seq_len = 176, 200
    n_q = seq_len - num_computed
    row = torch.arange(1000, 1000 + 20, dtype=torch.int64)
    selection = [b * block_size for b in (0, 1, 5, 9)]

    gathered_row, gathered_seq_len = _prefill_gather(
        row, block_size, selection, turn_start, seq_len
    )
    visible = _kernel_visible_positions(gathered_row, gathered_seq_len, row, block_size)

    assert visible[-n_q:] == list(range(seq_len - n_q, seq_len)), (
        f"the last {n_q} gathered entries are {visible[-n_q:]}, expected "
        f"{list(range(seq_len - n_q, seq_len))} -- bottom-right causal "
        f"alignment reads exactly these, so anything else means every query "
        f"in this chunk attends the wrong keys"
    )


def test_prefill_gather_tail_ignores_holes_the_selection_leaves_inside_it():
    """The force-kept tail is CONTIGUOUS, not "whatever the selection
    happens to name at or above the turn start". A selection that covers
    block 11 but not block 10 must still yield 10,11,12 -- punching a hole
    in the tail would break the alignment the previous test pins."""
    block_size, turn_start, seq_len = 16, 160, 200
    row = torch.arange(1000, 1000 + 20, dtype=torch.int64)
    # Block 11 named, blocks 10 and 12 not -- all three must appear anyway.
    selection = [0 * block_size, 11 * block_size]

    gathered_row, _ = _prefill_gather(row, block_size, selection, turn_start, seq_len)

    assert gathered_row.tolist() == [1000, 1010, 1011, 1012]


def test_prefill_gather_turn_zero_degenerates_to_dense():
    """Turn 0's prefill is where the context's KV is COMPUTED for the first
    time. Restricting it would poison the persistent cache every later
    turn's selection reads from, so `turn_start == 0` must produce a no-op
    -- and does, without a special case: the tail spans everything."""
    block_size, seq_len = 16, 200
    row = torch.arange(1000, 1000 + 20, dtype=torch.int64)
    selection = [0, 16, 32]  # aggressively small, and irrelevant here

    assert _prefill_gather(row, block_size, selection, 0, seq_len) is None


def test_prefill_gather_keeps_earlier_chunks_of_the_same_turn_visible():
    """Chunked prefill: a later chunk must still see the turn's earlier
    chunks. They are below `num_computed` but at or above `turn_start`, so
    only the contiguous tail rule covers them -- the selection cannot, it
    was registered before any of this turn's blocks existed as history."""
    block_size, turn_start = 16, 160
    row = torch.arange(1000, 1000 + 24, dtype=torch.int64)
    selection = [0 * block_size]

    # Chunk 1 ends at 192; chunk 2 ends at 240.
    _, seq_len_1 = _prefill_gather(row, block_size, selection, turn_start, 192)
    gathered_2, seq_len_2 = _prefill_gather(row, block_size, selection, turn_start, 240)
    visible_2 = _kernel_visible_positions(gathered_2, seq_len_2, row, block_size)

    # Everything from turn_start onward is visible in chunk 2, including
    # the positions chunk 1 computed.
    assert set(range(turn_start, 240)).issubset(set(visible_2))
    assert seq_len_1 == block_size + (192 - turn_start)


def test_prefill_gather_accounts_for_a_partially_filled_last_block():
    """Same class of bug as the decode path's confirmed off-by-one: the
    last gathered block is generally NOT full, and reporting it as full
    lets the kernel read uninitialized slots past the real occupancy."""
    block_size, turn_start = 16, 160
    row = torch.arange(1000, 1000 + 20, dtype=torch.int64)
    selection = [0 * block_size, 1 * block_size]

    for seq_len, expected_tail_tokens in ((193, 33), (200, 40), (208, 48)):
        _, gathered_seq_len = _prefill_gather(
            row, block_size, selection, turn_start, seq_len
        )
        # 2 selected history blocks + the real token count from turn_start on.
        assert gathered_seq_len == 2 * block_size + expected_tail_tokens, seq_len
        assert expected_tail_tokens == seq_len - turn_start


def test_prefill_gather_raises_on_a_turn_start_past_what_is_allocated():
    """A `turn_start` beyond the request's own written stream means the
    driver's resident-length tracking has drifted from the engine's.
    Gathering on it would read unwritten (or another request's) blocks, so
    it must fail loudly rather than silently -- same discipline as the
    decode path's out-of-range check."""
    block_size = 16
    row = torch.arange(1000, 1000 + 20, dtype=torch.int64)
    try:
        _prefill_gather(row, block_size, [0], turn_start=400, seq_len=200)
        raise AssertionError("expected ValueError for an out-of-range turn_start")
    except ValueError as e:
        assert "prefill turn-start block" in str(e), e


# ---------------------------------------------------------------------------
# Sparse PREFILL -- runner level
# ---------------------------------------------------------------------------


def _make_prefill_runner(runner_module, block_size=16, num_computed=176,
                         num_prompt=240, step_seq_len=200):
    return _make_fake_runner(
        runner_module, block_size,
        req_ids=["r0"],
        num_computed_tokens=[num_computed],
        num_prompt_tokens=[num_prompt],   # delta not finished -> a prefill chunk
        step_seq_lens=[step_seq_len],
    )


def test_apply_overrides_gathers_prefill_when_a_turn_start_is_registered():
    """The opt-in scope end to end through the runner: same chunk the
    default leaves untouched, now restricted."""
    runner_module = _load_sparse_target_runner()
    block_size, turn_start = 16, 160
    runner, attn_metadata = _make_prefill_runner(runner_module, block_size)
    original_row = attn_metadata["layer.0"].block_table[0].clone()
    selection = [b * block_size for b in (0, 1, 5, 9)]

    sparse_selection_registry.clear()
    try:
        sparse_selection_registry.register("r0", selection, turn_start)
        runner._apply_sparse_attention_overrides(attn_metadata)
    finally:
        sparse_selection_registry.clear()

    for name, layer in attn_metadata.items():
        assert layer.block_table[0, :7].tolist() == [
            1000, 1001, 1005, 1009, 1010, 1011, 1012
        ], name
        assert layer.block_table[0, 7:].abs().sum().item() == 0, name
        assert int(layer.seq_lens[0]) == 6 * block_size + 8, name
        # The two fields the decode path had confirmed bugs in must be
        # fixed up on this path too -- a stale max_seq_len or AOT schedule
        # sized for the pre-gather span is just as wrong here.
        assert layer.max_seq_len == 6 * block_size + 8, name
        assert layer.scheduler_metadata is None, name

    visible = _kernel_visible_positions(
        attn_metadata["layer.0"].block_table[0],
        int(attn_metadata["layer.0"].seq_lens[0]),
        original_row, block_size,
    )
    n_q = 200 - 176
    assert visible[-n_q:] == list(range(200 - n_q, 200))


def test_apply_overrides_records_prefill_chunks_separately_from_decode_steps():
    """`pop_prefill_steps` and `pop_attended_lens` feed different FLOP
    functions -- one charges a multi-token chunk, the other charges one
    decode row per entry. A prefill chunk landing in the decode
    accumulator would silently inflate the decode column."""
    runner_module = _load_sparse_target_runner()
    block_size, turn_start = 16, 160
    runner, attn_metadata = _make_prefill_runner(runner_module, block_size)

    sparse_selection_registry.clear()
    try:
        sparse_selection_registry.register(
            "r0", [b * block_size for b in (0, 1, 5, 9)], turn_start
        )
        runner._apply_sparse_attention_overrides(attn_metadata)
    finally:
        sparse_selection_registry.clear()

    assert runner.pop_attended_lens("r0") == []
    assert runner.pop_prefill_steps("r0") == [(200 - 176, 6 * block_size + 8)]
    # Popping resets, same contract as the decode accumulator.
    assert runner.pop_prefill_steps("r0") == []


def test_apply_overrides_charges_a_dense_prefill_chunk_at_its_full_length():
    """A chunk the gather declined to restrict (turn 0, or a selection
    already spanning everything) is a DENSE chunk, not a free one. Same
    reasoning as the decode accumulator's `gathered is None` handling: skip
    it and the measured prefill FLOPs understate what the GPU did."""
    runner_module = _load_sparse_target_runner()
    block_size = 16
    runner, attn_metadata = _make_prefill_runner(runner_module, block_size)
    before = attn_metadata["layer.0"].block_table.clone()

    sparse_selection_registry.clear()
    try:
        # turn_start=0 -> degenerate, nothing patched.
        sparse_selection_registry.register("r0", [0, 16, 32], 0)
        runner._apply_sparse_attention_overrides(attn_metadata)
    finally:
        sparse_selection_registry.clear()

    assert torch.equal(attn_metadata["layer.0"].block_table, before)
    assert runner.pop_prefill_steps("r0") == [(200 - 176, 200)]


def test_apply_overrides_leaves_prefill_dense_while_recomputing_older_tokens():
    """`num_computed < turn_start` means the engine is (re)computing tokens
    from BEFORE this turn -- a session resumption that missed the prefix
    cache. Their KV is being written for the first time, so restricting
    them would poison the cache, and the tail-contiguity invariant does not
    hold for them either. Dense is the only safe answer."""
    runner_module = _load_sparse_target_runner()
    block_size = 16
    runner, attn_metadata = _make_prefill_runner(
        runner_module, block_size, num_computed=64, step_seq_len=96,
    )
    before = {name: (layer.block_table.clone(), layer.seq_lens.clone())
              for name, layer in attn_metadata.items()}

    sparse_selection_registry.clear()
    try:
        sparse_selection_registry.register("r0", [0, 16, 32], 160)
        runner._apply_sparse_attention_overrides(attn_metadata)
    finally:
        sparse_selection_registry.clear()

    for name, layer in attn_metadata.items():
        old_bt, old_sl = before[name]
        assert torch.equal(layer.block_table, old_bt), name
        assert torch.equal(layer.seq_lens, old_sl), name
    assert runner.pop_prefill_steps("r0") == []


def test_prefill_base_view_cache_invalidates_on_the_generation_counter():
    """Same cache-invalidation contract (and the same `id()`-collision bug
    it must not reintroduce) as the decode path's two caches."""
    runner_module = _load_sparse_target_runner()
    block_size, turn_start = 16, 160
    runner, attn_metadata = _make_prefill_runner(runner_module, block_size)

    sparse_selection_registry.clear()
    try:
        sparse_selection_registry.register("r0", [0 * block_size], turn_start)
        runner._apply_sparse_attention_overrides(attn_metadata)
        first = attn_metadata["layer.0"].block_table[0, :4].tolist()

        runner2, attn_metadata_2 = _make_prefill_runner(runner_module, block_size)
        runner2._sparse_prefill_base_view_cache = runner._prefill_base_view_cache()
        runner2._sparse_base_block_indices_cache = runner._base_block_indices_cache()
        sparse_selection_registry.register("r0", [5 * block_size], turn_start)
        runner2._apply_sparse_attention_overrides(attn_metadata_2)
        second = attn_metadata_2["layer.0"].block_table[0, :4].tolist()
    finally:
        sparse_selection_registry.clear()

    assert first == [1000, 1010, 1011, 1012]
    assert second == [1005, 1010, 1011, 1012], (
        f"got {second} -- a stale cached view means the second turn is "
        f"attending the FIRST turn's selection"
    )


def test_target_sparse_prefill_flops_matches_the_analytic_model_when_dense():
    """The measured model must reduce to the analytic one on a chunk that
    was not actually restricted -- otherwise a `--sparse-prefill` run's
    turn 0 (always dense) would not be comparable to a default run's."""
    cfg = _llama31_8b_flop_cfg()
    n_cached, n_new = 4096, 512

    measured = target_sparse_prefill_flops(cfg, [(n_new, n_cached + n_new)])
    analytic = target_prefill_flops(
        cfg, prompt_len=n_cached + n_new, num_cached=n_cached
    )
    assert measured == analytic


def test_target_sparse_prefill_flops_charges_one_lm_head_row_per_turn():
    """Not one per chunk -- only the final chunk produces logits, and only
    for the one sampled token. Same convention as `target_prefill_flops`."""
    cfg = _llama31_8b_flop_cfg()
    chunks = [(256, 4096), (256, 4352)]

    total = target_sparse_prefill_flops(cfg, chunks)
    without_lm_head = sum(
        n * cfg.linear_flops_per_token + cfg.attn_prefill_flops(n, a - n)
        for n, a in chunks
    )
    assert total == without_lm_head + cfg.lm_head_flops
    assert target_sparse_prefill_flops(cfg, []) == 0


def test_target_sparse_prefill_flops_is_cheaper_when_the_gather_bites():
    """The whole point of the scope: a restricted chunk must cost strictly
    less than the same chunk read densely."""
    cfg = _llama31_8b_flop_cfg()
    n_new = 512
    dense = target_sparse_prefill_flops(cfg, [(n_new, 8192 + n_new)])
    sparse = target_sparse_prefill_flops(cfg, [(n_new, 2048 + n_new)])
    assert sparse < dense


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
        gen1, positions1, turn_start1 = sparse_selection_registry.get_with_generation("req-1")
        assert positions1 == [0, 4, 8]
        assert turn_start1 is None, "decode-only scope is the default"

        sparse_selection_registry.register("req-1", [12, 16])
        gen2, positions2, turn_start2 = sparse_selection_registry.get_with_generation("req-1")
        assert positions2 == [12, 16]
        assert turn_start2 is None
        assert gen2 != gen1, "generation must change across registrations"

        # A second request_id's own generation sequence is independent --
        # the counter is global/monotonic, not per-request_id, but that
        # must never cause two DIFFERENT requests' current generations to
        # collide either.
        sparse_selection_registry.register("req-2", [1, 2])
        gen_req2, _, _ = sparse_selection_registry.get_with_generation("req-2")
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


# --------------------------------------------------------------------------
# flops_model.py -- the analytic FLOP model (see its module docstring)
# --------------------------------------------------------------------------

def _llama31_8b_flop_cfg():
    return ModelFlopConfig(
        num_layers=32, hidden=4096, num_heads=32, num_kv_heads=8,
        head_dim=128, intermediate=14336, vocab=128256,
    )


def _tiny_flop_cfg():
    return ModelFlopConfig(
        num_layers=2, hidden=8, num_heads=2, num_kv_heads=1,
        head_dim=4, intermediate=16, vocab=32,
    )


def test_flops_prefill_attention_matches_brute_force():
    """Closed form vs. an explicit per-query-token loop -- the one place an
    off-by-one in the causal triangle would hide."""
    cfg = _tiny_flop_cfg()
    per_pair = cfg.num_layers * 4 * cfg.num_heads * cfg.head_dim
    for n_new, n_cached in [(1, 0), (5, 0), (1, 100), (7, 13), (64, 4096)]:
        brute = sum(n_cached + i + 1 for i in range(n_new)) * per_pair
        assert cfg.attn_prefill_flops(n_new, n_cached) == brute, (n_new, n_cached)


def test_flops_dense_term_is_exactly_two_n():
    """`linear_flops_per_token` must equal 2 x non-embedding params -- the
    standard cross-check that catches a transposed projection dimension."""
    cfg = _llama31_8b_flop_cfg()
    params = cfg.num_layers * (
        cfg.hidden * (cfg.num_heads + 2 * cfg.num_kv_heads) * cfg.head_dim  # QKV
        + cfg.num_heads * cfg.head_dim * cfg.hidden                          # O
        + 3 * cfg.hidden * cfg.intermediate                                  # SwiGLU
    )
    assert cfg.linear_flops_per_token == 2 * params
    # ~6.98B non-embedding params for Llama-3.1-8B -> ~13.96 GFLOP/token.
    assert 13.9e9 < cfg.linear_flops_per_token < 14.0e9


def test_flops_lm_head_charged_once_per_prefill_not_per_token():
    """Charging lm_head per prefill token would dwarf the prefill itself at
    SCBench context lengths -- pin it to exactly one row."""
    cfg = _llama31_8b_flop_cfg()
    prompt_len = 100_000
    got = target_prefill_flops(cfg, prompt_len, num_cached=0)
    expected = (
        prompt_len * cfg.linear_flops_per_token
        + cfg.attn_prefill_flops(prompt_len, 0)
        + cfg.lm_head_flops
    )
    assert got == expected
    # A fully-cached prefill computes nothing at all, lm_head included.
    assert target_prefill_flops(cfg, prompt_len, num_cached=prompt_len) == 0


def test_flops_gqa_shrinks_qkv_but_not_attention():
    """num_kv_heads shrinks the KV cache (bytes_per_token_kv's axis) and the
    QKV projection -- never the attention arithmetic."""
    gqa = _llama31_8b_flop_cfg()
    mha = ModelFlopConfig(**{**gqa.__dict__, "num_kv_heads": gqa.num_heads})
    assert gqa.qkv_flops_per_token < mha.qkv_flops_per_token
    assert gqa.attn_prefill_flops(64, 1024) == mha.attn_prefill_flops(64, 1024)
    assert gqa.attn_decode_step_flops(4096) == mha.attn_decode_step_flops(4096)


def test_flops_scoring_is_qk_only_so_half_of_full_attention():
    """compute_attention_score never contracts against V, so it must cost
    exactly half of what an equivalent full attention would."""
    cfg = _tiny_flop_cfg()
    look_ahead, ctx_len = 8, 4096
    full = cfg.num_layers * 4 * cfg.num_heads * cfg.head_dim * look_ahead * ctx_len
    assert cfg.scoring_flops(look_ahead, ctx_len) * 2 == full
    # Linear in context length -- the property that makes it worth its own
    # column rather than folding into the speculator forward cost.
    assert cfg.scoring_flops(8, 8192) == 2 * cfg.scoring_flops(8, 4096)


def test_flops_sparse_decode_is_cheaper_than_dense_and_monotone():
    cfg = _llama31_8b_flop_cfg()
    dense = dense_decode_attended_lens(prompt_len=32768, num_decode_steps=64)
    half = [n // 2 for n in dense]
    quarter = [n // 4 for n in dense]
    f_dense = target_decode_flops(cfg, dense)
    f_half = target_decode_flops(cfg, half)
    f_quarter = target_decode_flops(cfg, quarter)
    assert f_quarter < f_half < f_dense
    # The per-step weight cost (linear + lm_head) is NOT saved by sparsity,
    # so halving attention must not halve the total -- if it does, the
    # weight term has gone missing.
    assert f_half > f_dense / 2


def test_dense_decode_attended_lens_is_one_short_of_out_len():
    """The first output token comes from the prefill's logits row, so N
    generated tokens cost N-1 decode steps."""
    lens = dense_decode_attended_lens(prompt_len=10, num_decode_steps=3)
    assert lens == [11, 12, 13]
    assert dense_decode_attended_lens(10, 0) == []
    assert dense_decode_attended_lens(10, -5) == []


def test_flops_breakdown_accumulates_and_attributes():
    a = FlopBreakdown(spec_prefill=10, spec_scoring=5, target_decode=85)
    b = FlopBreakdown(spec_lookahead=100, target_prefill=100)
    total = FlopBreakdown()
    total += a
    total += b
    assert total.total == 300
    assert total.speculator_total == 115
    assert abs(total.speculator_fraction - 115 / 300) < 1e-12
    assert FlopBreakdown().speculator_fraction == 0.0
    d = a.as_dict()
    assert d["total"] == 100 and d["spec_prefill"] == 10 and d["target_prefill"] == 0


def test_flops_speculator_turn_respects_prefix_cache_and_lookahead():
    cfg = _tiny_flop_cfg()
    pool_len, look_ahead = 1000, 4
    cold = speculator_turn_flops(cfg, pool_len, num_cached=0, look_ahead=look_ahead)
    warm = speculator_turn_flops(cfg, pool_len, num_cached=900, look_ahead=look_ahead)
    # Prefix-cache hits shrink prefill only; lookahead and scoring both
    # still run over the FULL pool -- ignoring that would make a warm turn
    # look nearly free, which it isn't.
    assert warm.spec_prefill < cold.spec_prefill
    assert warm.spec_lookahead == cold.spec_lookahead
    assert warm.spec_scoring == cold.spec_scoring
    # Lookahead steps attend a growing sequence: pool_len+1 .. pool_len+k.
    expected_lookahead = sum(
        cfg.linear_flops_per_token + cfg.lm_head_flops
        + cfg.attn_decode_step_flops(pool_len + 1 + j)
        for j in range(look_ahead)
    )
    assert cold.spec_lookahead == expected_lookahead
    # EOS on the first candidate token: no lookahead, and nothing to score.
    none = speculator_turn_flops(cfg, pool_len, num_cached=0, look_ahead=0)
    assert none.spec_lookahead == 0 and none.spec_scoring == 0
    assert none.spec_prefill == cold.spec_prefill


def test_flops_model_config_hf_fallbacks():
    """Same fallbacks as bytes_per_token_kv: a config with neither head_dim
    nor num_key_value_heads must still resolve."""
    class _HF:
        num_hidden_layers = 4
        hidden_size = 64
        num_attention_heads = 8
        intermediate_size = 256
        vocab_size = 1000

    cfg = model_flop_config(_HF())
    assert cfg.num_kv_heads == 8       # defaulted to num_attention_heads
    assert cfg.head_dim == 8           # hidden_size // num_attention_heads

    class _HFGqa(_HF):
        num_key_value_heads = 2
        head_dim = 16

    gqa = model_flop_config(_HFGqa())
    assert gqa.num_kv_heads == 2 and gqa.head_dim == 16


def test_flops_hand_computed_tiny_breakdown():
    """Pins the stage split itself, not just the total, against numbers
    worked out by hand rather than by re-running the implementation."""
    cfg = ModelFlopConfig(
        num_layers=1, hidden=2, num_heads=1, num_kv_heads=1,
        head_dim=2, intermediate=4, vocab=8,
    )
    # qkv = 2*2*(1+2)*2 = 24; o = 2*1*2*2 = 8; mlp = 6*2*4 = 48 -> 80/token
    assert cfg.linear_flops_per_token == 80
    assert cfg.lm_head_flops == 32  # 2 * hidden * vocab
    # per query/key pair = 4*1*2 = 8
    assert cfg._attn_flops_per_query_key_pair == 8
    # prefill 3 tokens, nothing cached: keys visited = 1+2+3 = 6 -> 48
    assert cfg.attn_prefill_flops(3, 0) == 48
    assert target_prefill_flops(cfg, prompt_len=3, num_cached=0) == 3 * 80 + 48 + 32
    # two decode steps attending 4 and 5 keys
    assert target_decode_flops(cfg, [4, 5]) == 2 * (80 + 32) + 8 * (4 + 5)


def test_flop_summary_fields_match_the_csv_schema():
    """Every key `_flop_summary_fields` emits must exist in CSV_FIELDS, in
    both the populated and the empty case -- a mismatch means DictWriter
    raises mid-run, after the experiment has already been paid for."""
    turn = FlopBreakdown(spec_prefill=1e12, spec_lookahead=2e12, spec_scoring=3e12,
                         target_prefill=4e12, target_decode=10e12)
    populated = _flop_summary_fields([turn, turn], elapsed=10.0, peak_tflops=312.0)
    empty = _flop_summary_fields([], elapsed=10.0, peak_tflops=None)
    assert set(populated) == set(empty)
    missing = set(populated) - set(CSV_FIELDS)
    assert not missing, f"emitted keys absent from CSV_FIELDS: {sorted(missing)}"
    assert all(v is None for v in empty.values())

    # Stage columns are per-turn MEANS, total_tflops is the experiment.
    assert populated["spec_prefill_tflops_per_turn_mean"] == 1.0
    assert populated["total_tflops_per_turn_mean"] == 20.0
    assert populated["total_tflops"] == 40.0
    assert abs(populated["speculator_flops_fraction"] - 6 / 20) < 1e-12
    assert populated["achieved_tflops_per_s"] == 4.0
    assert abs(populated["mfu"] - 4.0 / 312.0) < 1e-12
    # No peak given -> no MFU claim, rather than a guessed one.
    assert _flop_summary_fields([turn], elapsed=1.0, peak_tflops=None)["mfu"] is None


def test_num_decode_steps_excludes_the_prefill_sampled_token():
    assert _num_decode_steps(0) == 0
    assert _num_decode_steps(1) == 0
    assert _num_decode_steps(64) == 63


def test_oracle_rows_pair_one_to_one_with_a_sparse_row():
    """An ORACLE-k{N} row is only interpretable as a CEILING for
    SPARSE-k{N}-g32 if the two differ in exactly one thing: which
    checkpoint scores. Everything the experiment matrix itself controls --
    keep rate, granularity, keep mode, and the driving loop implied by the
    mode -- must match its partner row, or the "gap" between them silently
    conflates estimator quality with whatever else drifted apart.

    Locked down here because the matrix is built by two separate loops in
    `_build_experiments` (one for ORACLE, one for SPARSE) that could
    trivially diverge -- e.g. someone extends KEEP_RATES for SPARSE only,
    leaving an oracle row with no partner, or flips ORACLE_GRANULARITY to a
    value no SPARSE row was ever run at."""
    oracle_ids = [eid for eid, cfg in EXPERIMENTS.items() if cfg["mode"] == "oracle"]
    assert oracle_ids, "no ORACLE-k* rows in the matrix at all"

    for eid in oracle_ids:
        cfg = EXPERIMENTS[eid]
        partner_id = f"SPARSE-k{int(cfg['keep_percentage'] * 100)}-g{cfg['granularity']}"
        assert partner_id in EXPERIMENTS, (
            f"{eid} has no SPARSE partner row {partner_id!r} to be a ceiling "
            f"for -- ORACLE_GRANULARITY and KEEP_RATES must stay inside the "
            f"SPARSE grid"
        )
        partner = EXPERIMENTS[partner_id]
        for field in ("keep_percentage", "granularity", "keep_mode"):
            assert cfg[field] == partner[field], (
                f"{eid} vs. {partner_id}: {field} differs "
                f"({cfg[field]!r} vs. {partner[field]!r})"
            )

    # And the ceiling covers every keep rate that has a SPARSE row at the
    # oracle's granularity -- a ceiling for 3 of 4 rates is a gap in the
    # attribution, not a smaller experiment.
    oracle_rates = {EXPERIMENTS[eid]["keep_percentage"] for eid in oracle_ids}
    sparse_rates_at_oracle_gran = {
        cfg["keep_percentage"]
        for cfg in EXPERIMENTS.values()
        if cfg["mode"] == "sparse"
        and cfg["granularity"] == EXPERIMENTS[oracle_ids[0]]["granularity"]
    }
    assert oracle_rates == sparse_rates_at_oracle_gran, (
        f"oracle keep rates {sorted(oracle_rates)} do not cover the SPARSE "
        f"rates at the same granularity {sorted(sparse_rates_at_oracle_gran)}"
    )


def _run_experiment_with_stubs(exp_id, **arg_overrides):
    """Drives `predict_scbench.run_experiment` for one row with every heavy
    dependency stubbed out (no vLLM, no GPU, no checkpoints, no dataset),
    and reports what it *tried* to construct: the target engine's kwargs,
    the scorer engine's kwargs, and which driving loop it dispatched to.

    Stubbing rather than mocking a real run is the point -- the thing under
    test is `run_experiment`'s own wiring decisions, which are pure
    branching on `exp_cfg["mode"]` and therefore fully exercisable on CPU.
    Everything downstream of those decisions (the engines, the loop) is
    validated on real hardware by the validate_*.py scripts instead."""
    import argparse
    import sys
    import tempfile
    import types
    from pathlib import Path as _Path

    import predict_scbench as ps

    calls = {"llm_kwargs": None, "proposer_kwargs": None, "loop": None,
             "row": None}

    class _FakeLLM:
        def __init__(self, **kwargs):
            calls["llm_kwargs"] = kwargs
            self.llm_engine = types.SimpleNamespace()

    class _FakeProposer:
        def __init__(self, **kwargs):
            calls["proposer_kwargs"] = kwargs

    class _FakeHFConfig:
        max_position_embeddings = 131072

    def _fake_loop(name):
        def loop(*args, **kwargs):
            calls["loop"] = name
            return [], {
                "ttfts": [], "out_lens": [], "actual_keep_rates": [],
                "num_cached_tokens_speculator": [], "num_skipped_too_large": 0,
                "turn_elapsed": [], "flops": [],
                "finish": {"stop": 0, "length": 0, "other": 0},
            }
        return loop

    import transformers

    import vllm_patch.proposer as proposer_mod

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM

    saved = {
        "vllm": sys.modules.get("vllm"),
        "proposer": proposer_mod.SpecPrefillProposer,
        "autoconfig": transformers.AutoConfig.from_pretrained,
        "autotok": transformers.AutoTokenizer.from_pretrained,
        "load": ps.load_conversations,
        "sparse": ps.run_sparse_attention,
        "specprefill": ps.run_specprefill,
        "baseline": ps.run_baseline,
        "csv": ps.CSV_PATH,
        "append": ps.append_csv_row,
        "out_dir": ps.OUT_DIR,
    }
    tmp_dir = tempfile.mkdtemp(prefix="oracle_wiring_test_")
    ps.OUT_DIR = _Path(tmp_dir)
    ps.CSV_PATH = ps.OUT_DIR / "all_runs.csv"
    sys.modules["vllm"] = fake_vllm
    proposer_mod.SpecPrefillProposer = _FakeProposer
    transformers.AutoConfig.from_pretrained = staticmethod(lambda *a, **k: _FakeHFConfig())
    transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: object())
    ps.load_conversations = lambda *a, **k: []
    ps.run_sparse_attention = _fake_loop("sparse")
    ps.run_specprefill = _fake_loop("specprefill")
    ps.run_baseline = _fake_loop("baseline")
    ps.append_csv_row = lambda row: calls.__setitem__("row", row)

    args = argparse.Namespace(
        samples=None, max_conversations=None, scbench_config=None,
        target_model="/ckpt/Llama-3.1-8B-Instruct",
        speculator_model="/ckpt/Llama-3.2-1B-Instruct",
        speculator_device=None,
        target_gpu_memory_utilization=0.85,
        # TP=1: the single-card default, so these stub runs keep
        # exercising the pre-TP placement behaviour.
        target_tensor_parallel_size=1,
        speculator_gpu_memory_utilization=0.2,
        target_max_num_batched_tokens=131072,
        speculator_max_num_batched_tokens=131072,
        target_prefill_chunk_tokens=None,
        scorer_prefill_chunk_tokens=None,
        oracle_scorer_model=None, oracle_scorer_device=None,
        oracle_scorer_gpu_memory_utilization=0.6,
        oracle_scorer_max_num_batched_tokens=None,
        early_scorer_gpu_memory_utilization=0.3,
        max_tokens=64, reps=1, peak_tflops=None,
        sparse_prefill=False,
        output_suffix="", head_set_from=None,
    )
    for k, v in arg_overrides.items():
        assert hasattr(args, k), f"unknown arg override {k!r}"
        setattr(args, k, v)

    try:
        ps.run_experiment(exp_id, EXPERIMENTS[exp_id], args)
    finally:
        if saved["vllm"] is None:
            del sys.modules["vllm"]
        else:
            sys.modules["vllm"] = saved["vllm"]
        proposer_mod.SpecPrefillProposer = saved["proposer"]
        transformers.AutoConfig.from_pretrained = saved["autoconfig"]
        transformers.AutoTokenizer.from_pretrained = saved["autotok"]
        ps.load_conversations = saved["load"]
        ps.run_sparse_attention = saved["sparse"]
        ps.run_specprefill = saved["specprefill"]
        ps.run_baseline = saved["baseline"]
        ps.CSV_PATH = saved["csv"]
        ps.append_csv_row = saved["append"]
        ps.OUT_DIR = saved["out_dir"]
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return calls


def test_oracle_row_scores_with_the_target_checkpoint_on_the_sparse_loop():
    """The whole claim of the ORACLE-k* row is "SPARSE-k{N}-g32 with exactly
    one thing changed: who scores." Four things have to hold for that claim
    to be true, and all four are `run_experiment` wiring decisions rather
    than anything the driving loop can check for itself:

    1. the target engine is the SPARSE one (persistent resumable session,
       block-gather worker) -- an oracle scored against the physically-
       pruned pipeline would be a ceiling for a different architecture;
    2. the scorer engine holds the TARGET checkpoint, not the 1B speculator;
    3. it gets its own memory budget (the 1B-sized 0.2 default would not fit
       an 8B scorer plus its growing per-conversation KV);
    4. the driving loop is `run_sparse_attention`, the same one the SPARSE
       rows use.

    Get any of them wrong and the run still completes and still writes a
    plausible-looking score -- it just wouldn't be a ceiling for the row it
    is being compared against. Hence a wiring test rather than trusting the
    numbers to look wrong."""
    oracle = _run_experiment_with_stubs("ORACLE-k20")

    assert oracle["llm_kwargs"]["worker_cls"] == (
        "vllm_patch.sparse_target_runner.SparseTargetWorker"
    )
    assert oracle["llm_kwargs"]["enable_prefix_caching"] is True
    # Required by the resumable-session mechanism -- see run_experiment's own
    # comment on the scheduler race this avoids.
    assert oracle["llm_kwargs"]["async_scheduling"] is False

    assert oracle["proposer_kwargs"]["speculator_model_path"] == "/ckpt/Llama-3.1-8B-Instruct"
    assert oracle["proposer_kwargs"]["gpu_memory_utilization"] == 0.6
    assert oracle["loop"] == "sparse"

    # Per-step batch size defaults to the context budget (previous
    # behavior), and both engines run with chunked prefill available so
    # --*-prefill-chunk-tokens is a working knob rather than a silently
    # ignored one.
    assert oracle["llm_kwargs"]["max_num_batched_tokens"] == 131072
    assert oracle["llm_kwargs"]["enable_chunked_prefill"] is True
    assert oracle["proposer_kwargs"]["max_num_batched_tokens"] == 131072
    assert oracle["proposer_kwargs"]["enable_chunked_prefill"] is True

    # ...and the chunk knob reaches the engines without moving the context
    # ceiling with it -- the whole point of separating them, since the
    # ceiling is also the conversation-skip threshold.
    chunked = _run_experiment_with_stubs(
        "ORACLE-k20", target_prefill_chunk_tokens=32768,
        scorer_prefill_chunk_tokens=32768,
    )
    assert chunked["llm_kwargs"]["max_num_batched_tokens"] == 32768
    assert chunked["llm_kwargs"]["max_model_len"] == oracle["llm_kwargs"]["max_model_len"]
    assert chunked["proposer_kwargs"]["max_num_batched_tokens"] == 32768
    assert chunked["proposer_kwargs"]["max_model_len"] == oracle["proposer_kwargs"]["max_model_len"]

    # ...and the SPARSE partner row differs in exactly the one field the
    # comparison is supposed to isolate.
    sparse = _run_experiment_with_stubs("SPARSE-k20-g32")
    assert sparse["proposer_kwargs"]["speculator_model_path"] == "/ckpt/Llama-3.2-1B-Instruct"
    assert sparse["loop"] == "sparse"
    assert sparse["llm_kwargs"] == oracle["llm_kwargs"]
    differing = {
        k for k in oracle["proposer_kwargs"]
        if oracle["proposer_kwargs"][k] != sparse["proposer_kwargs"][k]
    }
    assert differing <= {"speculator_model_path", "gpu_memory_utilization"}, (
        f"ORACLE and its SPARSE partner differ in more than the scorer "
        f"checkpoint and its memory budget: {sorted(differing)}"
    )


def test_oracle_scorer_sharing_the_targets_gpu_fails_before_the_run_starts():
    """The first real ORACLE-k20 run died with a CUDA OOM inside an
    activation kernel, tens of minutes in, holding 76.6GiB -- more than
    either engine's own configured budget, i.e. two 8B engines on one card.
    That traceback cannot name the culprit: `SpecPrefillProposer` places the
    scorer by rewriting CUDA_VISIBLE_DEVICES for the child process, so every
    engine calls its own device "GPU 0" regardless of which physical card it
    holds.

    So the placement is checked up front instead, and the check has to
    RAISE rather than warn: a warning at minute 0 of an hours-long sweep is
    read after the OOM, not before it."""
    try:
        _run_experiment_with_stubs("ORACLE-k20", oracle_scorer_device="cuda:0")
    except RuntimeError as e:
        assert "same GPU as the target" in str(e)
        assert "--oracle-scorer-device" in str(e), (
            "the error must name the flag that fixes it"
        )
    else:
        raise AssertionError(
            "an oracle scorer placed on device 0 (the target's own device) "
            "must fail before either engine allocates, not OOM mid-run"
        )

    # The 1B speculator sharing the target's GPU is a different question --
    # ~2GB of weights, which is why the SPARSE rows have always been free to
    # do it -- so the guard must not fire for them.
    sparse = _run_experiment_with_stubs("SPARSE-k20-g32", speculator_device="cuda:0")
    assert sparse["loop"] == "sparse"


def test_scoring_layer_indices_selects_late_layers_and_never_empties():
    """The layer-restriction policy, as integers. Every named selection drops
    EARLY layers -- layers 0-1 are positional/sink-dominated, and under the
    default `max` aggregation one peaked early head can set the whole
    importance vector by itself (ACCURACY_IMPROVEMENTS.md §1.2)."""
    assert scoring_layer_indices(32, None) == list(range(32))
    assert scoring_layer_indices(32, "skip_first2") == list(range(2, 32))
    assert scoring_layer_indices(32, "second_half") == list(range(16, 32))
    assert scoring_layer_indices(32, "last_quarter") == list(range(24, 32))
    # The 1B speculator has 16 layers, the 8B oracle scorer 32 -- the same
    # selection name must mean "the same fraction of the model" on both,
    # or a variant's result would not transfer between the two scorers.
    assert scoring_layer_indices(16, "second_half") == list(range(8, 16))
    assert scoring_layer_indices(16, "last_quarter") == list(range(12, 16))

    # Never empty: an empty selection would mean averaging over zero layers,
    # which is a silent NaN score vector rather than an error.
    for num_layers in (1, 2, 3, 4):
        for selection in (None, "skip_first2", "second_half", "last_quarter"):
            got = scoring_layer_indices(num_layers, selection)
            assert got, f"{selection!r} emptied the layer set at {num_layers} layers"
            assert max(got) == num_layers - 1
            assert min(got) >= 0

    try:
        scoring_layer_indices(32, "no_such_selection")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown score_layers must raise, not silently pass through")


def test_default_scoring_config_is_bit_identical_to_the_reference():
    """The scoring variants are only safe to add if leaving them unset
    reproduces the reference implementation EXACTLY -- every published row
    (M000's partners, the 12-cell SPARSE grid, ORACLE-k20) was measured with
    `max` over all layers, and those numbers are the baseline the variants
    are judged against. A default that drifted would silently invalidate the
    comparison rather than fail loudly."""
    torch.manual_seed(0)
    LAYERS, HEADS, LOOKAHEAD, CTX = 6, 4, 3, 64
    attn = [torch.randn(LAYERS, HEADS, LOOKAHEAD, CTX)]

    default_cfg = SpecConfig(keep_strategy="percentage", pool_kernel_size=13)
    assert default_cfg.score_aggregation == "max"
    assert default_cfg.score_layers is None

    # The reference pipeline, inlined: softmax -> pool -> max over
    # (layer, head) -> mean over lookahead.
    ref = torch.nn.functional.softmax(attn[0], dim=-1, dtype=torch.float32).to(attn[0].dtype)
    ref = ref.flatten(0, 1)
    ref = torch.nn.functional.avg_pool1d(ref, kernel_size=13, padding=6, stride=1)
    ref = ref.max(0)[0].mean(0)

    got = aggregate_attention_score(attn, default_cfg)[0]
    assert torch.equal(got, ref), "default scoring drifted from the reference pipeline"


def test_scoring_variants_change_the_selection_without_changing_its_shape():
    """Each variant must produce a real, different ranking -- a config knob
    that silently does nothing is worse than no knob, because a null result
    from it would be read as "this idea doesn't help" rather than "this idea
    was never tested". Shape and finiteness are checked alongside, since
    everything downstream (chunk selection, the block gather) assumes one
    finite score per prompt token."""
    torch.manual_seed(0)
    LAYERS, HEADS, LOOKAHEAD, CTX = 8, 4, 3, 128
    attn = [torch.randn(LAYERS, HEADS, LOOKAHEAD, CTX)]

    scores = {}
    for aggregation in ("max", "mean", "zmean"):
        for layers in (None, "skip_first2", "second_half", "last_quarter"):
            cfg = SpecConfig(
                keep_strategy="percentage", pool_kernel_size=13,
                score_aggregation=aggregation, score_layers=layers,
            )
            out = aggregate_attention_score(attn, cfg)[0]
            assert out.shape == (CTX,), f"{aggregation}/{layers} changed the score shape"
            assert torch.isfinite(out).all(), f"{aggregation}/{layers} produced non-finite scores"
            scores[(aggregation, layers)] = out

    # Every configuration ranks the context differently from the reference.
    reference = scores[("max", None)]
    ref_order = torch.argsort(reference, descending=True)
    for key, out in scores.items():
        if key == ("max", None):
            continue
        order = torch.argsort(out, descending=True)
        assert not torch.equal(order, ref_order), (
            f"variant {key} produced the reference ranking -- it is not "
            f"actually varying anything"
        )


def test_zmean_survives_a_constant_attention_head():
    """`zmean` divides by each head's standard deviation across the context.
    A head whose pooled distribution is genuinely flat has zero variance --
    a real case for a sink-dominated head after average pooling, not a
    hypothetical -- and without the epsilon that one head turns the entire
    score vector into NaN, silently, for the whole turn."""
    LAYERS, HEADS, LOOKAHEAD, CTX = 2, 2, 1, 32
    attn = torch.randn(LAYERS, HEADS, LOOKAHEAD, CTX)
    attn[0, 0] = 5.0  # one perfectly flat head

    cfg = SpecConfig(keep_strategy="percentage", pool_kernel_size=13, score_aggregation="zmean")
    out = aggregate_attention_score([attn], cfg)[0]
    assert torch.isfinite(out).all(), "a constant head produced NaN/inf scores under zmean"


def test_gold_token_positions_maps_char_spans_to_token_indices():
    """The §1.3 gate ranks heads by how much attention they put on the gold
    span, so a wrong span silently corrupts every number it reports -- and
    wrongly, not obviously: the run still completes and still prints a
    plausible ceiling.

    The rule under test is the overlap condition. A token counts if it
    overlaps the gold's character range AT ALL (partial overlap included --
    tokenizers merge across a boundary, and half a gold token is still gold),
    and tokens merely ADJACENT to the range do not count.

    A stub tokenizer stands in for a real one deliberately: what is being
    tested is this function's index arithmetic over an offset mapping, not
    any tokenizer's behavior, and the CPU suite has no model weights.
    """
    from diagnose_retrieval_heads import gold_token_positions

    class _StubTok:
        """Splits on spaces, reporting real character offsets."""
        def __init__(self, text):
            self.spans = []
            pos = 0
            for word in text.split(" "):
                self.spans.append((pos, pos + len(word)))
                pos += len(word) + 1

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            return {"offset_mapping": self.spans}

    context = "alpha beta gamma delta epsilon"
    #          0-4   6-9  11-15 17-21 23-29
    tok = _StubTok(context)

    # Exact single-token match.
    assert gold_token_positions(tok, context, "gamma") == [2]
    # Multi-token span.
    assert gold_token_positions(tok, context, "beta gamma") == [1, 2]
    # Partial overlap at both ends still counts the touched tokens, and the
    # untouched neighbours are excluded.
    assert gold_token_positions(tok, context, "ta gam") == [1, 2]
    # First and last tokens are reachable (no fencepost at either end).
    assert gold_token_positions(tok, context, "alpha") == [0]
    assert gold_token_positions(tok, context, "epsilon") == [4]
    # Absent gold is an empty result, not an exception and not a bogus span --
    # a legitimate outcome for a non-retrieval config, which the caller skips.
    assert gold_token_positions(tok, context, "not-in-context") == []


def test_jaccard_bounds_head_set_stability():
    """Head stability is the gate's second, independently fatal question: a
    high ceiling is worthless if the useful heads differ per input, because
    a retrieval head set is by definition static. Jaccard is how that is
    reported, so its degenerate cases have to be right -- identical sets must
    read 1.0 and disjoint sets 0.0, or the conclusion inverts."""
    from diagnose_retrieval_heads import jaccard

    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    assert jaccard({1, 2, 3}, {4, 5, 6}) == 0.0
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    # Two empty sets share nothing to be stable ABOUT -- 0.0, not a
    # divide-by-zero and not a spurious 1.0 that would read as perfect
    # stability.
    assert jaccard(set(), set()) == 0.0


def test_score_head_set_masks_exactly_the_named_heads():
    """§1.3's whole mechanism: only the listed heads vote. Verified by
    equivalence rather than by inspection -- scoring with `score_head_set=S`
    must equal scoring an attention tensor that only ever contained S's rows.
    If the indices were interpreted against the wrong axis (layer instead of
    layer*head is the easy mistake, since the tensor is [L, H, ...] until it
    is flattened), this fails; eyeballing the output would not."""
    torch.manual_seed(0)
    LAYERS, HEADS, LOOKAHEAD, CTX = 4, 3, 2, 48
    attn = torch.randn(LAYERS, HEADS, LOOKAHEAD, CTX)

    # Heads are indexed into the FLATTENED layer*head axis: layer l, head h
    # is index l * HEADS + h.
    chosen = [0 * HEADS + 1, 2 * HEADS + 0, 3 * HEADS + 2]

    masked_cfg = SpecConfig(keep_strategy="percentage", pool_kernel_size=13,
                            score_head_set=chosen)
    got = aggregate_attention_score([attn], masked_cfg)[0]

    # The same computation with a tensor that physically contains only those
    # rows, reshaped as a 3-layer/1-head model so the reference path sees
    # exactly the chosen distributions and nothing else.
    only_chosen = attn.reshape(LAYERS * HEADS, LOOKAHEAD, CTX)[chosen]
    equivalent = only_chosen.reshape(len(chosen), 1, LOOKAHEAD, CTX)
    ref_cfg = SpecConfig(keep_strategy="percentage", pool_kernel_size=13)
    expected = aggregate_attention_score([equivalent], ref_cfg)[0]

    assert torch.equal(got, expected), (
        "score_head_set did not select the heads its indices name"
    )

    # And it genuinely differs from letting every head vote -- a mask that
    # changed nothing would make a null §1.3 result unfalsifiable.
    all_heads = aggregate_attention_score([attn], ref_cfg)[0]
    assert not torch.equal(got, all_heads)


def test_score_head_set_rejects_configs_that_would_silently_mislead():
    """Three ways a head set can be wrong while still running to completion
    and writing a plausible-looking score. All three have to raise.

    The last one is the subtle one: `score_layers` removes layers from the
    very axis `score_head_set`'s indices are counted along, so combining them
    silently reinterprets every index as a different head."""
    # Empty list -- would score with no heads and produce NaN importance.
    try:
        SpecConfig(keep_strategy="percentage", score_head_set=[])
    except AssertionError as e:
        assert "empty" in str(e)
    else:
        raise AssertionError("an empty score_head_set must raise")

    # Duplicates -- would vote twice under `mean`, reweighting the aggregate.
    try:
        SpecConfig(keep_strategy="percentage", score_head_set=[3, 3, 7])
    except AssertionError as e:
        assert "duplicate" in str(e)
    else:
        raise AssertionError("a duplicated head must raise")

    # Combined with score_layers -- indices would point at different heads.
    try:
        SpecConfig(keep_strategy="percentage", score_head_set=[3],
                   score_layers="second_half")
    except AssertionError as e:
        assert "score_layers" in str(e)
    else:
        raise AssertionError("score_head_set + score_layers must raise")

    # A head list from a bigger checkpoint selects nothing here, which must
    # be an error rather than a silent all-head fallback: head indices do not
    # transfer between models.
    attn = torch.randn(2, 2, 1, 16)  # 4 heads total
    cfg = SpecConfig(keep_strategy="percentage", pool_kernel_size=13,
                     score_head_set=[500, 501])
    try:
        aggregate_attention_score([attn], cfg)
    except ValueError as e:
        assert "different checkpoint" in str(e)
    else:
        raise AssertionError("an out-of-range head list must raise, not silently pass")



# --------------------------------------------------------------------------
# EARLY-k*-g32-L<n> -- the target's own first n layers as the speculator
# --------------------------------------------------------------------------

def test_truncated_layer_weight_filter_keeps_exactly_the_owned_layers():
    """The mechanism that makes an EARLY row loadable at all.

    `hf_overrides={"num_hidden_layers": n}` builds n decoder layers and
    nothing where layers n.. used to be (`make_layers` only pads with
    `PPMissingLayer` for PIPELINE ranks), while the weight loader still
    yields every tensor in the checkpoint -- so `LlamaModel.load_weights`
    would `KeyError` on `layers.<n>....`. This predicate is what drops them,
    and it has to be exact in both directions: dropping too much silently
    leaves a layer at its initialization values (a plausible-looking run that
    measures nothing), keeping too much crashes the load.

    Non-layer weights are always kept -- `embed_tokens` and `norm` live on
    `LlamaModel` itself, `lm_head` one level up, and all three are needed
    whatever n is."""
    for i in range(4):
        assert keep_weight_for_layer_range(f"layers.{i}.self_attn.q_proj.weight", 0, 4)
    for i in (4, 5, 16, 31):
        assert not keep_weight_for_layer_range(f"layers.{i}.self_attn.q_proj.weight", 0, 4)

    for name in ("embed_tokens.weight", "norm.weight", "lm_head.weight"):
        assert keep_weight_for_layer_range(name, 0, 4), name

    # A prefix match is not enough: `layers.10....` must not be kept just
    # because it starts with the same characters as `layers.1....`. The
    # trailing dot in the pattern is what makes this hold.
    assert not keep_weight_for_layer_range("layers.10.mlp.up_proj.weight", 0, 2)
    assert keep_weight_for_layer_range("layers.1.mlp.up_proj.weight", 0, 2)

    # Honors start_layer too, so the filter stays correct under real pipeline
    # parallelism instead of only under PP=1.
    assert not keep_weight_for_layer_range("layers.0.mlp.up_proj.weight", 2, 4)
    assert keep_weight_for_layer_range("layers.2.mlp.up_proj.weight", 2, 4)


def test_truncated_layer_weight_filter_is_a_no_op_on_an_untruncated_model():
    """Installed unconditionally rather than behind a flag, so this is the
    property that keeps every already-published SPARSE/ORACLE row loading
    byte-identically: when the stack owns every layer the checkpoint has,
    nothing is ever filtered."""
    names = (
        ["embed_tokens.weight", "norm.weight", "lm_head.weight"]
        + [f"layers.{i}.self_attn.qkv_proj.weight" for i in range(32)]
        + [f"layers.{i}.mlp.gate_up_proj.weight" for i in range(32)]
    )
    assert all(keep_weight_for_layer_range(n, 0, 32) for n in names)


def test_early_rows_exist_at_every_layer_budget_with_a_sparse_partner():
    """The grid itself: n = 1..8 at the k20-g32 probe point, plus the k60/k80
    economics follow-ups.

    n stops at 8 for an arithmetic reason, not a taste one -- `r = n/32`, so
    n=8 is exactly the 1B speculator's own cost ratio, and past it the family
    is strictly worse than the status quo on the axis it exists to improve.
    That also makes `EARLY-k20-g32-L8` vs. `SPARSE-k20-g32` an equal-cost
    head-to-head, which only means anything if the partner row actually
    exists at the same keep rate and granularity -- the same invariant the
    ORACLE rows already assert."""
    early = {eid: cfg for eid, cfg in EXPERIMENTS.items() if cfg["mode"] == "early"}
    assert set(early) == (
        {f"EARLY-k20-g32-L{n}" for n in range(1, 9)}
        | {f"EARLY-k{r}-g32-L{n}" for r in (60, 80) for n in (2, 4)}
    ), sorted(early)

    for exp_id, cfg in early.items():
        assert cfg["granularity"] == "32"
        assert cfg["keep_mode"] == "keep"
        n = cfg["scorer_num_layers"]
        assert 1 <= n <= 8, f"{exp_id}: r = n/32 makes n > 8 pointless"
        assert exp_id.endswith(f"-L{n}"), (
            f"{exp_id} does not name the layer budget it actually runs"
        )
        partner = f"SPARSE-k{int(cfg['keep_percentage'] * 100)}-g32"
        assert partner in EXPERIMENTS, f"{exp_id} has no 1B-scorer partner"
        assert EXPERIMENTS[partner]["keep_percentage"] == cfg["keep_percentage"]


def test_early_row_truncates_the_target_checkpoint_on_the_sparse_loop():
    """Same wiring claim the ORACLE test makes, with one addition that is the
    whole point of the family: the scorer engine must be the TARGET
    checkpoint carrying `hf_overrides={"num_hidden_layers": n}`.

    Without the override the row still runs and still writes a
    plausible-looking score -- it would just be a duplicate ORACLE row under
    an EARLY id, i.e. a full-cost scorer recorded as a cheap one, which is
    the exact claim the family exists to test."""
    early = _run_experiment_with_stubs("EARLY-k20-g32-L2")

    assert early["proposer_kwargs"]["speculator_model_path"] == "/ckpt/Llama-3.1-8B-Instruct"
    assert early["proposer_kwargs"]["hf_overrides"] == {"num_hidden_layers": 2}
    assert early["loop"] == "sparse"
    assert early["llm_kwargs"]["worker_cls"] == (
        "vllm_patch.sparse_target_runner.SparseTargetWorker"
    )
    assert early["llm_kwargs"]["async_scheduling"] is False

    # Its own memory fraction, not the oracle's: 0.6 is documented as a full
    # 8B's ~16GB of weights plus 128 KiB/token of KV, and a 2-layer
    # truncation is neither.
    assert early["proposer_kwargs"]["gpu_memory_utilization"] == 0.3

    # The target engine is untouched by any of this -- an EARLY row differs
    # from its SPARSE partner only in who scores.
    sparse = _run_experiment_with_stubs("SPARSE-k20-g32")
    assert early["llm_kwargs"] == sparse["llm_kwargs"]
    assert "hf_overrides" not in sparse["proposer_kwargs"], (
        "a non-EARLY row must not carry a layer override at all -- passing "
        "one with the full layer count would still change the load path"
    )

    # ORACLE-k20 is the same checkpoint at full depth, i.e. this family's own
    # n=32 ceiling; the ONLY proposer difference may be the override and the
    # memory fraction.
    oracle = _run_experiment_with_stubs("ORACLE-k20")
    differing = {
        k for k in set(early["proposer_kwargs"]) | set(oracle["proposer_kwargs"])
        if early["proposer_kwargs"].get(k) != oracle["proposer_kwargs"].get(k)
    }
    assert differing == {"hf_overrides", "gpu_memory_utilization"}, sorted(differing)


def test_early_row_label_names_the_truncation():
    """`exp_id` carries the layer budget, but `label` is what sits next to
    the scorer path in all_runs.csv -- and "Llama-3.1-8B-Instruct" alone
    reads as the full 8B. Same append-to-label convention the head sets and
    the prefill scope already use, for the same reason."""
    row = _run_experiment_with_stubs("EARLY-k20-g32-L2")["row"]
    assert "first2layers" in row["label"]
    assert row["mode"] == "early", (
        "an EARLY row must not record itself as `oracle`: the oracle row is "
        "the accuracy CEILING, this one is a cheap approximation of it"
    )


def test_speculator_flops_scale_exactly_with_the_layer_budget():
    """The family's entire economic claim, stated as arithmetic the FLOP
    model has to keep reproducing: the first n layers of the target cost
    `n/32` of what all 32 do.

    This is what makes `r = n/32` true, and therefore what makes the win
    condition `(d + o)(1 - r - k) > 12r` predict anything. Checked here
    rather than trusted because nothing else would notice if
    `_speculator_flop_config` stopped reading the engine's OVERRIDDEN
    hf_config and started reading the checkpoint's own -- every FLOP column
    would just silently report the full 8B's cost under an EARLY id."""
    full = _llama31_8b_flop_cfg()
    for n in (1, 2, 4, 8):
        truncated = ModelFlopConfig(**{**full.__dict__, "num_layers": n})
        kwargs = dict(pool_len=4096, num_cached=0, look_ahead=8)
        a = speculator_turn_flops(truncated, **kwargs)
        b = speculator_turn_flops(full, **kwargs)
        # Prefill and scoring are pure per-layer work, so the ratio is exact.
        assert a.spec_prefill * 32 == b.spec_prefill * n
        assert a.spec_scoring * 32 == b.spec_scoring * n
        # Lookahead carries `lm_head` too, which is layer-count-independent,
        # so it is bounded rather than exact -- and must still shrink.
        assert a.spec_lookahead < b.spec_lookahead



# --------------------------------------------------------------------------
# Query-tail scoring source (ACCURACY_IMPROVEMENTS.md section 5a)
# --------------------------------------------------------------------------

def _tail_rows_over_a_prefill(prompt_len, tail_n, chunk):
    """Absolute positions `prompt_tail_subslice` retains across a whole
    chunked prefill, in the order it retains them."""
    got = []
    num_computed = 0
    while num_computed < prompt_len:
        num_scheduled = min(chunk, prompt_len - num_computed)
        sub = prompt_tail_subslice(num_computed, num_scheduled, prompt_len, tail_n)
        if sub is not None:
            lo, hi = sub
            assert 0 <= lo < hi <= num_scheduled, (lo, hi, num_scheduled)
            got.extend(range(num_computed + lo, num_computed + hi))
        num_computed += num_scheduled
    return got


def test_prompt_tail_subslice_reconstructs_the_tail_exactly():
    """The property the whole section-5a design rests on: concatenating what
    the capture hook retains, across however many chunked-prefill forward
    calls a request takes, yields exactly the prompt's last `tail_n`
    positions -- in order, once each, none missing and none extra.

    Swept across chunk sizes that put the boundary in every interesting place
    relative to the tail: the tail split across two calls, the tail entirely
    inside one call, calls wholly before the tail, and a chunk size that does
    not divide the prompt length."""
    for prompt_len in (1, 2, 7, 64, 1000, 118000):
        for tail_n in (1, 3, 8, 16):
            for chunk in (1, 2, 3, 7, 64, 999, 32768, 10**6):
                got = _tail_rows_over_a_prefill(prompt_len, tail_n, chunk)
                expected = list(range(max(0, prompt_len - tail_n), prompt_len))
                assert got == expected, (prompt_len, tail_n, chunk, got[:5], expected[:5])


def test_prompt_tail_subslice_excludes_generated_tokens():
    """Generated tokens sit at positions >= prompt_len, so the SAME predicate
    that selects the tail also rejects them -- no second rule, and no reliance
    on shape (a 1-token prefill chunk and a decode step are indistinguishable
    by shape, which is exactly the ambiguity `end_capture`'s docstring flags).
    """
    prompt_len = 100
    for step in range(8):  # decode steps: one token each, at 100, 101, ...
        assert prompt_tail_subslice(prompt_len + step, 1, prompt_len, 8) is None

    # ...while the final PREFILL token, also a 1-token slice, IS retained.
    assert prompt_tail_subslice(prompt_len - 1, 1, prompt_len, 8) == (0, 1)


def test_prompt_tail_subslice_degenerate_inputs():
    """Never returns an empty or inverted range, and never asks for rows a
    step does not have -- a caller slices `q[start+lo:start+hi]` with these
    directly, so an out-of-range pair would silently capture the wrong
    request's queries rather than fail."""
    # tail longer than the prompt: clamps to the whole prompt, not negative.
    assert prompt_tail_subslice(0, 5, 5, 999) == (0, 5)
    # nothing scheduled, and a nonsensical tail size: no capture, no crash.
    assert prompt_tail_subslice(0, 0, 100, 8) is None
    assert prompt_tail_subslice(0, 10, 100, 0) is None
    assert prompt_tail_subslice(0, 10, 100, -1) is None
    # a chunk entirely before the tail region.
    assert prompt_tail_subslice(0, 50, 100, 8) is None


def test_prompt_tail_subslice_under_a_real_prefix_cache_hit():
    """The dominant production shape, with the measured numbers.

    A steady-state scoring turn submits the whole candidate pool but hits the
    prefix cache for nearly all of it -- `num_cached_tokens_speculator_mean`
    was 99,320 on the EARLY-k20-g32-L8 run, against a ~61-token delta. So the
    tail arrives in ONE short forward call whose absolute positions start deep
    into the prompt, which is the case a shape-based rule gets wrong (61
    tokens is not 1, so `is_decode_query_slice` rejects it) and a
    position-based one gets right."""
    num_cached, delta = 99320, 61
    prompt_len = num_cached + delta
    sub = prompt_tail_subslice(num_cached, delta, prompt_len, 8)
    assert sub == (delta - 8, delta)
    lo, hi = sub
    assert list(range(num_cached + lo, num_cached + hi)) == list(
        range(prompt_len - 8, prompt_len)
    )


def test_prompt_tail_subslice_is_bounded_by_tail_n():
    """The memory property that makes this affordable where capturing the
    whole prefill is not. `is_decode_query_slice` exists because retaining
    prefill slices pinned 30.3GiB and OOM'd the first ORACLE-k20 run; this
    predicate retains at most `tail_n` rows per forward call regardless of
    how large that call is."""
    for chunk in (1, 32768, 118000):
        sub = prompt_tail_subslice(0, chunk, chunk, 8)
        assert sub is not None
        lo, hi = sub
        assert hi - lo <= 8


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


# ---------------------------------------------------------------------------
# Interleaved sliding-window support (Gemma 3/3n/4, Llama 4). See the porting
# plan's blockers A3 (sliding layers voting on positions they cannot attend
# to), A4 (wrong attention scale), and A5 (KV-shared layers voting twice).
# ---------------------------------------------------------------------------

# Gemma 4's larger models interleave 5 sliding layers per global one; E2B/E4B
# use 4:1. Both patterns are exercised below.
_GEMMA4_5TO1 = ["sliding_attention"] * 5 + ["full_attention"]
_GEMMA4_4TO1 = ["sliding_attention"] * 4 + ["full_attention"]


class _FakeImpl:
    def __init__(self, scale, logits_soft_cap=None):
        self.scale = scale
        self.logits_soft_cap = logits_soft_cap


class _FakeAttention:
    """Stands in for a vLLM `Attention` module. `layer_geometry_from_
    attention_layers` only ever calls `getattr`, which is what makes the
    geometry builder testable with no checkpoint and no GPU."""

    def __init__(self, scale, sliding_window=None, kv_sharing_target_layer_name=None,
                 logits_soft_cap=None):
        self.sliding_window = sliding_window
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.impl = _FakeImpl(scale, logits_soft_cap)


def test_scoring_layer_indices_global_only_keeps_just_the_full_attention_layers():
    """Blocker A3. On a 5:1 interleave, 5 of every 6 layers can never attend
    beyond the sliding window, so their score for a distant position is a
    number the model never computes in real inference -- and `max` over
    (layer, head) lets any ONE of them decide a token's importance."""
    layer_types = _GEMMA4_5TO1 * 5  # 30 layers, the 26B-A4B's depth
    got = scoring_layer_indices(30, "global_only", layer_types)
    assert got == [5, 11, 17, 23, 29]
    assert len(got) == 5, "5:1 interleave must leave 1 in 6 layers voting"

    # 4:1 (E2B/E4B) -- same policy, different density.
    four_to_one = _GEMMA4_4TO1 * 6  # 30 layers
    assert scoring_layer_indices(30, "global_only", four_to_one) == [4, 9, 14, 19, 24, 29]

    # A uniform-attention model has no sliding layers, so this degenerates to
    # "every layer" rather than becoming a special case to remember.
    assert scoring_layer_indices(4, "global_only", ["full_attention"] * 4) == [0, 1, 2, 3]
    # "attention" is the Llama-4/hybrid-SSM spelling for the same thing.
    assert scoring_layer_indices(2, "global_only", ["attention"] * 2) == [0, 1]


def test_scoring_layer_indices_global_only_refuses_to_guess():
    """Each failure mode raises rather than silently mis-selecting, because
    every one of them produces a plausible-looking score vector."""
    try:
        scoring_layer_indices(30, "global_only")
    except ValueError as exc:
        assert "layer_types" in str(exc)
    else:
        raise AssertionError("global_only without layer_types must raise")

    # A geometry built for a different (or untruncated) checkpoint.
    try:
        scoring_layer_indices(30, "global_only", _GEMMA4_5TO1)
    except ValueError as exc:
        assert "6" in str(exc) and "30" in str(exc)
    else:
        raise AssertionError("a layer_types/num_layers mismatch must raise")

    # An unrecognised type must not be silently counted as global -- that is
    # exactly the "admit a local layer's phantom vote" failure this prevents.
    try:
        scoring_layer_indices(2, "global_only", ["full_attention", "mamba"])
    except ValueError as exc:
        assert "mamba" in str(exc)
    else:
        raise AssertionError("an unknown layer type must raise, not be assumed global")

    # A model with no global layer cannot score a long context at all.
    try:
        scoring_layer_indices(3, "global_only", ["sliding_attention"] * 3)
    except ValueError as exc:
        assert "no layer" in str(exc)
    else:
        raise AssertionError("an all-sliding model must raise rather than score")


def test_kv_shared_layers_are_kept_by_default():
    """A KV-shared layer is NOT a duplicate vote, and this is the regression
    test for having once believed it was.

    `gpu_model_runner.initialize_kv_cache_tensors` aliases such a layer's
    `attn.kv_cache` to its target's tensor, so the K read back is the
    target's REAL K -- while the Q is the layer's own. The resulting
    distribution is as distinct as any two layers'. On Gemma-4-E2B, where 20
    of 35 layers are KV-shared, dropping them cut `global_only` from 7 voting
    layers to 3."""
    kv_shared = [False] * 4 + [True] * 2
    assert scoring_layer_indices(6, None, None, kv_shared) == [0, 1, 2, 3, 4, 5]

    layer_types = ["sliding_attention", "full_attention"] * 3
    assert scoring_layer_indices(6, "global_only", layer_types, kv_shared) == [1, 3, 5]


def test_scoring_layer_indices_drops_kv_shared_layers_when_asked():
    """The opt-in path, for a caller reading from a hand-built dummy cache
    that does not reproduce the engine's aliasing -- there a shared layer's
    cache is never written, so scoring it means scoring uninitialized
    memory."""
    kv_shared = [False] * 4 + [True] * 2
    assert scoring_layer_indices(6, None, None, kv_shared, True) == [0, 1, 2, 3]

    # Composes with a layer selection rather than replacing it.
    layer_types = ["sliding_attention", "full_attention"] * 3
    assert scoring_layer_indices(6, "global_only", layer_types, kv_shared, True) == [1, 3]

    # No sharing -> nothing dropped even when asked. The Llama case.
    for selection in (None, "skip_first2", "second_half", "last_quarter"):
        assert scoring_layer_indices(16, selection, None, [False] * 16, True) == (
            scoring_layer_indices(16, selection)
        )

    # Degenerate: an all-shared model keeps the pre-filter selection rather
    # than returning nothing (an empty set is a silent NaN downstream).
    assert scoring_layer_indices(3, None, None, [True] * 3, True) == [0, 1, 2]


def test_layer_geometry_from_attention_layers_reads_the_live_modules():
    """The builder reads what the KERNEL actually uses, not what the config
    says: the scale lives on the backend impl (`impl.scale`), the window and
    the KV-sharing target on the `Attention` module itself."""
    layers = [
        _FakeAttention(scale=1.0, sliding_window=1024),
        _FakeAttention(scale=1.0, sliding_window=1024),
        _FakeAttention(scale=1.0, sliding_window=None),          # global
        _FakeAttention(scale=1.0, sliding_window=1024,
                       kv_sharing_target_layer_name="layers.0.self_attn.attn"),
    ]
    geo = layer_geometry_from_attention_layers(layers)
    assert geo.layer_types == [
        "sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
    ]
    assert geo.scales == [1.0, 1.0, 1.0, 1.0]
    assert geo.kv_shared == [False, False, False, True]
    assert geo.logits_soft_cap is None
    assert not geo.is_noop()

    # A Llama-shaped model: uniform 1/sqrt(head_dim), no windows, no sharing.
    llama_scale = 128 ** -0.5
    llama_geo = layer_geometry_from_attention_layers(
        [_FakeAttention(scale=llama_scale) for _ in range(16)]
    )
    assert llama_geo.layer_types == ["full_attention"] * 16
    assert llama_geo.kv_shared == [False] * 16
    assert all(abs(s - llama_scale) < 1e-12 for s in llama_geo.scales)

    # Softcapping is collected once, and disagreement raises rather than
    # silently picking one -- LayerGeometry can only carry a single scalar.
    capped = layer_geometry_from_attention_layers(
        [_FakeAttention(scale=1.0, logits_soft_cap=50.0) for _ in range(3)]
    )
    assert capped.logits_soft_cap == 50.0
    try:
        layer_geometry_from_attention_layers([
            _FakeAttention(scale=1.0, logits_soft_cap=50.0),
            _FakeAttention(scale=1.0, logits_soft_cap=30.0),
        ])
    except ValueError as exc:
        assert "logits_soft_cap" in str(exc)
    else:
        raise AssertionError("disagreeing softcaps must raise")


def test_layer_geometry_refuses_to_guess_a_missing_scale():
    """Falling back to 1/sqrt(head_dim) is right for Llama and wrong for
    Gemma 4 (`scaling = 1.0`), and the difference is a softmax TEMPERATURE,
    not a constant factor -- so an unreadable scale must fail loudly."""
    class _NoScale:
        sliding_window = None
        kv_sharing_target_layer_name = None
        impl = None

    try:
        layer_geometry_from_attention_layers([_NoScale()])
    except ValueError as exc:
        assert "scale" in str(exc)
    else:
        raise AssertionError("a missing attention scale must raise")


def test_compute_attention_score_uses_the_layers_own_scale():
    """Blocker A4. `1/sqrt(head_dim)` is the model's scale only by
    coincidence. With a per-layer scale supplied, the raw logits must be
    scaled by exactly that."""
    num_layers, look_ahead, ctx, heads, head_dim = 2, 3, 7, 4, 8
    query_buffer, key_buffer = _synthetic_qk(
        num_layers=num_layers, num_samples=1, look_ahead=look_ahead,
        num_heads=heads, num_kv_heads=heads, head_dim=head_dim, ctx_len=ctx,
    )

    default = compute_attention_score(query_buffer, key_buffer, [look_ahead])[0]
    # Gemma 4's scale: 1.0, not 1/sqrt(d).
    ones = LayerGeometry(scales=[1.0] * num_layers)
    got = compute_attention_score(query_buffer, key_buffer, [look_ahead], ones)[0]
    assert torch.allclose(got, default * math.sqrt(head_dim), atol=1e-3, rtol=1e-3)

    # Per-layer scales really are per-layer: a model whose head_dim differs by
    # layer type gets a DIFFERENT softmax temperature per type, which is the
    # bias this fixes.
    mixed = LayerGeometry(scales=[1.0, 0.25])
    got = compute_attention_score(query_buffer, key_buffer, [look_ahead], mixed)[0]
    assert torch.allclose(got[0], default[0] * math.sqrt(head_dim), atol=1e-3, rtol=1e-3)
    assert torch.allclose(got[1], default[1] * math.sqrt(head_dim) * 0.25,
                          atol=1e-3, rtol=1e-3)

    # A geometry built for a different layer count must raise, not broadcast.
    try:
        compute_attention_score(query_buffer, key_buffer, [look_ahead],
                                LayerGeometry(scales=[1.0]))
    except ValueError as exc:
        assert "scales" in str(exc)
    else:
        raise AssertionError("a scales/layer-count mismatch must raise")


def test_compute_attention_score_applies_logit_softcapping():
    """Gemma's `attn_logit_softcapping` is applied by the attention kernel
    before its own softmax; omitting it here leaves the scoring softmax
    reading tails the model itself never sees."""
    num_layers, look_ahead, ctx, heads, head_dim = 1, 2, 5, 2, 4
    query_buffer, key_buffer = _synthetic_qk(
        num_layers=num_layers, num_samples=1, look_ahead=look_ahead,
        num_heads=heads, num_kv_heads=heads, head_dim=head_dim, ctx_len=ctx,
    )
    cap = 3.0
    uncapped = compute_attention_score(
        query_buffer, key_buffer, [look_ahead], LayerGeometry(scales=[1.0])
    )[0]
    capped = compute_attention_score(
        query_buffer, key_buffer, [look_ahead],
        LayerGeometry(scales=[1.0], logits_soft_cap=cap),
    )[0]
    assert torch.allclose(capped, cap * torch.tanh(uncapped.float() / cap).to(capped.dtype),
                          atol=1e-3, rtol=1e-3)
    assert capped.abs().max().item() <= cap + 1e-3


def test_layer_geometry_is_a_provable_noop_for_uniform_models():
    """The whole point of defaulting `geometry` to None -- and of a Llama-
    shaped geometry being all-`full_attention`, all-False, no softcap -- is
    that every already-published row is reproduced identically rather than
    'probably unchanged'."""
    num_layers, look_ahead, ctx, heads, head_dim = 4, 3, 11, 4, 16
    query_buffer, key_buffer = _synthetic_qk(
        num_layers=num_layers, num_samples=1, look_ahead=look_ahead,
        num_heads=heads, num_kv_heads=heads, head_dim=head_dim, ctx_len=ctx,
    )
    cfg = SpecConfig(keep_strategy="percentage",
                     keep_kwargs={"percentage": 0.5, "chunk": True, "chunk_size": 2},
                     pool_kernel_size=3)
    per_layer_keys = [k[0] for k in key_buffer]

    reference = score_and_select_indices(query_buffer, per_layer_keys, look_ahead, cfg)

    # An all-None geometry is declared a no-op and must behave as one.
    empty = LayerGeometry()
    assert empty.is_noop()
    assert score_and_select_indices(
        query_buffer, per_layer_keys, look_ahead, cfg, empty
    ) == reference

    # And so must the geometry a real uniform-attention model produces.
    llama_geo = layer_geometry_from_attention_layers(
        [_FakeAttention(scale=head_dim ** -0.5) for _ in range(num_layers)]
    )
    assert score_and_select_indices(
        query_buffer, per_layer_keys, look_ahead, cfg, llama_geo
    ) == reference


def test_score_head_set_refuses_a_geometry_that_renumbers_the_head_axis():
    """`score_head_set` indexes the FULL flattened layer*head axis. A geometry
    that drops layers (kv_shared) renumbers that axis WITHOUT score_layers
    being set, so `SpecConfig.__post_init__`'s existing mutual-exclusion check
    cannot catch it. Scoring with a silently different head set would be
    unfalsifiable in the results."""
    num_layers, look_ahead, ctx, heads, head_dim = 4, 2, 9, 2, 8
    query_buffer, key_buffer = _synthetic_qk(
        num_layers=num_layers, num_samples=1, look_ahead=look_ahead,
        num_heads=heads, num_kv_heads=heads, head_dim=head_dim, ctx_len=ctx,
    )
    attn = compute_attention_score(query_buffer, key_buffer, [look_ahead])
    cfg = SpecConfig(keep_strategy="percentage",
                     keep_kwargs={"percentage": 0.5}, score_head_set=[0, 3])

    cfg.drop_kv_shared_layers = True
    geo = LayerGeometry(kv_shared=[False, False, False, True])
    try:
        aggregate_attention_score(attn, cfg, geo)
    except ValueError as exc:
        assert "score_head_set" in str(exc)
    else:
        raise AssertionError("a layer-dropping geometry must not silently renumber heads")

    # Without a dropping geometry the same config is still fine.
    assert aggregate_attention_score(attn, cfg, LayerGeometry())[0].shape[0] == ctx


def test_global_only_is_accepted_by_the_config_surface():
    """A scoring mode the experiment matrix cannot name is a mode that cannot
    be run -- `SpecConfig.__post_init__` validates against a closed set."""
    cfg = SpecConfig(keep_strategy="percentage", keep_kwargs={"percentage": 0.2},
                     score_layers="global_only")
    assert cfg.score_layers == "global_only"


def test_global_only_end_to_end_ignores_a_sliding_layers_phantom_votes():
    """The property the whole `global_only` mode exists for, exercised through
    the real scoring pipeline rather than the index policy alone.

    Layer 0 is sliding and is given a saturated score on an early context
    position it could never actually attend to; layer 1 is global and has a
    milder, genuine preference for a late position. Under the default
    all-layer `max` the sliding layer's phantom vote wins outright. Under
    `global_only` that layer is not in the tensor at all.

    `pool_kernel_size=0` so the smoothing pass cannot blur the two peaks into
    each other -- this test is about WHICH LAYER decides, not about pooling.
    """
    look_ahead, ctx, heads, head_dim = 1, 16, 1, 4
    query_buffer, key_buffer = _synthetic_qk(
        num_layers=2, num_samples=1, look_ahead=look_ahead,
        num_heads=heads, num_kv_heads=heads, head_dim=head_dim, ctx_len=ctx,
    )
    attn = compute_attention_score(query_buffer, key_buffer, [look_ahead])[0].clone()
    attn.zero_()
    attn[0, :, :, 1] = 50.0    # sliding layer, saturated, on a position it
                               # can never reach in real inference
    attn[1, :, :, 14] = 2.0    # global layer, a real but milder preference

    # ceil(16 * 0.0625) == 1: exactly one position survives, so the assertion
    # is "who won", with no room for both to be kept.
    keep_one = {"percentage": 0.0625}
    geo = LayerGeometry(layer_types=["sliding_attention", "full_attention"])

    all_cfg = SpecConfig(keep_strategy="percentage", keep_kwargs=keep_one,
                         pool_kernel_size=0)
    all_layers = chunk_select_from_smoothed_attention(
        aggregate_attention_score([attn], all_cfg), all_cfg
    )[0].tolist()
    assert all_layers == [1], f"expected the phantom vote to win, got {all_layers}"

    global_cfg = SpecConfig(keep_strategy="percentage", keep_kwargs=keep_one,
                            pool_kernel_size=0, score_layers="global_only")
    global_only = chunk_select_from_smoothed_attention(
        aggregate_attention_score([attn], global_cfg, geo), global_cfg
    )[0].tolist()
    assert global_only == [14], f"expected the global layer to decide, got {global_only}"


# ---------------------------------------------------------------------------
# Locating the decoder attention modules (`vllm_patch/model_structure.py`).
# The wrapper shape differs per architecture and the failure mode is quiet:
# a wrong walk hooks nothing, and the first symptom is an empty query buffer
# scoring as NaN several steps later.
# ---------------------------------------------------------------------------

from vllm_patch.model_structure import find_attention_modules, unwrap_text_stack


class _Bag:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _StubAttention(_Bag):
    """Named so `find_attention_modules`' MLA guard sees a non-MLA class."""


def _stub_layers(n, attn_cls=_StubAttention):
    return [_Bag(self_attn=_Bag(attn=attn_cls(head_size=128, num_kv_heads=8)))
            for _ in range(n)]


def test_unwrap_text_stack_handles_both_wrapper_shapes():
    """Llama loads as `LlamaForCausalLM` (one `.model` hop). Gemma 4 loads as
    `Gemma4ForConditionalGeneration`, whose text stack is a whole
    `Gemma4ForCausalLM` under `.language_model` -- confirmed by the sibling
    single-turn pipeline on real hardware, and the reason a `.model`-only walk
    finds nothing on that checkpoint."""
    llama = _Bag(model=_Bag(layers=_stub_layers(4)))
    assert unwrap_text_stack(llama) is llama.model

    gemma4_mm = _Bag(language_model=_Bag(model=_Bag(layers=_stub_layers(6))))
    assert unwrap_text_stack(gemma4_mm) is gemma4_mm.language_model.model

    # Already the text stack: no hop at all.
    bare = _Bag(layers=_stub_layers(2))
    assert unwrap_text_stack(bare) is bare


def test_unwrap_text_stack_raises_on_an_unrecognised_shape():
    """Silently hooking zero layers is the failure this prevents."""
    try:
        unwrap_text_stack(_Bag(encoder=_Bag(blocks=[])))
    except NotImplementedError as exc:
        assert "layers" in str(exc)
    else:
        raise AssertionError("an unrecognised model shape must raise")


def test_find_attention_modules_returns_the_inner_attention_in_layer_order():
    """It must return the vLLM `Attention` (`layer.self_attn.attn`), not the
    model's own attention wrapper -- that is what lets the capture hook read
    the post-RoPE query as an ARGUMENT instead of recomputing it, and what
    makes per-layer `head_size`/`num_kv_heads` available for a model with
    heterogeneous head dims across layer types."""
    model = _Bag(model=_Bag(layers=_stub_layers(3)))
    found = find_attention_modules(model)
    assert len(found) == 3
    assert [id(a) for a in found] == [id(l.self_attn.attn) for l in model.model.layers]
    assert all(hasattr(a, "head_size") and hasattr(a, "num_kv_heads") for a in found)


def test_find_attention_modules_rejects_shapes_it_cannot_read():
    class _MLAAttention(_Bag):
        pass

    # MLA caches a compressed latent, not K/V, so the K read-back would not be
    # reading keys at all.
    mla = _Bag(model=_Bag(layers=_stub_layers(2, attn_cls=_MLAAttention)))
    try:
        find_attention_modules(mla)
    except NotImplementedError as exc:
        assert "MLA" in str(exc)
    else:
        raise AssertionError("an MLA stack must raise rather than be read as K/V")

    # A layer with no `.self_attn.attn`.
    odd = _Bag(model=_Bag(layers=[_Bag(mlp=_Bag())]))
    try:
        find_attention_modules(odd)
    except NotImplementedError as exc:
        assert "self_attn" in str(exc)
    else:
        raise AssertionError("a layer without an Attention module must raise")

    # An empty stack would produce an empty query buffer, i.e. a NaN score.
    try:
        find_attention_modules(_Bag(model=_Bag(layers=[])))
    except NotImplementedError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("an empty decoder stack must raise")


# ---------------------------------------------------------------------------
# Cross-layer KV sharing + a single (UniformType) KV cache group. vLLM appends
# a KV-sharing layer to its target group's `layer_names` but not to the group
# spec's `kv_cache_specs`, so the stock runner KeyErrors on it. Confirmed on
# real hardware with Gemma-4-E2B (layer 15).
# ---------------------------------------------------------------------------

from vllm_patch.kv_cache_utils import shared_layer_specs_to_backfill


def test_shared_layer_specs_to_backfill_borrows_the_targets_spec():
    """Gemma 4 shares KV in its last `num_kv_shared_layers` layers, each
    reusing the last non-shared layer of the SAME attention type -- so a
    shared layer's correct spec is its target's, not a guess."""
    specs = ["layers.0.self_attn.attn", "layers.1.self_attn.attn"]
    layer_names = specs + ["layers.2.self_attn.attn", "layers.3.self_attn.attn"]
    shared = {
        "layers.2.self_attn.attn": "layers.0.self_attn.attn",
        "layers.3.self_attn.attn": "layers.1.self_attn.attn",
    }
    assert shared_layer_specs_to_backfill(layer_names, specs, shared) == shared


def test_shared_layer_specs_to_backfill_is_a_noop_without_sharing():
    """The Llama case: every layer owns its spec, nothing to backfill. This is
    what keeps the override off the published rows' path."""
    specs = [f"layers.{i}.self_attn.attn" for i in range(16)]
    assert shared_layer_specs_to_backfill(specs, specs, {}) == {}


def test_shared_layer_specs_to_backfill_does_not_mask_other_missing_layers():
    """A layer missing for a reason OTHER than KV sharing must be left alone,
    so the stock KeyError still fires. Silently inventing a spec for it would
    turn a loud startup failure into a wrong cache layout at runtime."""
    specs = ["layers.0.self_attn.attn"]
    layer_names = specs + ["layers.9.self_attn.attn"]

    # Not in the sharing map at all.
    assert shared_layer_specs_to_backfill(layer_names, specs, {}) == {}

    # In the map, but pointing at a target that has no spec either.
    dangling = {"layers.9.self_attn.attn": "layers.8.self_attn.attn"}
    assert shared_layer_specs_to_backfill(layer_names, specs, dangling) == {}


# ---------------------------------------------------------------------------
# The sliding-window gate (diagnose_sliding_window_votes.py). The measurement
# is "what share of winning (layer, head) votes came from a layer that could
# never have attended to the position it won".
# ---------------------------------------------------------------------------

from vllm_patch.scoring import phantom_vote_counts


def test_phantom_vote_counts_only_counts_out_of_window_sliding_wins():
    """A vote is phantom only if BOTH hold: the winner is a sliding layer,
    and the position it won is further away than that layer's own window."""
    # 2 layers: 0 sliding (window 4), 1 full attention.
    layer_windows = [4, None]
    # 1 lookahead step whose query sits at position 10; 6 context positions.
    #                     pos: 0  1  2  3  4  5
    winning_layers = [[0, 1, 0, 1, 0, 0]]
    query_positions = [10]

    # Distances from the query: 10, 9, 8, 7, 6, 5 -- all beyond the window 4.
    # So every position won by layer 0 is phantom; layer 1 never is.
    phantom, total, _null = phantom_vote_counts(
        winning_layers, layer_windows, [0, 1, 2, 3, 4, 5], query_positions
    )
    assert (phantom, total) == (4, 6)

    # A full-attention winner is never phantom, however far away.
    phantom, total, _null = phantom_vote_counts(
        winning_layers, layer_windows, [1, 3], query_positions
    )
    assert (phantom, total) == (0, 2)

    # Inside the window is not phantom: position 7 is distance 3 <= 4.
    inside = [[0, 0]]
    phantom, total, _null = phantom_vote_counts(inside, layer_windows, [0, 1], [3])
    assert (phantom, total) == (0, 2)


def test_phantom_vote_counts_is_zero_for_a_uniform_attention_model():
    """The Llama case, and the reason the comparison sample matters: with no
    sliding layer there is nothing to attribute, so a 0% rate says nothing
    about selection quality."""
    layer_windows = [None] * 4
    winning_layers = [[0, 1, 2, 3], [3, 2, 1, 0]]
    phantom, total, _null = phantom_vote_counts(
        winning_layers, layer_windows, [0, 1, 2, 3], [100, 101]
    )
    assert phantom == 0
    assert total == 8


def test_phantom_vote_counts_requires_one_query_position_per_step():
    """Each lookahead step's query sits at a different absolute position, so a
    single global one would mismeasure every step but the first."""
    try:
        phantom_vote_counts([[0], [0]], [4], [0], [10])
    except ValueError as exc:
        assert "query position" in str(exc)
    else:
        raise AssertionError("a query-position/step mismatch must raise")


def test_winning_layers_matches_production_and_maps_through_layer_restriction():
    """The gate's whole credibility rests on measuring what production
    computed, so the winner collection must not perturb the importance vector
    it comes from -- and must report ORIGINAL layer indices even when
    `global_only` has restricted the tensor."""
    num_layers, look_ahead, ctx, heads, head_dim = 6, 2, 12, 2, 8
    query_buffer, key_buffer = _synthetic_qk(
        num_layers=num_layers, num_samples=1, look_ahead=look_ahead,
        num_heads=heads, num_kv_heads=heads, head_dim=head_dim, ctx_len=ctx,
    )
    attn = compute_attention_score(query_buffer, key_buffer, [look_ahead])
    cfg = SpecConfig(keep_strategy="percentage", keep_kwargs={"percentage": 0.5},
                     pool_kernel_size=3)

    reference = aggregate_attention_score(attn, cfg)[0]
    winners = []
    with_winners = aggregate_attention_score(attn, cfg, None, winning_layers=winners)[0]
    assert torch.equal(with_winners, reference), "collecting winners changed the score"
    assert winners[0].shape == (look_ahead, ctx)
    assert int(winners[0].max()) < num_layers

    # Under global_only, only layers 1 and 3 and 5 vote -- and the reported
    # indices must be those ORIGINAL numbers, not 0/1/2 within the subset.
    layer_types = ["sliding_attention", "full_attention"] * 3
    geo = LayerGeometry(layer_types=layer_types)
    gcfg = SpecConfig(keep_strategy="percentage", keep_kwargs={"percentage": 0.5},
                      pool_kernel_size=3, score_layers="global_only")
    gwinners = []
    aggregate_attention_score(attn, gcfg, geo, winning_layers=gwinners)
    assert set(gwinners[0].flatten().tolist()) <= {1, 3, 5}


def test_winning_layers_refuses_aggregations_that_have_no_argmax():
    """`mean`/`zmean` have no winner, and `score_head_set` renumbers the axis
    the winner maps through -- both must raise rather than report a number
    that looks plausible and means something else."""
    query_buffer, key_buffer = _synthetic_qk(
        num_layers=2, num_samples=1, look_ahead=1,
        num_heads=2, num_kv_heads=2, head_dim=4, ctx_len=6,
    )
    attn = compute_attention_score(query_buffer, key_buffer, [1])

    for aggregation in ("mean", "zmean"):
        cfg = SpecConfig(keep_strategy="percentage", keep_kwargs={"percentage": 0.5},
                         score_aggregation=aggregation)
        try:
            aggregate_attention_score(attn, cfg, None, winning_layers=[])
        except ValueError as exc:
            assert "max" in str(exc)
        else:
            raise AssertionError(f"{aggregation} must refuse to report winners")

    cfg = SpecConfig(keep_strategy="percentage", keep_kwargs={"percentage": 0.5},
                     score_head_set=[0, 1])
    try:
        aggregate_attention_score(attn, cfg, None, winning_layers=[])
    except ValueError as exc:
        assert "score_head_set" in str(exc)
    else:
        raise AssertionError("score_head_set must refuse to report winners")


def test_phantom_vote_counts_reports_the_random_winner_null():
    """The headline rate is uninterpretable without it. On a model that is
    mostly sliding layers, most wins land on a sliding layer by composition
    alone -- so a high raw rate can be exactly what "layer choice carries no
    signal" looks like. The null makes the excess visible."""
    # 4 layers, 3 sliding (window 4), 1 full attention.
    layer_windows = [4, 4, 4, None]
    query_positions = [100]

    # Every position is far outside the window, so any of the 3 sliding
    # layers would be a phantom win: null = 3/4 per vote.
    winning = [[0, 1, 2, 3]]
    phantom, total, null = phantom_vote_counts(
        winning, layer_windows, [0, 1, 2, 3], query_positions
    )
    assert (phantom, total) == (3, 4)
    assert abs(null - 3.0) < 1e-9      # 4 votes x 3/4

    # Inside every window, nothing COULD be phantom, so the null is 0 too --
    # the null tracks DISTANCE, not just layer composition. (The row must
    # span the context, since it is indexed by position.)
    near_row = [[0] * 100]
    phantom, total, null = phantom_vote_counts(
        near_row, layer_windows, [98, 99], [100]
    )
    assert (phantom, total) == (0, 2), "distance 1-2 is inside the window 4"
    assert null == 0.0

    # An all-sliding model has a null of 1.0 per far vote: a 100% rate there
    # is not evidence of anything.
    phantom, total, null = phantom_vote_counts(
        [[0, 1]], [4, 4], [0, 1], [100]
    )  # both positions ~100 away, both layers sliding
    assert (phantom, total) == (2, 2)
    assert abs(null - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Window masking: the third scoring mode, between "score sliding layers over
# the whole context" (default, measurably corrupted) and "drop them entirely"
# (global_only).
# ---------------------------------------------------------------------------


def test_mask_to_window_silences_a_sliding_layer_outside_its_window():
    """A masked position gets -inf before the softmax, so that layer's mass
    there is 0 and it cannot win the `max`. Inside the window the layer is
    untouched -- that is the whole point of masking over dropping."""
    look_ahead, ctx, heads, head_dim = 1, 20, 1, 4
    query_buffer, key_buffer = _synthetic_qk(
        num_layers=2, num_samples=1, look_ahead=look_ahead,
        num_heads=heads, num_kv_heads=heads, head_dim=head_dim, ctx_len=ctx,
    )
    # Layer 0 sliding with a window of 5; layer 1 full attention.
    geo = LayerGeometry(scales=[1.0, 1.0], sliding_windows=[5, None])

    masked = compute_attention_score(
        query_buffer, key_buffer, [look_ahead], geo, mask_to_window=True
    )[0]

    # Lookahead step 0 sits at position ctx, so context position i is
    # (ctx - i) behind it. Window 5 keeps only i >= ctx - 5.
    assert torch.isinf(masked[0, :, 0, : ctx - 5]).all(), "far positions must be -inf"
    assert torch.isfinite(masked[0, :, 0, ctx - 5:]).all(), "near positions must survive"
    # The full-attention layer is never masked.
    assert torch.isfinite(masked[1]).all()

    # Unmasked, nothing is -inf -- the default path is unchanged.
    unmasked = compute_attention_score(query_buffer, key_buffer, [look_ahead], geo)[0]
    assert torch.isfinite(unmasked).all()


def test_mask_to_window_leaves_a_full_attention_model_untouched():
    """No sliding layer means nothing to mask, so this must be a provable
    no-op rather than a mode that quietly perturbs uniform models."""
    query_buffer, key_buffer = _synthetic_qk(
        num_layers=3, num_samples=1, look_ahead=2,
        num_heads=2, num_kv_heads=2, head_dim=8, ctx_len=16,
    )
    geo = LayerGeometry(scales=[1.0] * 3, sliding_windows=[None] * 3)
    plain = compute_attention_score(query_buffer, key_buffer, [2], geo)[0]
    masked = compute_attention_score(
        query_buffer, key_buffer, [2], geo, mask_to_window=True
    )[0]
    assert torch.equal(plain, masked)


def test_mask_to_window_requires_the_windows_it_masks_by():
    """Silently not masking would look identical to masking on a model whose
    layers all fit their window -- so a missing geometry must raise."""
    query_buffer, key_buffer = _synthetic_qk(
        num_layers=1, num_samples=1, look_ahead=1,
        num_heads=1, num_kv_heads=1, head_dim=4, ctx_len=8,
    )
    try:
        compute_attention_score(query_buffer, key_buffer, [1], None, mask_to_window=True)
    except ValueError as exc:
        assert "sliding_windows" in str(exc)
    else:
        raise AssertionError("mask_to_window without windows must raise")


def test_masked_sliding_layer_cannot_win_a_far_vote():
    """The end-to-end property, and the reason this is an alternative to
    global_only: after masking, a sliding layer's score at range is zero
    post-softmax, so the winner at a far position is always a full-attention
    layer -- exactly what global_only achieves by deletion instead."""
    look_ahead, ctx = 1, 30
    query_buffer, key_buffer = _synthetic_qk(
        num_layers=2, num_samples=1, look_ahead=look_ahead,
        num_heads=1, num_kv_heads=1, head_dim=4, ctx_len=ctx,
    )
    geo = LayerGeometry(scales=[1.0, 1.0], sliding_windows=[5, None])
    cfg = SpecConfig(keep_strategy="percentage", keep_kwargs={"percentage": 0.5},
                     pool_kernel_size=0)

    attn = compute_attention_score(
        query_buffer, key_buffer, [look_ahead], geo, mask_to_window=True
    )
    winners = []
    aggregate_attention_score(attn, cfg, geo, winning_layers=winners)
    far = winners[0][0, : ctx - 5].tolist()
    assert set(far) == {1}, f"a masked sliding layer won a far vote: {set(far)}"


def test_score_mode_comparison_rows_differ_only_in_how_they_score():
    """The three-way comparison is only readable if the rows are identical
    apart from the scoring mode -- same path, same keep rate, same
    granularity. A control row that differs in two things measures neither."""
    from predict_scbench import EXPERIMENTS, SCORE_MODE_PROBE

    rate, gran = SCORE_MODE_PROBE
    prefix = f"SPARSE-k{int(rate * 100)}-g{gran}-"
    rows = {
        k: v for k, v in EXPERIMENTS.items()
        if k.startswith(prefix) and k.rsplit("-", 1)[-1] in
        ("unmasked", "global", "masked")
    }
    assert set(rows) == {prefix + s for s in ("unmasked", "global", "masked")}

    for cfg in rows.values():
        # The SPARSE architecture -- this pipeline's actual contribution, and
        # what every published row uses. Grading the three scorers on the
        # simpler physical-pruning path would have measured them on a
        # substrate the project does not otherwise report.
        assert cfg["mode"] == "sparse"
        assert cfg["keep_percentage"] == rate
        assert cfg["granularity"] == gran
        assert cfg["keep_mode"] == "keep"

    # The control must carry NO scoring flags: it is the default every
    # published row used, measured here rather than quoted from history.
    control = rows[prefix + "unmasked"]
    assert control.get("score_layers") is None
    assert control.get("mask_sliding_window", False) is False

    # And the two fixes must differ from it in exactly one way each.
    assert rows[prefix + "global"]["score_layers"] == "global_only"
    assert rows[prefix + "global"].get("mask_sliding_window", False) is False
    assert rows[prefix + "masked"]["mask_sliding_window"] is True
    assert rows[prefix + "masked"].get("score_layers") is None


def test_mask_sliding_window_reaches_the_speculator_config():
    """The flag has to survive the driver -> SpecConfig hop, since scoring
    runs in the speculator's own process and reads only what is passed."""
    cfg = SpecConfig(keep_strategy="percentage", keep_kwargs={"percentage": 0.3},
                     mask_sliding_window=True)
    assert cfg.mask_sliding_window is True
    # Default off, so nothing already measured changes underneath.
    assert SpecConfig(keep_strategy="percentage",
                      keep_kwargs={"percentage": 0.3}).mask_sliding_window is False


def test_sliding_window_layers_are_excluded_from_the_gather():
    """Correctness, not optimisation: a gather compacts the KV view, and a
    sliding-window kernel reads window membership from a key's index within
    `seqused_k` -- so a compacted view masks the wrong keys entirely."""
    from vllm_patch.model_structure import gatherable_layer_names

    class _Attn:
        def __init__(self, window):
            self.sliding_window = window

    # Gemma-4-style 4:1 interleave.
    layers = {f"l{i}": _Attn(512 if i % 5 != 4 else None) for i in range(10)}
    assert gatherable_layer_names(layers, layers) == {"l4", "l9"}

    # Uniform attention (Llama): every layer gatherable, a provable no-op.
    uniform = {f"l{i}": _Attn(None) for i in range(4)}
    assert gatherable_layer_names(uniform, uniform) == set(uniform)

    # A layer absent from the registry is treated as gatherable -- that is
    # the no-lookup-needed case, and it must not silently disable the gather.
    assert gatherable_layer_names(["unknown"], {}) == {"unknown"}

    # All-sliding leaves nothing to restrict; the runner turns this into a
    # loud failure rather than a silent no-op.
    all_sliding = {f"l{i}": _Attn(512) for i in range(3)}
    assert gatherable_layer_names(all_sliding, all_sliding) == set()


def test_scorer_placement_rejects_a_collision_with_the_targets_tp_ranks():
    """The failure this prevents is invisible in its own traceback: every
    engine calls its own device "GPU 0", so a mid-run activation OOM cannot
    say which engine or which card. Before TP, index 0 was the only possible
    collision and an existence check sufficed; with TP it is every index
    below the rank count."""
    from predict_scbench import scorer_placement_error

    from predict_scbench import scorer_placement_warning

    # Sharing is a WARNING, not an error: the SPARSE rows' small speculator
    # has always been allowed to share the target's card. Only the oracle
    # case is a hard error, checked separately, because there the arithmetic
    # is provably impossible rather than merely tight.
    assert scorer_placement_error(0, 1, 2, None) is None
    assert scorer_placement_warning(0, 1)
    assert scorer_placement_warning(1, 1) is None

    # TP=2: ranks own devices 0 AND 1, so cuda:1 now shares a card too --
    # the case every pre-TP instinct misses, since index 0 used to be the
    # only collision.
    assert scorer_placement_warning(1, 2)
    assert scorer_placement_warning(2, 2) is None
    assert "cuda:2" in scorer_placement_warning(1, 2)

    # A scorer index that does not exist is still a hard error, and says so.
    assert "does not exist" in scorer_placement_error(5, 1, 2, None)

    # More TP ranks than GPUs.
    assert "exceeds" in scorer_placement_error(3, 4, 2, None)

    # No CUDA in this process (CPU dry run): not this function's business.
    assert scorer_placement_error(1, 1, 0, None) is None
    # No scorer device requested at all.
    assert scorer_placement_error(None, 2, 4, None) is None


def test_native_context_length_reads_the_text_config_not_the_wrapper():
    """A natively multimodal checkpoint has no `max_position_embeddings` on
    its top-level config at all -- Gemma 4 raised
    `AttributeError: 'Gemma4Config' object has no attribute
    'max_position_embeddings'` mid-run, a message that says nothing about the
    text/wrapper split behind it. Several scripts clamp their context budgets
    against this value, so it is read in one place now."""
    import vllm_patch.model_structure as ms

    class _TextConfig:
        max_position_embeddings = 131072

    class _Wrapper:
        """No max_position_embeddings of its own -- the Gemma 4 shape."""
        def get_text_config(self):
            return _TextConfig()

    class _TextOnly:
        """Llama: get_text_config() returns self, so this is a no-op there."""
        max_position_embeddings = 8192

        def get_text_config(self):
            return self

    class _Ancient:
        """A config predating get_text_config() must still work."""
        max_position_embeddings = 4096

    original = ms.__dict__.get("AutoConfig")
    try:
        for stub, expected in ((_Wrapper(), 131072), (_TextOnly(), 8192),
                               (_Ancient(), 4096)):
            fake = type("AutoConfig", (), {
                "from_pretrained": staticmethod(lambda *a, **k: stub)})
            import transformers
            saved = transformers.AutoConfig
            transformers.AutoConfig = fake
            try:
                assert ms.native_context_length("ignored") == expected
            finally:
                transformers.AutoConfig = saved
    finally:
        if original is not None:
            ms.AutoConfig = original

    # A config with neither attribute returns None rather than raising --
    # every caller already guards `is not None` before clamping.
    class _Nothing:
        def get_text_config(self):
            return self

    import transformers
    saved = transformers.AutoConfig
    transformers.AutoConfig = type("AutoConfig", (), {
        "from_pretrained": staticmethod(lambda *a, **k: _Nothing())})
    try:
        assert ms.native_context_length("ignored") is None
    finally:
        transformers.AutoConfig = saved


def test_load_tokenizer_only_overrides_a_malformed_extra_special_tokens():
    """Gemma 4 declares `extra_special_tokens` as a LIST, which transformers
    5.14.x rejects with `AttributeError: 'list' object has no attribute
    'keys'` -- a message that never mentions the tokenizer config behind it.
    The override must fire ONLY for that malformed shape, so a well-formed
    checkpoint keeps the stock path."""
    import json
    import tempfile

    import transformers
    from vllm_patch.model_structure import load_tokenizer

    captured = {}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return "tokenizer"

    saved = transformers.AutoTokenizer
    transformers.AutoTokenizer = _FakeAutoTokenizer
    try:
        def _load(config):
            with tempfile.TemporaryDirectory() as d:
                if config is not None:
                    with open(f"{d}/tokenizer_config.json", "w", encoding="utf-8") as f:
                        json.dump(config, f)
                return load_tokenizer(d, trust_remote_code=True)

        # The Gemma 4 shape: overridden, and the caller's kwargs survive.
        _load({"extra_special_tokens": ["<|video|>"]})
        assert captured.get("extra_special_tokens") == {}
        assert captured.get("trust_remote_code") is True

        # Well-formed mapping: left alone, so the checkpoint's own aliases
        # are still registered.
        _load({"extra_special_tokens": {"video_token": "<|video|>"}})
        assert "extra_special_tokens" not in captured

        # Absent, empty list, and no config at all: nothing to override.
        _load({})
        assert "extra_special_tokens" not in captured
        _load({"extra_special_tokens": []})
        assert "extra_special_tokens" not in captured
        _load(None)
        assert "extra_special_tokens" not in captured
    finally:
        transformers.AutoTokenizer = saved


def _fake_group(block_size, layer_names):
    return type("Group", (), {
        "kv_cache_spec": type("Spec", (), {"block_size": block_size})(),
        "layer_names": list(layer_names),
    })()


def test_gather_resolves_block_size_from_the_gathered_layers_own_group():
    """With the hybrid KV cache manager enabled, an interleaved model has a
    group per attention type -- and their block sizes can DIFFER, because
    `unify_kv_cache_spec_page_size` equalises page sizes by multiplying the
    smaller-page layers' block_size. The gather is block-granular, so taking
    `cache_config.block_size` instead of the gathered group's own value would
    silently select different tokens than the row name claims."""
    runner_module = _load_sparse_target_runner()
    runner = object.__new__(runner_module.SparseTargetGPUModelRunner)

    attn_metadata = {f"l{i}": object() for i in range(6)}
    # l1, l3, l5 are full attention (gatherable) at block_size 16; the
    # sliding layers sit in their own group whose block_size was bumped.
    runner._gatherable_layer_names_cache = {"l1", "l3", "l5"}
    runner.kv_cache_config = type("KVCacheConfig", (), {"kv_cache_groups": [
        _fake_group(32, ["l0", "l2", "l4"]),
        _fake_group(16, ["l1", "l3", "l5"]),
    ]})()
    cache_config = type("CacheConfig", (), {"block_size": 16})()
    runner.vllm_config = type("VllmConfig", (), {"cache_config": cache_config})()

    assert runner._gatherable_group_block_size(attn_metadata) == 16, (
        "must take the GATHERED group's block size, not the sliding group's"
    )
    # Cached: layer types cannot change after load.
    runner.kv_cache_config = None
    assert runner._gatherable_group_block_size(attn_metadata) == 16


def test_gather_refuses_when_the_gathered_layers_span_groups():
    """Writing one group's block ids into another group's metadata reads
    unrelated physical memory with nothing to notice it, so this raises and
    names the flag that collapses the groups."""
    runner_module = _load_sparse_target_runner()
    runner = object.__new__(runner_module.SparseTargetGPUModelRunner)

    attn_metadata = {f"l{i}": object() for i in range(4)}
    runner._gatherable_layer_names_cache = {"l0", "l2"}
    runner.kv_cache_config = type("KVCacheConfig", (), {"kv_cache_groups": [
        _fake_group(16, ["l0", "l1"]),
        _fake_group(16, ["l2", "l3"]),
    ]})()
    cache_config = type("CacheConfig", (), {"block_size": 16})()
    runner.vllm_config = type("VllmConfig", (), {"cache_config": cache_config})()

    try:
        runner._gatherable_group_block_size(attn_metadata)
    except NotImplementedError as exc:
        assert "span" in str(exc)
        assert "disable_hybrid_kv_cache_manager" in str(exc), (
            "the error must name the flag that fixes it"
        )
    else:
        raise AssertionError("gathered layers spanning groups must raise")


def test_model_flop_config_reads_the_text_config_and_refuses_shapes_it_cannot_model():
    """`ModelFlopConfig` is flat -- one head_dim, one num_kv_heads, one
    intermediate. Gemma 4 breaks that twice over (per-layer-type attention
    geometry, MoE), so returning an approximate number would put an estimate
    in the results CSV beside genuinely measured ones with nothing marking
    it. Refusing is the honest answer."""
    from flops_model import model_flop_config

    class _Cfg:
        def __init__(self, **kw):
            self.num_attention_heads = 32
            self.num_hidden_layers = 32
            self.hidden_size = 4096
            self.intermediate_size = 14336
            self.vocab_size = 128256
            self.num_key_value_heads = 8
            self.head_dim = 128
            for k, v in kw.items():
                setattr(self, k, v)

        def get_text_config(self):
            return self

    # Llama-shaped: modelled exactly as before.
    cfg = model_flop_config(_Cfg())
    assert cfg is not None and cfg.num_layers == 32 and cfg.head_dim == 128

    # Interleaved attention -- one head_dim cannot describe both layer types.
    assert model_flop_config(_Cfg(
        layer_types=["sliding_attention"] * 4 + ["full_attention"])) is None
    # Heterogeneous head dims, even without layer_types.
    assert model_flop_config(_Cfg(head_dim=256, global_head_dim=512)) is None
    # MoE -- intermediate_size alone does not describe the MLP cost.
    assert model_flop_config(_Cfg(enable_moe_block=True)) is None
    assert model_flop_config(_Cfg(num_experts=128)) is None

    # Uniform layer_types is NOT heterogeneous, and global_head_dim equal to
    # head_dim is not either -- neither should trip the refusal.
    assert model_flop_config(_Cfg(layer_types=["full_attention"] * 32)) is not None
    assert model_flop_config(_Cfg(head_dim=128, global_head_dim=128)) is not None

    # The wrapper case: a multimodal config with no attention fields of its
    # own must be resolved through get_text_config(), not crash.
    class _Wrapper:
        def get_text_config(self):
            return _Cfg()

    assert model_flop_config(_Wrapper()) is not None


def test_speculator_worker_rpc_wrappers_match_their_runner_methods():
    """`collective_rpc` dispatches BY NAME to `SpeculatorWorker`, which
    forwards to the identically-named `SpeculatorGPUModelRunner` method. Add
    a parameter to one and not the other and nothing complains until the
    first scored turn, minutes into a run:

        TypeError: SpeculatorWorker.end_capture_and_score() takes from 6 to 9
        positional arguments but 10 were given

    That is exactly what happened when `mask_sliding_window` was threaded
    through the runner alone. Compared via `ast` rather than `inspect`
    because `speculator_worker.py` imports `vllm.v1.worker.gpu_model_runner`
    at module scope and is not importable in this CPU-only suite -- the same
    reason `model_structure.py` and `model_truncation.py` exist as separate
    vLLM-free modules."""
    import ast

    source = (Path(__file__).parent / "vllm_patch" / "speculator_worker.py").read_text(
        encoding="utf-8"
    )
    classes = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
    }
    for required in ("SpeculatorGPUModelRunner", "SpeculatorWorker"):
        assert required in classes, f"{required} not found -- was it renamed?"

    def signatures(cls):
        return {
            m.name: [a.arg for a in m.args.args[1:]]  # drop self
            for m in cls.body
            if isinstance(m, ast.FunctionDef) and not m.name.startswith("_")
        }

    runner = signatures(classes["SpeculatorGPUModelRunner"])
    worker = signatures(classes["SpeculatorWorker"])
    shared = sorted(set(runner) & set(worker))
    assert "end_capture_and_score" in shared, (
        "the scoring entry point must be paired -- if it was renamed, this "
        "test is no longer guarding the thing it exists for"
    )

    mismatched = {
        name: (runner[name], worker[name])
        for name in shared
        if runner[name] != worker[name]
    }
    assert not mismatched, (
        "RPC wrapper signatures drifted from their runner methods: "
        + "; ".join(
            f"{name}: runner{r} vs worker{w}" for name, (r, w) in mismatched.items()
        )
    )


if __name__ == "__main__":
    _run_all()
