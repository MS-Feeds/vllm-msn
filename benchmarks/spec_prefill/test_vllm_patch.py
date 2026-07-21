"""Standalone tests for the SpecPrefill Algorithm 1 pieces built in
vllm_patch/ (lines 1, 3-16, and pruning_registry.py's half of 18-19) -- see
EXPERIMENT_PLAN.md and the two approved plans for scope.

Covers everything testable without vLLM's full runtime or a real model
checkpoint: scoring.py's math (lines 12, 14, 16), prefill_split.py (line 1),
kv_cache_utils.py's backend-layout logic (part of line 11), and
pruning_registry.py's PruneRecord store (the driver/runner handoff for lines
18-19). The attention-hook/model-loading/lookahead-loop pieces in
proposer.py, pruner.py's end-to-end orchestration, and model_runner.py's/
worker.py's actual runner-integration behavior all require a running vLLM
engine with a loaded model and are not exercised here -- see
validate_proposer.py and each module's own "residual risk" notes for what a
live run would need to confirm.

Run with: python3 test_vllm_patch.py
(no pytest dependency assumed; plain asserts + a runner, matching
evaluation_pipeline/test_hardware_metrics.py's convention of a
if __name__ == "__main__" driver -- adjust here if this repo standardizes on
pytest later.)
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from vllm_patch.config import SpecConfig
from vllm_patch.kv_cache_utils import _find_kv_split_dim, gather_keys_for_slots
from vllm_patch.prefill_split import split_prefill_decode_requests
from vllm_patch import pruning_registry
from vllm_patch.pruning_registry import PruneRecord
from vllm_patch.scoring import (
    aggregate_attention_score,
    chunk_select_from_smoothed_attention,
    compute_attention_score,
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
    """num_heads != num_kv_heads must repeat-interleave K to match, not error."""
    num_layers, num_samples, num_heads, num_kv_heads, head_dim, look_ahead, ctx_len = (
        2, 2, 4, 2, 8, 3, 16,
    )
    query_buffer, key_buffer = _synthetic_qk(
        num_layers, num_samples, num_heads, num_kv_heads, head_dim, look_ahead, ctx_len
    )
    attn_scores = compute_attention_score(query_buffer, key_buffer, [look_ahead] * num_samples)
    assert attn_scores[0].shape == (num_layers, num_heads, look_ahead, ctx_len)


def test_compute_attention_score_single_layer():
    query_buffer, key_buffer = _synthetic_qk(1, 2, 4, 4, 8, 3, 16)
    attn_scores = compute_attention_score(query_buffer, key_buffer, [3, 3])
    assert attn_scores[0].shape == (1, 4, 3, 16)


def test_compute_attention_score_uneven_lookahead():
    """Per-sample actual_look_ahead_cnts differ (early EOS on one sample)."""
    num_layers, num_samples, num_heads, num_kv_heads, head_dim, look_ahead, ctx_len = (
        2, 2, 4, 4, 8, 3, 16,
    )
    query_buffer, key_buffer = _synthetic_qk(
        num_layers, num_samples, num_heads, num_kv_heads, head_dim, look_ahead, ctx_len
    )
    attn_scores = compute_attention_score(query_buffer, key_buffer, [look_ahead, 1])
    assert attn_scores[0].shape[2] == look_ahead
    assert attn_scores[1].shape[2] == 1


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

    # percentage=1.0 edge case: keep everything
    cfg_keep_all = SpecConfig(
        keep_strategy="percentage", keep_kwargs={"percentage": 1.0}, look_ahead_cnt=3
    )
    kept_all = chunk_select_from_smoothed_attention(token_importance, cfg_keep_all)
    assert kept_all[0].numel() == 16

    # no pooling (pool_kernel_size=None) must not error
    cfg_no_pool = SpecConfig(
        keep_strategy="percentage",
        keep_kwargs={"chunk": True, "chunk_size": 4, "percentage": 0.5},
        look_ahead_cnt=3,
        pool_kernel_size=None,
    )
    ti_no_pool = aggregate_attention_score(attn_scores, cfg_no_pool)
    assert ti_no_pool[0].shape == (16,)


def test_kv_split_dim_triton_layout():
    """(num_blocks, 2, block_size, num_kv_heads, head_size) -> split_dim=1."""

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


def test_kv_split_dim_flash_layout():
    """(2, num_blocks, block_size, num_kv_heads, head_size) -> split_dim=0."""

    class FakeFlashBackend:
        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
            return (2, num_blocks, block_size, num_kv_heads, head_size)

        @classmethod
        def get_kv_cache_block_dim(cls, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
            _S = 1234567
            return cls.get_kv_cache_shape(_S, block_size, num_kv_heads, head_size).index(_S)

    block_size, num_kv_heads, head_size, num_blocks = 16, 4, 128, 100
    kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_size)
    split_dim = _find_kv_split_dim(FakeFlashBackend, kv_cache, block_size, num_kv_heads, head_size)
    assert split_dim == 0


def test_kv_split_dim_shape_mismatch_raises():
    class FakeBackend:
        @staticmethod
        def get_name():
            return "FakeBackend"

        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
            return (num_blocks, 2, block_size, num_kv_heads, head_size)

        @classmethod
        def get_kv_cache_block_dim(cls, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
            _S = 1234567
            return cls.get_kv_cache_shape(_S, block_size, num_kv_heads, head_size).index(_S)

    wrong_shape_cache = torch.zeros(3, 5)  # doesn't match FakeBackend's declared shape
    try:
        _find_kv_split_dim(FakeBackend, wrong_shape_cache, 16, 4, 128)
        raise AssertionError("expected ValueError for mismatched shape")
    except ValueError:
        pass


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


def test_flatten_slot_index_roundtrip_both_layouts():
    """End-to-end: unbind -> flatten -> index by slot must recover the exact
    (block_id, offset) element, for both known backend layouts."""
    block_size, num_kv_heads, head_size, num_blocks = 16, 4, 8, 10

    def roundtrip(kv_cache: torch.Tensor, split_dim: int):
        k_cache, _v_cache = kv_cache.unbind(split_dim)
        assert k_cache.shape[0] == num_blocks and k_cache.shape[1] == block_size
        flat = k_cache.reshape(-1, num_kv_heads, head_size)
        block_id, offset = 5, 3
        slot = block_id * block_size + offset
        assert torch.equal(flat[slot], k_cache[block_id, offset])

    triton_cache = torch.arange(
        num_blocks * 2 * block_size * num_kv_heads * head_size, dtype=torch.float32
    ).reshape(num_blocks, 2, block_size, num_kv_heads, head_size)
    roundtrip(triton_cache, split_dim=1)

    flash_cache = torch.arange(
        2 * num_blocks * block_size * num_kv_heads * head_size, dtype=torch.float32
    ).reshape(2, num_blocks, block_size, num_kv_heads, head_size)
    roundtrip(flash_cache, split_dim=0)


def test_prune_record_validation():
    # valid record
    record = PruneRecord(kept_positions=[0, 3, 5], orig_len=8)
    assert record.num_kept == 3
    assert record.decode_offset == 5  # 8 - 3

    # kept_positions longer than orig_len must raise
    try:
        PruneRecord(kept_positions=[0, 1, 2, 3], orig_len=3)
        raise AssertionError("expected ValueError for kept_positions > orig_len")
    except ValueError:
        pass

    # a kept position outside [0, orig_len) must raise
    try:
        PruneRecord(kept_positions=[0, 8], orig_len=8)
        raise AssertionError("expected ValueError for out-of-range position")
    except ValueError:
        pass


def test_pruning_registry_lifecycle():
    pruning_registry.clear()
    try:
        assert pruning_registry.get("req-1") is None

        pruning_registry.register("req-1", kept_positions=[0, 2, 4], orig_len=10)
        record = pruning_registry.get("req-1")
        assert record is not None
        assert record.kept_positions == [0, 2, 4]
        assert record.orig_len == 10
        assert record.num_kept == 3
        assert record.decode_offset == 7

        # unregistered request stays None
        assert pruning_registry.get("req-2") is None

        # re-registering overwrites, doesn't accumulate
        pruning_registry.register("req-1", kept_positions=[1], orig_len=5)
        record = pruning_registry.get("req-1")
        assert record.kept_positions == [1]
        assert record.orig_len == 5

        pruning_registry.discard("req-1")
        assert pruning_registry.get("req-1") is None

        pruning_registry.register("req-a", kept_positions=[0], orig_len=1)
        pruning_registry.register("req-b", kept_positions=[0], orig_len=1)
        pruning_registry.discard_finished(["req-a", "req-nonexistent"])
        assert pruning_registry.get("req-a") is None
        assert pruning_registry.get("req-b") is not None
    finally:
        pruning_registry.clear()


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
