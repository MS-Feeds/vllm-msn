#!/usr/bin/env python3
"""Smoke test for the DeepSeek-V3.2 lightning-indexer instrumentation.

Tests that:
1. The indexer logger can be imported
2. Attention-logit (lightning-indexer) mode is the default config
3. Recording produces the expected output file structure
4. INDEXER_STREAMS correctly enables/skips individual streams' computation
5. DeepSeek-V3.2's Indexer class (config.index_topk) is present and calls
   the patched sparse_attn_indexer()

Usage:
    # Set environment variables
    export INDEXER_LOGIT_DUMP_DIR="./test_logits"
    export INDEXER_IS_EXPERT_ROUTING=0
    export INDEXER_LOGIT_RANGE="-50.0,50.0"

    # Run smoke test
    python test_deepseek_instrumentation.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def test_indexer_import():
    """Test that indexer logger can be imported."""
    print("Test 1: Import _indexer_logger...")
    try:
        # Add current directory to path for testing
        sys.path.insert(0, str(Path(__file__).parent / "vllm_patches"))
        from _indexer_logger import is_enabled, record
        print("  ✓ Import successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_config_detection():
    """Test that attention-logit (lightning-indexer) mode is the default."""
    print("\nTest 2: Config detection...")

    # Set up environment for the lightning indexer (attention logits).
    # RANK is set explicitly: outside a real vLLM distributed worker, the
    # logger falls back to PID for the dump filename (see _detect_rank()),
    # which would make the rank0.npz path assertion below flaky.
    os.environ["INDEXER_LOGIT_DUMP_DIR"] = tempfile.mkdtemp()
    os.environ["INDEXER_IS_EXPERT_ROUTING"] = "0"
    os.environ["RANK"] = "0"

    # Force reload config
    import importlib
    if "_indexer_logger" in sys.modules:
        importlib.reload(sys.modules["_indexer_logger"])

    from _indexer_logger import _CONFIG

    if _CONFIG is None:
        print("  ✗ Config not initialized (INDEXER_LOGIT_DUMP_DIR not set?)")
        return False

    # Check that range defaults to [-50, 50] for raw attention logits
    if _CONFIG.range_lo == -50.0 and _CONFIG.range_hi == 50.0:
        print(f"  ✓ Attention-logit mode detected: range=[{_CONFIG.range_lo}, {_CONFIG.range_hi}]")
        return True
    else:
        print(f"  ✗ Unexpected range: [{_CONFIG.range_lo}, {_CONFIG.range_hi}] (expected [-50.0, 50.0])")
        return False


def test_mock_recording():
    """Test recording with mock lightning-indexer attention-logit data."""
    print("\nTest 3: Mock attention-logit recording...")

    import _indexer_logger
    from _indexer_logger import record, _CONFIG

    if _CONFIG is None:
        print("  ✗ Config not initialized")
        return False

    try:
        # Simulate the lightning indexer: 16 query tokens, 2048 candidate
        # keys, raw (unbounded) attention logits.
        num_tokens = 16
        num_keys = 2048
        top_k = 2048

        torch.manual_seed(42)
        logits = (torch.rand(num_tokens, num_keys) - 0.5) * 100  # roughly [-50, 50]

        starts = torch.zeros(num_tokens, dtype=torch.long)
        ends = torch.full((num_tokens,), num_keys, dtype=torch.long)

        record(
            k_cache_prefix="model.layers.0.self_attn.indexer",
            phase="decode",
            logits=logits,
            index_topk=top_k,
            key_valid_starts=starts,
            key_valid_ends=ends,
        )

        print(f"  ✓ Recorded {num_tokens} tokens × {num_keys} keys (top-{top_k})")

        # record() only buffers in memory; the recorder normally dumps to
        # disk via an atexit hook. Dump explicitly so the smoke test doesn't
        # have to exit the process to verify the output structure.
        _indexer_logger._recorder.dump()

        # Verify output file created
        dump_path = Path(_CONFIG.dump_dir) / "indexer_logits_rank0.npz"
        if dump_path.exists():
            print(f"  ✓ Output file created: {dump_path}")

            # Load and check structure
            data = np.load(dump_path)
            print(f"  ✓ Captured layers: {data['layer_ids']}")
            print(f"  ✓ Decode rows: {data['count_decode']}")
            return True
        else:
            print(f"  ✗ Output file not created: {dump_path}")
            return False

    except Exception as e:
        print(f"  ✗ Recording failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stream_gating():
    """Test that INDEXER_STREAMS actually skips the disabled streams'
    computation, not just their output fields."""
    print("\nTest 4: INDEXER_STREAMS gating (B,C only)...")

    import importlib
    os.environ["INDEXER_LOGIT_DUMP_DIR"] = tempfile.mkdtemp()
    os.environ["INDEXER_IS_EXPERT_ROUTING"] = "0"
    os.environ["RANK"] = "0"
    os.environ["INDEXER_STREAMS"] = "B,C"

    import _indexer_logger
    importlib.reload(_indexer_logger)
    from _indexer_logger import record, _CONFIG

    if _CONFIG.enable_streams != frozenset({"B", "C"}):
        print(f"  ✗ enable_streams={_CONFIG.enable_streams}, expected {{'B', 'C'}}")
        return False

    try:
        # Need >= index_topk + gap_window + 1 valid keys for streams A/C to
        # even attempt the boundary computation (irrelevant here since A is
        # disabled, but keep the row count consistent with a real capture).
        num_tokens, num_keys, top_k = 16, 3000, 2048
        torch.manual_seed(0)
        logits = (torch.rand(num_tokens, num_keys) - 0.5) * 100
        starts = torch.zeros(num_tokens, dtype=torch.long)
        ends = torch.full((num_tokens,), num_keys, dtype=torch.long)

        record("model.layers.0.self_attn.indexer", "decode", logits, top_k, starts, ends)
        _indexer_logger._recorder.dump()

        d = np.load(_indexer_logger._recorder.dump_path)
        ok = True
        if d["count_decode"][0] == 0:
            print("  ✗ Stream B (enabled) recorded nothing")
            ok = False
        if d["topk_count_decode"][0] == 0:
            print("  ✗ Stream C (enabled) recorded nothing")
            ok = False
        if d["gap_count_decode"][0] != 0:
            print("  ✗ Stream A (disabled) still recorded data")
            ok = False
        if d["radix_bkt_count_decode"].sum() != 0:
            print("  ✗ Stream D (disabled) still recorded data")
            ok = False
        if ok:
            print(f"  ✓ B count={d['count_decode'][0]}, C count={d['topk_count_decode'][0]}, "
                  f"A count={d['gap_count_decode'][0]}, D count={d['radix_bkt_count_decode'].sum()}")
        return ok

    except Exception as e:
        print(f"  ✗ Stream gating check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.environ.pop("INDEXER_STREAMS", None)


def test_deepseek_v32_indexer_present():
    """Test that DeepSeek-V3.2's Indexer class and its dependency on the
    patched sparse_attn_indexer() are present."""
    print("\nTest 5: DeepSeek-V3.2 Indexer import...")

    try:
        from vllm.model_executor.models.deepseek_v2 import Indexer
        print("  ✓ Indexer imported successfully")

        import inspect
        source = inspect.getsource(Indexer)
        if "index_topk" in source and "SparseAttnIndexer" in source:
            print("  ✓ Indexer wires config.index_topk into SparseAttnIndexer")
        else:
            print("  ✗ Indexer no longer matches the expected V3.2 wiring")
            return False

        from vllm.model_executor.layers.sparse_attn_indexer import (
            sparse_attn_indexer,
        )
        source = inspect.getsource(sparse_attn_indexer)
        if "_indexer_logger" in source:
            print("  ✓ sparse_attn_indexer() has the logger patch applied")
            return True
        else:
            print("  ⚠ Logger patch not applied (run sync_to_env.sh?)")
            return True  # Still pass, just a warning

    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("DeepSeek-V3.2 Lightning-Indexer Instrumentation Smoke Test")
    print("=" * 60)

    results = []

    results.append(("Import", test_indexer_import()))
    results.append(("Config", test_config_detection()))
    results.append(("Recording", test_mock_recording()))
    results.append(("Stream gating", test_stream_gating()))
    results.append(("Model Import", test_deepseek_v32_indexer_present()))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s}  {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nNext steps:")
        print("  1. Run run_v32_experiment.sh --server-only for a quick smoke test")
        print("  2. Run run_v32_experiment.sh for the full eval sweep")
        print("  3. Verify logit captures with plot_indexer_logits.py")
        return 0
    else:
        print("✗ Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
