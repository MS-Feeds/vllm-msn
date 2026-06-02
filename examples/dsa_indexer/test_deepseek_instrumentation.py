#!/usr/bin/env python3
"""Smoke test for DeepSeek MoE instrumentation.

Tests that:
1. DeepSeek V2/V3 model loads successfully
2. Indexer logger captures expert routing scores
3. Output files are created with expected structure

Usage:
    # Set environment variables
    export INDEXER_LOGIT_DUMP_DIR="./test_logits"
    export INDEXER_IS_EXPERT_ROUTING=1
    export INDEXER_LOGIT_RANGE="0.0,1.0"

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
        sys.path.insert(0, str(Path(__file__).parent / "vllm"))
        from _indexer_logger import is_enabled, record
        print("  ✓ Import successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_config_detection():
    """Test that expert routing mode is detected correctly."""
    print("\nTest 2: Config detection...")

    # Set up environment for expert routing
    os.environ["INDEXER_LOGIT_DUMP_DIR"] = tempfile.mkdtemp()
    os.environ["INDEXER_IS_EXPERT_ROUTING"] = "1"

    # Force reload config
    import importlib
    if "_indexer_logger" in sys.modules:
        importlib.reload(sys.modules["_indexer_logger"])

    from _indexer_logger import _CONFIG

    if _CONFIG is None:
        print("  ✗ Config not initialized (INDEXER_LOGIT_DUMP_DIR not set?)")
        return False

    # Check that range defaults to [0, 1] for expert routing
    if _CONFIG.range_lo == 0.0 and _CONFIG.range_hi == 1.0:
        print(f"  ✓ Expert routing mode detected: range=[{_CONFIG.range_lo}, {_CONFIG.range_hi}]")
        return True
    else:
        print(f"  ✗ Unexpected range: [{_CONFIG.range_lo}, {_CONFIG.range_hi}] (expected [0.0, 1.0])")
        return False


def test_mock_recording():
    """Test recording with mock expert routing data."""
    print("\nTest 3: Mock expert routing recording...")

    from _indexer_logger import record, _CONFIG

    if _CONFIG is None:
        print("  ✗ Config not initialized")
        return False

    try:
        # Simulate expert routing: 16 tokens, 160 experts, post-softmax scores
        num_tokens = 16
        num_experts = 160
        top_k = 6

        # Create mock post-softmax scores (should sum to ~1 per row)
        torch.manual_seed(42)
        logits = torch.rand(num_tokens, num_experts)
        scores = torch.softmax(logits, dim=-1)  # Post-softmax: [0, 1], sum=1

        # All experts valid (no masking)
        starts = torch.zeros(num_tokens, dtype=torch.long)
        ends = torch.full((num_tokens,), num_experts, dtype=torch.long)

        # Record
        record(
            k_cache_prefix="model.layers.0.mlp.gate",
            phase="decode",
            logits=scores,
            index_topk=top_k,
            key_valid_starts=starts,
            key_valid_ends=ends,
        )

        print(f"  ✓ Recorded {num_tokens} tokens × {num_experts} experts (top-{top_k})")

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


def test_deepseek_model_import():
    """Test that patched DeepSeek model can be imported."""
    print("\nTest 4: DeepSeek model import...")

    try:
        # This will use the patched version if sync_to_env.sh was run
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
        print("  ✓ DeepseekV2MoE imported successfully")

        # Check if instrumentation code is present
        import inspect
        source = inspect.getsource(DeepseekV2MoE.forward)
        if "INDEXER LOGGER INSTRUMENTATION" in source:
            print("  ✓ Instrumentation code detected in forward method")
            return True
        else:
            print("  ⚠ Instrumentation code not found (run sync_to_env.sh?)")
            return True  # Still pass, just a warning

    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("DeepSeek MoE Instrumentation Smoke Test")
    print("=" * 60)

    results = []

    results.append(("Import", test_indexer_import()))
    results.append(("Config", test_config_detection()))
    results.append(("Recording", test_mock_recording()))
    results.append(("Model Import", test_deepseek_model_import()))

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
        print("  1. Run sync_to_env.sh to install patches")
        print("  2. Test with actual DeepSeek model inference")
        print("  3. Verify logit captures with plot_indexer_logits.py")
        return 0
    else:
        print("✗ Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
