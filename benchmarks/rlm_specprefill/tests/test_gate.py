"""Tests for target_stage/gate.py. Pure Python, no vllm/GPU needed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from target_stage.gate import load_n_min, should_compress  # noqa: E402


def test_should_compress_above_threshold():
    assert should_compress(n_tokens=500, n_min=100) is True


def test_should_compress_below_threshold():
    assert should_compress(n_tokens=50, n_min=100) is False


def test_should_compress_strictly_greater_not_greater_equal():
    """N > N_min per ../rlm_specprefill_ablation_plan.md's SPECPREFILL GATE
    section -- exactly at N_min should NOT trigger compression (the gate is
    calibrated to be the crossover point where compression starts paying
    for itself, so equality should fall on the "not yet worth it" side)."""
    assert should_compress(n_tokens=100, n_min=100) is False


def test_load_n_min_missing_file_raises_clear_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    try:
        load_n_min(missing_path)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "sweep_n_min.py" in str(e)  # points the user at the fix, not just "file not found"


def test_load_n_min_reads_calibrated_value(tmp_path):
    path = tmp_path / "n_min.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_min": 4096, "curve": [[1000, 1.2], [4096, 0.9]]}, f)

    assert load_n_min(path) == 4096
