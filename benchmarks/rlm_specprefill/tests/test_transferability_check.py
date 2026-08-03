"""Tests for calibration/transferability_check.py's pure-Python pieces
(reference-curve loading, recall computation, curve comparison) -- no
vllm/GPU needed. The GPU-only sweep orchestration (run_rlm_format_sweep,
which builds real SpecPrefill engines) is not exercised here; see
REPRODUCE.md's GPU-node steps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from calibration.transferability_check import (  # noqa: E402
    compare_curves,
    compute_recall,
    load_reference_curve,
)


# ---------------------------------------------------------------------------
# load_reference_curve
# ---------------------------------------------------------------------------


def _write_result_json(path: Path, overall_accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall_accuracy, "by_difficulty": {}, "by_domain": {}, "counts": {}}, f)


def test_load_reference_curve_sorted_by_keep_percentage(tmp_path):
    _write_result_json(tmp_path / "P001_result.json", 92.0)  # keep=1.0
    _write_result_json(tmp_path / "P004_result.json", 88.0)  # keep=0.5
    _write_result_json(tmp_path / "P002_result.json", 70.0)  # keep=0.1

    curve = load_reference_curve(tmp_path)

    assert curve == [(0.1, 70.0), (0.5, 88.0), (1.0, 92.0)]


def test_load_reference_curve_skips_missing_experiments(tmp_path):
    _write_result_json(tmp_path / "P002_result.json", 70.0)
    _write_result_json(tmp_path / "P006_result.json", 90.0)
    # P001, P003, P004, P005 not present -- should be silently skipped.

    curve = load_reference_curve(tmp_path)

    assert curve == [(0.1, 70.0), (0.9, 90.0)]


def test_load_reference_curve_raises_when_nothing_found(tmp_path):
    with pytest.raises(ValueError, match="No .*result.json"):
        load_reference_curve(tmp_path)


# ---------------------------------------------------------------------------
# compute_recall
# ---------------------------------------------------------------------------


def test_compute_recall_all_found():
    assert compute_recall(["123", "456"], "the values are 123 and 456 here") == 1.0


def test_compute_recall_partial():
    assert compute_recall(["123", "456"], "only 123 is here") == 0.5


def test_compute_recall_none_found():
    assert compute_recall(["123", "456"], "nothing relevant here") == 0.0


def test_compute_recall_no_needles_is_vacuously_full_recall():
    assert compute_recall([], "any text") == 1.0


# ---------------------------------------------------------------------------
# compare_curves
# ---------------------------------------------------------------------------


def test_compare_curves_matches_nearest_reference_point():
    reference = [(0.1, 70.0), (0.5, 88.0), (1.0, 92.0)]
    ours = [(0.5, 0.87)]  # 87% recall, matches keep=0.5 -> 88% reference exactly

    comparison = compare_curves(reference, ours, threshold_pct=15.0)

    assert len(comparison["points"]) == 1
    point = comparison["points"][0]
    assert point["percentage"] == 0.5
    assert point["reference_percentage_matched"] == 0.5
    assert point["our_recall_pct"] == 87.0
    assert point["reference_accuracy_pct"] == 88.0
    assert point["delta"] == pytest.approx(-1.0)
    assert point["diverges"] is False


def test_compare_curves_flags_divergence_beyond_threshold():
    reference = [(0.5, 88.0)]
    ours = [(0.5, 0.40)]  # 40% recall vs 88% reference accuracy -- way off

    comparison = compare_curves(reference, ours, threshold_pct=15.0)

    assert comparison["points"][0]["diverges"] is True
    assert comparison["any_divergence"] is True


def test_compare_curves_no_divergence_when_all_points_within_threshold():
    reference = [(0.1, 70.0), (0.5, 88.0), (0.9, 91.0)]
    ours = [(0.1, 0.68), (0.5, 0.85), (0.9, 0.90)]  # all within a few points

    comparison = compare_curves(reference, ours, threshold_pct=15.0)

    assert comparison["any_divergence"] is False
    assert all(not p["diverges"] for p in comparison["points"])


def test_compare_curves_matches_to_nearest_when_percentages_dont_align_exactly():
    """Our sweep's percentages don't have to exactly match the reference
    sweep's -- nearest-point matching handles that."""
    reference = [(0.1, 70.0), (0.9, 91.0)]
    ours = [(0.2, 0.72)]  # closer to reference's 0.1 than 0.9

    comparison = compare_curves(reference, ours, threshold_pct=15.0)

    assert comparison["points"][0]["reference_percentage_matched"] == 0.1
