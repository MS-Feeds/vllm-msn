"""Tests for analysis/aggregate_metrics.py against synthetic JSONL fixtures
shaped exactly like runner/run_arm.py's real predictions.jsonl/timing.jsonl
output -- no GPU/vllm/live-API run needed to exercise the aggregation
logic itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.aggregate_metrics import (  # noqa: E402
    aggregate_arm_metrics,
    load_arm_metrics,
    percentile_summary,
    render_arm_summary,
    score_accuracy,
)
from eval_data.schema import EvalSample  # noqa: E402


def _write_jsonl_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _timing_row(sample_id: str, **overrides) -> dict:
    row = {
        "sample_id": sample_id, "wall_time_s": 5.0,
        "t_rlm_root": 3.0, "t_repl_compute": 0.1, "t_rlm_subcalls": 0.9,
        "n_iterations": 3, "n_direct_subcalls": 1, "max_realized_depth": 1, "total_calls": 4,
    }
    row.update(overrides)
    return row


def _prediction_row(sample_id: str, **overrides) -> dict:
    row = {
        "id": sample_id, "pred": "the answer", "finish_reason": "stop",
        "n_output_tokens": 10, "n_prompt_tokens_full": 1000, "n_prompt_tokens_kept": None,
        "keep_rate": None, "ttft_ms": 50.0, "generation_time_s": 1.0,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# load_arm_metrics
# ---------------------------------------------------------------------------


def test_load_arm_metrics_joins_by_id(tmp_path):
    arm_dir = tmp_path / "A"
    _write_jsonl_rows([_timing_row("s1"), _timing_row("s2")], arm_dir / "timing.jsonl")
    _write_jsonl_rows([_prediction_row("s1"), _prediction_row("s2")], arm_dir / "predictions.jsonl")

    metrics = load_arm_metrics(arm_dir)

    assert len(metrics) == 2
    assert {m.sample_id for m in metrics} == {"s1", "s2"}
    for m in metrics:
        assert m.generation_time_s == 1.0
        assert m.pred == "the answer"


def test_load_arm_metrics_missing_predictions_still_yields_timing_only_rows(tmp_path):
    """--dry-run arms have timing.jsonl but no predictions.jsonl at all --
    must not crash, must yield rows with None target-stage fields."""
    arm_dir = tmp_path / "A"
    _write_jsonl_rows([_timing_row("s1")], arm_dir / "timing.jsonl")
    # No predictions.jsonl written.

    metrics = load_arm_metrics(arm_dir)

    assert len(metrics) == 1
    assert metrics[0].generation_time_s is None
    assert metrics[0].pred is None
    assert metrics[0].f is None


def test_load_arm_metrics_partial_predictions_some_samples_unanswered(tmp_path):
    """A sample skipped by answer_batch (over budget) has a timing.jsonl
    row but no predictions.jsonl row -- must surface as None, not crash or
    silently drop the sample."""
    arm_dir = tmp_path / "C"
    _write_jsonl_rows([_timing_row("s1"), _timing_row("s2")], arm_dir / "timing.jsonl")
    _write_jsonl_rows([_prediction_row("s1")], arm_dir / "predictions.jsonl")  # s2 missing

    metrics = load_arm_metrics(arm_dir)
    by_id = {m.sample_id: m for m in metrics}

    assert by_id["s1"].generation_time_s == 1.0
    assert by_id["s2"].generation_time_s is None


def test_load_arm_metrics_raises_when_timing_jsonl_missing(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="timing.jsonl"):
        load_arm_metrics(tmp_path / "nonexistent_arm")


# ---------------------------------------------------------------------------
# SampleMetrics.f / spec_prefill_applied
# ---------------------------------------------------------------------------


def test_sample_metrics_f_computation():
    from analysis.aggregate_metrics import SampleMetrics

    m = SampleMetrics(
        sample_id="s1", t_rlm_root=8.0, t_repl_compute=1.0, t_rlm_subcalls=1.0,
        n_iterations=3, n_direct_subcalls=0, max_realized_depth=0, total_calls=3,
        generation_time_s=10.0, ttft_ms=100.0, keep_rate=None,
        n_prompt_tokens_full=1000, n_prompt_tokens_kept=None, pred="x",
    )
    # t_rlm_total = 10.0, generation_time_s = 10.0 -> f = 10 / 20 = 0.5
    assert m.f == 0.5


def test_sample_metrics_f_none_without_target_stage_result():
    from analysis.aggregate_metrics import SampleMetrics

    m = SampleMetrics(
        sample_id="s1", t_rlm_root=8.0, t_repl_compute=1.0, t_rlm_subcalls=1.0,
        n_iterations=3, n_direct_subcalls=0, max_realized_depth=0, total_calls=3,
        generation_time_s=None, ttft_ms=None, keep_rate=None,
        n_prompt_tokens_full=None, n_prompt_tokens_kept=None, pred=None,
    )
    assert m.f is None


def test_sample_metrics_spec_prefill_applied():
    from analysis.aggregate_metrics import SampleMetrics

    plain = SampleMetrics(
        sample_id="s1", t_rlm_root=1.0, t_repl_compute=0.0, t_rlm_subcalls=0.0,
        n_iterations=1, n_direct_subcalls=0, max_realized_depth=0, total_calls=1,
        generation_time_s=1.0, ttft_ms=10.0, keep_rate=None,
        n_prompt_tokens_full=100, n_prompt_tokens_kept=None, pred="x",
    )
    pruned = SampleMetrics(
        sample_id="s2", t_rlm_root=1.0, t_repl_compute=0.0, t_rlm_subcalls=0.0,
        n_iterations=1, n_direct_subcalls=0, max_realized_depth=0, total_calls=1,
        generation_time_s=1.0, ttft_ms=10.0, keep_rate=0.5,
        n_prompt_tokens_full=100, n_prompt_tokens_kept=50, pred="x",
    )
    assert plain.spec_prefill_applied is False
    assert pruned.spec_prefill_applied is True


# ---------------------------------------------------------------------------
# percentile_summary / aggregate_arm_metrics
# ---------------------------------------------------------------------------


def test_percentile_summary_empty():
    summary = percentile_summary([])
    assert summary["n"] == 0
    assert summary["mean"] is None
    assert summary["p50"] is None


def test_percentile_summary_basic():
    summary = percentile_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["n"] == 5
    assert summary["mean"] == 3.0
    assert summary["p50"] == 3.0
    assert summary["min"] == 1.0
    assert summary["max"] == 5.0


def test_aggregate_arm_metrics_f_distribution(tmp_path):
    arm_dir = tmp_path / "A"
    _write_jsonl_rows(
        [_timing_row("s1", t_rlm_root=9.0, t_repl_compute=0.0, t_rlm_subcalls=0.0),
         _timing_row("s2", t_rlm_root=1.0, t_repl_compute=0.0, t_rlm_subcalls=0.0)],
        arm_dir / "timing.jsonl",
    )
    _write_jsonl_rows(
        [_prediction_row("s1", generation_time_s=1.0),   # f = 1/10 = 0.1
         _prediction_row("s2", generation_time_s=9.0)],  # f = 9/10 = 0.9
        arm_dir / "predictions.jsonl",
    )

    metrics = load_arm_metrics(arm_dir)
    agg = aggregate_arm_metrics(metrics)

    assert agg["n_samples"] == 2
    assert agg["n_target_answered"] == 2
    assert agg["f_distribution"]["n"] == 2
    assert agg["f_distribution"]["min"] == 0.1
    assert agg["f_distribution"]["max"] == 0.9


def test_aggregate_arm_metrics_gate_split_for_arm_c(tmp_path):
    """Arm C mixes plain (n_prompt_tokens_kept=None) and SpecPrefill
    (n_prompt_tokens_kept set) answers -- gate_split must count each
    correctly, purely from the keep-field presence, no separate flag
    needed."""
    arm_dir = tmp_path / "C"
    _write_jsonl_rows([_timing_row("skip1"), _timing_row("skip2"), _timing_row("compress1")], arm_dir / "timing.jsonl")
    _write_jsonl_rows(
        [
            _prediction_row("skip1", n_prompt_tokens_kept=None, keep_rate=None),
            _prediction_row("skip2", n_prompt_tokens_kept=None, keep_rate=None),
            _prediction_row("compress1", n_prompt_tokens_kept=300, keep_rate=0.3),
        ],
        arm_dir / "predictions.jsonl",
    )

    metrics = load_arm_metrics(arm_dir)
    agg = aggregate_arm_metrics(metrics)

    assert agg["gate_split"] == {"spec_prefill_applied": 1, "skipped": 2}
    assert agg["keep_rate_r"]["n"] == 1  # only the compressed sample has a keep_rate
    assert agg["keep_rate_r"]["mean"] == 0.3


def test_aggregate_arm_metrics_gate_split_none_when_no_target_results(tmp_path):
    """--dry-run results (timing.jsonl only, no predictions) should report
    gate_split as None, not a misleading {0, 0}."""
    arm_dir = tmp_path / "A"
    _write_jsonl_rows([_timing_row("s1")], arm_dir / "timing.jsonl")

    metrics = load_arm_metrics(arm_dir)
    agg = aggregate_arm_metrics(metrics)

    assert agg["gate_split"] is None
    assert agg["n_target_answered"] == 0


def test_aggregate_arm_metrics_depth_and_fanout_distribution(tmp_path):
    arm_dir = tmp_path / "A"
    _write_jsonl_rows(
        [
            _timing_row("s1", max_realized_depth=0, n_direct_subcalls=0),
            _timing_row("s2", max_realized_depth=2, n_direct_subcalls=5),
            _timing_row("s3", max_realized_depth=1, n_direct_subcalls=2),
        ],
        arm_dir / "timing.jsonl",
    )

    metrics = load_arm_metrics(arm_dir)
    agg = aggregate_arm_metrics(metrics)

    assert agg["realized_depth"]["max"] == 2.0
    assert agg["realized_fanout"]["max"] == 5.0
    assert agg["realized_depth"]["n"] == 3


# ---------------------------------------------------------------------------
# score_accuracy
# ---------------------------------------------------------------------------


def test_score_accuracy_niah_recall(tmp_path):
    arm_dir = tmp_path / "A"
    _write_jsonl_rows([_timing_row("n1")], arm_dir / "timing.jsonl")
    _write_jsonl_rows([_prediction_row("n1", pred="the value is 999888777")], arm_dir / "predictions.jsonl")

    sample = EvalSample(
        id="n1", source="synthetic_niah", context="...", question="find it", answer="",
        extra={"needles": [{"label": "NIAH_SECRET_1", "value": "999888777"}]},
    )

    metrics = load_arm_metrics(arm_dir)
    acc = score_accuracy(metrics, {"n1": sample})

    assert acc["synthetic_niah_mean_recall"] == 1.0
    assert acc["synthetic_niah_n"] == 1


def test_score_accuracy_longbench_substring_match(tmp_path):
    arm_dir = tmp_path / "A"
    _write_jsonl_rows([_timing_row("lb1")], arm_dir / "timing.jsonl")
    _write_jsonl_rows(
        [_prediction_row("lb1", pred="Based on the excerpts, the capital of France is Paris.")],
        arm_dir / "predictions.jsonl",
    )

    sample = EvalSample(
        id="lb1", source="longbench_v2_long", context="...", question="What is the capital?", answer="B",
        extra={"choices": ["London", "Paris", "Berlin", "Madrid"]},
    )

    metrics = load_arm_metrics(arm_dir)
    acc = score_accuracy(metrics, {"lb1": sample})

    assert acc["longbench_v2_approx_accuracy"] == 1.0
    assert acc["longbench_v2_n"] == 1


def test_score_accuracy_longbench_wrong_answer_not_matched(tmp_path):
    arm_dir = tmp_path / "A"
    _write_jsonl_rows([_timing_row("lb1")], arm_dir / "timing.jsonl")
    _write_jsonl_rows([_prediction_row("lb1", pred="The capital is Berlin.")], arm_dir / "predictions.jsonl")

    sample = EvalSample(
        id="lb1", source="longbench_v2_long", context="...", question="What is the capital?", answer="B",
        extra={"choices": ["London", "Paris", "Berlin", "Madrid"]},
    )

    metrics = load_arm_metrics(arm_dir)
    acc = score_accuracy(metrics, {"lb1": sample})

    assert acc["longbench_v2_approx_accuracy"] == 0.0


def test_score_accuracy_matches_any_longbench_length_bucket_by_prefix(tmp_path):
    """eval_data/prep_longbench_v2_long.py's --length flag can produce
    longbench_v2_short/_medium/_long sources -- score_accuracy must match
    all of them (prefix check), not just the original "_long" exact
    string, or switching to a shorter bucket for faster iteration would
    silently stop being scored at all (falling into n_unscored_source)."""
    arm_dir = tmp_path / "A"
    _write_jsonl_rows([_timing_row("lb1")], arm_dir / "timing.jsonl")
    _write_jsonl_rows(
        [_prediction_row("lb1", pred="Based on the excerpts, the capital of France is Paris.")],
        arm_dir / "predictions.jsonl",
    )

    sample = EvalSample(
        id="lb1", source="longbench_v2_short", context="...", question="What is the capital?", answer="B",
        extra={"choices": ["London", "Paris", "Berlin", "Madrid"]},
    )

    metrics = load_arm_metrics(arm_dir)
    acc = score_accuracy(metrics, {"lb1": sample})

    assert acc["longbench_v2_approx_accuracy"] == 1.0
    assert acc["longbench_v2_n"] == 1
    assert acc["n_unscored_source"] == 0


def test_score_accuracy_counts_missing_predictions_and_unscored_sources(tmp_path):
    arm_dir = tmp_path / "A"
    _write_jsonl_rows([_timing_row("no_pred"), _timing_row("unknown_source")], arm_dir / "timing.jsonl")
    _write_jsonl_rows([_prediction_row("unknown_source")], arm_dir / "predictions.jsonl")  # no_pred has none

    sample = EvalSample(id="unknown_source", source="some_other_dataset", context="...", question="q", answer="")

    metrics = load_arm_metrics(arm_dir)
    acc = score_accuracy(metrics, {"unknown_source": sample})

    assert acc["n_no_prediction"] == 1  # no_pred
    assert acc["n_unscored_source"] == 1  # unknown_source's dataset source isn't recognized


# ---------------------------------------------------------------------------
# render_arm_summary (smoke -- just confirm it doesn't crash on realistic
# and edge-case aggregates)
# ---------------------------------------------------------------------------


def test_render_arm_summary_full_arm(tmp_path):
    arm_dir = tmp_path / "A"
    _write_jsonl_rows([_timing_row("s1")], arm_dir / "timing.jsonl")
    _write_jsonl_rows([_prediction_row("s1")], arm_dir / "predictions.jsonl")

    metrics = load_arm_metrics(arm_dir)
    agg = aggregate_arm_metrics(metrics)

    output = render_arm_summary("A", agg)
    assert "Arm A" in output
    assert "f (target share" in output


def test_render_arm_summary_dry_run_arm_no_crash(tmp_path):
    """A --dry-run arm has timing.jsonl but no target-stage results at all
    -- must render something sane, not crash on None formatting."""
    arm_dir = tmp_path / "A"
    _write_jsonl_rows([_timing_row("s1")], arm_dir / "timing.jsonl")

    metrics = load_arm_metrics(arm_dir)
    agg = aggregate_arm_metrics(metrics)

    output = render_arm_summary("A", agg)
    assert "Arm A" in output


def test_render_arm_summary_empty_arm_no_crash(tmp_path):
    arm_dir = tmp_path / "A"
    _write_jsonl_rows([], arm_dir / "timing.jsonl")

    metrics = load_arm_metrics(arm_dir)
    agg = aggregate_arm_metrics(metrics)

    output = render_arm_summary("A", agg)
    assert "Arm A" in output
