#!/usr/bin/env python3
"""Aggregates one arm's `runner/run_arm.py` output (`predictions.jsonl` +
`timing.jsonl`) into the metrics ../rlm_specprefill_ablation_plan.md's
METRICS section asks for: `f`-distribution, `T_RLM_root`/`T_RLM_subcalls`/
`T_REPL_compute`, realized depth/fan-out, call counts, keep rate `r`, gate
invoked/skipped split, TTFT, and (best-effort, see `score_accuracy`'s
docstring) accuracy/recall.

Pure Python -- reads JSONL, no vllm/GPU/network needed. Unit-tested against
synthetic JSONL fixtures shaped like `run_arm.py`'s real output (not
requiring an actual GPU run to have happened).

Explicitly NOT computed here (per the ablation doc's METRICS list): answer
faithfulness (needs an LLM-judge pass, out of scope for a pure aggregation
script) and GPU memory/throughput (needs external monitoring during the
run, not derivable from these JSONL files after the fact).

Usage:
    python3 analysis/aggregate_metrics.py --results-dir results --arms A,B,C \\
        --dataset eval_data/longbench_v2_long_samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))

from eval_data.schema import EvalSample, read_jsonl  # noqa: E402
from target_stage.vllm_offline_engine import percentile  # noqa: E402


def _read_jsonl_dicts(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Per-sample joined metrics.
# ---------------------------------------------------------------------------


@dataclass
class SampleMetrics:
    """One sample's data joined from timing.jsonl (always present, from the
    RLM-evidence stage) and predictions.jsonl (present only once the
    target-model stage has run for this sample -- None fields below mean
    "no target-stage result yet", e.g. a --dry-run or an over-budget
    sample skipped by target_stage/vllm_offline_engine.py::answer_batch)."""

    sample_id: str
    t_rlm_root: float
    t_repl_compute: float
    t_rlm_subcalls: float
    n_iterations: int
    n_direct_subcalls: int
    max_realized_depth: int
    total_calls: int
    generation_time_s: float | None
    ttft_ms: float | None
    keep_rate: float | None
    n_prompt_tokens_full: int | None
    n_prompt_tokens_kept: int | None
    pred: str | None

    @property
    def t_rlm_total(self) -> float:
        return self.t_rlm_root + self.t_repl_compute + self.t_rlm_subcalls

    @property
    def f(self) -> float | None:
        """f = T_target / T_total, per ../rlm_specprefill_ablation_plan.md's
        LATENCY MODEL. `generation_time_s` is `target_stage/vllm_offline_engine.py`'s
        real (not estimated) per-request wall-clock time -- see that
        module's `drive_engine_to_completion`/`answer_batch` docstrings.
        None if this sample has no target-stage result."""
        if self.generation_time_s is None:
            return None
        total = self.t_rlm_total + self.generation_time_s
        if total <= 0:
            return None
        return self.generation_time_s / total

    @property
    def spec_prefill_applied(self) -> bool:
        return self.n_prompt_tokens_kept is not None


def load_arm_metrics(arm_dir: Path) -> list[SampleMetrics]:
    """Joins `{arm_dir}/predictions.jsonl` (target-stage output, may not
    exist yet -- e.g. after a `--dry-run`) with `{arm_dir}/timing.jsonl`
    (RLM-evidence-stage output, always written by `run_arm.py`) by sample
    id. `timing.jsonl` is authoritative for which samples exist in this
    arm at all -- a sample without a `predictions.jsonl` row just gets
    `None` target-stage fields, not dropped."""
    predictions_by_id: dict[str, dict] = {}
    predictions_path = arm_dir / "predictions.jsonl"
    if predictions_path.exists():
        for row in _read_jsonl_dicts(predictions_path):
            predictions_by_id[row["id"]] = row

    timing_path = arm_dir / "timing.jsonl"
    if not timing_path.exists():
        raise FileNotFoundError(f"{timing_path} not found -- run runner/run_arm.py for this arm first.")

    metrics: list[SampleMetrics] = []
    for row in _read_jsonl_dicts(timing_path):
        pred_row = predictions_by_id.get(row["sample_id"])
        metrics.append(
            SampleMetrics(
                sample_id=row["sample_id"],
                t_rlm_root=row.get("t_rlm_root", 0.0),
                t_repl_compute=row.get("t_repl_compute", 0.0),
                t_rlm_subcalls=row.get("t_rlm_subcalls", 0.0),
                n_iterations=row.get("n_iterations", 0),
                n_direct_subcalls=row.get("n_direct_subcalls", 0),
                max_realized_depth=row.get("max_realized_depth", 0),
                total_calls=row.get("total_calls", 0),
                generation_time_s=pred_row.get("generation_time_s") if pred_row else None,
                ttft_ms=pred_row.get("ttft_ms") if pred_row else None,
                keep_rate=pred_row.get("keep_rate") if pred_row else None,
                n_prompt_tokens_full=pred_row.get("n_prompt_tokens_full") if pred_row else None,
                n_prompt_tokens_kept=pred_row.get("n_prompt_tokens_kept") if pred_row else None,
                pred=pred_row.get("pred") if pred_row else None,
            )
        )
    return metrics


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def percentile_summary(values: list[float]) -> dict[str, float | int | None]:
    """Reuses target_stage/vllm_offline_engine.py's own `percentile` helper
    (identical p90-style interpolation already used for budget sizing)
    rather than a second implementation. Per the ablation doc: "Report `f`
    as a distribution (percentiles), not a mean" -- applied uniformly to
    every timing/count metric here, not just `f`."""
    sorted_vals = sorted(values)
    return {
        "n": len(values),
        "mean": (sum(values) / len(values)) if values else None,
        "p50": percentile(sorted_vals, 0.5),
        "p90": percentile(sorted_vals, 0.9),
        "p99": percentile(sorted_vals, 0.99),
        "min": sorted_vals[0] if sorted_vals else None,
        "max": sorted_vals[-1] if sorted_vals else None,
    }


def aggregate_arm_metrics(metrics: list[SampleMetrics]) -> dict[str, Any]:
    f_values = [m.f for m in metrics if m.f is not None]
    ttft_values = [m.ttft_ms for m in metrics if m.ttft_ms is not None]
    keep_rates = [m.keep_rate for m in metrics if m.keep_rate is not None]
    generation_times = [m.generation_time_s for m in metrics if m.generation_time_s is not None]

    n_target_answered = sum(1 for m in metrics if m.generation_time_s is not None)
    n_spec_prefill_applied = sum(1 for m in metrics if m.spec_prefill_applied)

    return {
        "n_samples": len(metrics),
        "n_target_answered": n_target_answered,
        "f_distribution": percentile_summary(f_values),
        "t_rlm_root_s": percentile_summary([m.t_rlm_root for m in metrics]),
        "t_repl_compute_s": percentile_summary([m.t_repl_compute for m in metrics]),
        "t_rlm_subcalls_s": percentile_summary([m.t_rlm_subcalls for m in metrics]),
        "t_target_s": percentile_summary(generation_times),
        "ttft_ms": percentile_summary(ttft_values),
        "realized_depth": percentile_summary([float(m.max_realized_depth) for m in metrics]),
        "realized_fanout": percentile_summary([float(m.n_direct_subcalls) for m in metrics]),
        "total_calls": percentile_summary([float(m.total_calls) for m in metrics]),
        "keep_rate_r": percentile_summary(keep_rates),
        # Gate invoked/skipped split -- meaningful for Arm C (mixed plain/
        # SpecPrefill per query); for Arm A every answered sample is
        # "skipped" (never SpecPrefill) and for Arm B every one is
        # "invoked" (always-on), which is expected, not a bug -- the split
        # is still reported uniformly across arms so a caller comparing
        # A/B/C doesn't have to special-case which arm it's reading.
        "gate_split": (
            {
                "spec_prefill_applied": n_spec_prefill_applied,
                "skipped": n_target_answered - n_spec_prefill_applied,
            }
            if n_target_answered
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Accuracy / recall (best-effort, source-aware -- see docstring).
# ---------------------------------------------------------------------------


def score_accuracy(metrics: list[SampleMetrics], samples_by_id: dict[str, EvalSample]) -> dict[str, Any]:
    """Per-source scoring, since this project's eval sets mix three ground-
    truth shapes (see eval_data/schema.py):

    - `synthetic_niah` samples (eval_data/gen_synthetic_niah.py's free-form,
      scaled multi-needle generator) and `s_niah` samples (eval_data/
      gen_s_niah.py's RULER-faithful single-needle S-NIAH generator, Hsieh
      et al. 2024 -- source prefix-matched, not tied to any one variant)
      both have well-defined ground truth (exact needle values in
      `sample.extra["needles"]`) -- scored via needle recall, reusing
      calibration/transferability_check.py's `compute_recall`. Reported
      under SEPARATE keys (not blended together): S-NIAH is a distinct,
      citable RULER benchmark whose numbers shouldn't be averaged in with
      the free-form multi-needle generator's (different needle count,
      different haystack style, different O() scaling).
    - `longbench_v2_*` samples (source is bucket-specific -- `longbench_v2_long`,
      `_short`, or `_medium`, per eval_data/prep_longbench_v2_long.py's
      `--length` flag; matched here by prefix, not tied to any one bucket)
      have a lettered multiple-choice ground truth, but this project's target prompt
      (target_stage/vllm_offline_engine.py) asks for a free-text answer,
      not a letter (see that module's docstring for why -- RLM's evidence
      excerpts, not a rendered multiple-choice prompt, are what the target
      sees). Reusing `spec_prefill_llama/grade_longbench_v2.py`'s exact-
      letter-extraction scoring isn't a clean fit as a result. Scored here
      via a best-effort substring check (does the free-text answer contain
      the correct choice's TEXT) instead -- flagged explicitly as an
      approximation, not exact-letter grading. A more faithful comparison
      would either change the target prompt to request a letter for
      LongBench-derived samples specifically, or run a separate LLM-judge
      pass -- noted as follow-up work, not implemented here.
    """
    from calibration.transferability_check import compute_recall

    niah_recalls: list[float] = []
    s_niah_recalls: list[float] = []
    longbench_matches: list[bool] = []
    n_no_prediction = 0
    n_unscored_source = 0

    for m in metrics:
        if m.pred is None:
            n_no_prediction += 1
            continue
        sample = samples_by_id.get(m.sample_id)
        if sample is None:
            n_unscored_source += 1
            continue

        if sample.source == "synthetic_niah":
            needle_values = [n["value"] for n in sample.extra.get("needles", [])]
            niah_recalls.append(compute_recall(needle_values, m.pred))
        elif sample.source.startswith("s_niah"):
            needle_values = [n["value"] for n in sample.extra.get("needles", [])]
            s_niah_recalls.append(compute_recall(needle_values, m.pred))
        elif sample.source.startswith("longbench_v2"):
            choices = sample.extra.get("choices", [])
            letter_index = {"A": 0, "B": 1, "C": 2, "D": 3}.get(sample.answer)
            correct_text = (
                choices[letter_index] if letter_index is not None and letter_index < len(choices) else None
            )
            longbench_matches.append(bool(correct_text) and correct_text.lower() in m.pred.lower())
        else:
            n_unscored_source += 1

    return {
        "synthetic_niah_mean_recall": (sum(niah_recalls) / len(niah_recalls)) if niah_recalls else None,
        "synthetic_niah_n": len(niah_recalls),
        "s_niah_mean_recall": (sum(s_niah_recalls) / len(s_niah_recalls)) if s_niah_recalls else None,
        "s_niah_n": len(s_niah_recalls),
        "longbench_v2_approx_accuracy": (
            (sum(longbench_matches) / len(longbench_matches)) if longbench_matches else None
        ),
        "longbench_v2_n": len(longbench_matches),
        "longbench_v2_scoring_note": (
            "best-effort substring match, NOT grade_longbench_v2.py's exact-letter "
            "scoring -- see score_accuracy's docstring. Covers any LongBench-v2 "
            "length bucket (source prefix-matched), not just 'long' -- check "
            "the underlying samples' own sample.source / sample.extra['length'] "
            "if you need to know which bucket these numbers came from."
        ),
        "n_no_prediction": n_no_prediction,
        "n_unscored_source": n_unscored_source,
    }


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def render_arm_summary(arm: str, agg: dict[str, Any]) -> str:
    if agg["n_samples"] == 0:
        return f"=== Arm {arm} (0 samples) ==="

    lines = [f"=== Arm {arm} ({agg['n_samples']} sample(s), {agg['n_target_answered']} answered) ==="]
    f_dist = agg["f_distribution"]
    if f_dist["n"]:
        lines.append(f"  f (target share of latency): p50={f_dist['p50']:.3f} p90={f_dist['p90']:.3f} mean={f_dist['mean']:.3f}")
    else:
        lines.append("  f: no target-stage results yet (dry-run or all samples skipped)")
    lines.append(
        f"  T_RLM_root p50={agg['t_rlm_root_s']['p50']:.2f}s  "
        f"T_REPL_compute p50={agg['t_repl_compute_s']['p50']:.4f}s  "
        f"T_RLM_subcalls p50={agg['t_rlm_subcalls_s']['p50']:.2f}s"
    )
    if agg["t_target_s"]["n"]:
        lines.append(f"  T_target p50={agg['t_target_s']['p50']:.2f}s  TTFT p50={agg['ttft_ms']['p50']:.0f}ms")
    lines.append(
        f"  realized depth p90={agg['realized_depth']['p90']:.0f}  "
        f"fan-out p90={agg['realized_fanout']['p90']:.0f}  "
        f"total_calls p50={agg['total_calls']['p50']:.0f}"
    )
    if agg["keep_rate_r"]["n"]:
        lines.append(f"  keep rate r: p50={agg['keep_rate_r']['p50']:.2f} mean={agg['keep_rate_r']['mean']:.2f}")
    if agg["gate_split"]:
        lines.append(
            f"  gate split: {agg['gate_split']['spec_prefill_applied']} SpecPrefill-applied, "
            f"{agg['gate_split']['skipped']} skipped"
        )
    if "accuracy" in agg:
        acc = agg["accuracy"]
        if acc["synthetic_niah_n"]:
            lines.append(f"  NIAH mean recall: {acc['synthetic_niah_mean_recall']:.2%} (n={acc['synthetic_niah_n']})")
        if acc["s_niah_n"]:
            lines.append(f"  S-NIAH mean recall: {acc['s_niah_mean_recall']:.2%} (n={acc['s_niah_n']})")
        if acc["longbench_v2_n"]:
            lines.append(
                f"  LongBench-v2 approx. accuracy: {acc['longbench_v2_approx_accuracy']:.2%} "
                f"(n={acc['longbench_v2_n']}, best-effort scoring, bucket per sample.source, see note)"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Matches runner/run_arm.py's DEFAULT_RESULTS_DIR fix -- both must honor
    # $RLM_SPECPREFILL_RESULTS_DIR the same way, or a run_arm.py invocation
    # that picked it up (e.g. the Qwen-Coder pairing's results/qwen_coder/
    # namespacing) silently can't be found by this script's own default.
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(os.environ.get("RLM_SPECPREFILL_RESULTS_DIR", str(THIS_DIR / "results"))),
    )
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Eval-set JSONL for accuracy/recall scoring against ground truth (optional -- omit to skip accuracy scoring).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    samples_by_id: dict[str, EvalSample] = {}
    if args.dataset is not None:
        for sample in read_jsonl(args.dataset):
            samples_by_id[sample.id] = sample

    report: dict[str, Any] = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        arm_dir = args.results_dir / arm
        if not (arm_dir / "timing.jsonl").exists():
            print(f"[aggregate_metrics] skipping arm {arm}: no timing.jsonl found at {arm_dir}")
            continue
        metrics = load_arm_metrics(arm_dir)
        agg = aggregate_arm_metrics(metrics)
        if samples_by_id:
            agg["accuracy"] = score_accuracy(metrics, samples_by_id)
        report[arm] = agg
        print(render_arm_summary(arm, agg))
        print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[aggregate_metrics] wrote {args.output}")


if __name__ == "__main__":
    main()
