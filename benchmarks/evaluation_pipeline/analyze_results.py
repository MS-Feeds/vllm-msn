#!/usr/bin/env python3
"""Aggregates evaluation pipeline results into a summary report.

No accuracy scoring in this pipeline -- reads all rows written by
run_pipeline.py's present() (results/all_runs.csv) and produces
results/summary.md with three sections, structurally mirroring
../gemma4_moe_benchmarks/analyze_results.py (comparison table -> best
result callout -> key-contribution deltas), swapped to this pipeline's
metrics:

    - out tok/s / QPS instead of just out tok/s (still the headline
      throughput number, but requests_per_second matters too since
      prompt lengths vary a lot across datasets)
    - grouped by dataset instead of scenario
    - delta pairs are "spec decode at k=N" vs "S000 baseline (no spec
      decode)" instead of the FP8/CUDA-graph ablation pairs
    - an extra acceptance-rate-by-k-by-dataset table, since the
      motivating question here (EXPERIMENT_PLAN.md) is whether the
      draft-length/acceptance-rate tradeoff differs by prompt/response
      shape (long AIME reasoning vs. code vs. short GPQA answers)

Usage:
    python3 analyze_results.py [--csv results/all_runs.csv]
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

BASELINE_EXP_ID = "S000"


def load_csv(csv_path: Path) -> dict[str, list[dict]]:
    """Load CSV and group rows by (exp_id, dataset)."""
    groups: dict[str, list[dict]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = f"{row['exp_id']}_{row['dataset']}"
            groups.setdefault(key, []).append(row)
    return groups


def _nums(rows: list[dict], field: str) -> list[float]:
    vals = []
    for r in rows:
        v = r.get(field, "")
        if v not in ("", "None", None):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return vals


def _stat(rows: list[dict], field: str) -> tuple[float | None, float | None]:
    vals = _nums(rows, field)
    if not vals:
        return None, None
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, stdev


def summarize_group(rows: list[dict]) -> dict:
    """Compute mean +/- stdev for numeric metrics across reps.
    acceptance_rate/mean_accept_length are None for spec_decode=False
    rows (nothing to average) -- _stat handles that the same way missing
    throughput numbers are handled in the reference script."""
    first = rows[0]
    output_tps_mean, output_tps_stdev = _stat(rows, "output_tps")
    qps_mean, qps_stdev = _stat(rows, "requests_per_second")
    acc_mean, acc_stdev = _stat(rows, "acceptance_rate")
    accept_len_mean, _ = _stat(rows, "mean_accept_length")
    return {
        "exp_id": first["exp_id"],
        "label": first["label"],
        "dataset": first["dataset"],
        "spec_decode": str(first["spec_decode"]).lower() in ("true", "1"),
        "mtp_k": first["mtp_k"],
        "reps": len(rows),
        "output_tps_mean": output_tps_mean,
        "output_tps_stdev": output_tps_stdev,
        "qps_mean": qps_mean,
        "qps_stdev": qps_stdev,
        "acceptance_rate_mean": acc_mean,
        "acceptance_rate_stdev": acc_stdev,
        "mean_accept_length_mean": accept_len_mean,
    }


def render_table(summaries: list[dict], dataset: str) -> str:
    rows = [s for s in summaries if s["dataset"] == dataset]
    rows.sort(key=lambda r: r["exp_id"])
    if not rows:
        return f"_No data for dataset {dataset}_\n"

    baseline = next((r["output_tps_mean"] for r in rows if r["exp_id"] == BASELINE_EXP_ID), None)

    lines: list[str] = []
    lines.append(f"### Dataset: {dataset}\n")
    lines.append(
        "| Exp | Label | k | out tok/s | +/-sigma | QPS | vs S000 | "
        "acceptance rate | mean accept len |"
    )
    lines.append("|-----|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in rows:
        otps = r["output_tps_mean"]
        otps_s = r["output_tps_stdev"] or 0.0
        otps_str = f"{otps:.1f}" if otps is not None else "FAIL"
        otps_sig = f"{otps_s:.1f}" if otps is not None else "-"
        vs_base = f"{otps / baseline:.3f}x" if (otps and baseline) else "-"
        qps_str = f"{r['qps_mean']:.3f}" if r["qps_mean"] is not None else "-"
        acc = r["acceptance_rate_mean"]
        acc_str = f"{acc:.3f}" if acc is not None else "n/a"
        accept_len = r["mean_accept_length_mean"]
        accept_len_str = f"{accept_len:.2f}" if accept_len is not None else "n/a"
        k_str = r["mtp_k"] if r["spec_decode"] else "-"

        lines.append(
            f"| {r['exp_id']} | {r['label'][:45]} | {k_str} | {otps_str} | {otps_sig} | "
            f"{qps_str} | {vs_base} | {acc_str} | {accept_len_str} |"
        )

    best = max((r for r in rows if r["output_tps_mean"]), key=lambda r: r["output_tps_mean"], default=None)
    if best:
        lines.append(f"\n**Best {dataset} result**: {best['exp_id']} -- {best['output_tps_mean']:.1f} output tok/s")
        if baseline:
            lines.append(f"  Speedup vs S000 baseline: {best['output_tps_mean'] / baseline:.3f}x")
    lines.append("")
    return "\n".join(lines)


def delta_str(summaries: list[dict], dataset: str, eid_on: str, eid_off: str, label: str) -> str:
    def get_tps(eid: str) -> float | None:
        key = f"{eid}_{dataset}"
        s = next((x for x in summaries if f"{x['exp_id']}_{x['dataset']}" == key), None)
        return s["output_tps_mean"] if s else None

    on_tps = get_tps(eid_on)
    off_tps = get_tps(eid_off)
    if on_tps and off_tps:
        delta = on_tps - off_tps
        pct = (delta / off_tps) * 100
        return f"  {label:<28} : {delta:+.1f} tok/s ({pct:+.1f}%)"
    return f"  {label:<28} : data missing ({eid_on} or {eid_off})"


def render_acceptance_by_k_table(summaries: list[dict], datasets: list[str], spec_exp_ids: list[str]) -> str:
    """Cross-dataset acceptance-rate/mean-accept-length by k -- answers
    whether the speculative-decoding tradeoff differs by prompt/response
    shape (see EXPERIMENT_PLAN.md)."""
    lines = ["### Acceptance rate by k, across datasets\n"]
    k_by_exp: dict[str, str] = {}
    for s in summaries:
        if s["exp_id"] in spec_exp_ids:
            k_by_exp[s["exp_id"]] = s["mtp_k"]
    header = "| Dataset | " + " | ".join(f"k={k_by_exp.get(eid, '?')}" for eid in spec_exp_ids) + " |"
    lines.append(header)
    lines.append("|---------|" + "---|" * len(spec_exp_ids))

    for dataset in datasets:
        cells = []
        for eid in spec_exp_ids:
            match = next(
                (
                    s
                    for s in summaries
                    if s["exp_id"] == eid and s["dataset"] == dataset
                ),
                None,
            )
            acc = match["acceptance_rate_mean"] if match else None
            cells.append(f"{acc:.3f}" if acc is not None else "n/a")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/all_runs.csv")
    ap.add_argument("--out", default="results/summary.md")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run run_experiments.sh first.")
        raise SystemExit(1)

    groups = load_csv(csv_path)
    if not groups:
        print("ERROR: CSV is empty or malformed.")
        raise SystemExit(1)

    summaries = [summarize_group(rows) for rows in groups.values()]
    datasets = sorted({s["dataset"] for s in summaries})
    exp_ids = sorted({s["exp_id"] for s in summaries})
    spec_exp_ids = sorted(
        (eid for eid in exp_ids if eid != BASELINE_EXP_ID),
        key=lambda eid: next(
            (s["mtp_k"] for s in summaries if s["exp_id"] == eid), "0"
        ),
    )

    md_lines: list[str] = [
        "# Gemma 4 MoE Speculative-Decoding Throughput Summary",
        "",
        "Generated by `analyze_results.py` from `results/all_runs.csv`. "
        "No accuracy scoring -- throughput and spec-decode behavior only.",
        "",
        "## Results by dataset",
        "",
    ]

    for ds in datasets:
        md_lines.append(render_table(summaries, ds))

    md_lines.append("---")
    md_lines.append(render_acceptance_by_k_table(summaries, datasets, spec_exp_ids))

    md_lines.append("---")
    md_lines.append("## Key configuration contribution (mean across reps)")
    md_lines.append("")
    md_lines.append(f"Delta vs {BASELINE_EXP_ID} (no speculative decoding), per dataset:")
    md_lines.append("")

    for ds in datasets:
        md_lines.append(f"**{ds}:**")
        for eid in spec_exp_ids:
            label = next((s["label"] for s in summaries if s["exp_id"] == eid), eid)
            md_lines.append(delta_str(summaries, ds, eid, BASELINE_EXP_ID, f"{label} ({eid}-{BASELINE_EXP_ID})"))
        md_lines.append("")

    md_content = "\n".join(md_lines)
    print(md_content)

    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_content, encoding="utf-8")
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
