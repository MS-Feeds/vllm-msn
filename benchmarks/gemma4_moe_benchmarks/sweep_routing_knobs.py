#!/usr/bin/env python3
"""Sweep Gemma4 MoE routing knobs over the offline benchmark driver.

This wrapper launches ``bench_experiment.py`` multiple times with different
values for:

- ``VLLM_GEMMA4_FUSED_ROUTING_RETRY_INTERVAL``
- ``VLLM_GEMMA4_ROUTING_SCRATCH_CACHE_SIZE``

Each combination writes into its own results directory under ``--out-root``.
After all runs complete, the script produces a compact CSV summary with mean
throughput metrics per configuration.

Example:
    python benchmarks/gemma4_moe_benchmarks/sweep_routing_knobs.py \
        --exp E006 --scenario sc1 --reps 2 \
        --retry-intervals 64,128,512 \
        --scratch-cache-sizes 4,8,16
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def _summarize_rows(rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    total_tps = [float(row["total_tps"]) for row in rows]
    output_tps = [float(row["output_tps"]) for row in rows]
    req_per_s = [float(row["requests_per_second"]) for row in rows]
    elapsed = [float(row["elapsed_time"]) for row in rows]

    def mean(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    def stdev(values: list[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    first = rows[0]
    return {
        "exp_id": first["exp_id"],
        "scenario": first["scenario"],
        "retry_interval": int(first["gemma4_fused_routing_retry_interval"]),
        "scratch_cache_size": int(first["gemma4_routing_scratch_cache_size"]),
        "reps": len(rows),
        "mean_elapsed_time": round(mean(elapsed), 3),
        "stdev_elapsed_time": round(stdev(elapsed), 3),
        "mean_requests_per_second": round(mean(req_per_s), 4),
        "stdev_requests_per_second": round(stdev(req_per_s), 4),
        "mean_output_tps": round(mean(output_tps), 2),
        "stdev_output_tps": round(stdev(output_tps), 2),
        "mean_total_tps": round(mean(total_tps), 2),
        "stdev_total_tps": round(stdev(total_tps), 2),
        "results_dir": first.get("_results_dir", ""),
    }


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep Gemma4 fused-routing retry and scratch-cache knobs."
    )
    parser.add_argument("--exp", required=True,
                        help="Experiment ID(s), comma-separated, e.g. E006 or E006,E012")
    parser.add_argument("--scenario", default="sc1", help="Scenario passed to bench_experiment.py")
    parser.add_argument("--reps", type=int, default=2, help="Repetitions per combination")
    parser.add_argument(
        "--retry-intervals",
        default="64,128,512",
        help="Comma-separated retry interval values",
    )
    parser.add_argument(
        "--scratch-cache-sizes",
        default="4,8,16",
        help="Comma-separated scratch cache sizes",
    )
    parser.add_argument(
        "--out-root",
        default="results_routing_knob_sweep",
        help="Directory that receives one subdirectory per configuration",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch bench_experiment.py",
    )
    args = parser.parse_args()

    retry_intervals = _parse_int_list(args.retry_intervals)
    scratch_cache_sizes = _parse_int_list(args.scratch_cache_sizes)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    experiment_script = script_dir / "bench_experiment.py"
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = script_dir / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | str]] = []

    for retry_interval in retry_intervals:
        for scratch_cache_size in scratch_cache_sizes:
            combo_name = (
                f"retry_{retry_interval}_scratch_{scratch_cache_size}"
            )
            results_dir = out_root / combo_name
            results_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env["VLLM_GEMMA4_FUSED_ROUTING_RETRY_INTERVAL"] = str(retry_interval)
            env["VLLM_GEMMA4_ROUTING_SCRATCH_CACHE_SIZE"] = str(scratch_cache_size)
            env["BENCH_RESULTS_DIR"] = str(results_dir)

            cmd = [
                args.python,
                str(experiment_script),
                "--exp",
                args.exp,
                "--scenario",
                args.scenario,
                "--reps",
                str(args.reps),
            ]
            print(
                f"\n=== sweep {combo_name} exp={args.exp} scenario={args.scenario} reps={args.reps} ===",
                flush=True,
            )
            subprocess.run(cmd, cwd=repo_root, env=env, check=True)

            rows = _read_rows(results_dir / "all_runs.csv")
            if not rows:
                continue
            for row in rows:
                row["_results_dir"] = str(results_dir)
            summary_rows.append(_summarize_rows(rows))

    summary_path = out_root / "routing_knob_summary.csv"
    fieldnames = [
        "exp_id",
        "scenario",
        "retry_interval",
        "scratch_cache_size",
        "reps",
        "mean_elapsed_time",
        "stdev_elapsed_time",
        "mean_requests_per_second",
        "stdev_requests_per_second",
        "mean_output_tps",
        "stdev_output_tps",
        "mean_total_tps",
        "stdev_total_tps",
        "results_dir",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())