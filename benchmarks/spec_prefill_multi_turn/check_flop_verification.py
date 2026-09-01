#!/usr/bin/env python3
"""Grades the FLOP model's end-to-end verification run against
`results/all_runs.csv`.

`validate_flops_model.py` checks the model against the HARDWARE (profiler
FLOPs, measured GPU time). This checks the same model against ITSELF once
`predict_scbench.py` has driven it over real conversations: that the stages
sum to the total, that the excl-turn-0 partition is real rather than empty
columns, and -- the one that actually bites -- that the sparse path's
accounting collapses onto the baseline's when the gather is a provable
no-op.

Deliberately reads only the CSV, so it needs no GPU and can be run on a
laptop against a copied-back results file.
"""

import argparse
import csv
import sys
from pathlib import Path

STAGES = ("spec_prefill", "spec_lookahead", "spec_scoring",
          "target_prefill", "target_decode")
SPECULATOR_STAGES = ("spec_prefill", "spec_lookahead", "spec_scoring")


def _f(row, key):
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def latest_rows(csv_path):
    """Most recent row per exp_id -- the file is append-only, so a re-run
    leaves the earlier attempt in place and a naive scan would grade it."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_id = {}
    for row in rows:
        by_id[row.get("exp_id", "")] = row
    return by_id


def print_breakdown(exp_id, row):
    print("\n--- {} ({} conv, {} turns) ---".format(
        exp_id, row.get("num_conversations") or "?",
        row.get("num_turns") or "?"))
    print("  {:>16} | {:>12} | {:>12}".format("stage", "all turns", "excl turn 0"))
    for stage in STAGES:
        allt = _f(row, stage + "_tflops_per_turn_mean")
        excl = _f(row, stage + "_tflops_per_turn_excl_turn0_mean")
        print("  {:>16} | {:>12} | {:>12}".format(
            stage,
            "--" if allt is None else "{:,.2f}".format(allt),
            "--" if excl is None else "{:,.2f}".format(excl)))
    for key in ("total_tflops_per_turn_mean",
                "total_tflops_per_turn_excl_turn0_mean",
                "total_tflops", "speculator_flops_fraction",
                "speculator_flops_fraction_excl_turn0",
                "spec_prefill_share_of_speculator",
                "spec_prefill_share_of_speculator_excl_turn0",
                "achieved_tflops_per_s", "mfu"):
        val = _f(row, key)
        print("  {:>44}: {}".format(
            key, "--" if val is None else "{:,.4f}".format(val)))


def check(label, ok, detail):
    print("  [{}] {}".format("PASS" if ok else "FAIL", label))
    if detail:
        print("         " + detail)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/all_runs.csv")
    ap.add_argument("--baseline", default="M000")
    ap.add_argument("--control", default="SPARSE-k100-g32-control")
    ap.add_argument("--sparse", default="SPARSE-k20-g32-masked")
    ap.add_argument("--peak-tflops", type=float, default=None)
    ap.add_argument("--tolerance-pct", type=float, default=2.0,
                    help="Band for the stage-sum and control-vs-baseline checks.")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print("no such file: {}".format(path))
        return 2
    by_id = latest_rows(path)

    wanted = [args.baseline, args.control, args.sparse]
    missing = [e for e in wanted if e not in by_id]
    present = [e for e in wanted if e in by_id]
    for exp_id in present:
        print_breakdown(exp_id, by_id[exp_id])

    print("\n=== checks ===")
    ok = True
    if missing:
        ok = check("all three rows present", False,
                   "missing from {}: {} -- the run did not get that far, "
                   "or used other --exp ids.".format(path, missing))

    # 1. Stages must sum to the reported total, in BOTH partitions. Catches a
    #    stage that is silently not being accumulated.
    for exp_id in present:
        row = by_id[exp_id]
        for suffix, name in (("", "all turns"), ("_excl_turn0", "excl turn 0")):
            parts = [_f(row, "{}_tflops_per_turn{}_mean".format(s, suffix))
                     for s in STAGES]
            total = _f(row, "total_tflops_per_turn{}_mean".format(suffix))
            if total is None or any(p is None for p in parts):
                ok = check("{}: stages populated ({})".format(exp_id, name), False,
                           "one or more FLOP columns are empty -- the model "
                           "config was unavailable, so nothing was charged.")
                continue
            summed = sum(parts)
            drift = abs(summed - total) / total * 100 if total else 0.0
            ok = check("{}: stages sum to total ({})".format(exp_id, name),
                       drift <= args.tolerance_pct,
                       "sum={:,.2f} total={:,.2f} drift={:.2f}%".format(
                           summed, total, drift)) and ok

    # 2. The baseline must charge NOTHING to the speculator: it has none.
    if args.baseline in by_id:
        row = by_id[args.baseline]
        vals = {s: _f(row, s + "_tflops_per_turn_mean") for s in SPECULATOR_STAGES}
        nonzero = {s: v for s, v in vals.items() if v}
        ok = check("{}: no speculator FLOPs charged".format(args.baseline),
                   not nonzero,
                   "nonzero: {}".format(nonzero) if nonzero else "") and ok

    # 3. THE one that matters. keep=100% makes the gather a provable no-op, so
    #    the target-side cost must land on the baseline's. If it does not, the
    #    sparse accounting is wrong independently of the model or the hardware.
    if args.baseline in by_id and args.control in by_id:
        base, ctrl = by_id[args.baseline], by_id[args.control]
        for stage in ("target_prefill", "target_decode"):
            b = _f(base, stage + "_tflops_per_turn_mean")
            c = _f(ctrl, stage + "_tflops_per_turn_mean")
            if b is None or c is None or not b:
                ok = check("control vs baseline: {}".format(stage), False,
                           "missing on one side") and ok
                continue
            drift = abs(c - b) / b * 100
            ok = check("control vs baseline: {} agree".format(stage),
                       drift <= args.tolerance_pct,
                       "baseline={:,.2f} control={:,.2f} drift={:.2f}%".format(
                           b, c, drift)) and ok

    # 4. The sparse row must actually be CHEAPER on the target than the
    #    control, or the gather is not biting and the row is measuring nothing.
    if args.control in by_id and args.sparse in by_id:
        c = _f(by_id[args.control], "target_decode_tflops_per_turn_mean")
        s = _f(by_id[args.sparse], "target_decode_tflops_per_turn_mean")
        if c and s:
            ok = check("sparse target_decode < control target_decode",
                       s < c,
                       "control={:,.2f} sparse={:,.2f} ({:.1f}% cheaper)".format(
                           c, s, (1 - s / c) * 100)) and ok

    # 5. Fractions must be fractions.
    for exp_id in present:
        row = by_id[exp_id]
        for key in ("speculator_flops_fraction",
                    "speculator_flops_fraction_excl_turn0",
                    "spec_prefill_share_of_speculator",
                    "spec_prefill_share_of_speculator_excl_turn0"):
            val = _f(row, key)
            if val is None:
                continue
            ok = check("{}: {} in [0,1]".format(exp_id, key),
                       0.0 <= val <= 1.0, "value={}".format(val)) and ok

    # 6. Above-peak is arithmetically impossible, so it proves over-counting.
    if args.peak_tflops:
        for exp_id in present:
            val = _f(by_id[exp_id], "achieved_tflops_per_s")
            if val is None:
                continue
            ok = check("{}: below peak".format(exp_id),
                       val <= args.peak_tflops,
                       "achieved={:,.2f} peak={:,.0f} MFU={:.1%}".format(
                           val, args.peak_tflops, val / args.peak_tflops)) and ok

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
