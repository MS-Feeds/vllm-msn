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
#: Wall-clock stages. `driver_overhead` is a residual, so these sum to the
#: turn clock by construction -- see timing_model.py.
TIME_STAGES = STAGES + ("driver_overhead",)


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
    leaves the earlier attempt in place and a naive scan would grade it.

    Returns the header too, because `csv.DictReader` gives `None` both for a
    column that is EMPTY and for one that does not exist in the file at all.
    Those are opposite diagnoses -- a real accounting bug versus a results
    file written by code that predates the column -- and this run hit exactly
    that ambiguity."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    by_id = {}
    for row in rows:
        by_id[row.get("exp_id", "")] = row
    return by_id, fieldnames


def report_missing_columns(fieldnames):
    """Which FLOP columns the file has no header for at all."""
    expected = []
    for suffix in ("", "_excl_turn0"):
        expected += ["{}_tflops_per_turn{}_mean".format(s, suffix) for s in STAGES]
        expected += ["total_tflops_per_turn{}_mean".format(suffix),
                     "speculator_flops_fraction{}".format(suffix),
                     "spec_prefill_share_of_speculator{}".format(suffix)]
        expected += ["{}_seconds_per_turn{}_mean".format(s, suffix)
                     for s in TIME_STAGES]
        expected += ["speculator_seconds_fraction{}".format(suffix)]
    absent = [c for c in expected if c not in fieldnames]
    if absent:
        print("\n=== columns ABSENT from the CSV header ===")
        print("  The file's HEADER predates these columns. That is a property")
        print("  of the results file, NOT of the code that produced the run:")
        print("  `ensure_csv_header` writes a header only when CREATING the")
        print("  file, so a CSV that already existed keeps its old header")
        print("  while new rows carry the full value count.")
        for col in absent:
            print("    " + col)
    return absent


def detect_column_shift(csv_path):
    """Rows carrying MORE values than the header has names.

    `csv.DictReader` files the overflow under the key `None`, so this is
    directly observable rather than inferred. It matters because the values
    that DO get names are then shifted: every column after the first added
    field reads its neighbour's number. Nothing about such a row looks
    wrong -- in the run this was written for, `achieved_tflops_per_s` read
    0.9987 (which was the prefill-share value) and `mfu` read 94.05 (which
    was the achieved throughput), both entirely plausible numbers.

    Not repairable by name: the names in the short header no longer point at
    the right values, so an affected row has to be re-measured."""
    shifted = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header_len = len(reader.fieldnames or [])
        for row in reader:
            overflow = row.get(None)
            if overflow:
                shifted.append((row.get("exp_id", "?"), row.get("ts", "?"),
                                header_len + len(overflow)))
    if shifted:
        print("\n=== rows whose VALUES are shifted against the header ===")
        print("  These rows have more values than the header has names, so")
        print("  every column after the first added field is reading its")
        print("  neighbour's value. They cannot be repaired by name -- the")
        print("  measurement has to be re-run. Affected:")
        for exp_id, ts, width in shifted:
            print("    {:<32} {}  ({} values vs {} header names)".format(
                exp_id, ts, width, header_len))
    return shifted


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

    print("  {:>16} | {:>12} | {:>12}   (seconds)".format(
        "stage", "all turns", "excl turn 0"))
    for stage in TIME_STAGES:
        allt = _f(row, stage + "_seconds_per_turn_mean")
        excl = _f(row, stage + "_seconds_per_turn_excl_turn0_mean")
        print("  {:>16} | {:>12} | {:>12}".format(
            stage,
            "--" if allt is None else "{:,.3f}".format(allt),
            "--" if excl is None else "{:,.3f}".format(excl)))
    for key in ("seconds_per_turn_mean", "seconds_per_turn_excl_turn0_mean",
                "speculator_seconds_fraction",
                "speculator_seconds_fraction_excl_turn0"):
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
    by_id, fieldnames = latest_rows(path)
    absent = report_missing_columns(fieldnames)
    shifted = detect_column_shift(path)

    wanted = [args.baseline, args.control, args.sparse]
    missing = [e for e in wanted if e not in by_id]
    present = [e for e in wanted if e in by_id]
    for exp_id in present:
        print_breakdown(exp_id, by_id[exp_id])

    print("\n=== checks ===")
    ok = True
    if shifted:
        ok = check("no rows are shifted against the header", False,
                   "{} row(s) affected -- every number reported below is "
                   "suspect. Re-run predict_scbench.py (it now migrates the "
                   "header on startup) and re-check.".format(len(shifted)))
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
                cols = ["{}_tflops_per_turn{}_mean".format(s, suffix)
                        for s in STAGES]
                cols.append("total_tflops_per_turn{}_mean".format(suffix))
                no_header = [c for c in cols if c not in fieldnames]
                if no_header:
                    detail = ("column(s) not in the CSV header at all: {} -- "
                              "stale code wrote this file, not a FLOP bug."
                              .format(no_header))
                else:
                    detail = ("column(s) present but EMPTY -- no turn with "
                              "that index was recorded, so `_stage_means` "
                              "returned None for every stage.")
                ok = check("{}: stages populated ({})".format(exp_id, name),
                           False, detail)
                continue
            summed = sum(parts)
            drift = abs(summed - total) / total * 100 if total else 0.0
            ok = check("{}: stages sum to total ({})".format(exp_id, name),
                       drift <= args.tolerance_pct,
                       "sum={:,.2f} total={:,.2f} drift={:.2f}%".format(
                           summed, total, drift)) and ok

    # 1b. Timing stages must sum to the measured turn clock. `driver_overhead`
    #     is a RESIDUAL, so this is an identity unless two stages overlap --
    #     i.e. it catches double-counting, which is the failure mode that
    #     would quietly re-create the misattribution this instrumentation
    #     exists to remove.
    for exp_id in present:
        row = by_id[exp_id]
        for suffix, clock, name in (
                ("", "seconds_per_turn_mean", "all turns"),
                ("_excl_turn0", "seconds_per_turn_excl_turn0_mean",
                 "excl turn 0")):
            parts = [_f(row, "{}_seconds_per_turn{}_mean".format(s, suffix))
                     for s in TIME_STAGES]
            measured = _f(row, clock)
            if measured is None or any(p is None for p in parts):
                cols = ["{}_seconds_per_turn{}_mean".format(s, suffix)
                        for s in TIME_STAGES]
                no_header = [c for c in cols if c not in fieldnames]
                detail = ("timing column(s) not in the CSV header: {} -- the "
                          "run predates the per-stage timings."
                          .format(no_header) if no_header else
                          "timing column(s) present but EMPTY")
                ok = check("{}: timing stages populated ({})".format(exp_id, name),
                           False, detail)
                continue
            summed = sum(parts)
            drift = abs(summed - measured) / measured * 100 if measured else 0.0
            ok = check("{}: timing stages sum to the turn clock ({})".format(
                           exp_id, name),
                       drift <= args.tolerance_pct,
                       "sum={:,.3f}s clock={:,.3f}s drift={:.2f}%".format(
                           summed, measured, drift)) and ok
            neg = [s for s, p in zip(TIME_STAGES, parts) if p is not None and p < 0]
            ok = check("{}: no negative timing stage ({})".format(exp_id, name),
                       not neg,
                       "negative: {} -- stages overlap, something is being "
                       "double-counted".format(neg) if neg else "") and ok

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

    # 5b. `achieved_tflops_per_s` is `total_tflops / elapsed_time`, and both
    #     inputs are in the same row -- so it can be recomputed rather than
    #     trusted. A mismatch means the throughput column is being derived
    #     from something other than what it claims.
    for exp_id in present:
        row = by_id[exp_id]
        total = _f(row, "total_tflops")
        elapsed = _f(row, "elapsed_time")
        reported = _f(row, "achieved_tflops_per_s")
        if total is None or elapsed is None or reported is None or not elapsed:
            continue
        recomputed = total / elapsed
        drift = (abs(recomputed - reported) / reported * 100) if reported else 0.0
        ok = check("{}: achieved_tflops_per_s == total_tflops/elapsed".format(exp_id),
                   drift <= args.tolerance_pct,
                   "reported={:,.2f} recomputed={:,.2f} "
                   "(total_tflops={:,.1f} elapsed={:,.1f}s) drift={:.1f}%".format(
                       reported, recomputed, total, elapsed, drift)) and ok

    # 6. Above-peak is arithmetically impossible, so it proves over-counting.
    #
    # Graded on `total_tflops / elapsed_time` RECOMPUTED from the row, never
    # on the `achieved_tflops_per_s` column. This check once passed on a run
    # whose throughput column read a bogus 1.00 TFLOP/s -- a falsification
    # bound that reads its input from a column it cannot verify will happily
    # confirm anything. The two inputs it uses instead are raw measurements
    # the driver could not have derived wrongly without check 1 also failing.
    if args.peak_tflops:
        for exp_id in present:
            row = by_id[exp_id]
            total = _f(row, "total_tflops")
            elapsed = _f(row, "elapsed_time")
            if total is None or elapsed is None or not elapsed:
                ok = check("{}: below peak".format(exp_id), False,
                           "cannot recompute throughput: total_tflops or "
                           "elapsed_time is missing") and ok
                continue
            val = total / elapsed
            ok = check("{}: below peak".format(exp_id),
                       val <= args.peak_tflops,
                       "achieved={:,.2f} peak={:,.0f} MFU={:.1%} "
                       "(recomputed, not read from the CSV column)".format(
                           val, args.peak_tflops, val / args.peak_tflops)) and ok

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
