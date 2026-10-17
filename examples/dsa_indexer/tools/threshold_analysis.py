#!/usr/bin/env python3
"""Dynamic-range analysis for the lightning indexer's raw MQA logits (Stream B).

For each layer and phase, finds the tightest *symmetric* range [-r, r] that
captures a target fraction of the observed raw-logit distribution (e.g.
99.9%), computed directly from the Stream B histogram. Useful for deciding
how tight a future quantization/storage range could be, relative to the
blanket capture range (typically [-50, 50]).

This is NOT a top-k pruning-safety analysis. Whether a given precision loss
would flip the top-k selection near the cutoff depends on the *relative gap*
between the k-th and (k+1)-th largest logit within each query row (Stream
A) -- information a Stream-B-only capture (a marginal histogram pooled
across all rows) does not contain. That's future work once Streams A/C/D
are back in scope; don't read pruning-safety conclusions into this output.

Usage:
    python tools/threshold_analysis.py indexer_logits/<run_id>/indexer_logits_rank0.npz
    python tools/threshold_analysis.py <path>.npz --coverages 0.99 0.999 0.9999
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Maximum representable magnitude per format (kept in sync with the same
# table in plot_indexer_logits.py; duplicated rather than imported so this
# script stays numpy-only and doesn't pick up a matplotlib dependency).
FORMAT_MAXES: list[tuple[str, float]] = [
    ("FP8 e4m3", 448.0),
    ("FP8 e5m2", 57344.0),
    ("FP16", 65504.0),
]


def compute_cdf(hist: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """CDF over histogram bin edges. Returns (edges, cdf)."""
    bins = len(hist)
    edges = np.linspace(lo, hi, bins + 1)
    total = hist.sum()
    if total == 0:
        return edges, np.zeros(bins + 1)
    cum = np.concatenate([[0], np.cumsum(hist)])
    return edges, cum / total


def quantile_from_cdf(edges: np.ndarray, cdf: np.ndarray, q: float) -> float:
    """Inverse CDF: smallest edge x where CDF(x) >= q."""
    idx = np.searchsorted(cdf, q)
    idx = min(idx, len(edges) - 1)
    return float(edges[idx])


def symmetric_range_for_coverage(hist: np.ndarray, lo: float, hi: float, coverage: float) -> float:
    """Tightest symmetric range [-r, r] capturing `coverage` fraction of mass,
    trimming (1-coverage)/2 from each tail."""
    edges, cdf = compute_cdf(hist, lo, hi)
    if cdf[-1] == 0:
        return float("nan")
    tail = (1.0 - coverage) / 2.0
    x_lo = quantile_from_cdf(edges, cdf, tail)
    x_hi = quantile_from_cdf(edges, cdf, 1.0 - tail)
    return max(abs(x_lo), abs(x_hi))


def analyze(npz_path: Path, coverages: tuple[float, ...]) -> None:
    d = np.load(npz_path)
    layers = d["layer_ids"]
    lo, hi = float(d["range_lo"]), float(d["range_hi"])
    strictest = max(coverages)

    print("# Lightning-indexer raw-logit dynamic-range analysis")
    print()
    print(f"**Data source**: `{npz_path.name}`")
    print(f"**Captured histogram range**: [{lo:g}, {hi:g}]")
    print()
    print("Reports the tightest *symmetric* range needed to capture a given "
          "fraction of the observed raw MQA-logit distribution, per layer "
          "per phase. Not a top-k pruning-safety statement -- see module "
          "docstring.")
    print()

    for phase in ("prefill", "decode"):
        hist = d[f"hist_{phase}"]
        count = d[f"count_{phase}"]

        if count.sum() == 0:
            print(f"## {phase.upper()}")
            print()
            print("*No data captured for this phase.*")
            print()
            continue

        print(f"## {phase.upper()}")
        print()
        header = ["Layer"] + [f"±@{c*100:g}%" for c in coverages] + ["obs min", "obs max"]
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["------:" if i > 0 else "------" for i in range(len(header))]) + "|")

        per_layer_ranges = {c: [] for c in coverages}
        for i, layer_id in enumerate(layers):
            if count[i] == 0:
                continue
            h = hist[i]
            row = [f"L{layer_id:02d}"]
            for c in coverages:
                r = symmetric_range_for_coverage(h, lo, hi, c)
                per_layer_ranges[c].append(r)
                row.append(f"{r:.2f}")
            row.append(f"{float(d[f'min_{phase}'][i]):.2f}")
            row.append(f"{float(d[f'max_{phase}'][i]):.2f}")
            print("| " + " | ".join(row) + " |")
        print()

        print(f"### {phase.capitalize()} summary")
        print()
        print("Worst-case range across all layers (the range needed if a single "
              "shared quantization range must cover every layer):")
        print()
        worst = {}
        for c in coverages:
            vals = [v for v in per_layer_ranges[c] if not np.isnan(v)]
            worst[c] = max(vals) if vals else float("nan")
            print(f"- **{c*100:g}% coverage**: ±{worst[c]:.2f}")
        print()

        print(f"Format dynamic-range fit at {strictest*100:g}% coverage "
              f"(±{worst[strictest]:.2f}) -- does the format's max representable "
              "magnitude comfortably cover it?")
        print()
        for name, fmax in FORMAT_MAXES:
            r = worst[strictest]
            verdict = ("SAFE" if r < 0.5 * fmax else
                       "MARGINAL" if r < fmax else "OVERFLOW")
            print(f"- `{name}` (max={fmax:g}): {verdict}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("npz_file", type=Path, help="Path to indexer_logits_rank*.npz file")
    parser.add_argument(
        "--coverages", type=float, nargs="+", default=[0.99, 0.999, 0.9999],
        help="Target coverage fractions (default: 0.99 0.999 0.9999)",
    )
    args = parser.parse_args()

    if not args.npz_file.exists():
        print(f"ERROR: File not found: {args.npz_file}", file=sys.stderr)
        return 1

    analyze(args.npz_file, tuple(args.coverages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
