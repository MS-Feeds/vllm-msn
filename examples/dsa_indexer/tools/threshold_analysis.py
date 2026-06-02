#!/usr/bin/env python3
"""Generate threshold recommendation table for DeepSeek Top-K pruning.

Computes P(score < threshold) for a range of threshold values, per layer.
Answers Joseph's question: "Can we prune values under 0.5?"

Usage:
    python tools/threshold_analysis.py indexer_logits/run01/indexer_logits_rank0.npz

Output:
    Markdown tables showing per-layer pruning recommendations.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def compute_cdf(hist: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute CDF from histogram.

    Returns:
        (edges, cdf): bin edges and cumulative probabilities
    """
    bins = len(hist)
    edges = np.linspace(lo, hi, bins + 1)
    total = hist.sum()
    if total == 0:
        return edges, np.zeros(bins + 1)
    cum = np.concatenate([[0], np.cumsum(hist)])
    cdf = cum / total
    return edges, cdf


def find_threshold_prob(hist: np.ndarray, lo: float, hi: float, threshold: float) -> float:
    """Find P(x < threshold) from histogram."""
    edges, cdf = compute_cdf(hist, lo, hi)
    idx = np.searchsorted(edges, threshold)
    if idx >= len(cdf):
        return 1.0
    return float(cdf[idx])


def recommend_threshold(hist: np.ndarray, lo: float, hi: float, max_loss: float = 0.01) -> float | None:
    """Find highest threshold where P(x < threshold) < max_loss.

    Returns None if no safe threshold exists.
    """
    edges, cdf = compute_cdf(hist, lo, hi)
    # Find largest threshold where CDF < max_loss
    valid = cdf < max_loss
    if not valid.any():
        return None
    return float(edges[np.where(valid)[0][-1]])


def format_percent(p: float) -> str:
    """Format probability as percentage with appropriate precision."""
    if p < 0.001:
        return f"{p*100:.2f}%"
    elif p < 0.01:
        return f"{p*100:.2f}%"
    elif p < 0.1:
        return f"{p*100:.1f}%"
    else:
        return f"{p*100:.0f}%"


def analyze_thresholds(npz_path: Path, max_loss: float = 0.01):
    """Generate threshold analysis tables."""
    d = np.load(npz_path)
    layers = d['layer_ids']
    bins = int(d['bins'])
    lo, hi = float(d['range_lo']), float(d['range_hi'])

    # Thresholds to test
    thresholds = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]

    print("# DeepSeek Top-K Threshold Analysis")
    print()
    print(f"**Model**: DeepSeek V2/V3 (assumed)")
    print(f"**Data source**: `{npz_path.name}`")
    print(f"**Max acceptable loss**: {max_loss*100:.1f}% (P(score < threshold) < {max_loss})")
    print(f"**Value range**: [{lo}, {hi}]")
    print()

    for phase in ('prefill', 'decode'):
        hist = d[f'hist_{phase}']  # [num_layers, bins]
        count = d[f'count_{phase}']

        if count.sum() == 0:
            print(f"## {phase.upper()}")
            print()
            print("*No data captured for this phase.*")
            print()
            continue

        print(f"## {phase.upper()}")
        print()
        print("P(score < threshold) — percentage of expert scores below threshold")
        print()

        # Build table header
        header = ["Layer"]
        for t in thresholds:
            if t < 1:
                header.append(f"< {t:.2f}")
            else:
                header.append(f"< {t:.0f}")
        header.append("Recommendation")

        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["------:" if i > 0 else "------" for i in range(len(header))]) + "|")

        for i, layer_id in enumerate(layers):
            if count[i] == 0:
                continue

            h = hist[i]
            row = [f"L{layer_id:02d}"]

            # Compute P(x < threshold) for each threshold
            for thresh in thresholds:
                p = find_threshold_prob(h, lo, hi, thresh)
                row.append(format_percent(p))

            # Recommendation: highest safe threshold
            rec_thresh = recommend_threshold(h, lo, hi, max_loss)
            if rec_thresh is not None and rec_thresh > 0:
                row.append(f"Prune < {rec_thresh:.3f}")
            else:
                row.append("No safe threshold")

            print("| " + " | ".join(row) + " |")

        print()

        # Aggregate statistics
        print(f"### {phase.capitalize()} Summary")
        print()

        valid_layers = [i for i, c in enumerate(count) if c > 0]
        if not valid_layers:
            print("*No valid layers.*")
            print()
            continue

        # Mean P(x < threshold) across layers
        print("**Mean P(score < threshold) across all layers:**")
        print()
        for thresh in thresholds:
            probs = [find_threshold_prob(hist[i], lo, hi, thresh) for i in valid_layers]
            mean_p = np.mean(probs)
            min_p = np.min(probs)
            max_p = np.max(probs)
            print(f"- `< {thresh:.2f}`: {format_percent(mean_p)} "
                  f"(range: {format_percent(min_p)} – {format_percent(max_p)})")
        print()

        # Safe threshold distribution
        rec_thresholds = []
        for i in valid_layers:
            rec = recommend_threshold(hist[i], lo, hi, max_loss)
            if rec is not None and rec > 0:
                rec_thresholds.append(rec)

        if rec_thresholds:
            print(f"**Recommended thresholds (< {max_loss*100:.1f}% loss):**")
            print()
            print(f"- Minimum (most conservative): {min(rec_thresholds):.3f}")
            print(f"- Median: {np.median(rec_thresholds):.3f}")
            print(f"- Maximum (most aggressive): {max(rec_thresholds):.3f}")
            print()
            print(f"**Suggestion**: Use threshold = {min(rec_thresholds):.3f} "
                  f"to be safe across all layers.")
        else:
            print(f"**No safe thresholds found** (all layers have > {max_loss*100:.1f}% "
                  f"of scores near zero).")

        print()

    # Cross-benchmark comparison (if available)
    # This requires storing benchmark labels during capture (future enhancement)
    print("## Interpretation")
    print()
    print("**For Joseph's question: \"Can we prune values under 0.5?\"**")
    print()

    # Check P(x < 0.5) across all layers and phases
    all_p_05 = []
    for phase in ('prefill', 'decode'):
        hist = d[f'hist_{phase}']
        count = d[f'count_{phase}']
        if count.sum() == 0:
            continue
        for i, c in enumerate(count):
            if c > 0:
                p = find_threshold_prob(hist[i], lo, hi, 0.5)
                all_p_05.append(p)

    if all_p_05:
        mean_loss = np.mean(all_p_05)
        max_loss_layer = np.max(all_p_05)

        if max_loss_layer < 0.01:
            print(f"✓ **YES**: Pruning < 0.5 is safe. "
                  f"Average loss: {format_percent(mean_loss)}, "
                  f"worst layer: {format_percent(max_loss_layer)}.")
        elif max_loss_layer < 0.05:
            print(f"⚠ **MAYBE**: Pruning < 0.5 has low but non-negligible loss. "
                  f"Average: {format_percent(mean_loss)}, "
                  f"worst layer: {format_percent(max_loss_layer)}. "
                  f"Recommend validation on accuracy benchmarks.")
        else:
            print(f"✗ **NO**: Pruning < 0.5 would discard {format_percent(mean_loss)} "
                  f"of scores on average (up to {format_percent(max_loss_layer)} "
                  f"in worst layer). This is likely too aggressive.")
    else:
        print("*Insufficient data to answer.*")

    print()

    print("## Notes")
    print()
    print("- These are **post-softmax** probabilities from the expert gating network.")
    print("- Values represent the probability assigned to each expert for routing.")
    print("- Top-K selection typically picks 6-8 experts with highest probabilities.")
    print("- Pruning thresholds should be validated on downstream accuracy metrics.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'npz_file',
        type=Path,
        help='Path to indexer_logits_rank*.npz file'
    )
    parser.add_argument(
        '--max-loss',
        type=float,
        default=0.01,
        help='Maximum acceptable loss fraction (default: 0.01 = 1%%)'
    )

    args = parser.parse_args()

    if not args.npz_file.exists():
        print(f"ERROR: File not found: {args.npz_file}", file=sys.stderr)
        return 1

    analyze_thresholds(args.npz_file, args.max_loss)
    return 0


if __name__ == '__main__':
    sys.exit(main())
