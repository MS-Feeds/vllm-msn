"""Plot indexer logit captures produced by vllm._indexer_logger.

Usage:
    python plot_indexer_logits.py \
        --inputs /path/to/indexer_logits_rank*.npz \
        --out figures/ [--aggregate]
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Format precision table. ε is "smallest relative perturbation that can flip a
# comparison" — for a floating-point format this is roughly 2^-(mantissa_bits).
# For top-bits radix sort it's 2^-(bits_inspected).
FORMATS: list[tuple[str, float, str]] = [
    ("BF16",       2 ** -7,  "tab:blue"),    # 7 mantissa bits → ε = 2^-7 ≈ 7.8e-3 (matches doc §1)
    ("FP16",       2 ** -10, "tab:cyan"),    # 10 mantissa bits → ε = 2^-10 ≈ 9.8e-4
    ("FP8 e4m3",   1 / 8,    "tab:red"),     # 3 mantissa bits → ε = 1/8 = 0.125
    ("FP8 e5m2",   1 / 4,    "tab:orange"),  # 2 mantissa bits → ε = 1/4 = 0.25
    ("4-bit radix",1 / 16,   "tab:purple"),  # top 4 bits → ε = 1/16 = 0.0625
    ("8-bit radix",1 / 256,  "tab:green"),   # top 8 bits → ε = 1/256 ≈ 3.9e-3
]

# Maximum representable magnitude per format (for range/saturation analysis).
FORMAT_MAXES: list[tuple[str, float, str]] = [
    ("FP8 e4m3", 448.0,   "tab:red"),
    ("FP8 e5m2", 57344.0, "tab:orange"),
    ("FP16",     65504.0, "tab:cyan"),
]


# ---------------------------------------------------------------------------
# Loading / aggregation
# ---------------------------------------------------------------------------


def _load(paths: list[Path]) -> list[dict]:
    return [dict(np.load(p)) for p in paths]


def _sum_aggregate(captures: list[dict]) -> dict:
    """Sum histograms / counts across TP ranks; take min over mins, max over maxes."""
    if len(captures) == 1:
        return captures[0]
    out: dict = {}
    # Use layer_ids from the first file; assume all ranks share the same layer set.
    ref_layers = captures[0]["layer_ids"]
    for c in captures[1:]:
        if not np.array_equal(c["layer_ids"], ref_layers):
            raise ValueError("layer_ids differ across ranks; cannot aggregate naively.")
    out["layer_ids"] = ref_layers
    # Pass scalars (k_index_topk, gap_window, ...) through from first capture.
    for key in ("k_index_topk", "gap_window", "gap_eps_norm", "gap_bins",
                "gap_range_lo", "gap_range_hi", "bins", "range_lo", "range_hi",
                "sample_k", "radix_bits_sweep", "radix_bkt_bins",
                "radix_best_bit_windows"):
        if key in captures[0]:
            out[key] = captures[0][key]

    # Sum-style aggregates. BitEntropy uses bit_ prefix; everything else uses
    # the standard hist_/count_/sum_/zero_/... substring patterns.
    sum_keys = [k for k in captures[0].keys()
                if (any(s in k for s in ("hist_", "count_", "above_", "below_",
                                          "skipped_", "sum_", "zero_"))
                    or k.startswith("bit_"))
                and not k.endswith(("_min_prefill", "_max_prefill",
                                     "_min_decode", "_max_decode"))]
    for k in sum_keys:
        out[k] = sum(c[k] for c in captures)

    # Min / max aggregates. Matches both prefix-style ("gap_min_prefill",
    # "topk_min_prefill") and Range stream's flat names ("min_prefill").
    min_max_suffixes = ("_prefill", "_decode")
    for k in captures[0].keys():
        is_min = (k.startswith("min_") or "_min_" in k) and k.endswith(min_max_suffixes)
        is_max = (k.startswith("max_") or "_max_" in k) and k.endswith(min_max_suffixes)
        if is_min:
            out[k] = np.min(np.stack([c[k] for c in captures]), axis=0)
        elif is_max:
            out[k] = np.max(np.stack([c[k] for c in captures]), axis=0)

    # Reservoirs: just concatenate then random-subsample to original K.
    if "sample_k" in out:
        K = int(out["sample_k"])
        for k in ("gap_sample_prefill", "gap_sample_decode",
                  "sample_prefill", "sample_decode",
                  "topk_sample_prefill", "topk_sample_decode"):
            if k in captures[0]:
                stacked = np.concatenate([c[k] for c in captures], axis=1)
                rng = np.random.default_rng(0)
                L = stacked.shape[0]
                out_arr = np.full((L, K), np.nan, dtype=np.float32)
                for i in range(L):
                    finite = stacked[i][np.isfinite(stacked[i])]
                    if finite.size == 0:
                        continue
                    if finite.size <= K:
                        out_arr[i, : finite.size] = finite
                    else:
                        out_arr[i] = rng.choice(finite, size=K, replace=False)
                out[k] = out_arr
    return out


# ---------------------------------------------------------------------------
# Histogram statistics
# ---------------------------------------------------------------------------


def _bin_centers(lo: float, hi: float, n: int) -> np.ndarray:
    edges = np.linspace(lo, hi, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _quantile_from_hist(hist: np.ndarray, lo: float, hi: float, q: float) -> float:
    """Compute a quantile from a histogram with bins uniformly spread in [lo, hi]."""
    total = hist.sum()
    if total <= 0:
        return float("nan")
    edges = np.linspace(lo, hi, len(hist) + 1)
    cum = np.cumsum(hist)
    target = q * total
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(hist) - 1)
    left = cum[idx - 1] if idx > 0 else 0
    bin_lo, bin_hi = edges[idx], edges[idx + 1]
    if hist[idx] == 0:
        return float(bin_lo)
    frac = (target - left) / hist[idx]
    return float(bin_lo + frac * (bin_hi - bin_lo))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_stream_a_panels(d: dict, out_dir: Path) -> None:
    """Per-layer relative-gap histogram panels with format-ε vertical lines."""
    layers = d["layer_ids"]
    L = len(layers)
    bins = int(d["gap_bins"])
    lo = float(d["gap_range_lo"])
    hi = float(d["gap_range_hi"])
    centers = _bin_centers(lo, hi, bins)

    cols = 6
    rows = math.ceil(L / cols)
    for phase in ("prefill", "decode"):
        hists = d[f"gap_hist_{phase}"]
        if hists.sum() == 0:
            continue
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 1.8),
                                 sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)
        for i, layer in enumerate(layers):
            ax = axes[i]
            h = hists[i]
            if h.sum() > 0:
                ax.bar(centers, h, width=(hi - lo) / bins, color="steelblue",
                       linewidth=0, log=True)
            for name, eps, col in FORMATS:
                ax.axvline(eps, color=col, linestyle="--", linewidth=0.6, alpha=0.7)
            ax.set_title(f"L{layer}", fontsize=8)
            ax.set_xlim(lo, hi * 0.5)
            ax.tick_params(labelsize=6)
        for j in range(L, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"Gap stream — relative-gap histogram per layer ({phase})", fontsize=12)
        fig.supxlabel("relative_gap = (logit[k] − logit[k+1]) / |logit[k]|")
        fig.supylabel("count (log)")
        # Legend for format markers
        handles = [plt.Line2D([0], [0], color=col, linestyle="--", label=name)
                   for name, _, col in FORMATS]
        fig.legend(handles=handles, loc="lower center", ncol=len(FORMATS), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        fig.savefig(out_dir / f"stream_a_panels_{phase}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_p1_by_layer(d: dict, out_dir: Path) -> None:
    """Per-layer p1 (lower 1% quantile) of the relative-gap distribution."""
    layers = d["layer_ids"]
    lo = float(d["gap_range_lo"])
    hi = float(d["gap_range_hi"])
    p1_prefill = np.array([_quantile_from_hist(d["gap_hist_prefill"][i], lo, hi, 0.01)
                           for i in range(len(layers))])
    p1_decode = np.array([_quantile_from_hist(d["gap_hist_decode"][i], lo, hi, 0.01)
                          for i in range(len(layers))])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(layers, p1_prefill, "o-", color="navy", label="prefill p1", markersize=4)
    ax.plot(layers, p1_decode, "s-", color="darkorange", label="decode p1", markersize=4)
    for name, eps, col in FORMATS:
        ax.axhline(eps, color=col, linestyle="--", linewidth=0.8, alpha=0.7, label=f"ε {name}")
    ax.set_yscale("log")
    ax.set_xlabel("layer index")
    ax.set_ylabel("p1 of relative_gap (log)")
    ax.set_title("Gap stream: per-layer p1 vs candidate-format ε")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "stream_a_p1_by_layer.png", dpi=140)
    plt.close(fig)

    # Save raw p1 values for inspection.
    np.savetxt(
        out_dir / "stream_a_p1.csv",
        np.column_stack([layers, p1_prefill, p1_decode]),
        header="layer_id,p1_prefill,p1_decode", delimiter=",", comments="",
    )


def plot_safety_heatmap(d: dict, out_dir: Path) -> None:
    """Heatmap: rows = layers, cols = formats, color = p1 / ε.

    Cells > 3 are 'comfortably safe', 1-3 are 'marginal', < 1 are 'unsafe'.
    """
    layers = d["layer_ids"]
    lo = float(d["gap_range_lo"])
    hi = float(d["gap_range_hi"])
    p1 = {phase: np.array([_quantile_from_hist(d[f"gap_hist_{phase}"][i], lo, hi, 0.01)
                            for i in range(len(layers))])
          for phase in ("prefill", "decode")}

    for phase in ("prefill", "decode"):
        if not np.any(np.isfinite(p1[phase])):
            continue
        names = [f for f, _, _ in FORMATS]
        eps_arr = np.array([e for _, e, _ in FORMATS])
        ratio = p1[phase][:, None] / eps_arr[None, :]   # [L, F]

        fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.0), max(6, len(layers) * 0.18)))
        im = ax.imshow(np.log10(np.clip(ratio, 1e-6, 1e6)),
                       aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=6)
        ax.set_xlabel("candidate format")
        ax.set_ylabel("layer")
        ax.set_title(f"Safety = log10(p1 / ε)  ·  {phase}\n(red = unsafe, green = safe)")
        for i in range(len(layers)):
            for j in range(len(names)):
                ax.text(j, i, f"{ratio[i, j]:.1f}", ha="center", va="center",
                        fontsize=5, color="black")
        fig.colorbar(im, ax=ax, label="log10(safety)")
        fig.tight_layout()
        fig.savefig(out_dir / f"safety_heatmap_{phase}.png", dpi=140)
        plt.close(fig)


def plot_stream_b_panels(d: dict, out_dir: Path) -> None:
    layers = d["layer_ids"]
    L = len(layers)
    bins = int(d["bins"])
    lo = float(d["range_lo"])
    hi = float(d["range_hi"])
    centers = _bin_centers(lo, hi, bins)
    cols = 6
    rows = math.ceil(L / cols)
    for phase in ("prefill", "decode"):
        hists = d[f"hist_{phase}"]
        if hists.sum() == 0:
            continue
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 1.8),
                                 sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)
        for i, layer in enumerate(layers):
            ax = axes[i]
            h = hists[i]
            if h.sum() > 0:
                ax.bar(centers, h, width=(hi - lo) / bins, color="darkseagreen",
                       linewidth=0, log=True)
            ax.set_title(f"L{layer}", fontsize=8)
            ax.tick_params(labelsize=6)
        for j in range(L, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"Range stream — raw logit histogram per layer ({phase})", fontsize=12)
        fig.supxlabel("logit value")
        fig.supylabel("count (log)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_dir / f"stream_b_panels_{phase}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_stream_b_summary(d: dict, out_dir: Path) -> None:
    layers = d["layer_ids"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for phase, color in [("prefill", "navy"), ("decode", "darkorange")]:
        cnt = d[f"count_{phase}"].astype(np.float64)
        if cnt.sum() == 0:
            continue
        mean = d[f"sum_{phase}"] / np.where(cnt > 0, cnt, 1)
        var = np.maximum(d[f"sum_sq_{phase}"] / np.where(cnt > 0, cnt, 1) - mean ** 2, 0)
        std = np.sqrt(var)
        ax.plot(layers, mean, "-", color=color, label=f"{phase} mean")
        ax.fill_between(layers, mean - std, mean + std, alpha=0.2, color=color,
                        label=f"{phase} ±1σ")
        ax.plot(layers, d[f"min_{phase}"], ":", color=color, alpha=0.6,
                label=f"{phase} min")
        ax.plot(layers, d[f"max_{phase}"], ":", color=color, alpha=0.6,
                label=f"{phase} max")
    ax.set_xlabel("layer index")
    ax.set_ylabel("logit value")
    ax.set_title("Range stream summary — mean ± std and min/max per layer")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "stream_b_summary.png", dpi=140)
    plt.close(fig)


def plot_stream_b_saturation(d: dict, out_dir: Path) -> None:
    layers = d["layer_ids"]
    fig, ax = plt.subplots(figsize=(11, 4))
    for phase, color in [("prefill", "navy"), ("decode", "darkorange")]:
        cnt = d[f"count_{phase}"].astype(np.float64)
        below = d[f"below_{phase}"]
        above = d[f"above_{phase}"]
        denom = np.where(cnt > 0, cnt, 1)
        frac = (below + above) / denom
        ax.plot(layers, frac, "-o", color=color, label=phase, markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("layer index")
    ax.set_ylabel("fraction outside histogram range (log)")
    ax.set_title("Range stream saturation — fraction of values landing in below_/above_ overflow")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "stream_b_saturation.png", dpi=140)
    plt.close(fig)


def plot_stream_c_summary(d: dict, out_dir: Path) -> None:
    """Per-layer top-k winner range, with format-max horizontal lines."""
    layers = d["layer_ids"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for phase, color in [("prefill", "navy"), ("decode", "darkorange")]:
        cnt = d[f"topk_count_{phase}"].astype(np.float64)
        if cnt.sum() == 0:
            continue
        denom = np.where(cnt > 0, cnt, 1)
        mean = d[f"topk_sum_{phase}"] / denom
        var = np.maximum(d[f"topk_sum_sq_{phase}"] / denom - mean ** 2, 0)
        std = np.sqrt(var)
        ax.plot(layers, mean, "-", color=color, label=f"{phase} winner mean")
        ax.fill_between(layers, mean - std, mean + std, alpha=0.2, color=color,
                        label=f"{phase} ±1σ")
        ax.plot(layers, d[f"topk_min_{phase}"], ":", color=color, alpha=0.6,
                label=f"{phase} min")
        ax.plot(layers, d[f"topk_max_{phase}"], ":", color=color, alpha=0.6,
                label=f"{phase} max")
    for name, fmax, color in FORMAT_MAXES:
        if abs(fmax) < 1e6:
            ax.axhline(fmax, color=color, linestyle="--", linewidth=0.6, alpha=0.7,
                       label=f"{name} max ({fmax:g})")
            ax.axhline(-fmax, color=color, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("layer index")
    ax.set_ylabel("winner logit value")
    ax.set_title("Winners stream — top-k winner dynamic range per layer")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "stream_c_winner_range.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cascade-TopD stream — radix boundary-bucket cascade plots
# ---------------------------------------------------------------------------

# kNumFinalItems in vLLM's top-k kernel (csrc/sampler.cu:338). A row's boundary
# bucket must fit in this many items for one-pass radix to succeed without
# cascading.
K_NUM_FINAL_ITEMS = 2048


def _stream_d_present(d: dict) -> bool:
    return "radix_bits_sweep" in d and "radix_bkt_hist_prefill" in d


def _fits_fraction(hist_row: np.ndarray, bkt_bins: int,
                   threshold: int = K_NUM_FINAL_ITEMS) -> float:
    """Fraction of rows whose boundary-bucket size ≤ `threshold`.

    The histogram bins are int64[bkt_bins]; bin i is the count of rows whose
    boundary-bucket size == i, with bin (bkt_bins-1) saturating (all sizes
    >= bkt_bins-1 land there).  If threshold >= bkt_bins, we cannot distinguish
    "fits" vs "overflow" within the saturating bin; return NaN as a flag.
    """
    total = int(hist_row.sum())
    if total == 0:
        return float("nan")
    if threshold + 1 > bkt_bins:
        # Saturating bin holds both "fits" and "overflow" cases; can't separate.
        return float("nan")
    return float(hist_row[: threshold + 1].sum() / total)


def plot_stream_d_cascade_rate(d: dict, out_dir: Path) -> None:
    """One-pass cascade success rate vs d (bits per radix scan).

    For each d ∈ radix_bits_sweep, fraction of rows where the boundary
    bucket size fits in `kNumFinalItems`. Per-layer faint lines + bold
    mean/median/worst-layer summary. Two panels: prefill, decode.
    """
    if not _stream_d_present(d):
        return
    ds = np.array(d["radix_bits_sweep"], dtype=int)
    bkt_bins = int(d["radix_bkt_bins"])
    layers = d["layer_ids"]
    nd, L = len(ds), len(layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, phase in zip(axes, ("prefill", "decode")):
        hist = d.get(f"radix_bkt_hist_{phase}")    # [nd, L, Bk]
        if hist is None or hist.sum() == 0:
            ax.set_title(f"{phase} — no data")
            ax.axis("off")
            continue
        fits = np.full((nd, L), np.nan)
        for di in range(nd):
            for li in range(L):
                fits[di, li] = _fits_fraction(hist[di, li], bkt_bins)
        # Per-layer faint
        for li in range(L):
            ax.plot(ds, fits[:, li], color="steelblue", alpha=0.12, linewidth=0.6)
        # Aggregate lines
        with np.errstate(invalid="ignore"):
            mean_ = np.nanmean(fits, axis=1)
            med_ = np.nanmedian(fits, axis=1)
            worst_ = np.nanmin(fits, axis=1)
        ax.plot(ds, mean_, "o-", color="darkblue", linewidth=2,
                label="layer mean", markersize=6)
        ax.plot(ds, med_, "s-", color="darkgreen", linewidth=2,
                label="layer median", markersize=6)
        ax.plot(ds, worst_, "x-", color="crimson", linewidth=1.6,
                label="worst layer", markersize=9)
        ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("d (bits per radix scan)")
        ax.set_xticks(ds)
        ax.set_title(f"{phase}: P(boundary bucket ≤ {K_NUM_FINAL_ITEMS}) vs d")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel(f"fraction of rows where 1-pass radix succeeds (fits ≤ {K_NUM_FINAL_ITEMS})")
    fig.suptitle("Cascade-TopD stream — cascade dial: one-pass success rate by bits-per-scan",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "stream_d_cascade_rate.png", dpi=140)
    plt.close(fig)


def plot_stream_d_summary(d: dict, out_dir: Path) -> None:
    """Per-layer mean ± std and max of boundary-bucket size, for each d.

    Reveals the layer-level structure: which layers concentrate cleanly
    (small buckets) vs which need deeper cascade.
    """
    if not _stream_d_present(d):
        return
    ds = np.array(d["radix_bits_sweep"], dtype=int)
    layers = d["layer_ids"]
    nd = len(ds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, phase in zip(axes, ("prefill", "decode")):
        count = d.get(f"radix_bkt_count_{phase}")    # [nd, L]
        sum_ = d.get(f"radix_bkt_sum_{phase}")
        sum_sq = d.get(f"radix_bkt_sum_sq_{phase}")
        bkt_max = d.get(f"radix_bkt_max_{phase}")
        if count is None or count.sum() == 0:
            ax.set_title(f"{phase} — no data")
            ax.axis("off")
            continue
        for di, dval in enumerate(ds):
            denom = np.where(count[di] > 0, count[di], 1).astype(np.float64)
            mean = sum_[di].astype(np.float64) / denom
            var = np.maximum(sum_sq[di].astype(np.float64) / denom - mean ** 2, 0)
            std = np.sqrt(var)
            color = plt.cm.viridis(di / max(nd - 1, 1))
            ax.plot(layers, mean, "-", color=color, linewidth=1.6,
                    label=f"d={dval} mean ({2**int(dval)} buckets)")
            ax.fill_between(layers, np.maximum(mean - std, 1e-1), mean + std,
                            alpha=0.12, color=color)
            ax.plot(layers, np.maximum(bkt_max[di], 1), ":", color=color,
                    alpha=0.6, linewidth=1)
        ax.axhline(K_NUM_FINAL_ITEMS, color="red", linestyle="--", linewidth=1,
                   alpha=0.6, label=f"kNumFinalItems={K_NUM_FINAL_ITEMS}")
        ax.set_yscale("log")
        ax.set_xlabel("layer index")
        ax.set_title(f"{phase} — solid = mean, shaded = ±σ, dotted = per-layer max")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)
    axes[0].set_ylabel("boundary bucket size (log)")
    fig.suptitle("Cascade-TopD stream — boundary-bucket size by layer × d", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "stream_d_summary.png", dpi=140)
    plt.close(fig)


def plot_stream_d_panels(d: dict, out_dir: Path) -> None:
    """Heatmap per d: rows = layer, columns = bucket-size bin.

    Shows distribution shape per layer. Red line at kNumFinalItems marks
    the one-pass-vs-cascade boundary.
    """
    if not _stream_d_present(d):
        return
    ds = np.array(d["radix_bits_sweep"], dtype=int)
    layers = d["layer_ids"]
    bkt_bins = int(d["radix_bkt_bins"])
    nd = len(ds)

    for phase in ("prefill", "decode"):
        hist = d.get(f"radix_bkt_hist_{phase}")
        if hist is None or hist.sum() == 0:
            continue
        ncols = min(nd, 5)
        nrows = math.ceil(nd / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.2),
                                  squeeze=False)
        for di, dval in enumerate(ds):
            ax = axes[di // ncols][di % ncols]
            h = hist[di].astype(np.float64)            # [L, Bk]
            row_sums = h.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1)
            normed = h / row_sums
            # log1p to compress dynamic range; values in [0,1]
            display = np.log1p(normed * 1000)          # roughly highlights up to 0.1% mass
            im = ax.imshow(display, aspect="auto", cmap="viridis", origin="lower",
                           interpolation="nearest")
            # Draw the kNumFinalItems line in bin coordinates
            if K_NUM_FINAL_ITEMS < bkt_bins:
                ax.axvline(K_NUM_FINAL_ITEMS, color="red", linestyle="--",
                           linewidth=1.2, alpha=0.9)
            ax.set_xlabel("boundary bucket size")
            ax.set_ylabel("layer index" if di % ncols == 0 else "")
            ax.set_title(f"d={dval}  ({2**int(dval)} buckets)", fontsize=10)
            # Y-axis label as layer ids (downsampled if many layers)
            n_yticks = min(len(layers), 10)
            yt_idx = np.linspace(0, len(layers) - 1, n_yticks).astype(int)
            ax.set_yticks(yt_idx)
            ax.set_yticklabels(layers[yt_idx], fontsize=7)
        for di in range(nd, nrows * ncols):
            axes[di // ncols][di % ncols].set_visible(False)
        fig.suptitle(
            f"Cascade-TopD stream — per-row boundary-bucket size distribution ({phase})\n"
            f"red dashed = kNumFinalItems={K_NUM_FINAL_ITEMS} (one-pass threshold)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"stream_d_panels_{phase}.png", dpi=130,
                    bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# BitEntropy stream — per-bit ones-count / bit-entropy plots
# ---------------------------------------------------------------------------


def _stream_e_present(d: dict) -> bool:
    return "bit_ones_raw_prefill" in d and "bit_total_prefill" in d


def _bit_entropy(ones: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Per-bit entropy fraction.

    ones: int64[..., 32] count of 1s per bit position.
    total: int64[...] total values.
    Returns float[..., 32]: min(p, 1-p) where p = ones/total. 0 = constant
    bit (useless), 0.5 = perfectly balanced (maximally informative). Returns
    NaN where total == 0.
    """
    total = total.astype(np.float64)
    if ones.ndim == 1:
        if total == 0:
            return np.full(32, np.nan, dtype=np.float64)
        p = ones.astype(np.float64) / total
        return np.minimum(p, 1.0 - p)
    # ones is [..., 32], total is [...]
    safe_total = np.where(total > 0, total, 1)
    p = ones.astype(np.float64) / safe_total[..., None]
    out = np.minimum(p, 1.0 - p)
    out[total == 0] = np.nan
    return out


def _annotate_fp32_layout(ax) -> None:
    """Draw vertical separators between sign / exponent / mantissa regions on a
    32-bit x-axis. Bits 0..22 = mantissa, 23..30 = exponent, 31 = sign."""
    # Mantissa | exponent | sign
    ax.axvline(22.5, color="white", linestyle="-", linewidth=1.5, alpha=0.7)
    ax.axvline(30.5, color="white", linestyle="-", linewidth=1.5, alpha=0.7)
    # Outer text labels — drawn on the axis below the bits, centered per region
    ax.text(11.0, -0.6, "mantissa (0..22)", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=8, color="black")
    ax.text(26.5, -0.6, "exp (23..30)", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=8, color="black")
    ax.text(31.0, -0.6, "sign (31)", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=8, color="black")


def plot_stream_e_bit_entropy(d: dict, out_dir: Path) -> None:
    """Per-layer × per-bit entropy heatmap.

    Two heatmaps stacked: raw FP32 bits (left) and sortable-uint32 bits
    (right), for each phase. Bit 31 = sign; 23..30 = exponent; 0..22 = mantissa.
    Constant bits show as black/blue (entropy ≈ 0); informative bits show as
    yellow/white (entropy ≈ 0.5).
    """
    if not _stream_e_present(d):
        return
    layers = d["layer_ids"]
    for phase in ("prefill", "decode"):
        ones_raw = d.get(f"bit_ones_raw_{phase}")
        ones_sort = d.get(f"bit_ones_sortable_{phase}")
        total = d.get(f"bit_total_{phase}")
        if ones_raw is None or total is None or total.sum() == 0:
            continue
        H_raw = _bit_entropy(ones_raw, total)        # [L, 32]
        H_sort = _bit_entropy(ones_sort, total)
        fig, axes = plt.subplots(
            1, 2, figsize=(16, max(6, len(layers) * 0.16)),
            gridspec_kw={"width_ratios": [1, 1]}, sharey=True,
        )
        for ax, mat, title in [(axes[0], H_raw, "raw FP32 bits"),
                                 (axes[1], H_sort, "sortable-uint32 bits")]:
            im = ax.imshow(mat, aspect="auto", cmap="inferno",
                           origin="lower", vmin=0, vmax=0.5, interpolation="nearest")
            ax.set_xticks(range(32))
            ax.set_xticklabels(range(32), fontsize=6)
            ax.set_xlabel("bit position")
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers, fontsize=6)
            ax.set_title(f"{title}  ({phase})")
            _annotate_fp32_layout(ax)
        axes[0].set_ylabel("layer index")
        fig.suptitle(
            f"BitEntropy stream — per-bit entropy (min(p, 1-p)) per layer ({phase})\n"
            "low = constant bit (wasted); high (~0.5) = maximally informative",
            fontsize=11,
        )
        cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
        cbar.set_label("bit entropy fraction")
        fig.savefig(out_dir / f"stream_e_bit_entropy_{phase}.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)


def plot_stream_e_best_bits(d: dict, out_dir: Path) -> None:
    """For each d ∈ {4,6,8,11,16}, compare 'top-d bits' (kernel default) vs
    'best-d bits by entropy' for the sortable-uint32 representation.

    Y axis: cumulative entropy of the chosen d bits, per layer.
    Higher = more informative bit selection. The gap between the two curves
    quantifies the bit-allocation improvement available.
    """
    if not _stream_e_present(d):
        return
    layers = d["layer_ids"]
    D_VALUES = [4, 6, 8, 11, 16]

    for phase in ("prefill", "decode"):
        ones_sort = d.get(f"bit_ones_sortable_{phase}")
        total = d.get(f"bit_total_{phase}")
        if ones_sort is None or total is None or total.sum() == 0:
            continue
        H = _bit_entropy(ones_sort, total)            # [L, 32]
        L = len(layers)
        # The kernel's "top-d bits" of the sortable form are bits 31, 30, ...,
        # 32-d. So top-d are at indices [32-d, 32).
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Panel 1: cumulative entropy curves
        ax = axes[0]
        for d_val in D_VALUES:
            kernel_bits = np.arange(32 - d_val, 32)
            kernel_score = np.nansum(H[:, kernel_bits], axis=1)
            # Best-d: pick top-d bits per layer by entropy
            best_score = np.zeros(L)
            for li in range(L):
                row = H[li]
                if np.all(np.isnan(row)):
                    best_score[li] = np.nan
                else:
                    best_score[li] = np.sort(np.nan_to_num(row, nan=-1))[-d_val:].sum()
            ax.plot(layers, kernel_score, "--",
                    label=f"d={d_val} top-{d_val} bits (kernel)", alpha=0.7)
            ax.plot(layers, best_score, "-",
                    label=f"d={d_val} best-{d_val} bits", linewidth=1.5)
        ax.set_xlabel("layer index")
        ax.set_ylabel("Σ entropy of chosen d bits (max 0.5·d)")
        ax.set_title(f"Bit-window choice — top-d vs best-by-entropy  ({phase})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)
        # Panel 2: average improvement vs d
        ax = axes[1]
        improvement = []
        d_axis = list(range(1, 17))
        for d_val in d_axis:
            kernel_bits = np.arange(32 - d_val, 32)
            kernel_avg = np.nansum(H[:, kernel_bits], axis=1)
            best_avg = []
            for li in range(L):
                row = H[li]
                if np.all(np.isnan(row)):
                    best_avg.append(np.nan)
                else:
                    best_avg.append(
                        np.sort(np.nan_to_num(row, nan=-1))[-d_val:].sum()
                    )
            best_avg = np.array(best_avg)
            with np.errstate(invalid="ignore"):
                imp = best_avg - kernel_avg
            improvement.append(np.nanmean(imp))
        ax.plot(d_axis, improvement, "o-", color="darkblue")
        ax.set_xlabel("d (bits per scan)")
        ax.set_ylabel("Σ entropy gain by best-bits over top-bits  (layer-mean)")
        ax.set_title(f"Available gain from bit reallocation  ({phase})")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"stream_e_best_bits_{phase}.png", dpi=140)
        plt.close(fig)


def plot_stream_e_bit_profile(d: dict, out_dir: Path) -> None:
    """Per-bit ones fraction profile, averaged across layers.

    Quick at-a-glance: which bits are biased / constant / balanced overall.
    """
    if not _stream_e_present(d):
        return
    layers = d["layer_ids"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, kind in zip(axes, ("raw", "sortable")):
        for phase, color in [("prefill", "navy"), ("decode", "darkorange")]:
            ones = d.get(f"bit_ones_{kind}_{phase}")
            total = d.get(f"bit_total_{phase}")
            if ones is None or total is None or total.sum() == 0:
                continue
            with np.errstate(invalid="ignore"):
                p = ones.astype(np.float64) / np.where(
                    total > 0, total, 1
                )[..., None]
                p[np.broadcast_to(total[..., None] == 0, p.shape)] = np.nan
            # Plot per-layer faint, plus mean / min / max
            for li in range(len(layers)):
                ax.plot(range(32), p[li], color=color, alpha=0.10, linewidth=0.6)
            mean = np.nanmean(p, axis=0)
            ax.plot(range(32), mean, "-", color=color, linewidth=2,
                    label=f"{phase} mean")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(bit == 1)")
        ax.set_title(f"{kind} FP32 bits")
        _annotate_fp32_layout(ax)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("bit position (0..31)")
    fig.suptitle("BitEntropy stream — per-bit P(bit=1) profile  (raw vs sortable)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "stream_e_bit_profile.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cascade-BestD stream — radix bucket plots with custom bit windows
# ---------------------------------------------------------------------------


def _stream_best_present(d: dict) -> bool:
    return ("radix_best_bit_windows" in d
            and "radix_best_bkt_hist_decode" in d)


def _best_window_label(start: int, width: int) -> str:
    return f"bits {start}..{start + width - 1} ({1 << width}b)"


def plot_stream_best_vs_top_cascade(d: dict, out_dir: Path) -> None:
    """Direct best-d vs top-d comparison: P(boundary bucket ≤ kNumFinalItems)
    at each candidate bit window.

    Two panels (prefill, decode). For each panel:
      X axis = number of bits (= log2 buckets) used by the bucket window.
      Y axis = P(one-pass success).
      Two series: top-d (kernel default) and best-d (run04 validation).
    """
    if not _stream_best_present(d) or not _stream_d_present(d):
        return
    ds_top = np.array(d["radix_bits_sweep"], dtype=int)
    windows = d["radix_best_bit_windows"]   # [ng, 2]
    bkt_bins = int(d["radix_bkt_bins"])
    layers = d["layer_ids"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, phase in zip(axes, ("prefill", "decode")):
        hist_top = d.get(f"radix_bkt_hist_{phase}")
        hist_best = d.get(f"radix_best_bkt_hist_{phase}")
        if hist_top is None or hist_best is None:
            ax.set_title(f"{phase} — no data")
            ax.axis("off")
            continue

        # Top-d fits-fraction
        top_fits_mean = []
        top_fits_worst = []
        for di in range(hist_top.shape[0]):
            fits = []
            for li in range(hist_top.shape[1]):
                f = _fits_fraction(hist_top[di, li], bkt_bins)
                if not np.isnan(f):
                    fits.append(f)
            if fits:
                top_fits_mean.append(np.mean(fits))
                top_fits_worst.append(np.min(fits))
            else:
                top_fits_mean.append(np.nan)
                top_fits_worst.append(np.nan)
        # Best-d fits-fraction
        best_fits_mean = []
        best_fits_worst = []
        for gi in range(hist_best.shape[0]):
            fits = []
            for li in range(hist_best.shape[1]):
                f = _fits_fraction(hist_best[gi, li], bkt_bins)
                if not np.isnan(f):
                    fits.append(f)
            if fits:
                best_fits_mean.append(np.mean(fits))
                best_fits_worst.append(np.min(fits))
            else:
                best_fits_mean.append(np.nan)
                best_fits_worst.append(np.nan)

        # X axis: log2(buckets) = width
        top_widths = ds_top
        best_widths = windows[:, 1]

        ax.plot(top_widths, top_fits_mean, "o-", color="crimson",
                label="top-d  (mean across layers)", markersize=7, linewidth=1.8)
        ax.plot(top_widths, top_fits_worst, "x--", color="crimson",
                label="top-d  (worst layer)", markersize=8, alpha=0.7)
        ax.plot(best_widths, best_fits_mean, "s-", color="darkblue",
                label="best-d (mean across layers)", markersize=7, linewidth=1.8)
        ax.plot(best_widths, best_fits_worst, "+--", color="darkblue",
                label="best-d (worst layer)", markersize=10, alpha=0.7)
        ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.3)
        ax.axhline(0.99, color="green", linewidth=0.5, alpha=0.3, linestyle=":")
        ax.set_xlabel("bits per radix pass (= log₂ buckets)")
        ax.set_title(f"{phase}: best-d vs top-d  ({K_NUM_FINAL_ITEMS}-fit rate)")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        # Annotate the best-d window indices on each marker
        for gi, (start, width) in enumerate(windows.tolist()):
            if not np.isnan(best_fits_mean[gi]):
                ax.annotate(f"[{start}..{start+width-1}]",
                            (width, best_fits_mean[gi]),
                            textcoords="offset points",
                            xytext=(8, -2), fontsize=7,
                            color="darkblue")
    axes[0].set_ylabel(f"fraction of rows fitting in {K_NUM_FINAL_ITEMS}")
    fig.suptitle(
        "Cascade-BestD vs Cascade-TopD — direct comparison of bit-window choice",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "cascade_best_vs_top.png", dpi=140)
    plt.close(fig)


def plot_stream_best_panels(d: dict, out_dir: Path) -> None:
    """Per-layer boundary-bucket-size distribution for each best-d window."""
    if not _stream_best_present(d):
        return
    windows = d["radix_best_bit_windows"]
    layers = d["layer_ids"]
    bkt_bins = int(d["radix_bkt_bins"])
    ng = len(windows)
    for phase in ("prefill", "decode"):
        hist = d.get(f"radix_best_bkt_hist_{phase}")
        if hist is None or hist.sum() == 0:
            continue
        ncols = min(ng, 3)
        nrows = math.ceil(ng / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5),
                                  squeeze=False)
        for gi, (start, width) in enumerate(windows.tolist()):
            ax = axes[gi // ncols][gi % ncols]
            h = hist[gi].astype(np.float64)
            row_sums = h.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1)
            normed = h / row_sums
            display = np.log1p(normed * 1000)
            ax.imshow(display, aspect="auto", cmap="viridis", origin="lower",
                      interpolation="nearest")
            if K_NUM_FINAL_ITEMS < bkt_bins:
                ax.axvline(K_NUM_FINAL_ITEMS, color="red", linestyle="--",
                           linewidth=1.2, alpha=0.9)
            ax.set_xlabel("boundary bucket size")
            ax.set_ylabel("layer index" if gi % ncols == 0 else "")
            ax.set_title(f"best-{width}: {_best_window_label(start, width)}",
                         fontsize=10)
            n_yticks = min(len(layers), 10)
            yt_idx = np.linspace(0, len(layers) - 1, n_yticks).astype(int)
            ax.set_yticks(yt_idx)
            ax.set_yticklabels(layers[yt_idx], fontsize=7)
        for gi in range(ng, nrows * ncols):
            axes[gi // ncols][gi % ncols].set_visible(False)
        fig.suptitle(
            f"Cascade-BestD stream — boundary-bucket distribution ({phase})\n"
            f"red dashed = kNumFinalItems={K_NUM_FINAL_ITEMS} threshold",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"cascade_best_panels_{phase}.png", dpi=130,
                    bbox_inches="tight")
        plt.close(fig)


def write_text_summary(d: dict, out_dir: Path) -> None:
    """One-page text dump: counts, p1, safety verdict."""
    layers = d["layer_ids"]
    lo = float(d["gap_range_lo"]); hi = float(d["gap_range_hi"])
    lines = []
    lines.append(f"k_index_topk={int(d['k_index_topk'])}  "
                 f"gap_window={int(d['gap_window'])}  "
                 f"gap_eps_norm={float(d['gap_eps_norm']):.1e}")
    lines.append(f"layers: {len(layers)}  "
                 f"prefill rows total: {int(d['gap_count_prefill'].sum())}  "
                 f"decode rows total: {int(d['gap_count_decode'].sum())}")
    lines.append("")
    for phase in ("prefill", "decode"):
        if d[f"gap_count_{phase}"].sum() == 0:
            lines.append(f"--- {phase}: no data captured")
            continue
        p1 = np.array([_quantile_from_hist(d[f"gap_hist_{phase}"][i], lo, hi, 0.01)
                       for i in range(len(layers))])
        worst_layer = int(layers[int(np.nanargmin(p1))])
        p1_min = float(np.nanmin(p1))
        lines.append(f"--- {phase} ---")
        lines.append(f"  p1 of relative_gap across layers: "
                     f"min={p1_min:.3e} (layer {worst_layer})   "
                     f"median={np.nanmedian(p1):.3e}   "
                     f"max={np.nanmax(p1):.3e}")
        lines.append("  Format safety (p1_min / ε; >3 safe, 1-3 marginal, <1 unsafe):")
        for name, eps, _ in FORMATS:
            r = p1_min / eps
            verdict = "SAFE" if r > 3 else ("MARGINAL" if r > 1 else "UNSAFE")
            lines.append(f"    {name:12s} ε={eps:.4f}   p1_min/ε={r:7.2f}   {verdict}")
        # Winners stream — max magnitude
        max_win = float(max(np.abs(d[f"topk_min_{phase}"]).max(),
                             np.abs(d[f"topk_max_{phase}"]).max()))
        lines.append(f"  Winner |value| max across layers: {max_win:.2f}")
        for name, fmax, _ in FORMAT_MAXES:
            verdict = "SAFE" if max_win < 0.5 * fmax else (
                "MARGINAL" if max_win < fmax else "OVERFLOW")
            lines.append(f"    {name:12s} max={fmax:>8.0f}    {verdict}")
        lines.append("")

    # ----- Cascade-TopD stream: dial -----
    if _stream_d_present(d):
        ds = np.array(d["radix_bits_sweep"], dtype=int)
        bkt_bins = int(d["radix_bkt_bins"])
        lines.append("--- Cascade-TopD stream — cascade dial (P(boundary bucket ≤ kNumFinalItems=2048)) ---")
        for phase in ("prefill", "decode"):
            hist = d.get(f"radix_bkt_hist_{phase}")
            if hist is None or hist.sum() == 0:
                continue
            lines.append(f"  {phase}:")
            for di, dval in enumerate(ds):
                fits = []
                for li in range(hist.shape[1]):
                    f = _fits_fraction(hist[di, li], bkt_bins)
                    if not np.isnan(f):
                        fits.append(f)
                if not fits:
                    continue
                arr = np.array(fits)
                lines.append(
                    f"    d={int(dval):2d} ({2**int(dval):6d} buckets):  "
                    f"P(fits)={arr.mean():.3f} mean   "
                    f"{arr.min():.3f} worst-layer   "
                    f"{(arr >= 0.99).sum()}/{len(arr)} layers ≥99%")
        lines.append("")

    # ----- BitEntropy stream: bit-allocation -----
    if _stream_e_present(d):
        lines.append("--- BitEntropy stream — bit allocation (sortable-uint32 form) ---")
        for phase in ("prefill", "decode"):
            ones = d.get(f"bit_ones_sortable_{phase}")
            total = d.get(f"bit_total_{phase}")
            if ones is None or total is None or total.sum() == 0:
                continue
            H = _bit_entropy(ones, total)   # [L, 32]
            # Layers with valid data
            valid = ~np.all(np.isnan(H), axis=1)
            H_valid = H[valid]
            if H_valid.size == 0:
                continue
            # Per-layer count of "informative" bits (entropy > 0.05)
            inf_bits = (H_valid > 0.05).sum(axis=1)
            lines.append(f"  {phase}:  layer-mean informative bits "
                         f"(entropy > 0.05): {inf_bits.mean():.1f}/32  "
                         f"min={int(inf_bits.min())}  max={int(inf_bits.max())}")
            # Top-d vs best-d entropy gap, averaged across layers
            lines.append(f"    d   Σ(top-d)  Σ(best-d)   gap")
            for d_val in (4, 6, 8, 11, 16):
                kbits = np.arange(32 - d_val, 32)
                top_sc = np.nansum(H_valid[:, kbits], axis=1)
                best_sc = np.array([
                    np.sort(np.nan_to_num(H_valid[li], nan=-1))[-d_val:].sum()
                    for li in range(H_valid.shape[0])
                ])
                lines.append(
                    f"   {d_val:>2d}    {top_sc.mean():.3f}     "
                    f"{best_sc.mean():.3f}    +{(best_sc - top_sc).mean():.3f}")
        lines.append("")

    # ----- Cascade-BestD stream: best-d direct measurement -----
    if _stream_best_present(d):
        windows = d["radix_best_bit_windows"]
        bkt_bins = int(d["radix_bkt_bins"])
        lines.append("--- Cascade-BestD stream — boundary fits at custom bit windows ---")
        for phase in ("prefill", "decode"):
            hist = d.get(f"radix_best_bkt_hist_{phase}")
            if hist is None or hist.sum() == 0:
                continue
            lines.append(f"  {phase}:")
            for gi, (start, width) in enumerate(windows.tolist()):
                fits = []
                for li in range(hist.shape[1]):
                    f = _fits_fraction(hist[gi, li], bkt_bins)
                    if not np.isnan(f):
                        fits.append(f)
                if not fits:
                    continue
                arr = np.array(fits)
                lines.append(
                    f"    bits {start:>2d}..{start+width-1:<2d} "
                    f"({1<<width:>5d} buckets):  "
                    f"P(fits)={arr.mean():.3f} mean   "
                    f"{arr.min():.3f} worst-layer   "
                    f"{(arr >= 0.99).sum()}/{len(arr)} layers >=99%")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    (out_dir / "summary.txt").write_text(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="One or more npz files (globs OK).")
    p.add_argument("--out", required=True, help="Output directory for figures.")
    p.add_argument("--aggregate", action="store_true",
                   help="Sum-aggregate histograms across all inputs (e.g., TP ranks).")
    args = p.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        expanded = glob.glob(raw)
        if not expanded:
            paths.append(Path(raw))
        else:
            paths.extend(Path(e) for e in expanded)
    if not paths:
        raise SystemExit(f"no inputs matched: {args.inputs}")
    print(f"loading {len(paths)} files: {[p.name for p in paths]}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    captures = _load(paths)
    if args.aggregate or len(captures) > 1:
        d = _sum_aggregate(captures)
        suffix = "aggregated"
    else:
        d = captures[0]
        suffix = paths[0].stem
    print(f"layers: {d['layer_ids']}")

    write_text_summary(d, out_dir)
    plot_stream_a_panels(d, out_dir)
    plot_p1_by_layer(d, out_dir)
    plot_safety_heatmap(d, out_dir)
    plot_stream_b_panels(d, out_dir)
    plot_stream_b_summary(d, out_dir)
    plot_stream_b_saturation(d, out_dir)
    plot_stream_c_summary(d, out_dir)
    # Cascade-TopD stream — cascade dial (only fires if present in npz)
    plot_stream_d_cascade_rate(d, out_dir)
    plot_stream_d_summary(d, out_dir)
    plot_stream_d_panels(d, out_dir)
    # BitEntropy stream — bit-allocation (only fires if present in npz)
    plot_stream_e_bit_entropy(d, out_dir)
    plot_stream_e_best_bits(d, out_dir)
    plot_stream_e_bit_profile(d, out_dir)
    # Cascade-BestD stream — direct best-d-vs-top-d comparison (run04+)
    plot_stream_best_vs_top_cascade(d, out_dir)
    plot_stream_best_panels(d, out_dir)
    print(f"\nFigures written to: {out_dir}")


if __name__ == "__main__":
    main()
