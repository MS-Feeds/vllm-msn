#!/usr/bin/env python3
"""Compares two or more predictions files on the EXACT SAME turns, with a
cluster bootstrap CI -- the tool for reading an ORACLE-k* run against the
SPARSE row it is a ceiling for.

## Why this exists rather than three `grade_scbench.py --predictions` calls

`grade_scbench.py` scores one predictions file against the full samples file
and reports `missing` for every turn that file doesn't cover. That's the right
behavior for grading a run, and the wrong one for COMPARING runs: a 20-
conversation `ORACLE-k20` graded that way produces a number over 100 turns,
and the published `SPARSE-k20-g32`/`M000` numbers are over 500 -- different
conversations, so any difference between them mixes the effect being measured
with which conversations each row happened to see. SCBench's per-conversation
scores vary enormously (an `in_match` conversation is 5 near-binary turns), so
that mix is not a rounding error.

This script instead intersects the (conversation_id, turn_idx) keys present in
EVERY file given and scores all of them on that common set only. Rows are
then directly subtractable, which is the entire point of a ceiling row.

## Why a cluster bootstrap, not a plain CI

Turns within a conversation share one context and are strongly correlated --
if the retrieval target survived selection, several turns tend to hit
together. Treating 100 turns as 100 independent samples would understate the
interval by roughly the square root of the per-conversation turn count.
Resampling CONVERSATIONS (with replacement, whole) respects that structure.
At 20 conversations the interval is wide; that is information, not a defect --
it is precisely what stops a 5-point gap being read as a finding.

Metric functions are imported from `grade_scbench.py`, never reimplemented,
so a number here is the same number that file would produce on the same turns.

Usage:
    python3 compare_ceiling.py \\
        results/M000_predictions.jsonl \\
        results/SPARSE-k20-g32_predictions.jsonl \\
        results/ORACLE-k20_predictions.jsonl

    # restrict to one SCBench config (recommended -- the metrics differ per
    # config, so a mixed "overall" number is not interpretable):
    python3 compare_ceiling.py --config scbench_kv results/*_predictions.jsonl
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from grade_scbench import DEFAULT_SAMPLES, _read_jsonl, score_turn


def load_scored(
    samples_by_key: dict, path: Path, config_filter: str | None
) -> dict[tuple[str, int], float]:
    """One predictions file -> {(conversation_id, turn_idx): score}.

    Turns whose key has no sample (or no metric for its config) are dropped
    rather than scored as 0 -- an unscorable turn is missing information, not
    a failed answer, and averaging it in as a zero would penalize whichever
    row happened to cover more of them.
    """
    scored: dict[tuple[str, int], float] = {}
    for row in _read_jsonl(path):
        key = (row["conversation_id"], row["turn_idx"])
        sample = samples_by_key.get(key)
        if sample is None:
            continue
        config, answer = sample
        if config_filter is not None and config != config_filter:
            continue
        score = score_turn(config, row["pred"], answer)
        if score is None:
            continue
        scored[key] = score
    return scored


def cluster_bootstrap_ci(
    per_conversation: dict[str, list[float]], iterations: int, seed: int
) -> tuple[float, float]:
    """95% CI for the mean score, resampling whole CONVERSATIONS with
    replacement. Returns (lo, hi) on the same 0-100 scale as the point
    estimate."""
    rng = random.Random(seed)
    conversation_ids = list(per_conversation)
    means = []
    for _ in range(iterations):
        picked = [rng.choice(conversation_ids) for _ in conversation_ids]
        turns = [s for cid in picked for s in per_conversation[cid]]
        if turns:
            means.append(100.0 * sum(turns) / len(turns))
    means.sort()
    if not means:
        return (float("nan"), float("nan"))
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return (lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path, nargs="+",
                        help="Two or more <exp_id>_predictions.jsonl files. The "
                             "FIRST is the reference every other row is "
                             "differenced against (put M000 first).")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--config", default=None,
                        help="Restrict to one SCBench config (e.g. scbench_kv). "
                             "Recommended: the three configs use different "
                             "metrics, so a mixed average means nothing.")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Cluster-bootstrap iterations (0 to skip).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--by-conversation", action="store_true",
                        help="Per-conversation score for every row, sorted by "
                             "context length -- for checking whether a subset "
                             "is unrepresentative along the length axis "
                             "(long-context conversations are where the "
                             "baseline itself starts failing, leaving nothing "
                             "for sparsification to lose).")
    parser.add_argument("--per-turn", action="store_true",
                        help="Also break every row down by turn_idx -- the "
                             "multi-turn-specific signal (does degradation "
                             "compound as the conversation grows?).")
    args = parser.parse_args()

    if len(args.predictions) < 2:
        parser.error("give at least two predictions files to compare")

    samples_by_key = {}
    context_chars: dict[str, int] = {}
    for sample in _read_jsonl(args.samples):
        context_chars[sample["id"]] = len(sample.get("context", ""))
        for turn_idx, turn in enumerate(sample["turns"]):
            samples_by_key[(sample["id"], turn_idx)] = (sample["config"], turn["answer"])

    scored_by_file = {}
    for path in args.predictions:
        if not path.exists():
            parser.error(f"no such predictions file: {path}")
        scored_by_file[path] = load_scored(samples_by_key, path, args.config)
        if not scored_by_file[path]:
            parser.error(
                f"{path.name} has no scorable turns"
                + (f" for config {args.config!r}" if args.config else "")
            )

    # The comparison set: turns EVERY file covers. Anything else would let a
    # row be judged on turns another row never attempted.
    common = set.intersection(*(set(v) for v in scored_by_file.values()))
    if not common:
        parser.error("these files share no scorable turns -- nothing to compare")

    per_file_coverage = {
        p.name: len(v) for p, v in scored_by_file.items()
    }
    conversations = {cid for cid, _ in common}
    print(f"comparison set: {len(common)} turns across {len(conversations)} "
          f"conversations"
          + (f", config={args.config}" if args.config else "")
          )
    dropped = {name: n - len(common) for name, n in per_file_coverage.items()}
    if any(dropped.values()):
        print("  turns each file had that others did not (excluded): "
              + ", ".join(f"{k} +{v}" for k, v in dropped.items() if v))
    print()

    reference = args.predictions[0]
    rows = []
    for path in args.predictions:
        scored = scored_by_file[path]
        by_conversation: dict[str, list[float]] = {}
        for key, score in scored.items():
            if key in common:
                by_conversation.setdefault(key[0], []).append(score)
        turns = [s for k, s in scored.items() if k in common]
        mean = 100.0 * sum(turns) / len(turns)
        ci = (
            cluster_bootstrap_ci(by_conversation, args.bootstrap, args.seed)
            if args.bootstrap else (float("nan"), float("nan"))
        )
        rows.append((path, mean, ci, by_conversation))

    ref_mean = rows[0][1]
    name_width = max(len(p.stem.replace("_predictions", "")) for p, *_ in rows)
    print(f"{'row'.ljust(name_width)}  {'score':>7}  {'95% CI':>16}  {'vs. ' + reference.stem.replace('_predictions', ''):>14}")
    print("-" * (name_width + 45))
    for path, mean, (lo, hi), _ in rows:
        label = path.stem.replace("_predictions", "").ljust(name_width)
        delta = "" if path == reference else f"{mean - ref_mean:+.1f}"
        ci_str = f"[{lo:.1f}, {hi:.1f}]" if lo == lo else "n/a"
        print(f"{label}  {mean:7.1f}  {ci_str:>16}  {delta:>14}")

    if args.per_turn:
        print("\nby turn_idx:")
        turn_indices = sorted({t for _, t in common})
        header = "row".ljust(name_width) + "".join(f"  turn{t:>2}" for t in turn_indices)
        print(header)
        print("-" * len(header))
        for path, _, _, _ in rows:
            scored = scored_by_file[path]
            cells = []
            for t in turn_indices:
                vals = [s for (c, ti), s in scored.items() if ti == t and (c, ti) in common]
                cells.append(f"  {100.0 * sum(vals) / len(vals):6.1f}" if vals else "     n/a")
            print(path.stem.replace("_predictions", "").ljust(name_width) + "".join(cells))

    # Is the comparison set representative of what each row covered?
    #
    # This is not a formality. A row restricted to a subset can score wildly
    # differently from its own published full-run number, and when it does,
    # every difference measured on that subset is being measured somewhere
    # the reference row does not behave the way the headline table says it
    # does. The failure mode it catches: a comparison set where the baseline
    # has little headroom left compresses every row into each other, and the
    # resulting "small gaps" say nothing about the thing under test.
    interesting = [
        (path, scored) for path, scored in scored_by_file.items()
        if len(scored) > len(common)
    ]
    if interesting:
        print("\nis this comparison set representative?")
        print(f"  {'row'.ljust(name_width)}  {'in set':>8}  {'outside':>8}  {'delta':>8}")
        for path, scored in interesting:
            inside = [sc for k, sc in scored.items() if k in common]
            outside = [sc for k, sc in scored.items() if k not in common]
            if not outside:
                continue
            mean_in = 100.0 * sum(inside) / len(inside)
            mean_out = 100.0 * sum(outside) / len(outside)
            label = path.stem.replace("_predictions", "").ljust(name_width)
            print(f"  {label}  {mean_in:8.1f}  {mean_out:8.1f}  {mean_in - mean_out:+8.1f}")
        print(
            "  A large negative delta means the shared conversations are the "
            "HARDER ones.\n  Gaps measured there are gaps measured where the "
            "baseline itself is weak."
        )

    if args.by_conversation:
        print("\nper conversation (sorted by context length):")
        cids = sorted(conversations, key=lambda c: context_chars.get(c, 0), reverse=True)
        header = "  " + "conversation".ljust(22) + f"{'ctx chars':>11}" + "".join(
            f"  {p.stem.replace('_predictions', '')[:12]:>12}" for p, *_ in rows
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cid in cids:
            cells = []
            for path, *_ in rows:
                vals = [
                    sc for k, sc in scored_by_file[path].items()
                    if k[0] == cid and k in common
                ]
                cells.append(f"  {100.0 * sum(vals) / len(vals):12.1f}" if vals else f"  {'n/a':>12}")
            print(f"  {cid[:22].ljust(22)}{context_chars.get(cid, 0):>11}" + "".join(cells))

    if len(rows) >= 3:
        print(
            "\nreading the two gaps (see ACCURACY_IMPROVEMENTS.md's Step 0): "
            "oracle-minus-sparse is the estimator's cost, "
            "M000-minus-oracle is the mechanism's. Both are only readable if "
            "the reference row has headroom on this set -- check the "
            "representativeness table above first."
        )
    if args.bootstrap:
        print(
            "\nCIs are a cluster bootstrap over conversations, not turns. "
            "Two rows whose intervals overlap heavily are not distinguishable "
            "at this sample size -- add conversations before believing the "
            "ordering."
        )


if __name__ == "__main__":
    main()
