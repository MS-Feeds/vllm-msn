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


def paired_delta_ci(
    row: dict[str, list[float]],
    ref: dict[str, list[float]],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """95% CI for (row - reference), resampling whole CONVERSATIONS and
    recomputing BOTH means on the same resample.

    This, not the overlap of two marginal CIs, is the statistic that says
    whether a gap is real. Every row here answered the SAME conversations,
    so the comparison is paired and the conversation-difficulty variance
    that dominates each row's own interval cancels out of the difference.
    Judging a paired difference by whether two marginal intervals overlap
    is a well-known way to call a real effect insignificant -- and it
    would have done exactly that here: at 100 conversations the first full
    ORACLE-k20 result had M000 [76.4, 86.4] and ORACLE-k20 [68.2, 78.6],
    which overlap, while the paired difference is a solid 8 points.
    """
    rng = random.Random(seed)
    conversation_ids = [c for c in row if c in ref]
    deltas = []
    for _ in range(iterations):
        picked = [rng.choice(conversation_ids) for _ in conversation_ids]
        row_turns = [s for cid in picked for s in row[cid]]
        ref_turns = [s for cid in picked for s in ref[cid]]
        if row_turns and ref_turns:
            deltas.append(
                100.0 * sum(row_turns) / len(row_turns)
                - 100.0 * sum(ref_turns) / len(ref_turns)
            )
    deltas.sort()
    if not deltas:
        return (float("nan"), float("nan"))
    return (
        deltas[int(0.025 * (len(deltas) - 1))],
        deltas[int(0.975 * (len(deltas) - 1))],
    )


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
    ref_by_conversation = rows[0][3]
    # Decimal places, chosen from the metric's own scale rather than fixed.
    # SCBench uses metrics on two different scales: `scbench_kv`/`qa_eng`
    # score 0-100, where one decimal is right, but `scbench_summary` is
    # ROUGE-L F1 on 0-1, where one decimal rounds 0.23 and 0.26 to "0.2" and
    # "0.3" and prints every delta as "+0.0" -- the table becomes unreadable
    # exactly when the differences are small enough to need reading.
    _scale = max((abs(m) for _, m, _, _ in rows), default=0.0)
    prec = 3 if _scale < 10.0 else 1
    name_width = max(len(p.stem.replace("_predictions", "")) for p, *_ in rows)
    ref_label = reference.stem.replace("_predictions", "")
    print(f"{'row'.ljust(name_width)}  {'score':>7}  {'95% CI':>16}  "
          f"{'vs. ' + ref_label:>10}  {'paired 95% CI of the delta':>28}")
    print("-" * (name_width + 70))
    for path, mean, (lo, hi), by_conversation in rows:
        label = path.stem.replace("_predictions", "").ljust(name_width)
        ci_str = f"[{lo:.{prec}f}, {hi:.{prec}f}]" if lo == lo else "n/a"
        if path == reference:
            delta, delta_ci = "", ""
        else:
            delta = f"{mean - ref_mean:+.{prec}f}"
            if args.bootstrap:
                dlo, dhi = paired_delta_ci(
                    by_conversation, ref_by_conversation, args.bootstrap, args.seed
                )
                # A delta whose paired interval excludes 0 is a real
                # difference, regardless of whether the two marginal CIs
                # to its left happen to overlap -- see paired_delta_ci.
                excludes_zero = "" if dlo <= 0.0 <= dhi else "  *"
                delta_ci = f"[{dlo:+.{prec}f}, {dhi:+.{prec}f}]{excludes_zero}"
            else:
                delta_ci = "n/a"
        print(f"{label}  {mean:7.{prec}f}  {ci_str:>16}  {delta:>10}  "
              f"{delta_ci:>28}")
    if args.bootstrap and len(rows) > 1:
        print("  * paired interval excludes 0 -- a real difference. Judge gaps "
              "by THIS column,\n    not by whether the marginal CIs overlap: "
              "every row answered the same\n    conversations, so the "
              "difficulty variance that dominates each marginal\n    interval "
              "cancels out of the paired difference.")

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

    # Agreement with the reference, turn by turn -- NOT just the means.
    #
    # Two rows can have nearly equal means and still disagree about almost
    # every individual turn, if one of them wins some and loses others and
    # the errors cancel. On a near-binary metric like `in_match` that is not
    # a corner case, it is the expected shape of a selection method that
    # scrambles WHICH turns succeed rather than uniformly losing some. The
    # means alone would report that as "no difference", which is the
    # opposite of what it means: a row that tracks the baseline turn for
    # turn is behaving like the baseline, and a row with the same mean and
    # no agreement is not.
    #
    # Real example this exists for: a 20-conversation ORACLE-k20 vs
    # SPARSE-k20-g32 comparison whose means were 54.0 and 52.0 -- 2 points
    # apart, readable as "the estimator buys nothing" -- while ORACLE agreed
    # with M000 on 14/20 conversations (mean |diff| 11.0) and SPARSE on 5/20
    # (mean |diff| 33.0). The estimator mattered enormously; its errors
    # simply happened to cancel in the mean at that keep rate.
    if len(rows) >= 2:
        ref_scored = scored_by_file[reference]
        print(f"\nagreement with {reference.stem.replace('_predictions', '')}, "
              f"turn by turn (not just in the mean):")
        print(f"  {'row'.ljust(name_width)}  {'mean |diff|':>12}  {'turns equal':>12}")
        for path, *_ in rows:
            if path == reference:
                continue
            scored = scored_by_file[path]
            diffs = [abs(scored[k] - ref_scored[k]) for k in common]
            equal = sum(1 for k in common if scored[k] == ref_scored[k])
            # Same units and precision as the score column above. It used
            # to be scaled by 100, which on a 0-1 metric printed 0.002 as
            # "0.2" -- indistinguishable from a score-column 0.2 meaning
            # something a hundred times larger.
            print(f"  {path.stem.replace('_predictions', '').ljust(name_width)}  "
                  f"{sum(diffs) / len(diffs):12.{prec}f}  "
                  f"{f'{equal}/{len(common)}':>12}")
        print(
            "  Rows with similar means but LOW agreement are not equivalent -- "
            "they are\n  disagreeing in both directions and cancelling. Read "
            "this before the means."
        )

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
