#!/usr/bin/env python3
"""Explains a per-turn-index accuracy gap in a graded sweep -- specifically
the "SPARSE turn 0 scores far below turns 1-4, while M000 doesn't" pattern.

## Why this script exists, and what it is testing

`grade_scbench.py`'s `by_turn_idx` breakdown showed, for every SPARSE keep
rate, a large turn-0 deficit (e.g. k40: 47 / 69 / 69 / 72 / 70) that M000
does not share (70 / 74 / 61 / 59 / 51). Before reading that as a property
of sparse attention, two structural facts about the driver have to be ruled
in or out, because both are pure prompt-construction effects with nothing to
do with the attention gather:

1. **Turn 0 is the only apples-to-apples turn in the whole sweep.**
   `run_baseline` builds turn 0's prompt as
   `chat_before + context + query + chat_after` (via `state.begin_turn`'s
   candidate_pool + force_keep_query), and `run_sparse_attention` builds
   turn 0's `delta_ids` as `chat_before + context + query + chat_after` --
   the same tokens. From turn 1 onward they diverge structurally: M000
   re-renders the ENTIRE conversation flattened into one user message
   (`predict_scbench.py`'s module docstring #3, including its own noted
   "turn-5 confusion cost"), while the sparse path uses genuine
   `<|eot_id|>`-delimited chat turns via `turn_boundary_ids`. So any
   M000-vs-SPARSE comparison at turns 1-4 is confounded by rendering, and
   SPARSE "beating" dense attention there is not a result.

2. **`in_match` is `ground_truth in prediction`** (`grade_scbench.py`), so
   verbosity and repetition tails cost nothing -- only a truncated or
   garbled value scores 0. With `--max-tokens` defaulting to 64, a turn-0
   answer that opens with a preamble can still lose the tail of a long
   value, and turn 0 is exactly the turn with no prior assistant message to
   imitate for terseness.

This script measures both against the real predictions, per turn index:
length/cap-hit distribution (truncation), whether a PREFIX of the gold
appears when the full gold does not (truncated-or-garbled retrieval, as
opposed to retrieving the wrong thing entirely), and whether a prediction
contains some OTHER turn's gold (fallback onto nearby prior-turn content
rather than the far, sparsified context).

## What it cannot tell you

The confound in (1) is not measurable from predictions alone -- it needs a
control row. `KEEP_RATES` has no 1.0 entry, so there is currently no run of
`run_sparse_attention` at full attention. Adding one isolates rendering from
sparsity: at keep=1.0 every ledger position plus every wrapper span is
registered, so the selection covers every resident block and
`compute_sparse_gather_view_incremental` returns `None` (its documented
degenerate no-op) on every decode step -- i.e. dense attention with the
sparse path's own prompt rendering. Compare that row's `by_turn_idx` against
M000's to size the rendering effect, and against SPARSE-k80's to size the
sparsity effect.

Usage:
    python3 diagnose_turn_index_gap.py --results-dir results
    python3 diagnose_turn_index_gap.py --experiments M000 SPARSE-k40-g32
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_SAMPLES = Path(__file__).parent / "datasets" / "scbench_samples.jsonl"
# Fraction of the gold string that must appear for a prediction to count as
# "found the right region but didn't land it" rather than a clean miss.
_PREFIX_FRACTION = 0.5
_MIN_PREFIX_CHARS = 6


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_golds(samples_path: Path) -> tuple[dict[tuple[str, int], str], dict[str, str]]:
    """`(conversation_id, turn_idx) -> gold answer` (matching how
    `grade_scbench.py` joins predictions back to the dataset), plus
    `conversation_id -> context` for the gold-position analysis below."""
    golds: dict[tuple[str, int], str] = {}
    contexts: dict[str, str] = {}
    for row in _read_jsonl(samples_path):
        contexts[str(row["id"])] = row.get("context") or ""
        for turn_idx, turn in enumerate(row.get("turns") or []):
            golds[(str(row["id"]), turn_idx)] = str(turn["answer"])
    return golds, contexts


def _gold_prefix(gold: str) -> str:
    return gold[: max(_MIN_PREFIX_CHARS, int(len(gold) * _PREFIX_FRACTION))]


def analyze_gold_position(
    predictions: list[dict],
    golds: dict[tuple[str, int], str],
    contexts: dict[str, str],
) -> tuple[dict, dict]:
    """Where in the shared context each turn's gold answer actually sits,
    and whether that position predicts failure.

    **The question this settles**, once truncation is ruled out (flat, short
    prediction lengths) and the turn-0 deficit is known to be all
    clean-miss: is turn 0 harder because its answers sit somewhere harder to
    retrieve from, or because it's the only turn with no worked example of
    the retrieval task already in context?

        mean position flat across turns  -> position is ruled out. Turns 1-4
            are easier because each one has prior `Question N: <key>
            Answer N: <value>` exchanges demonstrating the task -- ordinary
            in-context learning, a property of the benchmark's multi-turn
            structure, not of this pipeline. There is nothing to fix.
        mean position lower at turn 0    -> turn 0's answers sit earlier in
            the context, and an attention-based scorer with recency bias
            would miss them by position alone.

    The second return value breaks clean-miss rate down by which quartile of
    the context the gold sits in, pooled across turns -- if position matters
    at all, it shows up here regardless of how it's distributed per turn.

    Positions are character offsets of the gold's FIRST occurrence in the
    raw context, normalized to [0,1]. Golds that don't appear verbatim
    (summarization-style configs) are skipped rather than counted at 0.
    """
    by_turn: dict[int, list[float]] = defaultdict(list)
    by_quartile: dict[int, dict] = defaultdict(lambda: {"n": 0, "miss": 0})

    for pred in predictions:
        conv_id = str(pred["conversation_id"])
        turn_idx = int(pred["turn_idx"])
        gold = golds.get((conv_id, turn_idx))
        context = contexts.get(conv_id)
        if not gold or not context:
            continue
        where = context.find(gold)
        if where < 0:
            continue
        rel = where / len(context)
        by_turn[turn_idx].append(rel)

        q = min(3, int(rel * 4))
        by_quartile[q]["n"] += 1
        if gold not in (pred["pred"] or ""):
            by_quartile[q]["miss"] += 1

    turn_rows = {
        t: {"n": len(v), "mean_rel_pos": statistics.mean(v)}
        for t, v in sorted(by_turn.items()) if v
    }
    quartile_rows = {
        q: {"n": b["n"], "miss_pct": 100.0 * b["miss"] / (b["n"] or 1)}
        for q, b in sorted(by_quartile.items())
    }
    return turn_rows, quartile_rows


def analyze(predictions: list[dict], golds: dict[tuple[str, int], str], max_tokens: int) -> dict:
    """Per-turn-index breakdown. `by_conv` lets the cross-turn check ask
    "does this prediction contain a DIFFERENT turn's gold?", which
    distinguishes 'fell back on nearby prior-turn content' from 'retrieved
    from the far context'."""
    by_conv: dict[str, dict[int, str]] = defaultdict(dict)
    for pred in predictions:
        by_conv[str(pred["conversation_id"])][int(pred["turn_idx"])] = pred["pred"]

    buckets: dict[int, dict] = defaultdict(lambda: {
        "n": 0, "hits": 0, "prefix_only": 0, "clean_miss": 0,
        "other_turn_gold": 0, "char_lens": [], "empty": 0,
    })

    for pred in predictions:
        conv_id = str(pred["conversation_id"])
        turn_idx = int(pred["turn_idx"])
        text = pred["pred"] or ""
        gold = golds.get((conv_id, turn_idx))
        if gold is None:
            continue  # same "missing" exclusion grade_scbench.py applies

        b = buckets[turn_idx]
        b["n"] += 1
        b["char_lens"].append(len(text))
        if not text.strip():
            b["empty"] += 1

        if gold in text:
            b["hits"] += 1
        elif _gold_prefix(gold) in text:
            b["prefix_only"] += 1
        else:
            b["clean_miss"] += 1

        others = [
            g for (c, t), g in golds.items()
            if c == conv_id and t != turn_idx and g and g in text
        ]
        if others:
            b["other_turn_gold"] += 1

    out = {}
    for turn_idx in sorted(buckets):
        b = buckets[turn_idx]
        n = b["n"] or 1
        lens = b["char_lens"] or [0]
        out[turn_idx] = {
            "n": b["n"],
            "score": 100.0 * b["hits"] / n,
            "prefix_only_pct": 100.0 * b["prefix_only"] / n,
            "clean_miss_pct": 100.0 * b["clean_miss"] / n,
            "other_turn_gold_pct": 100.0 * b["other_turn_gold"] / n,
            "empty_pct": 100.0 * b["empty"] / n,
            "mean_chars": statistics.mean(lens),
            "p95_chars": sorted(lens)[int(0.95 * (len(lens) - 1))],
        }
    return out


def _print_table(exp_id: str, rows: dict) -> None:
    print(f"\n=== {exp_id} ===")
    print(f"{'turn':>4} {'n':>5} {'score':>7} {'prefix-only':>12} "
          f"{'clean-miss':>11} {'other-gold':>11} {'mean len':>9} {'p95 len':>8}")
    for turn_idx, r in rows.items():
        print(f"{turn_idx:>4} {r['n']:>5} {r['score']:>7.1f} "
              f"{r['prefix_only_pct']:>11.1f}% {r['clean_miss_pct']:>10.1f}% "
              f"{r['other_turn_gold_pct']:>10.1f}% {r['mean_chars']:>9.0f} {r['p95_chars']:>8.0f}")
    if len(rows) > 1:
        turn0 = rows[min(rows)]
        rest = [r for t, r in rows.items() if t != min(rows)]
        rest_score = statistics.mean(r["score"] for r in rest)
        rest_prefix = statistics.mean(r["prefix_only_pct"] for r in rest)
        rest_len = statistics.mean(r["mean_chars"] for r in rest)
        print(f"\n  turn 0 vs rest: score {turn0['score']:.1f} vs {rest_score:.1f} "
              f"({turn0['score'] - rest_score:+.1f})")
        print(f"  prefix-only (truncated/garbled): {turn0['prefix_only_pct']:.1f}% vs "
              f"{rest_prefix:.1f}%  -- a turn-0 excess here means the value was "
              f"found but not landed (truncation at --max-tokens, or corruption)")
        print(f"  mean pred length: {turn0['mean_chars']:.0f} vs {rest_len:.0f} chars "
              f"-- a longer turn 0 means no prior assistant turn to imitate for terseness")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Experiment ids to analyze (default: every "
                             "*_predictions.jsonl in --results-dir).")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="The run's --max-tokens, for the cap-hit note.")
    args = parser.parse_args()

    if not args.samples.exists():
        parser.error(f"{args.samples} not found -- run datasets/prep_scbench.py first")
    golds, contexts = _load_golds(args.samples)
    print(f"[diagnose_turn_index_gap] loaded {len(golds)} gold answers from {args.samples}")

    if args.experiments:
        pred_files = [args.results_dir / f"{e}_predictions.jsonl" for e in args.experiments]
    else:
        pred_files = sorted(args.results_dir.glob("*_predictions.jsonl"))
    if not pred_files:
        parser.error(f"no *_predictions.jsonl found in {args.results_dir}")

    for pred_file in pred_files:
        if not pred_file.exists():
            print(f"[diagnose_turn_index_gap] SKIP {pred_file} (not found)")
            continue
        exp_id = pred_file.name[: -len("_predictions.jsonl")]
        predictions = _read_jsonl(pred_file)
        rows = analyze(predictions, golds, args.max_tokens)
        _print_table(exp_id, rows)

        turn_pos, quartile = analyze_gold_position(predictions, golds, contexts)
        if turn_pos:
            print(f"\n  where the gold sits in the context (0.0 = start, 1.0 = end):")
            for turn_idx, r in turn_pos.items():
                print(f"    turn {turn_idx}: mean {r['mean_rel_pos']:.3f}  (n={r['n']})")
            spread = (max(r["mean_rel_pos"] for r in turn_pos.values())
                      - min(r["mean_rel_pos"] for r in turn_pos.values()))
            print(f"    spread across turns: {spread:.3f} "
                  f"-- near 0 means position is NOT what makes turn 0 harder")
            print(f"  clean-miss by context quartile (pooled across turns):")
            for q, r in quartile.items():
                print(f"    Q{q + 1} [{q * 0.25:.2f}-{(q + 1) * 0.25:.2f}]: "
                      f"{r['miss_pct']:.1f}% miss  (n={r['n']})")

    print("\nReading the table:")
    print("  prefix-only high at turn 0  -> truncation/garbling, not a retrieval miss.")
    print("                                 Re-run turn 0 with a larger --max-tokens to confirm.")
    print("  clean-miss high at turn 0   -> genuine retrieval failure from the sparsified")
    print("                                 far context; the expected cost of sparsity, and")
    print("                                 turn 0 is the only turn with no nearer fallback.")
    print("  other-gold high at turns>0  -> later turns are answerable from nearby prior-turn")
    print("                                 content, which the gather barely touches.")
    print("\nNeither reading is settled without a dense control that shares the sparse path's")
    print("prompt rendering -- see this script's module docstring on adding a keep=1.0 row.")


if __name__ == "__main__":
    main()
