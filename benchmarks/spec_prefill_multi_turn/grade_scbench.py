#!/usr/bin/env python3
"""Scores a multi-turn predictions file against `datasets/prep_scbench.py`'s
samples. Unlike `../spec_prefill_llama/grade_longbench_v2.py` (single-turn,
exact-match multiple-choice only), SCBench needs PER-TASK metrics -- ported
directly from the official `microsoft/MInference/scbench/eval_utils.py`
(fetched and confirmed at write time, not guessed -- see module docstring
below for exactly which functions were ported and why) rather than
reimplementing scoring conventions blind, since exact-match semantics (e.g.
what counts as a substring match) matter for reproducibility against
published SCBench numbers.

Per-config metric assignment (the official repo has no single dispatch
table visible in `eval_utils.py` mapping config name -> metric -- this is a
documented judgment call for this pipeline's 3 MVP configs, not a verbatim
port of a table that doesn't exist upstream):

- `scbench_kv`      -> `in_match`   (substring containment -- exact
                        key-value retrieval, ported verbatim)
- `scbench_qa_eng`  -> `qa_f1_score` (token-overlap F1 after normalization
                        -- standard free-form QA metric, ported verbatim)
- `scbench_summary` -> `rouge_l_f1` (ROUGE-L F-score -- the official repo's
                        `rouge_score` calls the external `rouge` PyPI
                        package, which isn't a dependency anywhere else in
                        this codebase; reimplemented here as a small,
                        dependency-free LCS-based ROUGE-L F1, same formula,
                        to avoid adding a new package for one metric)

Predictions file format (produced by `predict_scbench.py`): one JSON object
per line, {"conversation_id", "turn_idx", "config", "pred"} -- one row PER
TURN, not per conversation (see predict_scbench.py's own docstring for why
this benchmark is driven per-turn).

Usage:
    python3 grade_scbench.py \\
        --samples datasets/scbench_samples.jsonl \\
        --predictions results/<exp_id>_predictions.jsonl \\
        --output results/scbench_result.json

    # Grade every completed run at once (globs results/*_predictions.jsonl,
    # writes results/<exp_id>_result.json per file plus a combined
    # results/all_scores.csv -- the accuracy-side counterpart to
    # predict_scbench.py's all_runs.csv on the timing side):
    python3 grade_scbench.py --batch
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Optional

DEFAULT_RESULTS_DIR = Path(os.environ.get("BENCH_RESULTS_DIR", "results"))
DEFAULT_SAMPLES = Path(__file__).parent / "datasets" / "scbench_samples.jsonl"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "scbench_result.json"


# ---------------------------------------------------------------------------
# Metrics -- ported from microsoft/MInference/scbench/eval_utils.py (in_match,
# normalize_answer, qa_f1_score's f1_score helper verbatim; rouge_l_f1 is a
# dependency-free reimplementation of the same ROUGE-L F-score formula their
# rouge_score() computes via the external `rouge` package -- see module
# docstring).
# ---------------------------------------------------------------------------


def in_match(prediction: str, ground_truth: str) -> float:
    """Ported verbatim (semantics, not literal Python) from eval_utils.py's
    `in_match`: 1.0 if the ground-truth string appears anywhere in the
    prediction, else 0.0."""
    return 1.0 if ground_truth in prediction else 0.0


def normalize_answer(s: str) -> str:
    """Ported verbatim from eval_utils.py's `normalize_answer`: lowercase,
    strip punctuation/articles, collapse whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _token_f1(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    """Ported verbatim from eval_utils.py's `f1_score` (as called by
    `qa_f1_score` with pre-tokenized lists, not raw strings -- `Counter`
    over a list of tokens counts token frequency, standard SQuAD-style F1)."""
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0 or not prediction_tokens or not ground_truth_tokens:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    """Ported from eval_utils.py's `qa_f1_score` (adapted to take a single
    (prediction, ground_truth) pair rather than that repo's `line` dict
    convention -- multiple reference answers per turn aren't part of this
    pipeline's `datasets/prep_scbench.py` schema, so the max-over-references
    loop in the original collapses to one iteration here)."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    return _token_f1(normalized_prediction.split(), normalized_ground_truth.split())


def rouge_l_f1(prediction: str, ground_truth: str) -> float:
    """Dependency-free ROUGE-L F-score (longest-common-subsequence based) --
    same statistic eval_utils.py's `rouge_score` computes via the external
    `rouge` package, reimplemented directly rather than adding that
    dependency (see module docstring). Tokenizes on whitespace after the
    same `normalize_answer` normalization used for `qa_f1_score`, for
    consistency within this file (the official implementation tokenizes
    differently internally, inside the `rouge` package -- this is a
    documented approximation, not a byte-exact port, flagged here rather
    than silently presented as identical)."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Standard O(n*m) LCS length via dynamic programming.
    n, m = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[n][m]

    if lcs_len == 0:
        return 0.0
    precision = lcs_len / n
    recall = lcs_len / m
    return (2 * precision * recall) / (precision + recall)


_METRIC_BY_CONFIG = {
    "scbench_kv": in_match,
    "scbench_qa_eng": qa_f1_score,
    "scbench_summary": rouge_l_f1,
}


def score_turn(config: str, prediction: str, answer: str) -> Optional[float]:
    """Returns None (not a score of 0) for a config with no assigned metric
    -- e.g. a config outside the MVP's 3-config default (see
    prep_scbench.py) that a caller ran anyway -- so it's visibly excluded
    from aggregates rather than silently scored 0."""
    metric_fn = _METRIC_BY_CONFIG.get(config)
    if metric_fn is None:
        return None
    return metric_fn(prediction, answer)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def grade(samples: list[dict], predictions: list[dict]) -> dict:
    """Joins predictions to samples by (conversation_id, turn_idx) and
    scores each turn with its config's metric. Breaks down by (config,
    turn_idx) -- the turn_idx axis is the multi-turn-specific signal
    EXPERIMENT_PLAN.md calls for (does accuracy degrade as the conversation
    gets longer?), which LongBench-v2's single-turn grading never needed.

    Same missing-vs-scored-wrong distinction as
    ../spec_prefill_llama/grade_longbench_v2.py: a turn with no prediction
    at all (e.g. skipped for exceeding a token budget) is `missing`,
    excluded from every score denominator; a turn that DID get a
    prediction is always scored (0.0 counts as a real, if bad, score, same
    as any other turn) -- there's no "unparseable" concept here, since
    these are continuous similarity metrics, not a discrete letter-match.
    """
    pred_by_key: dict[tuple[str, int], str] = {}
    for p in predictions:
        pred_by_key[(p["conversation_id"], p["turn_idx"])] = p["pred"]

    total_turns = 0
    matched = 0
    missing = 0
    no_metric = 0

    scores_by_config: dict[str, list[float]] = {}
    scores_by_turn_idx: dict[int, list[float]] = {}
    scores_by_config_and_turn: dict[tuple[str, int], list[float]] = {}
    overall: list[float] = []
    per_turn: list[dict] = []

    for sample in samples:
        conversation_id = sample["id"]
        config = sample["config"]
        for turn_idx, turn in enumerate(sample["turns"]):
            total_turns += 1
            key = (conversation_id, turn_idx)
            question = turn["input"]
            answer = turn["answer"]

            if key not in pred_by_key:
                missing += 1
                per_turn.append(
                    {
                        "conversation_id": conversation_id,
                        "turn_idx": turn_idx,
                        "config": config,
                        "input": question,
                        "answer": answer,
                        "pred": None,
                        "score": None,
                    }
                )
                continue

            pred = pred_by_key[key]
            score = score_turn(config, pred, answer)
            if score is None:
                no_metric += 1
                per_turn.append(
                    {
                        "conversation_id": conversation_id,
                        "turn_idx": turn_idx,
                        "config": config,
                        "input": question,
                        "answer": answer,
                        "pred": pred,
                        "score": None,
                    }
                )
                continue

            matched += 1
            overall.append(score)
            scores_by_config.setdefault(config, []).append(score)
            scores_by_turn_idx.setdefault(turn_idx, []).append(score)
            scores_by_config_and_turn.setdefault((config, turn_idx), []).append(score)
            per_turn.append(
                {
                    "conversation_id": conversation_id,
                    "turn_idx": turn_idx,
                    "config": config,
                    "input": question,
                    "answer": answer,
                    "pred": pred,
                    "score": score,
                }
            )

    def _avg(xs: list[float]) -> float:
        return 100.0 * sum(xs) / len(xs) if xs else 0.0

    return {
        "overall": _avg(overall),
        "by_config": {k: _avg(v) for k, v in sorted(scores_by_config.items())},
        "by_turn_idx": {k: _avg(v) for k, v in sorted(scores_by_turn_idx.items())},
        "by_config_and_turn_idx": {
            f"{cfg}/turn{idx}": _avg(v)
            for (cfg, idx), v in sorted(scores_by_config_and_turn.items())
        },
        "counts": {
            "total_turns": total_turns,
            "matched": matched,
            "missing": missing,
            "no_metric_assigned": no_metric,
        },
        "per_turn": per_turn,
    }


def write_per_turn_csv(result: dict, path: Path) -> None:
    import csv

    fieldnames = ["conversation_id", "turn_idx", "config", "input", "answer", "pred", "score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["per_turn"]:
            writer.writerow(row)


def render_summary_table(result: dict) -> str:
    lines = []
    counts = result["counts"]
    lines.append(
        f"**Overall score: {result['overall']:.2f}** "
        f"(over {counts['matched']} matched of {counts['total_turns']} total turns -- "
        f"{counts['missing']} missing/skipped, {counts['no_metric_assigned']} with no "
        f"metric assigned, both excluded from this %)"
    )
    lines.append("")
    lines.append("| Config | Score |")
    lines.append("|---|---:|")
    for key, score in result["by_config"].items():
        lines.append(f"| {key} | {score:.2f} |")
    lines.append("")
    lines.append("| Turn index | Score |")
    lines.append("|---|---:|")
    for key, score in result["by_turn_idx"].items():
        lines.append(f"| {key} | {score:.2f} |")
    return "\n".join(lines)


def run_batch(samples_path: Path, results_dir: Path) -> None:
    """Grades every `<exp_id>_predictions.jsonl` file in `results_dir` --
    the naming convention `predict_scbench.py` itself writes
    (`OUT_DIR / f"{exp_id}_predictions.jsonl"`), so `exp_id` is recovered
    by stripping that fixed suffix off each matched filename, not parsed
    out of file content.

    Writes the same per-experiment `<exp_id>_result.json` single-file mode
    already writes (so nothing downstream that reads those individually
    needs to change), PLUS one combined `all_scores.csv` -- the accuracy
    counterpart to `predict_scbench.py`'s own `all_runs.csv` (timing),
    letting the two be joined on `exp_id` for a full picture.

    **Per-config columns tolerate config-filtered runs.** A run made with
    predict_scbench.py's own `--scbench-config` (e.g. only scbench_kv
    predicted) grades correctly here with NO special-casing needed: `grade`
    already treats every OTHER config's turns as `missing` (excluded from
    every average, not scored 0 -- see that function's own docstring), so
    `by_config` for such a run simply won't have keys for the configs that
    were never predicted. The CSV's column set is the UNION across every
    experiment found, so a config-filtered exp_id just gets an empty cell
    for the configs it never touched, rather than a misleading 0.00 that
    would look like a real (bad) score."""
    samples = _read_jsonl(samples_path)
    pred_files = sorted(results_dir.glob("*_predictions.jsonl"))
    if not pred_files:
        print(f"[grade_scbench] no *_predictions.jsonl files found in {results_dir}")
        return

    summary_rows = []
    for pred_file in pred_files:
        exp_id = pred_file.name[: -len("_predictions.jsonl")]
        predictions = _read_jsonl(pred_file)
        result = grade(samples, predictions)
        counts = result["counts"]

        aggregate_result = {k: v for k, v in result.items() if k != "per_turn"}
        out_path = results_dir / f"{exp_id}_result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(aggregate_result, f, indent=2)

        row = {
            "exp_id": exp_id,
            "overall": round(result["overall"], 4),
            "total_turns": counts["total_turns"],
            "matched": counts["matched"],
            "missing": counts["missing"],
            "no_metric_assigned": counts["no_metric_assigned"],
        }
        row.update({k: round(v, 4) for k, v in result["by_config"].items()})
        summary_rows.append(row)
        print(f"[grade_scbench] {exp_id}: overall={result['overall']:.2f} "
              f"({counts['matched']}/{counts['total_turns']} turns matched) -> {out_path}")

    fixed_cols = ["exp_id", "overall", "total_turns", "matched", "missing", "no_metric_assigned"]
    config_cols = sorted({k for row in summary_rows for k in row if k not in fixed_cols})
    fieldnames = fixed_cols + config_cols

    summary_path = results_dir / "all_scores.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"\n[grade_scbench] wrote combined summary ({len(summary_rows)} "
          f"experiments) -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade SCBench multi-turn predictions")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--predictions", type=Path, default=None,
        help="Required unless --batch is given.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-turn-output", type=Path, default=None)
    parser.add_argument(
        "--batch", action="store_true",
        help="Grade every <exp_id>_predictions.jsonl file in --results-dir "
             "instead of a single --predictions file -- see module "
             "docstring for what this writes.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Directory to glob for *_predictions.jsonl in --batch mode "
             "(defaults to $BENCH_RESULTS_DIR if set, else 'results', "
             "matching predict_scbench.py's own default output location).",
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(args.samples, args.results_dir)
        return

    if args.predictions is None:
        parser.error("--predictions is required unless --batch is given")

    samples = _read_jsonl(args.samples)
    predictions = _read_jsonl(args.predictions)
    result = grade(samples, predictions)

    if args.per_turn_output is not None:
        args.per_turn_output.parent.mkdir(parents=True, exist_ok=True)
        write_per_turn_csv(result, args.per_turn_output)
        print(f"[grade_scbench] wrote per-turn comparison -> {args.per_turn_output}")

    aggregate_result = {k: v for k, v in result.items() if k != "per_turn"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(aggregate_result, f, indent=2)

    print(render_summary_table(result))
    print(f"\n[grade_scbench] wrote {args.output}")


if __name__ == "__main__":
    main()
