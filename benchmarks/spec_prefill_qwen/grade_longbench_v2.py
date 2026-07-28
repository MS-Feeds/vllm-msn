#!/usr/bin/env python3
"""Scores a predictions file against `datasets/prep_longbench_v2.py`'s samples
(see EXPERIMENT_PLAN.md's "Experiment matrix": accuracy is the only metric
LongBench v2 scoring needs here, unlike LongBench v1's per-task F1/ROUGE/
classification metrics -- see the cloned reference repo's
speculative_prefill/speculative_prefill/eval/long_bench/eval.py for that older
harness, which this is deliberately NOT reused from, per EXPERIMENT_PLAN.md's
"Implementation status" #3).

Predictions file format (produced by a not-yet-built prediction-generation
script -- see EXPERIMENT_PLAN.md's "Implementation status" #3 and
REPRODUCE.md step 4): one JSON object per line,
{"id": <matches a sample's "id">, "pred": "<raw model output text>"}.

Answer extraction mirrors LongBench v2's own official eval convention: look
for an explicit "answer is X" statement first, then fall back to the first
standalone A-D letter in the text. A prediction that matches neither is
counted as incorrect (same as a wrong letter) but tracked separately as
"unparseable" so a systematic extraction failure is visible rather than
silently blended into the wrong-answer count.

Usage:
    python3 grade_longbench_v2.py \\
        --samples datasets/longbench_v2_samples.jsonl \\
        --predictions <path-to-predictions.jsonl> \\
        --output results/longbench_v2_result.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

DEFAULT_SAMPLES = Path(__file__).parent / "datasets" / "longbench_v2_samples.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "longbench_v2_result.json"

_ANSWER_IS_RE = re.compile(r"answer is\s*\(?([A-D])\)?", re.IGNORECASE)
_STANDALONE_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_answer_letter(raw_text: str) -> Optional[str]:
    """Extracts a single A-D answer letter from a raw model output string.
    Returns None if neither pattern matches (unparseable)."""
    match = _ANSWER_IS_RE.search(raw_text)
    if match:
        return match.group(1).upper()
    match = _STANDALONE_LETTER_RE.search(raw_text)
    if match:
        return match.group(1).upper()
    return None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _accuracy(results: list[bool]) -> float:
    return 100.0 * sum(results) / len(results) if results else 0.0


def grade(samples: list[dict], predictions: list[dict]) -> dict:
    """Joins predictions to samples by id and computes overall accuracy plus
    breakdowns by difficulty and domain."""
    pred_by_id = {p["id"]: p["pred"] for p in predictions}

    total = len(samples)
    matched = 0
    missing = 0
    unparseable = 0

    correct_by_difficulty: dict[str, list[bool]] = {}
    correct_by_domain: dict[str, list[bool]] = {}
    overall: list[bool] = []

    for sample in samples:
        sample_id = sample["id"]
        difficulty = sample.get("difficulty") or "unknown"
        domain = sample.get("domain") or "unknown"

        if sample_id not in pred_by_id:
            missing += 1
            is_correct = False
        else:
            matched += 1
            extracted = extract_answer_letter(pred_by_id[sample_id])
            if extracted is None:
                unparseable += 1
            is_correct = extracted == sample["answer"]

        overall.append(is_correct)
        correct_by_difficulty.setdefault(difficulty, []).append(is_correct)
        correct_by_domain.setdefault(domain, []).append(is_correct)

    return {
        "overall": _accuracy(overall),
        "by_difficulty": {k: _accuracy(v) for k, v in sorted(correct_by_difficulty.items())},
        "by_domain": {k: _accuracy(v) for k, v in sorted(correct_by_domain.items())},
        "counts": {
            "total": total,
            "matched": matched,
            "missing": missing,
            "unparseable": unparseable,
        },
    }


def render_summary_table(result: dict) -> str:
    """Markdown summary, same group -> stat -> table shape as
    ../gemma4_moe_benchmarks/analyze_results.py's render_table (adapted from
    throughput stats to accuracy; no shared module exists to import that
    code from -- see grep results across this repo for prior art)."""
    lines = []
    counts = result["counts"]
    lines.append(
        f"**Overall accuracy: {result['overall']:.2f}%** "
        f"({counts['matched']}/{counts['total']} matched, "
        f"{counts['missing']} missing, {counts['unparseable']} unparseable)"
    )
    lines.append("")
    lines.append("| Difficulty | Accuracy |")
    lines.append("|---|---:|")
    for key, acc in result["by_difficulty"].items():
        lines.append(f"| {key} | {acc:.2f}% |")
    lines.append("")
    lines.append("| Domain | Accuracy |")
    lines.append("|---|---:|")
    for key, acc in result["by_domain"].items():
        lines.append(f"| {key} | {acc:.2f}% |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade LongBench v2 predictions")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    samples = _read_jsonl(args.samples)
    predictions = _read_jsonl(args.predictions)
    result = grade(samples, predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(render_summary_table(result))
    print(f"\n[grade_longbench_v2] wrote {args.output}")


if __name__ == "__main__":
    main()
