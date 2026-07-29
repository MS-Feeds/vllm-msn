#!/usr/bin/env python3
"""Prepares the "short" (<32k word) subset of LongBench v2 for SpecPrefill's
keep-rate sweep (see ../EXPERIMENT_PLAN.md's experiment matrix P001-P006).

Downloads `THUDM/LongBench-v2` via the `datasets` library (not a plain CSV
download like ../../evaluation_pipeline/datasets/prep_aime.py /
prep_gpqa_diamond.py use -- LongBench v2 ships as an HF `datasets`-library
dataset (parquet), not a single CSV file. This follows the same loading
convention the cloned reference repo's own v1 harness already uses for the
sibling v1 dataset -- see ../speculative_prefill/speculative_prefill/eval/
long_bench/pred_vllm.py's `load_dataset('THUDM/LongBench', ...)` call), filters
to the official "short" length bucket (LongBench v2's own categorical `length`
field, defined by the paper as <32k words -- see ../EXPERIMENT_PLAN.md's
"Benchmark" section), and writes datasets/longbench_v2_samples.jsonl as
{"id", "domain", "sub_domain", "difficulty", "length", "prompt", "context",
"question", "choices": [...], "answer": "A"} rows. "context"/"question" are
stored separately (not just folded into "prompt") so predict_longbench_v2.py
can prune only the document context and always force-keep the question/
choices/instruction intact -- see that script's own docstring for the real
gibberish-output bug this fixes.

This script writes one combined raw text block as "prompt" (context + question
+ lettered choices + an answer-format instruction), NOT chat-template-rendered
-- rendering the final prompt happens at eval time (a prediction-generation
script, not yet built -- see ../EXPERIMENT_PLAN.md's "Implementation status"
#3), matching the convention in
../../evaluation_pipeline/datasets/prep_aime.py and prep_gpqa_diamond.py.

NOTE: LongBench v2's real column schema has not been verified against a live
download in the environment this script was written in (no network/HF access
there) -- `_REQUIRED_COLUMNS` below is checked explicitly with a clear error
listing the actual columns found, rather than silently mis-mapping fields if
the real schema differs from what's expected here.

Usage:
    python3 prep_longbench_v2.py --max-keep -1   # keep all "short" rows
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

HF_DATASET_NAME = "THUDM/LongBench-v2"
DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"
DEFAULT_OUTPUT = Path(__file__).parent / "longbench_v2_samples.jsonl"

CHOICE_LETTERS = ["A", "B", "C", "D"]
_REQUIRED_COLUMNS = [
    "_id",
    "domain",
    "sub_domain",
    "difficulty",
    "length",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
    "answer",
    "context",
]

# LongBench v2's own definition of "short" (see ../EXPERIMENT_PLAN.md's
# "Benchmark" section: "short questions (<32k words)"). The `length` column is
# used as the authoritative filter (it's the label the dataset's own authors
# assigned); this threshold is only used for the sanity cross-check below, not
# to re-derive the filter from scratch.
_SHORT_WORD_THRESHOLD = 32_000

_PROMPT_TEMPLATE = """Please read the following long text and answer the question below.

{context}

Question: {question}
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Answer with ONLY a single letter."""


def _resolve_hf_token(explicit: str | None) -> str | None:
    return explicit or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN"
    )


def load_longbench_v2_rows(cache_dir: Path, hf_token: str | None) -> list[dict]:
    """Loads the full THUDM/LongBench-v2 dataset via the `datasets` library
    (its own cache handles re-downloads, unlike the requests-based
    download_and_cache_csv helper the CSV-based prep scripts use)."""
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prep_longbench_v2] loading {HF_DATASET_NAME} (cache_dir={cache_dir})", flush=True)
    ds = load_dataset(
        HF_DATASET_NAME,
        split="train",
        cache_dir=str(cache_dir),
        token=hf_token,
    )
    rows = list(ds)
    if rows and not all(col in rows[0] for col in _REQUIRED_COLUMNS):
        raise KeyError(
            f"Expected columns {_REQUIRED_COLUMNS} not all found in "
            f"{HF_DATASET_NAME}. Available columns: {sorted(rows[0].keys())}"
        )
    return rows


def build_longbench_v2_samples(
    rows: list[dict],
    max_keep: int = -1,
    seed: int = 42,
) -> list[dict]:
    """Filters to the "short" length bucket and formats each row into
    {"id", "domain", "sub_domain", "difficulty", "length", "prompt",
    "choices", "answer"}."""
    samples: list[dict] = []
    skipped_not_short = 0
    skipped_bad_row = 0
    word_count_mismatches = 0

    for row in rows:
        if row.get("length") != "short":
            skipped_not_short += 1
            continue

        question = (row.get("question") or "").strip()
        context = (row.get("context") or "").strip()
        choices = [
            (row.get(f"choice_{letter}") or "").strip() for letter in CHOICE_LETTERS
        ]
        answer = (row.get("answer") or "").strip().upper()
        if not question or not context or not all(choices) or answer not in CHOICE_LETTERS:
            skipped_bad_row += 1
            continue

        # Sanity cross-check only -- `length == "short"` above is the
        # authoritative filter (the dataset's own label), not this word count.
        # See module docstring: this can't be verified against a live
        # download in the environment this was written in.
        word_count = len(context.split()) + len(question.split())
        if word_count >= _SHORT_WORD_THRESHOLD:
            word_count_mismatches += 1

        prompt = _PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            choice_A=choices[0],
            choice_B=choices[1],
            choice_C=choices[2],
            choice_D=choices[3],
        )

        samples.append(
            {
                "id": row.get("_id"),
                "domain": row.get("domain"),
                "sub_domain": row.get("sub_domain"),
                "difficulty": row.get("difficulty"),
                "length": row.get("length"),
                "prompt": prompt,
                # context/question stored separately (not just folded into
                # "prompt") so predict_longbench_v2.py can prune ONLY the
                # document context and always force-keep the question/
                # choices/instruction intact -- confirmed on real hardware
                # (2026-07-24): pruning the whole combined "prompt" as one
                # blob risks SpecPrefill's chunk-based scorer dropping the
                # trailing "Answer with a single letter..." instruction
                # entirely at aggressive keep rates, since it competes for a
                # spot against thousands of document chunks with no special
                # protection. That produced pure gibberish (the model never
                # learns it's supposed to answer a multiple-choice question)
                # -- not a position-restoration bug, a scoring-scope bug.
                "context": context,
                "question": question,
                "choices": choices,
                "answer": answer,
            }
        )

    print(
        f"[prep_longbench_v2] loaded={len(rows)} kept_short={len(samples)} "
        f"skipped_not_short={skipped_not_short} skipped_bad_row={skipped_bad_row}"
    )
    if word_count_mismatches:
        print(
            f"[prep_longbench_v2] WARNING: {word_count_mismatches} row(s) labeled "
            f"length='short' have a computed word count >= {_SHORT_WORD_THRESHOLD} "
            f"-- the dataset's own 'short' label was still used as the filter, "
            f"but this is worth checking against the real schema (see module "
            f"docstring's NOTE)."
        )

    if max_keep >= 0 and len(samples) > max_keep:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_keep)
        samples.sort(key=lambda s: s["id"])

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LongBench v2 'short' eval samples")
    parser.add_argument("--hf-token", default=None, help="Defaults to $HF_TOKEN")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--max-keep",
        type=int,
        default=-1,
        help="Cap on number of samples; -1 keeps all available 'short' rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    token = _resolve_hf_token(args.hf_token)
    rows = load_longbench_v2_rows(args.cache_dir, token)
    samples = build_longbench_v2_samples(rows, max_keep=args.max_keep, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prep_longbench_v2] wrote {len(samples)} rows -> {args.output}")


if __name__ == "__main__":
    main()
