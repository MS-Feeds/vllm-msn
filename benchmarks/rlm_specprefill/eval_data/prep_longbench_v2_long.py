#!/usr/bin/env python3
"""Prepares the "long" (>128k word) subset of LongBench v2 for the RLM +
SpecPrefill ablation's eval set.

This is `../../spec_prefill_llama/datasets/prep_longbench_v2.py` with the
length filter flipped from `"short"` to `"long"` -- that script's own
docstring explains why LongBench v2's `length` categorical field (not a
recomputed word count) is the authoritative filter, why `context`/`question`
are kept separate rather than folded into one formatted prompt string, and
why the real column schema was never verified against a live download in
the environment that script was written in (no network/HF access there) --
all of that carries over unchanged here; only the target bucket differs.

Per ../rlm_specprefill_ablation_plan.md's EVAL SET CONSTRAINTS: RLM is only
worth using well above the point where a base model would do fine on its
own, so this ablation restricts to contexts comfortably above ~131K tokens.
LongBench v2's own "long" bucket (>128k *words*) is the closest existing
categorical label to that -- but words != tokens, so this script's output is
an intermediate candidate pool, NOT the final filtered eval set. Run
filter_by_token_length.py on its output to apply the real >131K-*token*
threshold against Llama's actual tokenizer before trusting any sample here
satisfies the ablation's own constraint.

Usage:
    python3 prep_longbench_v2_long.py --max-keep -1   # keep all "long" rows
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schema import EvalSample, write_jsonl  # noqa: E402

HF_DATASET_NAME = "THUDM/LongBench-v2"
DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"
DEFAULT_OUTPUT = Path(__file__).parent / "longbench_v2_long_samples.jsonl"

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

# LongBench v2's own definition of "long" (paper: >128k words). Used only as
# a sanity cross-check below, not to re-derive the filter -- `length ==
# "long"` (the dataset's own label) is authoritative, matching the sibling
# "short" prep script's convention.
_LONG_WORD_THRESHOLD = 128_000


def _resolve_hf_token(explicit: str | None) -> str | None:
    return explicit or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def load_longbench_v2_rows(cache_dir: Path, hf_token: str | None) -> list[dict]:
    """Loads the full THUDM/LongBench-v2 dataset via the `datasets` library."""
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prep_longbench_v2_long] loading {HF_DATASET_NAME} (cache_dir={cache_dir})", flush=True)
    ds = load_dataset(HF_DATASET_NAME, split="train", cache_dir=str(cache_dir), token=hf_token)
    rows = list(ds)
    if rows and not all(col in rows[0] for col in _REQUIRED_COLUMNS):
        raise KeyError(
            f"Expected columns {_REQUIRED_COLUMNS} not all found in {HF_DATASET_NAME}. "
            f"Available columns: {sorted(rows[0].keys())}"
        )
    return rows


def build_long_samples(rows: list[dict], max_keep: int = -1, seed: int = 42) -> list[EvalSample]:
    """Filters to the "long" length bucket and formats each row as an
    EvalSample, with multiple-choice fields kept in `extra`."""
    samples: list[EvalSample] = []
    skipped_not_long = 0
    skipped_bad_row = 0
    word_count_mismatches = 0

    for row in rows:
        if row.get("length") != "long":
            skipped_not_long += 1
            continue

        question = (row.get("question") or "").strip()
        context = (row.get("context") or "").strip()
        choices = [(row.get(f"choice_{letter}") or "").strip() for letter in CHOICE_LETTERS]
        answer = (row.get("answer") or "").strip().upper()
        if not question or not context or not all(choices) or answer not in CHOICE_LETTERS:
            skipped_bad_row += 1
            continue

        # Sanity cross-check only -- length == "long" above is authoritative.
        word_count = len(context.split()) + len(question.split())
        if word_count < _LONG_WORD_THRESHOLD:
            word_count_mismatches += 1

        samples.append(
            EvalSample(
                id=str(row.get("_id")),
                source="longbench_v2_long",
                context=context,
                question=question,
                answer=answer,
                extra={
                    "domain": row.get("domain"),
                    "sub_domain": row.get("sub_domain"),
                    "difficulty": row.get("difficulty"),
                    "length": row.get("length"),
                    "choices": choices,
                    "word_count": word_count,
                },
            )
        )

    print(
        f"[prep_longbench_v2_long] loaded={len(rows)} kept_long={len(samples)} "
        f"skipped_not_long={skipped_not_long} skipped_bad_row={skipped_bad_row}"
    )
    if word_count_mismatches:
        print(
            f"[prep_longbench_v2_long] WARNING: {word_count_mismatches} row(s) labeled "
            f"length='long' have a computed word count < {_LONG_WORD_THRESHOLD} -- the "
            f"dataset's own 'long' label was still used as the filter, but this is worth "
            f"checking against the real schema once a live download is available."
        )

    if max_keep >= 0 and len(samples) > max_keep:
        import random

        rng = random.Random(seed)
        samples = rng.sample(samples, max_keep)
        samples.sort(key=lambda s: s.id)

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LongBench v2 'long' eval samples")
    parser.add_argument("--hf-token", default=None, help="Defaults to $HF_TOKEN")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--max-keep",
        type=int,
        default=-1,
        help="Cap on number of samples; -1 keeps all available 'long' rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    token = _resolve_hf_token(args.hf_token)
    rows = load_longbench_v2_rows(args.cache_dir, token)
    samples = build_long_samples(rows, max_keep=args.max_keep, seed=args.seed)

    write_jsonl(samples, args.output)
    print(f"[prep_longbench_v2_long] wrote {len(samples)} rows -> {args.output}")


if __name__ == "__main__":
    main()
