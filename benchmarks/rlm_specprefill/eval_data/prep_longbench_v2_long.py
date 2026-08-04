#!/usr/bin/env python3
"""Prepares a length-bucketed subset of LongBench v2 for the RLM +
SpecPrefill ablation's eval set. Default bucket is "long" (>128k word)
-- the file/function names below still say "long" for that reason (the
original, still-primary use case), but `--length` accepts any of
LongBench v2's own buckets (`short`/`medium`/`long`).

This is `../../spec_prefill_llama/datasets/prep_longbench_v2.py` with the
length filter made selectable instead of hardcoded to `"short"` -- that
script's own docstring explains why LongBench v2's `length` categorical
field (not a recomputed word count) is the authoritative filter, why
`context`/`question` are kept separate rather than folded into one
formatted prompt string, and why the real column schema was never verified
against a live download in the environment that script was written in (no
network/HF access there) -- all of that carries over unchanged here.

Per ../rlm_specprefill_ablation_plan.md's EVAL SET CONSTRAINTS: RLM is only
worth using well above the point where a base model would do fine on its
own, so the ablation's OWN reported numbers should restrict to contexts
comfortably above ~131K tokens -- `--length long` (the default) is the
closest existing categorical label to that, but words != tokens, so its
output is an intermediate candidate pool, NOT the final filtered eval set
without also running filter_by_token_length.py on it. `--length short`
(or `medium`) deliberately steps outside that constraint -- useful for
fast pipeline iteration/debugging (much smaller contexts, far fewer RLM
REPL turns needed per sample, so far less exposure to slow/hung samples),
but NOT a substitute for the long-bucket eval set when actually reporting
this ablation's results -- see the constraints doc for why a short context
confounds "SpecPrefill didn't help" with "RLM shouldn't have been used
here at all."

Usage:
    python3 prep_longbench_v2_long.py --max-keep -1                # long bucket (default)
    python3 prep_longbench_v2_long.py --length short --max-keep -1  # short bucket
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
VALID_LENGTHS = ("short", "medium", "long")


def default_output_for(length: str) -> Path:
    """Length-aware output path so `--length short` and the default
    `--length long` runs don't clobber each other's output file."""
    return Path(__file__).parent / f"longbench_v2_{length}_samples.jsonl"


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
# a sanity cross-check below when length="long" specifically, not to
# re-derive the filter -- `length == "long"` (the dataset's own label) is
# authoritative, matching the sibling "short" prep script's convention. No
# equivalent cross-check threshold is defined for "short"/"medium" -- their
# own categorical label is trusted as-is without a secondary word-count
# sanity check, since this script was never validated against a live
# download to derive one.
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


def build_long_samples(
    rows: list[dict], max_keep: int = -1, seed: int = 42, length: str = "long"
) -> list[EvalSample]:
    """Filters to the given length bucket (`short`/`medium`/`long`, LongBench
    v2's own categorical label) and formats each row as an EvalSample, with
    multiple-choice fields kept in `extra`. Name kept as `build_long_samples`
    for backward compatibility with existing callers/tests -- `length`
    controls which bucket is actually built, `long` remains the default."""
    if length not in VALID_LENGTHS:
        raise ValueError(f"length must be one of {VALID_LENGTHS}, got {length!r}")

    samples: list[EvalSample] = []
    skipped_wrong_length = 0
    skipped_bad_row = 0
    word_count_mismatches = 0

    for row in rows:
        if row.get("length") != length:
            skipped_wrong_length += 1
            continue

        question = (row.get("question") or "").strip()
        context = (row.get("context") or "").strip()
        choices = [(row.get(f"choice_{letter}") or "").strip() for letter in CHOICE_LETTERS]
        answer = (row.get("answer") or "").strip().upper()
        if not question or not context or not all(choices) or answer not in CHOICE_LETTERS:
            skipped_bad_row += 1
            continue

        word_count = len(context.split()) + len(question.split())
        # Sanity cross-check only, and only meaningful for "long" -- no
        # equivalent LongBench-v2-published threshold exists for
        # short/medium to cross-check against (see _LONG_WORD_THRESHOLD's
        # own comment), so this is skipped entirely for those buckets
        # rather than checking against a made-up number.
        if length == "long" and word_count < _LONG_WORD_THRESHOLD:
            word_count_mismatches += 1

        samples.append(
            EvalSample(
                id=str(row.get("_id")),
                source=f"longbench_v2_{length}",
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
        f"[prep_longbench_v2_long] loaded={len(rows)} kept_{length}={len(samples)} "
        f"skipped_wrong_length={skipped_wrong_length} skipped_bad_row={skipped_bad_row}"
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
    parser = argparse.ArgumentParser(description="Prepare a length-bucketed LongBench v2 eval sample set")
    parser.add_argument("--hf-token", default=None, help="Defaults to $HF_TOKEN")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--length",
        choices=VALID_LENGTHS,
        default="long",
        help=(
            "Which LongBench v2 length bucket to filter to. 'long' (default) "
            "is what the RLM+SpecPrefill ablation's own eval-set constraint "
            "calls for (see module docstring). 'short'/'medium' step outside "
            "that constraint -- useful for fast pipeline iteration/debugging "
            "(much smaller contexts, far less exposure to slow/hung RLM "
            "samples), not a substitute for 'long' when reporting real "
            "ablation results."
        ),
    )
    parser.add_argument(
        "--max-keep",
        type=int,
        default=-1,
        help="Cap on number of samples; -1 keeps all available rows in the selected bucket.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to longbench_v2_<length>_samples.jsonl (length-aware, so different buckets don't clobber each other).",
    )
    args = parser.parse_args()

    output = args.output if args.output is not None else default_output_for(args.length)

    token = _resolve_hf_token(args.hf_token)
    rows = load_longbench_v2_rows(args.cache_dir, token)
    samples = build_long_samples(rows, max_keep=args.max_keep, seed=args.seed, length=args.length)

    write_jsonl(samples, output)
    print(f"[prep_longbench_v2_long] wrote {len(samples)} rows -> {output}")


if __name__ == "__main__":
    main()
