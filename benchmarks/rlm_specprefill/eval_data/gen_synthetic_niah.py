#!/usr/bin/env python3
"""Scaled-up, multi-needle needle-in-a-haystack (NIAH) generator.

Generalizes ../../rlm/examples/quickstart_anthropic.py's single-needle
haystack (one hidden `SECRET_NUMBER=<digits>` line among ~50K lines of
random filler) into a parametric generator: configurable context size,
configurable needle count, and known ground-truth needle values/positions
recorded alongside each sample.

Why this exists alongside the LongBench v2 "long" bucket
(prep_longbench_v2_long.py): LongBench v2's multiple-choice answers don't
give a clean signal for the *evidence-recall* metric --
../rlm_specprefill_ablation_plan.md's METRICS section explicitly flags
evidence recall as needing "a well-defined ground truth ... ill-defined for
open-ended QA." A NIAH sample's ground truth is exact substrings at known
positions, so evidence recall reduces to "does the extracted evidence
contain each needle's exact value" -- no ambiguity. This is also the
day-1 candidate pool for calibration/sweep_n_min.py, since it lets you
generate candidate texts at exact target sizes rather than waiting on real
RLM output to happen to cluster usefully.

IMPORTANT: `--context-tokens` here is a GENERATION-TIME size target using a
rough chars-per-token estimate (matches
../../rlm/rlm/utils/token_utils.py's CHARS_PER_TOKEN_ESTIMATE=4 convention,
reused here ONLY for sizing filler text, not as a stand-in for the real
eval-set threshold check). It is NOT the authoritative token count -- run
filter_by_token_length.py on this script's output (against Llama's real
tokenizer) the same as for the LongBench-v2-derived samples, to get an
exact `context_tokens` value and confirm samples actually clear the >131K
threshold before trusting them.

Usage:
    python3 gen_synthetic_niah.py --context-tokens 150000,300000,500000 --n-needles 3
"""

from __future__ import annotations

import argparse
import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schema import EvalSample, write_jsonl  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).parent / "synthetic_niah_samples.jsonl"

# Matches rlm/utils/token_utils.py's own fallback estimate -- see module
# docstring above for why this is sizing-only, not the real threshold check.
CHARS_PER_TOKEN_ESTIMATE = 4

_LINE_CHARS = 120  # matches quickstart_anthropic.py's filler-line width


def _gen_filler_line(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase + " ", k=_LINE_CHARS))


def _gen_needle_value(rng: random.Random) -> str:
    return str(rng.randint(100_000_000, 999_999_999))


def generate_sample(
    rng: random.Random,
    context_tokens_target: int,
    n_needles: int,
    sample_idx: int,
) -> EvalSample:
    target_chars = context_tokens_target * CHARS_PER_TOKEN_ESTIMATE
    n_lines = max(n_needles + 1, target_chars // (_LINE_CHARS + 1))  # +1 for the newline joiner

    lines = [_gen_filler_line(rng) for _ in range(n_lines)]

    # Spread needles evenly across the document (not clustered), so recall
    # isn't trivially easy from a single lucky slice, and record ground
    # truth (label, value, approximate line position) for scoring.
    needles: list[dict] = []
    insert_positions = sorted(rng.sample(range(n_lines), n_needles))
    for offset, line_idx in enumerate(insert_positions):
        label = f"NIAH_SECRET_{offset + 1}"
        value = _gen_needle_value(rng)
        lines.insert(line_idx + offset, f"{label}={value}")  # +offset: earlier inserts shift later indices
        needles.append({"label": label, "value": value, "insert_line_index": line_idx})

    context = "\n".join(lines)

    question = (
        f"The context contains {n_needles} hidden line(s), each matching the pattern "
        f"NIAH_SECRET_<n>=<digits>, scattered among filler text. Find ALL of them and "
        f"return each as '<label>=<value>', one per line, sorted by label."
    )
    answer = "\n".join(f"{n['label']}={n['value']}" for n in sorted(needles, key=lambda x: x["label"]))

    return EvalSample(
        id=f"niah_{context_tokens_target}tok_{n_needles}needles_{sample_idx}",
        source="synthetic_niah",
        context=context,
        question=question,
        answer=answer,
        context_tokens=None,  # exact count populated by filter_by_token_length.py
        extra={
            "context_tokens_target": context_tokens_target,
            "n_needles": n_needles,
            "needles": needles,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scaled multi-needle NIAH samples")
    parser.add_argument(
        "--context-tokens",
        default="150000,300000,500000",
        help="Comma-separated target context sizes in (estimated) tokens.",
    )
    parser.add_argument("--n-needles", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1, help="Samples generated per context-size bucket.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    context_token_targets = [int(x) for x in args.context_tokens.split(",") if x.strip()]
    rng = random.Random(args.seed)

    samples: list[EvalSample] = []
    for target in context_token_targets:
        for rep in range(args.repeats):
            samples.append(generate_sample(rng, target, args.n_needles, rep))

    write_jsonl(samples, args.output)
    print(
        f"[gen_synthetic_niah] generated {len(samples)} samples "
        f"({len(context_token_targets)} size(s) x {args.repeats} repeat(s)) -> {args.output}"
    )
    print(
        "[gen_synthetic_niah] NOTE: context sizes are estimates -- run "
        "filter_by_token_length.py against the real Llama tokenizer to get "
        "exact context_tokens values before trusting the >131K threshold."
    )


if __name__ == "__main__":
    main()
