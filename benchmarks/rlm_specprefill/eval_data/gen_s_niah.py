#!/usr/bin/env python3
"""RULER-faithful single-needle-in-a-haystack (S-NIAH) generator.

Implements the S-NIAH task as described in the RLM paper, citing RULER
(Hsieh et al., 2024, "RULER: What's the Real Context Size of Your
Long-Context Language Models?"):

    S-NIAH. Following the single needle-in-the-haystack task in RULER, we
    consider a set of 50 single tasks that require finding a specific
    phrase or number in a large set of unrelated text. Here, the
    information being sought scales as O(1) with respect to input length.

Distinct from ../gen_synthetic_niah.py, which is a free-form, SCALED
multi-needle generator (needle count is an independent knob, default 3,
random-ASCII-gibberish haystack) used as calibration/sweep_n_min.py's
day-1 candidate-text pool -- a different job than a citable,
standard-shaped single-needle benchmark. This script exists alongside
that one, not as a replacement: exactly ONE needle per sample (no
`--n-needles` knob at all), a haystack of repeated/shuffled simple "noise"
filler sentences (not random-ASCII gibberish, and not a real-essay corpus
that would need a network download), and a `--n-tasks` count (default 50,
matching the RLM paper's own quoted number) generated per context-length
bucket.

The needle itself (a fixed 9-digit number, or a fixed two-word phrase) is
constant-size regardless of context length -- this is the O(1) property
the RLM paper's own description calls out, and it's covered by an explicit
unit test in ../tests/test_gen_s_niah.py.

IMPORTANT: `--context-tokens` here is a GENERATION-TIME size target using a
rough chars-per-token estimate (same CHARS_PER_TOKEN_ESTIMATE=4 convention
as ../gen_synthetic_niah.py and ../../rlm/rlm/utils/token_utils.py). It is
NOT the authoritative token count -- run filter_by_token_length.py on this
script's output (against the real target-model tokenizer) to get an exact
`context_tokens` value before trusting a sample against the ablation's
>131K threshold, same as for the LongBench-v2-derived and multi-needle-NIAH
samples.

Usage:
    python3 gen_s_niah.py --context-tokens 150000,300000,500000 --n-tasks 50
    python3 gen_s_niah.py --context-tokens 500,1000 --n-tasks 2 --needle-type number  # fast smoke test
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schema import EvalSample, write_jsonl  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).parent / "s_niah_samples.jsonl"

# Matches gen_synthetic_niah.py's / rlm/utils/token_utils.py's own fallback
# estimate -- see module docstring above for why this is sizing-only, not
# the real threshold check.
CHARS_PER_TOKEN_ESTIMATE = 4

VALID_NEEDLE_TYPES = ("number", "phrase", "mixed")

# RULER's own simplest S-NIAH haystack style: a small, hardcoded, repeated
# "noise" haystack -- deliberately NOT random-ASCII gibberish (weaker
# distractor, gen_synthetic_niah.py's style) and NOT a real-essay corpus
# (would need a network download this project's other eval_data/ scripts
# avoid where possible). Sampled with replacement + shuffled per document,
# so no two samples' haystacks are identical even at the same size.
_NOISE_SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
    "The cat sat on the mat.",
    "It was a quiet afternoon.",
    "The train arrived on time.",
    "Birds were singing in the trees.",
    "The coffee had gone cold.",
    "A gentle breeze moved through the open window.",
    "The library was silent except for turning pages.",
]

# Small, self-contained word lists for the "phrase" needle type -- no
# network/corpus dependency. 16 x 16 = 256 possible phrases.
_ADJECTIVES = [
    "crimson", "silent", "frozen", "hollow", "amber", "distant",
    "velvet", "brittle", "hidden", "golden", "quiet", "sudden",
    "ancient", "narrow", "polished", "faint",
]
_NOUNS = [
    "falcon", "lantern", "harbor", "compass", "orchard", "thunder",
    "ribbon", "granite", "meadow", "citadel", "whisper", "anchor",
    "ember", "canyon", "voyage", "mosaic",
]

_NEEDLE_TEMPLATES = {
    "number": "The special magic number for this passage is: {value}.",
    "phrase": "The special magic phrase for this passage is: {value}.",
}


def _gen_needle_value(rng: random.Random, needle_type: str) -> str:
    """O(1) in context length by construction -- a fixed-digit-count number
    or a fixed two-word phrase, never scaled with haystack size."""
    if needle_type == "number":
        return str(rng.randint(100_000_000, 999_999_999))  # always 9 digits
    if needle_type == "phrase":
        return f"{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)}"
    raise ValueError(f"needle_type must be 'number' or 'phrase', got {needle_type!r}")


def generate_sample(
    rng: random.Random,
    context_tokens_target: int,
    needle_type: str,
    sample_idx: int,
) -> EvalSample:
    """`needle_type` is 'number', 'phrase', or 'mixed' ('mixed' resolves to
    one or the other per-call via `rng.choice`, matching the RLM paper's
    own "phrase or number" framing)."""
    if needle_type not in VALID_NEEDLE_TYPES:
        raise ValueError(f"needle_type must be one of {VALID_NEEDLE_TYPES}, got {needle_type!r}")
    resolved_type = rng.choice(["number", "phrase"]) if needle_type == "mixed" else needle_type
    value = _gen_needle_value(rng, resolved_type)
    needle_sentence = _NEEDLE_TEMPLATES[resolved_type].format(value=value)

    target_chars = context_tokens_target * CHARS_PER_TOKEN_ESTIMATE
    avg_sentence_chars = sum(len(s) for s in _NOISE_SENTENCES) // len(_NOISE_SENTENCES)
    n_sentences = max(2, target_chars // (avg_sentence_chars + 1))  # +1 for joiner space

    sentences = [rng.choice(_NOISE_SENTENCES) for _ in range(n_sentences)]
    insert_idx = rng.randrange(len(sentences) + 1)
    sentences.insert(insert_idx, needle_sentence)
    context = " ".join(sentences)

    label = "NIAH_SECRET_1"
    what = "number" if resolved_type == "number" else "phrase"
    question = (
        f"The context above contains one hidden sentence stating a special magic {what}. "
        f"Find it and return ONLY the {what} itself, with no other text."
    )
    answer = value

    return EvalSample(
        id=f"s_niah_{context_tokens_target}tok_{resolved_type}_{sample_idx}",
        source="s_niah",
        context=context,
        question=question,
        answer=answer,
        context_tokens=None,  # exact count populated by filter_by_token_length.py
        extra={
            "context_tokens_target": context_tokens_target,
            "n_needles": 1,
            "needle_type": resolved_type,
            "needles": [{"label": label, "value": value, "insert_line_index": insert_idx}],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RULER-style single-needle S-NIAH samples (Hsieh et al., 2024)"
    )
    parser.add_argument(
        "--context-tokens",
        default="150000,300000,500000",
        help="Comma-separated target context sizes in (estimated) tokens.",
    )
    parser.add_argument(
        "--needle-type",
        choices=VALID_NEEDLE_TYPES,
        default="mixed",
        help=(
            "'number' or 'phrase' pins every sample to that needle type; "
            "'mixed' (default) picks per-sample, matching the spec's own "
            "'phrase or number' framing."
        ),
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=50,
        help="Samples generated per context-size bucket -- 50 matches the RLM paper's own quoted S-NIAH task count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    context_token_targets = [int(x) for x in args.context_tokens.split(",") if x.strip()]
    rng = random.Random(args.seed)

    samples: list[EvalSample] = []
    for target in context_token_targets:
        for task_idx in range(args.n_tasks):
            samples.append(generate_sample(rng, target, args.needle_type, task_idx))

    write_jsonl(samples, args.output)
    print(
        f"[gen_s_niah] generated {len(samples)} samples "
        f"({len(context_token_targets)} size(s) x {args.n_tasks} task(s)) -> {args.output}"
    )
    print(
        "[gen_s_niah] NOTE: context sizes are estimates -- run "
        "filter_by_token_length.py against the real target-model tokenizer "
        "to get exact context_tokens values before trusting the >131K threshold."
    )


if __name__ == "__main__":
    main()
