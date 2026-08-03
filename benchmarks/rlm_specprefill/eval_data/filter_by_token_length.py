#!/usr/bin/env python3
"""Applies the ablation's real >131K-*token* eval-set constraint, measured
against the TARGET model's own tokenizer (Llama), not an estimate.

Why not reuse ../../rlm/rlm/utils/token_utils.py's `count_tokens`: that
utility is tiktoken-based with a chars/4 fallback, and exists for the
Anthropic ROOT model's own compaction accounting against Anthropic's context
limit -- a different model, a different purpose. Its `MODEL_CONTEXT_LIMITS`
table has no Llama entry at all (checked directly -- only OpenAI/Anthropic/
Gemini/Qwen/Kimi/GLM keys are present), so `get_context_limit()` would
silently fall back to its generic `DEFAULT_CONTEXT_LIMIT` (128,000) for any
Llama model name -- an even less specific guess than a wrong per-model
entry would be. Reusing it here would risk the exact failure mode this
script exists to avoid: a document that LOOKS like >131K tokens under a
generic estimate but is actually well under that in Llama's real tokenizer
(or vice versa) shouldn't silently define whether "RLM should have been used
here" per ../rlm_specprefill_ablation_plan.md's EVAL SET CONSTRAINTS.

Token count is computed over `context` alone (not `context + question`) --
the ablation's >131K threshold is about the massive-context regime RLM is
meant for; the question is comparatively tiny and doesn't change which
regime a sample falls into.

Usage:
    python3 filter_by_token_length.py \\
        --input longbench_v2_long_samples.jsonl \\
        --tokenizer-path $LLAMA31_8B_MODEL_PATH \\
        --min-tokens 131000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schema import EvalSample, read_jsonl, write_jsonl  # noqa: E402


def load_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "filter_by_token_length.py needs `transformers` to load the real "
            "Llama tokenizer (pip install transformers). This is a GPU-node "
            "step, not expected to run on a machine without the target "
            "model's checkpoint available."
        ) from e
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def count_tokens_real(tokenizer, text: str) -> int:
    # add_special_tokens=False: we want the raw document's token footprint,
    # not the count including whatever chat-template/BOS overhead a full
    # request would add -- that's the target-serving stage's concern
    # (target_stage/vllm_offline_engine.py), not this filter's.
    return len(tokenizer.encode(text, add_special_tokens=False))


def filter_samples(
    samples: list[EvalSample], tokenizer, min_tokens: int, max_tokens: int | None = None
) -> tuple[list[EvalSample], dict[str, int]]:
    kept: list[EvalSample] = []
    stats = {"total": len(samples), "kept": 0, "below_min": 0, "above_max": 0}

    for sample in samples:
        n_tokens = count_tokens_real(tokenizer, sample.context)
        sample.context_tokens = n_tokens

        if n_tokens < min_tokens:
            stats["below_min"] += 1
            continue
        if max_tokens is not None and n_tokens > max_tokens:
            stats["above_max"] += 1
            continue

        kept.append(sample)
        stats["kept"] += 1

    return kept, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter an eval-set JSONL to samples above a real-tokenizer token-count threshold"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <input>.filtered.jsonl")
    parser.add_argument("--tokenizer-path", required=True, help="Local path or HF hub id, e.g. $LLAMA31_8B_MODEL_PATH")
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=131_000,
        help="Per ../rlm_specprefill_ablation_plan.md's EVAL SET CONSTRAINTS: 'comfortably above ~131K tokens'.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional upper bound, e.g. to stay within a sane evidence-extraction budget.",
    )
    args = parser.parse_args()

    output = args.output or args.input.with_suffix(".filtered.jsonl")

    samples = read_jsonl(args.input)
    print(f"[filter_by_token_length] loaded {len(samples)} samples from {args.input}")

    tokenizer = load_tokenizer(args.tokenizer_path)
    kept, stats = filter_samples(samples, tokenizer, args.min_tokens, args.max_tokens)

    write_jsonl(kept, output)
    print(
        f"[filter_by_token_length] total={stats['total']} kept={stats['kept']} "
        f"below_min={stats['below_min']} above_max={stats['above_max']} -> {output}"
    )
    if stats["kept"] == 0:
        print(
            "[filter_by_token_length] WARNING: 0 samples survived the filter -- "
            "the input pool may not actually reach the >131K-token regime under "
            "the real tokenizer, even if it was labeled 'long' by word count."
        )


if __name__ == "__main__":
    main()
