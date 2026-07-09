#!/usr/bin/env python3
"""Prepares a GPQA Diamond sample subset for the evaluation pipeline.

This pipeline measures throughput/spec-decode metrics only (see
../README.md) -- there is no accuracy scoring. GPQA Diamond is used
purely as a source of realistic, short multiple-choice-style prompts
(a useful contrast to AIME's long reasoning chains and LiveCodeBench's
code-generation prompts).

Downloads gpqa_diamond.csv (~198 expert-written multiple-choice science
questions) from https://huggingface.co/datasets/Idavidrein/gpqa, shuffles
each question's 4 answer choices into a random order (the source data
always lists the correct answer first/separately -- using that order
verbatim would leak a positional bias if this were ever used for scoring
later), and writes datasets/gpqa_diamond_samples.jsonl as
{"id", "subdomain", "prompt", "choices": [...], "answer": "A"} rows.
"choices"/"answer" are kept as harmless provenance metadata but nothing
downstream in this pipeline consumes them.

NOTE: this dataset is gated on Hugging Face -- you must (1) accept its
terms at https://huggingface.co/datasets/Idavidrein/gpqa while logged in,
and (2) have HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) set in the environment,
same as gemma4_moe_benchmarks/.env_exports.sh already does for model
downloads. Anonymous requests get a 401.

This script writes the raw question text as "prompt" (not chat-rendered,
and not pre-formatted with lettered options) -- rendering the final
multiple-choice prompt happens in run_pipeline.py at eval time, matching
the convention in ../../gemma4_moe_benchmarks/prep_dataset.py.

Usage:
    python3 prep_gpqa_diamond.py --max-keep -1   # keep all ~198
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import requests

HF_CSV_URL = "https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_diamond.csv"
DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"
DEFAULT_OUTPUT = Path(__file__).parent / "gpqa_diamond_samples.jsonl"

CHOICE_LETTERS = ["A", "B", "C", "D"]


def _resolve_hf_token(explicit: str | None) -> str | None:
    return explicit or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN"
    )


def download_and_cache_csv(url: str, cache_dir: Path, token: str | None) -> Path:
    """Download the source CSV once and reuse it on subsequent runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / url.split("/")[-1]
    if dest.exists():
        return dest

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    print(f"[prep_gpqa_diamond] downloading {url}", flush=True)
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 401:
        raise RuntimeError(
            "401 from Hugging Face for Idavidrein/gpqa: no valid token was "
            "sent. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in the "
            "environment before running this script -- see "
            "gemma4_moe_benchmarks/.env_exports.sh for the existing "
            "convention."
        )
    if resp.status_code == 403:
        raise RuntimeError(
            "403 from Hugging Face for Idavidrein/gpqa: your token is "
            "valid but that account hasn't been granted access to this "
            "gated dataset yet. Log in as that account at "
            "https://huggingface.co/datasets/Idavidrein/gpqa and "
            "request/accept access there (may require manual approval), "
            "then retry."
        )
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def build_gpqa_samples(
    csv_path: Path,
    max_keep: int = -1,
    seed: int = 42,
) -> list[dict]:
    """Parse the GPQA Diamond CSV into
    {"id", "subdomain", "prompt", "choices", "answer"} rows, with choices
    shuffled per-question so the correct answer isn't always in the same
    position."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = [
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    ]
    if rows and not all(col in rows[0] for col in required):
        raise KeyError(
            f"Expected columns {required} not all found in "
            f"{csv_path}. Available columns: {sorted(rows[0].keys())}"
        )

    rng = random.Random(seed)
    samples: list[dict] = []
    skipped = 0
    for i, row in enumerate(rows):
        question = (row.get("Question") or "").strip()
        correct = (row.get("Correct Answer") or "").strip()
        incorrect = [
            (row.get(f"Incorrect Answer {n}") or "").strip() for n in (1, 2, 3)
        ]
        if not question or not correct or not all(incorrect):
            skipped += 1
            continue

        options = [correct] + incorrect
        order = list(range(4))
        rng.shuffle(order)
        shuffled_choices = [options[j] for j in order]
        correct_letter = CHOICE_LETTERS[order.index(0)]

        samples.append(
            {
                "id": row.get("Record ID") or f"gpqa_diamond-{i}",
                "subdomain": row.get("High-level domain") or row.get("Subdomain"),
                "prompt": question,
                "choices": shuffled_choices,
                "answer": correct_letter,
            }
        )

    print(f"[prep_gpqa_diamond] parsed={len(rows)} kept={len(samples)} skipped={skipped}")

    if max_keep >= 0 and len(samples) > max_keep:
        samples = rng.sample(samples, max_keep)
        samples.sort(key=lambda s: s["id"])

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GPQA Diamond eval samples")
    parser.add_argument("--hf-csv-url", default=HF_CSV_URL)
    parser.add_argument("--hf-token", default=None, help="Defaults to $HF_TOKEN")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--max-keep",
        type=int,
        default=-1,
        help="Cap on number of samples; -1 keeps all available (~198).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    token = _resolve_hf_token(args.hf_token)
    csv_path = download_and_cache_csv(args.hf_csv_url, args.cache_dir, token)
    samples = build_gpqa_samples(csv_path, max_keep=args.max_keep, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prep_gpqa_diamond] wrote {len(samples)} rows -> {args.output}")


if __name__ == "__main__":
    main()
