#!/usr/bin/env python3
"""Prepares an AIME sample subset for the evaluation pipeline.

This pipeline measures throughput/spec-decode metrics only (see
../README.md) -- there is no accuracy scoring. AIME is used purely as a
source of realistic, varied-length reasoning prompts.

Downloads the AIME 1983-2024 problem set (933 problems; no single year has
enough problems on its own, so this pools across years) from the
`gneubig/aime-1983-2024` Hugging Face dataset, normalizes it, and writes
datasets/aime_samples.jsonl as {"id", "year", "prompt", "answer": <int>} rows.
"answer" is kept as optional provenance metadata (harmless, already
extracted) but is not consumed by anything downstream in this pipeline.

This script writes the raw problem statement as "prompt" (not
chat-template-rendered) -- rendering happens in run_pipeline.py at eval
time, matching the convention in ../../gemma4_moe_benchmarks/prep_dataset.py.

Usage:
    python3 prep_aime.py --max-keep 300
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import requests

# This CSV is text-only -- diagrams referenced by the original PDF are not
# carried over. Rows matching this pattern depend on a visual the model
# never receives, so they're excluded to keep the prompt set realistic
# (an unanswerable prompt isn't a representative throughput workload).
DIAGRAM_DEPENDENCY_PATTERN = re.compile(
    r"\b(figure|diagram|shown below|shown above|picture)\b"
    r"|\.(png|jpe?g|gif)\b|pdfresizer",
    re.IGNORECASE,
)

HF_CSV_URL = (
    "https://huggingface.co/datasets/gneubig/aime-1983-2024/"
    "resolve/main/AIME_Dataset_1983_2024.csv"
)
DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"
DEFAULT_OUTPUT = Path(__file__).parent / "aime_samples.jsonl"


def download_and_cache_csv(url: str, cache_dir: Path) -> Path:
    """Download the source CSV once and reuse it on subsequent runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / url.split("/")[-1]
    if dest.exists():
        return dest
    print(f"[prep_aime] downloading {url}", flush=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def parse_answer(raw: str) -> int | None:
    """AIME answers are integers 0-999. Returns None if unparseable
    (e.g. the 2022-II-8 row, whose recorded answer is the free-text
    "080 or 081 (both were accepted)")."""
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return None
    if not (0 <= value <= 999):
        return None
    return value


def build_aime_samples(
    csv_path: Path,
    max_keep: int = 300,
    seed: int = 42,
) -> list[dict]:
    """Parse the AIME CSV into {"id", "year", "prompt", "answer"} rows."""
    import random

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    samples: list[dict] = []
    skipped_unparseable = 0
    skipped_diagram = 0
    for row in rows:
        question = (row.get("Question") or "").strip()
        answer = parse_answer(row.get("Answer") or "")
        if not question or answer is None:
            skipped_unparseable += 1
            continue
        if DIAGRAM_DEPENDENCY_PATTERN.search(question):
            skipped_diagram += 1
            continue
        samples.append(
            {
                "id": row.get("ID"),
                "year": row.get("Year"),
                "prompt": question,
                "answer": answer,
            }
        )

    print(
        f"[prep_aime] parsed={len(rows)} kept={len(samples)} "
        f"skipped_unparseable={skipped_unparseable} skipped_diagram={skipped_diagram}"
    )

    if max_keep >= 0 and len(samples) > max_keep:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_keep)
        samples.sort(key=lambda s: (s["year"], s["id"]))

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AIME eval samples")
    parser.add_argument("--hf-csv-url", default=HF_CSV_URL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--max-keep",
        type=int,
        default=300,
        help="Cap on number of samples; -1 keeps all available (~932).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    csv_path = download_and_cache_csv(args.hf_csv_url, args.cache_dir)
    samples = build_aime_samples(csv_path, max_keep=args.max_keep, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prep_aime] wrote {len(samples)} rows -> {args.output}")


if __name__ == "__main__":
    main()
