#!/usr/bin/env python3
"""Prepares a LiveCodeBench sample subset for the evaluation pipeline.

This pipeline measures throughput/spec-decode metrics only (see
../README.md) -- there is no accuracy scoring, so this script only needs
realistic prompt text, not ground-truth test cases. It streams
livecodebench/code_generation_lite from Hugging Face (not gated) and stops
once enough valid samples are collected -- the source files are large
(test6.jsonl, the default, is ~134MB; test.jsonl is ~1.25GB), so this
deliberately never downloads one in full.

Streaming these files here has hit genuine mid-download connection drops
(ChunkedEncodingError/IncompleteRead) at different byte offsets on
different runs -- ordinary network flakiness, not something a smaller
file avoids on its own. download_with_resume() below retries on failure
using HTTP Range requests to continue from the exact byte offset already
read, rather than restarting from scratch.

Writes datasets/livecodebench_samples.jsonl as
{"id", "platform", "difficulty", "prompt"} rows, where "prompt" is the
problem statement with starter_code appended when present (so the prompt
is realistic/complete, matching what a real LiveCodeBench request would
contain). Never touches public_test_cases or private_test_cases -- the
latter is base64+zlib+pickle encoded, which would be an unsafe
deserialization risk from an external dataset, and neither is needed now
that scoring is out of scope.

This script writes raw prompt text (not chat-rendered) -- rendering
happens in run_pipeline.py at eval time, matching the convention in
../../gemma4_moe_benchmarks/prep_dataset.py.

Usage:
    python3 prep_livecodebench.py --max-keep 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

HF_JSONL_URL = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/"
    "resolve/main/{filename}"
)
DEFAULT_OUTPUT = Path(__file__).parent / "livecodebench_samples.jsonl"


def _parse_row(line: bytes):
    """Returns a normalized sample dict, or None if the line should be
    skipped (malformed / no problem text)."""
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None

    question_content = (row.get("question_content") or "").strip()
    if not question_content:
        return None

    starter_code = (row.get("starter_code") or "").strip()
    prompt = question_content
    if starter_code:
        prompt = f"{question_content}\n\n{starter_code}"

    return {
        "id": row.get("question_id"),
        "platform": row.get("platform"),
        "difficulty": row.get("difficulty"),
        "prompt": prompt,
    }


def stream_livecodebench_samples(
    filename: str = "test6.jsonl",
    max_keep: int = 300,
    seen_ids: set | None = None,
    max_retries: int = 8,
    retry_backoff_s: float = 3.0,
):
    """Streams the source file line-by-line, resuming via HTTP Range
    requests on connection failure instead of restarting from byte 0.
    Stops as soon as len(seen_ids) reaches max_keep -- these files are
    large and we never want to read one in full for a few-hundred-sample
    subset.

    seen_ids is mutated in place and may be shared across multiple calls
    (different filenames) so a caller can pull remaining samples from a
    fallback file once the first one runs out, without double-counting
    ids that appear in both (release snapshots overlap)."""
    url = HF_JSONL_URL.format(filename=filename)

    if seen_ids is None:
        seen_ids = set()
    scanned = 0
    skipped = 0
    byte_offset = 0
    buffer = b""
    attempt = 0

    def process_buffered_lines():
        nonlocal buffer, scanned, skipped
        results = []
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if not line.strip():
                continue
            scanned += 1
            sample = _parse_row(line)
            if sample is None:
                skipped += 1
                continue
            if sample["id"] in seen_ids:
                # Duplicate: already yielded from this file (retry overlap)
                # or from an earlier fallback file (release snapshots
                # overlap heavily).
                continue
            seen_ids.add(sample["id"])
            results.append(sample)
            if len(seen_ids) >= max_keep:
                break
        return results

    while len(seen_ids) < max_keep and attempt <= max_retries:
        attempt += 1
        headers = {"Range": f"bytes={byte_offset}-"} if byte_offset else {}
        label = "resuming" if byte_offset else "streaming"
        print(
            f"[prep_livecodebench] {label} {url} "
            f"(attempt {attempt}/{max_retries + 1}, byte_offset={byte_offset})",
            flush=True,
        )
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
                if byte_offset and resp.status_code == 200:
                    # Server ignored the Range request (no resume support) --
                    # bail rather than silently re-scan from the start with
                    # a byte_offset that no longer lines up.
                    raise RuntimeError(
                        "server does not support Range resume (got 200, "
                        "expected 206); can't safely continue"
                    )
                if byte_offset and resp.status_code == 416:
                    # Requested bytes past end-of-file: the previous
                    # attempt actually finished the whole file (byte_offset
                    # landed exactly on its size), it just didn't contain
                    # max_keep valid rows. Not an error -- stop cleanly.
                    print(
                        f"[prep_livecodebench] reached end of file at "
                        f"byte_offset={byte_offset}; only {len(seen_ids)} "
                        f"valid rows available so far (< max_keep={max_keep})",
                        flush=True,
                    )
                    break
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    byte_offset += len(chunk)
                    buffer += chunk
                    for sample in process_buffered_lines():
                        yield sample
                    if len(seen_ids) >= max_keep:
                        break
            if len(seen_ids) >= max_keep:
                break
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as e:
            print(
                f"[prep_livecodebench] connection error at byte_offset="
                f"{byte_offset} (kept={len(seen_ids)} so far): {e}",
                flush=True,
            )
            if attempt > max_retries:
                raise
            time.sleep(retry_backoff_s)
            continue

    print(
        f"[prep_livecodebench] file={filename} scanned={scanned} "
        f"total_kept={len(seen_ids)} skipped={skipped} attempts={attempt}"
    )


# Ascending by size (see conversation: HEAD-checked sizes). Release
# snapshots overlap heavily, so falling through accumulates *new* unique
# problems rather than re-scanning duplicates from scratch.
FALLBACK_FILES = ["test6.jsonl", "test5.jsonl", "test3.jsonl", "test2.jsonl"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LiveCodeBench eval samples")
    parser.add_argument(
        "--hf-jsonl-name",
        default=None,
        help=(
            "Single file to stream from, skipping the fallback sequence "
            f"(default fallback order: {FALLBACK_FILES})."
        ),
    )
    parser.add_argument("--max-keep", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    files = [args.hf_jsonl_name] if args.hf_jsonl_name else FALLBACK_FILES

    seen_ids: set = set()
    samples: list[dict] = []
    for filename in files:
        if len(seen_ids) >= args.max_keep:
            break
        before = len(seen_ids)
        samples.extend(
            stream_livecodebench_samples(
                filename=filename,
                max_keep=args.max_keep,
                seen_ids=seen_ids,
                max_retries=args.max_retries,
            )
        )
        print(
            f"[prep_livecodebench] after {filename}: "
            f"+{len(seen_ids) - before} new, {len(seen_ids)}/{args.max_keep} total",
            flush=True,
        )

    if len(seen_ids) < args.max_keep:
        print(
            f"[prep_livecodebench] WARNING: only found {len(seen_ids)}/"
            f"{args.max_keep} unique samples across all fallback files "
            f"{files}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prep_livecodebench] wrote {len(samples)} rows -> {args.output}")


if __name__ == "__main__":
    main()
