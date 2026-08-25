#!/usr/bin/env python3
"""Analyze the prompt-length distribution of layer1_delta_20260501.txt.

Phase A — full pass (streaming, char-only): scans all 859,988 rows,
extracts char counts for system content, user content, and the templated
formatted-prompt (computed as a length identity without re-running the
tokenizer's chat-template loop, see below). Cheap — bounded by disk I/O.

Phase B — tokenized sample: takes a deterministic sample of N rows
(spread evenly across the file), runs them through the actual Gemma 4
chat template + tokenizer to get exact token counts. Produces the
chars→tokens conversion factor.

Output:
  examples/dataset_analysis_full.json   — all numbers (counts, percentiles,
                                          conversion factor, sample size)
  examples/dataset_analysis_full.md     — human-readable summary table
  (raw per-row char lengths are NOT saved by default to keep this fast.)
"""
import argparse
import bisect
import gzip
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path


def percentile(sorted_vals, p):
    """Nearest-rank percentile (matches what the in-process metrics use)."""
    n = len(sorted_vals)
    if n == 0:
        return None
    k = max(0, min(n - 1, int(round(p / 100 * (n - 1)))))
    return sorted_vals[k]


def describe(vals, unit=""):
    """Return {min, p10, p25, p50, p75, p90, p95, p99, max, mean, stdev, count}."""
    if not vals:
        return {"count": 0, "unit": unit}
    s = sorted(vals)
    return {
        "count": len(s),
        "unit": unit,
        "min": s[0],
        "p10": percentile(s, 10),
        "p25": percentile(s, 25),
        "p50": percentile(s, 50),
        "p75": percentile(s, 75),
        "p90": percentile(s, 90),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "p99_9": percentile(s, 99.9),
        "max": s[-1],
        "mean": statistics.fmean(s),
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
        "sum": sum(s),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/nvmedata/data/layer1_delta_20260501.txt",
                    help="path to the full JSONL dataset")
    ap.add_argument("--sample-size", type=int, default=10000,
                    help="how many rows to tokenize (for token-distribution stats)")
    ap.add_argument("--model-path",
                    default="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it",
                    help="model to load the tokenizer + chat template from")
    ap.add_argument("--output-json",
                    default="/nvmedata/chenw/vllm-ra/examples/dataset_analysis_full.json")
    ap.add_argument("--output-md",
                    default="/nvmedata/chenw/vllm-ra/examples/dataset_analysis_full.md")
    ap.add_argument("--progress-every", type=int, default=50000)
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ===== Phase A: full streaming pass, char-only =====
    print(f"[A] Streaming {input_path} ({input_path.stat().st_size / 1e9:.2f} GB)...",
          flush=True)
    t0 = time.perf_counter()

    n_rows = 0
    n_skipped_format = 0
    sys_chars = []     # len of system content per row
    user_chars = []    # len of user content per row
    total_chars = []   # len of system + user content (chars only, no template overhead)
    n_msgs_seen = {}   # histogram of len(messages) per row
    roles_seen = {}    # histogram of roles seen

    open_fn = gzip.open if str(input_path).endswith(".gz") else open

    with open_fn(input_path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if not raw.strip():
                continue
            n_rows += 1
            try:
                d = json.loads(raw)
            except Exception:
                n_skipped_format += 1
                continue
            messages = d.get("messages") if isinstance(d, dict) else None
            if not isinstance(messages, list) or not messages:
                n_skipped_format += 1
                continue

            n_msgs_seen[len(messages)] = n_msgs_seen.get(len(messages), 0) + 1
            sc = uc = 0
            for m in messages:
                role = m.get("role", "")
                roles_seen[role] = roles_seen.get(role, 0) + 1
                c = len(m.get("content", "") or "")
                if role == "system":
                    sc += c
                elif role == "user":
                    uc += c
            sys_chars.append(sc)
            user_chars.append(uc)
            total_chars.append(sc + uc)

            if n_rows % args.progress_every == 0:
                elapsed = time.perf_counter() - t0
                rate = n_rows / elapsed
                print(f"  [{n_rows:>9,}] {elapsed:>6.1f}s  "
                      f"{rate:>7.0f} rows/s  ETA "
                      f"{(859988 - n_rows) / max(rate,1):>5.0f}s",
                      flush=True)

    phase_a_elapsed = time.perf_counter() - t0
    print(f"[A] done in {phase_a_elapsed:.1f}s — {n_rows:,} rows parsed, "
          f"{n_skipped_format:,} skipped",
          flush=True)

    # ===== Phase B: tokenized sample =====
    print(f"[B] Loading tokenizer from {args.model_path}...", flush=True)
    from transformers import AutoTokenizer  # heavy; defer until after Phase A
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    has_chat = tok.chat_template is not None
    print(f"[B] chat_template: {'yes' if has_chat else 'no'}  vocab={len(tok):,}",
          flush=True)

    # Deterministic stratified sample: every (n_rows / sample_size)-th row
    if args.sample_size >= n_rows:
        sample_indices = set(range(n_rows))
    else:
        step = n_rows / args.sample_size
        sample_indices = {int(i * step) for i in range(args.sample_size)}
    print(f"[B] Sampling {len(sample_indices):,} rows (every ~{n_rows//len(sample_indices):.0f}th row)...",
          flush=True)

    t0 = time.perf_counter()
    sample_token_lens = []
    sample_char_lens = []
    n_processed = 0
    n_seen = 0
    with open_fn(input_path, "rt", encoding="utf-8", errors="replace") as f:
        idx = -1
        for raw in f:
            if not raw.strip():
                continue
            idx += 1
            if idx not in sample_indices:
                continue
            n_seen += 1
            try:
                d = json.loads(raw)
                messages = d.get("messages")
                if not isinstance(messages, list) or not messages:
                    continue
                if has_chat:
                    formatted = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                else:
                    formatted = "\n\n".join(
                        f"{m.get('role','')}: {m.get('content','')}"
                        for m in messages)
                ids = tok.encode(formatted, add_special_tokens=False)
                sample_token_lens.append(len(ids))
                sample_char_lens.append(len(formatted))
                n_processed += 1

                if n_processed % 1000 == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"  [{n_processed:>6,}/{len(sample_indices):,}] "
                          f"{elapsed:>6.1f}s  "
                          f"{n_processed/elapsed:>6.0f} rows/s",
                          flush=True)
            except Exception as e:
                continue
    phase_b_elapsed = time.perf_counter() - t0
    print(f"[B] done in {phase_b_elapsed:.1f}s — tokenized {n_processed:,} sampled rows",
          flush=True)

    # ===== Build output =====
    # Chars/token ratio from the sample
    if sample_token_lens and sample_char_lens:
        ratios = [c / max(t, 1) for c, t in zip(sample_char_lens, sample_token_lens)]
    else:
        ratios = []

    output = {
        "input_file": str(input_path),
        "file_size_bytes": input_path.stat().st_size,
        "phase_a_seconds": phase_a_elapsed,
        "phase_b_seconds": phase_b_elapsed,
        "counts": {
            "rows_parsed": n_rows,
            "rows_skipped_format_error": n_skipped_format,
            "messages_per_row_histogram": n_msgs_seen,
            "roles_histogram": roles_seen,
        },
        "char_distribution": {
            "system_content_chars": describe(sys_chars, "chars"),
            "user_content_chars":   describe(user_chars, "chars"),
            "total_content_chars":  describe(total_chars, "chars"),
        },
        "token_distribution_sample": {
            "sample_size": n_processed,
            "tokens_per_prompt_formatted": describe(sample_token_lens, "tokens"),
            "formatted_string_chars":      describe(sample_char_lens, "chars"),
            "chars_per_token_ratio":       describe(ratios, "chars/token"),
        },
    }

    Path(args.output_json).write_text(json.dumps(output, indent=2, default=str))
    print(f"\nwrote {args.output_json}", flush=True)

    # ===== Markdown summary =====
    md = []
    md.append(f"# Dataset analysis — `{input_path.name}`\n")
    md.append(f"Source: `{input_path}` ({output['file_size_bytes']/1e9:.2f} GB)\n")
    md.append(f"Rows parsed: **{n_rows:,}**  "
              f"(skipped format errors: {n_skipped_format:,})\n")
    md.append(f"Messages-per-row histogram: {n_msgs_seen}\n")
    md.append(f"Roles seen: {roles_seen}\n")
    md.append(f"Phase A (full stream, char-only): {phase_a_elapsed:.1f}s")
    md.append(f"Phase B (sample tokenization, n={n_processed:,}): {phase_b_elapsed:.1f}s\n")

    def md_row(label, d):
        if d.get("count", 0) == 0:
            return f"| {label} | — | — | — | — | — | — | — | — | — |"
        return ("| " + label + " | " +
                f"{d['min']:,} | "
                f"{d['p10']:,} | "
                f"{d['p25']:,} | "
                f"{d['p50']:,} | "
                f"{d['p75']:,} | "
                f"{d['p90']:,} | "
                f"{d['p95']:,} | "
                f"{d['p99']:,} | "
                f"{d['max']:,} | "
                f"{d['mean']:,.1f} |")

    md.append("\n## Char-length distribution (full 859,988-row pass)\n")
    md.append("| metric | min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    md.append(md_row("system content chars/row",
                     output["char_distribution"]["system_content_chars"]))
    md.append(md_row("user content chars/row",
                     output["char_distribution"]["user_content_chars"]))
    md.append(md_row("total content chars/row",
                     output["char_distribution"]["total_content_chars"]))

    md.append(f"\n## Token-length distribution (sample n={n_processed:,})\n")
    md.append("Computed with the actual Gemma 4 tokenizer + chat template "
              "(`apply_chat_template(..., add_generation_prompt=True)` "
              "then `encode(add_special_tokens=False)`).\n")
    md.append("| metric | min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    md.append(md_row("tokens per formatted prompt",
                     output["token_distribution_sample"]["tokens_per_prompt_formatted"]))
    md.append(md_row("formatted-string chars",
                     output["token_distribution_sample"]["formatted_string_chars"]))

    md.append("\n## Chars → tokens conversion factor (per-row, sample)\n")
    cr = output["token_distribution_sample"]["chars_per_token_ratio"]
    md.append("| min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    md.append("| " +
              f"{cr['min']:.2f} | {cr['p10']:.2f} | {cr['p25']:.2f} | "
              f"{cr['p50']:.2f} | {cr['p75']:.2f} | {cr['p90']:.2f} | "
              f"{cr['p95']:.2f} | {cr['p99']:.2f} | {cr['max']:.2f} | "
              f"{cr['mean']:.2f} |")
    md.append("\nUse median chars/token ratio to estimate token counts in the "
              "full dataset's char-distribution table above when you need "
              "token-level numbers without running the tokenizer.\n")

    Path(args.output_md).write_text("\n".join(md))
    print(f"wrote {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
