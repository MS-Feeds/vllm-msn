#!/usr/bin/env python3
"""Drive a vLLM server with random samples from LiveCodeBench, AIME, and GPQA Diamond.

Designed for two purposes:
1. Smoke-test the running vLLM service with realistic, diverse prompts.
2. Generate driving traffic for the lightning-indexer logit experiment, so the
   per-layer histograms get populated with non-trivial inputs.

Output: a JSONL file with one row per prompt: {source, id, prompt, ok, latency,
completion, usage} so you can inspect responses afterward.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Missing 'datasets'. Install with: pip install datasets")


def load_livecodebench(n: int, seed: int):
    try:
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[livecodebench] skipped: {e}", file=sys.stderr)
        return []
    rows = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for r in rows:
        body = r.get("question_content") or r.get("question_title") or str(r)
        prompt = (
            "You are an expert competitive programmer. Solve the following "
            "problem. Show your reasoning, then provide a final solution.\n\n"
            f"{body}"
        )
        out.append(
            {
                "source": "livecodebench",
                "id": str(r.get("question_id", "?")),
                "prompt": prompt,
            }
        )
    return out


def load_aime(n: int, seed: int):
    try:
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    except Exception as e:
        print(f"[aime] skipped: {e}", file=sys.stderr)
        return []
    rows = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for r in rows:
        prompt = (
            "Solve the following AIME problem step by step. The answer is an "
            "integer between 0 and 999. Put your final answer in \\boxed{}.\n\n"
            f"{r['Problem']}"
        )
        out.append(
            {
                "source": "aime",
                "id": str(r.get("ID", "?")),
                "prompt": prompt,
            }
        )
    return out


def load_gpqa_diamond(n: int, seed: int):
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e:
        print(
            f"[gpqa_diamond] skipped (often gated; run "
            f"`huggingface-cli login` first): {e}",
            file=sys.stderr,
        )
        return []
    rows = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for r in rows:
        rng = random.Random(seed ^ (hash(r.get("Question", "")) & 0xFFFFFFFF))
        choices = [
            r["Correct Answer"],
            r["Incorrect Answer 1"],
            r["Incorrect Answer 2"],
            r["Incorrect Answer 3"],
        ]
        order = list(range(4))
        rng.shuffle(order)
        labeled = "\n".join(
            f"({chr(ord('A') + i)}) {choices[order[i]]}" for i in range(4)
        )
        prompt = (
            "Answer the following multiple-choice question. Reason step by "
            "step, then end with 'Answer: X' where X is the letter A-D.\n\n"
            f"Question: {r['Question']}\n\nChoices:\n{labeled}"
        )
        out.append(
            {
                "source": "gpqa_diamond",
                "id": str(r.get("Record ID", "?")),
                "prompt": prompt,
            }
        )
    return out


LOADERS = {
    "livecodebench": load_livecodebench,
    "aime": load_aime,
    "gpqa_diamond": load_gpqa_diamond,
}


def send(url: str, model: str, prompt: str, max_tokens: int,
         temperature: float, timeout: float) -> dict:
    t0 = time.time()
    try:
        r = requests.post(
            f"{url}/v1/chat/completions",
            json=dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=timeout,
        )
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": str(e), "latency": time.time() - t0}
    dt = time.time() - t0
    if r.status_code != 200:
        return {
            "ok": False,
            "status": r.status_code,
            "body": r.text[:500],
            "latency": dt,
        }
    j = r.json()
    return {
        "ok": True,
        "latency": dt,
        "completion": j["choices"][0]["message"]["content"],
        "usage": j.get("usage"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="glm-5.1")
    ap.add_argument(
        "--n-per-dataset", type=int, default=5,
        help="Random samples to draw from each enabled dataset (cap at size).",
    )
    ap.add_argument(
        "--total", type=int, default=0,
        help="If >0, ignore --n-per-dataset and sample exactly this many "
             "prompts with replacement from the union of all datasets. Each "
             "row is tagged with `replicate_idx` for repeat tracking.",
    )
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--concurrency", type=int, default=1,
        help="In-flight requests. >1 lets vLLM continuous-batch.",
    )
    ap.add_argument(
        "--datasets", nargs="+", default=list(LOADERS),
        choices=list(LOADERS),
    )
    ap.add_argument(
        "--out",
        default="/mnt/remote/guangtaow/logs/eval_prompts_results.jsonl",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If set, load an existing --out file and skip items whose idx "
             "already has ok=True. Appends to the file instead of truncating.",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    if args.total > 0:
        # Pull all available samples from each dataset, then sample N with replacement.
        pool = []
        for d in args.datasets:
            pool.extend(LOADERS[d](10**9, args.seed))
        if not pool:
            sys.exit("No prompts loaded — all selected datasets failed.")
        rng = random.Random(args.seed)
        items = []
        for i in range(args.total):
            base = dict(rng.choice(pool))
            base["replicate_idx"] = i
            items.append(base)
        # Shuffling is not needed (already random with replacement) but harmless.
        n_unique = len({(it["source"], it["id"]) for it in items})
        print(
            f"Sampled {len(items)} prompts WITH REPLACEMENT from a pool of "
            f"{len(pool)} ({n_unique} unique appear in this run)."
        )
    else:
        items = []
        for d in args.datasets:
            items.extend(LOADERS[d](args.n_per_dataset, args.seed))
        if not items:
            sys.exit("No prompts loaded — all selected datasets failed.")
        random.shuffle(items)

    print(
        f"Sending {len(items)} prompts to {args.url} as model={args.model} "
        f"(max_tokens={args.max_tokens}, T={args.temperature}, "
        f"concurrency={args.concurrency})"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- resume: load already-completed successes --------------------------------
    done_idx: set[int] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("ok") and "idx" in rec:
                    done_idx.add(rec["idx"])
            except json.JSONDecodeError:
                pass
        print(
            f"[resume] Found {len(done_idx)} already-completed items in {out_path}; "
            "will skip them."
        )
        items = [(i, it) for i, it in enumerate(items) if i not in done_idx]
        if not items:
            print("All items already completed — nothing to do.")
            return
        print(f"[resume] {len(items)} items remaining to run.")
    else:
        out_path.write_text("")  # truncate
        items = list(enumerate(items))
    # -----------------------------------------------------------------------------

    n_ok = n_fail = 0
    write_lock = Lock()
    progress_lock = Lock()
    progress = {"done": 0, "ok": 0, "fail": 0}
    t_start = time.time()

    def run_one(idx_item):
        idx, item = idx_item
        res = send(
            args.url, args.model, item["prompt"],
            args.max_tokens, args.temperature, args.timeout,
        )
        row = {**item, **res, "idx": idx}
        with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with progress_lock:
            progress["done"] += 1
            done = progress["done"]
            if res["ok"]:
                progress["ok"] += 1
            else:
                progress["fail"] += 1
            elapsed = time.time() - t_start
            rate = done / max(elapsed, 1e-6)
            eta = (len(items) - done) / max(rate, 1e-6)
            tag = (
                f"ok {res['latency']:.1f}s "
                f"({(res.get('usage') or {}).get('completion_tokens', '?')} tok)"
                if res["ok"]
                else f"FAIL: {(res.get('error') or (res.get('body','') or ''))[:80]}"
            )
            print(
                f"[{done}/{len(items)}] {item['source']}/{item['id']} {tag}"
                f"  | rate={rate:.2f}/s eta={eta/60:.1f}m",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(run_one, id_item) for id_item in items]
        for _ in as_completed(futures):
            pass

    n_ok = progress["ok"]
    n_fail = progress["fail"]
    total_t = time.time() - t_start
    print(
        f"\nDone in {total_t/60:.1f} min: {n_ok} ok, {n_fail} failed. "
        f"Results: {out_path}"
    )


if __name__ == "__main__":
    main()
