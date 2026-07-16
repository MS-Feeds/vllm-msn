#!/usr/bin/env python3
"""Single experiment executor for autobench.

Runs one vLLM offline benchmark with a given config JSON.
Outputs results to local ./run_results/ and ./results.tsv only.
The calling job script handles copying results to mount (Cosmos).

IMPORTANT: Environment variables must be set BEFORE importing this script
(vLLM freezes env at import time). The AML job command handles this.

Usage:
    python run_experiment.py --config config.json
    python run_experiment.py --config config.json --dry-run
    python run_experiment.py --config config.json --prompts 200 --reps 2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config_space import (
    config_summary,
    config_to_env_vars,
    config_to_llm_kwargs,
    validate_config,
)

SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "datasets" / "sc1_delta_v2.jsonl"

SCENARIO = dict(
    output_len=8192,
)

MODEL_TEXT_ONLY = os.environ.get(
    "GEMMA4_TEXT_ONLY_MODEL_PATH", "google/gemma-4-27b-it-text-only"
)
MODEL_ASSISTANT = os.environ.get(
    "GEMMA4_ASSISTANT_MODEL_PATH", "google/gemma-4-27b-it-assistant"
)

RESULTS_TSV = SCRIPT_DIR / "results.tsv"
RESULTS_DIR = SCRIPT_DIR / "run_results"


def load_prompts(n: int) -> list[str]:
    prompts: list[str] = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts.append(d["prompt"])
            if len(prompts) >= n:
                break
    if not prompts:
        raise FileNotFoundError(f"Dataset empty or not found: {DATASET_PATH}")
    return prompts


def render_chat(tok, raw_prompts: list[str]) -> list[str]:
    out = []
    for p in raw_prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out.append(text)
    return out


def percentile(sorted_vals: list, q: float):
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def run_single(cfg: dict, num_prompts: int, reps: int) -> dict:
    """Run benchmark with the given config. Returns result dict."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model = cfg.get("model", MODEL_TEXT_ONLY)

    llm_kwargs = config_to_llm_kwargs(
        cfg, {"max_num_batched_tokens": cfg.get("max_num_batched_tokens", 16384)}
    )
    llm_kwargs["model"] = model

    if "speculative_config" in cfg:
        llm_kwargs["speculative_config"] = cfg["speculative_config"]
    else:
        spec_tokens = cfg.get("spec_tokens", 0)
        if spec_tokens > 0:
            llm_kwargs["spec_model"] = MODEL_ASSISTANT
            llm_kwargs["spec_tokens"] = spec_tokens

    summary = config_summary(cfg)
    print(f"\n{'='*60}", flush=True)
    print(f"  Config: {summary}", flush=True)
    print(f"  Model: {model}", flush=True)
    print(f"  Prompts: {num_prompts}, Reps: {reps}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Use local_files_only=True for absolute local paths (e.g. /tmp/models/...)
    # to avoid HFValidationError: Repo id must be in 'namespace/repo' form.
    _is_local = model.startswith("/") or (len(model) > 1 and model[1] == ":")
    tok = AutoTokenizer.from_pretrained(
        model, trust_remote_code=True, local_files_only=_is_local
    )
    raw_prompts = load_prompts(num_prompts)
    prompts = render_chat(tok, raw_prompts)
    print(f"Loaded {len(prompts)} prompts", flush=True)

    t_engine = time.time()
    llm = LLM(**llm_kwargs)
    engine_time = time.time() - t_engine
    print(f"Engine built in {engine_time:.1f}s", flush=True)

    all_output_tps = []
    all_total_tps = []
    best_row = None

    for rep in range(1, reps + 1):
        sampling = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=SCENARIO["output_len"],
            seed=rep,
            ignore_eos=False,
        )
        print(f"\n--- Rep {rep}/{reps} ---", flush=True)
        t0 = time.time()
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        elapsed = time.time() - t0

        prompt_total = 0
        output_total = 0
        out_lens: list[int] = []
        for o in outputs:
            prompt_total += len(o.prompt_token_ids)
            for comp in o.outputs:
                n_out = len(comp.token_ids)
                output_total += n_out
                out_lens.append(n_out)

        total = prompt_total + output_total
        output_tps = output_total / elapsed
        total_tps = total / elapsed
        prompt_tps = prompt_total / elapsed

        all_output_tps.append(output_tps)
        all_total_tps.append(total_tps)

        print(
            f"  elapsed={elapsed:.1f}s  output_tps={output_tps:.1f}  "
            f"total_tps={total_tps:.1f}  prompt_tps={prompt_tps:.1f}",
            flush=True,
        )

        if best_row is None or output_tps > best_row["output_tps"]:
            out_lens_sorted = sorted(out_lens)
            best_row = {
                "output_tps": round(output_tps, 2),
                "total_tps": round(total_tps, 2),
                "prompt_tps": round(prompt_tps, 2),
                "elapsed_time": round(elapsed, 2),
                "num_prompts": len(prompts),
                "prompt_tokens_total": prompt_total,
                "output_tokens_total": output_total,
                "out_len_mean": round(statistics.mean(out_lens), 1),
                "out_len_p50": int(percentile(out_lens_sorted, 0.5)),
                "out_len_p90": int(percentile(out_lens_sorted, 0.9)),
            }

    mean_output_tps = statistics.mean(all_output_tps)
    mean_total_tps = statistics.mean(all_total_tps)
    stdev_output_tps = statistics.stdev(all_output_tps) if len(all_output_tps) > 1 else 0.0

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    result = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "config_summary": summary,
        "mean_output_tps": round(mean_output_tps, 2),
        "mean_total_tps": round(mean_total_tps, 2),
        "stdev_output_tps": round(stdev_output_tps, 2),
        "reps": reps,
        "num_prompts": num_prompts,
        "engine_build_time": round(engine_time, 1),
        "best_rep": best_row,
        "status": "ok",
    }

    print(f"\n--- RESULT ---", flush=True)
    print(f"output_tps: {mean_output_tps:.2f} ± {stdev_output_tps:.2f}", flush=True)
    print(f"total_tps: {mean_total_tps:.2f}", flush=True)

    return result


def save_result(result: dict):
    """Save results to local files only. Job script copies to Cosmos mount."""
    RESULTS_DIR.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result["run_id"] = run_id

    result_file = RESULTS_DIR / f"{run_id}.json"
    with result_file.open("w") as f:
        json.dump(result, f, indent=2, default=str)

    header_needed = not RESULTS_TSV.exists()
    with RESULTS_TSV.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("run_id\toutput_tps\ttotal_tps\tstdev\tstatus\tconfig_summary\n")
        f.write(
            f"{run_id}\t"
            f"{result['mean_output_tps']}\t"
            f"{result['mean_total_tps']}\t"
            f"{result['stdev_output_tps']}\t"
            f"{result['status']}\t"
            f"{result['config_summary']}\n"
        )

    print(f"Results saved: {result_file}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Run single autobench experiment")
    ap.add_argument("--config", required=True, help="Path to config JSON file")
    ap.add_argument("--prompts", type=int, default=1000,
                    help="Number of prompts (default: 1000, matches sc1 scenario)")
    ap.add_argument("--reps", type=int, default=2, help="Repetitions (default: 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate config and print args without running")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    valid, err = validate_config(cfg)
    if not valid:
        print(f"ERROR: Invalid config: {err}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Config: {json.dumps(cfg, indent=2)}")
        print(f"Summary: {config_summary(cfg)}")
        print(f"Env vars: {json.dumps(config_to_env_vars(cfg), indent=2)}")
        llm_kw = config_to_llm_kwargs(cfg, {})
        print(f"LLM kwargs: {json.dumps(llm_kw, indent=2, default=str)}")
        print("Validation: PASSED")
        return 0

    try:
        result = run_single(cfg, num_prompts=args.prompts, reps=args.reps)
        save_result(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        crash_result = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "config": cfg,
            "config_summary": config_summary(cfg),
            "mean_output_tps": 0.0,
            "mean_total_tps": 0.0,
            "stdev_output_tps": 0.0,
            "reps": args.reps,
            "num_prompts": args.prompts,
            "status": "crash",
            "error": str(e),
        }
        save_result(crash_result)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
