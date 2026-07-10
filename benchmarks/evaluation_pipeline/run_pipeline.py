#!/usr/bin/env python3
"""Main driver for the evaluation pipeline: initialize -> evaluate -> present.

This pipeline measures throughput and speculative-decoding behavior only
-- no accuracy scoring. AIME/LiveCodeBench/GPQA Diamond prompts (see
datasets/prep_*.py) are used purely as a realistic, varied-length prompt
mix, not for grading.

IMPORTANT -- environment variables must be set BEFORE this script is
imported (before vllm is imported). Do NOT call this script directly.
Use run_experiments.sh, which sources gemma4_moe_benchmarks/.env_exports.sh
first. Mirrors the same convention as
../gemma4_moe_benchmarks/bench_experiment.py.

Usage (via run_experiments.sh):
    run_experiments.sh --all             # all speculative-decoding configs
    run_experiments.sh S000              # single config (baseline, no spec decode)
    run_experiments.sh S001,S003         # subset

Direct call (env vars must already be set):
    python3 run_pipeline.py --exp S003 --reps 2
    python3 run_pipeline.py --list
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hardware_metrics
from metrics import diff_spec_decode_counters, snapshot_spec_decode_counters

# ---------------------------------------------------------------------------
# Output paths (mirrors gemma4_moe_benchmarks/bench_experiment.py)
# ---------------------------------------------------------------------------
OUT_DIR = Path(os.environ.get("EVAL_RESULTS_DIR", "results"))
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "all_runs.csv"

CSV_FIELDS = [
    "ts", "exp_id", "label", "dataset",
    "spec_decode", "mtp_k",
    "rep", "seed", "num_prompts",
    "elapsed_time", "requests_per_second",
    "prompt_tokens_total", "output_tokens_total", "output_tps",
    "acceptance_rate", "mean_accept_length",
    "num_draft_tokens", "num_accepted_tokens", "num_drafts",
    "mfu", "mbu",
]

# ---------------------------------------------------------------------------
# Model paths -- same env vars as bench_experiment.py, sourced from
# gemma4_moe_benchmarks/.env_exports.sh via run_experiments.sh.
# ---------------------------------------------------------------------------
MODEL_BASE = os.environ.get("GEMMA4_MODEL_PATH", "google/gemma-4-26B-A4B-it")
MODEL_ASSISTANT = os.environ.get("GEMMA4_ASSISTANT_MODEL_PATH", MODEL_BASE + "-assistant")

# ---------------------------------------------------------------------------
# Datasets -- prepared by datasets/prep_*.py. max_tokens is per-dataset
# since prompt/response shape differs a lot (AIME needs long reasoning
# chains, GPQA is short multiple-choice, LiveCodeBench needs a full code
# solution).
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "aime": dict(path="datasets/aime_samples.jsonl", max_tokens=8192),
    "gpqa_diamond": dict(path="datasets/gpqa_diamond_samples.jsonl", max_tokens=1024),
    "livecodebench": dict(path="datasets/livecodebench_samples.jsonl", max_tokens=4096),
}

# ---------------------------------------------------------------------------
# Speculative-decoding sweep. spec_decode=False -> no draft model at all;
# spec_decode=True -> MODEL_ASSISTANT as draft model with the given k.
# ---------------------------------------------------------------------------
EXPERIMENTS: dict[str, dict] = {
    "S000": dict(label="No speculative decoding (baseline)", spec_decode=False, mtp_k=0),
    "S001": dict(label="MTP k=1", spec_decode=True, mtp_k=1),
    "S002": dict(label="MTP k=2", spec_decode=True, mtp_k=2),
    "S003": dict(label="MTP k=3", spec_decode=True, mtp_k=3),
    "S004": dict(label="MTP k=4", spec_decode=True, mtp_k=4),
    "S005": dict(label="MTP k=5", spec_decode=True, mtp_k=5),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_csv_header():
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(row: dict):
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def load_prompts(dataset_path: str) -> list[str]:
    """Reads a datasets/*.jsonl file (written by prep_*.py) and returns
    the raw "prompt" field of every row. Mirrors
    gemma4_moe_benchmarks/bench_experiment.py's load_prompts(), minus the
    count cap -- these datasets are already sized at prep time."""
    prompts: list[str] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["prompt"])
    if not prompts:
        raise FileNotFoundError(
            f"Dataset empty or not found: {dataset_path}\n"
            "Run the matching datasets/prep_*.py first."
        )
    return prompts


def render_chat(tok, raw_prompts: list[str]) -> list[str]:
    """Same convention as bench_experiment.py: fold each raw prompt into
    a single user-role chat turn."""
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in raw_prompts
    ]


# ---------------------------------------------------------------------------
# Stage 1: initialize
# ---------------------------------------------------------------------------

def initialize_engine(exp_cfg: dict):
    """Builds one vllm.LLM for an experiment config. Reuses the
    llm_kwargs / spec_model / spec_tokens pattern from
    gemma4_moe_benchmarks/bench_experiment.py:498-517. Reused across all
    three datasets for this experiment (same model/spec config, only the
    prompts change) -- rebuilding per-dataset would be wasteful."""
    from vllm import LLM

    llm_kwargs: dict = dict(
        model=MODEL_BASE,
        trust_remote_code=True,
        seed=0,
        # LLM.__init__ defaults disable_log_stats=True (see
        # vllm/entrypoints/llm.py:253-254) unless explicitly overridden,
        # which makes llm.get_metrics() raise "Stat logging disabled" --
        # metrics.py needs this on to read spec-decode counters.
        disable_log_stats=False,
    )
    if exp_cfg["spec_decode"]:
        llm_kwargs["spec_model"] = MODEL_ASSISTANT
        llm_kwargs["spec_tokens"] = exp_cfg["mtp_k"]

    print(f"[run_pipeline] building LLM: {llm_kwargs}", flush=True)
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[run_pipeline] engine built in {time.time() - t0:.1f}s", flush=True)
    return llm


# ---------------------------------------------------------------------------
# Stage 2: evaluate
# ---------------------------------------------------------------------------

def evaluate(
    llm,
    tok,
    dataset_name: str,
    ds_cfg: dict,
    exp_id: str,
    exp_cfg: dict,
    rep: int,
    hf_config,
    active_params: dict[str, int],
    gpu_specs: dict[str, float] | None,
) -> dict:
    """Runs one (experiment x dataset x rep) generation pass: loads
    prompts, renders the chat template, calls llm.generate(), times it,
    and diffs spec-decode counters before/after. No scoring step -- see
    module docstring.

    hf_config/active_params/gpu_specs are computed once per experiment
    (see run_one_experiment) and passed in rather than recomputed here --
    they don't depend on the dataset or rep, only on the model/hardware."""
    from vllm import SamplingParams

    raw_prompts = load_prompts(ds_cfg["path"])
    prompts = render_chat(tok, raw_prompts)

    seed = rep
    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=ds_cfg["max_tokens"],
        seed=seed,
    )

    tag = f"{exp_id}_{dataset_name}_rep{rep}"
    print(f"\n--- RUN {tag} seed={seed} num_prompts={len(prompts)} ---", flush=True)

    before = snapshot_spec_decode_counters(llm)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    elapsed = time.time() - t0
    after = snapshot_spec_decode_counters(llm)
    spec = diff_spec_decode_counters(before, after)

    prompt_total = sum(len(o.prompt_token_ids) for o in outputs)
    output_total = sum(len(o.outputs[0].token_ids) for o in outputs)

    # MFU uses total tokens (prefill + decode both consume FLOPs); MBU
    # uses output tokens only (it's specifically the decode-phase,
    # memory-bound metric -- see hardware_metrics.py module docstring).
    # gpu_specs is None on unrecognized hardware -- report n/a rather
    # than a number computed against a guessed peak spec.
    mfu = mbu = None
    if gpu_specs is not None and elapsed > 0:
        avg_context_len = (prompt_total + output_total) / max(len(prompts), 1)
        flops_per_tok = hardware_metrics.flops_per_token(
            active_params, avg_context_len, hf_config
        )
        bytes_per_tok = hardware_metrics.bytes_per_token(active_params)
        total_tokens = prompt_total + output_total
        mfu = hardware_metrics.compute_mfu(
            total_tokens / elapsed, flops_per_tok, gpu_specs["peak_tflops_bf16"]
        )
        # NOT output_total / elapsed. Weights are read from HBM once per
        # scheduler step, and one step produces one token for every
        # concurrently-running request -- so "bytes moved" scales with
        # decode *steps*, not with raw output-token count. Dividing by
        # num_prompts (an approximation for concurrent batch size, since
        # all prompts are submitted in a single generate() call) converts
        # aggregate tokens/sec back into steps/sec. Using output_total/elapsed
        # directly effectively counts one full weight-read per request per
        # token instead of one per step -- that's what was producing
        # MBU >> 100% (physically impossible; you cannot sustain more than
        # the hardware's rated peak bandwidth).
        #
        # Under speculative decoding this needs a second correction: one
        # target-model verification pass (one weight-read) doesn't yield
        # one accepted token per sequence, it yields mean_accept_length of
        # them on average (that value already includes the guaranteed
        # "bonus" token every round produces regardless of draft
        # acceptance -- see metrics.py). Dividing by num_prompts alone
        # still counts every accepted token as its own weight-read, which
        # overcounts verification passes by mean_accept_length -- the same
        # category of bug as the original one, just a smaller multiplier
        # (mean_accept_length instead of concurrent batch size). When
        # spec_decode is off there's no drafting at all, so 1 token really
        # is 1 step -- mean_accept_length is defined as 1 in that case.
        accept_len_divisor = spec["mean_accept_length"] if spec else 1.0
        decode_steps_per_second = (
            output_total / max(len(prompts), 1) / accept_len_divisor
        ) / elapsed
        mbu = hardware_metrics.compute_mbu(
            decode_steps_per_second, bytes_per_tok, gpu_specs["peak_bandwidth_gbps"]
        )

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "exp_id": exp_id,
        "label": exp_cfg["label"],
        "dataset": dataset_name,
        "spec_decode": exp_cfg["spec_decode"],
        "mtp_k": exp_cfg["mtp_k"],
        "rep": rep,
        "seed": seed,
        "num_prompts": len(prompts),
        "elapsed_time": round(elapsed, 3),
        "requests_per_second": round(len(prompts) / elapsed, 4) if elapsed > 0 else 0.0,
        "prompt_tokens_total": prompt_total,
        "output_tokens_total": output_total,
        "output_tps": round(output_total / elapsed, 2) if elapsed > 0 else 0.0,
        "acceptance_rate": round(spec["acceptance_rate"], 4) if spec else None,
        "mean_accept_length": round(spec["mean_accept_length"], 3) if spec else None,
        "num_draft_tokens": spec["num_draft_tokens"] if spec else None,
        "num_accepted_tokens": spec["num_accepted_tokens"] if spec else None,
        "num_drafts": spec["num_drafts"] if spec else None,
        "mfu": round(mfu, 4) if mfu is not None else None,
        "mbu": round(mbu, 4) if mbu is not None else None,
    }

    acc_str = f"{row['acceptance_rate']:.3f}" if row["acceptance_rate"] is not None else "n/a"
    mfu_str = f"{row['mfu']:.3f}" if row["mfu"] is not None else "n/a (unrecognized GPU)"
    mbu_str = f"{row['mbu']:.3f}" if row["mbu"] is not None else "n/a (unrecognized GPU)"
    print(
        f"  elapsed={elapsed:.1f}s  req/s={row['requests_per_second']:.3f}  "
        f"out_tok/s={row['output_tps']:.0f}  acceptance_rate={acc_str}  "
        f"MFU={mfu_str}  MBU={mbu_str}",
        flush=True,
    )
    return row


# ---------------------------------------------------------------------------
# Stage 3: present
# ---------------------------------------------------------------------------

def present(rows: list[dict]) -> None:
    """Writes one JSON file per (experiment x dataset x rep) plus an
    appended CSV row, mirroring bench_experiment.py's
    OUT_DIR/{tag}.json + all_runs.csv convention."""
    ensure_csv_header()
    for row in rows:
        append_csv_row(row)
        tag = f"{row['exp_id']}_{row['dataset']}_rep{row['rep']}"
        with (OUT_DIR / f"{tag}.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)


def summarize(exp_id: str, label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"[SUMMARY] {exp_id}: no successful runs", flush=True)
        return

    def m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None, None
        if len(vals) == 1:
            return vals[0], 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    by_dataset: dict[str, list[dict]] = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    print(f"\n[SUMMARY] {exp_id} | {label}", flush=True)
    for dataset, ds_rows in by_dataset.items():
        r_m, r_s = m("requests_per_second")
        r_m, r_s = (
            statistics.mean([x["requests_per_second"] for x in ds_rows]),
            statistics.stdev([x["requests_per_second"] for x in ds_rows])
            if len(ds_rows) > 1
            else 0.0,
        )
        acc_vals = [x["acceptance_rate"] for x in ds_rows if x["acceptance_rate"] is not None]
        acc_str = f"{statistics.mean(acc_vals):.3f}" if acc_vals else "n/a"
        mfu_vals = [x["mfu"] for x in ds_rows if x["mfu"] is not None]
        mbu_vals = [x["mbu"] for x in ds_rows if x["mbu"] is not None]
        mfu_str = f"{statistics.mean(mfu_vals):.3f}" if mfu_vals else "n/a"
        mbu_str = f"{statistics.mean(mbu_vals):.3f}" if mbu_vals else "n/a"
        print(
            f"  {dataset:<14} reps={len(ds_rows)}  "
            f"requests/sec={r_m:.4f}+/-{r_s:.4f}  acceptance_rate={acc_str}  "
            f"MFU={mfu_str}  MBU={mbu_str}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_one_experiment(exp_id: str, exp_cfg: dict, dataset_names: list[str], reps: int) -> list[dict]:
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)

    # Model-architecture and hardware facts are constant across every
    # dataset/rep in this experiment -- compute once, not per evaluate() call.
    hf_config = AutoConfig.from_pretrained(MODEL_BASE, trust_remote_code=True)
    active_params = hardware_metrics.compute_active_params(hf_config)
    gpu_specs = hardware_metrics.detect_gpu_specs()
    if gpu_specs is None:
        print(
            "[run_pipeline] WARNING: unrecognized GPU for MFU/MBU -- "
            "see hardware_metrics.GPU_SPECS. Reporting mfu/mbu as n/a.",
            flush=True,
        )
    else:
        print(
            f"[run_pipeline] active_params="
            f"{active_params['total_active_params']/1e9:.2f}B  "
            f"gpu_specs={gpu_specs}",
            flush=True,
        )

    llm = initialize_engine(exp_cfg)

    rows: list[dict] = []
    try:
        for dataset_name in dataset_names:
            ds_cfg = DATASETS[dataset_name]
            for rep in range(1, reps + 1):
                row = evaluate(
                    llm, tok, dataset_name, ds_cfg, exp_id, exp_cfg, rep,
                    hf_config, active_params, gpu_specs,
                )
                rows.append(row)
    finally:
        present(rows)
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    summarize(exp_id, exp_cfg["label"], rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one or more speculative-decoding throughput experiments."
    )
    ap.add_argument(
        "--exp",
        help="Experiment ID(s), comma-separated (e.g. S000 or S000,S003). "
        "Omit with --all to run every experiment.",
    )
    ap.add_argument("--all", action="store_true", help="Run every experiment in EXPERIMENTS")
    ap.add_argument(
        "--datasets",
        default=",".join(DATASETS.keys()),
        help=f"Comma-separated dataset subset (default: all of {list(DATASETS.keys())})",
    )
    ap.add_argument("--reps", type=int, default=2, help="Repetitions per (experiment, dataset)")
    ap.add_argument("--list", action="store_true", help="Print the experiment matrix and exit")
    args = ap.parse_args()

    if args.list:
        print(f"{'ID':<6}  {'label'}")
        print("-" * 60)
        for eid, ecfg in EXPERIMENTS.items():
            print(f"{eid:<6}  {ecfg['label']}")
        return 0

    if not args.exp and not args.all:
        print("ERROR: pass --exp <IDs> or --all", file=sys.stderr)
        return 1

    exp_ids = list(EXPERIMENTS.keys()) if args.all else [x.strip() for x in args.exp.split(",")]
    dataset_names = [x.strip() for x in args.datasets.split(",")]
    for d in dataset_names:
        if d not in DATASETS:
            print(f"ERROR: unknown dataset '{d}'. Valid: {list(DATASETS.keys())}", file=sys.stderr)
            return 1

    failed_exp_ids: list[str] = []
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(
                f"ERROR: unknown experiment ID '{exp_id}'. Valid: {list(EXPERIMENTS.keys())}",
                file=sys.stderr,
            )
            return 1
        exp_cfg = EXPERIMENTS[exp_id]

        try:
            run_one_experiment(exp_id, exp_cfg, dataset_names, args.reps)
        except Exception as e:
            print(f"!!! experiment {exp_id} FAILED: {e}", flush=True)
            import traceback

            traceback.print_exc()
            failed_exp_ids.append(exp_id)

    if failed_exp_ids:
        print(f"\nDone with failures. failed={','.join(failed_exp_ids)}", flush=True)
        return 1

    print("\nAll experiments completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
