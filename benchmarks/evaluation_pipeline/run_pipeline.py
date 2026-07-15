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
    run_experiments.sh --all                       # all spec-decode configs (suite=spec, default)
    run_experiments.sh S000                        # single config (baseline, no spec decode)
    run_experiments.sh S001,S003                   # subset
    run_experiments.sh --suite batch --all          # all max_num_seqs batch-size configs
    run_experiments.sh --suite batch B003           # single batch-size config
    run_experiments.sh --suite cross --all          # all mns x MTP-k cross configs
    run_experiments.sh --suite cross X003           # single cross config
    run_experiments.sh --suite cross_hi --all       # all high-range mns x MTP-k configs
    run_experiments.sh --suite cross_hi Y007        # single high-range cross config

Direct call (env vars must already be set):
    python3 run_pipeline.py --exp S003 --reps 2
    python3 run_pipeline.py --suite batch --exp B003 --reps 2
    python3 run_pipeline.py --suite cross --exp X003 --reps 2
    python3 run_pipeline.py --suite cross_hi --exp Y007 --reps 2
    python3 run_pipeline.py --list
    python3 run_pipeline.py --suite batch --list
    python3 run_pipeline.py --suite cross --list
    python3 run_pipeline.py --suite cross_hi --list

Four independent sweep suites share this driver (same initialize_engine/
evaluate/present/summarize -- see EXPERIMENT_PLAN.md,
EXPERIMENT_PLAN_MAX_NUM_SEQS.md, EXPERIMENT_PLAN_MNS_SPEC_CROSS.md,
EXPERIMENT_PLAN_MNS_SPEC_CROSS_HI.md):
  - "spec"  (default) -- SPEC_EXPERIMENTS (S0xx): spec-decode on/off, MTP
    draft length k, crossed with dataset. max_num_seqs held at
    DEFAULT_MAX_NUM_SEQS for every row.
  - "batch" -- BATCH_EXPERIMENTS (B0xx): max_num_seqs swept, spec decode
    held off for every row.
  - "cross" -- CROSS_EXPERIMENTS (X0xx): max_num_seqs in {128, 256} x MTP
    k in {1, 3, 5}, spec decode on for every row -- the batch x spec-decode
    combination the other two suites deliberately avoid crossing.
  - "cross_hi" -- CROSS_HI_EXPERIMENTS (Y0xx): same idea as "cross" but a
    higher range -- max_num_seqs in {128, 256, 512} x MTP k in {4, 6, 8}.
    The mns=512 rows use a dedicated larger LiveCodeBench sample (same one
    BATCH_EXPERIMENTS' B006 uses), since the default datasets are too small
    for mns=512 to actually bind.
  These are deliberately kept in separate dicts / selected via --suite
  rather than merged into one matrix, so --all can never silently cross
  suites in one run -- --suite picks exactly one.
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
from metrics import (
    decode_window_seconds,
    diff_spec_decode_counters,
    snapshot_spec_decode_counters,
)

# ---------------------------------------------------------------------------
# Output paths (mirrors gemma4_moe_benchmarks/bench_experiment.py)
# ---------------------------------------------------------------------------
OUT_DIR = Path(os.environ.get("EVAL_RESULTS_DIR", "results"))
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "all_runs.csv"

CSV_FIELDS = [
    "ts", "exp_id", "label", "dataset",
    "spec_decode", "mtp_k", "max_num_seqs",
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
    # Dedicated larger sample (not the 300-prompt "livecodebench" set
    # above) -- exists so max_num_seqs=512 (B006, BATCH_EXPERIMENTS below)
    # has enough submitted prompts for the cap to actually bind. All three
    # datasets above (198-300 prompts) are too small for that: batch_size
    # = min(len(prompts), max_num_seqs) would just clamp to len(prompts)
    # regardless of the configured cap, making a "mns=512" experiment
    # against them indistinguishable from an uncapped run. 554 prompts,
    # not the 600 originally targeted -- prep_livecodebench.py's whole
    # FALLBACK_FILES chain (test6/test5/test3/test2.jsonl) only had 554
    # unique valid rows total; still comfortably above 512, so mns=512
    # binds correctly.
    "livecodebench_mns512": dict(path="datasets/livecodebench_mns512_samples.jsonl", max_tokens=4096),
}

# ---------------------------------------------------------------------------
# max_num_seqs -- vLLM's own default (see vllm/config/scheduler.py's
# SchedulerConfig.DEFAULT_MAX_NUM_SEQS). Every experiment in both suites
# below is run with an EXPLICIT max_num_seqs (never left implicit), set to
# this constant unless an experiment overrides it, so the batch_size figure
# fed into hardware_metrics' MBU calculation (see evaluate() below) always
# matches what the engine actually ran with, rather than assuming it.
# ---------------------------------------------------------------------------
DEFAULT_MAX_NUM_SEQS = 128

DEFAULT_MAX_NUM_BATCHED_TOKENS = 16384

# ---------------------------------------------------------------------------
# Suite 1/2 -- SPEC_EXPERIMENTS ("spec", the default suite). Speculative-
# decoding sweep. spec_decode=False -> no draft model at all; spec_decode=True
# -> MODEL_ASSISTANT as draft model with the given k. max_num_seqs held at
# DEFAULT_MAX_NUM_SEQS for every row -- this suite doesn't vary it (see
# EXPERIMENT_PLAN_MAX_NUM_SEQS.md for why that's a separate suite instead of
# a crossed axis here).
#
# An experiment may optionally set "datasets" to a fixed list of dataset
# names -- this overrides whatever --datasets was passed on the CLI, for
# experiments that are only meaningful (or only worth the runtime) against
# one specific dataset. It may also set "include_in_all=False" to be
# skipped by --all -- run_one_experiment/main() below honor both.
# ---------------------------------------------------------------------------
SPEC_EXPERIMENTS: dict[str, dict] = {
    "S000": dict(label="No speculative decoding (baseline)", spec_decode=False, mtp_k=0),
    "S001": dict(label="MTP k=1", spec_decode=True, mtp_k=1),
    "S002": dict(label="MTP k=2", spec_decode=True, mtp_k=2),
    "S003": dict(label="MTP k=3", spec_decode=True, mtp_k=3),
    "S004": dict(label="MTP k=4", spec_decode=True, mtp_k=4),
    "S005": dict(label="MTP k=5", spec_decode=True, mtp_k=5),
    "S006": dict(
        label="MTP k=3, gpqa_diamond only (single-dataset smoke test)",
        spec_decode=True,
        mtp_k=3,
        datasets=["gpqa_diamond"],
        include_in_all=False,
    ),
}

# ---------------------------------------------------------------------------
# Suite 2/2 -- BATCH_EXPERIMENTS ("batch"). max_num_seqs sweep -- see
# EXPERIMENT_PLAN_MAX_NUM_SEQS.md. spec_decode held off for every row (kept
# out of this suite entirely to avoid crossing two axes at once); only
# max_num_seqs varies. B004 (mns=128) is the same value SPEC_EXPERIMENTS
# implicitly runs at (DEFAULT_MAX_NUM_SEQS) -- i.e. the true no-op control.
# ---------------------------------------------------------------------------
BATCH_EXPERIMENTS: dict[str, dict] = {
    "B001": dict(label="Batch size sweep: max_num_seqs=16", spec_decode=False, mtp_k=0, max_num_seqs=16),
    "B002": dict(label="Batch size sweep: max_num_seqs=32", spec_decode=False, mtp_k=0, max_num_seqs=32),
    "B003": dict(label="Batch size sweep: max_num_seqs=64", spec_decode=False, mtp_k=0, max_num_seqs=64),
    "B004": dict(
        label="Batch size sweep: max_num_seqs=128 (control, matches DEFAULT_MAX_NUM_SEQS)",
        spec_decode=False, mtp_k=0, max_num_seqs=128,
    ),
    "B005": dict(label="Batch size sweep: max_num_seqs=256", spec_decode=False, mtp_k=0, max_num_seqs=256),
    "B006": dict(
        label="Batch size sweep: max_num_seqs=512 (554-prompt LiveCodeBench, dedicated sample)",
        spec_decode=False,
        mtp_k=0,
        max_num_seqs=512,
        # Overrides --datasets: B001-B005's 198-300-prompt datasets can't
        # make mns=512 actually bind (see DATASETS' livecodebench_mns512
        # comment above) -- this experiment needs its own larger sample.
        datasets=["livecodebench_mns512"],
    ),
}

# ---------------------------------------------------------------------------
# Suite 3/3 -- CROSS_EXPERIMENTS ("cross"). The batch-size x spec-decode
# cross-sweep BATCH_EXPERIMENTS' own comment above flagged as deliberately
# deferred (crossing two axes at once there would have confounded
# BATCH_EXPERIMENTS' isolated max_num_seqs measurement). Full mns x k grid:
# max_num_seqs in {128, 256} x MTP k in {1, 3, 5} -- 6 experiments, spec
# decode on for every row (unlike BATCH_EXPERIMENTS, which holds it off).
#
# mns=256 against gpqa_diamond (198 prompts) doesn't bind -- same caveat as
# BATCH_EXPERIMENTS' B005 (see EXPERIMENT_PLAN_MAX_NUM_SEQS.md); left as-is
# here since 256 vs "uncapped" is still a meaningful comparison against the
# mns=128 rows for that one dataset, just not a true 256-concurrency test.
# ---------------------------------------------------------------------------
CROSS_EXPERIMENTS: dict[str, dict] = {
    "X001": dict(label="mns=128 x MTP k=1", spec_decode=True, mtp_k=1, max_num_seqs=128),
    "X002": dict(label="mns=128 x MTP k=3", spec_decode=True, mtp_k=3, max_num_seqs=128),
    "X003": dict(label="mns=128 x MTP k=5", spec_decode=True, mtp_k=5, max_num_seqs=128),
    "X004": dict(label="mns=256 x MTP k=1", spec_decode=True, mtp_k=1, max_num_seqs=256),
    "X005": dict(label="mns=256 x MTP k=3", spec_decode=True, mtp_k=3, max_num_seqs=256),
    "X006": dict(label="mns=256 x MTP k=5", spec_decode=True, mtp_k=5, max_num_seqs=256),
}

# ---------------------------------------------------------------------------
# Suite 4/4 -- CROSS_HI_EXPERIMENTS ("cross_hi"). A separate suite (not more
# rows appended to CROSS_EXPERIMENTS) covering a higher mns x k range: mns in
# {128, 256, 512} x MTP k in {4, 6, 8}, spec decode on for every row -- see
# EXPERIMENT_PLAN_MNS_SPEC_CROSS_HI.md.
#
# mns=512 needs its own dataset override, same reasoning as BATCH_EXPERIMENTS'
# B006: the default 3 datasets (198-300 prompts) can't make mns=512 actually
# bind (batch_size = min(len(prompts), max_num_seqs) would just clamp to
# len(prompts)), so Y007-Y009 use the dedicated 554-prompt
# "livecodebench_mns512" sample instead of --datasets' default 3.
#
# mns=256 against gpqa_diamond (198 prompts) doesn't bind either, for
# Y004-Y006 -- same caveat as CROSS_EXPERIMENTS' X004-X006 and
# BATCH_EXPERIMENTS' B005.
# ---------------------------------------------------------------------------
CROSS_HI_EXPERIMENTS: dict[str, dict] = {
    "Y001": dict(label="mns=128 x MTP k=4", spec_decode=True, mtp_k=4, max_num_seqs=128),
    "Y002": dict(label="mns=128 x MTP k=6", spec_decode=True, mtp_k=6, max_num_seqs=128),
    "Y003": dict(label="mns=128 x MTP k=8", spec_decode=True, mtp_k=8, max_num_seqs=128),
    "Y004": dict(label="mns=256 x MTP k=4", spec_decode=True, mtp_k=4, max_num_seqs=256),
    "Y005": dict(label="mns=256 x MTP k=6", spec_decode=True, mtp_k=6, max_num_seqs=256),
    "Y006": dict(label="mns=256 x MTP k=8", spec_decode=True, mtp_k=8, max_num_seqs=256),
    "Y007": dict(
        label="mns=512 x MTP k=4 (554-prompt LiveCodeBench, dedicated sample)",
        spec_decode=True, mtp_k=4, max_num_seqs=512,
        datasets=["livecodebench_mns512"],
    ),
    "Y008": dict(
        label="mns=512 x MTP k=6 (554-prompt LiveCodeBench, dedicated sample)",
        spec_decode=True, mtp_k=6, max_num_seqs=512,
        datasets=["livecodebench_mns512"],
    ),
    "Y009": dict(
        label="mns=512 x MTP k=8 (554-prompt LiveCodeBench, dedicated sample)",
        spec_decode=True, mtp_k=8, max_num_seqs=512,
        datasets=["livecodebench_mns512"],
    ),
}

# Suite name -> matrix, for --suite dispatch in main(). Suites are mutually
# exclusive by design -- --suite selects exactly one dict, so --all (or an
# --exp list) can never silently mix experiments from more than one suite
# in the same run.
SUITES: dict[str, dict[str, dict]] = {
    "spec": SPEC_EXPERIMENTS,
    "batch": BATCH_EXPERIMENTS,
    "cross": CROSS_EXPERIMENTS,
    "cross_hi": CROSS_HI_EXPERIMENTS,
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
        # Always explicit, never left to vLLM's own default -- evaluate()'s
        # batch_size clamp (see below) needs to know exactly what the engine
        # ran with, not assume it matches vllm's internal default forever.
        max_num_seqs=exp_cfg.get("max_num_seqs", DEFAULT_MAX_NUM_SEQS),
        max_num_batched_tokens=exp_cfg.get("max_num_batched_tokens", DEFAULT_MAX_NUM_BATCHED_TOKENS),
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
    draft_hf_config=None,
    draft_active_params: dict[str, int] | None = None,
) -> dict:
    """Runs one (experiment x dataset x rep) generation pass: loads
    prompts, renders the chat template, calls llm.generate(), times it,
    and diffs spec-decode counters before/after. No scoring step -- see
    module docstring.

    hf_config/active_params/gpu_specs are computed once per experiment
    (see run_one_experiment) and passed in rather than recomputed here --
    they don't depend on the dataset or rep, only on the model/hardware.
    draft_hf_config/draft_active_params are None when exp_cfg["spec_decode"]
    is False (no draft model at all); when set, they feed the
    *_spec_decode hardware_metrics functions so MFU/MBU include the
    draft model's own FLOPs/bytes -- see hardware_metrics.py's DRAFT
    MODEL section."""
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

    # MFU covers prefill + decode (both consume FLOPs); MBU uses output
    # tokens only (it's specifically the decode-phase, memory-bound
    # metric -- see hardware_metrics.py module docstring). gpu_specs is
    # None on unrecognized hardware -- report n/a rather than a number
    # computed against a guessed peak spec.
    # Hoisted above the gpu_specs branch: needed for the row dict
    # unconditionally (recorded even when mfu/mbu are n/a on unrecognized
    # hardware), not just for the batch_size clamp inside it.
    max_num_seqs = exp_cfg.get("max_num_seqs", DEFAULT_MAX_NUM_SEQS)

    mfu = mbu = None
    if gpu_specs is not None and elapsed > 0:
        # Midpoint, not endpoint: context grows from prompt_len up to
        # prompt_len+output_len over the course of decoding, so the
        # STEP-AVERAGED context length (what actually matters -- FLOPs/
        # KV-bytes are per-step quantities, summed/averaged over every
        # step, not just the last one) is close to prompt_len +
        # output_len/2 for roughly linear growth, not the final length.
        # Using the final length here overestimated both the attention
        # FLOPs term and (once KV cache bytes were added) the KV-bytes
        # term by close to 2x for long generations. This holds under
        # speculative decoding too: decode_steps_per_second below already
        # converts to rounds/sec via mean_accept_length, so as long as
        # acceptance behavior is roughly stationary across the sequence,
        # the round-averaged cache length is still ~output_len/2 past the
        # prompt, just reached via fewer/larger jumps instead of many +1
        # steps -- there's no per-round accept-length data available here
        # to correct for non-stationary acceptance (e.g. degrading at
        # longer contexts), so this remains a first-order approximation.
        avg_context_len = (prompt_total + output_total / 2) / max(len(prompts), 1)
        # NOT max(len(prompts), 1) alone -- that's the number of prompts
        # SUBMITTED to this generate() call, not the number actually
        # resident and decoding at once. Once max_num_seqs caps concurrency
        # below len(prompts) (true today for the default mns=128 against
        # this pipeline's 198-300-prompt datasets, and the whole point of
        # the max_num_seqs sweep in EXPERIMENT_PLAN_MAX_NUM_SEQS.md), the
        # KV-cache byte term below (which batch_size scales -- see
        # hardware_metrics.bytes_per_decode_step's docstring) would be
        # overstated by using the submitted count instead of the cap. This
        # clamp is a static approximation, not a real measurement -- actual
        # concurrency ramps up at the start of a run and drains unevenly at
        # the end as shorter responses finish before longer ones, so this
        # slightly overstates the true time-averaged concurrency too, just
        # less than using len(prompts) unclamped would.
        batch_size = max(min(len(prompts), max_num_seqs), 1)
        mtp_k = exp_cfg["mtp_k"] if exp_cfg["spec_decode"] else 0
        # Needed before flops_per_tok now too (not just decode_steps_per_second
        # below): the draft model's per-round FLOPs/bytes contribution must be
        # converted from "per verification round" into "per accepted output
        # token" units the same way, since flops_per_token_spec_decode (like
        # bytes_per_decode_step_spec_decode) returns a per-round figure -- see
        # hardware_metrics.py's flops_per_token_spec_decode docstring.
        accept_len_divisor = spec["mean_accept_length"] if spec else 1.0
        decode_flops_per_round = hardware_metrics.flops_per_token_spec_decode(
            active_params, avg_context_len, hf_config, mtp_k,
            draft_hf_config, draft_active_params,
        )
        decode_flops_per_tok = decode_flops_per_round / accept_len_divisor
        bytes_per_step = hardware_metrics.bytes_per_decode_step_spec_decode(
            active_params, avg_context_len, hf_config, batch_size, mtp_k,
            draft_hf_config, draft_active_params,
        )

        # MFU is computed as two SEPARATE achieved-FLOPs terms, summed --
        # NOT one blended (prompt_total+output_total)/elapsed *
        # flops_per_tok multiplication. decode_flops_per_tok is a
        # decode-round-derived, per-ACCEPTED-token quantity: it's already
        # divided by mean_accept_length and (under spec decode) already
        # includes the draft model's FLOPs. Multiplying that by a token
        # rate that also includes PREFILL tokens would silently
        # attribute both of those decode-only corrections to prefill
        # tokens, which never go through a verification round or draft
        # model at all -- prefill tokens should be charged one full,
        # undivided target forward pass each, nothing else. (Confirmed by
        # hand-worked example: this bug OVERSTATES MFU under spec decode --
        # decode_flops_per_tok's numerator scales by (mtp_k+1), the
        # target's verification pass processing mtp_k+1 positions at
        # once, but is only divided by mean_accept_length, which is
        # always <= mtp_k+1 since not every candidate gets accepted, so
        # decode_flops_per_tok is systematically >= a prefill token's
        # true cost -- blending it onto prefill inflates the total.)
        #
        # prefill_avg_context_len approximates each prefill token's own
        # (much smaller, growing-within-the-prompt) attention context: a
        # length-L prompt processed in one causal forward pass has tokens
        # attending to 1..L positions, averaging L/2 -- NOT the decode
        # side's prompt_len+output_len/2 (which assumes the full prompt
        # is already cached, true only once decoding has started).
        #
        # NOT batch_size -- this is a PER-REQUEST average (mirrors
        # avg_context_len above, which divides by len(prompts) for the
        # same reason: a prompt's own attention context depends only on
        # its own length, not on how many other requests happen to be
        # running concurrently). batch_size is a concurrency figure
        # (clamped by max_num_seqs) and using it here was silently
        # correct only by coincidence before the max_num_seqs clamp was
        # introduced, back when batch_size always equaled len(prompts);
        # once max_num_seqs < len(prompts) (exactly what the batch-size
        # sweep tests), dividing by batch_size instead of len(prompts)
        # overstates this by a factor of len(prompts)/batch_size.
        prefill_avg_context_len = (prompt_total / max(len(prompts), 1)) / 2
        prefill_flops_per_tok = hardware_metrics.flops_per_token(
            active_params, prefill_avg_context_len, hf_config
        )
        mfu = hardware_metrics.compute_mfu(
            prompt_total / elapsed, prefill_flops_per_tok, gpu_specs["peak_tflops_bf16"]
        ) + hardware_metrics.compute_mfu(
            output_total / elapsed, decode_flops_per_tok, gpu_specs["peak_tflops_bf16"]
        )
        # NOT output_total / elapsed. Weights are read from HBM once per
        # scheduler step, and one step produces one token for every
        # concurrently-running request -- so weight bytes scale with
        # decode *steps*, not with raw output-token count (KV cache bytes
        # are different -- see bytes_per_decode_step, which already folds
        # in the batch_size scaling for the KV term above; this
        # decode_steps_per_second is purely the step-rate half of the
        # equation, feeding compute_mbu's tokens_per_second arg which it
        # multiplies by bytes_per_step -- passing raw aggregate tokens/sec
        # there instead, or the same value used for MFU's aggregate
        # throughput, effectively counts one full weight-read per request
        # per token instead of one per step). Dividing by num_prompts (an
        # approximation for concurrent batch size, since all prompts are
        # submitted in a single generate() call) converts aggregate
        # tokens/sec back into steps/sec. Using output_total/elapsed
        # directly instead was what was producing MBU >> 100% (physically
        # impossible; you cannot sustain more than the hardware's rated
        # peak bandwidth).
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
        #
        # The time denominator ALSO must not be the whole generate() call's
        # `elapsed` -- that includes prefill, which is compute-bound and
        # produces zero decode steps, so it dilutes decode_steps_per_second
        # (and therefore MBU) below what was actually achieved during
        # decode. decode_window_seconds() derives the decode-only wall time
        # from each request's own first/last-token timestamps instead (see
        # metrics.py) -- None (log_stats disabled, or every generation was
        # a single bonus token) means MBU can't be computed, not that it's
        # zero. accept_len_divisor was already computed above (also needed
        # by decode_flops_per_tok now).
        decode_elapsed = decode_window_seconds(outputs)
        if decode_elapsed is not None:
            decode_steps_per_second = (
                output_total / batch_size / accept_len_divisor
            ) / decode_elapsed
            mbu = hardware_metrics.compute_mbu(
                decode_steps_per_second, bytes_per_step, gpu_specs["peak_bandwidth_gbps"]
            )

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "exp_id": exp_id,
        "label": exp_cfg["label"],
        "dataset": dataset_name,
        "spec_decode": exp_cfg["spec_decode"],
        "mtp_k": exp_cfg["mtp_k"],
        "max_num_seqs": max_num_seqs,
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

def run_one_experiment(
    exp_id: str, exp_cfg: dict, dataset_names: list[str], reps: int, nsight: bool = False,
) -> list[dict]:
    from transformers import AutoConfig, AutoTokenizer

    # An experiment's own "datasets" (if set) overrides whatever
    # --datasets was passed on the CLI -- see EXPERIMENTS above.
    dataset_names = exp_cfg.get("datasets", dataset_names)

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

    # Draft ("assistant") model architecture facts, only needed when this
    # experiment actually runs speculative decoding -- feeds the
    # *_spec_decode hardware_metrics functions so MFU/MBU include the
    # draft model's own FLOPs/bytes instead of silently staying
    # target-only (see hardware_metrics.py's DRAFT MODEL section).
    draft_hf_config = None
    draft_active_params = None
    if exp_cfg["spec_decode"]:
        draft_hf_config = AutoConfig.from_pretrained(MODEL_ASSISTANT, trust_remote_code=True)
        draft_active_params = hardware_metrics.compute_draft_active_params(draft_hf_config)
        if gpu_specs is not None:
            print(
                f"[run_pipeline] draft_active_params="
                f"{draft_active_params['total_active_params']/1e9:.3f}B "
                f"(per forward pass, x{exp_cfg['mtp_k']} passes/round)",
                flush=True,
            )

    llm = initialize_engine(exp_cfg)

    # nsight=True brackets ONLY the generate() loop below (not engine
    # init -- weight loading/compilation warmup is one-time overhead
    # that would dominate and isn't representative of steady-state
    # throughput) with cudaProfilerStart/Stop, so that when this whole
    # process is launched under `nsys profile
    # --capture-range=cudaProfilerApi ...`, the capture window is scoped
    # to exactly this one experiment's requests instead of the entire
    # multi-experiment run. This call alone does NOT start Nsight itself
    # -- see EXPERIMENT_PLAN_MNS_SPEC_CROSS.md's Nsight section (or
    # REPRODUCE.md) for the required external nsys/ncu invocation and
    # setup. The NVTX range is a second, complementary marker that still
    # works even in a plain full-trace nsys run (no --capture-range),
    # since it lets the exp_id be located/filtered in the timeline after
    # the fact without needing the cudaProfilerApi start/stop to have
    # been honored.
    nvtx_pushed = False
    if nsight:
        try:
            import torch

            torch.cuda.nvtx.range_push(f"nsight_{exp_id}")
            nvtx_pushed = True
            torch.cuda.cudart().cudaProfilerStart()
            print(f"[run_pipeline] Nsight capture window opened for {exp_id}", flush=True)
        except Exception as e:
            print(f"[run_pipeline] WARNING: could not start CUDA profiler for Nsight: {e}", flush=True)

    rows: list[dict] = []
    try:
        for dataset_name in dataset_names:
            ds_cfg = DATASETS[dataset_name]
            for rep in range(1, reps + 1):
                row = evaluate(
                    llm, tok, dataset_name, ds_cfg, exp_id, exp_cfg, rep,
                    hf_config, active_params, gpu_specs,
                    draft_hf_config, draft_active_params,
                )
                rows.append(row)
    finally:
        if nsight:
            try:
                import torch

                torch.cuda.cudart().cudaProfilerStop()
                print(f"[run_pipeline] Nsight capture window closed for {exp_id}", flush=True)
            except Exception as e:
                print(f"[run_pipeline] WARNING: could not stop CUDA profiler for Nsight: {e}", flush=True)
            if nvtx_pushed:
                torch.cuda.nvtx.range_pop()

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
        description="Run one or more throughput experiments (spec-decode or batch-size suite)."
    )
    ap.add_argument(
        "--suite",
        choices=sorted(SUITES.keys()),
        default="spec",
        help="Which experiment matrix to select IDs from: 'spec' (SPEC_EXPERIMENTS, "
        "S0xx -- spec-decode/k sweep, default), 'batch' (BATCH_EXPERIMENTS, B0xx -- "
        "max_num_seqs sweep, spec decode held off), 'cross' (CROSS_EXPERIMENTS, "
        "X0xx -- max_num_seqs {128,256} x MTP k {1,3,5} grid), or 'cross_hi' "
        "(CROSS_HI_EXPERIMENTS, Y0xx -- max_num_seqs {128,256,512} x MTP k {4,6,8} "
        "grid). Suites are never merged into one --all so a single run can't "
        "silently cross suites -- pick one per invocation. See EXPERIMENT_PLAN.md, "
        "EXPERIMENT_PLAN_MAX_NUM_SEQS.md, EXPERIMENT_PLAN_MNS_SPEC_CROSS.md, "
        "EXPERIMENT_PLAN_MNS_SPEC_CROSS_HI.md.",
    )
    ap.add_argument(
        "--exp",
        help="Experiment ID(s) from the selected --suite, comma-separated "
        "(e.g. S000 or S000,S003; with --suite batch, B001 or B001,B003; "
        "with --suite cross, X001 or X001,X004; with --suite cross_hi, "
        "Y001 or Y001,Y007). "
        "Omit with --all to run every experiment in the suite.",
    )
    ap.add_argument("--all", action="store_true", help="Run every experiment in the selected --suite")
    ap.add_argument(
        "--datasets",
        default=",".join(DATASETS.keys()),
        help=f"Comma-separated dataset subset (default: all of {list(DATASETS.keys())})",
    )
    ap.add_argument("--reps", type=int, default=2, help="Repetitions per (experiment, dataset)")
    ap.add_argument("--list", action="store_true", help="Print the selected suite's experiment matrix and exit")
    ap.add_argument(
        "--nsight-exp",
        default=None,
        metavar="EXP_ID",
        help="Bracket ONLY this experiment ID's generate() calls (not engine init) with "
        "cudaProfilerStart/Stop + an NVTX range, so an external `nsys profile "
        "--capture-range=cudaProfilerApi ...` (or ncu) wrapping this whole process "
        "captures just that one experiment instead of the entire --exp/--all run. "
        "This flag alone does NOT launch Nsight -- see REPRODUCE.md's Nsight section "
        "for the required external command and setup. Must be one of the IDs actually "
        "being run this invocation.",
    )
    args = ap.parse_args()

    experiments = SUITES[args.suite]

    if args.list:
        print(f"suite={args.suite}")
        print(f"{'ID':<6}  {'label'}")
        print("-" * 60)
        for eid, ecfg in experiments.items():
            print(f"{eid:<6}  {ecfg['label']}")
        return 0

    if not args.exp and not args.all:
        print("ERROR: pass --exp <IDs> or --all", file=sys.stderr)
        return 1

    exp_ids = (
        [eid for eid, ecfg in experiments.items() if ecfg.get("include_in_all", True)]
        if args.all
        else [x.strip() for x in args.exp.split(",")]
    )
    dataset_names = [x.strip() for x in args.datasets.split(",")]
    for d in dataset_names:
        if d not in DATASETS:
            print(f"ERROR: unknown dataset '{d}'. Valid: {list(DATASETS.keys())}", file=sys.stderr)
            return 1

    if args.nsight_exp is not None and args.nsight_exp not in exp_ids:
        # Fail fast rather than silently profiling nothing -- a typo'd or
        # stale --nsight-exp value would otherwise run the whole batch
        # normally with no indication the capture window never opened.
        print(
            f"ERROR: --nsight-exp '{args.nsight_exp}' is not among the experiment(s) "
            f"being run this invocation ({', '.join(exp_ids)}).",
            file=sys.stderr,
        )
        return 1

    failed_exp_ids: list[str] = []
    for exp_id in exp_ids:
        if exp_id not in experiments:
            print(
                f"ERROR: unknown experiment ID '{exp_id}' for suite '{args.suite}'. "
                f"Valid: {list(experiments.keys())}",
                file=sys.stderr,
            )
            return 1
        exp_cfg = experiments[exp_id]

        try:
            run_one_experiment(
                exp_id, exp_cfg, dataset_names, args.reps, nsight=(exp_id == args.nsight_exp),
            )
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
