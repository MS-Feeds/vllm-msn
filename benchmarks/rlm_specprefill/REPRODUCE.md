# Reproduction Steps

Status: all 11 build-order steps in `IMPLEMENTATION_PLAN.md` are done (env
setup, `eval_data/` prep, evidence-extraction, the evidence cache, timing
decomposition, `target_stage/`, `runner/`, `calibration/`, and `analysis/`)
— **85/85 unit tests pass**, plus real Anthropic API smoke tests for the
RLM-evidence half. **Nothing in this project has been run against a GPU or
vLLM** — it was built on a Windows machine with no GPU and `vllm` not
installed there. Steps 0-4 below are runnable on any machine with Anthropic
API access; steps 5 onward are GPU-node-only and have never been executed.
See `IMPLEMENTATION_PLAN.md`'s Verification section for the exact list of
what's been checked vs. not, and its load-bearing findings for real issues
already discovered along the way: two bugs in dependencies this project
builds on (`AnthropicClient` not forwarding `sampling_args`, and RLM's
max-iterations fallback path crashing on an empty-content API response —
both handled, not blockers), plus a real instrumentation gap this project
had to fix itself (the offline-batched vLLM driving pattern doesn't expose
true per-request latency for free — needed for `f` to be a real per-sample
number, not an estimate).

## 0. Environment

Two independent stacks, both needed:

**Root model (Claude via Anthropic API)** — just needs `anthropic` +
`python-dotenv` (already a dependency of `../rlm/pyproject.toml`) and a real
`ANTHROPIC_API_KEY` in `../rlm/.env`.

**Target/speculator (self-hosted Llama via this fork's vLLM)** — same
environment as `../spec_prefill_llama/REPRODUCE.md` step 1 (conda env,
`VLLM_USE_PRECOMPILED=1 pip install -e .`, `pytest`, `datasets`,
`transformers`). Do not duplicate that setup here; follow that file's step 1
verbatim on the GPU node, then:

```bash
export HF_TOKEN=<your token>
source benchmarks/rlm_specprefill/.env_exports.sh   # sources rlm/.env + spec_prefill_llama/.env_exports.sh
```

Both Llama checkpoints (`meta-llama/Llama-3.1-8B-Instruct`,
`meta-llama/Llama-3.2-1B-Instruct`) are **gated on Hugging Face** — request
access to both on the HF model pages before downloading, or step 1 below
will fail with a 401/403, not a missing-file error.

Confirm both halves loaded (the script prints a summary):
```
[rlm_specprefill] env loaded. ANTHROPIC_API_KEY set: yes
[rlm_specprefill] LLAMA31_8B_MODEL_PATH=/scratch/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/...
```

Also install this project's own Python deps on the GPU node (not pinned to
a requirements file yet — install as needed): `pyyaml`, `python-dotenv`,
`anthropic`, and the `rlm` package itself (`pip install -e ../rlm` or
equivalent, so `import rlm` resolves).

## 1. Model checkpoints

Same as `../spec_prefill_llama/REPRODUCE.md` step 3 (Llama-3.1-8B target,
Llama-3.2-1B speculator) — not duplicated here. Verify `LLAMA31_8B_MODEL_PATH` /
`LLAMA32_1B_MODEL_PATH` resolve to real, complete snapshots (not just
tokenizer/config files) before proceeding.

## 2. Validate SpecPrefill itself on real hardware (blocking for Arms B/C only)

`vllm_patch/`'s Algorithm 1 implementation has **never been run on real
hardware** for the Llama pairing. This must pass before Arms B/C (which use
SpecPrefill) are trustworthy — it does not block Arm A (no SpecPrefill
involved at all), and it does not block the target_stage smoke tests in
step 5 below that only exercise the plain (non-pruned) engine path. Follow
`../spec_prefill_llama/REPRODUCE.md` steps 5.1-5.3 verbatim:

```bash
cd benchmarks/spec_prefill_llama
python3 test_vllm_patch.py                                        # no GPU needed
python3 validate_proposer.py --model $LLAMA32_1B_MODEL_PATH        # GPU
python3 validate_runner_integration.py \
    --target-model $LLAMA31_8B_MODEL_PATH \
    --speculator-model $LLAMA32_1B_MODEL_PATH                     # GPU
```

If any of these fail, do not proceed to Arm B/C testing below (step 6) —
fix or re-scope SpecPrefill validation first. Arm A / target_stage's plain
path (step 5) has no dependency on this passing.

## 3. Build the >131K-token eval set

Runnable now, no GPU needed (HF Hub tokenizer download only, not weights):

```bash
cd benchmarks/rlm_specprefill
python3 eval_data/prep_longbench_v2_long.py --max-keep -1
python3 eval_data/filter_by_token_length.py \
    --input eval_data/longbench_v2_long_samples.jsonl \
    --tokenizer-path $LLAMA31_8B_MODEL_PATH \
    --min-tokens 131000
python3 eval_data/gen_synthetic_niah.py --context-tokens 150000,300000,500000 --n-needles 3
```

Per `../rlm/rlm_specprefill_ablation_plan.md`'s EVAL SET CONSTRAINTS: only
contexts comfortably above ~131K tokens go into the eval set — below that,
prior work shows RLM underperforms the base model, which would confound
"SpecPrefill didn't help" with "RLM shouldn't have been used here."

**For the GPU-node smoke tests in steps 5-6 below, don't start with this
full eval set** — generate a tiny dataset first (cheap, fast, isolates
target_stage bugs from eval-set-scale problems):

```bash
python3 eval_data/gen_synthetic_niah.py --context-tokens 500,1000 --n-needles 1 --repeats 1 \
    --output eval_data/tiny_smoke_samples.jsonl
```

## 4. Smoke-test the RLM evidence-extraction + caching stage (no GPU)

Both runnable on any machine with Anthropic API access — confirmed working
during implementation (see IMPLEMENTATION_PLAN.md's Verification section):

```bash
cd benchmarks/rlm_specprefill
python3 rlm_stage/evidence_rlm.py --smoke-test      # RLM alone
python3 runner/smoke_test.py                        # RLM + evidence_cache.py together
```

If either fails with an `IndexError` from deep inside `rlm/clients/anthropic.py`,
that's the known `_default_answer` max-iterations issue documented in
IMPLEMENTATION_PLAN.md's load-bearing findings — re-run once or twice before
assuming something regressed; `runner/smoke_test.py` already retries this
automatically.

## 5. GPU-node smoke test: target_stage's plain (non-pruned) path

**Do this before running a full arm sweep.** `target_stage/vllm_offline_engine.py`
has never touched a real vLLM engine — validate it in isolation, on the
tiny dataset from step 3, before trusting `run_arm.py` end-to-end.

**5a. Pre-warm the evidence cache** (no GPU needed — do this from a login
node or any machine with Anthropic API access, so the GPU-node run below
doesn't spend GPU time waiting on Claude API calls):

```bash
cd benchmarks/rlm_specprefill
python3 runner/run_arm.py --arm A --dataset eval_data/tiny_smoke_samples.jsonl --dry-run
```

Confirm `results/A/timing.jsonl` has one row per sample and
`results/evidence_cache/` has one `.json` file per sample.

**5b. Run Arm A for real** (GPU-node, now touches vLLM for the first time
in this project's history):

```bash
python3 runner/run_arm.py --arm A --dataset eval_data/tiny_smoke_samples.jsonl \
    --target-model $LLAMA31_8B_MODEL_PATH
```

This re-uses the cached evidence from 5a (no new RLM/Anthropic calls) and
exercises `build_plain_target_engine` → `_submit_plain` →
`drive_engine_to_completion` → `answer_batch` for the first time against a
real engine. Check `results/A/predictions.jsonl` — each row should have a
non-empty `pred`, `finish_reason` of `"stop"` (not `"length"`, unless the
answer is genuinely getting cut off by `max_tokens`), and a plausible
`ttft_ms`.

**If this fails**, the most likely first-run issues (none of these have
been hit yet since this has never run — listed because they're the classes
of bug `predict_longbench_v2.py`'s own history warns about, not because
they're confirmed here):
- `max_num_batched_tokens` / `max_model_len` sizing — `vllm_offline_engine.py`'s
  `DEFAULT_MAX_NUM_BATCHED_TOKENS` (32768) is a guess sized for *compressed
  evidence*, not raw context; if evidence excerpts come out larger than
  expected, pass `--target-max-num-batched-tokens` explicitly (not yet a
  `run_arm.py` CLI flag — pass `max_num_batched_tokens=` directly to
  `build_plain_target_engine` if scripting this manually).
- Chat template / `enable_thinking=False` behavior — confirm Llama-3.1-8B's
  chat template still accepts that kwarg the way `predict_longbench_v2.py`
  found it does; if not, `chat_wrapper_pieces`/`build_prompt_pieces` in
  `vllm_offline_engine.py` need the same fix applied there.

## 6. GPU-node smoke test: target_stage's SpecPrefill path (Arms B/C)

**Blocked on step 2 passing.** Once it does:

```bash
python3 runner/run_arm.py --arm B --dataset eval_data/tiny_smoke_samples.jsonl \
    --target-model $LLAMA31_8B_MODEL_PATH --speculator-model $LLAMA32_1B_MODEL_PATH
```

Check `results/B/predictions.jsonl` — same sanity checks as step 5b, plus:
`keep_rate` should be close to `configs/spec_config_always_on.yaml`'s
configured `percentage` (0.5 by default) for samples whose evidence is
large enough for chunk-based pruning to meaningfully apply; very short
excerpts may round-trip near 100% kept regardless (nothing to prune away
usefully at that scale) — not a bug if so.

**Arm C** needs an `N_min` value. `calibration/sweep_n_min.py` (step 8
below) is what's supposed to produce a real, calibrated `configs/n_min.json`
— but that calibration wants a real candidate-text pool, not this tiny
2-sample smoke dataset. For this smoke test specifically, pass `--n-min`
manually with a plausible guess (e.g. a few
hundred tokens, well below the tiny dataset's evidence size, so the gate
actually exercises the "compress" path rather than trivially skipping
everything):

```bash
python3 runner/run_arm.py --arm C --dataset eval_data/tiny_smoke_samples.jsonl \
    --target-model $LLAMA31_8B_MODEL_PATH --speculator-model $LLAMA32_1B_MODEL_PATH \
    --n-min 50
```

Check the printed gate split (`Arm C gate (N_min=50): N skip (plain), M
compress (SpecPrefill)`) makes sense given the dataset, and that
`results/C/predictions.jsonl` has one row per sample regardless of which
bucket it went through.

## 7. Arm-A-only pass over the real eval set + the `f` go/no-go checkpoint

Once steps 5-6 pass on the tiny dataset, scale up to the real eval set from
step 3:

```bash
python3 runner/run_arm.py --arm A --dataset eval_data/longbench_v2_long_samples.jsonl \
    --target-model $LLAMA31_8B_MODEL_PATH
python3 analysis/aggregate_metrics.py --results-dir results --arms A \
    --dataset eval_data/longbench_v2_long_samples.jsonl
```

Per the ablation doc: compute `f` (target share of total latency) as a
**distribution** before investing further in Arms B/C — `aggregate_metrics.py`
prints exactly that (`f (target share of latency): p50=... p90=... mean=...`),
using the REAL per-request `generation_time_s` `target_stage/vllm_offline_engine.py`
records during `answer_batch` (not an estimate — see `IMPLEMENTATION_PLAN.md`'s
load-bearing findings for why this needed adding on top of
`predict_longbench_v2.py`'s own instrumentation, which never tracked true
per-request latency). If `f` is small (RLM's own search dominates), no
amount of SpecPrefill tuning yields a large end-to-end win — stop and
reconsider before step 8.

## 8. Calibrate `N_min`

Decoupled from live RLM by design (see `IMPLEMENTATION_PLAN.md` decision 8)
— never re-invokes RLM, sweeps fixed-size candidate-text bins against the
target+speculator pair alone:

```bash
python3 calibration/sweep_n_min.py \
    --evidence-cache results/evidence_cache --synthetic-niah eval_data/synthetic_niah_samples.jsonl \
    --target-model $LLAMA31_8B_MODEL_PATH --speculator-model $LLAMA32_1B_MODEL_PATH
```

Writes `configs/n_min.json` (`{"n_min": ..., "curve": [...]}`), which
`runner/run_arm.py --arm C` reads by default (no `--n-min` needed once this
exists — it was only a manual override for step 6's early smoke test).
Check the printed per-bin timings: SpecPrefill should get *relatively*
faster as `N` grows (fixed speculator-invocation overhead amortizing over a
larger prefill) — if it doesn't, something about the SpecPrefill path is
off, not just this calibration.

Then check score transferability (per the ablation doc: SpecPrefill was
validated on prompt-formatted benchmarks, not RLM's sliced/reordered
excerpt format — verify the keep-rate/quality relationship still holds
before trusting the `N_min` above):

```bash
# Needs spec_prefill_llama's own P001-P006 sweep + grading done first
# (that project's REPRODUCE.md) -- {exp_id}_result.json files in its results/ dir.
python3 calibration/transferability_check.py \
    --reference-dir ../spec_prefill_llama/results --niah-samples eval_data/synthetic_niah_samples.jsonl \
    --target-model $LLAMA31_8B_MODEL_PATH --speculator-model $LLAMA32_1B_MODEL_PATH
```

Read the printed table: any row marked `diverges? YES` means SpecPrefill's
quality/keep-rate relationship doesn't transfer cleanly to RLM's evidence
format at that keep percentage — per the ablation doc, recalibrate (or at
minimum treat Arm B/C's results at that keep rate with real skepticism)
rather than proceeding as if `N_min` alone settles the question.

## 9. Full A/B/C sweep over the real eval set

```bash
python3 runner/run_all_arms.py --dataset eval_data/longbench_v2_long_samples.jsonl \
    --target-model $LLAMA31_8B_MODEL_PATH --speculator-model $LLAMA32_1B_MODEL_PATH
python3 analysis/aggregate_metrics.py --results-dir results --arms A,B,C \
    --dataset eval_data/longbench_v2_long_samples.jsonl --output results/aggregate_report.json
```

`run_all_arms.py` sequences Arm A (reusing step 7's cached evidence — no
re-run) → Arm B → Arm C, and within Arm C, the plain-engine pass over the
gate's "skip" bucket before the SpecPrefill-engine pass over the "compress"
bucket — at most one vLLM `LLM` instance is ever live (see
`IMPLEMENTATION_PLAN.md` decision 3).

`aggregate_metrics.py` prints one summary block per arm (`f`-distribution,
`T_RLM_root`/`T_REPL_compute`/`T_RLM_subcalls`, realized depth/fan-out,
keep rate `r`, gate invoked/skipped split for Arm C, TTFT, and accuracy —
NIAH recall exactly, LongBench-v2 via a best-effort substring match, see
that module's `score_accuracy` docstring for why it's not
`grade_longbench_v2.py`'s exact-letter scoring) and writes the full report
to `--output` as JSON. Compare the three arms' `f` and accuracy side by
side from that file to answer the ablation's actual question: does
SpecPrefill (Arm B/C) meaningfully speed up the pipeline over Arm A without
a real accuracy cost, and does the gate (Arm C) capture most of that win
without paying Arm B's always-on overhead on queries too small to benefit.

## Expected runtime / hardware

TBD — inherits `../spec_prefill_llama/REPRODUCE.md`'s own "Expected runtime /
hardware" uncertainty (Llama-3.1-8B + Llama-3.2-1B, ~9B combined parameters,
likely fits on a single GPU, not yet confirmed) plus this project's own
root-model API latency/cost (depends on
dataset size and RLM's own convergence behavior — see the max-iterations
finding above — not yet measured at scale).
