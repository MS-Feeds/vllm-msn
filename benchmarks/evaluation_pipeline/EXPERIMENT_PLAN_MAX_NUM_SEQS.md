# Batch-Size (`max_num_seqs`) — Experiment Plan

Status: **infrastructure implemented, sweep not yet run**. Gaps #1-#3
below are done in `run_pipeline.py`/`run_experiments.sh` — see each gap
for what changed. Nothing in "Proposed experiment matrix" has actually
been executed yet.

## How to run this suite

This sweep and the sibling spec-decode sweep (`EXPERIMENT_PLAN.md`) now
share one driver (`run_pipeline.py`), selected via `--suite`:

```bash
./run_experiments.sh --suite batch --all              # all of B001-B005
./run_experiments.sh --suite batch B003                # single config
python3 run_pipeline.py --suite batch --list           # print the matrix
```

`--suite` defaults to `spec` (the existing `S0xx` sweep), so no existing
invocation's behavior changes. The two suites are deliberately never
merged into one `--all` — see gap #1 below.

## Goal

Identify the impact of batch size on throughput using **`max_num_seqs`**
— vLLM's direct cap on the number of concurrently-scheduled sequences —
as the *only* deliberately varied parameter. Everything else (dataset,
`max_model_len`, `gpu_memory_utilization`, spec_decode, quantization,
etc.) stays fixed across the sweep.

## Relationship to `../gemma4_moe_benchmarks/EXPERIMENT_PLAN.md` Group C

That plan already ran this exact axis — Group C (`E007`-`E009`:
mns=64/192/256 vs. control `E006` at mns=128; `E017`-`E018`: mns=16/32
for tighter-memory GPUs) — on its own workload (`sc1_delta_v2.jsonl`,
1000 prompts, fixed FP8 + CUDA graphs + MTP k=5 + text-only stack). It
found mns=64 outperformed 128/192/256 on the 40G mock run (see that
file's "Main findings" table).

This plan reruns the same axis inside `evaluation_pipeline`'s own
infrastructure instead — its three datasets (AIME/GPQA Diamond/
LiveCodeBench, 198-300 prompts each; see `run_pipeline.py`'s `DATASETS`)
and its MFU/MBU/spec-decode metric collection — rather than reusing
`gemma4_moe_benchmarks`'s workload and results. Proposed IDs use a
`B0xx` prefix ("Batch-size sweep") to avoid colliding with either
sibling's existing `E0xx`/`S0xx` namespaces.

Everything else needed already existed: the three datasets, the
spec-decode counters (unused here since spec_decode is held off — see
below), and the MFU/decode-window/GPU-spec machinery in
`hardware_metrics.py`.

## Design decisions

- **Spec decode held off (`spec_decode=False`) for every row.** Crossing
  `max_num_seqs` with the spec-decode/k axis that `EXPERIMENT_PLAN.md`
  (the `S0xx` sweep) already owns would confound two variables at once.
  If a later cross-sweep (batch size x k) is wanted, that's a separate
  follow-up plan, not this one.
- **`max_model_len` held at whatever `initialize_engine()`'s current
  default resolves to** (today: unset -> vLLM derives it from the model
  config; `run_pipeline.py` never overrides it). Recorded per run for
  auditability but not varied — this plan is intentionally the
  complement of `EXPERIMENT_PLAN_MAX_MODEL_LEN`-style sweeps, not a
  merge with them.
- **All three datasets included**, matching this pipeline's existing
  per-dataset-shape rationale (AIME long reasoning chains, GPQA short
  answers, LiveCodeBench full solutions) — batch-size sensitivity may
  differ by response-length profile the same way the k-sweep's
  acceptance-rate tradeoff does.
- **`num_prompts` must exceed the largest `max_num_seqs` tested** for
  the smaller values in the sweep to actually bind and create queuing
  pressure. All three existing datasets (198-300 prompts) already
  satisfy this against the proposed matrix below.

## Proposed experiment matrix (draft — not yet run)

| ID | Label | `max_num_seqs` | spec_decode | datasets |
|---|---|---:|:---:|---|
| B001 | mns=16 | 16 | off | aime, gpqa_diamond, livecodebench |
| B002 | mns=32 | 32 | off | " |
| B003 | mns=64 | 64 | off | " |
| B004 | mns=128 (vLLM-ish default control) | 128 | off | " |
| B005 | mns=256 | 256 | off | " |

Mirrors the value set already explored in `gemma4_moe_benchmarks` Group
C/D so results are comparable across the two pipelines' workloads, not
because those exact values are necessarily optimal for this dataset mix
— worth revisiting once gap #2's real-concurrency measurement exists,
in case it reveals the effective cap saturates earlier/later than
128-256 on these particular prompt lengths.

## Metrics captured per run

Reused as-is: requests/sec, elapsed time, output tokens/sec, MFU, MBU
(now using the clamped `batch_size` from gap #2), acceptance-rate fields
(will be null — spec decode is off for this whole sweep).

New: `max_num_seqs` column (gap #3, done). Not yet added: an
`achieved_avg_concurrency` field from the more rigorous real-time
sampler discussed in gap #2 and open question #1 — if/when that's
built, the CSV should record the real observed batch size alongside the
configured cap rather than assuming they're equal (the clamp assumes
the cap always binds for the whole run, which overstates true
time-averaged concurrency for reasons noted in the code comment).

## Expected signal

Throughput (req/s, output tok/s) should rise with `max_num_seqs` while
GPU memory/compute have headroom, then flatten or regress once KV cache
or compute saturates — matching the shape `gemma4_moe_benchmarks` Group
C already observed (mns=64 beating 128/192/256 in its 40G mock run).
Whether that same inflection point holds on this pipeline's shorter,
more varied-length datasets (vs. `sc1_delta_v2`'s longer, more uniform
prompts) is the actual question this plan answers — if the optimum
shifts by dataset shape, that's the interesting result.

## Open questions before running the sweep

1. Whether the more rigorous real-time concurrency sampler (background
   thread polling `vllm:num_requests_running`) is worth building later —
   deferred for now per the decision to start with the minimal clamp
   (gap #2). Revisit if the clamp's known bias (it assumes the cap binds
   for the whole run, not just part of it) turns out to matter once real
   numbers come back from B001-B005.
2. Confirm the `max_num_seqs` sweep values above (16/32/64/128/256) are
   the right range for the target hardware, or whether to match
   `gemma4_moe_benchmarks`'s exact set instead (which also includes 192).
3. Confirm reps per (experiment x dataset) — existing pipeline default
   is 2 (see `run_pipeline.py`'s `--reps` default); no reason assumed to
   change it here.
