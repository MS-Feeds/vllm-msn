# Top-K KV Cache Selection for Multi-Turn Conversation

Evaluates whether **SpecPrefill** (draft-model-based prefill token
preselection) generalizes from single-shot long-context QA to long, growing
**multi-turn conversations**, where context/prompt size is the serving
bottleneck across every turn, not just once. Target: Llama-3.1-8B-Instruct.
Speculator: Llama-3.2-1B-Instruct. Dataset: SCBench (3 configs:
`scbench_qa_eng`/`scbench_kv`/`scbench_summary`). Grid: keep-rate x
KV-entry-granularity, plus an oracle upper bound, at the protocol's `KEEP`
history-retention setting first.

## Contents

- `EXPERIMENT_PLAN.md` — the full protocol: motivation, architectural
  decisions (golden-context mode, persistent speculator engine,
  KEEP/DISCARD candidate pools, oracle upper bound), experiment matrix
  (`M000`/`ORACLE-k*`/`SPARSE-k*-g*`), success criteria, and an
  extensive numbered "findings" log of hardware bugs found and fixed while
  building the SPARSE pipeline.
- `REPRODUCE.md` — environment setup, checkpoint downloads, validation steps.
- `.env_exports.sh` — local env config (model paths, HF token).
- `vllm_patch/` — the multi-turn Algorithm 1 implementation (SPARSE pipeline).
- `test_vllm_patch.py` — CPU-only unit tests (no GPU needed). **Currently
  87/87 passing** (re-run `python3 test_vllm_patch.py` to confirm; grows
  as the pipeline grows, so re-check the count rather than trusting a
  stale figure).
- `validate_proposer.py` / `validate_runner_integration.py` /
  `validate_resumable_session.py` / `validate_sparse_attention.py` —
  GPU-node validation scripts, one per architectural mechanism. See
  "Validation results" below and `REPRODUCE.md` step 4.
- `diagnose_*.py` — targeted diagnostic scripts written to root-cause
  specific real-hardware bugs/anomalies found during experiments (see
  "Diagnostics" below).
- `datasets/prep_scbench.py` — downloads `microsoft/SCBench`'s 3 MVP
  configs.
- `predict_scbench.py` — runs the experiment matrix (per-conversation,
  per-turn sequential driving loop); writes a per-turn predictions JSONL
  per experiment.
- `grade_scbench.py` — scores predictions against `prep_scbench.py`'s
  samples, with per-SCBench-config metrics ported from the official
  `microsoft/MInference/scbench/eval_utils.py`.
- `compare_ceiling.py` — compares several predictions files on the turns
  they ALL cover, with a cluster-bootstrap CI over conversations. The tool
  for reading an `ORACLE-k*` run against the `SPARSE` row it bounds; grading
  each separately against the full samples file would compare different
  conversation sets.
- `ACCURACY_IMPROVEMENTS.md` — the candidate accuracy improvements drawn
  from the two papers and from this pipeline's own code, gated on the
  oracle result and ranked by effort.
- `flops_model.py` / `validate_flops_model.py` — analytic FLOP model for
  the combined speculator + target system (pure Python, no GPU needed) and
  its GPU-node validation harness.
- `sparse_decode_microbench.py` / `dense_context_sweep.py` /
  `stock_context_sweep.py` / `stock_vllm_control.py` /
  `gpu_vs_host_timing.py` / `ncu_kv_bytes_probe.py` — a family of
  microbenchmarks built to pin down decode-step latency vs. a
  memory-bandwidth roofline model (see "Microbenchmarks" below).
- `datasets/` — SCBench prompt sets (gitignored).
- `results/` — gitignored output directory (empty in this checkout).

Not implemented in this pass (confirmed out of scope with the user): the
protocol's baseline comparison methods (H2O, StreamingLLM, Quest, KVzip,
HeadKV) and head-level (rather than token-level) selection.

---

## The pipeline

**SPARSE-k\*-g\*** (persistent cache + sparse attention): the target
retains its full conversation KV cache persistently across every turn —
nothing is ever discarded — via resumable-session KV persistence, and a
decode-only block-table gather restricts attention to a subset of that
already-resident cache. What's compressed is decode-time attention compute,
not the KV cache itself. KV granularity is `16`/`32`/`64` (block-gather is
block-granular). History mode is self-generated (future turns are built
from the model's own actual output — the resumable-session mechanism has
no hook to substitute golden reference text). `keep_mode` is `keep` only
(nothing is ever evicted, so DISCARD's reason for existing doesn't apply).

---

## Experiment matrix (as configured)

Confirmed MVP scope: 3 SCBench configs (`scbench_qa_eng`/`scbench_kv`/
`scbench_summary`), no baseline methods implemented this pass.

| ID | Label | Keep rate | KV granularity | Keep mode |
|---|---|---:|---:|---|
| M000 | Baseline (no pruning) | 100% | — | — |
| ORACLE-k{80,60,40,20} | Oracle upper bound (target checkpoint scores instead of the speculator) | 80/60/40/20% | 32 (pairs with SPARSE-k\*-g32) | keep |
| SPARSE-k{80,60,40,20}-g{16,32,64} | Persistent cache + sparse attention | 80/60/40/20% | 16/32/64 | keep (only) |

Shared SpecPrefill hyperparameters (matches `../spec_prefill_llama/`'s
single-turn matrix): BF16, look-ahead count **8**, `pool_kernel_size`
**13**, `enforce_eager=True` on both engines.

`predict_scbench.py --list` prints this matrix programmatically.

### ORACLE-k\*: what it is and what it bounds

`ORACLE-k{N}` is `SPARSE-k{N}-g32` with exactly one thing changed: the
importance scores come from the **target checkpoint** (Llama-3.1-8B) scoring
its own attention, instead of the 1B speculator's estimate of it. Same
driving loop, same persistent session, same block-gather, same prompt
rendering, same keep rate — the scorer engine just holds a different
checkpoint, on the same GPU slot the speculator would have used. So the
sweep's unexplained `scbench_kv` degradation splits into two measurable
halves:

| Gap | What it measures |
|---|---|
| `ORACLE-k{N}` − `SPARSE-k{N}-g32` | what the 1B speculator's estimation error costs |
| `M000` − `ORACLE-k{N}` | what block-granular sparse decode costs with the estimator held as good as this method allows |

Whichever dominates says where accuracy work should go: better scoring
(layer/head aggregation, retrieval-head filtering, lookahead source) versus
better allocation and mechanism (block dilation, nucleus budgets, per-layer
selection, mid-turn re-selection). The graded SPARSE sweep alone cannot tell
them apart, which is why this row runs first.

**What it is not.** The scorer's lookahead queries come from tokens it
generated itself, so this is the ceiling for "what if the draft model were as
good as the target?" — SpecPrefill's own self-speculation limit — not "what if
we knew the right answer?". A golden-answer teacher-forced variant would be
strictly stronger; `vllm_patch/pruner.py::compute_oracle_kept_pairs` is kept
as its entry point. See `EXPERIMENT_PLAN.md`'s "Oracle upper bound" for why
the wired-up version took the separate-engine route (short version: a scoring
pass driven through the target's own engine would write into the persistent
conversation session it is supposed to be scoring).

**Cost.** The scoring pass is an 8B prefill instead of a 1B one, ~8x per
turn. Prefix caching confines most of that to each conversation's turn 0 in
KEEP mode, but an oracle row is still meaningfully slower than its SPARSE
partner — which is why it pairs with one representative granularity (32)
rather than the full cross.

**Memory.** `--oracle-scorer-gpu-memory-utilization` defaults to **0.6**, not
the target's 0.85, and that gap is load-bearing rather than conservative:
scoring materializes per-layer K for the whole context (~5.8GB in bf16 at 88k
tokens for the 8B, vs. ~1.4GB for the 1B) plus the attention-score tensor and
its fp32 softmax, and all of it is allocated from the headroom vLLM did *not*
reserve. Too high a value OOMs during scoring, not during generation. The
1B speculator's own 0.2 default never exposed this because it left 64GB free.

Run it with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python predict_scbench.py --exp oracle --target-prefill-chunk-tokens 32768 --scorer-prefill-chunk-tokens 32768
```

**Why the chunk flags** (learned from the first real ORACLE-k20 run, which
OOM'd on the **scorer** — dumped batch `scbench_kv-0::turn0`,
`prompt_token_ids_len=124009`, `max_tokens=9`, i.e. `proposer.py`'s scoring
request, not the target's `::sparse-session`): SCBench's longest contexts are
~124k tokens, and with `max_num_batched_tokens` at the context ceiling the
whole thing goes through one forward pass. At Llama-3.1-8B's
`intermediate_size` of 14336, a single SiluAndMul output for that batch is
**3.31GiB** in bf16, with several such buffers live at once — tens of GiB of
transients that no `gpu_memory_utilization` reserves, because vLLM sizes the
KV pool from a *profiled* activation peak that a 124k-token batch blows past.

Chunking alone was **not** enough, and the second OOM said why: it dropped
the failed allocation from 3.31GiB to 896MiB, but the scorer process was
still holding **78.00GiB allocated** on a 79.25GiB card, with only 72MiB
reserved-but-unallocated (so: not fragmentation). The rest was a *retained*
buffer, not a transient — see below.

### The retained-query bug (fixed)

`speculator_worker.py`'s capture hook appended **every** forward call's query
slice, prefill chunks included, and left `end_capture` to filter them out by
shape afterwards. Each captured slice is a *view*, so one retained prefill
slice pins that whole forward's query tensor, per layer, until the turn's
scoring pass runs:

| Scorer | prefill queries retained @124k tokens | budget @util | card left free |
|---|---:|---:|---:|
| Llama-3.2-1B (SPARSE) | 16 × 0.47 = **7.6 GiB** | 15.9 GiB @0.2 | ~63 GiB |
| Llama-3.1-8B (ORACLE) | 32 × 0.95 = **30.3 GiB** | 47.5 GiB @0.6 | ~32 GiB |

15GiB weights + ~30GiB KV pool + 30.3GiB of prefill queries ≈ the 78.00GiB
observed. It is **invariant to prefill chunk size** — chunking splits the same
total across more, smaller views — which is exactly why capping
`max_num_batched_tokens` shrank the transient and not this.

**Why no SPARSE run ever surfaced it**: the 1B's 7.6GiB was absorbed by the
~63GiB the card had free at `--speculator-gpu-memory-utilization 0.2`. The
same bug was there the whole time, just paid for out of headroom. The 8B
scorer has 2x less headroom and retains 4x more.

The fix applies the same "a decode step is exactly 1 token" rule at capture
time instead of after the fact (`kv_cache_utils.is_decode_query_slice`), so
prefill queries are never retained at all; `end_capture`'s shape filter stays
as the second line of defense. The scorer also runs *first* —
`compute_pruned_turn` precedes the turn's `add_request` — so on turn 0 of the
first conversation it prefills 124k tokens before the target does anything,
which is why an oracle run died there rather than anywhere downstream. The
target itself is not known to be marginal: it survived the entire published
SPARSE sweep at 0.85 with the same token stream.

`--target-prefill-chunk-tokens` / `--scorer-prefill-chunk-tokens` set the
engines' per-step batch size **without** moving `--*-max-num-batched-tokens`,
which stays the context ceiling *and* the conversation-skip threshold —
lowering that instead would silently skip the longest conversations rather
than serve them (watch `num_skipped_too_large`). Chunking costs nothing in
accuracy: `sparse_target_runner.py` leaves every step with
`num_computed < num_prompt` at full unrestricted attention, and
`speculator_worker.py::end_capture` filters captured queries by shape
(exactly-1-token entries) precisely so any number of leading prefill chunks
is harmless. Both flags default to off, so no already-measured row changes
its batching underneath the published sweep.

Placement is now checked and printed before either engine allocates: the
scorer must not land on the target's own GPU (two 8B engines do not fit on
one card), and the log line names the model, device, utilization, and
`CUDA_VISIBLE_DEVICES` for both engines — a CUDA OOM traceback cannot, since
every engine calls its own device "GPU 0".

`scbench_kv` is where the SPARSE degradation actually lives, so for a first
read: `--exp ORACLE-k20 --scbench-config scbench_kv --max-conversations 20`
against the matching `SPARSE-k20-g32` slice is the cheapest run that answers
the attribution question at the grid's worst-case corner.

---

## Results: full SCBench graded sweep (M000 + all 12 SPARSE-k\*-g\* configs)

The current, most complete real results: M000 baseline plus the full
`SPARSE-k{20,40,60,80}-g{16,32,64}` grid (12 configs), graded by
`grade_scbench.py` against all 3 SCBench configs. `total_turns=1201` is the
same in all three tables because grading was run against the full combined
samples file each time; `matched` is the number of those 1201 turns that
actually belong to that config and have a scorable prediction (the rest are
`missing` — turns belonging to the *other* two configs, not a failure of
this run). This means `scbench_qa_eng` (~24 conversations, 117 matched
turns) and `scbench_summary` (~70 conversations, 350 matched turns) carry
noticeably more sampling noise than `scbench_kv` (~100 conversations, 500
matched turns).

### `scbench_kv` (`in_match`, exact/substring retrieval — 500/1201 turns matched)

| Config | Score | Δ vs. M000 | Δ vs. M000 (relative) |
|---|---:|---:|---:|
| **M000** | **81.6** | — | — |
| SPARSE-k20-g16 | 46.6 | −35.0 | −42.9% |
| SPARSE-k20-g32 | 56.6 | −25.0 | −30.6% |
| SPARSE-k20-g64 | 62.2 | −19.4 | −23.8% |
| SPARSE-k40-g16 | 62.0 | −19.6 | −24.0% |
| SPARSE-k40-g32 | 65.4 | −16.2 | −19.9% |
| SPARSE-k40-g64 | 70.8 | −10.8 | −13.2% |
| SPARSE-k60-g16 | 72.6 | −9.0 | −11.0% |
| SPARSE-k60-g32 | 72.8 | −8.8 | −10.8% |
| SPARSE-k60-g64 | 76.8 | −4.8 | −5.9% |
| SPARSE-k80-g16 | 78.6 | −3.0 | −3.7% |
| SPARSE-k80-g32 | 79.2 | −2.4 | −2.9% |
| SPARSE-k80-g64 | 80.8 | −0.8 | −1.0% |

**Clean, monotonic dose-response**: score rises with both keep rate and KV
granularity, exactly as expected if sparsification is genuinely degrading
retrieval and coarser/larger blocks preserve more useful context per
selected unit. **M000 outperforms every single SPARSE config** on this
metric — the earlier partial sweep's apparent "SPARSE beats M000" pattern
(see "Superseded" below) does not hold up once the full grid is graded,
consistent with that pattern having been a rendering artifact all along.
Against the plan's "≤5% score drop" success criterion: only
**SPARSE-k80-g64** (−1.0% relative, −0.8 pts absolute) and **SPARSE-k60-g64**
(−5.9% relative, −4.8 pts absolute) come close; every `g16` and `g32`
config, and everything at `k20`/`k40`, misses it by a wide margin. `k20-g16`
is the worst case, losing 43% of its relative score.

### `scbench_qa_eng` (`qa_f1_score`, token-overlap F1 — 117/1201 turns matched)

| Config | Score | Δ vs. M000 |
|---|---:|---:|
| **M000** | **27.26** | — |
| SPARSE-k20-g16 | 26.12 | −1.14 |
| SPARSE-k20-g32 | 26.25 | −1.01 |
| SPARSE-k20-g64 | 26.89 | −0.37 |
| SPARSE-k40-g16 | 26.55 | −0.71 |
| SPARSE-k40-g32 | 27.20 | −0.06 |
| SPARSE-k40-g64 | 27.10 | −0.16 |
| SPARSE-k60-g16 | 27.99 | +0.73 |
| SPARSE-k60-g32 | 28.57 | +1.31 |
| SPARSE-k60-g64 | 25.62 | −1.64 |
| SPARSE-k80-g16 | 27.99 | +0.73 |
| SPARSE-k80-g32 | 26.63 | −0.63 |
| SPARSE-k80-g64 | 28.37 | +1.11 |

Against that baseline, every SPARSE config lands within **±1.64 points**
(~6% relative) of M000, with no clear keep-rate or granularity trend —
`k60-g32` is the best (+1.31) and `k60-g64` the worst (−1.64) with no
consistent ordering between them elsewhere in the grid. This reads as
near-parity/noise at this sample size (only ~24 conversations, the
smallest of the three configs) rather than either a real SPARSE
improvement or a real degradation — consistent with the flat pattern seen
on `scbench_summary` below, and unlike the large, clean, monotonic
degradation seen on `scbench_kv` above.

### `scbench_summary` (`rouge_l_f1` — 350/1201 turns matched)

| Config | Score | Δ vs. M000 |
|---|---:|---:|
| **M000** | **36.04** | — |
| SPARSE-k20-g16 | 36.01 | −0.03 |
| SPARSE-k20-g32 | 35.92 | −0.12 |
| SPARSE-k20-g64 | 35.84 | −0.20 |
| SPARSE-k40-g16 | 36.28 | +0.24 |
| SPARSE-k40-g32 | 36.16 | +0.12 |
| SPARSE-k40-g64 | 36.32 | +0.28 |
| SPARSE-k60-g16 | 36.10 | +0.06 |
| SPARSE-k60-g32 | 36.40 | +0.36 |
| SPARSE-k60-g64 | 36.46 | +0.42 |
| SPARSE-k80-g16 | 35.94 | −0.10 |
| SPARSE-k80-g32 | 36.04 | ≈0.00 |
| SPARSE-k80-g64 | 36.04 | ≈0.00 |

**Flat across the entire grid.** Every SPARSE config lands within ±0.5
points (~1% relative) of M000, with no visible trend by keep rate or
granularity — even `k20-g16` (only 20% of context kept, coarsest
granularity) shows essentially no degradation. Two plausible, currently
undistinguished explanations: (a) global-summary quality genuinely doesn't
depend much on which specific tokens the sparse gather selects, as long as
each selected block carries the same amount of the source document's
"gist," or (b) `rouge_l_f1`'s LCS-based scoring is simply insensitive
enough (compared to `in_match`'s exact-substring requirement) to hide a
real, smaller degradation. Not yet investigated further.

---

## FLOP model (analytic, not hardware-measured)

`flops_model.py` is a pure-Python (no torch/vLLM import) analytic FLOP
accounting model for the combined speculator + target system, on
per-turn token counts measured during a run.

**Key analytic finding** (a model projection, not a profiler measurement —
label it as such): at 77k context and 64 output tokens, **SPARSE runs at
~120% of M000's FLOPs at every keep rate** — i.e. under this accounting,
SPARSE's benefit is not a FLOP reduction at all, it's a **memory-bandwidth**
reduction (fewer KV bytes read per decode step), which the FLOP model
doesn't capture. This distinction matters for reading the microbenchmark
section above: SPARSE's case for being faster rests entirely on the
roofline/bandwidth story, not on doing less arithmetic.

The speculator's own scoring pass is negligible in this accounting: ~0.04
TFLOP vs. ~538 TFLOP for that same speculator's prefill at 77k context
(~0.007% of the turn) on the 1B-parameter speculator.

Conventions used: 1 MAC = 2 FLOPs; GQA does not reduce attention FLOPs
(only QKV projection scales with `num_kv_heads`); `lm_head` is charged once
per logits row, not once per token; RMSNorm/RoPE/softmax/residual/
elementwise ops are excluded (documented as <1% of total). Validated
structurally by `test_vllm_patch.py`'s FLOP unit tests (all passing, see
above) and intended to be cross-checked against real GPU execution by
`validate_flops_model.py` — no captured pass/fail result for that
cross-check exists in this checkout yet.

---

## FLOP results (measured, from real SCBench runs)


`all_runs_summary.csv` on the scbench-summary dataset

| Config | Total TFLOP/turn | Δ TFLOP vs. M000 | Target decode TFLOP/turn | Achieved TFLOP/s | Sec/turn (excl. turn 0) | Δ latency vs. M000 |
|---|---:|---:|---:|---:|---:|---:|
| **M000** | 858.0 | — | 3.88 | 149.5 | 1.27 | — |
| SPARSE-k80-g16 | 1039.4 | +21.1% | 3.48 | 140.6 | 1.87 | +47.0% |
| SPARSE-k80-g32 | 1039.3 | +21.1% | 3.39 | 139.4 | 1.92 | +50.9% |
| SPARSE-k80-g64 | 1039.3 | +21.1% | 3.36 | 141.2 | 1.84 | +45.0% |
| SPARSE-k60-g16 | 1038.9 | +21.1% | 2.95 | 141.1 | 1.83 | +44.3% |
| SPARSE-k60-g32 | 1038.7 | +21.1% | 2.82 | 140.7 | 1.85 | +45.9% |
| SPARSE-k60-g64 | 1038.7 | +21.1% | 2.75 | 142.1 | 1.79 | +40.9% |
| SPARSE-k40-g16 | 1038.2 | +21.0% | 2.31 | 142.0 | 1.79 | +40.9% |
| SPARSE-k40-g32 | 1038.1 | +21.0% | 2.19 | 142.5 | 1.76 | +38.8% |
| SPARSE-k40-g64 | 1038.0 | +21.0% | 2.12 | 142.1 | 1.78 | +40.2% |
| SPARSE-k20-g16 | 1037.5 | +20.9% | 1.62 | 142.4 | 1.76 | +38.5% |
| SPARSE-k20-g32 | 1037.5 | +20.9% | 1.54 | 143.0 | 1.73 | +36.3% |
| SPARSE-k20-g64 | 1037.4 | +20.9% | 1.48 | 142.4 | 1.75 | +38.2% |


## Benchmark

**SCBench** (arXiv:2412.10319, `microsoft/SCBench` on Hugging Face) --
`scbench_qa_eng` (semantic retrieval / free-form QA), `scbench_kv` (string /
exact retrieval), `scbench_summary` (global-information / summarization).
Confirmed empirically as exactly 5 turns sharing one long context per row
across all 3 MVP configs (not the HF dataset card's stated "2-4"). Metrics:
`in_match` for `scbench_kv`, token-overlap `qa_f1_score` for
`scbench_qa_eng`, a dependency-free LCS-based ROUGE-L reimplementation for
`scbench_summary` — all ported from `microsoft/MInference/scbench/
eval_utils.py`.

---

## Success criteria

- **Score drop ≤5% vs. M000, per config**: assessed on the full-grid
  aggregate above (not yet broken down by `turn_idx` — that needs a re-run
  of the diagnostics against the new sweep). **`scbench_kv`**: only
  `SPARSE-k80-g64` (−1.0%) and `SPARSE-k60-g64` (−5.9%, borderline) come
  close; every other config misses by a wide margin, worst case −43% at
  `k20-g16`. **`scbench_summary`**: every config passes trivially (all
  within ~1% of M000, flat across the whole grid). **`scbench_qa_eng`**:
  every config passes (all within ~6% relative / ±1.64 points of M000's
  27.26, no clear keep-rate/granularity trend — noise-dominated at this
  sample size rather than a real effect).
- Report TTFT/throughput improvement over M000 for each keep-rate row — deal with later
- Report oracle rows as an accuracy ceiling — wired up, not yet run. Read
  as the two gaps in "ORACLE-k\*: what it is and what it bounds" above,
  not as a single number.

---

## What to run next

1. **`--exp oracle`** — the ceiling rows, now wired up. Everything else on
   this list is speculative until the two gaps above are measured, because
   they decide whether accuracy work belongs in the scorer or in the
   selection/gather mechanism.
2. Then, guided by that split, the candidates in
   **[ACCURACY_IMPROVEMENTS.md](ACCURACY_IMPROVEMENTS.md)** — scoring-signal
   changes if the estimator is what's losing points, allocation/mechanism
   changes if it isn't. That document's Step 0 is the decision rule.
3. Per-`turn_idx` score breakdowns for the existing sweep — the aggregate
   tables hide whether degradation compounds across turns, which is the
   multi-turn thesis itself.

---

## References

- [SpecPrefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation](https://arxiv.org/abs/2502.02789) (ICML 2025)
- [SCBench: A KV Cache-Centric Analysis of Long-Context Methods](https://arxiv.org/pdf/2412.10319)
- `microsoft/SCBench` (Hugging Face) / `microsoft/MInference` (GitHub, `scbench/` -- reference harness `eval_utils.py`'s metric functions ported into `grade_scbench.py`)
- `../spec_prefill_llama/EXPERIMENT_PLAN.md` -- the single-turn pipeline this was built on top of
- `../spec_prefill_llama/REPRODUCE.md` -- environment setup this pipeline's own `REPRODUCE.md` follows the same conventions as

---

## Files in this directory

| File | Purpose |
|---|---|
| `EXPERIMENT_PLAN.md` | Full protocol + architectural decisions + numbered bug-fix findings log |
| `README.md` | This file — experiment overview, parameters, and results |
| `REPRODUCE.md` | Environment setup + reproduction steps |
| `.env_exports.sh` | Local env config (model paths, HF token) |
| `vllm_patch/` | The multi-turn Algorithm 1 implementation (SPARSE pipeline) |
| `test_vllm_patch.py` | CPU-only unit tests — 87/87 passing |
| `validate_proposer.py` | GPU-node validation: persistent speculator engine, cross-turn KV read-back |
| `validate_runner_integration.py` | GPU-node validation: `worker_cls` wiring + multi-turn RoPE position-override correctness |
| `validate_resumable_session.py` | GPU-node validation: target-side session persistence (TTFT evidence) |
| `validate_sparse_attention.py` | GPU-node validation: decode-step block-gather sparse attention (needle-in-haystack) |
| `diagnose_turn_index_gap.py` | Diagnoses the turn-0 accuracy dip: rendering-mismatch vs. truncation hypotheses |
| `diagnose_gold_survival.py` | Diagnoses the turn-0 accuracy dip: does the gold answer survive selection? |
| `diagnose_speculator_selection.py` | Decodes the speculator's actual selected text, to isolate selection-side vs. target-side bugs |
| `diagnose_target_gather_metadata.py` | Checks target-side gathered block count against an independently computed expectation |
| `diagnose_h1_metadata.py` | Dumps attention-metadata length fields to catch stale-field bugs (found `max_seq_len` staleness) |
| `sparse_decode_microbench.py` / `dense_context_sweep.py` / `stock_context_sweep.py` / `stock_vllm_control.py` / `gpu_vs_host_timing.py` / `ncu_kv_bytes_probe.py` | Decode-latency-vs-bandwidth-roofline microbenchmark family |
| `flops_model.py` | Analytic FLOP model (pure Python, no GPU needed) |
| `validate_flops_model.py` | GPU-node validation of the FLOP model against real profiler output |
| `datasets/prep_scbench.py` | Downloads `microsoft/SCBench`'s 3 MVP configs, writes `datasets/scbench_samples.jsonl` |
| `predict_scbench.py` | Runs the M000/ORACLE-k\*/SPARSE-k\*-g\* matrix, writes a per-turn predictions JSONL per experiment |
| `grade_scbench.py` | Scores a predictions file against `prep_scbench.py`'s samples, per-config metrics |
| `compare_ceiling.py` | Compares predictions files on their common turns, with a cluster-bootstrap CI — for reading ORACLE-k\* against its SPARSE partner |
| `ACCURACY_IMPROVEMENTS.md` | Candidate accuracy improvements, gated on the oracle result and ranked by effort |
| `datasets/` | SCBench prep output (gitignored) |
| `results/` | Output directory (gitignored, empty in this checkout) |
