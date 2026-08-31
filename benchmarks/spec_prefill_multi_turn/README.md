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
  154/154 passing** (re-run `python3 test_vllm_patch.py` to confirm; grows
  as the pipeline grows, so re-check the count rather than trusting a
  stale figure).
- `validate_proposer.py` / `validate_runner_integration.py` /
  `validate_resumable_session.py` / `validate_sparse_attention.py` —
  GPU-node validation scripts, one per architectural mechanism. See
  "Validation results" below and `REPRODUCE.md` step 4.
- `diagnose_*.py` — targeted diagnostic scripts written to root-cause
  specific real-hardware bugs/anomalies found during experiments (see
  "Diagnostics" below). `diagnose_sliding_window_votes.py` is the
  interleaved-attention gate — see "Gemma 4 / interleaved-attention support".
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
block-table gather restricts attention to a subset of that already-resident
cache. What's compressed is attention compute, not the KV cache itself. KV
granularity is `16`/`32`/`64` (block-gather is block-granular). History
mode is self-generated (future turns are built from the model's own actual
output — the resumable-session mechanism has no hook to substitute golden
reference text). `keep_mode` is `keep` only (nothing is ever evicted, so
DISCARD's reason for existing doesn't apply).

The gather's **scope** is a run-level choice, and it changes what the row
measures:

| scope | flag | turn N's prefill | turn N's decode |
| --- | --- | --- | --- |
| decode-only (default) | — | dense over the full resident cache | restricted |
| prefill + decode | `--sparse-prefill` | restricted to selected blocks + a contiguous tail covering the turn's own tokens | restricted |

Every published `SPARSE-k*`/`ORACLE-k*` row was measured under decode-only,
which is why it stays the default; `--sparse-prefill` rows are tagged
`[prefill=sparse]` in the `label` column so the two never get compared by
accident. Turn 0 is dense under both — its prefill is where the context's
KV is computed for the first time, and computing it under a restricted view
would poison the persistent cache every later turn's selection reads from.
Under `--sparse-prefill` the `target_prefill` FLOPs are measured per prefill
chunk rather than derived analytically, since the analytic model assumes
every new token attends every cached token.

---

## Experiment matrix (as configured)

Confirmed MVP scope: 3 SCBench configs (`scbench_qa_eng`/`scbench_kv`/
`scbench_summary`), no baseline methods implemented this pass.

| ID | Label | Keep rate | KV granularity | Keep mode |
|---|---|---:|---:|---|
| M000 | Baseline (no pruning) | 100% | — | — |
| ORACLE-k{80,60,40,20} | Oracle upper bound (target checkpoint scores instead of the speculator) | 80/60/40/20% | 32 (pairs with SPARSE-k\*-g32) | keep |
| SPARSE-k{80,60,40,20}-g{16,32,64} | Persistent cache + sparse attention | 80/60/40/20% | 16/32/64 | keep (only) |
| EARLY-k20-g32-L{1..8} | Scorer = the target's own first n layers (`r = n/32`) | 20% | 32 | keep |
| EARLY-k{60,80}-g32-L{2,4} | Same, at the keep rates a cheap `r` unlocks | 60/80% | 32 | keep |

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

`scbench_kv` is where the SPARSE degradation actually lives, so that is the
config to run first.

> **`--max-conversations N` is a biased sample, not a cheap one.** It takes
> the first N rows in file order ([predict_scbench.py:347](predict_scbench.py:347)),
> and `scbench_kv`'s file order is ordered by difficulty. Measured, not
> theorized: M000 scores **57.0** on the first 20 conversations and **87.75**
> on the other 80, against its 81.60 full-set figure. Every gap shrinks with
> it — M000 − SPARSE-k20-g32 is 25.0 points on the full set and 5.0 points on
> that prefix, because a baseline that has already lost the points has none
> left for sparsification to take. A 20-conversation ceiling run is therefore
> not a cheap preview of the full one; it is a different, unreadable
> measurement. Run the full config, or add a seeded random sample.
>
> (Conversation ids are HF row ids and step by 5 — `kv-0, kv-5, … kv-95` is
> the first 20 conversations, not a stride across 100. Easy to misread as a
> spread sample; it isn't one.)

---


### EARLY-k\*-g32-L\<n>: the target's own first *n* layers as the scorer

`EARLY-k{K}-g32-L{n}` is `SPARSE-k{K}-g32` with the 1B speculator replaced by
the **target checkpoint truncated to its first n layers** — the same engine,
the same driving loop, the same block-gather, built as
`LLM(model=<target>, hf_overrides={"num_hidden_layers": n})`.

It exists because the scorer/target cost ratio then needs no measurement:

> `A = 32 · 4 · 32 · 128`, `B = n · 4 · 32 · 128`, so **`r = n/32` exactly.**

and `SPECULATION_ECONOMICS.md`'s win condition `(d + o)(1 - r - k) > 12r`
turns entirely on `r`. At today's `r = 1/4` the useful keep rate can never
exceed 75%, while `k80` is the only rate clearing a 5% accuracy drop on
`scbench_kv` — the structural bind that section describes. At `n = 2`,
`r = 1/16` and both `k60` and `k80` clear the ceiling.

| n | r | max useful keep | fixed overhead `12r` |
|---:|---:|---:|---:|
| 1 | 1/32 | 96.9% | 0.375 |
| 2 | 1/16 | 93.8% | 0.75 |
| 4 | 1/8 | 87.5% | 1.5 |
| 8 | 1/4 | 75.0% | 3.0 |

**Why the sweep is 1..8.** `n = 8` reproduces the 1B speculator's `r`
exactly, so `EARLY-k20-g32-L8` vs. `SPARSE-k20-g32` is a controlled,
equal-cost head-to-head — and past n=8 the family is strictly worse than the
status quo on the axis it exists to improve. `ORACLE-k{K}` is this family's
own `n = 32` end point.

**Rows.** `n = 1..8` at the `k20-g32` probe point (the corner where the
estimator gap is largest, so scorers are distinguishable above the noise —
at `k80` almost nothing is pruned and every `n` looks alike), plus
`EARLY-k{60,80}-g32-L{2,4}` for the economics question that `k20` cannot
answer. `--exp early` selects all 12; `--scbench-config` and `--chunk-size`
work exactly as they do for every other family.

**Run the gate first.** `diagnose_retrieval_heads.py --layer-prefix-budgets
1,2,3,4,5,6,7,8,16,32 --speculator-model <target>` measures gold-answer
survival for **every n in one speculator-only pass**, no target engine and no
grading — a layer prefix is just the first `n · num_heads` rows of the
flattened `layer*head` axis the §1.3 fixed-head-set machinery already takes.
Same "measure the ceiling before building toward it" move as `ORACLE-k20`.

**Two things to read the results against.**

1. The gate is an *upper bound*, not a prediction of the EARLY rows. Its
   lookahead tokens still come from the full 32-layer forward pass, so it
   isolates *"is early-layer attention informative"* from *"can a truncated
   model produce usable lookahead queries."*
2. A truncated scorer decodes lookahead through the target's final norm and
   `lm_head`, trained for layer-32 outputs — an untuned early-exit head. Those
   tokens degrade as `n` shrinks. That is a real property of the proposal, not
   a bug, but it means a poor EARLY row has two possible causes and the gate is
   what separates them.

**Cost caveat.** These rows run the scorer as a *separate engine*, so they
still pay its own prefill (turn 0's measured +21% penalty). A fused
implementation — reading the attention out of the target's own first n layers
mid-prefill — removes that entirely. The grid measures the pessimistic bound.

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

### ORACLE-k20: the ceiling, and what the 25-point gap is made of

Run over the same 100 conversations / 500 turns as every number above.

| Row | Score | vs. M000 |
|---|---:|---:|
| M000 | 81.6 | — |
| **ORACLE-k20** | **73.6** | **−8.0** |
| SPARSE-k20-g32 | 56.6 | −25.0 |

| Term | Points | Share |
|---|---:|---:|
| **Estimator** — the 1B speculator's estimation error (`ORACLE` − `SPARSE`) | **17.0** | **68%** |
| **Mechanism** — block-granular sparse decode with a perfect-as-this-method-allows estimator (`M000` − `ORACLE`) | 8.0 | 32% |

**Roughly 2:1 in favour of the scorer being the problem.** Since the scoring
pass is ~0.007% of a turn's FLOPs, those 17 points are the cheapest available.
Per-turn, neither gap compounds with `turn_idx` (mechanism 9/8/8/12/3,
estimator 23/14/18/9/21) — degradation is a per-turn selection problem
repeated five times, not an accumulating one. Turn 0 is the worst turn for
every row including M000 (70.0, vs 86.0 at turn 4) and is where sparsification
costs most (−32, of which 23 is the estimator). See
[ACCURACY_IMPROVEMENTS.md](ACCURACY_IMPROVEMENTS.md) for what this selects.

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

## Gemma 4 / interleaved-attention support (gate phase, in progress)

Target for this work: **Gemma-4-31B (dense)** with **Gemma-4-E2B-it** as
speculator. Scoped deliberately to a gate-plus-scoring-fidelity phase, then a
re-decision — see the porting plan for the full blocker analysis.

**What is ready.** The speculator-side scoring path no longer assumes Llama:

- The query-capture hook is architecture-generic. It hooks `Attention.forward`
  and reads the post-RoPE query as an *argument* rather than reproducing a
  model's attention `forward` body, so it cannot drift from the model it was
  copied from. Gemma 4 gives such a copy plenty to drift on — `q_norm` applied
  per-head *before* RoPE, a `use_k_eq_v` branch deriving V from the pre-norm K,
  and a KV-shared branch that RoPEs Q only. The Llama-only `NotImplementedError`
  gate is gone as a result, not joined by a second branch.
- `scoring.LayerGeometry` carries the model's own per-layer facts, read off the
  live `Attention` modules (not `hf_config`): the attention **scale** — Gemma 4
  uses `scaling = 1.0`, not `1/sqrt(head_dim)` — plus `attn_logit_softcapping`,
  `layer_types`, and which layers are KV-shared.
- **Three ways to score an interleaved model**, to be graded against each
  other rather than argued about:
  1. *unmasked* (the default, and what every published row used) — every
     layer scores the whole context. Measured on Gemma-4-E2B / `scbench_kv`
     at keep=0.3: sliding layers win **+14.3 points over a random-winner
     null** on the positions selection kept, against +6.1 on positions it
     pruned. Their long-range `Q·K` is uncalibrated, and `max` selects
     extremes.
  2. `score_layers="global_only"` — drop the sliding layers from the vote.
  3. `mask_sliding_window=True` — mask each sliding layer to its own window
     before the softmax, keeping its real opinion about what it can see.
     More principled than (2), but it recovers signal only near the query:
     past 512 tokens a sliding layer is silent either way, so both leave the
     same 7 of 35 full-attention layers deciding long-range retrieval.
- `score_layers="global_only"` restricts scoring to full-attention layers. On a
  5:1 interleave, 5 of every 6 layers can never attend beyond a 512–1024 token
  window, so their score for a distant position is a number the model never
  computes — and `max` over (layer, head) lets any one of them decide a token's
  importance by itself.
- Cross-layer-KV-sharing layers are **kept** in the vote by default. An
  earlier version dropped them on the premise that their K is a duplicate;
  that was wrong. `initialize_kv_cache_tensors` aliases such a layer's
  `attn.kv_cache` to its target's tensor, so the K read back is the target's
  *real* K while the Q is the layer's own — a distinct distribution, not a
  duplicate vote. On Gemma-4-E2B (20 of 35 layers KV-shared, matching the
  technical report's stated ratio) dropping them cut `global_only` from 7
  voting layers to 3. `SpecConfig.drop_kv_shared_layers` re-enables the drop
  for a caller whose K read-back does not reproduce that aliasing.

Every one of these is a **provable no-op on a uniform-attention model**:
an unsupplied geometry keeps `1/sqrt(head_dim)` with no cap, and a Llama-shaped
geometry reports all-`full_attention`, all-False, no softcap. The published
Llama rows above are reproduced unchanged, which
`test_layer_geometry_is_a_provable_noop_for_uniform_models` checks directly
rather than asserting.

**The multi-group problem is confirmed on real hardware, not just traced.**
A Gemma-4-E2B speculator raised, out of `kv_cache_utils._find_kv_split_dim`:

```
KV cache shape (138140, 2, 32, 1, 256) does not match TRITON_ATTN's declared
shape (138140, 2, 16, 1, 256) for block_size=16, num_kv_heads=1, head_size=256
```

The configured block size is 16; the allocated one for that layer is 32.
E2B's sliding layers are `head_dim=256, num_kv_heads=1` — a 16KiB page against
the global layers' 32KiB — and `unify_kv_cache_spec_page_size` equalises page
sizes by *multiplying the smaller-page layers' block_size*. The result is two
KV cache groups with different block sizes, which breaks every site that
assumes one global `block_size` or reads `block_table[0]`.

The speculator engine now forces `disable_hybrid_kv_cache_manager=True`
(`proposer.py`), collapsing the specs into a single group; K read-back takes
its block size from that group's own spec and raises with an actionable
message if the group count is ever not 1. Both are no-ops on a
uniform-attention model.

**What is NOT ready — do not run a `SPARSE-k*` row on Gemma 4 yet.**
`sparse_target_runner.py` writes one gathered `block_table` into *every*
layer's attention metadata, and `speculator_worker.py` reads
`input_batch.block_table[0]`. Both assume a single KV-cache group. An
interleaved model has two or more, with *different block sizes*, and neither
site raises — they produce garbage K and garbage attention silently. See
`speculator_worker.py`'s "Known risk areas" #1 for the traced chain through
`get_kv_cache_groups` and for the `--disable-hybrid-kv-cache-manager` escape
hatch that restores the single-group invariant. A startup assertion for that is
the next piece of work, not something already in place.

Two further caveats worth knowing before reading any Gemma 4 number:

- **Sliding layers must be excluded from the gather entirely, not patched.**
  The gather *compacts* the KV view, and a sliding-window kernel decides window
  membership from a key's index within `seqused_k` — so compaction shifts every
  key's apparent distance from the query. The contiguous force-kept tail that
  fixes this for causal masking cannot fix it here, because a window must be
  contiguous in *true* positions and a top-k block selection is not.
- **The headroom is 5–6× smaller than on Llama.** The mechanism saves attention
  compute by restricting how much resident KV each query reads; on Gemma 4 most
  layers already read only 512–1024 tokens. The saving lives on the ~1/6 global
  layers, so the keep-rate/accuracy curves above do not transfer, and
  `SPECULATION_ECONOMICS.md`'s win condition needs both of its terms re-derived.

**The gate: `diagnose_sliding_window_votes.py`.** Reports what fraction of
winning (layer, head) votes came from a sliding layer scoring a position it
could never attend to, split between the positions the selection KEPT and a
random sample of those it pruned away. The comparison is the evidence — a bare
rate means little. Migrated here from
`../spec_prefill/verify_sliding_window_hypothesis.py` (now marked superseded),
and more trustworthy for three reasons: it shares production's aggregation via
`aggregate_attention_score`'s `winning_layers` out-list instead of copying the
softmax→pool→max chain; it reads a real engine's KV cache rather than a dummy
one where 20 of E2B's 35 KV-shared layers would score uninitialized memory; and
it uses each layer's own window instead of inferring "sliding" from a head-dim
proxy that reports 0% on any uniform-head-dim checkpoint.

```bash
python3 diagnose_sliding_window_votes.py --speculator-model "$GEMMA4_E2B_MODEL_PATH" --keep-percentage 0.3
```

Run it again with `--score-layers global_only`: that should drive the rate to
0% by construction, which is the check that the harness measures what it
claims before its answer is believed.

**Measured (Gemma-4-E2B-it, `scbench_kv`, keep=0.3, 10 conversations, all
context-truncated):**

| mode | phantom rate | random-winner null | excess |
|---|---:|---:|---:|
| unmasked (default) — KEPT | 93.3% | 79.0% | **+14.3** |
| unmasked (default) — PRUNED-AWAY | 86.1% | 80.0% | +6.1 |
| `--score-layers global_only` | 0.0% | — | 0 |
| `--mask-sliding-window` | ~0.0% | — | ~0 |

The null matters: 28 of 35 layers are sliding, so ~80% of wins land on one by
composition alone. The finding is the **excess** — sliding layers win more
than their share, and more than twice as much on the positions selection keeps
as on those it discards. Both fixes zero the metric by construction, so the
gate cannot rank them; that needs grading (below).

### Grading the three modes

`SPARSE-k20-g32-{unmasked,global,masked}` — the head-to-head, selectable as
`--exp scoremode`. On the **SPARSE architecture**, this pipeline's actual
contribution and what every published row uses, at the same `k20-g32` probe
point as the existing scoring variants (the low-keep corner is where scorers
are distinguishable at all — at k80 almost nothing is pruned and every scorer
looks alike).

```bash
python3 predict_scbench.py --exp scoremode --scbench-config scbench_kv --target-tensor-parallel-size 2 --speculator-device cuda:2 --speculator-gpu-memory-utilization 0.5 --target-prefill-chunk-tokens 32768 --scorer-prefill-chunk-tokens 32768
```

`--target-tensor-parallel-size` shards the target across GPUs; the speculator
is always TP=1. Ranks take the first N visible devices, so `--speculator-device`
must be N or higher — sharing is warned about, not refused, since a small
scorer sharing a card has always been legitimate here. Sizing on 80GB cards:

| target | weights (bf16) | TP | GPUs total |
|---|---:|---:|---:|
| Gemma-4-26B-A4B | ~49 GB | 1 | 2 |
| Gemma-4-31B | ~62 GB | **2** | **3** |

At TP=1 a 31B target leaves ~4GB for KV and activations at 0.85 — not enough
for a long context, and less than it looks because the sparse path retains KV
outside the sliding window.

**TP needs `TORCHDYNAMO_DISABLE=1`** (pinned in `.env_exports.sh`, alongside a
`TORCHINDUCTOR_COMPILE_THREADS=1` that is kept but is NOT sufficient on its
own -- `AsyncCompile.wakeup()` calls `use_process_pool()`, which spawns the
pool as part of deciding whether to use one, so the thread-count knob never
gets a chance to prevent the spawn).
Without it a TP run dies during `profile_run` with `AssertionError: daemonic
processes are not allowed to have children` — vLLM's `MultiprocExecutor`
spawns its workers as daemonic processes, and inductor then tries to start its
own compile-worker pool inside one. The trigger is
`VocabParallelEmbedding.forward`'s `@torch.compile`d
`get_masked_input_and_mask`, which runs only when `tp_size > 1`. Note that
`enforce_eager=True` does not prevent this: it disables vLLM's compilation and
CUDA graphs, not a bare `@torch.compile` decorator inside a layer.

```bash
python3 compare_ceiling.py --config scbench_kv results/SPARSE-k20-g32-unmasked_predictions.jsonl results/SPARSE-k20-g32-global_predictions.jsonl results/SPARSE-k20-g32-masked_predictions.jsonl
```

**Two Gemma-4 blockers had to be closed for SPARSE to be correct here**, and
both are now in place:

- **The gathered layers must share one KV cache group** — but only they.
  The hybrid KV cache manager stays **enabled** on the target, so sliding
  layers keep their own group and their own windowed allocation.
  `sparse_target_runner._gatherable_group_block_size` resolves the block table
  and block size from whichever group holds the full-attention layers, and
  raises if they straddle groups.

  An earlier version forced `disable_hybrid_kv_cache_manager=True` instead.
  That guaranteed one group, but budgeted every sliding layer for the full
  context it can never read: on Gemma-4-31B, 55.03 GiB of KV needed against
  34.05 GiB available, so the engine would not start. The speculator engine
  still sets the flag — `unmasked` scoring reads `Q·K` over the whole context
  for *every* layer, so dropping the sliding layers' out-of-window KV there
  would feed the control row recycled blocks.

  Note the block size a group actually uses can differ from the configured
  one: `unify_kv_cache_spec_page_size` equalises page sizes by multiplying the
  smaller-page layers' `block_size`. Since the gather is block-granular, the
  runner logs the block size it actually used, which is the real granularity
  regardless of what the row name says.
- **Sliding-window layers are excluded from the gather.** Correctness, not
  tuning: the gather compacts the KV view, and a sliding-window kernel reads
  window membership from a key's index within `seqused_k`, so a compacted view
  masks the wrong keys. The contiguous force-kept tail that fixes the
  analogous causal-masking problem cannot help — a window must be contiguous
  in *true* positions, which a top-k block selection is not.

Nothing is lost by that exclusion: a sliding layer already reads at most its
own window, so it never had long-context KV traffic to save. On Gemma-4-E2B
that leaves the gather operating on 7 of 35 layers — the same 1-in-6 figure
the economics reach from the compute side.

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
| `test_vllm_patch.py` | CPU-only unit tests — 154/154 passing |
| `validate_proposer.py` | GPU-node validation: persistent speculator engine, cross-turn KV read-back |
| `validate_runner_integration.py` | GPU-node validation: `worker_cls` wiring + multi-turn RoPE position-override correctness |
| `validate_resumable_session.py` | GPU-node validation: target-side session persistence (TTFT evidence) |
| `validate_sparse_attention.py` | GPU-node validation: decode-step block-gather sparse attention (needle-in-haystack) |
| `diagnose_turn_index_gap.py` | Diagnoses the turn-0 accuracy dip: rendering-mismatch vs. truncation hypotheses |
| `diagnose_gold_survival.py` | Diagnoses the turn-0 accuracy dip: does the gold answer survive selection? |
| `diagnose_retrieval_heads.py` | The §1.3 gate: upper bound on retrieval-head filtering (heads picked using the gold position — cheating, so it is a ceiling), plus whether those heads are stable enough for a fixed head set to exist. `--layer-prefix-budgets` reuses the same fixed-head-set machinery as the cheap gate for the `EARLY-k*-g32-L<n>` family: gold survival when only the first n layers vote, every n in one pass |
| `diagnose_speculator_selection.py` | Decodes the speculator's actual selected text, to isolate selection-side vs. target-side bugs |
| `diagnose_target_gather_metadata.py` | Checks target-side gathered block count against an independently computed expectation |
| `diagnose_h1_metadata.py` | Dumps attention-metadata length fields to catch stale-field bugs (found `max_seq_len` staleness) |
| `sparse_decode_microbench.py` / `dense_context_sweep.py` / `stock_context_sweep.py` / `stock_vllm_control.py` / `gpu_vs_host_timing.py` / `ncu_kv_bytes_probe.py` | Decode-latency-vs-bandwidth-roofline microbenchmark family |
| `flops_model.py` | Analytic FLOP model (pure Python, no GPU needed) |
| `validate_flops_model.py` | GPU-node validation of the FLOP model against real profiler output |
| `datasets/prep_scbench.py` | Downloads `microsoft/SCBench`'s 3 MVP configs, writes `datasets/scbench_samples.jsonl` |
| `predict_scbench.py` | Runs the M000/ORACLE-k\*/SPARSE-k\*-g\*/EARLY-k\*-g32-L\<n> matrix, writes a per-turn predictions JSONL per experiment |
| `grade_scbench.py` | Scores a predictions file against `prep_scbench.py`'s samples, per-config metrics |
| `SPARSE-k20-g32-*` rows | Scoring-variant sweep (`--exp score`) — `score_aggregation`/`score_layers`, targeting the 17-point estimator gap the oracle measured |
| `compare_ceiling.py` | Compares predictions files on their common turns, with a cluster-bootstrap CI — for reading ORACLE-k\* against its SPARSE partner |
| `ACCURACY_IMPROVEMENTS.md` | Candidate accuracy improvements, gated on the oracle result and ranked by effort |
| `datasets/` | SCBench prep output (gitignored) |
| `results/` | Output directory (gitignored, empty in this checkout) |
