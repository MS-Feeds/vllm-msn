# Speculative Prefill (SpecPrefill) — Experiment Plan

Status: **blocked — SpecPrefill not implemented in this vLLM fork** (see
"Implementation status" below for the concrete porting checklist).

| | |
|---|---|
| **Target model** | Gemma-4-26B-A4B-it (thinking mode off, greedy decoding) |
| **Speculator** | Gemma-4-E2B-it |
| **Precision** | BF16 |
| **Infra** | vLLM (this fork, V1 engine) |
| **Benchmark** | LongBench v2 — "short" questions (<32k words) subset |
| **Hardware** | 2x A100, 160GB total GPU HBM |
| **ETA** | 7 days |

Reference the paper in this directory and linked at the bottom of this document for details

---

## Motivation

SpecPrefill is a training-free token preselection mechanism that accelerates TTFT
during the prefill stage by using a smaller "speculator" model to select only the
most important tokens for the base model, reducing its prefill cost.

Because self-attention compute grows as O(n²), TTFT increases superlinearly with
sequence length. SpecPrefill should therefore meaningfully improve QPS on long
inputs.

This protocol evaluates SpecPrefill specifically on long inputs with reasoning-based
short output, aiming to answer:

1. How low can token keep rates go before performance degrades?
2. Can SpecPrefill match or surpass baseline accuracy?
3. How much can SpecPrefill improve QPS?

---

## Algorithm reference

A functional reference implementation has been cloned locally to
[`speculative_prefill/`](speculative_prefill/) (`github.com/Jingyu6/speculative_prefill`,
the ICML 2025 paper's own repo — see References). The mechanism, in its own terms:

1. **Draft scoring**: the speculator model prefills the full prompt, then runs a few
   *lookahead decode* steps, capturing its post-RoPE query vectors from attention
   layers during those steps.
2. **Importance scoring**: `softmax(Q_lookahead @ K_prompt^T / sqrt(d))`, average-pooled
   per chunk, max-reduced over layers/heads, mean-reduced over lookahead steps —
   producing one importance score per prompt chunk.
3. **Chunk selection**: the prompt is grouped into fixed-size chunks; the top-k% of
   chunks by importance score are kept.
4. **Sparse prefill**: the target model prefills only the selected tokens. Original
   position IDs are preserved via RoPE patching (RoPE is relative — `Q_m @ K_p^T`
   depends only on `m - p` — so selected tokens keep correct relative positions
   during subsequent decode) so quality is preserved despite dropped tokens.

The cloned repo exposes this as a config surface via
[`speculative_prefill/vllm_patch/config.py`](speculative_prefill/speculative_prefill/vllm_patch/config.py)'s
`SpecConfig` dataclass — this is the actual knob set this protocol's experiment
matrix is built from:

| Field | Meaning | Protocol value |
|---|---|---|
| `keep_strategy` | Selection strategy (only `"percentage"` is currently supported) | `percentage` |
| `keep_kwargs.chunk` | Whether to group tokens into chunks before scoring | `True` |
| `keep_kwargs.chunk_size` | Chunk size for scoring/selection | `64` (see note below) |
| `keep_kwargs.percentage` | Fraction of chunks to keep | swept: 0.1 / 0.3 / 0.5 / 0.7 / 0.9 |
| `look_ahead_cnt` | Number of lookahead decode steps used for scoring | `8` |
| `pool_kernel_size` | Pooling kernel size for chunk score aggregation | inherited from repo's `_full` configs (13) unless retuned |

---

## Implementation status

SpecPrefill does **not** exist natively in vLLM (confirmed: `vllm-project/vllm#39060`
is an open, unimplemented feature request) or in this fork. The cloned reference repo
is real and functional — its `experiments/run_long_bench_70b.sh` and
`configs/config_p{1,3,5,7,9}_full_lah8.yaml` are exactly what this protocol's keep-rate
sweep and look-ahead=8 setting are modeled on — but it is **not directly runnable
against this fork as-is**. Porting checklist:

1. **Engine mismatch.** The repo's `requirements.txt` pins `vllm==0.6.3.post1` (the
   old **v0 engine**). Its monkey-patch
   ([`speculative_prefill/vllm_patch/scheduler.py`](speculative_prefill/speculative_prefill/vllm_patch/scheduler.py))
   targets `vllm.core.scheduler.Scheduler` / `vllm.core.interfaces.AllocStatus` —
   modules that don't exist in this fork's vLLM (V1 engine; scheduling now lives
   under `vllm.v1.core.sched.scheduler`). This needs a port against V1's scheduler
   internals, not a version bump.
2. **Model support.** The repo's own README states: *"Currently only support Llama
   model as the base and speculator."* Gemma 4's MoE architecture and heterogeneous
   attention head dimensions (see `gemma4_moe_benchmarks/EXPERIMENT_PLAN.md`'s
   TRITON_ATTN note) mean the patch's attention-hooking code likely needs
   Gemma4-specific work before the speculator/target pairing here is viable.
3. **Benchmark mismatch.** The repo's `eval/long_bench/pred_vllm.py` only targets
   dataset `THUDM/LongBench` (**v1** — free-form QA/summarization with per-task
   metrics). This protocol calls for **LongBench v2** (multiple-choice format, with
   a length field used for the short/<32k-word filter) — a new prediction+eval
   script is needed; the v1 harness can't be reused as-is.
4. **Missing model path.** This directory's own `.env_exports.sh` (not the
   shared one in `gemma4_moe_benchmarks/`) has a commented-out
   `GEMMA4_E2B_MODEL_PATH` placeholder, but the checkpoint itself isn't
   downloaded yet — see `REPRODUCE.md` step 3 for the exact `hf download`
   command; uncomment and fill in the real snapshot path once it completes.

This plan's experiment matrix, dataset scoping, and success criteria below are ready
to execute once items 1–4 land.

---

## SpecPrefill settings

- BF16 precision
- Chunk-based attention scoring, chunk size **32** (protocol value — note the cloned
  repo's own configs default to `chunk_size: 32`;)
- Look-ahead count: **8**
- `enforce_eager=True`, `enable_chunked_prefill=False` — required by the reference
  implementation's own setup notes (printed at patch-apply time), not just the paper

---

## Experiment matrix

| ID | Label | Keep rate | Chunk size | Look-ahead | Based on (reference repo config) |
|---|---|---:|---:|---:|---|
| P001 | Baseline (no SpecPrefill) | 100% | — | — | `experiments/run_long_bench_70b.sh` baseline branch |
| P002 | Keep 10% | 10% | 64 | 8 | `config_p1_full_lah8.yaml` |
| P003 | Keep 30% | 30% | 64 | 8 | `config_p3_full_lah8.yaml` |
| P004 | Keep 50% | 50% | 64 | 8 | `config_p5_full_lah8.yaml` |
| P005 | Keep 70% | 70% | 64 | 8 | `config_p7_full_lah8.yaml` |
| P006 | Keep 90% | 90% | 64 | 8 | `config_p9_full_lah8.yaml` |

Metrics captured per run: accuracy (LongBench v2 scoring), QPS, TTFT.

---

## Benchmark

**LongBench v2**, restricted for this protocol round to "short" questions
(<32k words). Dataset prep and grader do not exist in this repo yet (Implementation
status #3) — future work, analogous in shape to
`evaluation_pipeline/datasets/prep_aime.py`.

---

## Success criteria

- Accuracy drop ≤5% compared to baseline (P001).
- Report QPS improvement over baseline for each keep-rate row (no fixed pass/fail
  threshold specified — the sweep itself is the signal for "how low can keep rate go").

---

## Resource requirements

- Gemma-26B and E2B: total 160GB GPU HBM
- 2x A100

**ETA**: 7 days

---

## References

- [SpecPrefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation](https://arxiv.org/abs/2502.02789) (ICML 2025)
- [vllm-project/vllm#39060](https://github.com/vllm-project/vllm/issues/39060) — open feature request tracking native vLLM support
- [Jingyu6/speculative_prefill](https://github.com/Jingyu6/speculative_prefill) — reference implementation; cloned locally at [`speculative_prefill/`](speculative_prefill/)

---

## Files in this directory

| File | Purpose |
|---|---|
| `EXPERIMENT_PLAN.md` | This file |
| `README.md` | Overview / index |
| `REPRODUCE.md` | Environment setup + reproduction steps |
| `.env_exports.sh` | Local env config (conda activation, `HF_TOKEN`, `GEMMA4_MODEL_PATH`, `GEMMA4_E2B_MODEL_PATH`) — separate from the shared one in `../gemma4_moe_benchmarks/`, see `REPRODUCE.md` step 1 |
| `vllm_patch/` | The Algorithm 1 implementation (`scoring.py`, `proposer.py`, `kv_cache_utils.py`, `prefill_split.py`, `pruning_registry.py`, `pruner.py`, `model_runner.py`, `worker.py`, `config.py`) — see its own `__init__.py` module map |
| `test_vllm_patch.py` | Unit tests for the engine-agnostic pieces of `vllm_patch/` (no GPU needed) |
| `validate_proposer.py` | GPU-node validation: speculator loading + attention hook, per-layer `head_dim` check |
| `validate_runner_integration.py` | GPU-node validation: `worker_cls` wiring + the position-override correctness check |
| `speculative_prefill/` | Cloned reference implementation (gitignored — has its own `.git`) |
| `datasets/` | LongBench v2 prep output (empty — prep script is future work) |
| `results/` | Output directory (gitignored) |
