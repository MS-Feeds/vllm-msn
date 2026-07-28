# Speculative Prefill (SpecPrefill) — Experiment Plan (Qwen3)

Status: **ported from `../spec_prefill/`'s validated Gemma4 pipeline —
`vllm_patch/`'s Qwen3 port is code-complete but unvalidated on real
hardware** (written on a machine with no GPU, same as the pipeline it was
ported from). See "Implementation status" below for what's inherited
unchanged vs. what was rewritten and still needs fresh confirmation, and
`REPRODUCE.md` step 5 for the validation scripts to run first on the GPU
node.

| | |
|---|---|
| **Target model** | Qwen3-8B (thinking mode off, greedy decoding) |
| **Speculator** | Qwen3-1.7B |
| **Precision** | BF16 |
| **Infra** | vLLM (this fork, V1 engine) |
| **Benchmark** | LongBench v2 — "short" questions (<32k words) subset |
| **Hardware** | TBD — see "Resource requirements" below |
| **ETA** | TBD |

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

Running the same protocol against a second model family (Qwen3, dense,
uniform attention) alongside `../spec_prefill/`'s Gemma4 (MoE, heterogeneous
attention) also answers a question that pipeline alone can't: how much of
SpecPrefill's behavior is architecture-dependent vs. general.

---

## Algorithm reference

A functional reference implementation is cloned locally at
`../spec_prefill/speculative_prefill/` (`github.com/Jingyu6/speculative_prefill`,
the ICML 2025 paper's own repo — see References; not duplicated into this
directory, see `REPRODUCE.md` step 2). The mechanism, in its own terms:

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
[`speculative_prefill/vllm_patch/config.py`](../spec_prefill/speculative_prefill/speculative_prefill/vllm_patch/config.py)'s
`SpecConfig` dataclass — this is the actual knob set this protocol's experiment
matrix is built from:

| Field | Meaning | Protocol value |
|---|---|---|
| `keep_strategy` | Selection strategy (only `"percentage"` is currently supported) | `percentage` |
| `keep_kwargs.chunk` | Whether to group tokens into chunks before scoring | `True` |
| `keep_kwargs.chunk_size` | Chunk size for scoring/selection | `64` |
| `keep_kwargs.percentage` | Fraction of chunks to keep | swept: 0.1 / 0.3 / 0.5 / 0.7 / 0.9 |
| `look_ahead_cnt` | Number of lookahead decode steps used for scoring | `8` |
| `pool_kernel_size` | Pooling kernel size for chunk score aggregation | `13`, matching the reference repo's own cited configs |

These are SpecPrefill algorithm hyperparameters (inherited unchanged from
`../spec_prefill/`'s matrix, itself modeled on the reference repo's own
`config_p{1,3,5,7,9}_full_lah8.yaml`), not facts about Gemma4 or Qwen3 — no
re-derivation needed for the model swap.

---

## Implementation status

This pipeline is a port of `../spec_prefill/`'s SpecPrefill implementation
(built and real-hardware-validated for `vllm_patch/proposer.py`'s speculator
loading against Gemma-4-E2B-it — see that directory's own
`EXPERIMENT_PLAN.md`), targeting Qwen3-8B/Qwen3-1.7B instead of
Gemma-4-26B-A4B-it/Gemma-4-E2B-it. What changed and why:

1. **Carried over unchanged (architecture-agnostic).** `vllm_patch/scoring.py`,
   `kv_cache_utils.py`, `prefill_split.py`, `pruning_registry.py`, `pruner.py`,
   `model_runner.py`, `worker.py` derive shapes from tensors/config at
   runtime and never reference a specific model architecture — no port work
   needed.
2. **Rewritten: `vllm_patch/proposer.py`'s query-capture hook and
   attention-layer discovery.** The Gemma4 version's hook was a faithful
   copy of `Gemma4Attention.forward`'s body, including branching on
   Gemma4-only concepts (`is_kv_shared_layer`, `use_k_eq_v`, a separate
   `v_norm`) that don't exist on `Qwen3Attention` — rewritten to mirror
   `Qwen3Attention.forward` instead (`vllm/model_executor/models/
   qwen3.py:145-162`), which is simpler (no branching, RMSNorm on q/k only).
   Layer discovery no longer needs Gemma-4-E2B-it's multimodal-wrapper
   unwrap (`Gemma4ForConditionalGeneration.get_language_model()`) since
   Qwen3-8B/1.7B load as plain, non-multimodal `Qwen3ForCausalLM`. See
   `proposer.py`'s own docstrings for the full before/after.
3. **Open question, to be answered empirically (not assumed) by
   `validate_proposer.py`'s Step A**: does Qwen3-1.7B have a uniform
   head_dim/num_kv_heads across layers? Expected yes — `Qwen3Attention.
   __init__` computes a single value per instance from config, with no
   per-layer heterogeneity possible by construction, unlike Gemma-4-E2B-it
   (confirmed heterogeneous: head_dim=256 on 28 sliding-attention layers,
   head_dim=512 on 7 full-attention layers, 35 layers total,
   num_kv_heads=1 throughout) — but not yet confirmed for Qwen3-1.7B on
   real hardware here.
4. **Re-derive, not assume: the chunked-prefill deviation.** The Gemma4
   pipeline settled on leaving `enable_chunked_prefill` at its
   model-supported default (on), rather than the paper's own stated
   `enable_chunked_prefill=False`, because Gemma-4-26B-A4B-it specifically
   requires `max_num_batched_tokens >= max_model_len` (262144) to disable
   chunking, and vLLM itself warns that model doesn't officially support
   disabling it. **Neither constraint is known to apply to Qwen3-8B**
   (dense, non-multimodal) — this needs to be checked fresh once
   `validate_runner_integration.py` runs against Qwen3-8B, not assumed to
   match Gemma4's conclusion. See that script's module docstring for the
   fuller history of what was tried and why.
5. **Benchmark harness — unchanged.** `datasets/prep_longbench_v2.py` and
   `grade_longbench_v2.py` have zero model-specific code and are reused
   verbatim. `predict_longbench_v2.py` needed only its two model-path env
   var names updated (`GEMMA4_MODEL_PATH`/`GEMMA4_E2B_MODEL_PATH` →
   `QWEN3_MODEL_PATH`/`QWEN3_1_7B_MODEL_PATH`); its engine-driving loop,
   TTFT/output-length metrics, and CSV/JSONL output are architecture-agnostic.
6. **Not ported**: the Gemma4 pipeline's diagnostic/bug-investigation
   scripts (`diag_worker.py`, `repro_chunked_prefill_bug.py`,
   `verify_sliding_window_hypothesis.py`). Both bugs they investigate are
   Gemma4-specific in their preconditions — a multimodal chunked-prefill
   token floor, and a sliding-window-vs-full-attention scoring-corruption
   hypothesis premised on Gemma4's heterogeneous attention-layer split —
   and neither precondition is expected to occur for Qwen3-8B/1.7B (dense,
   non-multimodal, uniform full attention across all layers, per
   `vllm/model_executor/models/qwen3.py`). If an analogous issue surfaces
   during Qwen3 validation, it should be re-diagnosed from scratch rather
   than assumed to be the same root cause.

This plan's experiment matrix and success criteria below are ready to
execute once the full pipeline (`validate_proposer.py`,
`validate_runner_integration.py` Step B/B2, `predict_longbench_v2.py`) is
validated end-to-end on real hardware for Qwen3 — see `REPRODUCE.md` steps
5–6.

---

## SpecPrefill settings

- BF16 precision
- Chunk-based attention scoring, chunk size **64**, `pool_kernel_size` **13**
  — matching the "Algorithm reference" table above and inherited unchanged
  from `../spec_prefill/`'s matrix (these are algorithm hyperparameters, not
  model-architecture facts).
- Look-ahead count: **8**
- `enforce_eager=True`. **`enable_chunked_prefill` is left at the model's
  own supported default (on) for now, matching `../spec_prefill/`'s
  approach as a starting point — NOT independently re-confirmed for
  Qwen3-8B.** The Gemma4 pipeline's specific reasons for this deviation
  (a multimodal-driven `max_num_batched_tokens` floor, and vLLM's explicit
  warning against disabling chunked prefill for that model) are Gemma4-only
  facts that may not apply to Qwen3-8B at all — see "Implementation
  status" #4 above and `validate_runner_integration.py`'s module docstring.
  Re-check this setting once Qwen3-8B has been validated, rather than
  assuming it should stay this way.

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

`predict_longbench_v2.py` additionally supports a standalone `SPEC`
experiment (Qwen3-1.7B generating alone, no pruning, no target model) as a
reference point for how much of the pruned target's accuracy is just
tracking the speculator's own ceiling — not part of the official matrix
above.

---

## Benchmark

**LongBench v2**, restricted for this protocol round to "short" questions
(<32k words). Dataset prep (`datasets/prep_longbench_v2.py`), prediction
generation (`predict_longbench_v2.py`), and grading (`grade_longbench_v2.py`)
are all ported unchanged (dataset/harness side) or with only env-var renames
(prediction driver) from `../spec_prefill/`.

---

## Success criteria

- Accuracy drop ≤5% compared to baseline (P001).
- Report QPS improvement over baseline for each keep-rate row (no fixed pass/fail
  threshold specified — the sweep itself is the signal for "how low can keep rate go").

---

## Resource requirements

**TBD — not yet derived.** Qwen3-8B (target) + Qwen3-1.7B (speculator) are
both dense models, combined roughly 10B parameters — far smaller than the
sibling Gemma4 pipeline's Gemma-4-26B-A4B-it (MoE) + Gemma-4-E2B-it, which
needed 2x A100 (160GB total GPU HBM). Expect a single GPU to be sufficient
here, but this should be confirmed empirically (via `REPRODUCE.md` step 5's
validation scripts) rather than assumed from parameter count alone.

**ETA**: TBD

---

## References

- [SpecPrefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation](https://arxiv.org/abs/2502.02789) (ICML 2025)
- [vllm-project/vllm#39060](https://github.com/vllm-project/vllm/issues/39060) — open feature request tracking native vLLM support
- [Jingyu6/speculative_prefill](https://github.com/Jingyu6/speculative_prefill) — reference implementation; cloned locally at [`../spec_prefill/speculative_prefill/`](../spec_prefill/speculative_prefill/)
- `../spec_prefill/EXPERIMENT_PLAN.md` — the Gemma4 pipeline this one was ported from

---

## Files in this directory

| File | Purpose |
|---|---|
| `EXPERIMENT_PLAN.md` | This file |
| `README.md` | Overview / index |
| `REPRODUCE.md` | Environment setup + reproduction steps |
| `.env_exports.sh` | Local env config (conda activation, `HF_TOKEN`, `QWEN3_MODEL_PATH`, `QWEN3_1_7B_MODEL_PATH`) — separate from the shared one in `../gemma4_moe_benchmarks/` and from `../spec_prefill/`'s own, see `REPRODUCE.md` step 1 |
| `vllm_patch/` | The Algorithm 1 implementation (`scoring.py`, `proposer.py`, `kv_cache_utils.py`, `prefill_split.py`, `pruning_registry.py`, `pruner.py`, `model_runner.py`, `worker.py`, `config.py`) — see its own `__init__.py` module map |
| `test_vllm_patch.py` | Unit tests for the engine-agnostic pieces of `vllm_patch/` (no GPU needed) |
| `validate_proposer.py` | GPU-node validation: speculator loading + attention hook, per-layer `head_dim` check |
| `validate_runner_integration.py` | GPU-node validation: `worker_cls` wiring + the position-override correctness check |
| `datasets/prep_longbench_v2.py` | Downloads `THUDM/LongBench-v2`, filters to the "short" (<32k word) subset, writes `datasets/longbench_v2_samples.jsonl` |
| `predict_longbench_v2.py` | Runs the P001–P006 keep-rate sweep (with/without SpecPrefill pruning) against those samples, writes a predictions JSONL per experiment |
| `grade_longbench_v2.py` | Scores a predictions file against `prep_longbench_v2.py`'s samples; writes `result.json` + prints a Markdown summary |
| `datasets/` | LongBench v2 prep output (`longbench_v2_samples.jsonl`, gitignored raw `.cache/`) |
| `results/` | Output directory (gitignored) |
