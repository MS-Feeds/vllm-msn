# Speculative Prefill (SpecPrefill) — Experiment Plan (Qwen3-Coder)

Status: **ported from `../spec_prefill_qwen/`'s Qwen3 pipeline (itself
ported from `../spec_prefill/`'s validated Gemma4 pipeline) —
`vllm_patch/`'s Qwen3-Coder port is code-complete but unvalidated on real
hardware** (written on a machine with no GPU, same as the pipelines it was
ported from). Unlike the dense Qwen3-8B/1.7B port, target and speculator
here are MoE (`Qwen3MoeForCausalLM`) -- confirmed this required zero
`vllm_patch/` logic changes (see "Implementation status" below), only
resource-planning changes (the target no longer fits on one GPU). See
"Implementation status" below for what's inherited unchanged vs. what still
needs fresh confirmation, and `REPRODUCE.md` step 5 for the validation
scripts to run first on the GPU node.

| | |
|---|---|
| **Target model** | Qwen3-Coder-480B-A35B-Instruct, MoE (thinking mode off, greedy decoding) -- 62 layers, 96 Q / 8 KV heads, 160 experts (8 active), native context 262144 |
| **Speculator** | Qwen3-Coder-30B-A3B-Instruct, MoE |
| **Precision** | BF16, or 4-bit AWQ for easier serving (confirmed available: `QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ`, ~236GB) |
| **Infra** | vLLM (this fork, V1 engine), target requires `tensor_parallel_size > 1`. Current default: `tensor_parallel_size=4` (see "Resource requirements" for why, not the AWQ checkpoint's own tested `tensor_parallel_size=8` + `enable_expert_parallel=True`) |
| **Benchmark** | LongBench v2 — "short" questions (<32k words) subset |
| **Hardware** | See "Resource requirements" below — 8x A100 80GB is a confirmed-workable target size at 4-bit AWQ |
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

Running the same protocol against additional model families/scales --
`../spec_prefill_llama/`'s Llama (dense), `../spec_prefill_qwen/`'s dense
Qwen3-8B/1.7B, `../spec_prefill/`'s Gemma4 (MoE, heterogeneous attention),
and this port's Qwen3-Coder-480B-A35B/30B-A3B (MoE, uniform attention, but
at a scale none of the others reach) -- answers a question no single
pipeline can: how much of SpecPrefill's behavior is architecture- and
scale-dependent vs. general.

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

This pipeline is a port of `../spec_prefill_qwen/`'s dense-Qwen3
implementation (itself built and real-hardware-validated for
`vllm_patch/proposer.py`'s speculator loading against Gemma-4-E2B-it, see
`../spec_prefill/EXPERIMENT_PLAN.md`), targeting
Qwen3-Coder-480B-A35B-Instruct/Qwen3-Coder-30B-A3B-Instruct instead of
Qwen3-8B/Qwen3-1.7B. What changed and why:

1. **Carried over unchanged, including the dense-Qwen3 port's query-capture
   hook and attention-layer discovery.** Confirmed directly against this
   fork's `vllm/model_executor/models/qwen3_moe.py`:
   `Qwen3MoeAttention.forward` (lines 343-361) is byte-identical to dense
   `Qwen3Attention.forward` (same `qkv_proj` split, same per-head
   `q_norm`/`k_norm`, same `rotary_emb` call), and `Qwen3MoeModel.
   layers[i].self_attn` matches the same structure the dense port's
   `_find_qwen3_attention_layers` already assumed. Unlike the dense-Qwen3
   port's own rewrite of the Gemma4 pipeline's hooks, **this MoE port
   required zero logic changes** to `vllm_patch/proposer.py` -- only
   docstring updates citing the MoE class names/line numbers. Also
   unchanged: `vllm_patch/scoring.py`, `kv_cache_utils.py`,
   `prefill_split.py`, `pruning_registry.py`, `pruner.py`, `model_runner.py`,
   `worker.py` (all derive shapes from tensors/config at runtime and never
   reference a specific model architecture).
2. **New resource-planning requirement the dense port never had: tensor
   parallelism for the target.** Qwen3-Coder-480B-A35B does not fit on a
   single GPU. `../rlm_specprefill/target_stage/vllm_offline_engine.py`
   gained a `tensor_parallel_size` parameter for this; `validate_runner_
   integration.py` here gained a matching `--target-tensor-parallel-size`
   flag. `vllm_patch/worker.py`/`model_runner.py` needed no changes for
   this -- they operate per-worker-process, and vLLM's own executor spawns
   one `SpecPrefillWorker` per TP rank automatically.
3. **Also new: the v1/v2 model-runner workaround is a no-op here, not
   load-bearing.** The dense-Qwen3 port needed `VLLM_USE_V2_MODEL_RUNNER=0`
   because this fork's `use_v2_model_runner` auto-selects a newer runner
   for dense, non-MoE architectures in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`
   (`{"Qwen3ForCausalLM"}`). Both models in this port are MoE
   (`Qwen3MoeForCausalLM`), which `use_v2_model_runner` excludes
   unconditionally (`not model_config.is_moe`) -- same situation the
   Gemma4/Llama ports were already in. The env var is kept anyway as a
   harmless explicit pin.
4. **Open question, to be answered empirically (not assumed) by
   `validate_proposer.py`'s Step A**: does Qwen3-Coder-30B-A3B have a
   uniform head_dim/num_kv_heads across layers? Expected yes --
   `Qwen3MoeAttention.__init__` computes a single value per instance from
   config, with no per-layer heterogeneity possible by construction (MoE
   routing affects only each layer's `mlp`, not `self_attn`), same as the
   dense port's own (still only empirically, not independently, confirmed)
   expectation for Qwen3-1.7B -- but not yet confirmed for Qwen3-Coder-30B-
   A3B on real hardware here either.
5. **Re-derive, not assume: the chunked-prefill deviation.** The Gemma4
   pipeline settled on leaving `enable_chunked_prefill` at its
   model-supported default (on), rather than the paper's own stated
   `enable_chunked_prefill=False`, because Gemma-4-26B-A4B-it specifically
   requires `max_num_batched_tokens >= max_model_len` (262144) to disable
   chunking, and vLLM itself warns that model doesn't officially support
   disabling it. The dense-Qwen3 port reasoned this likely didn't apply to
   Qwen3-8B (dense, non-multimodal, smaller native context) but never
   confirmed it. **For THIS port, the opposite prior may hold**:
   Qwen3-Coder-480B-A35B is MoE, like Gemma-4-26B-A4B-it, not dense like
   Qwen3-8B -- so Gemma4's reasoning may carry over here more than it did
   to the dense port. Needs to be checked fresh once
   `validate_runner_integration.py` runs against the real checkpoint, not
   assumed either way. See that script's module docstring for the fuller
   history of what was tried and why.
6. **Benchmark harness — unchanged.** `datasets/prep_longbench_v2.py` and
   `grade_longbench_v2.py` have zero model-specific code and are reused
   verbatim. `predict_longbench_v2.py` needed only its two model-path env
   var names updated (`QWEN3_MODEL_PATH`/`QWEN3_1_7B_MODEL_PATH` →
   `QWEN3_CODER_480B_MODEL_PATH`/`QWEN3_CODER_30B_MODEL_PATH`); its
   engine-driving loop, TTFT/output-length metrics, and CSV/JSONL output
   are architecture-agnostic.
7. **Not ported** (inherited decision from the dense-Qwen3 port): the
   Gemma4 pipeline's diagnostic/bug-investigation scripts (`diag_worker.py`,
   `repro_chunked_prefill_bug.py`, `verify_sliding_window_hypothesis.py`).
   The sliding-window-vs-full-attention hypothesis is premised on Gemma4's
   heterogeneous attention-layer split, which doesn't apply to Qwen3-Coder
   (uniform full attention across all layers, confirmed via
   `vllm/model_executor/models/qwen3_moe.py`) -- MoE routing in the `mlp`
   doesn't reintroduce that heterogeneity. If an analogous issue surfaces
   during Qwen3-Coder validation, it should be re-diagnosed from scratch
   rather than assumed to be the same root cause.
8. **Inherited, unresolved caveat from the dense-Qwen3 port**:
   `worker.py`/`model_runner.py` still carry `TEMPORARY diagnostic` print
   statements from an intermittent (~1-in-4) race condition in
   `register_prune_record`'s cross-process registration, never confirmed
   fixed there. Not being fixed in this port either -- re-check before
   trusting Arm B/C-equivalent runs here.

This plan's experiment matrix and success criteria below are ready to
execute once the full pipeline (`validate_proposer.py`,
`validate_runner_integration.py` Step B/B2, `predict_longbench_v2.py`) is
validated end-to-end on real hardware for Qwen3-Coder — see `REPRODUCE.md`
steps 5–6.

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
  Qwen3-Coder-480B-A35B.** The Gemma4 pipeline's specific reasons for this
  deviation (a multimodal-driven `max_num_batched_tokens` floor, and
  vLLM's explicit warning against disabling chunked prefill for that
  model) may apply MORE here than to the dense Qwen3-8B port, since this
  target is also MoE — see "Implementation status" #5 above and
  `validate_runner_integration.py`'s module docstring. Re-check this
  setting once Qwen3-Coder-480B-A35B has been validated, rather than
  assuming it should stay this way.
- **Target requires `tensor_parallel_size > 1`** (new for this port —
  Qwen3-Coder-480B-A35B does not fit on one GPU); the speculator
  (Qwen3-Coder-30B-A3B) remains `tensor_model_parallel_size=1`, standalone
  on its own GPU, same scope as every other port here.

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

`predict_longbench_v2.py` additionally supports a standalone `P000`
experiment (Qwen3-Coder-30B-A3B generating alone, no pruning, no target
model) as a reference point for how much of the pruned target's accuracy is
just tracking the speculator's own ceiling — not part of the official
matrix above.

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

Qualitatively very different from every other port in this repo.
Qwen3-Coder-480B-A35B-Instruct (target, ~480B total/~35B active params,
MoE) does **not** fit on a single GPU at BF16 (~960GB), or on the
2x-A100-class setup the sibling Gemma4 pipeline (Gemma-4-26B-A4B-it +
Gemma-4-E2B-it, 160GB total HBM) used.

**8x A100 80GB (640GB total HBM), the config actually targeted here, is
confirmed workable at 4-bit but not at BF16/FP8**:

| Precision | Total target weight size | Per-GPU at `tensor_parallel_size=8` |
|---|---:|---:|
| BF16 | ~960 GB | ~120 GB — does not fit |
| FP8 | ~480 GB | ~60 GB — tight, little KV-cache room |
| 4-bit AWQ | ~236 GB (confirmed: `QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ` repo size) | ~30 GB — comfortable |

For 4-bit: this fork's `FusedMoE` layer has a real AWQ-Marlin quant method
(`vllm/model_executor/layers/quantization/awq_marlin.py`'s
`AWQMarlinMoEMethod`), and `Qwen3MoeSparseMoeBlock` threads `quant_config`
straight through — no MoE-specific blocker for this architecture. vLLM
auto-detects the quantization method from the checkpoint's own
`config.json`, so no `quantization=` argument is needed; just point
`QWEN3_CODER_480B_MODEL_PATH` at the quantized checkpoint. **Avoid
pre-quantized bitsandbytes checkpoints** if you need `tensor_parallel_size
> 1` — this fork's bnb loader explicitly rejects TP>1 for pre-quantized
models (only pipeline parallelism is supported there).

**The `tensor_parallel_size=8` + `--enable-expert-parallel` requirement**:
the QuantTrio AWQ card documents that at 8-GPU tensor parallelism,
`--enable-expert-parallel` is REQUIRED or "the expert tensors cannot be
evenly split across tensor parallel ranks" (their wording) — confirmed a
real `LLM(...)`/`ParallelConfig` field in this fork. `runner/run_arm.py`,
`validate_runner_integration.py`, and the calibration scripts all gained a
`--target-enable-expert-parallel` flag for this (default off, so it
doesn't change behavior for checkpoints/TP sizes that don't need it).
96 Q-heads/8 KV-heads means `tensor_parallel_size` ∈ {1, 2, 4, 8} all
divide evenly for this architecture (verified against
`Qwen3MoeAttention.__init__`'s own head-count assertions) — 8 is what the
checkpoint's own card was tested against.

**The speculator/target GPU-placement conflict this creates, and the
current default**: Qwen3-Coder-30B-A3B-Instruct (speculator) is kept
**unquantized** by `vllm_patch/proposer.py`'s design
(`_create_speculator_vllm_config` clears `quant_config`) and needs its own
~60GB GPU. At `tensor_parallel_size=8` the target occupies *every* GPU,
leaving none free for the speculator — stacking it onto a GPU already
holding a ~30GB (4-bit) target shard would exceed 80GB.

**Current default: `tensor_parallel_size=4`**, not the checkpoint's own
tested 8 (still a valid divisor per the head-count check above) — target
uses 4 GPUs (~59GB/GPU at 4-bit, repo-size/4), speculator gets a dedicated
GPU among the remaining 4 (~60GB, comfortable), 3 GPUs spare. Set via
`TARGET_TENSOR_PARALLEL_SIZE=4` in `.env_exports.sh` (and re-exported by
`../rlm_specprefill/.env_exports_qwen_coder.sh`), which `runner/run_arm.py`,
`run_all_arms.py`, the calibration scripts, and `validate_runner_
integration.py` all read as their `--target-tensor-parallel-size` default
— no flag needs to be passed explicitly for this starting point.
`--target-enable-expert-parallel` is **not** known to be required at TP=4
(unlike the checkpoint's own tested TP=8, where it's REQUIRED) — leave it
off unless an "expert tensors don't split evenly" error appears.

To move to `tensor_parallel_size=8` (the checkpoint's own tested config,
using all 8 GPUs) later, the speculator's placement needs to be resolved
first — untested here, but two paths:
1. Quantize the speculator too, to fit alongside a target shard on one
   GPU — `proposer.py`'s `quant_config=None` only strips the field it
   would otherwise inherit from the *target's* `VllmConfig`, not
   necessarily anything the speculator's own `ModelConfig` would
   auto-detect from a quantized speculator checkpoint's own `config.json`
   — verify this is actually decoupled before relying on it.
2. Accept a 9-GPU-equivalent layout isn't possible on an 8-GPU node and
   stay at TP=4 (or another divisor <8) permanently for this hardware.

**ETA**: TBD

---

## References

- [SpecPrefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation](https://arxiv.org/abs/2502.02789) (ICML 2025)
- [vllm-project/vllm#39060](https://github.com/vllm-project/vllm/issues/39060) — open feature request tracking native vLLM support
- [Jingyu6/speculative_prefill](https://github.com/Jingyu6/speculative_prefill) — reference implementation; cloned locally at [`../spec_prefill/speculative_prefill/`](../spec_prefill/speculative_prefill/)
- `../spec_prefill/EXPERIMENT_PLAN.md` — the original Gemma4 pipeline
- `../spec_prefill_qwen/EXPERIMENT_PLAN.md` — the dense-Qwen3 (Qwen3-8B/1.7B) pipeline this one was directly ported from

---

## Files in this directory

| File | Purpose |
|---|---|
| `EXPERIMENT_PLAN.md` | This file |
| `README.md` | Overview / index |
| `REPRODUCE.md` | Environment setup + reproduction steps |
| `.env_exports.sh` | Local env config (conda activation, `HF_TOKEN`, `QWEN3_CODER_480B_MODEL_PATH`, `QWEN3_CODER_30B_MODEL_PATH`) — separate from the shared one in `../gemma4_moe_benchmarks/` and from `../spec_prefill/`'s own, see `REPRODUCE.md` step 1 |
| `vllm_patch/` | The Algorithm 1 implementation (`scoring.py`, `proposer.py`, `kv_cache_utils.py`, `prefill_split.py`, `pruning_registry.py`, `pruner.py`, `model_runner.py`, `worker.py`, `config.py`) — see its own `__init__.py` module map |
| `test_vllm_patch.py` | Unit tests for the engine-agnostic pieces of `vllm_patch/` (no GPU needed) |
| `validate_proposer.py` | GPU-node validation: speculator loading + attention hook, per-layer `head_dim` check |
| `validate_runner_integration.py` | GPU-node validation: `worker_cls` wiring + the position-override correctness check |
| `datasets/prep_longbench_v2.py` | Downloads `THUDM/LongBench-v2`, filters to the "short" (<32k word) subset, writes `datasets/longbench_v2_samples.jsonl` |
| `predict_longbench_v2.py` | Runs the P001–P006 keep-rate sweep (with/without SpecPrefill pruning) against those samples, writes a predictions JSONL per experiment |
| `grade_longbench_v2.py` | Scores a predictions file against `prep_longbench_v2.py`'s samples; writes `result.json` + prints a Markdown summary |
| `datasets/` | LongBench v2 prep output (`longbench_v2_samples.jsonl`, gitignored raw `.cache/`) |
| `results/` | Output directory (gitignored) |
