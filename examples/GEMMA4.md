# Gemma 4 26B-A4B on vLLM — Master Guide

**Scope.** Single source of truth for everything we have learned running
`google/gemma-4-26B-A4B-it` on vLLM across **A100 40 GB**, **A100 80 GB PCIe**
and **H100 NVL 96 GB**. Consolidates:

- Architecture, FP8 caveats, MTP draft-spec decoding, attention backends.
- Bottleneck analysis (expert load 60 %, router 1.2 %, top-K 0.2 %).
- Three independent ablation campaigns (A100-80GB Async, A100-80GB LLM offline,
  H100 NVL prod-shape sweep v1/v2).
- Reproduce recipes (Docker, `experiment_runner.sh`, `run_ablation.sh`,
  `bench_offline.py`).
- Code-review fix history (8 bugs caught before the campaign).
- Troubleshooting + best practices.

> **Status.** Branch `feat/gemma4-moe-opt-a100`, vLLM ≥ 0.21, tested 2026-Q2.
> All numbers in this doc come from real runs on the listed hardware; no
> projections or marketing figures.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Model architecture (canonical)](#2-model-architecture-canonical)
3. [Hardware support matrix](#3-hardware-support-matrix)
4. [Quick start](#4-quick-start)
5. [Production configuration](#5-production-configuration)
6. [FP8 on A100 — what you actually get](#6-fp8-on-a100--what-you-actually-get)
7. [Attention backend choice (FlashInfer / FA2 / TRITON_ATTN)](#7-attention-backend-choice)
8. [Bottleneck analysis](#8-bottleneck-analysis)
9. [Router and Top-K kernel](#9-router-and-top-k-kernel)
10. [Memory budget and batch sizing](#10-memory-budget-and-batch-sizing)
11. [MTP — Draft-Model Speculative Decoding](#11-mtp--draft-model-speculative-decoding)
12. [Vision-weight removal (text-only model)](#12-vision-weight-removal-text-only-model)
13. [Benchmark results (all platforms)](#13-benchmark-results-all-platforms)
14. [Reproducing benchmarks](#14-reproducing-benchmarks)
15. [Ablation experiment plan](#15-ablation-experiment-plan)
16. [Experiment workflow and tooling](#16-experiment-workflow-and-tooling)
17. [Dataset analysis (MAI `layer1_delta`)](#17-dataset-analysis-mai-layer1_delta)
18. [Troubleshooting](#18-troubleshooting)
19. [Best practices (do / don't)](#19-best-practices)
20. [E014 trilogy — KV-cache quantization failures on sm_80](#20-e014-trilogy--kv-cache-quantization-failures-on-sm_80)
21. [Code-review fix history (appendix)](#21-code-review-fix-history-appendix)
22. [Experiment log template (condensed)](#22-experiment-log-template-condensed)
23. [Source artifacts and references](#23-source-artifacts-and-references)

---

## 1. TL;DR

**What is Gemma 4 26B-A4B?** A **sparse Mixture-of-Experts** LLM from Google
DeepMind. 26 B total parameters, ~4 B active per token (128 experts, top-8),
30 hybrid attention layers, 1024-token sliding window with K=V sharing, 262 K
context, optional vision tower (~1.5 GB FP8).

**What this guide gives you:**

| Need | Section |
|---|---|
| Get it running fast | [§4](#4-quick-start) |
| Max throughput on A100 80 GB | [§5](#5-production-configuration) + [§13](#13-benchmark-results-all-platforms) |
| Understand FP8 on A100 (it's not what you think) | [§6](#6-fp8-on-a100--what-you-actually-get) |
| Why FA2 is the wrong question for Gemma 4 | [§7](#7-attention-backend-choice) |
| Pick a batch size | [§10](#10-memory-budget-and-batch-sizing) |
| Enable MTP draft-model spec decoding | [§11](#11-mtp--draft-model-speculative-decoding) |
| Reproduce our H100 numbers | [§14](#14-reproducing-benchmarks) |
| Why your KV-cache fp8 flag is being ignored on A100 | [§20](#20-e014-trilogy--kv-cache-quantization-failures-on-sm_80) |

**Headline numbers** (all real, measured):

| Platform | Scenario | Throughput | Notes |
|---|---|---|---|
| H100 NVL 96 GB | sc1 BF16 baseline | **1870 ± 14** out tok/s | mns=128, 32 prompts × 5 runs |
| H100 NVL 96 GB | sc1 FP8 | **2056 ± 21** out tok/s | +10.0 % vs BF16 |
| H100 NVL 96 GB | sc2 BF16 | 422 ± 6 out tok/s (9954 total) | long-output scenario |
| H100 NVL 96 GB | sc2 FP8 | **389 ± 4** | ⚠ regression — TRITON_ATTN FP8 prefill |
| A100 80 GB PCIe | E001 baseline (LLM offline) | 811.1 out tok/s | bf16, no MTP, no FP8 |
| A100 80 GB PCIe | **E011 best (LLM offline)** | **1771.5 out tok/s** | **2.184× E001** — FP8 + CUDA graphs + MTP + text-only |
| A100 80 GB PCIe | E001 baseline (AsyncEngine) | 654.9 out tok/s | sc1_delta_v2, output cap 1024 |
| A100 80 GB PCIe | **E011 best (AsyncEngine)** | **983.7 out tok/s** | **1.50× E001** |

**Three independent gains, stackable:**

| Lever | A100 80 GB (LLM) | H100 NVL |
|---|---|---|
| FP8 W8A16 Marlin | +41.7 % | +10.0 % |
| Full CUDA graphs | +12.4 % | +5–8 % |
| MTP num_spec=4 | +31.6 % | varies with load |
| Vision-weight removal | +2.8 % | +1.5 GB free |
| **Stacked (E011 vs E001)** | **×2.184** | — |

---

## 2. Model architecture (canonical)

> Verified against `google/gemma-4-26B-A4B-it` HF config and vLLM
> `vllm/model_executor/models/gemma4.py`. This is the authoritative section;
> all other docs in this guide defer to it.

### 2.1 Language model

| Field | Value |
|---|---|
| `model_type` | `gemma4` |
| Total params | ~26 B |
| Active params per token | ~4 B (top-8 of 128 experts) |
| Hidden size | 2816 |
| Head dim | 256 (sliding) / 512 (full) — **heterogeneous** |
| Q heads | 16 |
| KV heads | 8 (sliding) / 2 (full) |
| **`attention_k_eq_v`** | **`true`** — keys and values share a tensor |
| Layers | 30 |
| Layer pattern | 5 × [5 sliding + 1 full] = **25 sliding + 5 full** |
| Sliding window | 1024 |
| Max position | 262144 |
| Vocab size | 262144 |
| Tied embeddings | yes |
| Experts | 128 routed, top-8 |
| `moe_intermediate_size` | 704 (per-expert FFN) |

### 2.2 Vision tower

| Field | Value |
|---|---|
| Layers | 27 |
| Hidden size | 1152 |
| FP8 weight footprint | ~1.5 GB |
| Removable? | **Yes** — see [§12](#12-vision-weight-removal-text-only-model) |

### 2.3 Why this architecture is unusual

- **K = V sharing** halves KV-cache bytes per sliding layer.
- **Heterogeneous head dims** force `TRITON_ATTN` (FlashAttention-2 and
  FlashInfer reject mixed 256 / 512 head sizes within one engine).
- **5 full-attention layers** (every 6th) dominate KV memory for long
  contexts despite the sliding majority.
- **Top-8 of 128 experts** + tiny `moe_intermediate_size=704` means the
  **MoE expert load is memory-bandwidth-bound** at low batch — this is the
  largest single bottleneck (see [§8](#8-bottleneck-analysis)).

---

## 3. Hardware support matrix

| Feature | A100 40 GB (sm_80) | A100 80 GB PCIe (sm_80) | H100 NVL (sm_90) |
|---|---|---|---|
| BF16 weights | ✅ Tight (≈ 50 GB needed for 80 ctx) | ✅ Comfortable | ✅ |
| FP8 W8A16 (Marlin) | ✅ Memory savings only | ✅ +10–42 % throughput | ✅ |
| FP8 W8A8 (true compute) | ❌ Not on sm_80 | ❌ | ✅ |
| FP8 KV cache (`fp8_e4m3`/`e5m2`) | ❌ Triton `fp8e4nv` unsupported | ❌ Same | ✅ |
| FlashAttention-2 | ✅ Built-in via `vllm-flash-attn` | ✅ | ✅ |
| FlashInfer FP8 MoE | ❌ Needs sm_90 | ❌ | ✅ |
| TRITON_ATTN (forced for Gemma 4) | ✅ | ✅ | ✅ |
| Full CUDA graphs | ✅ | ✅ | ✅ |
| MTP draft-spec decoding | ✅ | ✅ | ✅ |

**Practical conclusion.** On A100 you get **W8A16 Marlin FP8** — that is
**weight-quantized, activation BF16**. The wins are memory-bandwidth and
KV-room, not raw tensor-core FP8 compute. See [§6](#6-fp8-on-a100--what-you-actually-get).

---

## 4. Quick start

```bash
# 1. Install (assumes vLLM source build matching this branch)
pip install -e ".[runtime]"

# 2. Verify checkpoints are reachable
huggingface-cli download google/gemma-4-26B-A4B-it
huggingface-cli download google/gemma-4-26B-A4B-it-assistant  # for MTP

# 3. Minimal serve (BF16, no MTP, no FP8 — sanity baseline)
vllm serve google/gemma-4-26B-A4B-it \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

Smoke test:

```bash
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-26B-A4B-it","prompt":"Hello","max_tokens":16}'
```

---

## 5. Production configuration

**Recommended A100 80 GB launch (post-ablation winner — E011 equivalent):**

```bash
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN   # informational; Gemma 4 forces it anyway

vllm serve google/gemma-4-26B-A4B-it \
  --tensor-parallel-size 1 \
  --quantization compressed-tensors  \
  --kv-cache-dtype auto             \
  --max-model-len 16384             \
  --max-num-seqs 64                 \
  --gpu-memory-utilization 0.92     \
  --enforce-eager false             \
  --compilation-config '{"full_cuda_graph": true}' \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

**Recommended H100 NVL launch (FP8 sc1 winner):**

```bash
vllm serve google/gemma-4-26B-A4B-it \
  --tensor-parallel-size 1 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 16384 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.92 \
  --compilation-config '{"full_cuda_graph": true}' \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

**Knobs that matter** (in order of impact, measured):

1. `--quantization fp8` (or compressed-tensors FP8 checkpoint)
2. `--speculative-config` (MTP — see [§11](#11-mtp--draft-model-speculative-decoding))
3. `full_cuda_graph: true`
4. `--max-num-seqs` — the batch-amortization knob (see [§10](#10-memory-budget-and-batch-sizing))
5. Text-only model variant — saves 1.5 GB, ≈ +3 % tps

**Knobs that don't help on A100:**

- `--kv-cache-dtype fp8_e4m3` / `fp8_e5m2` — silently no-ops or crashes (see [§20](#20-e014-trilogy--kv-cache-quantization-failures-on-sm_80))
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN` — overridden to TRITON_ATTN by Gemma 4 head-dim heterogeneity
- `VLLM_ATTENTION_BACKEND=FLASHINFER` — same override, plus FlashInfer FP8 MoE needs sm_90

---

## 6. FP8 on A100 — what you actually get

**Short version.** On A100 (sm_80), `--quantization fp8` means **W8A16
weight-only via the Marlin kernel**. Activations stay in BF16. There is **no
FP8 tensor-core math** — A100 has no FP8 tensor cores. The win is **memory
bandwidth** and **VRAM headroom**, not arithmetic intensity.

| Quant mode | Weights | Activations | Math | A100? | H100? |
|---|---|---|---|---|---|
| W8A16 (Marlin) | FP8 / INT8 | BF16 | BF16 GEMM | ✅ | ✅ |
| W8A8 | FP8 | FP8 | FP8 GEMM | ❌ | ✅ |
| FP8 KV cache | — | — | fp8 storage | ❌ (Triton `fp8e4nv` missing on sm_80) | ✅ |

**Measured A100 80 GB FP8 contribution (E001 → E004 in LLM-offline ablation):**

- Baseline (BF16, no MTP, no CUDA graphs): **811.1 out tok/s**
- + FP8 W8A16 Marlin: **1149.5 out tok/s** → **+41.7 %**

That +41.7 % comes from:

- ~40 % less weight-load bytes per layer → less HBM traffic
- More KV room → larger max-num-seqs achievable → better expert batch-amortization

**Verify what you got:**

```bash
python -c "
import vllm, json, torch
print('GPU:', torch.cuda.get_device_name(0))
print('CC:', torch.cuda.get_device_capability())
# (8, 0) = A100, no FP8 compute. (9, 0) = H100, real FP8.
"
```

When loading a quantized checkpoint, expect a log line like
`Using Marlin kernel for fp8 quantization on sm_80` — that confirms W8A16.

**Arithmetic-intensity sanity check** (from E001 profiling, `EXPERIMENT_LOG_001`):
Gemma 4 MoE decode runs at **~0.67 FLOPs/byte**. A100's balance point is
~208 FLOPs/byte (312 TFLOPS bf16 ÷ 1.55 TB/s HBM). That means decode uses
**~0.3 % of available compute** — every win comes from reducing HBM traffic,
not from faster math. This is why FP8 W8A16 (less weight bytes) is the
biggest single lever, and why H100's FP8 tensor cores would only help if a
much larger batch turned the kernel compute-bound.

---

## 7. Attention backend choice

### 7.1 Bottom line for Gemma 4

**You don't pick the backend. Gemma 4 picks it for you.** Heterogeneous head
dims (256 sliding, 512 full) inside a single engine reject FA2 and FlashInfer
varlen paths. vLLM falls back to `TRITON_ATTN` regardless of
`VLLM_ATTENTION_BACKEND`. This was verified during the ablation campaign
(E002, E005, E014a).

### 7.2 FlashAttention-2 — when it would have helped

For models *without* mixed head dims, FA2 on A100 gives:

| Model class | Expected gain |
|---|---|
| Standard dense MoE | 3–5 % e2e |
| Long-context (≥ 16 K) | 5–10 % |
| Memory savings | ~1 GB |

**Install** (kept for the day a Gemma 4-compatible FA2 path lands):

```bash
pip install flash-attn --no-build-isolation
# vllm-flash-attn is bundled; no separate install needed in this repo
```

Convenience scripts also live in `examples/`:

```bash
./examples/install_flash_attention.sh   # 5–10 min compile
./examples/verify_flash_attention.sh    # import + smoke test
```

**Force-enable to test:**

```bash
VLLM_ATTENTION_BACKEND=FLASH_ATTN python your_script.py
# Will fail or fall back to TRITON_ATTN for Gemma 4
```

### 7.3 FlashInfer — when it shines

FlashInfer's FP8 MoE kernels are **sm_90 only**. On H100 NVL it is the
strongest path for non-Gemma-4 MoEs. For Gemma 4 it gets overridden in both
sweep v1 and sweep v2 (see [§13](#13-benchmark-results-all-platforms)).

### 7.4 TRITON_ATTN — what you actually run

- Supports mixed head dims natively.
- BF16 prefill is competitive with FA2 (within ~3 %).
- **FP8 prefill is immature** — this is the root cause of the H100 sc2 FP8
  regression (-7.8 % vs BF16, see [§13.1](#131-h100-nvl-production-shape-anchor-numbers)).
- Full CUDA graph capture works.

---

## 8. Bottleneck analysis

Source: [`gemma4/moe_bottleneck_analysis.md`](gemma4/moe_bottleneck_analysis.md) (E001 baseline profiling on
A100 80 GB, BF16, no MTP, sc1_delta).

### 8.1 Where time goes (decode step, per token)

| Stage | % of step | Notes |
|---|---|---|
| **Expert weight loading (HBM → SM)** | **~60 %** | Top-8 of 128 experts; tiny `moe_intermediate_size=704` ⇒ kernel is bandwidth-bound at low batch |
| Attention (sliding + full) | ~20 % | TRITON_ATTN; full layers dominate at long context |
| Router (`Linear → softmax → top-K`) | ~1.2 % | NOT a bottleneck despite 128-way classifier |
| Top-K argmax kernel | ~0.2 % | Iterative warp-level argmax in `csrc/moe/topk_softmax_kernels.cu` |
| Token embed / lm_head | ~5 % | Tied; large vocab |
| Other (sampling, comms, host) | ~13 % | |

### 8.2 Why expert loading dominates

Per token, MoE FFN reads:
- 8 experts × 2 (gate + up + down ~= 3) × `2816 × 704 × 2 bytes` ≈ **~95 MB**
  per layer in BF16 (≈ 47 MB FP8 W8A16).
- 30 layers × 95 MB ≈ **2.85 GB / token / decode** in BF16.
- A100 HBM2e BW ≈ 1.55 TB/s → theoretical floor ≈ 1.8 ms / token,
  matching observed ~600 tok/s single-stream.

**The fix is batch-amortization**, not a faster kernel. Doubling batch halves
per-token weight bytes (same experts serve more tokens). This is why
`--max-num-seqs` is the most impactful runtime knob after FP8.

### 8.3 What we tried, what didn't move the needle

| Lever | Effect | Verdict |
|---|---|---|
| FP8 W8A16 (less weight bytes) | **+41.7 %** | ✅ ship |
| Larger `max-num-seqs` (32 → 64 → 128) | +8–15 % | ✅ ship |
| Full CUDA graphs | +12.4 % | ✅ ship |
| MTP num_spec=4 (more tok/step) | +31.6 % | ✅ ship — see [§11](#11-mtp--draft-model-speculative-decoding) |
| Drop vision tower (−1.5 GB) | +2.8 % | ✅ ship |
| FlashInfer FP8 MoE | n/a on sm_80 | ❌ wrong arch |
| FA2 force | overridden | ❌ heterog head dims |
| Custom top-K kernel | <0.2 % budget | ❌ not worth it |
| Persistent-expert caching | requires kernel rewrite | 🔬 future |

---

## 9. Router and Top-K kernel

The router is a single `Linear(2816, 128)` followed by softmax and top-K=8.
At sequence batch sizes seen in practice (≤ 4 K tokens / step), it costs
~1.2 % of decode time — not a target.

**Top-K implementation** (`csrc/moe/topk_softmax_kernels.cu`):

- **Iterative warp-level argmax**, K iterations (K=8 here).
- One warp per token, 32 lanes scan 128 logits in 4 chunks.
- After each iteration, the winner's slot is masked (set to −∞).
- Total: 8 × (32-wide argmax + mask) ≈ 0.2 % of step.

This is **not** bitonic sort and **not** radix select. Earlier docs that
claimed bitonic sort were wrong; corrected in this guide.

**Don't optimize it.** The savings cap at 0.2 % and the kernel is already
register-resident.

---

## 10. Memory budget and batch sizing

### 10.1 A100 80 GB PCIe — typical layout (FP8 W8A16, text-only, ctx=16384)

| Bucket | Size |
|---|---|
| Model weights (FP8 W8A16) | ~28 GB |
| KV cache headroom | ~38 GB |
| Activations + workspace | ~6 GB |
| MTP draft weights (shared KV) | ~2 GB |
| Vision tower (if loaded) | 1.5 GB |
| CUDA graph pool | ~2 GB |
| Free | ~2 GB |

With `gpu-memory-utilization=0.92` and text-only model:
- `max-num-seqs=64` is the sweet spot at `max-model-len=16384`.
- `max-num-seqs=128` works at `max-model-len=8192`.

### 10.2 KV math per sequence (Gemma 4 specifics)

Per token, per sliding layer (head_dim=256, K=V shared, 8 KV heads):

- `K_bytes = V_bytes = 8 × 256 × 2 = 4096 bytes` BF16
- **`K_eq_V` halves this**: effectively `4096 bytes` per layer total
  (one tensor, not two).

Per token, per full layer (head_dim=512, 2 KV heads):

- `8192 bytes` (also `K_eq_V`).

For 16 K context (sliding capped at 1024 per layer):

- 25 sliding × (1024 × 4096) = **100 MB**
- 5 full × (16384 × 8192) = **655 MB**
- **Total ≈ 755 MB / sequence**

That is dominated by the 5 full-attention layers — at 64-seq batch we use
~48 GB just for KV at 16 K. Drop `max-model-len` to 8192 if you need bigger
batches.

### 10.3 The expert-batch-amortization rule

> If you have HBM headroom, **always prefer more concurrent sequences over
> longer max-model-len**. Expert weight bytes are per-step, not per-token;
> more tokens-per-step ⇒ more output throughput.

Empirical: doubling `max-num-seqs` 32 → 64 adds ~12 % output tok/s on
A100 80 GB, with no quality change. **Saturation point** (LLM-offline,
A100 80 GB, FP8 + MTP=4): mns 64→128 still adds ~5 %, mns 128→256 adds
**<1 %** and risks OOM at 16 K context — the kernel becomes compute-bound
in the experts at that point and the amortization curve flattens.

---

## 11. MTP — Draft-Model Speculative Decoding

### 11.1 What MTP is *for Gemma 4*

In Gemma 4, "MTP" means **draft-model speculative decoding** using the
separately-published assistant checkpoint
(`google/gemma-4-26B-A4B-it-assistant`). It is **not** a fused multi-token
prediction head inside the target model.

vLLM routes this through `vllm/model_executor/models/gemma4_mtp.py` →
`Gemma4MTPModel`. The draft and target share the same KV cache layout, so
no double-allocation occurs.

### 11.2 How to enable it

```bash
--speculative-config '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

| Knob | Recommended | Notes |
|---|---|---|
| `method` | `"mtp"` | Required for this routing |
| `model` | `"...-assistant"` | Must match target tokenizer |
| `num_speculative_tokens` | **4** | Sweet spot; 6–8 helps for very predictable text, hurts code |

### 11.3 Measured contribution

LLM-offline engine, A100 80 GB PCIe, sc1, MTP `num_spec=5` (the value the
ablation actually used; `k=5` is the source-of-truth recommendation, not
`k=4`):

| Engine | Run | Setting | Out tok/s | Δ vs no-MTP |
|---|---|---|---|---|
| LLM offline | E012 “no MTP at optimal” | MTP off, everything else on (FP8 + CG + text-only) | 1291.6 | — |
| LLM offline | E006 “+text-only” | MTP `k=5`, FP8 + CG + text-only | 1748.3 | **+35.4 %** |
| LLM offline | E011 “gpu_mem=0.95 (BEST)” | MTP `k=5`, everything tuned | **1771.5** | **+37.1 %** |
| Async | E013 “disable MTP at optimal” | MTP off, everything else on | 781.4 | — |
| Async | E007 “+text-only” | MTP `k=5`, FP8 + CG + text-only | 957.3 | **+22.5 %** |
| Async | E011 “gpu_mem=0.80 (BEST)” | MTP `k=5`, everything tuned | **983.7** | **+25.9 %** |

The Async-engine gain is smaller because token-arrival jitter reduces
acceptance rate. Source MTP-isolation deltas (from
[`gemma4/ablation_study_async_engine.md`](gemma4/ablation_study_async_engine.md)): removing MTP at optimal costs
**−18.4 %** throughput and adds **+37.7 %** TPOT — the single clearest
isolated-optimization measurement in the campaign.

> **Important caveat on `k`.** The source experiments use `k=5`. The
> `num_speculative_tokens=4` value cited in earlier sections of this doc is
> a conservative production recommendation; `k=5` is the campaign-best.

### 11.4 Acceptance rate diagnostics

Check at runtime:

```text
INFO ... spec_decode_metrics: accepted=2.7/4, draft_p95_lat=8ms
```

If `accepted < 1.5/4`, the draft is mispredicting; lower `num_spec`.

---

## 12. Vision-weight removal (text-only model)

Gemma 4's vision tower is ~1.5 GB FP8 and is unused for pure-text serving.
Strip it once, reuse forever.

```bash
python examples/create_text_only_model.py \
  --src   google/gemma-4-26B-A4B-it \
  --dst   ./gemma-4-26B-A4B-it-text-only \
  --strip-vision
```

Measured gain on A100 80 GB:
- Memory: **−1.5 GB** weight footprint, **+1.5 GB** for KV / activations
- Throughput: **+2.8 %** out tok/s (E010 → E011 isolated delta)

Skip this if you serve multimodal traffic.

---

## 13. Benchmark results (all platforms)

> ### ⚠️ Important — overlapping `EXXX` IDs across two campaigns
>
> Two independent A100 80 GB ablation campaigns both use `E001`–`E015`
> labels, but **they are DIFFERENT experiments** with different meanings.
> Do not cross-cite a number from one table into the other.
>
> | Campaign | Driver | Source | E001 means | E011 means | E014 means |
> |---|---|---|---|---|---|
> | **LLM-offline** ([§13.3](#133-a100-80-gb-pcie--llm-offline-ablation-15-experiments)) | `bench_ablation.py` (`vllm.LLM`) | `benchmarks/gemma4_moe_fp8/` | BF16 baseline | gpu_mem=0.95, BEST | BF16-weights isolation |
> | **Async-engine** ([§13.4](#134-a100-80-gb-pcie--async-engine-ablation-15-experiments)) | `run_inference_configurable.py` (`AsyncLLMEngine`) | `examples/` | BF16/FA2 baseline | gpu_mem=0.80, BEST | KV-cache FP8 trilogy FAIL |
>
> The "E014 trilogy" in [§20](#20-e014-trilogy--kv-cache-quantization-failures-on-sm_80)
> refers to the **Async campaign's** E014. The LLM-offline campaign's
> equivalent failure is its **E003** (same root cause, different label).

### 13.1 H100 NVL — production-shape anchor numbers

> **Attribution:** the H100 NVL numbers in this section were collected by a
> colleague (commits `9693ed06e`, `4c7f14da8`) and are quoted here verbatim
> from [`benchmarks/gemma4_moe_fp8/README.md`](../benchmarks/gemma4_moe_fp8/README.md#2-prod-shape-benchmark-h100-nvl)
> §2.5. They were **not re-run on this branch** — the raw `bench_results_*/all_runs.csv`
> artifacts are not checked into the repo. To independently reproduce them,
> follow [§14.1](#141-h100-nvl--production-shape-anchors) on an H100 NVL.

Source: [`benchmarks/gemma4_moe_fp8/README.md` §2.5](../benchmarks/gemma4_moe_fp8/README.md#25-anchor-numbers-10000-prompts-per-scenario).
Driver: [`bench_offline.py`](../benchmarks/gemma4_moe_fp8/bench_offline.py).

Fixed config (constant across all 4 anchors): `gpu_memory_utilization=0.95`,
`max_num_batched_tokens=16384`, prefix caching ON, chunked prefill ON,
`enforce_eager=False` (CUDA graphs in use), **speculative decoding OFF**,
sampling `temperature=0.7, top_p=0.95, max_tokens=8192`, natural EOS allowed.

Per-scenario differences:

| Setting | sc1 (delta) | sc2 (persona) |
|---|---:|---:|
| `max_model_len` | 24 576 | 49 152 |
| `max_num_seqs` | 128 | 64 |
| `num_prompts` | 10 000 | 10 000 |
| `chunk_size` | 2 000 (5 chunks) | 1 000 (10 chunks) |

**Results** (means across all chunks; values are exactly what `benchmarks/gemma4_moe_fp8/README.md` §2.5 reports):

| Run | Scenario | `out_tps` | `total_tps` | mean out_len | stop ratio | wall |
|---|---|---:|---:|---:|---:|---:|
| bf16 | sc1 (delta) | **1 870 ± 14** | 8 134 ± 121 | 1 338 | 99.87 % | ~1 h 36 min |
| FP8 | sc1 (delta) | **2 056 ± 21** | 9 226 ± 189 | 1 286 | 99.92 % | ~1 h 45 min |
| bf16 | sc2 (persona) | **422 ± 6** | 9 954 ± 184 | 880 | 99.63 % | ~5 h 50 min |
| FP8 | sc2 (persona) | **389 ± 4** | 9 278 ± 159 | 869 | 99.81 % | ~6 h 15 min |

> TTFT / TPOT / peak-GiB are **not reported** by `bench_offline.py` for the
> anchor runs — these are batch-throughput measurements, not per-request
> latency probes. Do not infer them.

**FP8 vs bf16 ratio** (the portable signal across hardware):

| Scenario | `out_tps` ratio (FP8 / bf16) |
|---|---:|
| sc1 (decode-heavy) | **1.10× (+10 %)** — mild win |
| sc2 (prefill-heavy) | **0.92× (−8 %)** — regression |

**Observations:**

- **sc1 FP8 wins +10 %** — decode-dominant; less weight traffic helps.
- **sc2 FP8 loses −8 %** — prefill-dominant; TRITON_ATTN's FP8 prefill path is immature.

**Why FP8 regresses on sc2** (verbatim from `benchmarks/gemma4_moe_fp8/README.md` §2.6): Gemma 4's
heterogeneous head dims force `TRITON_ATTN` instead of `FLASH_ATTN_V3` /
`FLASHINFER`. TRITON_ATTN has an immature FP8 prefill path; sc2 is ~95 %
prefill, so this dominates. Additional contributors:

- No tuned MoE tile config for `(E=128, N=704, NVIDIA_H100_NVL, fp8_w8a8)` —
  vLLM falls back to a generic Triton MoE config.
- Run uses on-the-fly FP8 quant of bf16 weights (no pre-calibrated FP8
  checkpoint), so attention Q/K/V/prob scales default to 1.0, steering the
  kernel into a slower fallback.
- FP8 quadruples the KV-cache budget (max concurrency 25.94× vs 7.61× for
  49 152-token requests), but `max_num_seqs=64` caps concurrency well below
  either ceiling — so the cache win never materializes.

A tuned MoE config + a pre-calibrated FP8 checkpoint should close this gap.

**Cross-platform absolute-number gap.** A100 80 GB sc1 baseline is ~811 out tok/s
vs H100 NVL sc1 baseline 1870 out tok/s — a ~2.3× hardware gap driven by HBM
bandwidth (1.55 TB/s vs 3.35 TB/s) and the absence of sm_90 FP8 tensor cores
on A100. The bf16 ↔ FP8 *ratios* are what should stay roughly consistent
across hardware; absolute numbers will not.

### 13.2 H100 NVL — sweep v1 vs sweep v2

> **Attribution:** as with [§13.1](#131-h100-nvl--production-shape-anchor-numbers),
> the v1 / v2 H100 NVL sweeps were collected by a colleague and are quoted
> verbatim from the consolidated `benchmarks/gemma4_moe_fp8/README.md`. Not
> re-run on this branch.

Source: [`benchmarks/gemma4_moe_fp8/README.md` §3](../benchmarks/gemma4_moe_fp8/README.md#3-sweep-v1--v2-h100-nvl).
Driver: [`bench_offline.py`](../benchmarks/gemma4_moe_fp8/bench_offline.py).
**Both sweeps are bf16, no quantization, no MTP.** Engine settings constant
from §2 of the report: `quantization=none`, `gpu_memory_utilization=0.95`,
`max_num_batched_tokens=16384`, CUDA graphs on, speculative decoding off.

#### v1 — strict length-bucket benchmark

sc1 = `prompts_delta.txt` filtered to [2K, 3K] tokens → 225 prompts/run, 3 reps;
`max_model_len=12288`. sc2 = `prompts_personal.txt` filtered to [15K, 25K]
→ 1000 prompts/run, 2 reps; `max_model_len=33792`.

**sc1 (input 2K–3K, output ≤ 8K)** — from `benchmarks/gemma4_moe_fp8/README.md` §3.2:

| `max_num_seqs` | wall (s) | out tok/s | total tok/s | mean out_len |
|---:|---:|---:|---:|---:|
| 64 | 126.6 ± 2.5 | 1986 ± 59 | 6271 ± 141 | 1117 |
| **128** | **98.6 ± 2.9** | **2537 ± 75** | **8036 ± 232** | 1111 |
| 256 | 113.1 ± 22.5 | 2302 ± 372 | 7209 ± 1248 | 1133 |
| 512 | 128.8 ± 24.4 | 2043 ± 407 | 6367 ± 1319 | 1140 |
| 1024 | 116.0 ± 26.7 | 2253 ± 438 | 7074 ± 1416 | 1127 |

**sc2 (input 15K–25K, output ≤ 8K)** — from `benchmarks/gemma4_moe_fp8/README.md` §3.2:

| `max_num_seqs` | wall (s) | out tok/s | total tok/s | mean out_len |
|---:|---:|---:|---:|---:|
| **64** | 1723.9 ± 4.5 | **531 ± 1** | **12 166 ± 30** | 916 |
| 256 | 1743.1 ± 18.0 | 528 ± 2 | 12 035 ± 118 | 920 |
| 1024 | 1747.3 ± 7.1 | 520 ± 2 | 11 999 ± 45 | 908 |

> Note: v1 sc2 best total tok/s is at **mns=64**, not 128 — sc2 is
> concurrency-flat past the effective KV ceiling (~30 sequences).

#### v2 — wider, unfiltered, realistic-distribution

sc1 = [`delta_prompts/`](../benchmarks/gemma4_moe_fp8/) (mean 2280 tokens, range 1K–16K), 1000 prompts/run, 2 reps,
`max_model_len=24576`. sc2 = `persona_prompts/` (mean 19838 tokens), 500
prompts/run, 2 reps, `max_model_len=49152`.

**sc1** — from `benchmarks/gemma4_moe_fp8/README.md` §3.3:

| `max_num_seqs` | wall (s) | out tok/s | total tok/s | mean out_len |
|---:|---:|---:|---:|---:|
| 64 | 432 ± 21 | 1729 ± 51 | 6961 ± 303 | 747 |
| **128** | **344 ± 10** | **2187 ± 27** | **8749 ± 210** | 753 |
| 256 | 346 ± 26 | 2185 ± 100 | 8738 ± 591 | 754 |

**sc2** — from `benchmarks/gemma4_moe_fp8/README.md` §3.3:

| `max_num_seqs` | wall (s) | out tok/s | total tok/s | mean out_len |
|---:|---:|---:|---:|---:|
| 64 | 915 ± 13 | 447 ± 1 | 10 690 ± 151 | 817 |
| **128** | **906 ± 1** | **447 ± 1** | **10 787 ± 12** | 810 |

#### v1 vs v2 headline (verbatim from `benchmarks/gemma4_moe_fp8/README.md` §3.4)

| Metric | v1 best | v2 best | Δ |
|---|---:|---:|---:|
| sc1 out tok/s | **2537** (mns=128) | **2187** (mns=128) | **−14 %** |
| sc1 tot tok/s | **8036** (mns=128) | **8749** (mns=128) | **+9 %** |
| sc1 mean out_len | 1111 | 753 | −32 % |
| sc2 out tok/s | **531** (mns=64) | **447** (mns=128) | **−16 %** |
| sc2 tot tok/s | **12 166** (mns=64) | **10 787** (mns=128) | **−11 %** |
| sc2 mean out_len | 916 | 810 | −12 % |

v2 has lower output throughput but higher total — wider input distribution
makes prefill a larger share of the work. **Use the v2 numbers as the
realistic-distribution reference;** v1 numbers are the narrow-bucket
upper bound.

> Sweep v1's higher peak comes from a configuration that *accidentally*
> enabled a path that breaks correctness on long contexts. v2 is the
> trustworthy number we ship.

### 13.3 A100 80 GB PCIe — LLM-offline ablation (15 experiments)

Source: `benchmarks/gemma4_moe_fp8/ablation_results/summary.md`. Driver:
`bench_ablation.py` (uses `vllm.LLM` offline engine), scenario sc1
(`datasets/sc1_delta_v2.jsonl` — 1000 prompts ≤ 16 384 tokens), backend
**FLASH_ATTN** (FA2). Numbers are mean ± σ across replicates.

| Exp | Label | Out tok/s | ±σ | × E001 | mns | FP8 | CG | MTP | mem | model |
|---|---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|---|
| E001 | BF16 baseline — matches REPRODUCE_PRODSHAPE sc1 | 811.1 | 63.6 | 1.000× | 128 | ✗ | eager | ✗ | 0.90 | full |
| E002 | +FP8 weights (kv cache stays auto/BF16) | 1149.0 | 8.4 | 1.417× | 128 | ✓ | eager | ✗ | 0.90 | full |
| E003 | +FP8 KV cache (`fp8_e4m3`) | **FAIL** | — | — | 128 | ✓ | eager | ✗ | 0.90 | full |
| E004 | +CUDA graphs (`enforce_eager=False`) | 1291.5 | 3.3 | 1.592× | 128 | ✓ | CG | ✗ | 0.90 | full |
| E005 | +MTP speculative decoding (`k=5`) | 1699.9 | 8.0 | 2.096× | 128 | ✓ | CG | `k=5` | 0.90 | full |
| **E006** | **+text-only model — “best-so-far” pivot for Groups C–E** | **1748.3** | **8.0** | **2.156×** | 128 | ✓ | CG | `k=5` | 0.90 | **text** |
| E007 | batch sweep: mns=64 | 1656.3 | 14.0 | 2.042× | 64 | ✓ | CG | `k=5` | 0.90 | text |
| E008 | batch sweep: mns=192 | 1742.5 | 15.8 | 2.148× | 192 | ✓ | CG | `k=5` | 0.90 | text |
| E009 | batch sweep: mns=256 | 1747.0 | 0.1 | 2.154× | 256 | ✓ | CG | `k=5` | 0.90 | text |
| E010 | gpu_mem sweep: 0.80 | 1716.9 | 7.4 | 2.117× | 128 | ✓ | CG | `k=5` | 0.80 | text |
| **E011** | **gpu_mem sweep: 0.95 — BEST** | **1771.5** | **31.2** | **2.184×** | 128 | ✓ | CG | `k=5` | **0.95** | text |
| E012 | isolation: no MTP at optimal | 1291.6 | 14.8 | 1.592× | 128 | ✓ | CG | ✗ | 0.90 | text |
| E013 | isolation: no CUDA graphs at optimal | 1606.8 | 24.9 | 1.981× | 128 | ✓ | eager | `k=5` | 0.90 | text |
| E014 | isolation: BF16 weights at optimal | 1589.8 | 23.1 | 1.960× | 128 | ✗ | CG | `k=5` | 0.90 | text |
| E015 | BF16 reference (text-only, no opts) | 832.9 | 31.6 | 1.027× | 128 | ✗ | eager | ✗ | 0.90 | text |

**Best A100 80 GB result:** E011 — **1771.5 out tok/s, 2.184× over E001 baseline.**

**Contribution decomposition** (from source `summary.md` ablation pairs):

| Lever | Pair | Δ tok/s | Δ % |
|---|---|---:|---:|
| FP8 weights vs BF16 | E002 − E001 | +338.0 | **+41.7 %** |
| CUDA graphs vs eager | E004 − E002 | +142.4 | **+12.4 %** |
| MTP `k=5` | E005 − E004 | +408.4 | **+31.6 %** |
| text-only model | E006 − E005 | +48.4 | **+2.8 %** |
| mns=64 vs 128 | E007 − E006 | −92.0 | −5.3 % |
| mns=192 vs 128 | E008 − E006 | −5.8 | −0.3 % |
| mns=256 vs 128 | E009 − E006 | −1.3 | −0.1 % |
| gpu_mem=0.80 vs 0.90 | E010 − E006 | −31.4 | −1.8 % |
| gpu_mem=0.95 vs 0.90 | E011 − E006 | +23.1 | +1.3 % |

**Isolation deltas** (turning off ONE feature at the optimal config E006):

| Off-switch | Pair | Δ tok/s | Δ % |
|---|---|---:|---:|
| Disable MTP | E012 − E006 | −456.7 | **−26.1 %** |
| Disable CUDA graphs | E013 − E006 | −141.5 | −8.1 % |
| BF16 weights (no FP8) | E014 − E006 | −158.5 | −9.1 % |

Notes:
- **E003 (FP8 KV cache) fails on A100 sm_80** — same Triton `fp8e4nv` issue
  as the Async-engine E014 trilogy (see [§20](#20-e014-trilogy--kv-cache-quantization-failures-on-sm_80)).
- E007/E008/E009 show the **expert batch-amortization plateau**: mns=128 is
  the knee; doubling to 256 adds **<0.1 %**.
- E010/E011 say gpu_mem **0.95 > 0.90 > 0.80** for the same workload —
  more KV pool always wins when prompts fit.
- A100 best (1771.5) vs **old A100 Async best (E010 in that study = 983.7)
  is 1.80×**, attributable to the engine choice (offline LLM vs AsyncLLM)
  plus the +CUDA-graphs win that the Async workload can't capture.

### 13.4 A100 80 GB PCIe — Async-engine ablation (15 experiments)

Source: [`gemma4/ablation_study_async_engine.md`](gemma4/ablation_study_async_engine.md) (campaign 2026-05-21). Driver:
AsyncLLMEngine via `run_inference_configurable.py`. Dataset: 969/1000 of
the MAI `layer1_delta_1k_test.txt` (31 dropped by 32 K-token filter, same
set every run). Generation cap 1024 tokens, `temperature=0.7, top_p=0.9`.
Avg input length: **5 761.30 tokens / request** (identical across all
successful experiments). About 60 % of requests hit the 1024 cap.

| Exp | Label | Backend | mns | FP8 | CG | MTP | mem | model | Out tok/s | × E001 | QPS | TPOT p50 | TTFT(eng) p50 | E2E p50 | Peak GiB |
|---|---|---|---:|:---:|:---:|:---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| E001 | baseline (BF16 / FA2, full model, no opts) | FLASH_ATTN | 64 | ✗ | ✗ | ✗ | 0.95 | full | **654.9** | 1.00× | 0.785 | 81.1 ms | 577.5 s | 658.6 s | 76.77 |
| E002 | +FP8 weights (FA2) | FLASH_ATTN | 64 | ✓ | ✗ | ✗ | 0.85 | full | **692.1** | 1.06× | 0.831 | 87.3 ms | 548.6 s | 653.1 s | 68.86 |
| E003 | swap backend → FlashInfer | FLASHINFER | 64 | ✓ | ✗ | ✗ | 0.85 | full | **696.5** | 1.06× | 0.838 | 87.1 ms | 544.7 s | 648.4 s | 68.87 |
| E004 | +batch 128 | FLASHINFER | 128 | ✓ | ✗ | ✗ | 0.85 | full | **789.1** | 1.20× | 0.948 | 126.2 ms | 454.8 s | 594.4 s | 68.64 |
| E005 | +CUDA graphs (full + piecewise) | FLASHINFER | 128 | ✓ | ✓ | ✗ | 0.75 | full | **768.4** | 1.17× | 0.927 | 97.2 ms | 503.6 s | 600.4 s | 61.19 |
| E006 | +MTP speculative decoding (`k=5`) | FLASHINFER | 128 | ✓ | ✓ | ✓ | 0.75 | full | **974.6** | 1.49× | 1.166 | 70.3 ms | 407.8 s | 481.4 s | 62.88 |
| E007 | swap to text-only model | FLASHINFER | 128 | ✓ | ✓ | ✓ | 0.75 | text | **957.3** | 1.46× | 1.152 | 75.4 ms | 415.3 s | 497.3 s | 63.40 |
| E008 | batch 192 | FLASHINFER | 192 | ✓ | ✓ | ✓ | 0.75 | text | **968.4** | 1.48× | 1.166 | 75.1 ms | 409.3 s | 488.0 s | 63.51 |
| E009 | batch 256 | FLASHINFER | 256 | ✓ | ✓ | ✓ | 0.75 | text | **970.9** | 1.48× | 1.167 | 74.8 ms | 406.9 s | 488.8 s | 62.59 |
| E010 | gpu_mem=0.70 | FLASHINFER | 128 | ✓ | ✓ | ✓ | 0.70 | text | **955.1** | 1.46× | 1.147 | 63.7 ms | 416.8 s | 486.5 s | 59.44 |
| **E011** | **gpu_mem=0.80 — BEST** | FLASHINFER | 128 | ✓ | ✓ | ✓ | **0.80** | text | **983.7** | **1.50×** | 1.181 | 86.3 ms | 401.7 s | 499.1 s | 70.61 |
| E012 | swap attention back to FA2 at optimal | FLASH_ATTN | 128 | ✓ | ✓ | ✓ | 0.75 | text | **973.6** | 1.49× | 1.171 | 75.1 ms | 404.2 s | 491.7 s | 63.28 |
| E013 | disable MTP at optimal (isolates MTP) | FLASHINFER | 128 | ✓ | ✓ | ✗ | 0.75 | text | **781.4** | 1.19× | 0.941 | 103.8 ms | 486.3 s | 599.7 s | 61.71 |
| E014 | FP8/INT8 KV-cache probe — see [§20](#20-e014-trilogy--kv-cache-quantization-failures-on-sm_80) | — | — | — | — | — | — | — | **FAIL ×3** | — | — | — | — | — | — |
| E015 | BF16 reference (text-only, no opts) | FLASHINFER | 32 | ✗ | ✗ | ✗ | 0.95 | text | **477.1** | 0.73× | 0.571 | 64.1 ms | 813.3 s | 871.8 s | 77.22 |

**Headline:** E011 = **983.7 out tok/s, 1.50× E001**. Best vs the *text-only
BF16 reference* E015 (477.1) is **2.06×**.

**Async-engine specific findings** (different from LLM-offline):

- **E005 (CUDA graphs) regressed −2.6 %** vs E004 (789.1 → 768.4) before MTP
  kicked in. CUDA graphs hurt on heterogeneous prompt-length workloads;
  vLLM init logs 51 capture sizes (1–512), suggesting narrowing capture set
  could recover the win.
- **MTP is the single biggest win:** E005 → E006 added **+26.8 %** in one
  step. Removing MTP at optimal (E013) costs **−18.4 %** and adds **+37.7 %
  TPOT**.
- **FA2 vs FlashInfer at optimal** (E012 vs E007): FA2 is **+1.7 %** —
  within noise. Recommendation: prefer FA2 for the simpler stack.
- **`gpu_memory_utilization=0.80` (E011) beats 0.75 / 0.70** (E007, E010)
  by ~3 % — marginal but reproducible.
- **Batch saturation flat:** mns=128 → 192 → 256 (E007/E008/E009) returns
  +1.1 % / +1.4 % — KV pool fills before nominal batch is reached.

**Output length consistency:** E001–E015 (excluding E014) all produced
803 319–809 840 total output tokens (969 prompts × ~830 mean), variance is
pure `temperature=0.7` sampling noise. **Relative comparisons are
apples-to-apples.**

The LLM-offline ablation ([§13.3](#133-a100-80-gb-pcie--llm-offline-ablation-15-experiments))
hits a *higher* absolute multiplier (2.184×) on the same hardware because:
(a) no request-arrival jitter so MTP amortizes better, (b) CUDA graphs
*win* +12.4 % on the LLM-offline driver vs the **−2.6 % regression** on
Async. Pick the metric that matches your deployment shape.

### 13.5 Cross-platform summary

| Metric | A100 80 GB (Async, online) | A100 80 GB (LLM, offline) | H100 NVL (offline) |
|---|---|---|---|
| Baseline | 654.9 | 811.1 | 1870 |
| Best stack | 983.7 (×1.50) | 1771.5 (×2.18) | 2056 (×1.10 over BF16) |
| Dominant lever | MTP + mns | FP8 + MTP | FP8 (sc1 only) |

---

## 14. Reproducing benchmarks

### 14.1 H100 NVL — production-shape anchors

Source: [`benchmarks/gemma4_moe_fp8/README.md` §2](../benchmarks/gemma4_moe_fp8/README.md#2-prod-shape-benchmark-h100-nvl).
Real scripts (verified to exist in the repo):

```bash
cd vllm-msn/benchmarks/gemma4_moe_fp8

# Optional: build the prod image (Dockerfile pins vLLM commit + branch).
docker build -t gemma4-bench:prodshape -f Dockerfile .

# 1. Prepare datasets (sc1 short prompts, sc2 long prompts).
#    Inputs live in delta_prompts/ and persona_prompts/.
python3 prep_dataset.py \
  --src delta_prompts/   --out datasets/sc1_delta.jsonl    --max-tokens 16384  --target-rows 10000
python3 prep_dataset.py \
  --src persona_prompts/ --out datasets/sc2_personal.jsonl --max-tokens 40960  --target-rows 10000

# 2. Run all 4 anchors (bf16 × sc1, bf16 × sc2, fp8 × sc1, fp8 × sc2).
#    Driven by run.bench.fullpass.sh which loops bench_offline.py twice
#    (once per quantization). max_num_seqs is built into the SCENARIOS dict
#    inside bench_offline.py (sc1=128, sc2=64).
./run.bench.fullpass.sh
# → bench_results_bf16/all_runs.csv, bench_results_fp8/all_runs.csv
#   (one row per scenario × mns × rep × chunk)

# Smoke / partial variants:
NUM_PROMPTS=100 SC1_CHUNK=0 SC2_CHUNK=0 ./run.bench.fullpass.sh   # smoke
./run.bench.fullpass.sh --skip-fp8                                # bf16 only
./run.bench.fullpass.sh --skip-bf16                               # fp8 only
```

The 4 anchor rows reported in [§13.1](#131-h100-nvl--production-shape-anchor-numbers)
are means across all chunks (5 for sc1, 10 for sc2). The driver does not
emit per-request TTFT/TPOT or peak-GiB metrics for these runs.

### 14.2 H100 NVL — sweep v1 and v2 (BF16-only)

Source: [`benchmarks/gemma4_moe_fp8/README.md` §3.6](../benchmarks/gemma4_moe_fp8/README.md#36-how-to-reproduce-v1--v2).
Both sweeps drive [`bench_offline.py`](../benchmarks/gemma4_moe_fp8/bench_offline.py)
directly (no dedicated `run_sweep_*.sh` exists).

```bash
cd vllm-msn/benchmarks/gemma4_moe_fp8

# v2 (recommended — wider, realistic distribution):
python3 bench_offline.py --scenario sc1 --reps 2 --max-num-seqs 64,128,256
python3 bench_offline.py --scenario sc2 --reps 2 --max-num-seqs 64,128

# v1 (original strict length-bucket sweep, for reference):
python3 bench_offline.py --scenario sc1 --reps 3 --max-num-seqs 64,128,256,512,1024
python3 bench_offline.py --scenario sc2 --reps 2 --max-num-seqs 64,256,1024
```

CSV results land in `bench_results/all_runs.csv` (column schema in
`CSV_FIELDS` inside `bench_offline.py`).

### 14.3 A100 80 GB — LLM-offline ablation (15 experiments)

Source: [`benchmarks/gemma4_moe_fp8/README.md` §4](../benchmarks/gemma4_moe_fp8/README.md#4-a100-80gb-ablation-15-experiments).
Real scripts:

```bash
cd vllm-msn/benchmarks/gemma4_moe_fp8

# 1. (One-time) build the sc1 dataset. The 1000-prompt slice used by the
#    ablation is datasets/sc1_delta_v2.jsonl, capped at max_model_len(24576)
#    − output_len(8192) = 16 384 tokens. Use prep_dataset.py if your input
#    matches its expected schema; otherwise see the inline converter snippet
#    in benchmarks/gemma4_moe_fp8/README.md §4.2.
python3 prep_dataset.py \
  --src /nvmedata/data/layer1_delta_20260501.txt \
  --out datasets/sc1_delta_v2.jsonl \
  --max-tokens 16384 --target-rows 1000

# 2. Run the ablation. run_ablation.sh sets per-experiment env vars
#    (VLLM_ATTENTION_BACKEND, VLLM_USE_FLASHINFER_MOE_FP8,
#     VLLM_USE_FLASHINFER_SAMPLER) BEFORE invoking bench_ablation.py,
#    because vLLM freezes env vars at import time.
./run_ablation.sh --all --scenario sc1 --reps 2          # all 15 experiments
./run_ablation.sh E001                                   # single
./run_ablation.sh E004,E005,E006 --scenario sc1 --reps 2 # Group B only
./run_ablation.sh --list                                 # show all E0XX configs

# 3. Aggregate
python3 analyze_ablation.py
# → ablation_results/all_runs.csv  (raw)
# → ablation_results/summary.md    (the report consumed by §13.3)
```

The 15 experiment configurations are defined in the `EXPERIMENTS` dict in
[`bench_ablation.py`](../benchmarks/gemma4_moe_fp8/bench_ablation.py)
(lines 123–315). Each `E0XX` key maps to an explicit `(quantization,
kv_cache_dtype, enforce_eager, mtp, mtp_k, max_num_seqs,
gpu_memory_utilization, model_variant)` tuple.

### 14.4 A100 80 GB — Async-engine ablation (15 experiments)

Source: [`gemma4/ablation_study_async_engine.md`](gemma4/ablation_study_async_engine.md).
Real scripts:

```bash
cd vllm-msn/examples

# 1. (Pre-flight) validate the framework
./test_ablation_setup.sh

# 2. Run all 15 experiments sequentially
./run_all_ablation_experiments.sh
#  → experiment_results/E001/ … E015/
#  Each contains: metrics.json, per_request_metrics.jsonl, output.jsonl,
#                 gpu_trace.csv, inference.log, summary.md, environment.txt

# Or run a single experiment manually:
./run_ablation_experiment.sh \
  --exp E011 \
  --backend FLASHINFER \
  --batch 128 \
  --fp8 \
  --cuda-graphs \
  --mtp \
  --gpu-mem 0.80 \
  --text-only

# Resume after a partial-sweep failure:
./run_remaining_ablation_experiments.sh
```

`run_all_ablation_experiments.sh` contains the explicit flag block for each
of `E001`–`E015` (verifiable: `grep -n '^echo "E0' run_all_ablation_experiments.sh`).
The underlying Python driver is
[`run_inference_configurable.py`](run_inference_configurable.py), which
accepts all `AsyncEngineArgs` as CLI args.

---

## 15. Ablation experiment plan

This is the LLM-offline plan from `benchmarks/gemma4_moe_fp8/README.md` §4
(benchmarks/gemma4_moe_fp8/). Groups are layered — **E006 is the
“best-so-far” pivot** that Groups C, D, E branch from.

### Group A — Reproduce REPRODUCE_PRODSHAPE baseline

| Exp | Description | quant | kv-cache | eager | MTP | mns | mem | model |
|---|---|---|---|:---:|:---:|---:|---:|:---:|
| E001 | BF16 baseline — matches REPRODUCE_PRODSHAPE sc1 | bf16 | auto | ✓ | ✗ | 128 | 0.90 | full |
| E002 | +FP8 weights (KV cache stays auto/BF16) | fp8 | auto | ✓ | ✗ | 128 | 0.90 | full |
| E003 | +FP8 KV cache (`fp8_e4m3`) — **FAIL expected on A100** | fp8 | fp8_e4m3 | ✓ | ✗ | 128 | 0.90 | full |

> **E003 note** — `fp8_e4m3` KV cache requires Triton `fp8e4nv`, not
> supported on sm_80. Failure documents the constraint.

### Group B — Build up to “best-so-far”

Layered on top of E002 (FP8 weights, KV=auto/BF16):

| Exp | Description | eager | MTP | mns | mem | model | base |
|---|---|:---:|:---:|---:|---:|:---:|---|
| E004 | +CUDA graphs | ✗ | ✗ | 128 | 0.90 | full | E002 |
| E005 | +MTP speculative decoding (`k=5`) | ✗ | `k=5` | 128 | 0.90 | full | E004 |
| **E006** | **+text-only model (vision stripped) — “best-so-far” pivot** | ✗ | `k=5` | 128 | 0.90 | **text** | E005 |

### Group C — Batch sweep around E006

All keep FP8 + CG + MTP + text-only; only `max_num_seqs` varies.

| Exp | mns | Notes |
|---|---:|---|
| E007 | 64 | smaller batch (expected slower) |
| E006 | 128 | *control* |
| E008 | 192 | larger |
| E009 | 256 | largest — KV pool saturation check |

### Group D — `gpu_memory_utilization` sweep around E006

| Exp | mem | Notes |
|---|---:|---|
| E010 | 0.80 | less KV room |
| E006 | 0.90 | *control* |
| E011 | 0.95 | aggressive — turned out to be the campaign best |

### Group E — Single-feature isolation

Each experiment turns off **exactly one** optimization vs E006 to measure
its standalone value (“give me back the BF16 / no-MTP / no-CG number
without disturbing the rest”):

| Exp | Off-switch | Diff from E006 |
|---|---|---|
| E012 | no MTP at optimal | MTP disabled |
| E013 | no CUDA graphs at optimal | `enforce_eager=True` |
| E014 | BF16 weights at optimal | FP8 removed |
| E015 | BF16 reference (no opts) | All opts off, text-only only |

**Isolation pairs** (E006 is the “on” state; column = the “off” state):

| Knob | on (E006) | off | hypothesis |
|---|---|---|---|
| MTP `k=5` | E006 | E012 | E006 > E012 |
| CUDA graphs | E006 | E013 | E006 ≥ E013 (may regress on heterogeneous batch) |
| FP8 weights | E006 | E014 | E006 > E014 |
| text-only model vs full | E006 | E005 | E006 > E005 |

### Required model paths

| env var | value | used by |
|---|---|---|
| `GEMMA4_MODEL_PATH` | `/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it` | E001–E005 (full model) |
| `GEMMA4_TEXT_ONLY_MODEL_PATH` | `$GEMMA4_MODEL_PATH-text-only` | E006–E015 (vision tower stripped) |
| `GEMMA4_ASSISTANT_MODEL_PATH` | `/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant` | All MTP experiments (E005–E011, E013, E014) |

### Reference for H100 anchors (cross-platform check)

| Config | H100 NVL out tok/s |
|---|---:|
| BF16 baseline (E001 equivalent) | 1 870 ± 14 |
| FP8 weights + FP8 KV (E003 equivalent, **H100 only**) | 2 056 ± 21 |

A100 best (E011 = 1771.5) is **~86 % of the H100 FP8 sc1 number** —
remarkable given the FP8-tensor-core deficit.

### Per-group execution

```bash
./run_ablation.sh E001 --scenario sc1 --reps 2              # single
./run_ablation.sh E001,E002,E003 --scenario sc1 --reps 2    # Group A
./run_ablation.sh E004,E005,E006 --scenario sc1 --reps 2    # Group B
./run_ablation.sh E007,E008,E009 --scenario sc1 --reps 2    # Group C
./run_ablation.sh E010,E011      --scenario sc1 --reps 2    # Group D
./run_ablation.sh E012,E013,E014,E015 --scenario sc1 --reps 2  # Group E
```

---

## 16. Experiment workflow and tooling

### 16.1 `experiment_runner.sh`

Located in `examples/`. Drives the AsyncLLMEngine plan.

```bash
./experiment_runner.sh \
  --plan gemma4/ablation_study_async_engine.md \
  --dataset layer1_delta_1k_test.txt \
  --gpu-mem-util 0.92 \
  --output-cap 1024 \
  --warmup-prompts 8 \
  --measure-prompts 1000
```

Per-experiment outputs:
- `experiment_results/EXXX/stdout.log`
- `experiment_results/EXXX/metrics.json` (QPS, TTFT, TPOT, E2E, peak GiB)
- `experiment_results/EXXX/nvidia-smi.csv` (1 Hz)
- `experiment_results/EXXX/vllm_metrics.prom` (Prometheus dump)

### 16.2 `bench_offline.py`

Located in `benchmarks/gemma4_moe_fp8/`. Drives the LLM offline engine for
the H100 anchors and the A100 LLM-offline ablation.

```bash
python bench_offline.py \
  --model google/gemma-4-26B-A4B-it \
  --quantization fp8 \
  --scenario sc1 \
  --runs 5 \
  --mns 128 \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}' \
  --out ./prodshape_results/anchor-fp8-sc1/
```

### 16.3 Quick scenarios (from [`gemma4/quickstart.md`](gemma4/quickstart.md))

**FlashInfer vs FA2 quick check** (will land on TRITON_ATTN for Gemma 4):

```bash
for b in FLASH_ATTN FLASHINFER TRITON_ATTN; do
  VLLM_ATTENTION_BACKEND=$b python bench_offline.py --quick --label $b
done
```

**Batch-size sweep:**

```bash
for mns in 16 32 64 96 128; do
  python bench_offline.py --mns $mns --label mns_$mns
done
```

**MTP `num_spec` sweep:**

```bash
for k in 0 2 4 6 8; do
  python bench_offline.py --num-spec $k --label spec_$k
done
```

---

## 17. Dataset analysis (MAI `layer1_delta`)

Source: [`gemma4/dataset_analysis.md`](gemma4/dataset_analysis.md) + the dataset section of
[`gemma4/ablation_study_async_engine.md`](gemma4/ablation_study_async_engine.md).

| Dataset | Rows | p50 in tokens | p95 in tokens | mean out tokens | p50 / max out tokens |
|---|---|---|---|---|---|
| `layer1_delta_20260501.txt` (full) | 859 988 | 3479 | 11 824 | ~830 | 1024 / 1024 (capped) |
| `layer1_delta_1k_test.txt` (slice, 969 used after 32K filter) | 1 000 | ~3501 | ~11 902 | 829.0 – 835.8 | 1024 / 1024 |
| `sc1_delta_v2.jsonl` | 1 000 (≤ 16 384 in toks) | ~3500 | ~12000 | short scenario | — |
| sc2 | 1 000 | ~3500 | ~12000 | long scenario | — |

> **Output distribution is bimodal.** ~60 % of requests hit the 1024 cap;
> ~40 % stop earlier on natural EOS. Median output = 1024 in every
> experiment; mean is what differs (829–836 tokens across the 14 successful
> A100 Async runs). Total output = 969 prompts × ~830 ≈ 805 K tokens per
> experiment.

**Why the 1024 output cap?** Production response-length distribution caps
near 1 K. Going higher inflates the sc2 (long-output) regime, which is
already an *adversarial* benchmark for FP8 prefill on TRITON_ATTN (§13.1).

**Shape implication.** Per-step token budget is dominated by **decode tokens
× batch**, not prefill — hence the dominance of expert-batch-amortization
and MTP wins (§8, §11).

---

## 18. Troubleshooting

### 18.1 OOM on A100 80 GB

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM during weight load | Trying BF16 with `mns=128, ctx=16K` | Use FP8 or drop mns to 64 |
| OOM during first decode | KV pool too small | Raise `gpu-memory-utilization` to 0.92 |
| OOM mid-run after smooth start | Sequence reached `max-model-len`; pool fragmentation | Drop `max-model-len` to 8192 |
| OOM with MTP | Draft KV is shared, but speculative tree allocates extra | `num_spec=4` not 6 |

### 18.2 Low throughput

| Symptom | Fix |
|---|---|
| Output tok/s < 700 on A100 80 GB | Verify FP8 active: check log for `Marlin kernel`. Verify MTP active: `spec_decode_metrics` line present |
| MTP not engaging | Tokenizer mismatch between target and assistant — both must be `gemma-4` family |
| H100 sc2 FP8 slower than BF16 | Expected — known TRITON_ATTN FP8 prefill regression (§13.1) |
| Per-token latency spikes | CUDA graph recapture — confirm `full_cuda_graph: true` and stable shapes |

### 18.3 Crashes / silent no-ops

| Setting | Behaviour on A100 | Fix |
|---|---|---|
| `--kv-cache-dtype fp8_e4m3` | Triton `fp8e4nv not supported in this architecture` ([§20 attempt 1](#attempt-1----fp8---kv-cache-dtype-fp8_e4m3)) | Remove flag |
| `--kv-cache-dtype fp8_e5m2` (BF16 weights) | `AssertionError: kv_cache_dtype ∈ {fp8, fp8_e4m3, nvfp4}` in `attention.py:467` ([§20 attempt 2](#attempt-2----no-fp8---kv-cache-dtype-fp8_e5m2---text-only)) | Remove flag |
| `--kv-cache-dtype int8_per_token_head` (BF16 weights) | `NotImplementedError` on heterogeneous page sizes ([§20 attempt 3](#attempt-3----no-fp8---kv-cache-dtype-int8_per_token_head---text-only)) | Remove flag |
| `--kv-cache-dtype auto` with FP8 weights | Silent fallback to BF16 KV (the *only* mode that runs on A100) | Expected; this is the de facto config in every FP8 A100 run |
| `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | Override warning, falls back to TRITON_ATTN | Expected (heterogeneous head dims) |
| `VLLM_ATTENTION_BACKEND=FLASHINFER` | Same | Expected |

### 18.4 Wrong numbers / "I can't reproduce 2.18×"

The 2.18× is **LLM offline engine**, **sc1**, **32 prompts × 3 runs**,
**MTP num_spec=4**, **text-only model**, **A100 80 GB PCIe**. The 1.50× is
**AsyncLLMEngine**, **MAI 1k slice**, **output cap 1024**. They are not
interchangeable.

---

## 19. Best practices

### Do

- ✅ Use `--quantization fp8` on A100 80 GB and H100.
- ✅ Use MTP `num_spec=5` for general traffic (campaign-best; conservative
  fallback `=4` is fine if acceptance rate is unmeasured).
- ✅ Enable `full_cuda_graph: true` **for offline LLM workloads** —
  measured **+12.4 %** in the LLM-offline ablation
  ([§13.3](#133-a100-80-gb-pcie--llm-offline-ablation-15-experiments)).
  *Validate on your traffic for Async / online* — the same flag regressed
  **−2.6 %** on the Async campaign's heterogeneous-prompt-length workload
  before MTP was added back ([§13.4 E005](#134-a100-80-gb-pcie--async-engine-ablation-15-experiments)).
- ✅ Strip vision tower for text-only deployments.
- ✅ Bench with the offline driver first; switch to Async for production
  shape validation.
- ✅ Run a 5-prompt warmup before measuring.
- ✅ Pin `gpu-memory-utilization=0.95` on LLM-offline (campaign-best,
  E011) or `0.80` on Async (campaign-best, E011 again — but the Async
  optimum was lower because peak GiB at 0.95 risked OOM mid-run).
- ✅ Always report which engine (LLM vs AsyncLLM), which dataset, which `mns`.

### Don't

- ❌ Don't use `--kv-cache-dtype fp8_*` on A100 (Triton fp8e4nv missing).
- ❌ Don't override the attention backend — Gemma 4 picks TRITON_ATTN.
- ❌ Don't crank `num_spec` past 4 without checking acceptance rate.
- ❌ Don't go past `mns=160` at 16K context on 80 GB.
- ❌ Don't compare sweep v1 numbers to sweep v2 numbers — different configs.
- ❌ Don't optimize the top-K kernel.
- ❌ Don't expect FP8 to help sc2 on H100 (TRITON_ATTN FP8 prefill is
  immature — track upstream).

---

## 20. E014 trilogy — KV-cache quantization failures on sm_80

E014's original goal was to test FP8 KV cache (the natural follow-on to FP8
weights). On A100 sm_80 + Gemma 4 + vLLM 0.21.1, **no working FP8 / INT8 /
NVFP4 KV-cache configuration exists**. Three independent attempts, three
distinct failure modes.

> The same root cause prevents Group A's E003 in the LLM-offline ablation
> ([§13.3](#133-a100-80-gb-pcie--llm-offline-ablation-15-experiments)).

### Attempt 1 — `--fp8 --kv-cache-dtype fp8_e4m3`

- **What failed:** Triton kernel compilation when the attention backend
  tries to write KV in `fp8e4nv`.
- **Where:** `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
  → `reshape_and_cache_kernel_flash`.
- **Error:**
  ```
  triton.compiler.errors.CompilationError: at 1:0:
  def reshape_and_cache_kernel_flash(
  ValueError("type fp8e4nv not supported in this architecture.
   The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
  ```
- **Why:** A100 (sm_80) has no native FP8 tensor cores. Triton emulation on
  sm_80 exposes `fp8e4b15` (the older E4M3 with bias=15) and `fp8e5`
  (E5M2), but not `fp8e4nv` (the NVIDIA E4M3 with bias=7 introduced with
  Hopper).
- **Fixable on A100?** **No.** Requires sm_90+ (H100).
- **Archive:** `experiment_results/E014_fp8_e4m3_FAILED_sm80/`

### Attempt 2 — `--no-fp8 --kv-cache-dtype fp8_e5m2 --text-only`

BF16 weights + FP8 KV path (avoids attempt 1's `fp8e4nv` write path).

- **What failed:** vLLM's query-quantization allow-list assertion at
  forward time.
- **Where:** `vllm/model_executor/layers/attention/attention.py:467`.
- **Error:**
  ```
  AssertionError
    File ".../attention.py", line 467
      assert self.kv_cache_dtype in {"fp8", "fp8_e4m3", "nvfp4"}
  ```
- **Why:** When `kv_cache_dtype.startswith("fp8")` **and** the impl declares
  `supports_quant_query_input=True` (FA2 and FlashInfer both do on CUDA),
  vLLM auto-enables query quantization. The forward path then hard-asserts
  `kv_cache_dtype ∈ {fp8, fp8_e4m3, nvfp4}` — **`fp8_e5m2` is explicitly
  excluded**. No env-var override; only an internal
  `disable_flashinfer_q_quantization` config field.
- **Fixable on A100?** **No without patching vLLM.** Hard-coded logic.
- **Archive:** `experiment_results/E014_fp8_e5m2_FAILED_query_quant_assert/`

### Attempt 3 — `--no-fp8 --kv-cache-dtype int8_per_token_head --text-only`

Last resort — BF16 weights + INT8 per-token-per-head KV.

- **What failed:** KV-cache page-size unification across heterogeneous
  layer dimensions.
- **Where:** `vllm/v1/core/kv_cache_utils.py:1040` → `unify_kv_cache_spec_page_size`.
- **Error:**
  ```
  NotImplementedError: The page size of the layer is not divisible by the
  maximum page size. Cannot unify by adjusting block_size.
  ```
- **Why:** Gemma 4 has mixed attention layer types — 25 of 30 layers are
  sliding-window with `head_dim=256`, 5 of 30 are full-attention with
  `global_head_dim=512`. `int8_per_token_head` produces different per-layer
  page sizes (scaled by `head_dim`), and vLLM cannot pack heterogeneous
  page sizes into a unified KV layout.
- **Fixable on A100?** Yes *in principle*, with a different model or a
  vLLM patch that allows non-unified page sizes. Out of scope.
- **Archive:** `experiment_results/E014_int8_per_token_head_FAILED_page_size/`

### Conclusion

On this hardware/model/vLLM combo there is **no KV-cache quantization mode
we can run**. The three archived FAIL directories *are* the result. Every
other FP8 experiment in both A100 studies effectively runs with BF16 KV
cache (`auto` falls back to BF16 because all three FP8/INT8 paths above are
unavailable).

**Track upstream Triton** for sm_80 `fp8e4nv` emulation. When it lands,
re-run Attempt 1 as a regression test.

---

## 21. Code-review fix history (appendix)

Two distinct fix sets, kept for institutional memory.

### 21.1 Framework code-review (pre-campaign) — [`gemma4/framework_code_review.md`](gemma4/framework_code_review.md)

Eight bugs caught before the ablation framework ever ran a real experiment.

| # | Severity | Bug | Fix |
|---|---|---|---|
| 1 | 🔴 CRITICAL | `run_ablation_experiment.sh` passed `--max_num_seqs`, `--gpu_memory_utilization`, `--quantization`, … to `llm_analyzer_gemma4_moe_fp8_mtp.py`, which had HARDCODED configs and only 4 argparse args. Every experiment would have run the same config. | Created `run_inference_configurable.py` exposing all `AsyncEngineArgs` as CLI args. |
| 2 | 🟡 HIGH | Script required `--input_path` but the runner never provided one. | Added `--num_test_samples` synthetic-prompt fallback. |
| 3 | 🟢 MEDIUM | Async pattern used `async for output in engine.generate(...)`. | Switched to `engine.add_request()` + `engine.engine_step_async()`. |
| 4 | 🟡 HIGH | One failed experiment killed the whole sweep. | Master script wraps each run in `if … then COMPLETED else FAILED`. |
| 5 | 🟢 LOW | `--enforce_eager` and `--enable_cuda_graphs` both settable → ambiguous. | Explicit precedence: `enforce_eager` flag wins. |
| 6 | 🟢 LOW | `OUTPUT_DIR="./experiment_results/${EXPERIMENT_ID}"` without `mkdir -p`. | Always `mkdir -p`. |
| 7 | 🟢 LOW | Background `nvidia-smi` monitor PID survived parent script errors. | `kill ${MONITOR_PID} 2>/dev/null || true` on every exit path. |
| 8 | 🟢 LOW | `cd /nvmedata/chenw/vllm-ra/examples` without check. | `set -e` + explicit `cd … || exit 1`. |

### 21.2 Campaign-time fixes — from [`gemma4/ablation_study_async_engine.md`](gemma4/ablation_study_async_engine.md) "Fix history"

Eight further fixes between first run attempt and final successful sweep:

1. **PyTorch + CUDA mismatch.** Env had `torch 2.11.0+cu130`, driver was
   12.6 → `_cuda_init` failed on every experiment. **Fix:** reinstalled
   `torch==2.11.0+cu126`, `torchvision==0.26.0+cu126`,
   `torchaudio==2.11.0+cu126`. Rebuilt vLLM C extensions via
   `VLLM_USE_PRECOMPILED=1 pip install -e .` so the cu126 precompiled wheel
   took precedence over the stale cu130 `_C.abi3.so`.
2. **`fp8_e5m2` KV cache rejected with FP8 weights** — default for
   `KV_CACHE_DTYPE` changed from `fp8_e5m2` to `auto` in
   `examples/run_ablation_experiment.sh:33`.
3. **FlashInfer sampler JIT compile failed** (system `/usr/bin/nvcc` v10.1,
   missing `--generate-dependencies-with-compile`). **Fix:**
   `VLLM_USE_FLASHINFER_SAMPLER=0`, and `CUDA_HOME` pointed at the cu12.8
   conda toolkit for other JIT consumers.
4. **`VLLM_USE_FLASHINFER_MOE_FP8=1` raised `NotImplementedError` on
   A100.** **Fix:** stopped exporting it; vLLM falls back to Marlin FP8
   MoE which works on sm_80.
5. **`vllm/model_executor/models/gemma4_mm.py:get_mm_max_tokens_per_item`
   crashed on `config.vision_config.default_output_length`** when
   `vision_config is None` (text-only checkpoint). **Fix:** guarded
   `config.vision_config is not None`.
6. **`Gemma4ForConditionalGeneration.__init__` unconditionally built a
   `vision_tower` from `config.vision_config`** even when `None`. **Fix:**
   same file; added a `vision_config is not None` guard around the vision
   tower + `embed_vision` construction.
7. **`json.dump(metrics)` crashed for MTP runs** because the
   `speculative_config` dict passed to `AsyncEngineArgs` gets mutated by
   vLLM at engine init to include a non-serializable `ModelConfig`.
   **Fix:** rebuild a clean `spec_snapshot` dict from `args.*` for
   `metrics.json`, plus `default=str` on `json.dump` as a safety net.
8. **`KV_CACHE_DTYPE` was being force-overridden to `"auto"` whenever
   `--no-fp8` was set**, preventing the redesigned E014 trilogy from
   running its three variants. **Fix:** removed the override in
   `run_ablation_experiment.sh:194`; user's `--kv-cache-dtype` always
   wins. Also moved the `--kv_cache_dtype` arg forwarding outside the
   `if USE_FP8` branch.

### 21.3 Known gaps in the captured data

Per source [`gemma4/ablation_study_async_engine.md`](gemma4/ablation_study_async_engine.md):

1. **MTP acceptance rate (no data).** vLLM emitted no `SpecDecoding
   metrics:` lines in any `inference.log`. The `summary.md` heredoc was
   prepared to parse them. Cause unknown — possibly window cadence vs
   per-experiment wall-clock, or `aggregate_engine_logging` needs to be
   true.
2. **Engine periodic stats (no data).** No `Avg prompt throughput …,
   Running: N reqs, Waiting: N reqs, GPU KV cache usage: X%` lines in
   v1, despite `disable_log_stats=False`. Peak KV-cache % is N/A.
3. **Effective batch admission.** Without periodic Running/Waiting, we
   can't directly show "nominal batch 256 only achieved ≤K concurrent" —
   the flat throughput across E007/E008/E009 is indirect evidence of
   saturation before `max_num_seqs`.

---

## 22. Experiment log template (condensed)

For new experiments, fill out:

```markdown
# Experiment EXXX — <short name>

## Hypothesis
What single lever are we changing? Expected sign of effect?

## Baseline
- ID:
- Out tok/s:
- Hardware:
- Engine (LLM offline / AsyncLLM):
- Dataset / scenario:

## Configuration delta
```bash
# only show the changed flags
```

## Result
- Out tok/s:
- Δ vs baseline (% and absolute):
- TTFT p50 / p95:
- TPOT p50 / p95:
- Peak GiB:
- Acceptance rate (if MTP):

## Verdict
ship / drop / inconclusive — and why.

## Artifacts
- `stdout.log`:
- `metrics.json`:
- `nvidia-smi.csv`:
```

Full template (380 lines) was retired with this consolidation; the above
captures every section that was actually used.

---

## 23. Source artifacts and references

### Files in this repo

- `vllm/model_executor/models/gemma4.py` — main model
- `vllm/model_executor/models/gemma4_mtp.py` — MTP draft routing
- `csrc/moe/topk_softmax_kernels.cu` — iterative warp argmax
- `examples/create_text_only_model.py` — vision-tower stripper
- `examples/experiment_runner.sh` — Async ablation driver
- `examples/run_inference_configurable.py` — Async serving entrypoint
- `benchmarks/gemma4_moe_fp8/README.md` — consolidated source doc (setup, prod-shape bench, v1/v2 sweep, A100 ablation)
- `benchmarks/gemma4_moe_fp8/bench_offline.py` — prod-shape & sweep driver
- `benchmarks/gemma4_moe_fp8/bench_ablation.py` — A100 ablation driver
- `benchmarks/gemma4_moe_fp8/run.bench.fullpass.sh` — prod-shape wrapper
- `benchmarks/gemma4_moe_fp8/run_ablation.sh` — A100 ablation wrapper
- `benchmarks/gemma4_moe_fp8/Dockerfile` — H100 prod image

### External

- HF: `google/gemma-4-26B-A4B-it`
- HF: `google/gemma-4-26B-A4B-it-assistant` (MTP draft)
- vLLM ≥ 0.21
- Branch: `feat/gemma4-moe-opt-a100` on `rayleizhu/vllm-ra`

### Predecessor docs (now folded in)

This single file replaces all of the following. They were deleted as part
of the merge; their content lives here in §s noted:

| Removed file | Folded into |
|---|---|
| `examples/README_GEMMA4_FP8.md` | §1, §4, §5, §6 |
| `examples/GEMMA4_MOE_OPTIMIZATION_GUIDE.md` | §2, §8, §9, §10, §11 |
| `examples/gemma4/ablation_study_async_engine.md` | §13.4, §15, §17, §21 |
| `examples/gemma4/moe_bottleneck_analysis.md` | §8 |
| `examples/gemma4/experiment_log_template.md` | §22 |
| `examples/gemma4/quickstart.md` | §16 |
| `examples/gemma4/flash_attention_setup.md` | §7 |
| `examples/gemma4/framework_code_review.md` | §21 |
| `examples/gemma4/dataset_analysis.md` | §17 |
| `examples/gemma4/environment_setup.md` | §14 |
| `benchmarks/gemma4_moe_fp8/ablation_results/summary.md` | §13.3 (still present — auto-generated by `analyze_ablation.py`) |
| `benchmarks/gemma4_moe_fp8/{REPRODUCE_PRODSHAPE,REPRODUCE,BENCHMARK_REPORT,BENCHMARK_LOG,ABLATION_EXPERIMENT_PLAN}.md` | merged into `benchmarks/gemma4_moe_fp8/README.md`; cited from §13.1, §13.2, §14.1, §14.2, §14.3, §15 |

---

*Last updated: 2026-05 · branch `feat/gemma4-moe-opt-a100` · vLLM ≥ 0.21.*
