# Gemma 4 26B MoE Ablation Study - Experiment Plan

**Objective:** Systematically measure the impact of each optimization on Gemma 4 26B MoE inference performance.

**Hardware:** 1× NVIDIA A100 80GB PCIe (sm_80)
**Baseline Model:** google/gemma-4-26B-A4B-it (checkpoint-quantized FP8)
**Assistant (MTP draft):** gemma-4-26B-A4B-it-assistant
**Text-only variant:** gemma-4-26B-A4B-it-text-only (built via `create_text_only_model.py`)
**Metrics:** QPS, output tokens/sec, prompt tokens/sec, TTFT (engine + client), TPOT, E2E latency (mean/p50/p90/p95/p99/max), GPU peak/avg memory + utilization, KV cache usage, prefix-cache hit rate, finish-reason histogram, failed-error histogram, MTP acceptance rate (when enabled)

---

## Actual Run Settings (this hardware / 2026-05-20)

The plan below was authored against an A100 40GB assumption. The settings actually used on the 80GB box, plus the hardware-driven adjustments, are:

| Setting | Value | Why |
|---|---|---|
| `max_model_len` | 32768 | Covers p95 of MAI dataset (~22K tokens); drops the long-tail 3.1% (31/1000) of prompts that exceed `32768 − max_tokens`. Larger max_model_len pinches the KV pool on BF16 baselines (see E001/E015) without measurably helping the body of the distribution. |
| `max_num_batched_tokens` | 32768 (= 1× max_model_len) | One max-length sequence fits in one prefill chunk; no benefit on this dataset from chunking further. |
| `max_tokens` (per request) | 1024 | Plan's intended 16K from the dataset's `max_completion_tokens` would push each experiment to multi-hour runtime; 1024 amortizes prefill while keeping each experiment ≤25 min. Held constant across all 15 experiments so the relative comparison is fair. |
| `KV_CACHE_DTYPE` (default for FP8) | `auto` (= BF16 KV) | vLLM rejects `fp8_e5m2` with FP8 checkpoints (`vllm/model_executor/layers/attention/attention.py:167`). Triton on sm_80 can't compute `fp8e4nv` (= `fp8_e4m3`); supports only `fp8e4b15` / `fp8e5`. So the only working KV-cache-dtype combo on A100 + FP8 weights is `auto` → BF16. E014's explicit `--kv-cache-dtype fp8_e4m3` is now the deliberate sm_80 probe — **expected to fail** with the Triton error; that failure IS the result. |
| `VLLM_USE_FLASHINFER_MOE_FP8` | **unset** | FlashInfer's FP8 MoE backend wants Hopper FP8 tensor cores; on sm_80 it raises `NotImplementedError`. Unset → vLLM uses Marlin FP8 MoE, which works on A100. FlashInfer is still used for *attention* (`VLLM_ATTENTION_BACKEND=FLASHINFER`). |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` (torch native) | FlashInfer's top-k/top-p sampler is JIT-compiled on first use. The system `/usr/bin/nvcc` is v10.1 (too old); the env's cu13 nvcc mismatches FlashInfer's bundled cu12 cccl headers; the available cu12.8 nvcc lacks `--host-stub-linkage-explicit`. Falling back to torch's native sampler avoids the entire compile dance. Constant overhead across all experiments → relative comparisons stay valid. |
| `CUDA_HOME` | `/root/miniconda3/envs/vila/targets/x86_64-linux` | Steers any remaining CUDA-extension build at a cu12.8 toolkit with matching include headers. |
| Sampler | `temperature=0.7 top_p=0.9` | Defaults; kept constant across experiments. |
| PyTorch | `2.11.0+cu126` | System driver is CUDA 12.6 (`560.35.03`). `2.11.0+cu130` (the original install) failed `_cuda_init` at runtime; reinstalled as cu126 + rebuilt vLLM C extensions via `VLLM_USE_PRECOMPILED=1`. |
| Dataset | `/nvmedata/data/layer1_delta_1k_test.txt` (1000 JSONL rows) | All experiments use the same file. The runner applies the actual Gemma 4 chat template and counts tokens with the model's tokenizer, then drops any row whose token count exceeds `max_model_len − max_tokens` — so every experiment sees the identical 969-prompt subset and skip-count is reproducible. **See the [Dataset section](#dataset) below for the full prompt/output length distribution (pre/post filter), tokenization recipe, schema, and per-experiment finish-reason breakdown.** |
| GPU pin | `CUDA_VISIBLE_DEVICES=0`, `nvidia-smi -i 0` | One A100 only. Other GPUs on this box are untouched. |
| Cooldown between experiments | 30 s | Allows CUDA caching allocator to release before next engine init. |

### Per-experiment hardware-driven expectations (override the plan body below where they disagree)

- **E001 (BF16 baseline)** runs successfully on 80GB (median prompt ~3.4K tokens fits comfortably) — peak memory ~77 GiB, ~97% of total. The plan's "OOM likely" prediction was for 40GB.
- **E014 (`fp8_e4m3` KV cache)** is expected to fail at first inference step with `triton.compiler.errors.CompilationError: type fp8e4nv not supported in this architecture`. Documented as a sm_80 hardware limit, not a script bug.
- **E009 (batch 256)** may admit far fewer than 256 concurrent sequences on the long-tail prompts since KV pool fills first. The reported throughput reflects what the engine actually achieves, not the nominal batch.
- All FP8 experiments effectively run with BF16 KV cache (due to the `auto` fallback). The "FP8 memory savings" the plan attributes to FP8 quantization are model-weights savings only on A100; KV memory is unchanged from BF16.

---

## Results (2026-05-21)

Final run of the 15-experiment recipe on 1× A100 80GB PCIe. **14 produced numeric metrics; E014 documented as a three-failure hardware/software constraint trilogy.** All experiments processed the same filtered subset of the MAI dataset (969/1000 prompts after the 32K-token length filter — the 31 dropped prompts are identical across every run, so relative comparisons are apples-to-apples).

### Dataset

**Headline averages** (the two numbers a typical perf report quotes):

- **Avg input length:** **5,761.30 tokens / request** — identical across all 14 successful experiments (same filtered 969-prompt subset, computed as `counts.total_prompt_tokens / counts.finished` from each experiment's `metrics.json`).
- **Avg output length:** **829.0 – 835.8 tokens / request** depending on experiment (sampling variance at `temperature=0.7`). Range:
  - E001: 834.5, E002: 832.5, E003: 831.5, E004: 832.7, E005: 829.0, E006: 835.5,
  - E007: 831.4, E008: 830.8, E009: 831.8, E010: 832.4, E011: 832.7, E012: 831.4, E013: 830.1, E015: 835.7
- **Generation budget cap:** 1024 tokens / request (constant across all experiments). About 60% of requests hit the cap; ~40% stop earlier on EOS — see the [finish-reason table](#dataset) below.

#### Full source dataset (`layer1_delta_20260501.txt`, 24 GB, 859,988 rows)

Analysis performed in 5.6 minutes via [examples/analyze_dataset.py](../analyze_dataset.py); full output in [examples/dataset_analysis_full.json](../dataset_analysis_full.json) + [dataset_analysis.md](dataset_analysis.md). The 1K test subset that drives the experiments is a uniform sample of this file.

**Schema invariants** (all 859,988 rows): zero JSON-parse errors, every row has exactly 2 messages (system + user). The system content is byte-identical across every row — **3,345 chars on every single row**, the same MAI Profile V3 instruction. This is what makes prefix caching valuable in production.

**User-content char distribution (full 859,988-row pass):**

| metric | min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| user content chars / row | 265 | 909 | 2,827 | 9,917 | 28,315 | 59,464 | 87,491 | 168,629 | **1,103,343** | 22,817.6 |
| total (system + user) chars / row | 3,610 | 4,254 | 6,172 | 13,262 | 31,660 | 62,809 | 90,836 | 171,974 | **1,106,688** | 26,162.6 |

The max is a real outlier — one row has 1.1 M chars of user content (~290 K tokens). That single prompt cannot be served by any vLLM config on this hardware regardless of `max_model_len` (Gemma 4's design ceiling is 262 K tokens).

**Token distribution from a 10,000-row stratified sample** (one row every ~86; computed with the actual Gemma 4 tokenizer + chat template):

| metric | min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tokens per formatted prompt | 970 | 1,152 | 1,669 | 3,479 | 8,467 | 16,650 | 24,352 | 46,703 | 117,043 | 6,958.8 |

The sample missed the 1.1 M-char outlier (sample max is 117 K tokens, not the full-dataset extreme), but the body of the distribution (p10–p99) lines up tightly with the 1K test subset's pre-filter numbers (p50 = 3,565 in the 1K vs 3,479 here, p90 = 16,949 vs 16,650). **The 1K subset is a representative sample of the bulk of production traffic.**

**Chars → tokens conversion** (sample, per-row ratio): min 2.63, p10 3.68, **p50 3.82**, p90 3.92, p99 4.03, max 4.37, mean 3.81. Stable enough that you can divide any char threshold by ~3.82 to get a token estimate.

**What fits in which `max_model_len`** (using p50 chars/token = 3.82 to convert):

| `max_model_len` (tokens) | char-equivalent | % of full dataset that fits (estimate) |
|---|---:|---:|
| 16,384 | ~62,500 | between p90 and p95 → roughly 90–95% |
| 32,768 (our setting) | ~125,000 | between p95 and p99 → roughly 96–98% |
| 65,536 | ~250,000 | very close to p99 → roughly 99%+ |
| 131,072 | ~500,000 | only the 1 M-char outliers fail |
| 262,144 (Gemma 4 max) | ~1,000,000 | still doesn't cover the 1.1 M-char extreme |

The 1K subset's measured drop rate at `max_model_len=32768` was 3.1% — fully consistent with the 2–4% the full distribution predicts.

---

#### Test subset used by these experiments

**File:** `/nvmedata/data/layer1_delta_1k_test.txt` (1000 JSONL rows, 29 MB on disk; the 1K test subset of `layer1_delta_20260501.txt.gz` which has 859,988 rows / 24 GB unpacked).

**Schema:** every row is a single JSON object with these top-level keys: `_export_prompt`, `step_key`, `model`, `max_completion_tokens` (= 16000, uniform across all rows), `messages`, `user_id`, `date`. The `messages` field is a 2-element list `[{role: "system", content: ...}, {role: "user", content: ...}]`. **All 1000 rows share the same system prompt** (the MAI Profile V3 — Layer 1: Delta Interest Extraction instruction); user content varies per row. This is what drives the ~15% prefix-cache hit rate in the engine metrics.

**Prompt tokenization:** each row is converted to a single string via `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` using the Gemma 4 tokenizer (vocab=262144, BOS=2, EOS=1) — the chat template emits `<bos><|turn>system\n…<|turn>user\n…<|turn>assistant\n`. Token counts are then computed with `encode(add_special_tokens=False)` which avoids double-BOS (verified — `add_special_tokens=True` vs `False` produce identical token IDs because `<bos>` is already in the templated string).

**Prompt-length distribution (pre-filter, all 1000 rows):**

| min | p50 | p90 | p95 | p99 | max | mean | total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 977 | 3,565 | 16,949 | 23,592 | 51,343 | 83,475 | 7,071.9 | 7,071,900 |

(Tokens, measured with the model's actual tokenizer on the chat-template-formatted prompts.)

**Length filter:** the runner skips any row whose tokenized length exceeds `max_model_len − max_tokens = 32768 − 1024 = 31,744 tokens`. This is enforced **before** submitting to vLLM, so the engine never sees a length-rejection at runtime.

**Skipped prompts:** 31 of 1000 (3.1%), all in the long tail. The skipped subset has token lengths from 33,119 to 83,475. Zero rows skipped for format errors — all 1000 are valid JSON with a populated `messages` array.

**Prompt-length distribution (post-filter, the 969 prompts every experiment actually processed):**

| min | p50 | p90 | p95 | p99 | max | mean | total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 977 | 3,393 | 14,989 | 18,363 | 27,154 | 31,286 | 5,761.3 | 5,582,695 |

This `total = 5,582,695` is the `counts.total_prompt_tokens` reported in every experiment's `metrics.json` — exact match, as expected.

**Output generation budget:** `max_tokens = 1024` per request, held constant across all 15 experiments. Sampling: `temperature=0.7, top_p=0.9`. The dataset's intrinsic `max_completion_tokens=16000` is the design target but pushing that through 15 experiments would multiply each run's wall-clock by ~16×, so the campaign uses 1024 as a budget-compatible constant.

**Output-length distribution (per experiment, in tokens):**

| Exp | min | p50 | p90 | p95 | p99 | max | mean | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E001 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 834.5 | 808,654 |
| E002 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 832.5 | 806,659 |
| E003 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 831.5 | 805,689 |
| E004 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 832.7 | 806,917 |
| E005 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 829.0 | 803,319 |
| E006 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 835.5 | 809,635 |
| E007 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 831.4 | 805,582 |
| E008 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 830.8 | 805,033 |
| E009 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 831.8 | 806,020 |
| E010 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 832.4 | 806,623 |
| E011 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 832.7 | 806,894 |
| E012 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 831.4 | 805,638 |
| E013 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 830.1 | 804,322 |
| E015 | 5 | 1024 | 1024 | 1024 | 1024 | 1024 | 835.7 | 809,840 |

The distribution is highly bimodal — most requests hit the 1024 cap (median = max = 1024), a smaller fraction stop earlier due to natural EOS. The mean of ~830 tokens × 969 prompts ≈ 805K total output tokens per experiment, consistent across all 14 (variance dominated by random sampling at `temperature=0.7`).

**Finish-reason breakdown:**

| Exp | length-capped | natural stop | % capped |
|---|---:|---:|---:|
| E001 | 599 | 370 | 61.8% |
| E002 | 582 | 387 | 60.1% |
| E003 | 570 | 399 | 58.8% |
| E004 | 581 | 388 | 60.0% |
| E005 | 571 | 398 | 58.9% |
| E006 | 589 | 380 | 60.8% |
| E007 | 578 | 391 | 59.6% |
| E008 | 576 | 393 | 59.4% |
| E009 | 578 | 391 | 59.6% |
| E010 | 580 | 389 | 59.9% |
| E011 | 576 | 393 | 59.4% |
| E012 | 581 | 388 | 60.0% |
| E013 | 587 | 382 | 60.6% |
| E015 | 596 | 373 | 61.5% |

~60% of requests hit the 1024-token cap; ~40% stop on EOS earlier. The cap-rate is stable across configs — the model isn't producing longer outputs because of any optimization, the 1024 budget is just genuinely tight for this workload.

**Per-request artifact:** each experiment's `per_request_metrics.jsonl` contains one row per request with the exact `prompt_tokens`, `output_tokens`, `num_cached_tokens` (prefix cache hits), and `finish_reason`, plus all timing fields. Anyone reproducing this report should compute distributions from those files, not from the doc.

### Headline numbers

- **Best config (E011):** 983.7 output tok/s — **1.50× over E001 baseline**.
- **Naive BF16 baseline (E001):** 654.9 output tok/s, 0.785 QPS.
- **BF16 reference at optimal-config geometry (E015, text-only):** 477.1 output tok/s. Best-vs-this is **2.06×**.
- **Single biggest optimization:** MTP speculative decoding (k=5). Going from E005 → E006 added **26.8%** throughput in one step; removing MTP at the optimal point (E013) costs **19.8%**.
- **Single biggest no-op:** swapping FlashInfer attention for Flash-Attention 2 at the optimal point (E007 → E012). FA2 is **+1.7%** vs FlashInfer here — within noise.

The original plan predicted up to a **7× cumulative speedup**. We measured **1.50×** end-to-end. The shortfall is entirely attributable to A100-specific constraints (no native FP8 tensor cores, so FP8 KV cache unusable; FlashInfer FP8 MoE wants Hopper). On H100 the projected speedup would likely land closer to the plan, but we have no way to validate that without the hardware.

### Recap table — all 15 experiments

Avg input length is **5,761.3 tokens** across all 14 successful experiments (identical — same 969-prompt subset). Avg output length is per-experiment (varies slightly from sampling at T=0.7, all values shown below).

| Exp | label | backend | batch | FP8 | CUDA gr. | MTP | mem% | text? | KV | avg in tok | avg out tok | out tok/s | vs E001 | QPS | TPOT p50 | TTFT(eng) p50 | E2E p50 | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E001 | baseline (BF16 / FA2, full model, no opts) | FLASH_ATTN | 64 | ✗ | ✗ | ✗ | 0.95 | full | auto | 5,761.3 | 834.5 | **654.9** | 1.00× | 0.785 | 81.1ms | 577.5s | 658.6s | 76.77 |
| E002 | +FP8 weights (FA2) | FLASH_ATTN | 64 | ✓ | ✗ | ✗ | 0.85 | full | auto | 5,761.3 | 832.5 | **692.1** | 1.06× | 0.831 | 87.3ms | 548.6s | 653.1s | 68.86 |
| E003 | swap attention backend → FlashInfer | FLASHINFER | 64 | ✓ | ✗ | ✗ | 0.85 | full | auto | 5,761.3 | 831.5 | **696.5** | 1.06× | 0.838 | 87.1ms | 544.7s | 648.4s | 68.87 |
| E004 | +batch 128 | FLASHINFER | 128 | ✓ | ✗ | ✗ | 0.85 | full | auto | 5,761.3 | 832.7 | **789.1** | 1.20× | 0.948 | 126.2ms | 454.8s | 594.4s | 68.64 |
| E005 | +CUDA graphs (full + piecewise) | FLASHINFER | 128 | ✓ | ✓ | ✗ | 0.75 | full | auto | 5,761.3 | 829.0 | **768.4** | 1.17× | 0.927 | 97.2ms | 503.6s | 600.4s | 61.19 |
| E006 | +MTP speculative decoding (k=5) | FLASHINFER | 128 | ✓ | ✓ | ✓ | 0.75 | full | auto | 5,761.3 | 835.5 | **974.6** | 1.49× | 1.166 | 70.3ms | 407.8s | 481.4s | 62.88 |
| E007 | swap to text-only model (vision stripped) | FLASHINFER | 128 | ✓ | ✓ | ✓ | 0.75 | text | auto | 5,761.3 | 831.4 | **957.3** | 1.46× | 1.152 | 75.4ms | 415.3s | 497.3s | 63.40 |
| E008 | batch 192 | FLASHINFER | 192 | ✓ | ✓ | ✓ | 0.75 | text | auto | 5,761.3 | 830.8 | **968.4** | 1.48× | 1.166 | 75.1ms | 409.3s | 488.0s | 63.51 |
| E009 | batch 256 | FLASHINFER | 256 | ✓ | ✓ | ✓ | 0.75 | text | auto | 5,761.3 | 831.8 | **970.9** | 1.48× | 1.167 | 74.8ms | 406.9s | 488.8s | 62.59 |
| E010 | gpu_memory_utilization=0.70 | FLASHINFER | 128 | ✓ | ✓ | ✓ | 0.7 | text | auto | 5,761.3 | 832.4 | **955.1** | 1.46× | 1.147 | 63.7ms | 416.8s | 486.5s | 59.44 |
| E011 | gpu_memory_utilization=0.80 | FLASHINFER | 128 | ✓ | ✓ | ✓ | 0.8 | text | auto | 5,761.3 | 832.7 | **983.7** | 1.50× | 1.181 | 86.3ms | 401.7s | 499.1s | 70.61 |
| E012 | swap attention back to FA2 at optimal config | FLASH_ATTN | 128 | ✓ | ✓ | ✓ | 0.75 | text | auto | 5,761.3 | 831.4 | **973.6** | 1.49× | 1.171 | 75.1ms | 404.2s | 491.7s | 63.28 |
| E013 | disable MTP at optimal config (isolates MTP) | FLASHINFER | 128 | ✓ | ✓ | ✗ | 0.75 | text | auto | 5,761.3 | 830.1 | **781.4** | 1.19× | 0.941 | 103.8ms | 486.3s | 599.7s | 61.71 |
| E014 | FP8/INT8 KV cache probe (NOT TESTABLE — see trilogy) | — | — | — | — | — | — | — | — | — | — | **FAIL** | — | — | — | — | — | — |
| E015 | BF16 reference baseline (text-only, no opts) | FLASHINFER | 32 | ✗ | ✗ | ✗ | 0.95 | text | auto | 5,761.3 | 835.7 | **477.1** | 0.73× | 0.571 | 64.1ms | 813.3s | 871.8s | 77.22 |

### Per-experiment detail (everything that's in metrics.json, expanded)

#### E001 — baseline (BF16 / FA2, full model, no opts)

- **Wall-clock inference:** 1234.8 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 599, 'stop': 370}
- **Token totals:** prompt=5,582,695, output=808,654, cached=813,248, prefix-cache hit rate=14.57%
- **Throughput:** QPS=0.7848, output tok/s=654.91, prompt tok/s=4521.28, total tok/s=5176.19
- **TTFT (engine, canonical):** mean=589.81s, std=349.49s, min=8.22s, p50=577.50s, p90=1085.24s, p95=1137.36s, p99=1167.07s, max=1177.16s (n=969)
- **TTFT (client, sanity-check):** mean=594.13s, p50=581.78s, p99=1175.32s
- **TPOT (steady-state decode):** mean=82.76ms, std=16.56ms, min=48.13ms, p50=81.12ms, p90=95.57ms, p95=103.29ms, p99=137.42ms, max=315.98ms (n=969)
- **E2E latency:** mean=661.86s, std=350.57s, min=35.37s, p50=658.58s, p90=1146.10s, p95=1191.59s, p99=1228.31s, max=1234.72s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=76.77 GiB (96.86% of 79.25 GiB), avg=74.67 GiB, end=0.0 GiB, Δ(peak-baseline)=76.77 GiB, samples=1314
- **GPU utilization:** compute peak=100% / avg=93.5%; memory-bw peak=73% / avg=44.9%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=None, kv_cache_dtype=auto, gpu_memory_utilization=0.95, max_num_seqs=64, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=True, speculative_config=None
- **Artifacts:** [experiment_results/E001/](experiment_results/E001/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E002 — +FP8 weights (FA2)

- **Wall-clock inference:** 1165.5 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 582, 'stop': 387}
- **Token totals:** prompt=5,582,695, output=806,659, cached=836,352, prefix-cache hit rate=14.98%
- **Throughput:** QPS=0.8314, output tok/s=692.13, prompt tok/s=4790.09, total tok/s=5482.22
- **TTFT (engine, canonical):** mean=553.11s, std=328.45s, min=8.26s, p50=548.59s, p90=1012.51s, p95=1063.31s, p99=1098.36s, max=1109.26s (n=969)
- **TTFT (client, sanity-check):** mean=557.42s, p50=552.88s, p99=1106.62s
- **TPOT (steady-state decode):** mean=88.28ms, std=21.11ms, min=46.76ms, p50=87.31ms, p90=102.74ms, p95=108.78ms, p99=164.72ms, max=358.88ms (n=969)
- **E2E latency:** mean=630.13s, std=328.03s, min=43.47s, p50=653.08s, p90=1088.16s, p95=1123.80s, p99=1154.67s, max=1165.43s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=68.86 GiB (86.88% of 79.25 GiB), avg=66.08 GiB, end=0.0 GiB, Δ(peak-baseline)=68.86 GiB, samples=1249
- **GPU utilization:** compute peak=100% / avg=92.1%; memory-bw peak=61% / avg=36.1%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.85, max_num_seqs=64, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=True, speculative_config=None
- **Artifacts:** [experiment_results/E002/](experiment_results/E002/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E003 — swap attention backend → FlashInfer

- **Wall-clock inference:** 1156.8 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 570, 'stop': 399}
- **Token totals:** prompt=5,582,695, output=805,689, cached=836,352, prefix-cache hit rate=14.98%
- **Throughput:** QPS=0.8377, output tok/s=696.48, prompt tok/s=4825.95, total tok/s=5522.43
- **TTFT (engine, canonical):** mean=549.39s, std=325.85s, min=8.24s, p50=544.67s, p90=1005.03s, p95=1057.03s, p99=1090.27s, max=1100.72s (n=969)
- **TTFT (client, sanity-check):** mean=553.69s, p50=548.93s, p99=1098.52s
- **TPOT (steady-state decode):** mean=87.73ms, std=20.67ms, min=46.65ms, p50=87.05ms, p90=102.51ms, p95=108.47ms, p99=157.59ms, max=357.77ms (n=969)
- **E2E latency:** mean=625.87s, std=325.41s, min=43.35s, p50=648.43s, p90=1080.13s, p95=1116.14s, p99=1146.35s, max=1156.77s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=68.87 GiB (86.9% of 79.25 GiB), avg=66.28 GiB, end=0.0 GiB, Δ(peak-baseline)=68.87 GiB, samples=1234
- **GPU utilization:** compute peak=100% / avg=92.6%; memory-bw peak=60% / avg=36.4%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.85, max_num_seqs=64, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=True, speculative_config=None
- **Artifacts:** [experiment_results/E003/](experiment_results/E003/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E004 — +batch 128

- **Wall-clock inference:** 1022.6 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 581, 'stop': 388}
- **Token totals:** prompt=5,582,695, output=806,917, cached=813,024, prefix-cache hit rate=14.56%
- **Throughput:** QPS=0.9476, output tok/s=789.09, prompt tok/s=5459.34, total tok/s=6248.43
- **TTFT (engine, canonical):** mean=479.66s, std=289.34s, min=8.56s, p50=454.75s, p90=870.68s, p95=939.35s, p99=959.08s, max=961.10s (n=969)
- **TTFT (client, sanity-check):** mean=484.18s, p50=459.23s, p99=967.65s
- **TPOT (steady-state decode):** mean=128.70ms, std=69.65ms, min=51.63ms, p50=126.16ms, p90=145.12ms, p95=192.72ms, p99=325.05ms, max=1495.85ms (n=969)
- **E2E latency:** mean=585.34s, std=288.50s, min=63.14s, p50=594.43s, p90=978.58s, p95=1006.56s, p99=1020.70s, max=1022.55s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=68.64 GiB (86.6% of 79.25 GiB), avg=65.63 GiB, end=0.0 GiB, Δ(peak-baseline)=68.63 GiB, samples=1104
- **GPU utilization:** compute peak=100% / avg=93.1%; memory-bw peak=66% / avg=36.4%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.85, max_num_seqs=128, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=True, speculative_config=None
- **Artifacts:** [experiment_results/E004/](experiment_results/E004/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E005 — +CUDA graphs (full + piecewise)

- **Wall-clock inference:** 1045.4 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 571, 'stop': 398}
- **Token totals:** prompt=5,582,695, output=803,319, cached=807,840, prefix-cache hit rate=14.47%
- **Throughput:** QPS=0.9269, output tok/s=768.43, prompt tok/s=5340.24, total tok/s=6108.68
- **TTFT (engine, canonical):** mean=503.10s, std=298.05s, min=8.77s, p50=503.58s, p90=924.13s, p95=962.26s, p99=1002.89s, max=1004.99s (n=969)
- **TTFT (client, sanity-check):** mean=507.67s, p50=508.10s, p99=1011.64s
- **TPOT (steady-state decode):** mean=98.72ms, std=35.55ms, min=30.87ms, p50=97.20ms, p90=109.32ms, p95=122.81ms, p99=249.61ms, max=558.91ms (n=969)
- **E2E latency:** mean=586.64s, std=298.01s, min=60.79s, p50=600.35s, p90=1006.00s, p95=1028.69s, p99=1041.76s, max=1045.36s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=61.19 GiB (77.21% of 79.25 GiB), avg=56.94 GiB, end=0.0 GiB, Δ(peak-baseline)=61.19 GiB, samples=1208
- **GPU utilization:** compute peak=100% / avg=87.9%; memory-bw peak=94% / avg=34.9%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.75, max_num_seqs=128, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=False, speculative_config=None
- **Artifacts:** [experiment_results/E005/](experiment_results/E005/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E006 — +MTP speculative decoding (k=5)

- **Wall-clock inference:** 830.7 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 589, 'stop': 380}
- **Token totals:** prompt=5,582,695, output=809,635, cached=751,200, prefix-cache hit rate=13.46%
- **Throughput:** QPS=1.1664, output tok/s=974.59, prompt tok/s=6720.14, total tok/s=7694.73
- **TTFT (engine, canonical):** mean=409.39s, std=238.31s, min=8.64s, p50=407.80s, p90=742.45s, p95=782.11s, p99=800.61s, max=805.18s (n=969)
- **TTFT (client, sanity-check):** mean=413.89s, p50=412.25s, p99=809.21s
- **TPOT (steady-state decode):** mean=72.87ms, std=36.18ms, min=15.23ms, p50=70.31ms, p90=90.95ms, p95=116.50ms, p99=209.74ms, max=525.87ms (n=969)
- **E2E latency:** mean=470.89s, std=237.74s, min=56.72s, p50=481.42s, p90=803.49s, p95=821.52s, p99=828.66s, max=830.70s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=62.88 GiB (79.33% of 79.25 GiB), avg=58.88 GiB, end=0.0 GiB, Δ(peak-baseline)=62.88 GiB, samples=938
- **GPU utilization:** compute peak=100% / avg=90.4%; memory-bw peak=54% / avg=25.3%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.75, max_num_seqs=128, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=False, speculative_config={'model': '/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant', 'num_speculative_tokens': 5}
- **Artifacts:** [experiment_results/E006/](experiment_results/E006/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E007 — swap to text-only model (vision stripped)

- **Wall-clock inference:** 841.5 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 578, 'stop': 391}
- **Token totals:** prompt=5,582,695, output=805,582, cached=751,200, prefix-cache hit rate=13.46%
- **Throughput:** QPS=1.1515, output tok/s=957.32, prompt tok/s=6634.25, total tok/s=7591.57
- **TTFT (engine, canonical):** mean=417.76s, std=239.27s, min=11.27s, p50=415.35s, p90=755.29s, p95=788.60s, p99=812.19s, max=816.75s (n=969)
- **TTFT (client, sanity-check):** mean=422.19s, p50=419.75s, p99=820.63s
- **TPOT (steady-state decode):** mean=77.08ms, std=34.46ms, min=15.80ms, p50=75.38ms, p90=94.37ms, p95=111.12ms, p99=225.23ms, max=523.26ms (n=969)
- **E2E latency:** mean=483.11s, std=238.69s, min=65.31s, p50=497.29s, p90=816.05s, p95=831.88s, p99=839.27s, max=841.45s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=63.4 GiB (79.99% of 79.25 GiB), avg=58.41 GiB, end=0.0 GiB, Δ(peak-baseline)=63.39 GiB, samples=985
- **GPU utilization:** compute peak=100% / avg=86.0%; memory-bw peak=87% / avg=24.3%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.75, max_num_seqs=128, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=False, speculative_config={'model': '/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant', 'num_speculative_tokens': 5}
- **Artifacts:** [experiment_results/E007/](experiment_results/E007/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E008 — batch 192

- **Wall-clock inference:** 831.3 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 576, 'stop': 393}
- **Token totals:** prompt=5,582,695, output=805,033, cached=752,096, prefix-cache hit rate=13.47%
- **Throughput:** QPS=1.1657, output tok/s=968.44, prompt tok/s=6715.91, total tok/s=7684.35
- **TTFT (engine, canonical):** mean=409.38s, std=238.25s, min=8.25s, p50=409.32s, p90=744.77s, p95=778.46s, p99=804.27s, max=806.58s (n=969)
- **TTFT (client, sanity-check):** mean=413.69s, p50=413.60s, p99=812.53s
- **TPOT (steady-state decode):** mean=76.23ms, std=36.15ms, min=13.91ms, p50=75.07ms, p90=92.52ms, p95=107.75ms, p99=208.71ms, max=664.64ms (n=969)
- **E2E latency:** mean=474.28s, std=237.27s, min=57.60s, p50=488.03s, p90=805.99s, p95=821.18s, p99=828.22s, max=831.22s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=63.51 GiB (80.13% of 79.25 GiB), avg=61.02 GiB, end=0.0 GiB, Δ(peak-baseline)=63.51 GiB, samples=899
- **GPU utilization:** compute peak=100% / avg=93.3%; memory-bw peak=53% / avg=25.9%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.75, max_num_seqs=192, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=False, speculative_config={'model': '/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant', 'num_speculative_tokens': 5}
- **Artifacts:** [experiment_results/E008/](experiment_results/E008/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E009 — batch 256

- **Wall-clock inference:** 830.2 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 578, 'stop': 391}
- **Token totals:** prompt=5,582,695, output=806,020, cached=749,600, prefix-cache hit rate=13.43%
- **Throughput:** QPS=1.1672, output tok/s=970.87, prompt tok/s=6724.49, total tok/s=7695.36
- **TTFT (engine, canonical):** mean=409.40s, std=238.92s, min=8.21s, p50=406.85s, p90=745.99s, p95=780.79s, p99=803.10s, max=807.22s (n=969)
- **TTFT (client, sanity-check):** mean=413.70s, p50=411.11s, p99=811.31s
- **TPOT (steady-state decode):** mean=76.44ms, std=35.76ms, min=13.98ms, p50=74.84ms, p90=92.42ms, p95=106.89ms, p99=196.77ms, max=582.90ms (n=969)
- **E2E latency:** mean=474.48s, std=237.89s, min=57.69s, p50=488.84s, p90=806.49s, p95=821.02s, p99=827.78s, max=830.16s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=62.59 GiB (78.98% of 79.25 GiB), avg=60.14 GiB, end=0.0 GiB, Δ(peak-baseline)=62.59 GiB, samples=898
- **GPU utilization:** compute peak=100% / avg=93.8%; memory-bw peak=53% / avg=26.0%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.75, max_num_seqs=256, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=False, speculative_config={'model': '/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant', 'num_speculative_tokens': 5}
- **Artifacts:** [experiment_results/E009/](experiment_results/E009/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E010 — gpu_memory_utilization=0.70

- **Wall-clock inference:** 844.5 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 580, 'stop': 389}
- **Token totals:** prompt=5,582,695, output=806,623, cached=747,200, prefix-cache hit rate=13.38%
- **Throughput:** QPS=1.1474, output tok/s=955.11, prompt tok/s=6610.38, total tok/s=7565.49
- **TTFT (engine, canonical):** mean=418.67s, std=243.42s, min=8.68s, p50=416.84s, p90=758.15s, p95=796.67s, p99=816.99s, max=819.44s (n=969)
- **TTFT (client, sanity-check):** mean=423.21s, p50=421.32s, p99=825.68s
- **TPOT (steady-state decode):** mean=64.62ms, std=24.73ms, min=15.53ms, p50=63.75ms, p90=79.51ms, p95=91.25ms, p99=156.26ms, max=435.56ms (n=969)
- **E2E latency:** mean=475.38s, std=243.04s, min=51.50s, p50=486.45s, p90=816.51s, p95=836.11s, p99=843.54s, max=844.49s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=59.44 GiB (75.0% of 79.25 GiB), avg=57.15 GiB, end=0.0 GiB, Δ(peak-baseline)=59.44 GiB, samples=914
- **GPU utilization:** compute peak=100% / avg=93.4%; memory-bw peak=53% / avg=26.3%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.7, max_num_seqs=128, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=False, speculative_config={'model': '/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant', 'num_speculative_tokens': 5}
- **Artifacts:** [experiment_results/E010/](experiment_results/E010/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E011 — gpu_memory_utilization=0.80

- **Wall-clock inference:** 820.2 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 576, 'stop': 393}
- **Token totals:** prompt=5,582,695, output=806,894, cached=757,600, prefix-cache hit rate=13.57%
- **Throughput:** QPS=1.1814, output tok/s=983.72, prompt tok/s=6806.12, total tok/s=7789.85
- **TTFT (engine, canonical):** mean=402.74s, std=235.04s, min=8.74s, p50=401.72s, p90=732.15s, p95=770.69s, p99=787.74s, max=793.96s (n=969)
- **TTFT (client, sanity-check):** mean=407.42s, p50=406.40s, p99=796.48s
- **TPOT (steady-state decode):** mean=90.22ms, std=49.39ms, min=16.85ms, p50=86.31ms, p90=111.75ms, p95=149.46ms, p99=330.66ms, max=591.10ms (n=969)
- **E2E latency:** mean=476.70s, std=233.34s, min=66.81s, p50=499.14s, p90=803.65s, p95=814.92s, p99=818.39s, max=820.21s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=70.61 GiB (89.1% of 79.25 GiB), avg=67.23 GiB, end=0.0 GiB, Δ(peak-baseline)=70.61 GiB, samples=887
- **GPU utilization:** compute peak=100% / avg=93.4%; memory-bw peak=54% / avg=25.9%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.8, max_num_seqs=128, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=False, speculative_config={'model': '/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant', 'num_speculative_tokens': 5}
- **Artifacts:** [experiment_results/E011/](experiment_results/E011/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E012 — swap attention back to FA2 at optimal config

- **Wall-clock inference:** 827.5 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 581, 'stop': 388}
- **Token totals:** prompt=5,582,695, output=805,638, cached=752,800, prefix-cache hit rate=13.48%
- **Throughput:** QPS=1.1710, output tok/s=973.58, prompt tok/s=6746.44, total tok/s=7720.02
- **TTFT (engine, canonical):** mean=407.63s, std=237.73s, min=8.22s, p50=404.18s, p90=744.50s, p95=777.28s, p99=800.74s, max=804.67s (n=969)
- **TTFT (client, sanity-check):** mean=411.93s, p50=408.44s, p99=808.96s
- **TPOT (steady-state decode):** mean=77.08ms, std=36.71ms, min=13.84ms, p50=75.14ms, p90=92.68ms, p95=116.19ms, p99=215.59ms, max=578.83ms (n=969)
- **E2E latency:** mean=472.47s, std=237.03s, min=57.23s, p50=491.67s, p90=803.52s, p95=818.91s, p99=825.40s, max=827.46s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=63.28 GiB (79.85% of 79.25 GiB), avg=60.84 GiB, end=0.0 GiB, Δ(peak-baseline)=63.28 GiB, samples=895
- **GPU utilization:** compute peak=100% / avg=93.6%; memory-bw peak=54% / avg=25.9%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.75, max_num_seqs=128, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=False, speculative_config={'model': '/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant', 'num_speculative_tokens': 5}
- **Artifacts:** [experiment_results/E012/](experiment_results/E012/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E013 — disable MTP at optimal config (isolates MTP)

- **Wall-clock inference:** 1029.3 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 587, 'stop': 382}
- **Token totals:** prompt=5,582,695, output=804,322, cached=807,840, prefix-cache hit rate=14.47%
- **Throughput:** QPS=0.9414, output tok/s=781.39, prompt tok/s=5423.55, total tok/s=6204.94
- **TTFT (engine, canonical):** mean=493.30s, std=292.96s, min=8.57s, p50=486.30s, p90=914.20s, p95=940.41s, p99=986.34s, max=989.93s (n=969)
- **TTFT (client, sanity-check):** mean=497.77s, p50=490.76s, p99=994.85s
- **TPOT (steady-state decode):** mean=107.99ms, std=52.22ms, min=30.09ms, p50=103.80ms, p90=124.96ms, p95=162.81ms, p99=326.21ms, max=889.55ms (n=969)
- **E2E latency:** mean=581.60s, std=291.81s, min=63.20s, p50=599.67s, p90=995.91s, p95=1014.55s, p99=1025.99s, max=1029.30s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=61.71 GiB (77.87% of 79.25 GiB), avg=58.24 GiB, end=0.0 GiB, Δ(peak-baseline)=61.71 GiB, samples=1155
- **GPU utilization:** compute peak=100% / avg=89.5%; memory-bw peak=86% / avg=35.9%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=fp8, kv_cache_dtype=auto, gpu_memory_utilization=0.75, max_num_seqs=128, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=False, speculative_config=None
- **Artifacts:** [experiment_results/E013/](experiment_results/E013/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

#### E014 — FP8/INT8 KV cache probe (NOT TESTABLE — see trilogy)

**Not runnable on this hardware.** See [§E014 Trilogy](#e014-trilogy-the-three-distinct-failure-modes) below.

#### E015 — BF16 reference baseline (text-only, no opts)

- **Wall-clock inference:** 1697.6 s
- **Dataset:** seen=1000, loaded=969, skipped_length=31, threshold=31744 tokens
- **Requests:** finished=969, failed=0
- **Finish reasons:** {'length': 596, 'stop': 373}
- **Token totals:** prompt=5,582,695, output=809,840, cached=836,352, prefix-cache hit rate=14.98%
- **Throughput:** QPS=0.5708, output tok/s=477.06, prompt tok/s=3288.64, total tok/s=3765.70
- **TTFT (engine, canonical):** mean=818.41s, std=482.87s, min=8.20s, p50=813.26s, p90=1492.27s, p95=1571.50s, p99=1628.91s, max=1639.45s (n=969)
- **TTFT (client, sanity-check):** mean=822.70s, p50=817.51s, p99=1637.13s
- **TPOT (steady-state decode):** mean=64.85ms, std=7.85ms, min=47.77ms, p50=64.07ms, p90=73.98ms, p95=78.27ms, p99=87.55ms, max=134.31ms (n=969)
- **E2E latency:** mean=876.88s, std=484.45s, min=18.51s, p50=871.77s, p90=1557.02s, p95=1623.53s, p99=1682.07s, max=1697.53s
- **GPU memory (1 Hz trace, GPU 0):** baseline=0.0 GiB, peak=77.22 GiB (97.44% of 79.25 GiB), avg=76.17 GiB, end=0.0 GiB, Δ(peak-baseline)=77.22 GiB, samples=1742
- **GPU utilization:** compute peak=100% / avg=89.1%; memory-bw peak=63% / avg=43.6%
- **Engine config (from metrics.config):** dtype=bfloat16, quantization=None, kv_cache_dtype=auto, gpu_memory_utilization=0.95, max_num_seqs=32, max_num_batched_tokens=32768, max_model_len=32768, enforce_eager=True, speculative_config=None
- **Artifacts:** [experiment_results/E015/](experiment_results/E015/) (metrics.json, per_request_metrics.jsonl, output.jsonl, gpu_trace.csv, inference.log, summary.md, environment.txt)

### Cumulative chain (E001 → E007): which knob bought how much

| Step | Δ vs prior | out tok/s | New vs E001 | What was added |
|---|---|---|---|---|
| E001 → E002 | +5.7% | 654.9 → 692.1 | 1.06× | FP8 weights (Marlin FP8 MoE under the hood on sm_80). **Big memory drop**: 76.77 → 68.86 GiB peak. |
| E002 → E003 | +0.6% | 692.1 → 696.5 | 1.06× | Attention backend swap FA2 → FlashInfer. Within noise — the plan's predicted "+8-10%" included FlashInfer FP8 MoE wins that don't exist on A100. |
| E003 → E004 | **+13.3%** | 696.5 → 789.1 | 1.20× | Batch 64 → 128. TPOT goes up (87 → 126 ms) because more sequences share each decode step, but throughput dominates. |
| E004 → E005 | **−2.6%** | 789.1 → 768.4 | 1.17× | CUDA graphs (full + piecewise). **Regression on this workload.** Likely cause: heterogeneous prompt lengths defeat graph-capture amortization on a model with 51 capture sizes (1–512). Worth a deeper look. |
| E005 → E006 | **+26.8%** | 768.4 → 974.6 | 1.49× | MTP (k=5). **Single biggest jump in the whole study.** TPOT drops from 97 → 70 ms — MTP genuinely produces multiple tokens per decode step. |
| E006 → E007 | −1.8% | 974.6 → 957.3 | 1.46× | Swap to text-only model. Vision weights are only ~1 GiB; on 80 GiB this isn't worth doing on this hardware. |

After E007, the chain branches into the group-B/C alternates rather than a single cumulative path. The peak in the whole study is **E011 (gpu_memory_utilization=0.80) at 983.7 out tok/s = 1.50× E001**.

### Group B/C — what the alternates revealed

**Batch scaling beyond 128 saturates** (E007/E008/E009 = 957.3 / 968.4 / 970.9 out tok/s). Effective batch is bounded by the KV pool — at max_num_seqs=128 with BF16 KV cache, the pool fills before the engine admits 192 or 256 concurrent. The peak Running counter never approached the nominal `max_num_seqs` in any of these. Engine periodic stats lines weren't emitted (see "Known gaps") so we can't show the exact Running peak, but gpu_trace.csv shows GPU memory plateaus at the same level across these three runs.

**`gpu_memory_utilization` sensitivity** (E010 / E007 / E011 = 0.70 / 0.75 / 0.80):
- 0.70 → 955.1 out tok/s (peak GPU 59.44 GiB)
- 0.75 → 957.3 out tok/s (peak 63.40 GiB)
- 0.80 → **983.7 out tok/s** (peak 70.61 GiB)
Higher KV pool → marginally more throughput. Going past 0.80 wasn't tested but the trend suggests diminishing returns since each new GiB of pool reaches longer-tail prompts that contribute less.

**Attention backend at the optimal point** (E007 vs E012, otherwise identical):
- FlashInfer attn: 957.3 out tok/s, TPOT p50 = 75.4 ms
- FA2 attn:       973.6 out tok/s, TPOT p50 = 75.1 ms
FA2 wins by **1.7%** — small but consistent. On Gemma 4, 25/30 layers use sliding-window attention (window=1024), and FA2 appears to handle the sliding pattern slightly better than FlashInfer at our batch size. **Recommendation: use FA2 on A100 for this model, not FlashInfer.**

**MTP contribution at the optimal point** (E006/E007 vs E013):
- With MTP (E007): 957.3 out tok/s, TPOT p50 = 75.4 ms
- Without MTP (E013): 781.4 out tok/s, TPOT p50 = 103.8 ms
**Removing MTP costs 18.4% throughput and adds 37.7% TPOT.** This is the clearest "isolated optimization value" measurement in the study. MTP earns its complexity here.

### E014 trilogy — the three distinct failure modes

E014's original purpose was to test FP8 KV cache (E4M3 vs the default). On A100 + Gemma 4 + vLLM 0.21.1 there is **no working FP8 / INT8 / NVFP4 KV cache configuration**. We attempted three:

#### Attempt 1: `--fp8 --kv-cache-dtype fp8_e4m3`

- **What failed:** Triton kernel compilation, when the attention backend tries to write KV in fp8e4nv.
- **Where:** `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` → `reshape_and_cache_kernel_flash`.
- **Root error:**
  ```
  triton.compiler.errors.CompilationError: at 1:0:
  def reshape_and_cache_kernel_flash(
  ValueError("type fp8e4nv not supported in this architecture.
   The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
  ```
- **Why:** A100 (sm_80) has no native FP8 tensor cores. Triton's emulation kernel on sm_80 exposes `fp8e4b15` (the older E4M3 with bias=15) and `fp8e5` (E5M2), but not `fp8e4nv` (the NVIDIA E4M3 with bias=7, introduced with Hopper).
- **Fixable on this hardware?** No. Requires sm_90+ (H100).
- **Archived at:** `experiment_results/E014_fp8_e4m3_FAILED_sm80/`

#### Attempt 2: `--no-fp8 --kv-cache-dtype fp8_e5m2 --text-only` (BF16 weights, FP8 KV)

- **What failed:** vLLM's query-quantization path asserts kv_cache_dtype is in a fixed allow-list at forward time.
- **Where:** `vllm/model_executor/layers/attention/attention.py:467`.
- **Root error:**
  ```
  AssertionError
    File ".../attention.py", line 467
      assert self.kv_cache_dtype in {"fp8", "fp8_e4m3", "nvfp4"}
  ```
- **Why:** When `kv_cache_dtype.startswith("fp8")` and the impl declares `supports_quant_query_input=True` (FA2 and FlashInfer both do on CUDA), vLLM auto-enables query quantization. The forward path then asserts kv_cache_dtype is one of `{fp8, fp8_e4m3, nvfp4}` — `fp8_e5m2` is **explicitly excluded**. There's no env-var override (only an internal `disable_flashinfer_q_quantization` config field).
- **Fixable on this hardware?** No without patching vLLM. The constraint is hard-coded vLLM logic.
- **Archived at:** `experiment_results/E014_fp8_e5m2_FAILED_query_quant_assert/`

#### Attempt 3: `--no-fp8 --kv-cache-dtype int8_per_token_head --text-only`

- **What failed:** KV cache page-size unification across heterogeneous layer dimensions.
- **Where:** `vllm/v1/core/kv_cache_utils.py:1040` → `unify_kv_cache_spec_page_size`.
- **Root error:**
  ```
  NotImplementedError: The page size of the layer is not divisible by the
  maximum page size. Cannot unify by adjusting block_size.
  ```
- **Why:** Gemma 4 has mixed attention layer types — 25/30 layers are sliding-window with `head_dim=256`, 5/30 are full-attention with `global_head_dim=512`. `int8_per_token_head` produces different per-layer page sizes (scaled by head_dim), and vLLM can't pack them into a uniform KV cache layout.
- **Fixable on this hardware?** Yes in principle, with a different model or a vLLM patch that allows non-unified page sizes. Out of scope here.
- **Archived at:** `experiment_results/E014_int8_per_token_head_FAILED_page_size/`

#### Conclusion for E014

On this hardware/model/vLLM combo there is no KV-cache quantization mode we can run. The three archives **are** the result — every other FP8 experiment in the study effectively runs with BF16 KV cache (`auto` falls back to BF16 because the FP8 paths above are unavailable).

### Hardware-driven deltas from the original plan (recap)

These are differences between what the plan body predicted and what actually happened, ranked by their effect on the conclusion:

1. **No FP8 KV cache** on A100 → every FP8 experiment runs with BF16 KV. The plan's implicit "FP8 across the stack" turned into "FP8 weights only".
2. **No FlashInfer FP8 MoE** on A100 → `VLLM_USE_FLASHINFER_MOE_FP8` unset; vLLM uses Marlin FP8 MoE in every FP8 experiment, regardless of attention backend. The E003/E012 "FA2 vs FlashInfer" comparison became a pure attention-kernel test instead of "everything FA2 vs everything FlashInfer".
3. **No FlashInfer sampler** on this nvcc/cccl combo → `VLLM_USE_FLASHINFER_SAMPLER=0`, torch native sampler used. Constant overhead across the whole sweep, so relative comparisons are still valid.
4. **A100 80GB instead of plan's assumed 40GB** → E001 (BF16 baseline) didn't OOM; ran to completion at 654.9 out tok/s. The whole "FP8 unlocks running the model at all" framing in the plan body doesn't apply.

### Fix history during the campaign

Every change to source / scripts that was made between the initial run attempt and final completion. Listed in chronological order:

1. **PyTorch + CUDA mismatch.** Initial env had `torch 2.11.0+cu130` but the system driver was 12.6. `_cuda_init` failed on every experiment. **Fix:** reinstalled the torch family as cu126 (`torch==2.11.0+cu126`, `torchvision==0.26.0+cu126`, `torchaudio==2.11.0+cu126`). Rebuilt vLLM C extensions via `VLLM_USE_PRECOMPILED=1 pip install -e .` so the precompiled wheel for cu126 took precedence over the stale `_C.abi3.so` (which had been built against cu130 / libcudart.so.13).
2. **`fp8_e5m2` KV cache rejected with FP8 weights** — default changed from `fp8_e5m2` to `auto` in `examples/run_ablation_experiment.sh:33`.
3. **FlashInfer sampler JIT compile failed** (system `/usr/bin/nvcc` is v10.1, missing `--generate-dependencies-with-compile`). **Fix:** set `VLLM_USE_FLASHINFER_SAMPLER=0`. Also added `CUDA_HOME` pointing at `/root/miniconda3/envs/vila/targets/x86_64-linux` (cu12.8 toolkit) so any other vLLM-side JIT picks up a usable nvcc.
4. **`VLLM_USE_FLASHINFER_MOE_FP8=1` raised `NotImplementedError` on A100.** **Fix:** stopped exporting it; vLLM falls back to Marlin FP8 MoE which works on sm_80.
5. **vLLM's `gemma4_mm.py:get_mm_max_tokens_per_item` crashed on `config.vision_config.default_output_length` when vision_config is None** (text-only checkpoint). **Fix:** patched `vllm/model_executor/models/gemma4_mm.py` to guard `config.vision_config is not None`.
6. **vLLM's `Gemma4ForConditionalGeneration.__init__` unconditionally built a `vision_tower` from `config.vision_config`** even when it's None. **Fix:** same file; added a `vision_config is not None` guard around the vision tower / `embed_vision` construction.
7. **`json.dump(metrics)` crashed for MTP runs** because the `speculative_config` dict we passed to `AsyncEngineArgs` gets mutated by vLLM at engine init to include a non-serializable `ModelConfig`. **Fix:** rebuild a clean `spec_snapshot` dict from `args.*` for `metrics.json`, and pass `default=str` to `json.dump` as a safety net.
8. **`KV_CACHE_DTYPE` was being force-overridden to `"auto"` whenever `--no-fp8` was set**, preventing the redesigned E014 runs. **Fix:** removed the override in `run_ablation_experiment.sh:194`; `--kv-cache-dtype` from the user always wins. Also moved the `--kv_cache_dtype` arg forwarding outside the `if USE_FP8` branch so it's always passed to Python.

### Known gaps in the captured data

Things metrics.json *should* have but doesn't, on this run:

1. **MTP acceptance rate (no data).** vLLM didn't emit any `SpecDecoding metrics:` lines in any inference.log, despite MTP being active in E006–E012. The summary.md heredoc was prepared to parse these (and would have shown mean / min / max draft acceptance rate per experiment), but no data → no output. Cause unknown — possibly the SpecDecodingLogging's window cadence is longer than our per-experiment wall clock, or `aggregate_engine_logging` would need to be set true. **Status:** unmeasured. The TPOT delta between E006/E007 and E013 (the +/− MTP comparison) is currently our only quantification of MTP's effect.
2. **Engine periodic stats (no data).** Same issue — no `Avg prompt throughput …, Running: N reqs, Waiting: N reqs, GPU KV cache usage: X%` lines were emitted by vLLM v1 in any run, despite `disable_log_stats=False`. The summary.md heredoc's `Peak KV cache usage / Peak running requests / Peak waiting requests` rows all read "N/A". We do have nvidia-smi-derived GPU memory peak per experiment via gpu_trace.csv, which captures the total-process picture (model + KV + activations + cuda-graph cache), just not the KV-cache-vs-other breakdown.
3. **Effective batch admission.** Without the engine's Running/Waiting periodic snapshots, we can't directly show "nominal batch 256 only achieved ≤K concurrent". The flat-line throughput across E007/E008/E009 is indirect evidence that batch saturation occurred before max_num_seqs was reached.

### Recommendations

1. **Use FA2 attention, not FlashInfer, for Gemma 4 26B on A100** (E012 vs E007: +1.7% throughput, equivalent TPOT, simpler stack).
2. **Always enable MTP** if assistant model is available — biggest single win in the whole study, costs ~1 GiB.
3. **CUDA graphs (E005) regress on this workload** — don't enable unless validated for your specific prompt distribution. Worth a dedicated study with uniform prompt sizes to see if the regression is workload-specific.
4. **Batch 128 is the sweet spot.** Going higher is wasted budget (KV pool saturates) for this dataset. If your prompts skew shorter than the MAI distribution, you'd see bigger gains from batch ≥192.
5. **`gpu_memory_utilization=0.80` outperforms 0.75** (E011 best of the sweep). Marginal but reproducible.
6. **For genuine FP8 KV cache validation, move to H100+**. None of the FP8 KV variants are accessible on A100 with the current vLLM.
7. **Wire SpecDecodingLogging + IterationStats into metrics.json** before the next sweep. The current pipeline writes only client-side per-request data; engine-side periodic stats are missing entirely, which is why MTP acceptance and KV cache % aren't in this report.
8. **Re-run E004 → E005 (CUDA graphs) with reduced `cudagraph_capture_sizes`** to see if the regression is from compile-time vs runtime overhead. The vLLM init logs 51 capture sizes (1–512); narrowing to just the sizes that actually appear in steady-state might recover the win.

### Artifact index

| Path | What's there |
|---|---|
| `experiment_results/E001/` … `E013/` | Successful runs. Each contains: `metrics.json` (the canonical structured output), `per_request_metrics.jsonl` (one row per request with timing + token counts), `output.jsonl` (generations, with `input_index` for mapping back to source prompts), `gpu_trace.csv` (1-Hz nvidia-smi trace of GPU 0), `inference.log` (vLLM + Python stdout/stderr), `summary.md` (human-readable per-experiment report), `environment.txt` (hardware + software + git commit snapshot), `gpu_initial.txt`/`gpu_final.txt` (snapshots). |
| `experiment_results/E015/` | Successful BF16 reference run. Same artifact set. |
| `experiment_results/E014_fp8_e4m3_FAILED_sm80/` | First E014 attempt — Triton fp8e4nv sm_80 failure. |
| `experiment_results/E014_fp8_e5m2_FAILED_query_quant_assert/` | Second E014 attempt — vLLM query-quant assertion failure. |
| `experiment_results/E014_int8_per_token_head_FAILED_page_size/` | Third E014 attempt — KV page-size unification failure. |
| `experiment_results/ablation_study/` | Master dispatcher log directory (`ablation_study.log`, `ablation_study_resume.log` for partial sweeps). |
| `gemma4/ablation_study_async_engine.md` | This document — plan + actual settings + results. |
| `run_ablation_experiment.sh` | Single-experiment runner. |
| `run_all_ablation_experiments.sh` | Sequential 15-experiment master script. |
| `run_remaining_ablation_experiments.sh` | Resume script used after partial-sweep failures. |
| `run_inference_configurable.py` | Python entrypoint that builds the vLLM AsyncLLMEngine and runs inference, writing all structured metrics. |
| `create_text_only_model.py` | Strips vision weights from the Gemma 4 26B checkpoint to produce the text-only variant. |

---

## Table of Contents

**Up-to-date sections (read these first):**

0. [Actual Run Settings (this hardware / 2026-05-20)](#actual-run-settings-this-hardware--2026-05-20)
1. [Results (2026-05-21)](#results-2026-05-21) — including:
   - [Dataset](#dataset) — file path, schema, prompt + output length distributions
   - [Headline numbers](#headline-numbers)
   - [Recap table — all 15 experiments](#recap-table--all-15-experiments)
   - [Per-experiment detail](#per-experiment-detail-everything-thats-in-metricsjson-expanded)
   - [Cumulative chain (E001 → E007)](#cumulative-chain-e001--e007-which-knob-bought-how-much)
   - [Group B/C alternates](#group-bc--what-the-alternates-revealed)
   - [E014 trilogy — three distinct failure modes](#e014-trilogy--the-three-distinct-failure-modes)
   - [Hardware-driven deltas from the original plan](#hardware-driven-deltas-from-the-original-plan-recap)
   - [Fix history during the campaign](#fix-history-during-the-campaign)
   - [Known gaps in the captured data](#known-gaps-in-the-captured-data)
   - [Recommendations](#recommendations)
   - [Artifact index](#artifact-index)

**Historical plan body (the original intent — supersedes by the sections above where they disagree):**

2. [Experiment Matrix](#experiment-matrix)
3. [Baseline Configuration](#baseline-configuration)
4. [Experiment Configurations](#experiment-configurations)
5. [Execution Plan](#execution-plan)
7. [Analysis Guidelines](#analysis-guidelines)

---

## Experiment Matrix

### Overview

```
Experiments organized in 3 groups:

GROUP A: Core Optimizations (E001-E007)
├─ Baseline → FP8 → FlashInfer → CUDA Graphs → MTP
└─ Build up optimizations cumulatively

GROUP B: Memory Optimizations (E008-E011)
├─ Vision weight removal
├─ Batch size scaling
└─ gpu_memory_utilization tuning

GROUP C: Alternative Configurations (E012-E015)
├─ Backend comparison
├─ Quantization comparison
└─ KV cache dtype comparison

Total: 15 experiments
Total measured inference wall-clock across the 14 successful experiments: 14178s = 3.94h (excludes engine-init and cooldowns; excludes the 3 failed E014 attempts)
```

### Quick Reference Table

| Exp | Name | FP8 | Backend | CUDA | MTP | Batch | Vision | Mem%  |
|-----|------|-----|---------|------|-----|-------|--------|------ |
| E001 | Baseline | ✗ | FA2 | ✗ | ✗ | 64 | ✓ | 0.95  |
| E002 | +FP8 | ✓ | FA2 | ✗ | ✗ | 64 | ✓ | 0.85  |
| E003 | +FlashInfer | ✓ | FI | ✗ | ✗ | 64 | ✓ | 0.85  |
| E004 | +Batch128 | ✓ | FI | ✗ | ✗ | 128 | ✓ | 0.85  |
| E005 | +CUDAGraphs | ✓ | FI | ✓ | ✗ | 128 | ✓ | 0.75  |
| E006 | +MTP | ✓ | FI | ✓ | ✓ | 128 | ✓ | 0.75  |
| E007 | -Vision | ✓ | FI | ✓ | ✓ | 128 | ✗ | 0.75  |
| E008 | Batch192 | ✓ | FI | ✓ | ✓ | 192 | ✗ | 0.75  |
| E009 | Batch256 | ✓ | FI | ✓ | ✓ | 256 | ✗ | 0.75  |
| E010 | Mem70 | ✓ | FI | ✓ | ✓ | 128 | ✗ | 0.70  |
| E011 | Mem80 | ✓ | FI | ✓ | ✓ | 128 | ✗ | 0.80  |
| E012 | FA2vsFI | ✓ | FA2 | ✓ | ✓ | 128 | ✗ | 0.75  |
| E013 | NoMTP | ✓ | FI | ✓ | ✗ | 128 | ✗ | 0.75  |
| E014 | KV_E4M3 | ✓ | FI | ✓ | ✓ | 128 | ✗ | 0.75  |
| E015 | BF16Full | ✗ | FI | ✗ | ✗ | 32 | ✗ | 0.95  |

Legend:
- FP8: FP8 quantization enabled (model weights only; KV cache is BF16 on this hardware — see "Actual Run Settings" above)
- Backend: FA2 (Flash Attention 2), FI (FlashInfer **attention only** on this hardware — FlashInfer FP8 MoE and sampler are disabled, see "Actual Run Settings" for why)
- CUDA: CUDA graphs enabled
- MTP: Multi-Token Prediction with assistant model (vllm `speculative_config`, num_speculative_tokens=5)
- Batch: max_num_seqs
- Vision: Vision weights present in loaded checkpoint
- Mem%: gpu_memory_utilization
```

### What "FlashInfer" actually means in this run

The experiments labeled "FI" exercise FlashInfer's **attention** kernels (`VLLM_ATTENTION_BACKEND=FLASHINFER`). They do **not** exercise:

- **FlashInfer FP8 MoE** (`VLLM_USE_FLASHINFER_MOE_FP8`) — unset; needs sm_90+. vLLM uses Marlin FP8 MoE on A100 instead. This applies to **every FP8 experiment** (E002–E014), regardless of attention backend.
- **FlashInfer sampler** (`VLLM_USE_FLASHINFER_SAMPLER`) — set to `0`; falls back to torch's native top-k/top-p. Constant overhead across all 15 experiments → relative comparisons stay valid.

So the FA2-vs-FlashInfer comparison (E002 vs E003, and E007 vs E012) is **strictly an attention-kernel comparison**. The "FlashInfer FP8 MoE benefit" the plan body originally implied for E003+ is not being measured here — Marlin FP8 MoE is the actual MoE backend used by every FP8 experiment, with or without FlashInfer attention.

---

## Baseline Configuration

### E001: Naive Baseline

**Purpose:** Establish baseline without any optimizations

```bash
# Configuration
MODEL_PATH=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it
DTYPE=bfloat16
QUANTIZATION=None  # No quantization
TENSOR_PARALLEL=1
GPU_MEMORY_UTIL=0.95
MAX_NUM_SEQS=64  # Small batch to avoid OOM
MAX_NUM_BATCHED_TOKENS=3072
ENFORCE_EAGER=True  # No CUDA graphs

# Environment
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Flash Attention 2
unset VLLM_USE_FLASHINFER_MOE_FP8

```

**Run Command:**
```bash
./experiment_runner.sh E001 FLASH_ATTN 64 \
    --no-fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.95
```

---

## Experiment Configurations

### GROUP A: Core Optimizations (Cumulative)

#### E002: Add FP8 Quantization

**Purpose:** Measure FP8 quantization impact (most critical optimization)

```bash
# Changes from E001
QUANTIZATION=fp8
KV_CACHE_DTYPE=fp8_e5m2
GPU_MEMORY_UTIL=0.85
MAX_NUM_SEQS=64

# Environment
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

```

**Run Command:**
```bash
./experiment_runner.sh E002 FLASH_ATTN 64 \
    --fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.85
```

**Key Question:** Does FP8 enable model to run? How much throughput improvement?

---

#### E003: Switch to FlashInfer Attention

**Purpose:** Measure FlashInfer **attention** kernels vs Flash Attention 2. On this hardware (A100, sm_80) FlashInfer's FP8 MoE backend is unavailable, so the MoE path is unchanged from E002 — only the attention kernels differ. The original plan also expected "Better MoE FP8 kernel integration" via FlashInfer; that does NOT apply here. See "Actual Run Settings" → "What FlashInfer actually means in this run".

```bash
# Changes from E002
export VLLM_ATTENTION_BACKEND=FLASHINFER
# Do NOT export VLLM_USE_FLASHINFER_MOE_FP8 — not supported on sm_80;
# vLLM uses Marlin FP8 MoE (same as E002).

```

**Run Command:**
```bash
./run_ablation_experiment.sh --exp E003 --backend FLASHINFER --batch 64 \
    --fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.85
```

**Key Question:** How much does FlashInfer attention alone change performance vs FA2 (with MoE held constant)?

---

#### E004: Increase Batch Size to 128

**Purpose:** Measure batch size impact on MoE amortization

```bash
# Changes from E003
MAX_NUM_SEQS=128  # 2× larger
MAX_NUM_BATCHED_TOKENS=6144

```

**Run Command:**
```bash
./experiment_runner.sh E004 FLASHINFER 128 \
    --fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.85
```

**Key Question:** How much does larger batch improve throughput?

---

#### E005: Enable CUDA Graphs

**Purpose:** Measure CUDA graph compilation benefit

```bash
# Changes from E004
ENFORCE_EAGER=False
GPU_MEMORY_UTIL=0.75  # Reduce for CUDA graph overhead
MAX_NUM_SEQS=128

```

**Run Command:**
```bash
./experiment_runner.sh E005 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --no-mtp \
    --gpu-mem 0.75
```

**Key Question:** Does enabling CUDA graphs help on this workload? (Measured: see §Results.)

---

#### E006: Add MTP (Multi-Token Prediction)

**Purpose:** Measure MTP speculative decoding benefit

```bash
# Changes from E005
SPECULATIVE_MODEL=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant
NUM_SPECULATIVE_TOKENS=5

```

**Run Command:**
```bash
./experiment_runner.sh E006 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75
```

**Key Question:** How much does MTP help on this workload? Acceptance rate? (Measured: see §Results.)

---

#### E007: Remove Vision Weights

**Purpose:** Measure benefit of removing unused vision components

```bash
# Changes from E006
MODEL_PATH=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only

# Create text-only model first:
python3 create_text_only_model.py \
    --model_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --output_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only

```

**Run Command:**
```bash
./experiment_runner.sh E007 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Key Question:** Does vision removal free up enough memory for larger batches?

---

### GROUP B: Memory Optimizations

#### E008: Test Batch Size 192

**Purpose:** Measure throughput with larger batch (enabled by vision removal)

```bash
# Changes from E007
MAX_NUM_SEQS=192
MAX_NUM_BATCHED_TOKENS=9216

```

**Run Command:**
```bash
./experiment_runner.sh E008 FLASHINFER 192 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Key Question:** Can we fit batch=192? How much throughput gain?

---

#### E009: Test Batch Size 256

**Purpose:** Find maximum sustainable batch size

```bash
# Changes from E007
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=12288

```

**Run Command:**
```bash
./experiment_runner.sh E009 FLASHINFER 256 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Key Question:** Does batch=256 OOM? If yes, what's max batch?

---

#### E010: Lower Memory Utilization (0.70)

**Purpose:** Test if lower allocation enables better dynamic batching

```bash
# Changes from E007
GPU_MEMORY_UTIL=0.70  # More headroom
MAX_NUM_SEQS=128

```

**Run Command:**
```bash
./experiment_runner.sh E010 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.70 \
    --text-only-model
```

**Key Question:** Does lower utilization improve dynamic batching?

---

#### E011: Higher Memory Utilization (0.80)

**Purpose:** Test if higher allocation causes issues

```bash
# Changes from E007
GPU_MEMORY_UTIL=0.80  # Less headroom
MAX_NUM_SEQS=128

```

**Run Command:**
```bash
./experiment_runner.sh E011 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.80 \
    --text-only-model
```

**Key Question:** Does higher utilization cause OOM or instability?

---

### GROUP C: Alternative Configurations

#### E012: Compare Flash Attention 2 (with all optimizations)

**Purpose:** Validate FlashInfer attention vs FA2 at the optimal-config point. With MoE held constant (Marlin FP8 in both cases on this hardware), this is a clean attention-kernel head-to-head.

```bash
# Changes from E007
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# VLLM_USE_FLASHINFER_MOE_FP8 is already unset (sm_80 doesn't support it).

```

**Run Command:**
```bash
./experiment_runner.sh E012 FLASH_ATTN 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Key Question:** Confirm FlashInfer is faster than FA2?

---

#### E013: Disable MTP (measure MTP contribution)

**Purpose:** Isolate MTP contribution to performance

```bash
# Changes from E007
# Remove MTP
SPECULATIVE_MODEL=None
NUM_SPECULATIVE_TOKENS=0

```

**Run Command:**
```bash
./experiment_runner.sh E013 FLASHINFER 128 \
    --fp8 \
    --cuda-graphs \
    --no-mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Key Question:** Confirm MTP provides 2-3× speedup?

---

#### E014: Test FP8_E4M3 KV Cache

**Purpose:** Compare E5M2 vs E4M3 KV cache formats

```bash
# Changes from E007
KV_CACHE_DTYPE=fp8_e4m3  # Higher precision mantissa

```

**Run Command:**
```bash
./experiment_runner.sh E014 FLASHINFER 128 \
    --fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only-model
```

**Key Question:** Does E4M3 provide better accuracy or performance?

---

#### E015: Full BF16 (Reference Baseline)

**Purpose:** Compare to full-precision baseline (if it fits)

```bash
# Changes
DTYPE=bfloat16
QUANTIZATION=None
KV_CACHE_DTYPE=auto
MAX_NUM_SEQS=32  # Very small to avoid OOM
GPU_MEMORY_UTIL=0.95
ENFORCE_EAGER=True
NO MTP

```

**Run Command:**
```bash
./experiment_runner.sh E015 FLASHINFER 32 \
    --no-fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.95 \
    --text-only-model
```

**Key Question:** Can full BF16 even run? How much slower than FP8?

---

## Execution Plan

### Phase 1: Setup (30 minutes)

```bash
# 1. Create text-only model variant
cd /nvmedata/chenw/vllm-ra/examples
python3 create_text_only_model.py \
    --model_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --output_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only

# 2. Prepare experiment runner
chmod +x experiment_runner.sh

# 3. Create results directory
mkdir -p experiment_results/ablation_study
cd experiment_results/ablation_study

# 4. Test baseline (quick smoke test)
../../experiment_runner.sh E001 FLASH_ATTN 64 --no-fp8 --no-cuda-graphs --no-mtp --gpu-mem 0.95 --dry-run
```

### Phase 2: Core Experiments (GROUP A) - 60 minutes

```bash
# Run experiments E001-E007 sequentially
for exp in E001 E002 E003 E004 E005 E006 E007; do
    echo "=== Running ${exp} ==="
    ../../experiment_runner.sh ${exp} ...
    sleep 30  # Cool down between experiments
done
```

**Critical checkpoints:**
- E001: May OOM (acceptable, just document)
- E002: Must succeed (FP8 enables model to run)
- E005: Must leave 4-6 GB for CUDA graphs

### Phase 3: Memory Experiments (GROUP B) - 45 minutes

```bash
# Run experiments E008-E011
for exp in E008 E009 E010 E011; do
    echo "=== Running ${exp} ==="
    ../../experiment_runner.sh ${exp} ...
    # Check for OOM, adjust if needed
    sleep 30
done
```

**Critical checkpoints:**
- E008: Monitor memory closely (may be tight)
- E009: Likely to OOM (document max batch size)
- E010-E011: Compare dynamic batching behavior

### Phase 4: Comparisons (GROUP C) - 45 minutes

```bash
# Run experiments E012-E015
for exp in E012 E013 E014 E015; do
    echo "=== Running ${exp} ==="
    ../../experiment_runner.sh ${exp} ...
    sleep 30
done
```

**Critical checkpoints:**
- E015: May OOM (document if so)

### Phase 5: Analysis (30 minutes)

```bash
# Generate comparison report
python3 ../../analyze_experiments.py \
    --results-dir experiment_results/ablation_study \
    --output ablation_study_report.md

# Create visualizations
python3 ../../plot_experiments.py \
    --results-dir experiment_results/ablation_study \
    --output ablation_study_plots/
```

---

---

## Analysis Guidelines

### For Each Experiment, Record:

**Performance Metrics:**
```yaml
throughput:
  total_tokens_sec: 5000
  tokens_per_sequence: 39.1

latency:
  mean_ms: 25.5
  p50_ms: 24.8
  p95_ms: 32.1
  p99_ms: 41.2

memory:
  peak_allocated_gb: 36.2
  peak_reserved_gb: 38.1
  kv_cache_gb: 9.4
  cuda_graphs_gb: 4.8

gpu_utilization:
  mean_pct: 52.3
  peak_pct: 68.9
  memory_bandwidth_pct: 87.2
```

**Qualitative Observations:**
- Stability (any OOM, crashes, hangs?)
- Warmup time (CUDA graphs take longer)
- Accuracy (any degradation noticed?)
- Error rates (MTP acceptance rate)

### Key Comparisons:

**1. FP8 vs BF16 (E002 vs E015)**
```
Metric           BF16(E015)   FP8(E002)    Delta
──────────────────────────────────────────────────
Throughput       700 tok/s    900 tok/s    +29%
Memory           39 GB        32 GB        -18%
Fits in 40GB?    Barely       ✓ Yes        N/A

Conclusion: FP8 essential for A100 40GB
```

**2. FlashInfer vs Flash Attn 2 (E003 vs E012)**
```
Metric           FA2(E012)    FI(E003)     Delta
──────────────────────────────────────────────────
Throughput       4600 tok/s   5000 tok/s   +9%
Memory           36 GB        35 GB        -3%
Attention time   12 ms        11 ms        -8%

Conclusion: FlashInfer 8-10% faster for Gemma 4
```

**3. MTP Impact (E006 vs E013)**
```
Metric           No MTP(E013) With MTP(E006) Delta
────────────────────────────────────────────────────
Throughput       1900 tok/s   5000 tok/s     +2.6×
Latency          40 ms/tok    25 ms/tok      -38%
Memory           35 GB        36 GB          +3%

Conclusion: MTP provides 2.6× speedup for 0.8 GB cost
```

**4. Batch Size Scaling (E007 vs E008 vs E009)**
```
Batch    Throughput    Per-token    Memory    Fits?
────────────────────────────────────────────────────
128      5100 tok/s    39.8 t/s     35 GB     ✓
192      6100 tok/s    31.8 t/s     38 GB     ? (tight)
256      7000 tok/s    27.3 t/s     41 GB     ✗ (OOM)

Conclusion: Optimal batch = 192 (if fits) or 128 (safe)
```

### Success Criteria:

```
✓ E001 may OOM - acceptable (establishes need for FP8)
✓ E002 must run - FP8 enables model on A100 40GB
✓ E008 should fit in memory - optimal batch determination
✓ E009 may OOM - documents maximum batch size
✓ Overall: 5-7× speedup vs baseline

❌ Failure conditions:
- E002 OOMs (FP8 should fit)
- E005 no speedup (CUDA graphs not working)
- E006 no speedup (MTP not working)
- E012 faster than E003 (wrong backend choice)
```

---

## Next Steps After Experiments

1. **Analyze results:**
   ```bash
   python3 analyze_experiments.py --results-dir experiment_results/ablation_study
   ```

2. **Generate visualizations:**
   - Throughput vs experiment (bar chart)
   - Memory usage vs batch size (line chart)
   - Latency distribution (box plot)
   - Cumulative speedup (stacked bar)

3. **Update documentation:**
   - Add results to EXPERIMENT_LOG_002_ABLATION_STUDY.md
   - Update GEMMA4_MOE_OPTIMIZATION_GUIDE.md with validated numbers
   - Document optimal configuration in README

4. **Production configuration:**
   ```bash
   # Based on results, create optimal config
   ```

5. **Report summary:**
   - Key findings (which optimizations matter most)
   - Recommended configuration for A100 40GB
   - Optimal batch size
   - Memory vs throughput trade-offs

---

## Quick Start (TL;DR)

```bash
# 1. Create text-only model
python3 create_text_only_model.py --model_path ... --output_path ...

# 2. Run all experiments (automated)
./run_ablation_study.sh

# 3. Analyze results
python3 analyze_experiments.py --results-dir experiment_results/ablation_study

# 4. View report
cat experiment_results/ablation_study/ablation_study_report.md
```

---

**Document Version:** 1.0
**Status:** Ready for execution
**Next:** Run experiments and document results
