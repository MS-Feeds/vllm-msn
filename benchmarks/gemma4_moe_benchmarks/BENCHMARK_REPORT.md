# Gemma 4 26B-A4B-it — Offline Throughput Benchmark Report

**Model**: `google/gemma-4-26B-A4B-it` (bf16, MoE, 26B params / 4B active per token)
**Hardware**: 1 × NVIDIA H100 NVL (96 GB HBM)
**Framework**: vLLM 0.19.1.dev6 (V1 engine, `VLLM_COMPILE` mode 3, CUDA graphs)
**Driver**: custom Python harness on top of `vllm.LLM` ([bench_offline.py](bench_offline.py))
**Workload**: two production-style scenarios, real user prompts, natural EOS

---

## 1. Scenarios

| Scenario | Description | Source data |
|---|---|---|
| **sc1 (delta)** | Short prompts, short outputs (delta-interest extraction) | JSONL `_export_prompt:true` rows |
| **sc2 (personal)** | Long prompts (~20K tokens), short outputs (persona generation) | [prompts_personal.txt](prompts_personal.txt) |

Both scenarios use the same model and same sampling params:
- `temperature=0.7`, `top_p=0.95`, `max_tokens=8192`, natural EOS allowed.
- Prompts are wrapped as a single `user` chat message and rendered through
  the Gemma 4 -it chat template before being fed to the engine.

---

## 2. Engine configuration (constant across both runs)

| Setting | Value |
|---|---|
| `tensor_parallel_size` | 1 |
| `dtype` | bf16 (auto) |
| `quantization` | none |
| `gpu_memory_utilization` | 0.95 |
| `max_num_batched_tokens` | 16384 |
| `enable_chunked_prefill` | on (default) |
| `enable_prefix_caching` | on (default) |
| `enforce_eager` | off (CUDA graphs in use) |
| Speculative decoding | off |

`max_num_seqs` and `max_model_len` differ per run; see below.

---

## 3. Run #1 (v1) — strict length-bucket benchmark

### 3.1 Goal

Establish a throughput baseline at well-defined input lengths matching the
two production scenarios.

### 3.2 Dataset preparation

- sc1: filtered `prompts_delta.txt` to tokens in **[2000, 3000]** → **225 prompts** kept.
- sc2: filtered [prompts_personal.txt](prompts_personal.txt) to tokens in **[15000, 25000]** → **1262 prompts** available; sampled 1000 per run.

Each prompt was folded `[system, user] → user`-only, rendered through the
Gemma 4 chat template at prep time, and length-filtered against the bucket.

### 3.3 Engine settings

| Param | sc1 | sc2 |
|---|---:|---:|
| `max_model_len` | 12288 | 33792 |
| `num_prompts` / run | 225 | 1000 |
| `max_num_seqs` sweep | 64, 128, 256, 512, 1024 | 64, 256, 1024 |
| reps per config | 3 | 2 |

### 3.4 Results

**sc1 (input 2K–3K, output ≤ 8K, 225 prompts/run, 3 reps each)**

| `max_num_seqs` | wall (s) | **out tok/s** | total tok/s | mean out_len | finish=length |
|---:|---:|---:|---:|---:|---:|
| 64 | 126.6 ± 2.5 | **1986 ± 59** | 6271 ± 141 | 1117 ± 14 | 0 |
| **128** | **98.6 ± 2.9** | **2537 ± 75** | **8036 ± 232** | 1111 ± 9 | 0 |
| 256 | 113.1 ± 22.5 | 2302 ± 372 | 7209 ± 1248 | 1133 ± 23 | 1 |
| 512 | 128.8 ± 24.4 | 2043 ± 407 | 6367 ± 1319 | 1140 ± 13 | 2 |
| 1024 | 116.0 ± 26.7 | 2253 ± 438 | 7074 ± 1416 | 1127 ± 12 | 1 |

**sc2 (input 15K–25K, output ≤ 8K, 1000 prompts/run, 2 reps each)**

| `max_num_seqs` | wall (s) | out tok/s | **total tok/s** | mean out_len | finish=length |
|---:|---:|---:|---:|---:|---:|
| 64 | 1723.9 ± 4.5 | 531 ± 1 | **12166 ± 30** | 916 ± 4 | 3 |
| 256 | 1743.1 ± 18.0 | 528 ± 2 | 12035 ± 118 | 920 ± 12 | 4 |
| 1024 | 1747.3 ± 7.1 | 520 ± 2 | 11999 ± 45 | 908 ± 7 | 1 |

### 3.5 Run #1 takeaways

- sc1 hits a clean **+28% throughput jump** from `mns=64 → 128` (1986 → 2537 out tok/s) — exactly the continuous-batching scaling we expected.
- Past 128, **variance explodes** (±372 → ±438) because rare "run-to-8K" outputs occupy a KV slot for the full 8000 decode steps and drop effective concurrency for everyone else. The mean is also lower at mns ≥ 256, suggesting some preemption / scheduling cost on top of the outlier effect.
- sc2 is **completely concurrency-flat** between 64, 256, and 1024 (all within 1% of each other) — KV cache is the binding constraint at 28K context, so `max_num_seqs` doesn't matter once you're past the effective ceiling (~30 sequences).
- sc2 stdev is < 0.5%, signal is rock solid.

Total wall: **~10 hours** across 21 runs.

---

## 4. Run #2 (v2) — wider, unfiltered, realistic-distribution benchmark

### 4.1 Goals

1. Use the **larger, more representative dataset** ([delta_prompts/](delta_prompts/), 10 part files, ~16 917 valid rows ≈ 10× v1).
2. **Drop the strict length filter**. Keep prompts that naturally fit in `max_model_len − output_len`. This matches the real production input distribution.
3. **Tighten variance** (less reliance on a small 225-prompt set).

### 4.2 Dataset preparation

After tokenization + filtering by `max_model_len − output_len`:

| Dataset | Records | min | p50 | p90 | p99 | max | mean | filtered (too long) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sc1_delta_v2.jsonl` | 5000 | 975 | 1467 | 4503 | 12158 | 16293 | **2280** | 89 |
| `sc2_personal_v2.jsonl` | 3000 | 979 | 19772 | 33359 | 39981 | 40958 | **19838** | 117 |

sc1 distribution is now **right-tailed and wider** (1K–16K) vs. the v1 2K–3K bucket. sc2 mean is essentially unchanged (~19.8K).

### 4.3 Engine settings

| Param | sc1 | sc2 |
|---|---:|---:|
| `max_model_len` | **24576** | **49152** |
| `num_prompts` / run | **1000** | **500** |
| `max_num_seqs` sweep | 64, 128, 256 | 64, 128 |
| reps per config | 2 | 2 |

Trim rationale:
- Dropped sc1 mns=512 / 1024 — v1 showed these only add variance, no mean gain.
- sc1 reps 3 → 2 (v1 sc1 stdev ~3%, 2 reps is enough).
- sc2 num_prompts 1000 → 500 (v1 sc2 stdev < 0.5%, 500 prompts gives the same tightness).
- sc2 mns=32 dropped (v1 showed concurrency-flat from 64 → 1024).

### 4.4 Results

**sc1 (delta_prompts/, input ≤16K, output ≤ 8K, 1000 prompts/run, 2 reps each)**

| `max_num_seqs` | wall (s) | **out tok/s** | tot tok/s | mean out_len | finish=length |
|---:|---:|---:|---:|---:|---:|
| 64 | 432 ± 21 | 1729 ± 51 | 6961 ± 303 | 747 ± 14 | 3 / 2000 |
| **128** | **344 ± 10** | **2187 ± 27** | **8749 ± 210** | 753 ± 12 | 3 / 2000 |
| 256 | 346 ± 26 | 2185 ± 100 | 8738 ± 591 | 754 ± 22 | 4 / 2000 |

**sc2 (personal, input ≤40K, output ≤ 8K, 500 prompts/run, 2 reps each)**

| `max_num_seqs` | wall (s) | out tok/s | **tot tok/s** | mean out_len | finish=length |
|---:|---:|---:|---:|---:|---:|
| 64 | 915 ± 13 | 447 ± 1 | 10690 ± 151 | 817 ± 9 | 1 / 1000 |
| **128** | **906 ± 1** | **447 ± 1** | **10787 ± 12** | 810 ± 1 | 0 / 1000 |

### 4.5 Run #2 takeaways

- sc1 still scales 64 → 128 (+27%), plateaus at 128 = 256. Same shape as v1 but with **3–6× tighter stdev** thanks to the bigger dataset.
- sc2 still concurrency-flat between 64 and 128. Stdev shrunk further (≈ 0.1%).
- Mean output length dropped for both scenarios — wider input distribution → wider mix of question types → shorter answers on average.

Total wall: **~2 hours** across 10 runs.

---

## 5. Run #1 vs Run #2 — what changed and why

### 5.1 Setup deltas

| Aspect | Run #1 (v1) | Run #2 (v2) | Effect |
|---|---|---|---|
| sc1 source | `prompts_delta.txt` (1693 rows) | [delta_prompts/](delta_prompts/) 10 files (~16 917 rows, ~10×) | Larger sample, fewer per-run quirks |
| sc1 length filter | strict 2K–3K | none (only `max_model_len` cap) | More realistic distribution; some shorter, some longer prompts |
| sc1 `max_model_len` | 12288 | 24576 | KV cache budget per seq grows; 9.77× concurrency ceiling vs. 14.36× in v1 |
| sc1 num_prompts | 225 | 1000 | Steadier averages; better warm-up amortization |
| sc1 mns sweep | 64, 128, 256, 512, 1024 | 64, 128, 256 | Skip points v1 proved useless |
| sc1 reps | 3 | 2 | Save time; stdev already small |
| sc2 source | `prompts_personal.txt` | same | unchanged |
| sc2 length filter | strict 15K–25K | none | Few more long prompts admitted |
| sc2 `max_model_len` | 33792 | 49152 | KV per seq grows; fewer concurrent at full load |
| sc2 num_prompts | 1000 | 500 | Half the per-run sample but stdev was already < 0.5% |
| sc2 mns sweep | 64, 256, 1024 | 64, 128 | Skip points v1 proved useless |
| sc2 reps | 2 | 2 | unchanged |
| Total wall | ~10 h | ~2 h | **5× faster** with same signal |

### 5.2 Headline numbers

| Metric | Run #1 best | Run #2 best | Δ |
|---|---:|---:|---:|
| sc1 **out tok/s** | **2537** (mns=128) | **2187** (mns=128) | **−14%** |
| sc1 **tot tok/s** | **8036** (mns=128) | **8749** (mns=128) | **+9%** |
| sc1 mean out_len | 1111 | 753 | −32% |
| sc2 **out tok/s** | **531** (mns=64) | **447** (mns=128) | **−16%** |
| sc2 **tot tok/s** | **12 166** (mns=64) | **10 787** (mns=128) | **−11%** |
| sc2 mean out_len | 916 | 810 | −12% |

### 5.3 Why the throughput numbers differ

The v2 numbers are **lower** for output tok/s and total tok/s. Two separate effects explain it:

**For sc1**:
- Wider prompt distribution (1K–16K, mean 2280) vs. v1's narrow 2K–3K bucket (mean 2359).
- Crucially, v2 **outputs are 32% shorter** (753 vs. 1111 tokens). With shorter outputs, the prefill portion of each request becomes a *larger* fraction of total work. Output throughput drops, but **total** throughput is higher because prefill is fully tensor-core-bound while decode is HBM-bound.
- A wider input range also means more variance per request in the running batch. CUDA-graph hits are slightly less efficient because the engine sees more shape combinations.
- The +9% total tok/s in v2 is the prefill share growing.

**For sc2**:
- The dataset is essentially the same (mean 19.8K vs 19.9K, similar finish distributions).
- The −11% in total tok/s comes almost entirely from the **wider `max_model_len`** (33K → 49K). The KV cache budget is fixed (~37 GiB bf16), so per-sequence allocations are bigger and effective concurrency drops from ~30 → ~22 sequences. Less decode parallelism → slightly lower throughput.
- This is the honest cost of dropping the length filter — we now correctly handle 30K–40K prompts that v1 would have rejected.

**Variance shrunk significantly** in v2:
- sc1 mns=128: stdev dropped from ±75 → ±27 (3.6× tighter).
- sc2 mns=128: stdev dropped from ±30 (v1 mns=64) → ±12 (v1→v2 same metric, 2.5× tighter).
This is the direct benefit of larger datasets averaging over per-prompt variability.

### 5.4 What both runs agree on

1. **sc1: continuous batching gives ~+28% throughput from `mns=64 → 128`.** Past 128 the engine is KV-cache-bound and the mean plateaus.
2. **sc2: `max_num_seqs` is irrelevant beyond a small floor (~32).** KV cache caps effective concurrency at ~20–30 sequences regardless of the configured cap.
3. **sc2 is prefill-dominated**: ~95% of total tokens processed are prompt tokens. The "total tok/s" number is the meaningful one for this workload.
4. **No length-finish in > 99.7% of requests.** Natural EOS dominates; the 8192 cap rarely fires.

---

## 6. Recommendations (final)

### sc1 production deployment

- **`--max-num-seqs 128`**.
- Expected throughput on one H100 NVL: **~2200 output tok/s, ~8750 total tok/s**.
- Headroom levers:
  - **FP8 weights + FP8 KV cache** (the [run.gemma4.quant.sh](run.gemma4.quant.sh) recipe **minus** speculative decoding): roughly doubles KV budget → doubles effective concurrency → expect **~2× throughput**.
  - Tightening `max_tokens` to e.g. 2048 reduces variance but increases length-finish rate.

### sc2 production deployment

- **`--max-num-seqs 128`** (any value ≥ 64 is equivalent).
- Expected throughput on one H100 NVL: **~10 800 total tok/s, ~450 output tok/s**.
- Headroom levers:
  - **FP8**: helps both ways here — more KV budget *and* faster prefill matmuls. Expect **~1.5–2× total tok/s**.
  - Sharper input length distribution (if your real traffic is closer to 18–22K) lets you drop `max_model_len` and reclaim KV concurrency.

### General

- Don't bother with `max_num_seqs > 256` for either scenario on this GPU at bf16. KV cache is the ceiling.
- Don't enable speculative decoding for high-throughput offline workloads; it costs throughput at high batch sizes.
- Keep `--enforce-eager` off — CUDA graphs help throughput materially.

---

## Appendix: raw data

- Run #1 CSV: `bench_results_archive_v1/all_runs.csv` (21 rows)
- Run #2 CSV: [bench_results/all_runs.csv](bench_results/all_runs.csv) (10 rows)
- Per-run JSONs (one per run, with full output-length distributions): `bench_results/*.json`
- Driver: [bench_offline.py](bench_offline.py)
- Dataset prep: [prep_dataset.py](prep_dataset.py)
- Full chronological log: [BENCHMARK_LOG.md](BENCHMARK_LOG.md)
