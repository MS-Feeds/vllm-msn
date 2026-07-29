# vLLM Gemma 4 — Setup & Benchmark Log

A running record of everything done in this workspace.

---

## Workspace overview

Files at start:

| File | Purpose |
|---|---|
| [Dockerfile](Dockerfile) | Image built on `vllm/vllm-openai:gemma4` with NCCL/SSH/multimodal deps |
| [run.gemma4.sh](run.gemma4.sh) | Baseline launcher for Gemma 4 26B-A4B-it |
| [run.gemma4.full.sh](run.gemma4.full.sh) | Gemma 4 31B-it with tool-calling + multimodal |
| [run.gemma4.quant.sh](run.gemma4.quant.sh) | Gemma 4 26B-A4B-it FP8 + n-gram speculative decoding |
| [run.qwen36.sh](run.qwen36.sh) | Qwen 3.6 35B-A3B with tool-calling, 256K context |
| [tool_chat_template_gemma4.jinja](tool_chat_template_gemma4.jinja) | Gemma 4 chat template (tool calls, multimodal, reasoning channel) |
| `prompts_delta.txt` | JSONL with `_export_prompt:true` rows (1693 valid, median ~1.4K tokens) — Scenario 1 source |
| `prompts_personal.txt` | JSONL with `_export_prompt:true` rows (4139 valid, median ~17K tokens) — Scenario 2 source |

Hardware: single **NVIDIA H100 NVL** (96 GB HBM).

---

## 1. Build & first launch (`run.gemma4.sh`)

Container image already present: `vllm-gemma4:local` (23.4 GB) built from [Dockerfile](Dockerfile).

### 1.1 Initial wrong assumption

I initially assumed `google/gemma-4-26B-A4B-it` was not a public HF model and
had the script changed to the base id `google/gemma-4-26B-A4B`. The base
ran fine but had **no chat template** in its tokenizer, so
`/v1/chat/completions` returned 400. Later verified via `huggingface.co/api`
that the `-it` variant **is** public — reverted.

### 1.2 First run command

```bash
docker run -d --name vllm-gemma4 --gpus all --ipc=host \
  -p 8100:8100 \
  -v "$PWD/hf_cache:/root/.cache/huggingface" \
  -v "$PWD/run.gemma4.sh:/workspace/run.gemma4.sh:ro" \
  --entrypoint bash \
  vllm-gemma4:local \
  /workspace/run.gemma4.sh 8100 1 32768 0.90 auto gemma4
```

Mounts:
- `./hf_cache → /root/.cache/huggingface` — keeps weights + torch.compile cache inside the workdir
- `./run.gemma4.sh → /workspace/run.gemma4.sh:ro` — picks up edits without image rebuild

Args: `PORT=8100 TP_SIZE=1 MAX_LEN=32768 GPU_UTIL=0.90 DTYPE=auto SERVED_NAME=gemma4`.

### 1.3 Startup timeline

| Phase | Cold (first ever) | Warm (compile cache + weights cached) |
|---|---|---|
| HF download | 1407 s (~48 GB) | 58 s (`-it` delta) |
| Weight load to GPU | 13 s | 8.7 s |
| `torch.compile` (VLLM_COMPILE mode 3) | 31 s | 31 s |
| CUDA graph capture (51 sizes) | ~6 s | ~6 s |
| **Total** | ~24 min | ~3 min 20 s |

Model loading footprint: **48.5 GiB** on GPU.
KV cache budget at `gpu-memory-utilization=0.9`: **30.5 GiB → 133,312 tokens**
(12.79× concurrency for 32K context).

### 1.4 Verification

```bash
curl -s http://localhost:8100/v1/completions \
  -d '{"model":"gemma4","prompt":"The capital of France is","max_tokens":20}'
# -> "the capital of France is one of the most visited cities in the world. ..."

curl -s http://localhost:8100/v1/chat/completions \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"What is vLLM?"}]}'
# -> "vLLM is a high-throughput and memory-efficient engine for LLM inference
#     and serving that utilizes PagedAttention to optimize KV cache management."
```

---

## 2. Theory recap

### 2.1 Why weights aren't in the image

Scripts resolve `$model` from `${_ModelDataPath_}/model` (Azure ML),
`./INPUT_model_dir`, or fall back to the HF id. Download happens implicitly
when `vllm serve` receives an HF id. Image stays small; auth/license stay
out of the image.

### 2.2 Chat template

A Jinja2 string in `tokenizer_config.json` that converts the structured
message list into the special-token prompt the model was trained on. Missing
template on the base model → 400 on `/v1/chat/completions`. The `-it`
variant has one built in. The repo's
[tool_chat_template_gemma4.jinja](tool_chat_template_gemma4.jinja) extends
this to serialize Gemma 4's tool-calling, reasoning channel, and multimodal
markers — used by [run.gemma4.full.sh](run.gemma4.full.sh).

### 2.3 Continuous batching (still needed offline)

Decode is HBM-bandwidth bound. One decode step on a 26B bf16 model streams
~52 GB to produce **1 token at batch=1** vs. **N tokens at batch=N**. Static
batching collapses to batch=1 whenever a sequence finishes early. vLLM's
continuous batching (re-pick the batch every iteration) + PagedAttention
(no padding) + chunked prefill (interleave new prefill with ongoing decodes)
are what keep the effective batch high.

### 2.4 "Stream" disambiguation

"Single-stream decode" = one request in flight, not one CUDA stream. vLLM
packs N sequences into one batched matmul on a **single** CUDA stream — the
parallelism is in the batch dimension of the tensor.

---

## 3. Benchmark plan

### 3.1 Tool

`vllm bench throughput` — offline, in-process engine, no HTTP overhead.

### 3.2 Knobs that matter (offline)

| Flag | Recommendation |
|---|---|
| `--max-num-seqs` | Sweep upward until throughput plateaus or KV cache fills |
| `--max-num-batched-tokens` | Large (e.g. 16K) — TTFT doesn't matter |
| `--gpu-memory-utilization` | Push to 0.95 |
| `--max-model-len` | Cap to actual workload max (smaller → more KV budget) |
| `--enable-chunked-prefill`, `--enable-prefix-caching` | Default on, keep |
| `--enforce-eager` | **Don't** — CUDA graphs help throughput |
| `--quantization fp8` + `--kv-cache-dtype fp8` | Biggest single H100 lever |
| Speculative decoding | **Off** — hurts at high batch |

### 3.3 Scenarios (user-stated)

| Scenario | Input | Output |
|---|---|---|
| 1 | 2K–3K tokens | up to 8K |
| 2 | ~20K tokens | up to 8K |

### 3.4 Dataset

Will use the workspace's real JSONL files, converted to ShareGPT-style JSON
for `vllm bench throughput --dataset-name sharegpt`. The ShareGPT loader
feeds `conversations[0]["value"]` to the engine **as raw text** (no chat
template applied), so we pre-render each prompt through Gemma 4's chat
template before writing the JSON.

Bucketing (strict, matches user spec):

| Bucket | Source | Token range |
|---|---|---|
| Scenario 1 | `prompts_delta.txt` | 2000–3000 |
| Scenario 2 | `prompts_personal.txt` | 15000–25000 |

Output policy: **natural EOS allowed, cap at 8192**. Reports will include
actual mean / p90 output length.

### 3.5 Sweep

| Run | input bucket | output cap | `--max-num-seqs` |
|---|---|---|---|
| 1a / 1b / 1c | 2K–3K | 8192 | 64 / 256 / 512 |
| 2a / 2b / 2c | 15K–25K | 8192 | 64 / 256 / 512 |

Common engine flags: `--gpu-memory-utilization 0.95`, `--max-num-batched-tokens 16384`,
`--max-model-len = ceil(p99_input + 8192)`, bf16 weights, `--seed 0`,
`--no-oversample`, 1000 prompts per run.

### 3.6 Reported metrics per run

- Wall time
- Request throughput (req/s)
- **Output-token throughput** (tok/s) — headline
- Prompt-token throughput (tok/s)
- Total-token throughput (tok/s)
- Actual mean / p90 output length
- Peak running batch size

---

## 4. Dataset preparation

### 4.1 ShareGPT path — failed

First wrote a converter to ShareGPT-style JSON. **Blocker:** vLLM's
`ShareGPTDataset.sample()` hard-codes `is_valid_sequence(..., max_prompt_len=1024,
max_total_len=2048)`. Our 2K–25K prompts would all be discarded. Not
configurable from the throughput CLI.

### 4.2 Pivot to `--dataset-name custom`

`CustomDataset` reads JSONL `{"prompt": "..."}` records, accepts any prompt
length, and applies the chat template internally as a single `user` message.
We fold `system + user` into one combined string. The rendered token count
differs from a true multi-turn `[system, user]` render by a few special
tokens — negligible for throughput measurement.

[prep_dataset.py](prep_dataset.py) now:
1. Reads `_export_prompt:true` rows.
2. Folds `[system, user]` → combined string (`[SYSTEM]\n...\n\n<user content>`).
3. Renders via Gemma 4 -it chat template with `role=user` to measure tokens.
4. Filters into the requested bucket.
5. Writes one `{"prompt": ...}` JSON object per line.

### 4.3 Results

| Scenario | Source | Bucket | Kept | min | median | p90 | max |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | `prompts_delta.txt` | 2000–3000 tok | **225** | 2000 | 2359 | 2825 | 2998 |
| 2 | `prompts_personal.txt` | 15000–25000 tok | **1262** | 15005 | 19937 | 24070 | 24998 |

Filtered out:
- Scenario 1: 1126 too_short + 342 too_long (of 1693 valid rows).
- Scenario 2: 1396 too_short + 1481 too_long (of 4139 valid rows).

Outputs: `datasets/scenario1_2k-3k.jsonl`, `datasets/scenario2_15k-25k.jsonl`.

### 4.4 No oversampling, deliberate

Scenario 1: `--num-prompts 225` (= dataset size → no oversampling).
Scenario 2: `--num-prompts 1000` (< dataset size 1262 → no oversampling).

---

## 5. Benchmark runs

### 5.1 Sweep matrix

| Scenario | input | output cap | num_prompts | max_num_seqs values | reps |
|---|---|---|---:|---|---:|
| 1 (delta)    | 2K–3K   | 8192 | 225  | 64 / 128 / 256 / 512 / 1024 | 3 |
| 2 (personal) | 15K–25K | 8192 | 1000 | 32 / 64 / 128 / 256 / 512 / 1024 | TBD |

Total: 15 + ~12–18 ≈ 27–33 runs.

### 5.2 Engine settings (all runs)

- bf16 weights
- `--gpu-memory-utilization 0.95`
- `--max-num-batched-tokens 16384`
- `--max-model-len`: 12288 (sc1) / 33792 (sc2)
- `--seed = rep #` (different seed per replicate for honest stdev)
- `--trust-remote-code`
- No speculative decoding, no FP8 quantization

### 5.3 Stdev approach

`vllm bench throughput`'s output JSON only contains aggregates (one number per
run). Stdev comes from **multiple reps per config with different seeds**.
[bench_offline.py](bench_offline.py) is a custom driver that uses `vllm.LLM`
directly — same engine as the bench tool — but lets us:

- Use **`vllm bench throughput --dataset-name custom`** was attempted first
  but the `custom` choice isn't wired into the `throughput` subcommand's
  parser in this image (it's only available for `vllm bench serve`). Switched
  to a Python driver.
- Capture per-request output length (mean, stdev, p50, p90, max), finish
  reasons, and timing.
- Reuse the same engine across reps within a config (saves ~95 s rebuild
  per rep — only pay the build cost once per `max_num_seqs`).

### 5.4 Sanity run

Before the full sweep, ran scenario 1 / `max_num_seqs=64` / 1 rep:

```
elapsed=129.0 s  req/s=1.74  out_tok/s=1922  total_tok/s=6125
out_len(mean ± sd) = 1101.93 ± 308.58  (p90=1493, max=2281)
finish: stop=225 / length=0
```

Key takeaways:
- **All 225 requests stopped at EOS naturally** at ~1100 tok mean — the 8K cap
  is *not* binding. Throughput is prefill-dominated for sc1.
- KV-cache budget: 161,920 tokens at `max_model_len=12288` →
  **14.36× concurrency** ceiling for sc1.
- Engine build: ~95 s (compile cache hit + load + CUDA graph capture).

### 5.5 Sweep plan (running)

Reduced sc2 from 6×3 to 3×2 to keep total time ≤ ~3 h:

| Scenario | max_num_seqs | reps | runs |
|---|---|---:|---:|
| 1 (delta, 2–3K in) | 64, 128, 256, 512, 1024 | 3 | 15 |
| 2 (personal, 15–25K in) | 64, 256, 1024 | 2 | 6 |

Launched as a detached container `bench-runner` running both sweeps back-to-back.

### 5.6 Run log

Status: **complete** (21/21 runs).

#### sc1 results (input 2K–3K, output ≤ 8K, 225 prompts/run, 3 reps each)

| `max_num_seqs` | wall (s) | **out tok/s** | total tok/s | mean out_len | finish=length |
|---:|---:|---:|---:|---:|---:|
| 64   | 126.6 ± 2.5  | **1986 ± 59**  | 6271 ± 141   | 1117 ± 14 | 0 |
| **128**  | **98.6 ± 2.9**   | **2537 ± 75**  | **8036 ± 232**   | 1111 ± 9  | 0 |
| 256  | 113.1 ± 22.5 | 2302 ± 372  | 7209 ± 1248  | 1133 ± 23 | 1 |
| 512  | 128.8 ± 24.4 | 2043 ± 407  | 6367 ± 1319  | 1140 ± 13 | 2 |
| 1024 | 116.0 ± 26.7 | 2253 ± 438  | 7074 ± 1416  | 1127 ± 12 | 1 |

Headline: **2537 ± 75 output tok/s at `mns=128`**. Past 128 the engine is
KV-cache-bound; mean throughput is the same but variance grows because rare
"run-to-8K" outputs occupy a KV slot for the full 8000 decode steps while
other sequences finish.

#### sc2 results (input 15K–25K, output ≤ 8K, 1000 prompts/run, 2 reps each)

| `max_num_seqs` | wall (s) | out tok/s | **total tok/s** | mean out_len | finish=length |
|---:|---:|---:|---:|---:|---:|
| 64   | 1723.9 ± 4.5  | 531 ± 1 | **12166 ± 30**  | 916 ± 4  | 3 |
| 256  | 1743.1 ± 18.0 | 528 ± 2 | 12035 ± 118 | 920 ± 12 | 4 |
| 1024 | 1747.3 ± 7.1  | 520 ± 2 | 11999 ± 45  | 908 ± 7  | 1 |

Headline: **~12 100 total tok/s** (prefill-dominated: ~95% of work is prefill
on these long prompts), ~530 output tok/s. Concurrency cap has **no effect**:
KV cache caps effective batch at ~30 sequences regardless of `max_num_seqs`.
Stdev across reps is < 0.5 %.

### 5.7 Recommendations

1. **sc1-like workload** (short prompts, naturally short outputs):
   - Set `--max-num-seqs 128`.
   - Expect ~2500 output tok/s, ~8000 total tok/s per H100 NVL.
   - To push further: FP8 weights + FP8 KV cache (see
     [run.gemma4.quant.sh](run.gemma4.quant.sh) *minus* speculative decoding)
     should ~2× this by doubling KV budget.

2. **sc2-like workload** (20K input):
   - `--max-num-seqs` doesn't matter; pick the smallest value that gives
     headroom (e.g. 64).
   - Expect ~12 000 total tok/s, ~530 output tok/s per H100 NVL.
   - Bottleneck is prefill compute + KV memory. FP8 should give ~1.5–2× by
     enlarging KV budget *and* speeding prefill matmuls.

3. **Variance interpretation**: sc1 reps with `mns ≥ 256` had a single
   "length=8192" output that dragged throughput down for ~5–6 minutes. Real
   property of workload + sampling, not the engine. Capping `max_tokens` more
   aggressively (e.g. 2048) would tighten throughput considerably.

---

## 6. Sweep v2 — wider data, no length filter

User concerns from v1:
1. Latency numbers weren't recorded (we only got run-level wall time).
2. v1 sc1 had high variance at `mns ≥ 256` due to the small 225-prompt set
   triggering rare "run-to-8K" outliers.
3. v1 used a strict 2K–3K and 15K–25K length filter — not representative.

User-supplied fix:
- New, larger delta dataset: 10 part files at
  [delta_prompts/](delta_prompts/) (~16,917 valid `_export_prompt` rows total,
  vs. 1,693 in v1).
- Drop the length filter. Keep only prompts that fit within `max_model_len`.

### 6.1 New data profile

Sampled 10,000 of 16,917 rows from delta_prompts/, tokenized via Gemma 4 -it:

| min | p10 | p50 | p90 | p95 | p99 | max | mean ± sd |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 971 | 1018 | 1497 | 4890 | 8008 | 21496 | 224601 | 2822 ± 5796 |

Right-tailed: 90% of prompts ≤ 5K tokens, but ~5% spike to 8K–200K+.

### 6.2 v2 sweep plan

| Scenario | dataset | `max_model_len` | num_prompts | reps | `max_num_seqs` |
|---|---|---:|---:|---:|---|
| sc1 (delta) | new 10-file pool | **24576** | 1000 | 2 | 64, 128, 256 |
| sc2 (personal) | unchanged | **49152** | 500 | 2 | 64, 128 |

`max_model_len` widened to drop only the > 95-percentile outliers and let the
common workload fit naturally:
- sc1: 24K context (allows ~16K input + 8K output). Drops ~5% (the > 16K
  prompts). Old v1 setting (12288) would have dropped ~10% of the new pool.
- sc2: 49K context. Drops < 1% of personal prompts.

Trims vs. naive plan (5 h → 1.9 h):
- Dropped sc1 mns=512 (per user) and mns=1024 (per user — also did so in v1).
- sc1 reps 3 → 2 (v1 sc1 stdev was already ~3%, 2 reps fine).
- sc2 num_prompts 1000 → 500 (v1 sc2 stdev was < 0.5%, 500 prompts gives
  similar tightness).
- sc2 dropped mns=32 (v1 data showed concurrency-flat for sc2 from 64 → 1024).

### 6.3 New v2 dataset stats

After tokenization + filtering by `max_model_len − output_len`:

| Dataset | Records | min | p50 | p90 | p99 | max | mean | filtered (too long) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sc1_delta_v2.jsonl`  | 5000 | 975 | 1467 | 4503 | 12158 | 16293 | 2280 | 89 |
| `sc2_personal_v2.jsonl` | 3000 | 979 | 19772 | 33359 | 39981 | 40958 | 19838 | 117 |

(We stop reading after 5000 / 3000 records; far more than the 1000 / 500
we'll actually use per run.)

### 6.4 Estimated wall time

Anchored on v1 per-prompt timings:

| Sweep | per-run wall (est.) | runs | sweep wall (est.) |
|---|---:|---:|---:|
| sc1 64,128,256 × 2 reps | ~430 s | 6 | ~43 min |
| sc2 64,128 × 2 reps | ~900 s | 4 | ~60 min |
| Engine builds (5 configs) | ~95 s | 5 | ~8 min |
| Dataset prep | — | — | ~3 min |
| **Total** | | | **~1.9 h** |

### 6.5 Run log

Status: **complete** (10/10 runs, total wall ~2 h, container exited 0).

#### sc1 v2 results (delta_prompts/, ≤16K input + ≤8K output, 1000 prompts/run)

| `max_num_seqs` | wall (s) | **out tok/s** | tot tok/s | mean out_len | finish=length |
|---:|---:|---:|---:|---:|---:|
| 64  | 432 ± 21 | 1729 ± 51 | 6961 ± 303 | 747 ± 14 | 3 / 2000 |
| **128** | **344 ± 10** | **2187 ± 27** | **8749 ± 210** | 753 ± 12 | 3 / 2000 |
| 256 | 346 ± 26 | 2185 ± 100 | 8738 ± 591 | 754 ± 22 | 4 / 2000 |

**Headline: ~2200 output tok/s, ~8750 total tok/s.** mns=128 and 256 give the
same mean — KV-cache budget caps effective concurrency (9.77× at
`max_model_len=24576`). mns=128 has tighter stdev so it's the recommendation.

#### sc2 v2 results (personal, ≤40K input + ≤8K output, 500 prompts/run)

| `max_num_seqs` | wall (s) | out tok/s | **tot tok/s** | mean out_len | finish=length |
|---:|---:|---:|---:|---:|---:|
| 64  | 915 ± 13 | 447 ± 1 | 10690 ± 151 | 817 ± 9 | 1 / 1000 |
| **128** | **906 ± 1** | **447 ± 1** | **10787 ± 12** | 810 ± 1 | 0 / 1000 |

**Headline: ~10800 total tok/s, ~447 output tok/s.** Concurrency cap doesn't
matter (KV-bound at ~25 effective concurrent seqs at 40K context). Stdev
microscopic (< 0.5%).

### 6.6 v1 vs v2 comparison

| Metric | v1 best | v2 best | Δ |
|---|---:|---:|---:|
| sc1 out tok/s | 2537 (mns=128) | 2187 (mns=128) | −14% |
| sc1 tot tok/s | 8036 (mns=128) | 8749 (mns=128) | +9% |
| sc1 mean out_len | 1111 | 753 | −32% |
| sc2 tot tok/s | 12166 (mns=64) | 10787 (mns=128) | −11% |
| sc2 mean out_len | 916 | 810 | −11% |

What changed:
- **sc1 outputs are shorter** in v2 (750 vs 1110 mean). v2 includes shorter
  prompts (down to ~1K tokens), whose answers tend to be shorter. Output tok/s
  drops; total tok/s goes up because prefill becomes a larger share.
- **sc2 saw wider context** (`max_model_len` 33K → 49K). Each sequence
  reserves more KV blocks → fewer concurrent → slightly lower throughput.
  Honest cost of removing the length filter.

### 6.7 Final recommendations

1. **sc1-like workload** (delta-style, mostly short prompts, short outputs):
   - `--max-num-seqs 128`.
   - Expect **~2200 output tok/s, ~8750 total tok/s** on one H100 NVL.
   - FP8 weights + FP8 KV cache should ~2× this by doubling KV budget.

2. **sc2-like workload** (personal-style, 20K+ input):
   - Any `--max-num-seqs ≥ 64`. mns=128 gives identical throughput with
     slightly tighter variance.
   - Expect **~10800 total tok/s, ~447 output tok/s** on one H100 NVL.
   - This is prefill-dominated; FP8 helps both ways (more KV budget + faster
     prefill matmuls). Expect ~1.5–2× with FP8.

3. **Variance**: with `max_tokens=8192` and the natural-EOS workload, ~0.15%
   of sc1 requests hit the length cap. Each one occupies a KV slot for the
   full 8192 decode steps, which is why mns=256 has wider stdev than mns=128.
   Tightening the cap (e.g. 2048) would reduce variance at the cost of more
   length-truncations.

---

## 7. Ablation study — Round 2 (H100 NVL)

### Background

An A100 80GB ablation study (15 experiments, E001–E015) was conducted on
2026-05-21, documented in `examples/EXPERIMENT_PLAN_ABLATION_STUDY.md`.
That study used `AsyncLLMEngine` with per-request TTFT/TPOT metrics on a
1K-row dataset (`layer1_delta_1k_test.txt`). Best A100 result: **983.7 output
tok/s** (E011: FP8 + FlashInfer + CUDA graphs + MTP k=5 + text-only +
gpu_mem=0.80) — 1.50× the A100 BF16 baseline.

This section describes Round 2: re-running the same 15-config ablation matrix
on the **H100 NVL** using the offline `vllm.LLM` framework from `bench_offline.py`.

Key differences vs. the A100 run:

| Dimension | A100 run | H100 Round 2 |
|-----------|----------|--------------|
| Hardware | A100 80GB PCIe (sm_80) | H100 NVL 96GB (sm_90) |
| Driver | vLLM async engine | vLLM offline `LLM()` |
| Dataset | `layer1_delta_1k_test.txt` | `datasets/sc1_delta_v2.jsonl` (sc1) |
| Output cap | 1024 tokens | 8192 tokens |
| FlashInfer FP8 MoE | ✗ (sm_80 unsupported) | ✓ (sm_90 native FP8) |
| FP8 KV cache (E014) | FAIL (Triton fp8e4nv) | Expected PASS |

H100 BF16 baseline from Sweep v2 (Section 6):
- **sc1: ~2187 output tok/s** (mns=128, max_model_len=24576)
- **sc2: ~447 output tok/s** (mns=128, max_model_len=49152)

### New scripts

| Script | Purpose |
|--------|---------|
| [`bench_experiment.py`](bench_experiment.py) | Ablation driver — 15-experiment matrix encoded as dicts; uses `vllm.LLM` offline |
| [`run_experiments.sh`](run_experiments.sh) | Shell wrapper — sets `VLLM_ATTENTION_BACKEND`, `VLLM_USE_FLASHINFER_MOE_FP8` per experiment before Python import; handles --all / subset runs |
| [`analyze_results.py`](analyze_results.py) | Reads `results/all_runs.csv`, prints comparison table vs A100 baselines, writes `results/summary.md` |

### Experiment matrix (carries over from A100 study)

| Exp | Config added | Key flags |
|-----|-------------|-----------|
| E001 | BF16 baseline | FA2, eager, no MTP, mns=64, mem=0.95 |
| E002 | +FP8 weights | FA2, eager, no MTP, mns=64, mem=0.85 |
| E003 | +FlashInfer attn (+ FP8 MoE on H100) | FI, eager, no MTP, mns=64, mem=0.85 |
| E004 | +batch 128 | FI, eager, no MTP, mns=128, mem=0.85 |
| E005 | +CUDA graphs | FI, CUDA-gr, no MTP, mns=128, mem=0.75 |
| E006 | +MTP k=5 | FI, CUDA-gr, MTP, mns=128, mem=0.75 |
| E007 | text-only model | FI, CUDA-gr, MTP, mns=128, mem=0.75, text |
| E008 | batch 192 | FI, CUDA-gr, MTP, mns=192, mem=0.75, text |
| E009 | batch 256 | FI, CUDA-gr, MTP, mns=256, mem=0.75, text |
| E010 | gpu_mem=0.70 | FI, CUDA-gr, MTP, mns=128, mem=0.70, text |
| E011 | gpu_mem=0.80 | FI, CUDA-gr, MTP, mns=128, mem=0.80, text |
| E012 | FA2 at optimal | FA2, CUDA-gr, MTP, mns=128, mem=0.75, text |
| E013 | no MTP | FI, CUDA-gr, no MTP, mns=128, mem=0.75, text |
| E014 | FP8 E4M3 KV cache | FI, CUDA-gr, MTP, FP8-kv, mns=128, mem=0.75, text |
| E015 | BF16 ref (text, no opts) | FI, eager, no MTP, mns=32, mem=0.95, text |

### How to run

```bash
cd benchmarks/gemma4_moe_benchmarks
chmod +x run_experiments.sh

# Export model paths (adjust to your mount points):
export GEMMA4_MODEL_PATH=/mnt/models/gemma-4-26B-A4B-it
export GEMMA4_TEXT_ONLY_MODEL_PATH=/mnt/models/gemma-4-26B-A4B-it-text-only
export GEMMA4_ASSISTANT_MODEL_PATH=/mnt/models/gemma-4-26B-A4B-it-assistant

# Run all 15 experiments on sc1 (≈ 4–6 h on H100 NVL):
./run_experiments.sh --all --scenario sc1 --reps 2

# Run the optimal config only (E011) to validate quickly:
./run_experiments.sh E011 --scenario sc1 --reps 3

# Run the key comparison subset (baseline, best, MTP ablation):
./run_experiments.sh E001,E011,E013 --scenario sc1 --reps 2

# After runs complete, generate comparison table:
python3 analyze_results.py
```

Results are written to `results/all_runs.csv` (cumulative) and
`results/<exp>_<scenario>_rep<N>.json` (per-run).
Summary table: `results/summary.md`.

### Expected outcomes (H100 vs A100 predictions)

| Dimension | A100 result | H100 prediction | Reason |
|-----------|------------|-----------------|--------|
| BF16 baseline (E001) | 654.9 tok/s | ~2100–2500 tok/s | H100 has ~3× the HBM BW + larger NVL memory |
| FP8 gain (E002 vs E001) | +5.7% | +10–20% | H100 has native FP8 tensor cores |
| FlashInfer + FP8 MoE (E003) | +0.6% over E002 | **+15–25% over E002** | `VLLM_USE_FLASHINFER_MOE_FP8=1` now active |
| MTP (E006 vs E005) | +26.8% | +15–30% | MTP acceptance rate model/workload dependent |
| CUDA graphs (E005 vs E004) | −2.6% | small gain expected | H100 graph overhead lower, prompts more uniform |
| FP8 KV cache (E014) | FAIL | expected PASS | Triton fp8e4nv supported on sm_90 |
| Overall best | 983.7 tok/s (A100) | ~3000–4000 tok/s | Proportional HBM BW gain + FP8 MoE improvement |

### Results

_To be filled in after runs complete._

See `results/summary.md` for auto-generated comparison table.





