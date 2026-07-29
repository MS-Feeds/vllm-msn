# Reproducing the Gemma 4 production-shaped throughput benchmarks

This guide reproduces two end-to-end throughput runs on
**Google Gemma 4 26B-A4B-it** under a single offline driver:

- **bf16 run** — full-precision baseline.
- **FP8 run**  — same workload with `--quantization fp8 --kv-cache-dtype fp8`.

Each run covers two production-shaped scenarios:

| Scenario | Description | Token shape |
|---|---|---|
| **sc1 (delta)**    | Decode-heavy: short prompts, moderate-length outputs.   | input ≤ 16 K, output ≤ 8 K |
| **sc2 (persona)**  | Prefill-heavy: long prompts, moderate-length outputs.   | input ≤ 40 K, output ≤ 8 K |

Each scenario uses 10 000 prompts. The vLLM engine is rebuilt between
scenarios because `max_model_len` differs; it is **not** rebuilt between
chunks within a scenario.

---

## 1. Prerequisites

### Hardware

- The anchor numbers in §6 are from a single **NVIDIA H100 NVL (96 GB HBM)**.
- bf16 weights take ~48 GB; the sc2 KV cache at `max_model_len=49152` needs
  another ~25–35 GB at bf16. FP8 roughly halves both.

### Software

- Docker with the NVIDIA Container Toolkit (`docker run --gpus all` works).
- Build the image once:
  ```bash
  docker build -t vllm-gemma4:local .
  ```
- Hugging Face access to `google/gemma-4-26B-A4B-it` (public; no token needed).

### Branch optimization assumptions

This repro assumes you are running this branch's Gemma4 path (not a vanilla
upstream wheel). The following code-level optimizations are part of the
baseline and apply to both bf16 and FP8 runs:

- Fused dual-RMSNorm Triton path in `vllm/model_executor/layers/gemma4_fused_ops.py`.
- Fused-path 3-D coverage in `vllm/model_executor/models/gemma4.py`:
  `[B, S, H]` tensors are flattened to `[B*S, H]` for the fused kernel and
  reshaped back after execution.
- Fused kernel layout guard: hidden dimension must be contiguous
  (`stride(-1) == 1`) for `x1/x2/residual`; otherwise the model uses the
  unfused fallback path.
- Redundant V-projection removal for `k_eq_v` full-attention layers in
  `vllm/model_executor/models/gemma4.py`.
- Text-only multimodal safety guards in `vllm/model_executor/models/gemma4_mm.py`.

If these are missing (for example on a different branch or official wheel),
absolute throughput can differ and the numbers in this document are not
directly comparable.

### Files

| Path | What |
|---|---|
| [Dockerfile](Dockerfile)                       | Image definition |
| [bench_offline.py](bench_offline.py)           | Python driver |
| [prep_dataset.py](prep_dataset.py)             | One-time dataset prep |
| [run.bench.fullpass.sh](run.bench.fullpass.sh) | Wrapper that runs both halves |
| `datasets/sc1_delta.jsonl`                     | sc1 prompts (10 000 rows) |
| `datasets/sc2_personal.jsonl`                  | sc2 prompts (10 000 rows) |
| `hf_cache/`                                    | Populated with ~48 GB of weights + compile cache on first run |

---

## 2. Datasets (one-time)

If the dataset JSONLs already exist, skip this section.

Inputs are upstream JSONL dumps in `delta_prompts/` (for sc1) and
`persona_prompts/` (for sc2). The prep step keeps only the prompt rows, folds
the `[system, user]` messages into a single string, filters by Gemma-4
rendered token count, and writes one `{"prompt": "..."}` JSON per line.

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD/hf_cache:/root/.cache/huggingface" \
  -v "$PWD:/work" -w /work \
  --entrypoint bash vllm-gemma4:local -c '
set -e
# sc1: short prompts, cap = max_model_len(24576) − output_len(8192) = 16384
python3 prep_dataset.py \
  --src delta_prompts/ \
  --dst datasets/sc1_delta.jsonl \
  --min-tokens 1 --max-tokens 16384 \
  --max-keep 10000

# sc2: long prompts, cap = max_model_len(49152) − output_len(8192) = 40960
python3 prep_dataset.py \
  --src persona_prompts/ \
  --dst datasets/sc2_personal.jsonl \
  --min-tokens 1 --max-tokens 40960 \
  --max-keep 10000
'
```

### Prompt selection semantics

- `prep_dataset.py` walks the input directory in **sorted filename order** and
  reads each file line-by-line, stopping at `--max-keep`.
- [bench_offline.py](bench_offline.py) `load_prompts(...)` then reads the
  prepped JSONL line-by-line, stopping at `--num-prompts`.
- **No shuffle, no random sampling.** bf16 and FP8 therefore see the same
  10 000 prompts in the same order — that is what makes the comparison
  apples-to-apples.
- If the upstream files are sorted by anything correlated with prompt length
  or topic (date, user-id, etc.), the first 10 000 may be skewed. Shuffle
  the prepped JSONL once with a fixed seed before running the bench if
  representativeness matters more than determinism.

### Expected length distributions

| Dataset | min | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| `sc1_delta.jsonl`    | ~975 | ~1 500  | ~4 500  | ~12 000 | ≤ 16 384 |
| `sc2_personal.jsonl` | ~980 | ~20 000 | ~33 000 | ~40 000 | ≤ 40 958 |

---

## 3. Engine settings

| Knob | sc1 (delta) | sc2 (persona) |
|---|---:|---:|
| `num_prompts`                  | 10 000   | 10 000   |
| `max_model_len`                | 24 576   | 49 152   |
| `max_num_batched_tokens`       | 16 384   | 16 384   |
| `gpu_memory_utilization`       | 0.90     | 0.90     |
| `max_num_seqs`                 | 128      | 64       |
| `output_len` cap (`max_tokens`)| 8 192    | 8 192    |
| `chunk_size`                   | 2 000    | 1 000    |
| reps                           | 1        | 1        |
| prefix caching                 | on (vLLM V1 default) | on |
| chunked prefill                | on (vLLM V1 default) | on |
| speculative decoding           | off      | off      |
| weights dtype                  | bf16 (Gemma 4's config) | bf16 |
| attention backend              | `TRITON_ATTN` (forced for Gemma 4) | same |

The FP8 run additionally passes `--quantization fp8 --kv-cache-dtype fp8`.
Everything else is identical between the two runs.

Sampling: `temperature=0.7`, `top_p=0.95`, `max_tokens=8192`, `seed=1`,
`ignore_eos=False`.

Note on the attention backend: Gemma 4 has heterogeneous head dimensions
(256 and 512), so vLLM forces `TRITON_ATTN` for both bf16 and FP8 to avoid
mixed-backend numerical divergence. This affects the FP8 result — see §6.

---

## 4. Quick start

[run.bench.fullpass.sh](run.bench.fullpass.sh) runs both halves sequentially
in one container with `set -euo pipefail` so any per-config failure aborts
the run immediately.

```bash
chmod +x run.bench.fullpass.sh
./run.bench.fullpass.sh
```

The script:

- Pre-checks the image and both datasets exist.
- Creates `bench_results_bf16/` and `bench_results_fp8/` as the host user.
- Runs (sc1 bf16 → sc2 bf16 → sc1 fp8 → sc2 fp8).

Useful overrides:

```bash
NUM_PROMPTS=100 SC1_CHUNK=0 SC2_CHUNK=0 ./run.bench.fullpass.sh   # smoke test
./run.bench.fullpass.sh --skip-fp8                                # bf16 only
./run.bench.fullpass.sh --skip-bf16                               # fp8 only
```

---

## 5. Manual version

If you'd rather run the two halves as separate `docker run` invocations
(e.g. on different nodes, or to keep one running while you iterate on the
other), use the blocks below. They are equivalent to what the wrapper script
runs.

### bf16

```bash
docker run --rm --name bench-bf16 --entrypoint bash --gpus all --ipc=host \
  -v "$PWD/hf_cache:/root/.cache/huggingface" \
  -v "$PWD:/work" -w /work \
  vllm-gemma4:local -c '
set -e
echo "=== bf16 start ==="; date -u

python3 bench_offline.py --scenario sc1 --reps 1 \
  --max-num-seqs 128 \
  --dataset datasets/sc1_delta.jsonl --num-prompts 10000 \
  --chunk-size 2000 \
  --gpu-mem-util 0.90 \
  --output-dir bench_results_bf16

python3 bench_offline.py --scenario sc2 --reps 1 \
  --max-num-seqs 64 \
  --dataset datasets/sc2_personal.jsonl --num-prompts 10000 \
  --chunk-size 1000 \
  --gpu-mem-util 0.90 \
  --output-dir bench_results_bf16

echo "=== bf16 done ==="; date -u
'
```

### FP8

Same commands plus `--quantization fp8 --kv-cache-dtype fp8`, into a
separate output directory:

```bash
docker run --rm --name bench-fp8 --entrypoint bash --gpus all --ipc=host \
  -v "$PWD/hf_cache:/root/.cache/huggingface" \
  -v "$PWD:/work" -w /work \
  vllm-gemma4:local -c '
set -e
echo "=== fp8 start ==="; date -u

python3 bench_offline.py --scenario sc1 --reps 1 \
  --max-num-seqs 128 \
  --dataset datasets/sc1_delta.jsonl --num-prompts 10000 \
  --chunk-size 2000 \
  --gpu-mem-util 0.90 \
  --quantization fp8 --kv-cache-dtype fp8 \
  --output-dir bench_results_fp8

python3 bench_offline.py --scenario sc2 --reps 1 \
  --max-num-seqs 64 \
  --dataset datasets/sc2_personal.jsonl --num-prompts 10000 \
  --chunk-size 1000 \
  --gpu-mem-util 0.90 \
  --quantization fp8 --kv-cache-dtype fp8 \
  --output-dir bench_results_fp8

echo "=== fp8 done ==="; date -u
'
```

### Flags worth knowing

- `--entrypoint bash` — the image's default `ENTRYPOINT` is `vllm serve`;
  without this override, the `bash -c '...'` blob is parsed as a CLI flag
  and the container exits before any Python runs.
- `--gpus all --ipc=host` — required by vLLM (multi-process workers use
  shared memory).

---

## 6. Anchor numbers and what to expect

Each run writes one CSV row plus one JSON file per chunk:

```
bench_results_<bf16|fp8>/
  all_runs.csv                            # one row per (scenario, mns, rep, chunk)
  sc1_mns128_rep1_chunk001of005.json
  ...
  sc2_mns64_rep1_chunk010of010.json
```

CSV columns are listed in `CSV_FIELDS` in [bench_offline.py](bench_offline.py).

### Reference results — single H100 NVL, 10 000 prompts per scenario

Means across all chunks (5 for sc1, 10 for sc2):

| Run  | Scenario      | `out_tps`  | `total_tps` | mean out_len | stop ratio | wall |
|---|---|---:|---:|---:|---:|---:|
| bf16 | sc1 (delta)   | 1 870 ± 14 | 8 134 ± 121 | 1 338 | 99.87 % | ~1 h 36 min |
| FP8  | sc1 (delta)   | 2 056 ± 21 | 9 226 ± 189 | 1 286 | 99.92 % | ~1 h 45 min |
| bf16 | sc2 (persona) | 422 ± 6    | 9 954 ± 184 | 880   | 99.63 % | ~5 h 50 min |
| FP8  | sc2 (persona) | 389 ± 4    | 9 278 ± 159 | 869   | 99.81 % | ~6 h 15 min |

### FP8 vs bf16 ratio (the portable signal across hardware)

| Scenario | `out_tps` ratio (FP8 / bf16) |
|---|---:|
| sc1 (decode-heavy)   | **1.10× (+10 %)**, mild win |
| sc2 (prefill-heavy)  | **0.92× (−8 %)**, regression |

If you reproduce on H100 NVL and your numbers are off by more than ~5 %,
suspect (in order): prefix caching disabled, wrong `max_model_len`,
`gpu_memory_utilization` too low, or a host process competing for the GPU.
On different hardware the absolute numbers will change — the bf16 ↔ FP8
ratios are what should stay roughly consistent.

### Why FP8 regresses on sc2

Gemma 4's heterogeneous attention head dimensions (256 and 512) force vLLM
to use the generic `TRITON_ATTN` backend instead of the much-faster
`FLASH_ATTN_V3` / `FLASHINFER` paths. `TRITON_ATTN` has an immature FP8
prefill path; sc2 is ~95 % prefill, so this dominates. On top of that:

- There is no tuned MoE tile config for `(E=128, N=704, NVIDIA_H100_NVL, fp8_w8a8)` —
  vLLM falls back to a generic Triton MoE config.
- The run uses on-the-fly FP8 quantization of bf16 weights (no pre-calibrated
  FP8 checkpoint), so attention Q / K / V / prob scales default to 1.0,
  steering the attention kernel into a slower fallback path.
- FP8 quadruples the KV-cache budget (max concurrency 25.94× vs 7.61× for
  49 152-token requests) but `max_num_seqs=64` caps concurrency well below
  either ceiling — so the cache win never materializes.

If a tuned MoE config and a pre-calibrated FP8 checkpoint become available,
the FP8 sc2 number should improve.

---

## 7. Verification checklist

1. Container exited with code **0**.
2. `all_runs.csv` has 16 lines (1 header + 5 sc1 rows + 10 sc2 rows) in each
   output directory.
3. Per-chunk JSON files:
   - `sc1_mns128_rep1_chunk{001..005}of005.json` (5)
   - `sc2_mns64_rep1_chunk{001..010}of010.json` (10)
4. `finish_length` per chunk is small (< 10) — almost all requests should
   stop on EOS, not on the 8192 length cap.
5. Headline `out_tps` per scenario within ±5 % of the anchors in §6.

---

## 8. Cleanup and partial re-runs

```bash
docker rm -f bench-runner-fullpass bench-bf16 bench-fp8 2>/dev/null
rm -rf bench_results_bf16 bench_results_fp8
```

If a chunked run dies mid-way:

- Per-chunk JSON files for completed chunks survive; the relaunch will
  **overwrite** them when it re-runs the same chunk.
- `all_runs.csv` is append-only — a naive relaunch produces **duplicate
  rows** for previously-completed chunks. Either delete the CSV first or
  dedupe afterwards on `(scenario, max_num_seqs, rep, chunk_index)`.
- There is no automatic resume-from-chunk-N. The relaunch starts at chunk 1.

The `hf_cache/` mount persists weights and the torch.compile cache. Keep it
across runs to avoid a 20–25 min cold rebuild.
