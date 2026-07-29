# Reproducing the Gemma 4 vLLM throughput benchmarks

End-to-end steps to rebuild the image, prepare datasets, and reproduce the
numbers in [BENCHMARK_LOG.md](BENCHMARK_LOG.md) §6 (sweep v2 — the current
[bench_results/](bench_results/)).

---

## 0. Requirements

- One **NVIDIA H100 NVL** (96 GB HBM) — or any single GPU with ≥ 80 GB HBM
  for bf16 26B + 49K context KV.
- Docker with NVIDIA Container Toolkit (`--gpus all` works).
- ~60 GB free disk for the HF cache (model weights ~48 GB + torch.compile cache).
- Hugging Face access to `google/gemma-4-26B-A4B-it`. The model is public; no
  token required for download.
- Inputs already in the workspace:
  - [prompts_personal.txt](prompts_personal.txt) — raw sc2 source (JSONL with
    `_export_prompt:true` rows).
  - [delta_prompts/](delta_prompts/) — 10 part files, raw sc1 source.

If [prompts_delta.txt](_unused/prompts_delta.txt) is needed (v1 sc1 source),
it lives in `_unused/`.

---

## 1. Build the image

```bash
cd /path/to/vllm_gemma_model
docker build -t vllm-gemma4:local .
```

The image is `FROM vllm/vllm-openai:gemma4` plus Azure-ML extras (see
[Dockerfile](Dockerfile)). vLLM, PyTorch, and FlashAttention are baked in.
The Python driver scripts ([bench_offline.py](bench_offline.py),
[prep_dataset.py](prep_dataset.py)) are **not** baked in — they are
bind-mounted in at runtime so edits don't require a rebuild.

---

## 2. Prepare datasets

Runs in the same image so the Gemma 4 tokenizer is available. Both commands
share two mounts:

- `./hf_cache → /root/.cache/huggingface` (model + tokenizer cache, persisted)
- `./ → /work` with `workdir=/work` (so [prep_dataset.py](prep_dataset.py) and
  the raw inputs are visible)

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD/hf_cache:/root/.cache/huggingface" \
  -v "$PWD:/work" -w /work \
  --entrypoint bash vllm-gemma4:local -c '
set -e
echo "--- sc1: delta_prompts/ , max-tokens=16384 (24K context - 8K output) ---"
python3 prep_dataset.py \
  --src delta_prompts/ \
  --dst datasets/sc1_delta_v2.jsonl \
  --min-tokens 1 --max-tokens 16384 --max-keep 5000

echo "--- sc2: prompts_personal.txt , max-tokens=40960 (49K context - 8K output - slack) ---"
python3 prep_dataset.py \
  --src prompts_personal.txt \
  --dst datasets/sc2_personal_v2.jsonl \
  --min-tokens 1 --max-tokens 40960 --max-keep 3000
'
```

What this does (see [prep_dataset.py](prep_dataset.py)):

1. Reads `_export_prompt:true` rows from the source(s).
2. Folds the production `[system, user]` messages into a single combined
   string (`[SYSTEM]\n…\n\n<user content>`) because vLLM's `CustomDataset`
   loader only supports one role.
3. Renders through the Gemma 4 -it chat template (role=user) and tokenizes
   to count tokens.
4. Drops prompts that wouldn't fit `max_model_len − output_len`.
5. Writes `{"prompt": "..."}` JSONL.

Expected outputs:

| Dataset | Records | min | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| `datasets/sc1_delta_v2.jsonl`  | 5000 | ~975 | ~1467 | ~4503 | ~12158 | ≤ 16384 |
| `datasets/sc2_personal_v2.jsonl` | 3000 | ~979 | ~19772 | ~33359 | ~39981 | ≤ 40958 |

First run downloads the Gemma 4 tokenizer (~few MB).

---

## 3. Run the benchmark sweep

This is the exact command used to produce [bench_results/](bench_results/)
(sweep v2). Detached container, runs both sweeps back-to-back, ~2 h total
on H100 NVL.

```bash
docker run -d --name bench-runner-v2 --gpus all --ipc=host \
  -v "$PWD/hf_cache:/root/.cache/huggingface" \
  -v "$PWD:/work" -w /work \
  --entrypoint bash vllm-gemma4:local -c '
set -e
echo "=== START SWEEP V2 ===" ; date -u
echo "--- sc1 sweep ---"
python3 bench_offline.py --scenario sc1 --reps 2 --max-num-seqs 64,128,256
echo "--- sc2 sweep ---"
python3 bench_offline.py --scenario sc2 --reps 2 --max-num-seqs 64,128
echo "=== DONE SWEEP V2 ===" ; date -u
'
```

Monitor:

```bash
docker logs -f bench-runner-v2
# or, after it exits:
docker logs bench-runner-v2 | tail -200
```

Per-config build cost (cold compile cache) is ~95 s. The engine is **rebuilt
per `max_num_seqs` value** and **reused across reps** within that value, so
reps 2 of each cell are cheap.

### To reproduce v1 (archived under [bench_results_archive_v1/](bench_results_archive_v1/))

```bash
# v1 needs the older v1 datasets, currently in _unused/datasets/:
#   _unused/datasets/scenario1_2k-3k.jsonl
#   _unused/datasets/scenario2_15k-25k.jsonl
# Restore them or re-run prep_dataset.py with the strict v1 buckets:
#   sc1: prompts_delta.txt, --min-tokens 2000 --max-tokens 3000
#   sc2: prompts_personal.txt, --min-tokens 15000 --max-tokens 25000
# Then point bench_offline.py SCENARIOS at those paths (or temporarily
# rename them to sc1_delta_v2.jsonl / sc2_personal_v2.jsonl).

docker run -d --name bench-runner --gpus all --ipc=host \
  -v "$PWD/hf_cache:/root/.cache/huggingface" \
  -v "$PWD:/work" -w /work \
  --entrypoint bash vllm-gemma4:local -c '
set -x
python3 bench_offline.py --scenario sc1 --reps 3 --max-num-seqs 64,128,256,512,1024
python3 bench_offline.py --scenario sc2 --reps 2 --max-num-seqs 64,256,1024
'
```

---

## 4. Engine settings (the same across all runs)

Defined in [bench_offline.py](bench_offline.py) — `SCENARIOS` dict:

| Knob | sc1 (delta) | sc2 (personal) |
|---|---:|---:|
| dataset | `datasets/sc1_delta_v2.jsonl` | `datasets/sc2_personal_v2.jsonl` |
| `num_prompts` (per rep) | 1000 | 500 |
| `output_len` cap | 8192 | 8192 |
| `max_model_len` | 24576 | 49152 |
| `max_num_batched_tokens` | 16384 | 16384 |
| `gpu_memory_utilization` | 0.95 | 0.95 |
| weights dtype | bf16 (image default) | bf16 |
| prefix caching | on (vLLM V1 default) | on |
| chunked prefill | on (vLLM V1 default) | on |
| FP8 / spec decoding | **off** | **off** |

Sampling: `temperature=0.7`, `top_p=0.95`, `max_tokens=8192`, `seed=rep#`
(different seed per rep for honest stdev), `ignore_eos=False`.

---

## 5. Outputs

After the sweep finishes:

```
bench_results/
  all_runs.csv              # one row per (scenario, max_num_seqs, rep)
  sc1_mns64_rep1.json       # per-run details incl. per-request out_lens
  sc1_mns64_rep2.json
  sc1_mns128_rep1.json
  ...
```

CSV columns (see `CSV_FIELDS` in [bench_offline.py](bench_offline.py#L32-L42)):

`ts, scenario, num_prompts, output_len_cap, max_model_len, max_num_batched_tokens, gpu_mem_util, max_num_seqs, rep, seed, elapsed_time, requests_per_second, prompt_tokens_total, output_tokens_total, total_tokens, prompt_tps, output_tps, total_tps, out_len_mean, out_len_stdev, out_len_p50, out_len_p90, out_len_max, finish_stop, finish_length, finish_other`

Headline numbers you should land near (H100 NVL, sweep v2):

| Scenario | Best config | out tok/s | total tok/s | mean out_len |
|---|---|---:|---:|---:|
| sc1 (delta)    | `mns=128` | ~2200 | ~8750  | ~750 |
| sc2 (personal) | `mns=128` |  ~447 | ~10800 | ~810 |

Stdev across the 2 reps should be < 1–2 % at the recommended `mns`.

---

## 6. Cleanup / re-runs

```bash
# Remove a finished bench container before re-running with the same name
docker rm bench-runner-v2

# Wipe results to do a clean re-run
rm -f bench_results/all_runs.csv bench_results/sc*_mns*_rep*.json

# The hf_cache mount persists weights + the torch.compile cache.
# Wiping it forces a ~25 min cold rebuild (HF download + compile).
```

---

## 7. Notes / known fidelity gaps

- **Roles are folded.** Production sends `[system, user]`; the bench sends
  one `user` turn containing `[SYSTEM]\n…\n\n<user content>`. Token count
  differs by a few special tokens. Throughput is faithful; output quality
  may differ slightly.
- **Prefix caching is on (vLLM V1 default).** The shared system header is
  hashed identically across requests in a sweep, so prefill cost is amortized
  for sc1. To measure cold prefill, pass `enable_prefix_caching=False` to
  `LLM(...)` in `bench_offline.py`.
- **Engine reuse across reps.** The same `LLM` instance handles rep1 → rep2
  within a `max_num_seqs` value, so APC also carries over between reps.
- **`bench_sweep.py` is not used.** Earlier CLI-based driver, kept in
  [_unused/](_unused/) for reference. The `vllm bench throughput
  --dataset-name custom` path it depends on is not wired up in this image's
  parser.
