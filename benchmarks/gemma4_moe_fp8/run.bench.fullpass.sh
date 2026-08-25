#!/usr/bin/env bash
# =============================================================================
# run.bench.fullpass.sh
#
# Reproduce the two production-shaped throughput runs on Gemma 4 26B-A4B-it
# documented in REPRODUCE_PRODSHAPE.md.
#
# What this runs (in order, all inside one container):
#   1. bf16 / sc1   10k delta   prompts   max_num_seqs=128  chunk=2000
#   2. bf16 / sc2   10k persona prompts   max_num_seqs=64   chunk=1000
#   3. fp8  / sc1   10k delta   prompts   max_num_seqs=128  chunk=2000
#   4. fp8  / sc2   10k persona prompts   max_num_seqs=64   chunk=1000
#
# Total wall time on a single H100 NVL: ~15 hours (sc2 dominates).
# Other hardware will scale up or down accordingly.
#
# Outputs:
#   bench_results_bf16/   5 sc1 chunks + 10 sc2 chunks + all_runs.csv
#   bench_results_fp8/    5 sc1 chunks + 10 sc2 chunks + all_runs.csv
#
# Prerequisites:
#   - Docker + NVIDIA Container Toolkit (`docker run --gpus all` works).
#   - Image `vllm-gemma4:local` already built from this repo's Dockerfile.
#   - HF cache mounted at ./hf_cache (will pull google/gemma-4-26B-A4B-it
#     weights on first run, ~48 GB).
#   - Datasets at datasets/sc1_delta.jsonl and datasets/sc2_personal.jsonl
#     (produce via prep_dataset.py — see REPRODUCE_PRODSHAPE.md §2).
#
# Usage:
#   ./run.bench.fullpass.sh                  # full 10k each, both halves
#   ./run.bench.fullpass.sh --skip-bf16      # only the FP8 half
#   ./run.bench.fullpass.sh --skip-fp8       # only the bf16 half
#   NUM_PROMPTS=100 ./run.bench.fullpass.sh  # smoke version
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Tunables (env-overridable so smoke tests / different hardware can adjust)
# -----------------------------------------------------------------------------
IMAGE="${IMAGE:-vllm-gemma4:local}"

# Number of prompts per scenario. num_prompts=0 means "all rows in the dataset".
NUM_PROMPTS="${NUM_PROMPTS:-10000}"

# Chunk sizes: each chunk writes a JSON + CSV row, so the run survives crashes
# at chunk boundaries. The engine is NOT rebuilt between chunks.
SC1_CHUNK="${SC1_CHUNK:-2000}"     # sc1 (short prompts) — 5 chunks for 10k
SC2_CHUNK="${SC2_CHUNK:-1000}"     # sc2 (long prompts)  — 10 chunks for 10k

# Best max_num_seqs values for this model on H100 NVL. Re-bench on different
# hardware before changing.
SC1_MNS="${SC1_MNS:-128}"
SC2_MNS="${SC2_MNS:-64}"

# gpu_memory_utilization. 0.90 is the verified-safe value on this image.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# Output directories
OUT_BF16="${OUT_BF16:-bench_results_bf16}"
OUT_FP8="${OUT_FP8:-bench_results_fp8}"

# Single container runs both halves sequentially
CONTAINER="${CONTAINER:-bench-runner-fullpass}"

# Datasets (prep'd via prep_dataset.py — see REPRODUCE_PRODSHAPE.md §2)
SC1_DATASET="${SC1_DATASET:-datasets/sc1_delta.jsonl}"
SC2_DATASET="${SC2_DATASET:-datasets/sc2_personal.jsonl}"

# -----------------------------------------------------------------------------
# CLI flags
# -----------------------------------------------------------------------------
SKIP_BF16=0
SKIP_FP8=0
for arg in "$@"; do
  case "$arg" in
    --skip-bf16) SKIP_BF16=1 ;;
    --skip-fp8)  SKIP_FP8=1 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
echo "=== pre-flight ==="
date -u

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: image '$IMAGE' not found. Build it first:"
  echo "  docker build -t $IMAGE ."
  exit 1
fi

if [[ ! -f "$SC1_DATASET" ]]; then
  echo "ERROR: sc1 dataset not found at $SC1_DATASET"
  echo "  See REPRODUCE_PRODSHAPE.md §2 for prep instructions."
  exit 1
fi
if [[ ! -f "$SC2_DATASET" ]]; then
  echo "ERROR: sc2 dataset not found at $SC2_DATASET"
  echo "  See REPRODUCE_PRODSHAPE.md §2 for prep instructions."
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "ERROR: container '$CONTAINER' already exists. Remove it first:"
  echo "  docker rm -f $CONTAINER"
  exit 1
fi

WORKDIR="$(pwd)"

# Pre-create output dirs so they're owned by the host user (otherwise the
# container creates them as root and `rm` will need sudo).
mkdir -p "$OUT_BF16" "$OUT_FP8"

echo "Image:        $IMAGE"
echo "Workdir:      $WORKDIR"
echo "Num prompts:  $NUM_PROMPTS per scenario"
echo "bf16 output:  $OUT_BF16   (skip=$SKIP_BF16)"
echo "fp8 output:   $OUT_FP8    (skip=$SKIP_FP8)"
echo

# -----------------------------------------------------------------------------
# Build the bash command that will run inside the container.
#
# One big bash -c blob keeps everything in a single container so the HF cache
# and torch.compile cache hit on the second engine build. `set -e` plus the
# bench script's non-zero exit code on any per-config failure means an error
# in the bf16 half aborts before the FP8 half starts.
# -----------------------------------------------------------------------------

run_bf16='
echo "=== START bf16 ==="
date -u

echo "--- bf16 / sc1 (delta, '"$NUM_PROMPTS"' prompts, mns='"$SC1_MNS"', chunk='"$SC1_CHUNK"') ---"
python3 bench_offline.py --scenario sc1 --reps 1 \
  --max-num-seqs '"$SC1_MNS"' \
  --dataset '"$SC1_DATASET"' --num-prompts '"$NUM_PROMPTS"' \
  --chunk-size '"$SC1_CHUNK"' \
  --gpu-mem-util '"$GPU_MEM_UTIL"' \
  --output-dir '"$OUT_BF16"'

echo "--- bf16 / sc2 (persona, '"$NUM_PROMPTS"' prompts, mns='"$SC2_MNS"', chunk='"$SC2_CHUNK"') ---"
python3 bench_offline.py --scenario sc2 --reps 1 \
  --max-num-seqs '"$SC2_MNS"' \
  --dataset '"$SC2_DATASET"' --num-prompts '"$NUM_PROMPTS"' \
  --chunk-size '"$SC2_CHUNK"' \
  --gpu-mem-util '"$GPU_MEM_UTIL"' \
  --output-dir '"$OUT_BF16"'

echo "=== DONE bf16 ==="
date -u
'

run_fp8='
echo "=== START fp8 (W+KV) ==="
date -u

echo "--- fp8 / sc1 (delta, '"$NUM_PROMPTS"' prompts, mns='"$SC1_MNS"', chunk='"$SC1_CHUNK"') ---"
python3 bench_offline.py --scenario sc1 --reps 1 \
  --max-num-seqs '"$SC1_MNS"' \
  --dataset '"$SC1_DATASET"' --num-prompts '"$NUM_PROMPTS"' \
  --chunk-size '"$SC1_CHUNK"' \
  --gpu-mem-util '"$GPU_MEM_UTIL"' \
  --quantization fp8 --kv-cache-dtype fp8 \
  --output-dir '"$OUT_FP8"'

echo "--- fp8 / sc2 (persona, '"$NUM_PROMPTS"' prompts, mns='"$SC2_MNS"', chunk='"$SC2_CHUNK"') ---"
python3 bench_offline.py --scenario sc2 --reps 1 \
  --max-num-seqs '"$SC2_MNS"' \
  --dataset '"$SC2_DATASET"' --num-prompts '"$NUM_PROMPTS"' \
  --chunk-size '"$SC2_CHUNK"' \
  --gpu-mem-util '"$GPU_MEM_UTIL"' \
  --quantization fp8 --kv-cache-dtype fp8 \
  --output-dir '"$OUT_FP8"'

echo "=== DONE fp8 ==="
date -u
'

container_script='set -euo pipefail'$'\n'
[[ "$SKIP_BF16" == 0 ]] && container_script+="$run_bf16"
[[ "$SKIP_FP8"  == 0 ]] && container_script+="$run_fp8"
container_script+=$'\necho "=== ALL DONE ==="\ndate -u\n'

# -----------------------------------------------------------------------------
# Launch the container.
#
# Required flags:
#   --gpus all          give the container the GPU
#   --ipc=host          required by vLLM for shared-memory IPC w/ workers
#   --entrypoint bash   the image's default ENTRYPOINT is `vllm serve`; we
#                       want bash so we can run our python driver instead.
#   -v hf_cache:...     persist HF weights + torch.compile cache across runs.
#   -v $PWD:/work       mount the repo so bench_offline.py / prep_dataset.py
#                       / datasets/ are visible inside the container without
#                       baking them into the image.
#   -w /work            cwd inside the container.
#
# Foreground (not detached) so the user can ctrl-C cleanly and so this
# script's exit code matches the container's. For unattended overnight runs,
# replace `--rm` with `-d` and tail with `docker logs -f $CONTAINER`.
# -----------------------------------------------------------------------------
echo "=== launching container '$CONTAINER' ==="
docker run --rm \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --entrypoint bash \
  -v "$WORKDIR/hf_cache:/root/.cache/huggingface" \
  -v "$WORKDIR:/work" -w /work \
  "$IMAGE" -c "$container_script"

rc=$?

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "=== run finished (exit code=$rc) ==="
date -u
if [[ "$rc" -eq 0 ]]; then
  echo
  echo "Result CSVs:"
  [[ "$SKIP_BF16" == 0 ]] && wc -l "$OUT_BF16/all_runs.csv" 2>/dev/null || true
  [[ "$SKIP_FP8"  == 0 ]] && wc -l "$OUT_FP8/all_runs.csv"  2>/dev/null || true
  echo
  echo "See REPRODUCE_PRODSHAPE.md §6 for expected numbers."
fi

exit "$rc"
