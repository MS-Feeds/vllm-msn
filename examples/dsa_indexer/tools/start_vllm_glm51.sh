#!/usr/bin/env bash
set -euo pipefail

# Fixed environment requested by user.
ENV_ROOT="/mnt/remote/guangtaow/conda_env/vllm_glm5_py312"
VLLM_BIN="$ENV_ROOT/bin/vllm"
MODEL_DIR="/mnt/remote/checkpoints/GLM-5.1"

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "ERROR: vLLM binary not found at: $VLLM_BIN" >&2
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model directory not found at: $MODEL_DIR" >&2
  exit 1
fi

PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
SERVED_NAME="${SERVED_NAME:-glm-5.1}"

# ---------------------------------------------------------------------------
# Indexer logit recorder (§5 of INDEXER_LOGIT_EXPERIMENT.md)
#   Set INDEXER_LOGIT_DUMP_DIR to enable recording; leave unset to disable.
#   Key insight from run02 (§13): INDEXER_GAP_RANGE_HI=0.04 pinned 59/78
#   prefill layers at the histogram floor. Use 0.001 so the true p1 resolves.
# ---------------------------------------------------------------------------
# INDEXER_LOGIT_DUMP_DIR is intentionally unset here; override from caller:
#   export INDEXER_LOGIT_DUMP_DIR=/mnt/remote/guangtaow/indexer_logits/run03
export INDEXER_GAP_RANGE_HI="${INDEXER_GAP_RANGE_HI:-0.001}"   # was 0.04 in run02
export INDEXER_LOGIT_BINS="${INDEXER_LOGIT_BINS:-4096}"
export INDEXER_GAP_BINS="${INDEXER_GAP_BINS:-4096}"
export INDEXER_GAP_WINDOW="${INDEXER_GAP_WINDOW:-8}"
export INDEXER_GAP_EPS_NORM="${INDEXER_GAP_EPS_NORM:-1e-6}"
export INDEXER_LOGIT_DUMP_EVERY="${INDEXER_LOGIT_DUMP_EVERY:-200}"  # periodic flush
export INDEXER_LOGIT_SAMPLE_K="${INDEXER_LOGIT_SAMPLE_K:-0}"
# Stream D — radix boundary-bucket sweep (§14 of INDEXER_LOGIT_EXPERIMENT.md)
# d values to probe; boundary-bucket size histogram tells how many radix passes needed.
export INDEXER_RADIX_BITS_SWEEP="${INDEXER_RADIX_BITS_SWEEP:-4,8,12,16}"   # dropped d=6 for throughput; covers the curve well enough
export INDEXER_RADIX_BKT_BINS="${INDEXER_RADIX_BKT_BINS:-4096}"   # was 256; needed >=2049 to resolve fits-in-kNumFinalItems=2048
# Stream G — radix sweep with CUSTOM bit windows (best-d-by-entropy, from run03 Stream E).
# Each entry is start_bit:width. Defaults are best-4/-8/-12 as contiguous mantissa slices.
export INDEXER_RADIX_BEST_BIT_WINDOWS="${INDEXER_RADIX_BEST_BIT_WINDOWS:-9:4,5:8,1:12}"
if [[ -n "${INDEXER_LOGIT_DUMP_DIR:-}" ]]; then
  mkdir -p "$INDEXER_LOGIT_DUMP_DIR"
  export INDEXER_LOGIT_DUMP_DIR
  echo "[indexer] Recording enabled → $INDEXER_LOGIT_DUMP_DIR" >&2
else
  echo "[indexer] Recording disabled (INDEXER_LOGIT_DUMP_DIR not set)" >&2
fi

# Default to all 8 GPUs.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

# If unset, all visible GPUs are used; set CUDA_VISIBLE_DEVICES before calling this script if needed.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
# Stabilize MoE init by avoiding a separate shared-experts CUDA stream.
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"

# Ensure cuBLAS and cuBLASLt from the correct CUDA 13 package are loaded BEFORE
# the system CUDA 13 path. The system /usr/local/cuda-13/lib64 sits first in
# LD_LIBRARY_PATH by default, but the Python package ships its own cublasLt build
# that must match its cublas. Prepending the package path fixes the ABI mismatch.
_NVIDIA_LIB="$ENV_ROOT/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "$_NVIDIA_LIB" ]]; then
  export LD_LIBRARY_PATH="$_NVIDIA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# DeepGEMM JIT-compiles FP8 MQA kernels at first use and requires nvcc >= 12.3.
# The system PATH may pick up an older /usr/bin/nvcc (CUDA 12.0). Prepend the
# CUDA 13 bin dir so `nvcc --version` returns >= 12.3.
if [[ -d /usr/local/cuda-13/bin ]]; then
  export PATH="/usr/local/cuda-13/bin:$PATH"
fi

exec "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --quantization fp8 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enforce-eager \
  --trust-remote-code \
  --port "$PORT"
