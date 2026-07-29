#!/bin/bash
set -xe

# --- 1. Positional Arguments ---
PORT=${1:-8100}
TP_SIZE=${2:-8}
MAX_LEN=${3:-262144}
GPU_UTIL=${4:-0.90}
SERVED_NAME=${5:-"qwen3.6"}

# --- 2. Environment Setup ---
if [[ -z "${_ModelDataPath_}" ]]; then
  echo "Assuming local Azure ML environment"
  model_dir="$(pwd)/INPUT_model_dir"
else
  echo "Using _ModelDataPath_: ${_ModelDataPath_}"
  model_dir="${_ModelDataPath_}/model"
fi

# --- 3. Model Path Resolution ---
model="Qwen/Qwen3.6-35B-A3B"
[[ -d "$model_dir" ]] && model="$model_dir"

# --- 4. Execute vLLM ---
echo "Starting vLLM server on port $PORT..."

vllm serve "$model" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --async-scheduling