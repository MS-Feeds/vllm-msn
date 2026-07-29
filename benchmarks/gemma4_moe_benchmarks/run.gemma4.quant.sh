#!/bin/bash
set -xe

# --- 1. Positional Arguments (with defaults) ---
PORT=${1:-8100}
TP_SIZE=${2:-1}
MAX_LEN=${3:-32768}
GPU_UTIL=${4:-0.95}
DTYPE=${5:-"auto"}
SERVED_NAME=${6:-"gemma4"} # <-- Added Default Service Name

# --- 2. Environment Setup ---
if [[ -z "${_ModelDataPath_}" ]]; then
  echo "Assuming local Azure ML environment"
  model_dir="$(pwd)/INPUT_model_dir"
else
  echo "Using _ModelDataPath_: ${_ModelDataPath_}"
  model_dir="${_ModelDataPath_}/model"
fi

# --- 3. Model Path Resolution ---
model="google/gemma-4-26B-A4B-it"
[[ -d "$model_dir" ]] && model="$model_dir"

# --- 4. Dynamic Configuration ---
export TORCH_CUDA_ARCH_LIST=$(python3 -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability(0))))")

chat_template_arg=""
[[ -f "$model_dir/template.jinja" ]] && chat_template_arg="--chat-template $model_dir/template.jinja"

# --- 5. Execute vLLM ---
echo "Starting server on port $PORT..."
echo "Serving Model As: $SERVED_NAME"

vllm serve "$model" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype "$DTYPE" \
  --quantization fp8 \
  --kv-cache-dtype auto \
  --trust-remote-code \
  --async-scheduling \
  --no-enable-log-requests \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"method": "ngram_gpu", "num_speculative_tokens": 5}' \
  --max-num-seqs 64 \
  $chat_template_arg