#!/bin/bash
set -xe

# --- 1. Positional Arguments ---
PORT=${1:-8100}
TP_SIZE=${2:-8}
MAX_LEN=${3:-32768}
GPU_UTIL=${4:-0.90}
SERVED_NAME=${5:-"gemma4"}
# FIXED: Reverted back to proper JSON string format!
MM_LIMITS=${6:-'{"image": 4, "video": 1}'}

# --- 2. Environment Setup ---
if [[ -z "${_ModelDataPath_}" ]]; then
  echo "Assuming local Azure ML environment"
  model_dir="$(pwd)/INPUT_model_dir"
else
  echo "Using _ModelDataPath_: ${_ModelDataPath_}"
  model_dir="${_ModelDataPath_}/model"
fi

# --- 3. Model Path Resolution ---
model="google/gemma-4-31B-it"
[[ -d "$model_dir" ]] && model="$model_dir"

# --- 4. Fetch the Official Chat Template ---
TEMPLATE_PATH="/workspace/tool_chat_template_gemma4.jinja"
if [ ! -f "$TEMPLATE_PATH" ]; then
    echo "Downloading Gemma 4 tool chat template from vLLM repository..."
    curl -sSLf "https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_gemma4.jinja" -o "$TEMPLATE_PATH"
fi

# --- 5. Execute vLLM ---
echo "Starting vLLM server on port $PORT..."

vllm serve "$model" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --chat-template "$TEMPLATE_PATH" \
  --limit-mm-per-prompt "$MM_LIMITS" \
  --async-scheduling