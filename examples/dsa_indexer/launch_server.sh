#!/usr/bin/env bash
# Quick vLLM Server Launch Script for DeepSeek Experiments
#
# Usage:
#   bash launch_server.sh [model_name] [port]
#
# Examples:
#   bash launch_server.sh deepseek-ai/DeepSeek-V3 8000
#   bash launch_server.sh deepseek-ai/DeepSeek-V2 8001

set -euo pipefail

# Configuration
MODEL="${1:-deepseek-ai/DeepSeek-V3}"
PORT="${2:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM="${GPU_MEM:-0.90}"
TP_SIZE="${TP_SIZE:-4}"

# Indexer configuration
export INDEXER_LOGIT_DUMP_DIR="${INDEXER_LOGIT_DUMP_DIR:-./indexer_logits/test_$(date +%Y%m%d_%H%M%S)}"
export INDEXER_IS_EXPERT_ROUTING=1
export INDEXER_LOGIT_RANGE="0.0,1.0"
export INDEXER_LOGIT_BINS=4096
export INDEXER_RADIX_BITS_SWEEP="4,6,8,12,16"

echo "=========================================="
echo "vLLM Server Launch"
echo "=========================================="
echo "Model:              $MODEL"
echo "Port:               $PORT"
echo "Max model length:   $MAX_MODEL_LEN"
echo "GPU memory util:    $GPU_MEM"
echo "Tensor parallel:    $TP_SIZE"
echo ""
echo "Indexer logging:    $INDEXER_LOGIT_DUMP_DIR"
echo "=========================================="
echo ""

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "ERROR: Port $PORT is already in use"
    echo "Use a different port or kill the existing process:"
    echo "  lsof -ti:$PORT | xargs kill -9"
    exit 1
fi

# Create log directory
mkdir -p logs
mkdir -p "$INDEXER_LOGIT_DUMP_DIR"

LOG_FILE="logs/server_$(basename $MODEL)_${PORT}_$(date +%Y%m%d_%H%M%S).log"

echo "Launching server... (log: $LOG_FILE)"
echo ""

# Launch server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM \
    --tensor-parallel-size $TP_SIZE \
    --trust-remote-code \
    --host $HOST \
    --port $PORT \
    --served-model-name "$(basename $MODEL)" \
    2>&1 | tee "$LOG_FILE"
