#!/usr/bin/env bash
# DeepSeek V4 Flash Server - Fixed with PYTHONPATH

set -euo pipefail

cd "$(dirname "$0")"

echo "=========================================="
echo "DeepSeek V4 Flash Server"
echo "=========================================="

# Configure environment
export PYTHONPATH="/nvmedata/chenw/vllm-ra:${PYTHONPATH:-}"
export HF_HOME="/nvmedata/hf_checkpoints"
export VLLM_USE_DEEP_GEMM=0
export VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE=256
export INDEXER_LOGIT_DUMP_DIR="./indexer_logits/deepseek_v4_flash_$(date +%Y%m%d_%H%M%S)"
export INDEXER_IS_EXPERT_ROUTING=1
export INDEXER_LOGIT_RANGE="0.0,1.0"

mkdir -p "$INDEXER_LOGIT_DUMP_DIR"
mkdir -p logs

echo "PYTHONPATH: $PYTHONPATH"
echo "Logit output: $INDEXER_LOGIT_DUMP_DIR"
echo "Model: deepseek-ai/DeepSeek-V4-Flash"
echo "Port: 8000"
echo ""

# Launch
/root/miniconda3/envs/vllm-ablation/bin/python -m vllm.entrypoints.openai.api_server \
    --model /nvmedata/hf_checkpoints/DeepSeek-V4-Flash \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --enforce-eager \
    --attention-backend TRITON_MLA_SPARSE \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name DeepSeek-V4-Flash \
    2>&1 | tee "logs/server_v4_flash_$(date +%Y%m%d_%H%M%S).log"
