#!/usr/bin/env bash
# Quick Launch: DeepSeek V4 Flash Server
#
# Usage: bash launch_v4_server.sh

set -euo pipefail

cd "$(dirname "$0")"

echo "=========================================="
echo "DeepSeek V4 Flash Server Launch"
echo "=========================================="
echo ""

# Step 1: Activate environment
echo "Step 1: Activating vllm-ablation environment..."
source /root/miniconda3/bin/activate vllm-ablation

# Step 2: Deploy patches
echo "Step 2: Deploying V4 patches..."
VLLM_INSTALL_PATH="/root/miniconda3/envs/vllm-ablation/lib/python3.10/site-packages/vllm" \
    bash sync_to_env.sh

# Step 3: Verify installation
echo ""
echo "Step 3: Verifying installation..."
python -c "from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM; print('✓ DeepSeek V4 imports OK')"

# Step 4: Configure indexer
echo ""
echo "Step 4: Configuring indexer..."
export INDEXER_LOGIT_DUMP_DIR="./indexer_logits/deepseek_v4_flash_$(date +%Y%m%d_%H%M%S)"
export INDEXER_IS_EXPERT_ROUTING=1
export INDEXER_LOGIT_RANGE="0.0,1.0"
export INDEXER_LOGIT_BINS=4096

mkdir -p "$INDEXER_LOGIT_DUMP_DIR"
echo "Logit dump: $INDEXER_LOGIT_DUMP_DIR"

# Step 5: Launch server
echo ""
echo "Step 5: Launching vLLM server..."
echo "Model: deepseek-ai/DeepSeek-V4-Flash"
echo "Port: 8000"
echo "TP size: 4"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name DeepSeek-V4-Flash \
    2>&1 | tee "logs/server_v4_flash_$(date +%Y%m%d_%H%M%S).log"
