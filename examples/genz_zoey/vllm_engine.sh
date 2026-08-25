MODEL_PATH=/nvmedata/hf_checkpoints/Qwen3-32B/
# MODEL_PATH=/nvmedata/hf_checkpoints/Qwen3-8B/

# Suppress INFO level logging
export PYTHONWARNINGS="ignore"
export LOGURU_LEVEL="ERROR"
export VLLM_LOG_LEVEL="ERROR"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --block-size 32 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 512 \
    --swap-space 8 \
    --disable-log-requests \
    # --enable-relay-attention
