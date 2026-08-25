#!/bin/bash
# Gemma 4 26B MoE with FP8 Quantization + MTP Assistant on A100 40GB
# Multi-Token Prediction for 2-3x speedup

VLLM_CMD=llm_analyzer_gemma4_moe_fp8_mtp.py

# Set logging levels
export PYTHONWARNINGS="ignore"
export LOGURU_LEVEL="ERROR"
export VLLM_LOG_LEVEL="ERROR"
export PYTHONUNBUFFERED=1

# A100 40GB Configuration
export TORCH_CUDA_ARCH_LIST="8.0"

# Attention Backend: FLASH_ATTN is optimal for MoE models on A100
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# FP8 Quantization for MoE
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_MOE_BACKEND=auto

# Model Configuration
MODEL_PATH=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it
ASSISTANT_PATH=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant

# Speculative Decoding with MTP
# The assistant model predicts 4-6 tokens ahead for faster inference
NUM_SPECULATIVE_TOKENS=5  # Typical range: 3-7

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/nvmedata/chenw/vllm-ra python3 $VLLM_CMD \
    --input_path /nvmedata/chenw/genz/genz_users_20k_format.tsv \
    --output_path /nvmedata/chenw/genz/genz_users_interests_gemma4_fp8_mtp.jsonl \
    --model_path $MODEL_PATH \
    --speculative_model $ASSISTANT_PATH \
    --num_speculative_tokens $NUM_SPECULATIVE_TOKENS \
    --batch_size 128

# Expected Performance with MTP + CUDA Graphs:
# - Throughput: 70-100 req/sec (MTP: 1.5-2x, CUDA graphs: +10-15%)
# - Latency: 35-70ms per request (optimized for online inference)
# - Memory: ~34-38GB (model + assistant + CUDA graphs overhead)
# - Memory breakdown:
#   * Main model (FP8): ~20-22GB
#   * Assistant model: ~0.8GB
#   * KV cache (FP8): ~6-8GB (reduced with gpu_memory_utilization=0.75)
#   * CUDA graphs: ~4-6GB (graphs for batch sizes 1,2,4,8,16,32,64,128)
#   * CUDA overhead: ~2-3GB
# - Speedup: 1.5-2x (MTP) + 10-15% (CUDA graphs) vs standard decoding
#
# Note: Reduced gpu_memory_utilization to 0.75 and max_num_seqs to 128
#       to accommodate CUDA graph memory overhead for online inference
