#!/bin/bash
# Gemma 4 26B MoE with FP8 Quantization on A100 40GB
# Optimized for memory-constrained environments

VLLM_CMD=llm_analyzer_vllm_oaas_async_relay.py

# Set logging levels
export PYTHONWARNINGS="ignore"
export LOGURU_LEVEL="ERROR"
export VLLM_LOG_LEVEL="ERROR"
export PYTHONUNBUFFERED=1

# A100 40GB Configuration
export TORCH_CUDA_ARCH_LIST="8.0"

# Attention Backend: FLASH_ATTN is optimal for MoE models on A100
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# FP8 Quantization for MoE - Enable FlashInfer MoE FP8 support
export VLLM_USE_FLASHINFER_MOE_FP8=1

# MoE Backend Selection (auto will choose best for FP8)
# Options: auto, triton, deep_gemm (for FP8), flashinfer_cutlass
export VLLM_MOE_BACKEND=auto

# Model Configuration - Gemma 4 26B MoE (A4B variant)
MODEL_PATH=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it

# A100 40GB Memory Optimization:
# - Use FP8 quantization for weights and KV cache
# - Reduce GPU memory utilization to 0.85 (leave headroom for FP8 ops)
# - Smaller batch sizes due to memory constraints

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/nvmedata/chenw/vllm-ra python3 $VLLM_CMD \
    --input_path /nvmedata/chenw/genz/genz_users_20k_format.tsv \
    --output_path /nvmedata/chenw/genz/genz_users_interests_gemma4_fp8.jsonl \
    --model_path $MODEL_PATH \
    --batch_size 128 \
    --enable_relay_attention \
    --max_model_len 8192 \
    --quantization fp8 \
    --kv_cache_dtype fp8_e5m2 \
    --gpu_memory_utilization 0.85 \
    --max_num_batched_tokens 8192 \
    --max_num_seqs 256
