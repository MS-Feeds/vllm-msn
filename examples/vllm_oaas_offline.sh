VLLM_CMD=llm_analyzer_vllm_oaas_async_relay.py
# VLLM_CMD=llm_analyzer_vllm_oaas_async.py

# Set logging levels
export PYTHONWARNINGS="ignore"
export LOGURU_LEVEL="ERROR"
export VLLM_LOG_LEVEL="ERROR"
export PYTHONUNBUFFERED=1
# run these in the same shell that starts vLLM
export TORCH_CUDA_ARCH_LIST="8.0"      # A100 architecture
# A100 + MoE Optimization: Use FLASH_ATTN (best for Gemma 4 MoE)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# Note: FLASHINFER is better for non-MoE models on A100
# For MoE models like Gemma 4, FLASH_ATTN is more stable and performant
# Or leave unset for autodetect (will choose FLASH_ATTN on A100)


# MODEL_PATH=/nvmedata/hf_checkpoints/Qwen3-8B/
MODEL_PATH=/nvmedata/hf_checkpoints/Llama-2-7b-chat-hf-bf16

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/nvmedata/chenw/vllm-ra python3 $VLLM_CMD \
    --input_path /nvmedata/chenw/genz/genz_users_20k_format.tsv \
    --output_path /nvmedata/chenw/genz/genz_users_interests_vllm_oaas_async.jsonl \
    --model_path $MODEL_PATH \
    --batch_size 256 \
    --enable_relay_attention