#!/bin/bash
set -euo pipefail

exec vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --moe-backend marlin \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --tokenizer-mode deepseek_v4 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --speculative-config \
    '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
    "$@"
