#!/bin/bash
# Automated Experiment Runner for Gemma 4 MoE
# Runs experiments with different configurations and logs results

set -e

# Configuration
EXPERIMENT_ID="${1:-E001}"
BACKEND="${2:-FLASHINFER}"  # FLASHINFER, FLASH_ATTN, or AUTO
BATCH_SIZE="${3:-128}"
OUTPUT_DIR="./experiments/${EXPERIMENT_ID}"

echo "=========================================="
echo "Gemma 4 MoE Experiment Runner"
echo "=========================================="
echo ""
echo "Experiment ID: ${EXPERIMENT_ID}"
echo "Backend: ${BACKEND}"
echo "Batch Size: ${BATCH_SIZE}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Activate environment
source /root/miniconda3/bin/activate vllm

# Log environment info
echo "Step 1: Logging environment information..."
{
    echo "# Experiment ${EXPERIMENT_ID} - Environment Info"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "## Hardware"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
    echo ""
    echo "## Software"
    python --version
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM: dev"
    python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" 2>/dev/null || echo "Flash Attention: Not installed"
    python -c "import flashinfer; print(f'FlashInfer: {flashinfer.__version__}')" 2>/dev/null || echo "FlashInfer: Not installed"
    echo ""
    echo "## Git Commit"
    git rev-parse HEAD
    git log -1 --oneline
} > "${OUTPUT_DIR}/environment.txt"

echo "✓ Environment info saved to ${OUTPUT_DIR}/environment.txt"
echo ""

# Set backend
echo "Step 2: Configuring attention backend..."
case "${BACKEND}" in
    FLASHINFER)
        export VLLM_ATTENTION_BACKEND=FLASHINFER
        export VLLM_USE_FLASHINFER_MOE_FP8=1
        echo "✓ Using FlashInfer backend"
        ;;
    FLASH_ATTN)
        export VLLM_ATTENTION_BACKEND=FLASH_ATTN
        unset VLLM_USE_FLASHINFER_MOE_FP8
        echo "✓ Using Flash Attention 2 backend"
        ;;
    AUTO)
        unset VLLM_ATTENTION_BACKEND
        unset VLLM_USE_FLASHINFER_MOE_FP8
        echo "✓ Using auto-detected backend"
        ;;
    *)
        echo "✗ Unknown backend: ${BACKEND}"
        exit 1
        ;;
esac
echo ""

# Start memory monitoring in background
echo "Step 3: Starting memory monitor..."
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv -l 1 > "${OUTPUT_DIR}/memory_trace.csv" &
MONITOR_PID=$!
echo "✓ Memory monitor started (PID: ${MONITOR_PID})"
echo ""

# Record initial state
echo "Step 4: Recording initial GPU state..."
nvidia-smi > "${OUTPUT_DIR}/gpu_initial.txt"
echo "✓ Initial state saved"
echo ""

# Run experiment
echo "Step 5: Running inference experiment..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/nvmedata/chenw/vllm-ra \
time python3 llm_analyzer_gemma4_moe_fp8_mtp.py \
    --input_path /nvmedata/chenw/genz/genz_users_20k_format.tsv \
    --output_path "${OUTPUT_DIR}/output.jsonl" \
    --model_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
    --speculative_model /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant \
    --num_speculative_tokens 5 \
    --batch_size ${BATCH_SIZE} \
    2>&1 | tee "${OUTPUT_DIR}/inference.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Duration: ${DURATION} seconds ($((DURATION / 60)) minutes)"
echo ""

# Stop memory monitoring
kill ${MONITOR_PID} 2>/dev/null || true
echo "✓ Memory monitor stopped"
echo ""

# Record final state
echo "Step 6: Recording final GPU state..."
nvidia-smi > "${OUTPUT_DIR}/gpu_final.txt"
echo "✓ Final state saved"
echo ""

# Analyze results
echo "Step 7: Analyzing results..."
{
    echo "# Experiment ${EXPERIMENT_ID} - Results Summary"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Backend: ${BACKEND}"
    echo "Batch Size: ${BATCH_SIZE}"
    echo ""

    echo "## Performance"
    echo "Duration: ${DURATION} seconds ($((DURATION / 60)) minutes)"

    if [ -f "${OUTPUT_DIR}/output.jsonl" ]; then
        SAMPLES=$(wc -l < "${OUTPUT_DIR}/output.jsonl")
        THROUGHPUT=$(echo "scale=2; ${SAMPLES} / ${DURATION}" | bc)
        echo "Samples processed: ${SAMPLES}"
        echo "Throughput: ${THROUGHPUT} samples/sec"
    else
        echo "Output file not found"
    fi

    echo ""
    echo "## Memory"
    if [ -f "${OUTPUT_DIR}/memory_trace.csv" ]; then
        # Skip header and get max memory
        MAX_MEM=$(tail -n +2 "${OUTPUT_DIR}/memory_trace.csv" | cut -d',' -f2 | sort -n | tail -1)
        AVG_MEM=$(tail -n +2 "${OUTPUT_DIR}/memory_trace.csv" | cut -d',' -f2 | awk '{sum+=$1; count++} END {printf "%.0f", sum/count}')
        echo "Peak memory: ${MAX_MEM} MB"
        echo "Average memory: ${AVG_MEM} MB"
    fi

    echo ""
    echo "## Files"
    echo "- Environment: ${OUTPUT_DIR}/environment.txt"
    echo "- Inference log: ${OUTPUT_DIR}/inference.log"
    echo "- Memory trace: ${OUTPUT_DIR}/memory_trace.csv"
    echo "- Output: ${OUTPUT_DIR}/output.jsonl"
    echo "- GPU initial: ${OUTPUT_DIR}/gpu_initial.txt"
    echo "- GPU final: ${OUTPUT_DIR}/gpu_final.txt"

} > "${OUTPUT_DIR}/summary.txt"

cat "${OUTPUT_DIR}/summary.txt"
echo ""

# Check for errors
echo "Step 8: Checking for errors..."
ERROR_COUNT=$(grep -ic "error\|exception\|fail" "${OUTPUT_DIR}/inference.log" || true)
if [ ${ERROR_COUNT} -gt 0 ]; then
    echo "⚠ Found ${ERROR_COUNT} error/exception mentions in log"
    echo "Review: ${OUTPUT_DIR}/inference.log"
else
    echo "✓ No errors found"
fi
echo ""

echo "=========================================="
echo "Experiment ${EXPERIMENT_ID} Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Next steps:"
echo "1. Review summary: cat ${OUTPUT_DIR}/summary.txt"
echo "2. Check logs: cat ${OUTPUT_DIR}/inference.log"
echo "3. Analyze memory: cat ${OUTPUT_DIR}/memory_trace.csv"
echo "4. Document findings in EXPERIMENT_LOG.md"
echo ""
