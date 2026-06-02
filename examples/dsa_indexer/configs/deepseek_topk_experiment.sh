#!/usr/bin/env bash
# DeepSeek Top-K Distribution Experiment
# 3 benchmarks × 100 prompts = 300 total
#
# Purpose: Capture post-softmax expert routing scores across math, code, and QA
#          benchmarks to analyze threshold-based pruning opportunities.
#
# Usage:
#   bash configs/deepseek_topk_experiment.sh [--dry-run]

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Model configuration
MODEL="${DEEPSEEK_MODEL:-deepseek-ai/DeepSeek-V3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"
GPU_MEM="${GPU_MEM:-0.90}"
TP_SIZE="${TP_SIZE:-4}"  # Tensor parallel size (adjust based on GPU count)

# Benchmark configuration
PROMPTS_PER_DATASET="${PROMPTS_PER_DATASET:-100}"
CONCURRENCY="${CONCURRENCY:-4}"  # In-flight requests for continuous batching
SEED="${SEED:-42}"

# Output configuration
RUN_ID="deepseek_v3_$(date +%Y%m%d_%H%M%S)"
LOGIT_DUMP_DIR="$PROJECT_ROOT/indexer_logits/$RUN_ID"
EVAL_LOG="$PROJECT_ROOT/logs/eval_${RUN_ID}.jsonl"
SERVER_LOG="$PROJECT_ROOT/logs/server_${RUN_ID}.log"

# Indexer logger configuration
export INDEXER_LOGIT_DUMP_DIR="$LOGIT_DUMP_DIR"
export INDEXER_IS_EXPERT_ROUTING=1
export INDEXER_LOGIT_RANGE="0.0,1.0"           # Post-softmax probabilities
export INDEXER_LOGIT_BINS=4096
export INDEXER_RADIX_BITS_SWEEP="4,6,8,12,16"
export INDEXER_RADIX_BKT_BINS=256
export INDEXER_LOGIT_SAMPLE_K=0                # Disable reservoir (not needed)
export INDEXER_LOGIT_DUMP_EVERY=0              # Dump only at exit

# Server configuration
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"

# =============================================================================
# Argument Parsing
# =============================================================================

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "DRY RUN MODE: Commands will be printed but not executed"
    echo ""
fi

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "=========================================="
echo "DeepSeek Top-K Experiment Configuration"
echo "=========================================="
echo "Model:              $MODEL"
echo "Max model length:   $MAX_MODEL_LEN"
echo "Max tokens/prompt:  $MAX_TOKENS"
echo "Temperature:        $TEMPERATURE"
echo "GPU memory util:    $GPU_MEM"
echo "Tensor parallel:    $TP_SIZE"
echo ""
echo "Prompts per dataset: $PROMPTS_PER_DATASET"
echo "Total prompts:       $((PROMPTS_PER_DATASET * 3))"
echo "Concurrency:         $CONCURRENCY"
echo "Random seed:         $SEED"
echo ""
echo "Run ID:             $RUN_ID"
echo "Logit dump dir:     $LOGIT_DUMP_DIR"
echo "Eval log:           $EVAL_LOG"
echo "Server log:         $SERVER_LOG"
echo "=========================================="
echo ""

# Check required tools
for cmd in python3 curl jq; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "ERROR: Required command not found: $cmd"
        exit 1
    fi
done

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU check skipped."
else
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "GPUs available: $GPU_COUNT"
    if [[ $GPU_COUNT -lt $TP_SIZE ]]; then
        echo "WARNING: TP_SIZE=$TP_SIZE but only $GPU_COUNT GPUs available"
        echo "Set TP_SIZE environment variable to match GPU count."
        if [[ $DRY_RUN -eq 0 ]]; then
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
fi

# Create output directories
mkdir -p "$(dirname "$EVAL_LOG")"
mkdir -p "$(dirname "$SERVER_LOG")"
mkdir -p "$LOGIT_DUMP_DIR"

echo "✓ Pre-flight checks complete"
echo ""

# =============================================================================
# Phase 1: Launch vLLM Server
# =============================================================================

echo "=========================================="
echo "Phase 1: Launching vLLM Server"
echo "=========================================="

LAUNCH_CMD="python3 -m vllm.entrypoints.openai.api_server \
    --model '$MODEL' \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM \
    --tensor-parallel-size $TP_SIZE \
    --trust-remote-code \
    --host $VLLM_HOST \
    --port $VLLM_PORT \
    --served-model-name '${MODEL##*/}' \
    2>&1 | tee '$SERVER_LOG'"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Would execute:"
    echo "$LAUNCH_CMD &"
    SERVER_PID="(dry-run)"
else
    echo "Launching server..."
    eval "$LAUNCH_CMD" &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    # Save PID for cleanup
    echo "$SERVER_PID" > "$PROJECT_ROOT/logs/server_${RUN_ID}.pid"

    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    MAX_WAIT=300  # 5 minutes
    ELAPSED=0
    WAIT_INTERVAL=5

    while [[ $ELAPSED -lt $MAX_WAIT ]]; do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo "✓ Server is ready (took ${ELAPSED}s)"
            break
        fi
        sleep $WAIT_INTERVAL
        ELAPSED=$((ELAPSED + WAIT_INTERVAL))
        echo "  ... still waiting (${ELAPSED}s / ${MAX_WAIT}s)"
    done

    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
        echo "ERROR: Server failed to start within ${MAX_WAIT}s"
        echo "Check server log: $SERVER_LOG"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
fi

echo ""

# =============================================================================
# Phase 2: Run Evaluation
# =============================================================================

echo "=========================================="
echo "Phase 2: Running Evaluation"
echo "=========================================="

EVAL_CMD="python3 '$PROJECT_ROOT/tools/run_eval_prompts.py' \
    --url 'http://localhost:$VLLM_PORT' \
    --model '${MODEL##*/}' \
    --datasets aime livecodebench gpqa_diamond \
    --n-per-dataset $PROMPTS_PER_DATASET \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --concurrency $CONCURRENCY \
    --seed $SEED \
    --out '$EVAL_LOG'"

echo "Datasets:"
echo "  - AIME (math):         $PROMPTS_PER_DATASET prompts"
echo "  - LiveCodeBench (code): $PROMPTS_PER_DATASET prompts"
echo "  - GPQA Diamond (QA):    $PROMPTS_PER_DATASET prompts"
echo "Total:                     $((PROMPTS_PER_DATASET * 3)) prompts"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Would execute:"
    echo "$EVAL_CMD"
else
    echo "Starting evaluation..."
    START_TIME=$(date +%s)

    if eval "$EVAL_CMD"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo ""
        echo "✓ Evaluation complete (took ${DURATION}s = $((DURATION / 60))m)"

        # Quick stats
        if [[ -f "$EVAL_LOG" ]]; then
            TOTAL=$(wc -l < "$EVAL_LOG")
            SUCCESS=$(grep -c '"ok":true' "$EVAL_LOG" || echo "0")
            FAILED=$((TOTAL - SUCCESS))
            echo "  Total:   $TOTAL prompts"
            echo "  Success: $SUCCESS"
            echo "  Failed:  $FAILED"
        fi
    else
        echo "ERROR: Evaluation failed"
        echo "Check eval log: $EVAL_LOG"
        if [[ -n "${SERVER_PID:-}" ]] && [[ "$SERVER_PID" != "(dry-run)" ]]; then
            kill $SERVER_PID 2>/dev/null || true
        fi
        exit 1
    fi
fi

echo ""

# =============================================================================
# Phase 3: Shutdown Server
# =============================================================================

echo "=========================================="
echo "Phase 3: Shutting Down Server"
echo "=========================================="

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Would kill server PID: $SERVER_PID"
else
    if [[ -n "${SERVER_PID:-}" ]] && [[ "$SERVER_PID" != "(dry-run)" ]]; then
        echo "Shutting down server (PID $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true

        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 $SERVER_PID 2>/dev/null; then
                echo "✓ Server stopped"
                break
            fi
            sleep 1
        done

        # Force kill if still running
        if kill -0 $SERVER_PID 2>/dev/null; then
            echo "Forcing server shutdown..."
            kill -9 $SERVER_PID 2>/dev/null || true
        fi
    fi
fi

echo ""

# =============================================================================
# Phase 4: Verify Output
# =============================================================================

echo "=========================================="
echo "Phase 4: Verifying Output"
echo "=========================================="

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Would verify:"
    echo "  - Logit captures in: $LOGIT_DUMP_DIR"
    echo "  - Eval results in:   $EVAL_LOG"
else
    # Check logit captures
    NPZ_FILES=("$LOGIT_DUMP_DIR"/*.npz)
    if [[ -e "${NPZ_FILES[0]}" ]]; then
        NPZ_COUNT=${#NPZ_FILES[@]}
        echo "✓ Logit captures: $NPZ_COUNT file(s)"
        for f in "${NPZ_FILES[@]}"; do
            echo "    $(basename "$f")"
        done

        # Quick check: load and print layer info
        echo ""
        echo "Sample capture structure:"
        python3 -c "
import numpy as np
d = np.load('${NPZ_FILES[0]}')
print(f'  Layers captured: {d[\"layer_ids\"]}')
print(f'  Prefill rows:    {d[\"count_prefill\"].sum():,}')
print(f'  Decode rows:     {d[\"count_decode\"].sum():,}')
print(f'  Histogram bins:  {d[\"bins\"]}')
print(f'  Range:           [{d[\"range_lo\"]}, {d[\"range_hi\"]}]')
" 2>/dev/null || echo "  (numpy check failed - install numpy to see details)"
    else
        echo "⚠ WARNING: No logit capture files found in $LOGIT_DUMP_DIR"
        echo "  Check that INDEXER_LOGIT_DUMP_DIR was set correctly"
        echo "  Check server log for errors: $SERVER_LOG"
    fi

    echo ""

    # Check eval log
    if [[ -f "$EVAL_LOG" ]]; then
        EVAL_LINES=$(wc -l < "$EVAL_LOG")
        echo "✓ Eval log: $EVAL_LINES lines"
        echo "    $EVAL_LOG"
    else
        echo "⚠ WARNING: Eval log not found: $EVAL_LOG"
    fi
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  Logit captures:  $LOGIT_DUMP_DIR/"
echo "  Eval results:    $EVAL_LOG"
echo "  Server log:      $SERVER_LOG"
echo ""
echo "Next steps:"
echo "  1. Analyze logit distributions:"
echo "     python3 tools/plot_indexer_logits.py \\"
echo "       --inputs '$LOGIT_DUMP_DIR'/*.npz \\"
echo "       --out figures/${RUN_ID} \\"
echo "       --aggregate"
echo ""
echo "  2. Generate threshold recommendations:"
echo "     python3 tools/threshold_analysis.py '$LOGIT_DUMP_DIR'/indexer_logits_rank0.npz \\"
echo "       > figures/${RUN_ID}/threshold_recommendations.txt"
echo ""
echo "  3. Review results:"
echo "     cat figures/${RUN_ID}/summary.txt"
echo ""
echo "=========================================="
