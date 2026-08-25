#!/bin/bash
# Master script to run all ablation study experiments
# Based on EXPERIMENT_PLAN_ABLATION_STUDY.md

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Gemma 4 MoE - Ablation Study"
echo "=========================================="
echo ""
echo "This will run 15 experiments:"
echo "  GROUP A (E001-E007): Core optimizations (cumulative)"
echo "  GROUP B (E008-E011): Memory optimizations"
echo "  GROUP C (E012-E015): Alternative configurations"
echo ""
echo "Estimated time: 2-3 hours"
echo ""

# Confirm
read -p "Proceed with all experiments? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create results directory
mkdir -p experiment_results/ablation_study
RESULTS_DIR="experiment_results/ablation_study"

# Log file
LOG_FILE="${RESULTS_DIR}/ablation_study.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo ""
echo "Starting ablation study at $(date)"
echo "Results directory: ${RESULTS_DIR}"
echo ""

# Helper function to run experiment
run_experiment() {
    local exp_id=$1
    shift
    local exp_args=("$@")

    echo ""
    echo "========================================"
    echo -e "${BLUE}Running Experiment ${exp_id}${NC}"
    echo "========================================"
    echo ""

    # Run experiment
    if ./run_ablation_experiment.sh --exp "${exp_id}" "${exp_args[@]}"; then
        echo -e "${GREEN}✓ ${exp_id} completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ ${exp_id} failed${NC}"
        return 1
    fi
}

# Track success/failure
declare -a COMPLETED=()
declare -a FAILED=()

# Cooldown between experiments
cooldown() {
    echo ""
    echo "Cooldown (30 seconds)..."
    sleep 30
    echo ""
}

# =============================================================================
# GROUP A: Core Optimizations (Cumulative Build-up)
# =============================================================================

echo "========================================"
echo "GROUP A: Core Optimizations"
echo "========================================"
echo ""

# E001: Naive Baseline. Uses the same max-model-len / max-batched-tokens as
# every other experiment so the input distribution is identical (some prompts
# in the MAI dataset exceed 16K tokens — those will fail length validation
# uniformly across all experiments and show up in metrics.json's
# failed_errors). Without this, E001 would process a different subset of
# prompts than the FP8 experiments and the comparison would be invalid.
# Expected to OOM on BF16 26B; the plan already accounts for this.
echo "E001: Baseline (no optimizations, may OOM)"
if run_experiment E001 \
    --backend FLASH_ATTN \
    --batch 64 \
    --no-fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.95; then
    COMPLETED+=("E001")
else
    FAILED+=("E001")
    echo -e "${YELLOW}⚠ E001 failed (expected if OOM), continuing...${NC}"
fi
cooldown

# E002: Add FP8
echo "E002: Add FP8 Quantization"
if run_experiment E002 \
    --backend FLASH_ATTN \
    --batch 64 \
    --fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.85; then
    COMPLETED+=("E002")
else
    FAILED+=("E002")
    # E002 is the baseline that every later FP8 experiment builds on, so its
    # failure is a strong signal — but DO NOT exit the master script: we still
    # want partial results from later experiments to surface in the summary so
    # the user can diagnose what went wrong (and what still works).
    echo -e "${RED}✗ E002 failed — FP8 path broken. Continuing so downstream experiments still produce data; review E002 logs first.${NC}"
fi
cooldown

# E003: Switch to FlashInfer
echo "E003: Switch to FlashInfer Backend"
if run_experiment E003 \
    --backend FLASHINFER \
    --batch 64 \
    --fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.85; then
    COMPLETED+=("E003")
else
    FAILED+=("E003")
fi
cooldown

# E004: Increase batch size
echo "E004: Increase Batch Size to 128"
if run_experiment E004 \
    --backend FLASHINFER \
    --batch 128 \
    --fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.85; then
    COMPLETED+=("E004")
else
    FAILED+=("E004")
fi
cooldown

# E005: Enable CUDA graphs
echo "E005: Enable CUDA Graphs"
if run_experiment E005 \
    --backend FLASHINFER \
    --batch 128 \
    --fp8 \
    --cuda-graphs \
    --no-mtp \
    --gpu-mem 0.75; then
    COMPLETED+=("E005")
else
    FAILED+=("E005")
    echo -e "${YELLOW}⚠ E005 failed (CUDA graphs issue), continuing...${NC}"
fi
cooldown

# E006: Add MTP
echo "E006: Add MTP (Multi-Token Prediction)"
if run_experiment E006 \
    --backend FLASHINFER \
    --batch 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75; then
    COMPLETED+=("E006")
else
    FAILED+=("E006")
    echo -e "${YELLOW}⚠ E006 failed (MTP issue), continuing...${NC}"
fi
cooldown

# E007: Remove vision weights
echo "E007: Use Text-Only Model (vision removed)"
if run_experiment E007 \
    --backend FLASHINFER \
    --batch 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only; then
    COMPLETED+=("E007")
else
    FAILED+=("E007")
fi
cooldown

# =============================================================================
# GROUP B: Memory Optimizations
# =============================================================================

echo "========================================"
echo "GROUP B: Memory Optimizations"
echo "========================================"
echo ""

# E008: Test batch=192
echo "E008: Test Larger Batch (192)"
if run_experiment E008 \
    --backend FLASHINFER \
    --batch 192 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only; then
    COMPLETED+=("E008")
else
    FAILED+=("E008")
    echo -e "${YELLOW}⚠ E008 failed (likely OOM), continuing...${NC}"
fi
cooldown

# E009: Test batch=256
echo "E009: Test Maximum Batch (256, may OOM)"
if run_experiment E009 \
    --backend FLASHINFER \
    --batch 256 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only; then
    COMPLETED+=("E009")
else
    FAILED+=("E009")
    echo -e "${YELLOW}⚠ E009 failed (expected if OOM), continuing...${NC}"
fi
cooldown

# E010: Lower memory util
echo "E010: Lower Memory Utilization (0.70)"
if run_experiment E010 \
    --backend FLASHINFER \
    --batch 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.70 \
    --text-only; then
    COMPLETED+=("E010")
else
    FAILED+=("E010")
fi
cooldown

# E011: Higher memory util
echo "E011: Higher Memory Utilization (0.80)"
if run_experiment E011 \
    --backend FLASHINFER \
    --batch 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.80 \
    --text-only; then
    COMPLETED+=("E011")
else
    FAILED+=("E011")
    echo -e "${YELLOW}⚠ E011 failed (may be unstable), continuing...${NC}"
fi
cooldown

# =============================================================================
# GROUP C: Alternative Configurations
# =============================================================================

echo "========================================"
echo "GROUP C: Alternative Configurations"
echo "========================================"
echo ""

# E012: Flash Attention 2 comparison
echo "E012: Flash Attention 2 (for comparison)"
if run_experiment E012 \
    --backend FLASH_ATTN \
    --batch 128 \
    --fp8 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only; then
    COMPLETED+=("E012")
else
    FAILED+=("E012")
fi
cooldown

# E013: No MTP (isolate MTP contribution)
echo "E013: No MTP (measure MTP contribution)"
if run_experiment E013 \
    --backend FLASHINFER \
    --batch 128 \
    --fp8 \
    --cuda-graphs \
    --no-mtp \
    --gpu-mem 0.75 \
    --text-only; then
    COMPLETED+=("E013")
else
    FAILED+=("E013")
fi
cooldown

# E014: FP8_E4M3 KV cache
echo "E014: Test FP8_E4M3 KV Cache Format"
if run_experiment E014 \
    --backend FLASHINFER \
    --batch 128 \
    --fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --cuda-graphs \
    --mtp \
    --gpu-mem 0.75 \
    --text-only; then
    COMPLETED+=("E014")
else
    FAILED+=("E014")
fi
cooldown

# E015: Full BF16 reference baseline (text-only). Same input footprint as
# every other experiment so the comparison is fair. Like E001, may OOM.
echo "E015: Full BF16 Baseline (may OOM)"
if run_experiment E015 \
    --backend FLASHINFER \
    --batch 32 \
    --no-fp8 \
    --no-cuda-graphs \
    --no-mtp \
    --gpu-mem 0.95 \
    --text-only; then
    COMPLETED+=("E015")
else
    FAILED+=("E015")
    echo -e "${YELLOW}⚠ E015 failed (expected if OOM), continuing...${NC}"
fi

# =============================================================================
# Final Summary
# =============================================================================

echo ""
echo "========================================"
echo "Ablation Study Complete!"
echo "========================================"
echo ""
echo "Completed at: $(date)"
echo ""
echo "Results Summary:"
echo "  Completed: ${#COMPLETED[@]} experiments"
echo "  Failed: ${#FAILED[@]} experiments"
echo ""

if [ ${#COMPLETED[@]} -gt 0 ]; then
    echo -e "${GREEN}Completed experiments:${NC}"
    for exp in "${COMPLETED[@]}"; do
        echo "  ✓ ${exp}"
    done
    echo ""
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "${RED}Failed experiments:${NC}"
    for exp in "${FAILED[@]}"; do
        echo "  ✗ ${exp}"
    done
    echo ""
fi

echo "Results directory: ${RESULTS_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Review experiment logs:"
echo "     ls -lh ${RESULTS_DIR}/*/summary.md"
echo ""
echo "  2. Compare results:"
echo "     python3 compare_experiments.py ${RESULTS_DIR}"
echo ""
echo "  3. Generate report:"
echo "     ./generate_ablation_report.sh ${RESULTS_DIR}"
echo ""
echo "  4. Document findings:"
echo "     vim EXPERIMENT_LOG_002_ABLATION_RESULTS.md"
echo ""
echo "Full log: ${LOG_FILE}"
echo ""

# Exit with appropriate code
if [ ${#FAILED[@]} -eq 0 ]; then
    exit 0
else
    exit 1
fi
