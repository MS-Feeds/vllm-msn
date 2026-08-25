#!/bin/bash
# Resume the ablation sweep at E006. E001-E005 already produced valid
# metrics.json on a previous run; only E006-E015 are still pending.
#
# Args for each experiment are identical to those in
# run_all_ablation_experiments.sh — keep the two files in sync if you change
# any experiment's recipe.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

mkdir -p experiment_results/ablation_study
LOG_FILE="experiment_results/ablation_study/ablation_study_resume.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=========================================="
echo "Resuming ablation study at E006"
echo "Started at $(date)"
echo "=========================================="
echo ""

run_experiment() {
    local exp_id=$1
    shift
    local exp_args=("$@")
    echo ""
    echo "========================================"
    echo -e "${BLUE}Running Experiment ${exp_id}${NC}"
    echo "========================================"
    if ./run_ablation_experiment.sh --exp "${exp_id}" "${exp_args[@]}"; then
        echo -e "${GREEN}✓ ${exp_id} completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ ${exp_id} failed${NC}"
        return 1
    fi
}

declare -a COMPLETED=()
declare -a FAILED=()

cooldown() {
    echo ""
    echo "Cooldown (30 seconds)..."
    sleep 30
    echo ""
}

# E006 already completed successfully in the previous resume run — skipping.
# (E001-E006 metrics.json files exist in experiment_results/.)

# E007: Text-only + MTP (the gemma4_mm vision_config patch is validated here)
echo "E007: Use Text-Only Model (vision removed)"
if run_experiment E007 \
    --backend FLASHINFER --batch 128 --fp8 --cuda-graphs --mtp --gpu-mem 0.75 --text-only; then
    COMPLETED+=("E007")
else
    FAILED+=("E007")
fi
cooldown

echo "E008: Test Larger Batch (192)"
if run_experiment E008 \
    --backend FLASHINFER --batch 192 --fp8 --cuda-graphs --mtp --gpu-mem 0.75 --text-only; then
    COMPLETED+=("E008")
else
    FAILED+=("E008"); echo -e "${YELLOW}⚠ E008 failed (likely OOM)${NC}"
fi
cooldown

echo "E009: Test Maximum Batch (256, may OOM)"
if run_experiment E009 \
    --backend FLASHINFER --batch 256 --fp8 --cuda-graphs --mtp --gpu-mem 0.75 --text-only; then
    COMPLETED+=("E009")
else
    FAILED+=("E009"); echo -e "${YELLOW}⚠ E009 failed (expected if OOM)${NC}"
fi
cooldown

echo "E010: Lower Memory Utilization (0.70)"
if run_experiment E010 \
    --backend FLASHINFER --batch 128 --fp8 --cuda-graphs --mtp --gpu-mem 0.70 --text-only; then
    COMPLETED+=("E010")
else
    FAILED+=("E010")
fi
cooldown

echo "E011: Higher Memory Utilization (0.80)"
if run_experiment E011 \
    --backend FLASHINFER --batch 128 --fp8 --cuda-graphs --mtp --gpu-mem 0.80 --text-only; then
    COMPLETED+=("E011")
else
    FAILED+=("E011"); echo -e "${YELLOW}⚠ E011 failed${NC}"
fi
cooldown

echo "E012: Flash Attention 2 (for comparison)"
if run_experiment E012 \
    --backend FLASH_ATTN --batch 128 --fp8 --cuda-graphs --mtp --gpu-mem 0.75 --text-only; then
    COMPLETED+=("E012")
else
    FAILED+=("E012")
fi
cooldown

echo "E013: No MTP (measure MTP contribution)"
if run_experiment E013 \
    --backend FLASHINFER --batch 128 --fp8 --cuda-graphs --no-mtp --gpu-mem 0.75 --text-only; then
    COMPLETED+=("E013")
else
    FAILED+=("E013")
fi
cooldown

echo "E014: Test FP8_E4M3 KV Cache Format (EXPECTED TO FAIL on sm_80)"
if run_experiment E014 \
    --backend FLASHINFER --batch 128 --fp8 --kv-cache-dtype fp8_e4m3 --cuda-graphs --mtp --gpu-mem 0.75 --text-only; then
    COMPLETED+=("E014")
else
    FAILED+=("E014"); echo -e "${YELLOW}⚠ E014 failed (expected — Triton sm_80 lacks fp8e4nv)${NC}"
fi
cooldown

echo "E015: Full BF16 Baseline (may OOM)"
if run_experiment E015 \
    --backend FLASHINFER --batch 32 --no-fp8 --no-cuda-graphs --no-mtp --gpu-mem 0.95 --text-only; then
    COMPLETED+=("E015")
else
    FAILED+=("E015"); echo -e "${YELLOW}⚠ E015 failed (expected if OOM)${NC}"
fi

echo ""
echo "=========================================="
echo "Resume sweep complete at $(date)"
echo "=========================================="
echo "Completed: ${#COMPLETED[@]} / 10"
for e in "${COMPLETED[@]}"; do echo -e "  ${GREEN}✓${NC} $e"; done
echo "Failed:    ${#FAILED[@]} / 10"
for e in "${FAILED[@]}"; do echo -e "  ${RED}✗${NC} $e"; done

if [ ${#FAILED[@]} -eq 0 ]; then exit 0; else exit 1; fi
