#!/bin/bash
# Test script to validate ablation study setup
# Runs quick checks without full inference

set -e

echo "=========================================="
echo "Ablation Study Setup Validation"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ERRORS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
        ERRORS=$((ERRORS + 1))
    fi
}

# =============================================================================
# 1. Check Required Files
# =============================================================================

echo "1. Checking required files..."
echo ""

test -f "run_ablation_experiment.sh"
check "run_ablation_experiment.sh exists"

test -f "run_all_ablation_experiments.sh"
check "run_all_ablation_experiments.sh exists"

test -f "run_inference_configurable.py"
check "run_inference_configurable.py exists"

test -f "EXPERIMENT_PLAN_ABLATION_STUDY.md"
check "EXPERIMENT_PLAN_ABLATION_STUDY.md exists"

test -f "create_text_only_model.py"
check "create_text_only_model.py exists"

test -x "run_ablation_experiment.sh"
check "run_ablation_experiment.sh is executable"

test -x "run_all_ablation_experiments.sh"
check "run_all_ablation_experiments.sh is executable"

test -x "run_inference_configurable.py"
check "run_inference_configurable.py is executable"

echo ""

# =============================================================================
# 2. Check Model Paths
# =============================================================================

echo "2. Checking model paths..."
echo ""

if [ -d "/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it" ]; then
    check "Main model exists"
else
    echo -e "${RED}✗${NC} Main model NOT found: /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant" ]; then
    check "Assistant model exists (for MTP)"
else
    echo -e "${YELLOW}⚠${NC} Assistant model NOT found (MTP experiments will fail)"
    echo "  Path: /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant"
fi

if [ -d "/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only" ]; then
    check "Text-only model exists"
else
    echo -e "${YELLOW}⚠${NC} Text-only model NOT found (will be created or experiments will use full model)"
    echo "  Run: python3 create_text_only_model.py to create it"
fi

echo ""

# =============================================================================
# 3. Check Python Environment
# =============================================================================

echo "3. Checking Python environment..."
echo ""

# Activate conda
source /root/miniconda3/bin/activate vllm-ablation 2>/dev/null || true

python3 --version > /dev/null 2>&1
check "Python 3 available"

python3 -c "import torch" 2>/dev/null
check "PyTorch installed"

python3 -c "import vllm" 2>/dev/null
check "vLLM installed"

python3 -c "from vllm import AsyncLLMEngine, AsyncEngineArgs" 2>/dev/null
check "vLLM AsyncEngine available"

# Check version
VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "  vLLM version: ${VLLM_VERSION}"

echo ""

# =============================================================================
# 4. Check GPU
# =============================================================================

echo "4. Checking GPU..."
echo ""

nvidia-smi > /dev/null 2>&1
check "nvidia-smi available"

GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -ge 1 ]; then
    check "At least 1 GPU detected ($GPU_COUNT GPUs)"
else
    echo -e "${RED}✗${NC} No GPUs detected!"
    ERRORS=$((ERRORS + 1))
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  GPU: ${GPU_NAME}"

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  Memory: ${GPU_MEM} MB"

if [ "$GPU_MEM" -ge 40000 ]; then
    check "GPU has >= 40GB memory"
else
    echo -e "${YELLOW}⚠${NC} GPU has < 40GB memory (experiments may OOM)"
fi

echo ""

# =============================================================================
# 5. Test Script Syntax
# =============================================================================

echo "5. Testing script syntax..."
echo ""

bash -n run_ablation_experiment.sh 2>/dev/null
check "run_ablation_experiment.sh syntax OK"

bash -n run_all_ablation_experiments.sh 2>/dev/null
check "run_all_ablation_experiments.sh syntax OK"

python3 -m py_compile run_inference_configurable.py 2>/dev/null
check "run_inference_configurable.py syntax OK"

echo ""

# =============================================================================
# 6. Test Dry Run
# =============================================================================

echo "6. Testing dry run mode..."
echo ""

./run_ablation_experiment.sh --exp TEST001 --batch 64 --fp8 --dry-run > /dev/null 2>&1
check "Dry run completed successfully"

echo ""

# =============================================================================
# 7. Test Help
# =============================================================================

echo "7. Testing help output..."
echo ""

./run_ablation_experiment.sh --help > /dev/null 2>&1
check "Help command works"

python3 run_inference_configurable.py --help > /dev/null 2>&1
check "Python script help works"

echo ""

# =============================================================================
# 8. Check Disk Space
# =============================================================================

echo "8. Checking disk space..."
echo ""

AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "  Available space: ${AVAILABLE_GB}GB"

if [ "$AVAILABLE_GB" -ge 20 ]; then
    check "Sufficient disk space (>= 20GB)"
else
    echo -e "${YELLOW}⚠${NC} Low disk space (< 20GB), experiments may fail"
    echo "  Need ~10-15GB for experiment logs and traces"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "=========================================="
    echo ""
    echo "Ready to run ablation study:"
    echo "  Single experiment: ./run_ablation_experiment.sh --exp E002 --batch 64 --fp8"
    echo "  All experiments:   ./run_all_ablation_experiments.sh"
    echo ""
    if [ ! -d "/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only" ]; then
        echo "Note: text-only model not present — build it before E007–E014:"
        echo "  python3 create_text_only_model.py \\"
        echo "    --model_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \\"
        echo "    --output_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only"
        echo ""
    fi
    exit 0
else
    echo -e "${RED}✗ ${ERRORS} error(s) found!${NC}"
    echo "=========================================="
    echo ""
    echo "Please fix the errors above before running experiments."
    echo ""
    exit 1
fi
