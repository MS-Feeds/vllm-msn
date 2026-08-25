#!/usr/bin/env bash
# run_ablation.sh — Launch one or more ablation experiments.
#
# Usage:
#   ./run_ablation.sh E001               # single experiment, sc1, 2 reps
#   ./run_ablation.sh E001,E003,E006     # comma-separated list
#   ./run_ablation.sh --all              # all 16 experiments
#   ./run_ablation.sh E011 --scenario sc2 --reps 3
#
# Environment variables (can override before calling):
#   GEMMA4_MODEL_PATH            full model checkpoint path
#   GEMMA4_TEXT_ONLY_MODEL_PATH  text-only checkpoint path
#   GEMMA4_ASSISTANT_MODEL_PATH  MTP assistant model path
#
# The script resolves VLLM_ATTENTION_BACKEND and FlashInfer knobs per
# experiment ID rather than having them hard-coded in bench_ablation.py
# (vllm imports freeze env vars at import time, so they must be set here).

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Default model paths (override via env)
# ---------------------------------------------------------------------------
: "${GEMMA4_MODEL_PATH:=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it}"
: "${GEMMA4_TEXT_ONLY_MODEL_PATH:=${GEMMA4_MODEL_PATH}-text-only}"
: "${GEMMA4_ASSISTANT_MODEL_PATH:=/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant}"
export GEMMA4_MODEL_PATH GEMMA4_TEXT_ONLY_MODEL_PATH GEMMA4_ASSISTANT_MODEL_PATH

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
EXP_IDS=""
SCENARIO="sc1"
REPS=2
RUN_ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)          RUN_ALL=1; shift ;;
    --scenario)     SCENARIO="$2"; shift 2 ;;
    --reps)         REPS="$2"; shift 2 ;;
    --list)
      python3 bench_ablation.py --exp E001 --list
      exit 0
      ;;
    -h|--help)
      sed -n '2,22p' "$0"     # print the usage block at top of script
      exit 0
      ;;
    *)              EXP_IDS="$1"; shift ;;
  esac
done

if [[ $RUN_ALL -eq 1 ]]; then
  EXP_IDS="E001,E002,E003,E004,E005,E006,E007,E008,E009,E010,E011,E012,E013,E014,E015,E016"
fi

if [[ -z "$EXP_IDS" ]]; then
  echo "ERROR: no experiment ID provided. Use --all or specify e.g. E001"
  exit 1
fi

# ---------------------------------------------------------------------------
# Per-experiment environment variable table.
# These must be set BEFORE vllm is imported, so this shell wrapper is the
# right place.
#
# NOTE: Gemma 4 has heterogeneous attention head dims (256 and 512), so vLLM
# forces TRITON_ATTN regardless of VLLM_ATTENTION_BACKEND.  The env var is
# set here only so it appears in logs for reproducibility — it has no runtime
# effect on this model.
#
# Mapping:
#   VLLM_ATTENTION_BACKEND          : FLASH_ATTN (default, all exps)
#   VLLM_USE_FLASHINFER_MOE_FP8     : 0 | 1  (H100 only; A100 must use 0)
#   VLLM_USE_FLASHINFER_SAMPLER     : 0 | 1  (JIT compile fix for old nvcc)
# ---------------------------------------------------------------------------
set_env_for_exp() {
  local exp="$1"
  # Default: safe values for all experiments
  export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
  export VLLM_USE_FLASHINFER_SAMPLER="0"

  # VLLM_USE_FLASHINFER_MOE_FP8: needed for FP8 MoE on H100 (sm_90+), must be 0 on A100.
  # Only enable for experiments that use FP8 weights (E002–E013).
  case "$exp" in
    E001|E014|E015|E016)
      # BF16 weights — FlashInfer FP8 MoE is irrelevant; keep 0
      export VLLM_USE_FLASHINFER_MOE_FP8="0"
      ;;
    E002|E003|E004|E005|E006|E007|E008|E009|E010|E011|E012|E013)
      # FP8 weights: enable FlashInfer FP8 MoE on H100, keep off on A100
      COMPUTE_CAP=$(python3 -c "import torch; cc=torch.cuda.get_device_capability(); print(cc[0]*10+cc[1])" 2>/dev/null || echo "0")
      if [[ "$COMPUTE_CAP" -ge 90 ]]; then
        export VLLM_USE_FLASHINFER_MOE_FP8="1"
        echo "  → H100 detected (sm_${COMPUTE_CAP}): enabling VLLM_USE_FLASHINFER_MOE_FP8=1"
      else
        export VLLM_USE_FLASHINFER_MOE_FP8="0"
        echo "  → non-H100 (sm_${COMPUTE_CAP}): VLLM_USE_FLASHINFER_MOE_FP8=0 (Marlin FP8 MoE fallback)"
      fi
      ;;
    *)
      export VLLM_USE_FLASHINFER_MOE_FP8="0"
      echo "WARNING: unknown exp '$exp' — using safe defaults"
      ;;
  esac

  echo "  ENV: VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND  VLLM_USE_FLASHINFER_MOE_FP8=$VLLM_USE_FLASHINFER_MOE_FP8  VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
IFS=',' read -ra EXPS <<< "$EXP_IDS"

echo "=================================================="
echo "  Ablation benchmark"
echo "  Experiments : ${EXP_IDS}"
echo "  Scenario    : ${SCENARIO}"
echo "  Reps        : ${REPS}"
echo "  $(date)"
echo "=================================================="

FAILED=()
for EXP in "${EXPS[@]}"; do
  EXP="$(echo "$EXP" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"
  echo ""
  echo ">>> Setting up environment for ${EXP}"
  set_env_for_exp "$EXP"

  echo ">>> Launching bench_ablation.py --exp ${EXP} --scenario ${SCENARIO} --reps ${REPS}"
  if python3 bench_ablation.py \
      --exp "${EXP}" \
      --scenario "${SCENARIO}" \
      --reps "${REPS}"; then
    echo ">>> ${EXP} COMPLETED"
  else
    echo ">>> ${EXP} FAILED (exit $?)"
    FAILED+=("$EXP")
  fi
done

echo ""
echo "=================================================="
echo "  Ablation run finished"
echo "  Results: ablation_results/all_runs.csv"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  FAILED experiments: ${FAILED[*]}"
  exit 1
fi
echo "  All experiments PASSED"
echo "=================================================="
