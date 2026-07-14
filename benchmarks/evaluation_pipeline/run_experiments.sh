#!/usr/bin/env bash
# run_experiments.sh — Launch one or more throughput experiments, from
# one of three independent suites (no accuracy scoring -- see README.md):
#   spec  (default) -- S0xx, spec-decode on/off + MTP draft length k
#   batch            -- B0xx, max_num_seqs sweep (see EXPERIMENT_PLAN_MAX_NUM_SEQS.md)
#   cross            -- X0xx, max_num_seqs x MTP k grid, spec decode on for every
#                        row (see EXPERIMENT_PLAN_MNS_SPEC_CROSS.md)
# Suites are never merged into one --all -- pick exactly one via --suite.
#
# Usage:
#   ./run_experiments.sh S000                    # single spec-suite experiment, all datasets, 2 reps
#   ./run_experiments.sh S001,S003                # comma-separated list
#   ./run_experiments.sh --all                     # every spec-suite experiment with include_in_all=True
#                                                    # (see run_pipeline.py --list; some, like
#                                                    # single-dataset smoke tests, are excluded)
#   ./run_experiments.sh S003 --datasets aime,gpqa_diamond --reps 3
#   ./run_experiments.sh --suite batch --all        # every batch-size (max_num_seqs) experiment
#   ./run_experiments.sh --suite batch B003         # single batch-size experiment
#   ./run_experiments.sh --suite cross --all        # every mns x MTP-k cross experiment
#   ./run_experiments.sh --suite cross X003         # single cross experiment
#   ./run_experiments.sh --list                    # print the (default suite's) experiment matrix and exit
#   ./run_experiments.sh --suite batch --list       # print the batch-suite matrix and exit
#   ./run_experiments.sh --suite cross --list       # print the cross-suite matrix and exit
#
# Environment variables (source ../gemma4_moe_benchmarks/.env_exports.sh
# first -- same file, same variables -- or set these yourself):
#   GEMMA4_MODEL_PATH            full model checkpoint path
#   GEMMA4_ASSISTANT_MODEL_PATH  MTP assistant/draft model path
#
# NOTE: unlike gemma4_moe_benchmarks/run_experiments.sh, this script does
# NOT loop per-experiment in bash. That script does because different
# experiment IDs there need different env vars (VLLM_USE_FLASHINFER_MOE_FP8)
# set before vLLM is imported. This pipeline's sweep (spec-decode on/off,
# k) doesn't vary any env var across experiments, so the whole
# comma-separated list is passed to run_pipeline.py in a single process,
# which already loops over it internally.

set -euo pipefail
cd "$(dirname "$0")"

: "${PYTHON_BIN:=python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN '$PYTHON_BIN' not found in PATH"
  exit 1
fi

# ---------------------------------------------------------------------------
# CUDA runtime library path (required for precompiled vLLM binaries).
# Same auto-detection as gemma4_moe_benchmarks/run_experiments.sh.
# ---------------------------------------------------------------------------
_NVIDIA_LIB_GLOBS=$($PYTHON_BIN - <<'PY' 2>/dev/null || true
import glob, os, sysconfig
sp = sysconfig.get_paths()["purelib"]
roots = []
for sub in ("nvidia/cu13/lib", "nvidia/cu12/lib"):
    roots.extend(glob.glob(os.path.join(sp, sub)))
roots.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(":".join(sorted(set(roots))))
PY
)
if [[ -n "${_NVIDIA_LIB_GLOBS}" ]]; then
  export LD_LIBRARY_PATH="${_NVIDIA_LIB_GLOBS}:${LD_LIBRARY_PATH:-}"
fi

# ---------------------------------------------------------------------------
# Default model paths (override via env, or source
# ../gemma4_moe_benchmarks/.env_exports.sh before calling this script).
# ---------------------------------------------------------------------------
: "${GEMMA4_MODEL_PATH:=google/gemma-4-26B-A4B-it}"
: "${GEMMA4_ASSISTANT_MODEL_PATH:=google/gemma-4-26B-A4B-it-assistant}"
export GEMMA4_MODEL_PATH GEMMA4_ASSISTANT_MODEL_PATH

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
EXP_IDS=()
DATASETS="aime,gpqa_diamond,livecodebench"
REPS=2
RUN_ALL=0
SUITE="spec"
DO_LIST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)       RUN_ALL=1; shift ;;
    --suite)     SUITE="$2"; shift 2 ;;
    --datasets)  DATASETS="$2"; shift 2 ;;
    --reps)      REPS="$2"; shift 2 ;;
    --list)      DO_LIST=1; shift ;;
    -h|--help)
      sed -n '2,24p' "$0"     # print the usage block at top of script
      exit 0
      ;;
    *)           EXP_IDS+=("$1"); shift ;;
  esac
done

# --list is handled after the full parse (not inline) so --suite batch --list
# forwards the suite it was actually given, rather than always listing the
# default suite.
if [[ $DO_LIST -eq 1 ]]; then
  "$PYTHON_BIN" run_pipeline.py --suite "$SUITE" --list
  exit 0
fi

if [[ $RUN_ALL -eq 0 ]]; then
  EXP_IDS_CSV=$(IFS=','; echo "${EXP_IDS[*]}")
  if [[ -z "$EXP_IDS_CSV" ]]; then
    echo "ERROR: no experiment ID provided. Use --all or specify e.g. S000 (or, with --suite batch, B001)"
    exit 1
  fi
fi
# When RUN_ALL, --all is forwarded to run_pipeline.py directly below rather
# than hardcoding the experiment list here -- run_pipeline.py's SUITES dict
# (SPEC_EXPERIMENTS/BATCH_EXPERIMENTS, see each entry's include_in_all flag)
# is the single source of truth for which experiments --all covers, so a new
# experiment can't be silently missed on one side but not the other.

# ---------------------------------------------------------------------------
# Preflight: fail fast if the selected Python env can't import what
# run_pipeline.py needs, rather than failing partway through a run.
# ---------------------------------------------------------------------------
echo "Python executable: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "Python version   : $($PYTHON_BIN -V 2>&1)"
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import vllm
import transformers
PY
then
  echo "ERROR: Python preflight failed. '$PYTHON_BIN' must import vllm and transformers."
  echo "Hint: source ../gemma4_moe_benchmarks/.env_exports.sh (activates the right conda env), or"
  echo "      prepend the intended env to PATH, e.g. 'export PATH=/opt/conda/envs/vllm-ablation/bin:\$PATH'"
  exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "=================================================="
echo "  Throughput pipeline (suite=${SUITE})"
if [[ $RUN_ALL -eq 1 ]]; then
  echo "  Experiments : --all (run_pipeline.py --suite ${SUITE} --list shows which IDs that covers)"
else
  echo "  Experiments : ${EXP_IDS_CSV}"
fi
echo "  Datasets    : ${DATASETS}"
echo "  Reps        : ${REPS}"
echo "  Model       : ${GEMMA4_MODEL_PATH}"
echo "  Assistant   : ${GEMMA4_ASSISTANT_MODEL_PATH}"
echo "  EVAL_RESULTS_DIR: ${EVAL_RESULTS_DIR:-results}"
echo "  $(date)"
echo "=================================================="

STATUS=0
if [[ $RUN_ALL -eq 1 ]]; then
  "$PYTHON_BIN" run_pipeline.py \
      --suite "${SUITE}" \
      --all \
      --datasets "${DATASETS}" \
      --reps "${REPS}" || STATUS=$?
else
  "$PYTHON_BIN" run_pipeline.py \
      --suite "${SUITE}" \
      --exp "${EXP_IDS_CSV}" \
      --datasets "${DATASETS}" \
      --reps "${REPS}" || STATUS=$?
fi

echo ""
echo "=================================================="
echo "  Pipeline run finished"
echo "  Results: ${EVAL_RESULTS_DIR:-results}/all_runs.csv"
if [[ $STATUS -ne 0 ]]; then
  echo "  FAILED (exit $STATUS) -- see run_pipeline.py output above for which experiment(s)"
  exit "$STATUS"
fi
echo "  All experiments PASSED"
echo "=================================================="
