#!/usr/bin/env bash
# Environment for running benchmarks/rlm_specprefill/ against the
# Qwen3-Coder-480B-A35B (target) / Qwen3-Coder-30B-A3B (speculator) pairing,
# on a GPU node. Source this (not execute): `source .env_exports_qwen_coder.sh`
#
# Sibling to .env_exports.sh (the original Llama pairing) -- same structure,
# pointed at ../spec_prefill_qwen_coder/ instead of ../spec_prefill_llama/.
# Both files can't be sourced usefully in the same shell session (they both
# set SPEC_PREFILL_LLAMA_DIR-equivalent + results/log dir vars) -- pick
# whichever pairing this run targets.

set -a  # export everything sourced below, including from the .env files

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Root model ---
# ANTHROPIC_API_KEY is still loaded (harmless if unused) since
# --root-backend=anthropic remains selectable per-invocation if you want to
# compare against a hosted-Claude root on this same pairing. rlm/.env holds
# ANTHROPIC_API_KEY (+ empty OPENAI_API_KEY/GEMINI_API_KEY placeholders,
# unused by this ablation). Same as .env_exports.sh -- see that file for
# the full rationale, not duplicated here.
if [ -f "$THIS_DIR/../rlm/.env" ]; then
    # shellcheck disable=SC1091
    source "$THIS_DIR/../rlm/.env"
else
    echo "WARNING: ../rlm/.env not found -- ANTHROPIC_API_KEY will not be set." >&2
fi

# --- Target/speculator (Qwen3-Coder via self-hosted vLLM) ---
# Pulls in QWEN3_CODER_480B_MODEL_PATH, QWEN3_CODER_30B_MODEL_PATH, HF_HOME,
# HF_TOKEN, VLLM_USE_V2_MODEL_RUNNER=0 (a harmless no-op pin for this MoE
# pairing, see that file's own comment), and the CUDA LD_LIBRARY_PATH fix --
# see ../spec_prefill_qwen_coder/.env_exports.sh for why each of these is
# set the way it is. Unlike the Llama pairing, the target here (480B-A35B)
# does NOT fit on a single GPU -- see REPRODUCE.md-equivalent guidance in
# that directory before assuming a GPU count.
if [ -f "$THIS_DIR/../spec_prefill_qwen_coder/.env_exports.sh" ]; then
    # shellcheck disable=SC1091
    source "$THIS_DIR/../spec_prefill_qwen_coder/.env_exports.sh"
else
    echo "WARNING: ../spec_prefill_qwen_coder/.env_exports.sh not found -- target/speculator model paths will not be set." >&2
fi

# --- Root backend selection (must come AFTER sourcing spec_prefill_qwen_coder's
# .env_exports.sh above, since ROOT_MODEL_PATH's default references
# QWEN3_CODER_480B_MODEL_PATH, which that sourcing is what actually sets) ---
# Defaults to root_backend='vllm' for THIS pairing -- self-hosted
# Qwen3-Coder-480B-A35B serving as both root (RLM's REPL orchestration) and
# SpecPrefill target, matching the RLM paper's default same-model-recursion
# design (see rlm_stage/evidence_rlm.py's module docstring for how this
# differs from the original Llama pairing's hosted-Claude root, and
# rlm_stage/root_vllm_server.py for the vllm-serve lifecycle this implies --
# runner/run_arm.py starts/stops it around the RLM evidence stage
# automatically).
export ROOT_BACKEND=vllm
export ROOT_BASE_URL="http://localhost:8000/v1"
export ROOT_PORT=8000
# Same checkpoint as the target by default -- set explicitly only if you
# want a different model/quantization serving as root than as target.
export ROOT_MODEL_PATH="${ROOT_MODEL_PATH:-$QWEN3_CODER_480B_MODEL_PATH}"

# --- This project's own additions ---
# target_stage/vllm_offline_engine.py's ensure_spec_prefill_on_path() needs
# spec_prefill_qwen_coder's vllm_patch/ on sys.path (it's not an installed
# package) -- resolved via this env var (pass
# --spec-prefill-dir-env=SPEC_PREFILL_QWEN_CODER_DIR to runner/run_arm.py
# or runner/run_all_arms.py to select it) rather than a hardcoded relative
# path, mirroring SPEC_PREFILL_LLAMA_DIR's own rationale in .env_exports.sh.
export SPEC_PREFILL_QWEN_CODER_DIR="$THIS_DIR/../spec_prefill_qwen_coder"

# Namespaced under results/qwen_coder/ (not results/, which the Llama
# pairing's .env_exports.sh points at) so a Qwen-Coder sweep never
# clobbers an existing Llama arm's results in the same results_dir.
export RLM_SPECPREFILL_RESULTS_DIR="${RLM_SPECPREFILL_RESULTS_DIR:-$THIS_DIR/results/qwen_coder}"
export RLM_SPECPREFILL_LOG_DIR="${RLM_SPECPREFILL_LOG_DIR:-$THIS_DIR/logs}"

set +a

echo "[rlm_specprefill] env loaded (Qwen3-Coder pairing). ANTHROPIC_API_KEY set: $([ -n "$ANTHROPIC_API_KEY" ] && echo yes || echo no)"
echo "[rlm_specprefill] QWEN3_CODER_480B_MODEL_PATH=${QWEN3_CODER_480B_MODEL_PATH:-<unset>}"
echo "[rlm_specprefill] QWEN3_CODER_30B_MODEL_PATH=${QWEN3_CODER_30B_MODEL_PATH:-<unset>}"
echo "[rlm_specprefill] TARGET_TENSOR_PARALLEL_SIZE=${TARGET_TENSOR_PARALLEL_SIZE:-<unset>} (run_arm.py's --target-tensor-parallel-size default; --target-enable-expert-parallel is NOT set by default at this TP size -- pass it explicitly if you hit an uneven expert-split error)"
echo "[rlm_specprefill] ROOT_BACKEND=${ROOT_BACKEND:-<unset>} ROOT_MODEL_PATH=${ROOT_MODEL_PATH:-<unset>} ROOT_BASE_URL=${ROOT_BASE_URL:-<unset>} (root_backend='vllm' means run_arm.py starts/stops its own vllm serve around the RLM evidence stage -- see rlm_stage/root_vllm_server.py)"
