#!/usr/bin/env bash
# Environment for running benchmarks/rlm_specprefill/ on a GPU node.
# Source this (not execute): `source .env_exports.sh`
#
# Ties together two independent projects' env setup -- neither is duplicated
# here, both are sourced/re-exported so this directory has a single entry
# point. See each project's own .env_exports.sh for the full rationale
# behind individual settings.

set -a  # export everything sourced below, including from the .env files

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Root model (Claude via Anthropic API) ---
# rlm/.env holds ANTHROPIC_API_KEY (+ empty OPENAI_API_KEY/GEMINI_API_KEY
# placeholders, unused by this ablation). python-dotenv's load_dotenv() in
# rlm_stage/evidence_rlm.py reads this directly too; sourcing it here as
# well makes the key available to any shell-level tooling (e.g. this repo's
# own tests/scripts run outside a Python process).
if [ -f "$THIS_DIR/../rlm/.env" ]; then
    # shellcheck disable=SC1091
    source "$THIS_DIR/../rlm/.env"
else
    echo "WARNING: ../rlm/.env not found -- ANTHROPIC_API_KEY will not be set." >&2
fi

# --- Target/speculator (Llama via self-hosted vLLM) ---
# Pulls in LLAMA31_8B_MODEL_PATH, LLAMA32_1B_MODEL_PATH, HF_HOME, HF_TOKEN,
# VLLM_USE_V2_MODEL_RUNNER=0, and the CUDA LD_LIBRARY_PATH fix -- see
# ../spec_prefill_llama/.env_exports.sh for why each of these is set the way
# it is. Unlike the sibling Qwen3 pipeline, VLLM_USE_V2_MODEL_RUNNER=0 isn't
# strictly load-bearing for Llama (this fork's v2-runner architecture list
# currently only includes Qwen3ForCausalLM, per that file's own comment) --
# kept anyway as an explicit pin, not a functional requirement here. Both
# Llama checkpoints are gated on Hugging Face -- see REPRODUCE.md step 1 for
# the access-request step before downloading.
if [ -f "$THIS_DIR/../spec_prefill_llama/.env_exports.sh" ]; then
    # shellcheck disable=SC1091
    source "$THIS_DIR/../spec_prefill_llama/.env_exports.sh"
else
    echo "WARNING: ../spec_prefill_llama/.env_exports.sh not found -- target/speculator model paths will not be set." >&2
fi

# --- This project's own additions ---
# target_stage/vllm_offline_engine.py needs spec_prefill_llama's vllm_patch/
# on sys.path (it's not an installed package) -- resolved via this env var
# rather than a hardcoded relative path, so it survives being run from a
# different CWD.
export SPEC_PREFILL_LLAMA_DIR="$THIS_DIR/../spec_prefill_llama"

# Where RLMLogger(log_dir=...) and results/* write to, resolved relative to
# THIS_DIR regardless of invocation CWD (mirrors spec_prefill_llama's
# BENCH_RESULTS_DIR convention).
export RLM_SPECPREFILL_RESULTS_DIR="${RLM_SPECPREFILL_RESULTS_DIR:-$THIS_DIR/results}"
export RLM_SPECPREFILL_LOG_DIR="${RLM_SPECPREFILL_LOG_DIR:-$THIS_DIR/logs}"

set +a

echo "[rlm_specprefill] env loaded. ANTHROPIC_API_KEY set: $([ -n "$ANTHROPIC_API_KEY" ] && echo yes || echo no)"
echo "[rlm_specprefill] LLAMA31_8B_MODEL_PATH=${LLAMA31_8B_MODEL_PATH:-<unset>}"
