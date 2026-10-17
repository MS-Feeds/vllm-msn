#!/usr/bin/env bash
# DeepSeek-V3.2 Lightning-Indexer Top-K Distribution Experiment
# 3 benchmarks × 100 prompts = 300 total
#
# Purpose: Capture raw attention-logit distributions from the DeepSeek-V3.2
#          lightning indexer (config.index_topk / sparse_attn_indexer.py)
#          across math, code, and QA benchmarks, to evaluate radix-sort
#          bit-width reduction for the sparse-attention top-k select.
#
#          This targets the same mechanism as the GLM-5.1 baseline in this
#          folder (they share sparse_attn_indexer.py) -- NOT MoE expert
#          routing top-k, which is a different, unrelated selection.
#
# Usage:
#   bash run_v32_experiment.sh [--dry-run] [--server-only | --eval-only]
#
#   --server-only   Launch the server, wait for it to be ready, print the
#                    URL, then return (leaving it running) instead of also
#                    running the 300-prompt eval sweep. Useful for a quick
#                    manual smoke-test of a checkpoint/config before
#                    committing to a full run.
#   --eval-only     Skip syncing patches and launching a server entirely --
#                    assumes one is already running at $VLLM_PORT (e.g. from
#                    a prior --server-only run) and just runs the eval sweep
#                    against it. Does not shut it down afterward, since this
#                    script didn't start it.

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# $HOME has a full 10GB NFS quota on this host (confirmed via `quota -s`).
# vLLM's various JIT/download caches default into $HOME/.cache/* (or
# $HOME/.<tool>/*) and will fail with "Disk quota exceeded" the moment they
# try to write -- redirect them onto scratch storage before anything else
# runs. Each of these was discovered the hard way, one crash at a time:
# flashinfer (FLASHINFER_WORKSPACE_BASE), then proactively for
# XDG_CACHE_HOME/HF_HOME/TRITON_CACHE_DIR, then flashinfer's *bundled*
# TensorRT-LLM DeepGEMM compiler (TRTLLM_DG_CACHE_DIR, see
# flashinfer/data/csrc/nv_internal/tensorrt_llm/deep_gemm/compiler.cuh) --
# that one throws an uncaught C++ exception (std::terminate, not a catchable
# Python error) straight out of a worker process, taking down the whole
# engine instead of just failing one request.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CACHE_BASE="${CACHE_BASE:-$REPO_ROOT/.cache}"
mkdir -p "$CACHE_BASE"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_BASE}"

# flashinfer appends its own ".cache/flashinfer" suffix onto this base (see
# flashinfer/jit/env.py: FLASHINFER_CACHE_DIR = FLASHINFER_BASE_DIR / ".cache"
# / "flashinfer") -- point it at REPO_ROOT, not CACHE_BASE, or the result
# doubles to "$CACHE_BASE/.cache/flashinfer".
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$REPO_ROOT}"
export HF_HOME="${HF_HOME:-$CACHE_BASE/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_BASE/triton}"
export TRTLLM_DG_CACHE_DIR="${TRTLLM_DG_CACHE_DIR:-$CACHE_BASE/tensorrt_llm}"

# That same TensorRT-LLM DeepGEMM compiler shells out to `nvcc` to JIT-compile
# any GEMM shape not already in its kernel cache -- it's not on PATH, but is
# bundled in the venv via the nvidia-cuda-nvcc pip package. Point at it
# directly (checked before CUDA_HOME by compiler.cuh); also set CUDA_HOME in
# case anything else goes looking for a CUDA toolkit root.
_NVCC_BIN="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc"
if [[ -x "$_NVCC_BIN" ]]; then
    export TRTLLM_DG_NVCC_COMPILER="${TRTLLM_DG_NVCC_COMPILER:-$_NVCC_BIN}"
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$_NVCC_BIN")")}"
fi

# vLLM's *own* DeepGEMM JIT cache (vllm/utils/deep_gemm.py) defaults
# DG_JIT_CACHE_DIR to $VLLM_CACHE_ROOT/deep_gemm, and VLLM_CACHE_ROOT
# defaults to ~/.cache/vllm -- under the same full $HOME, and does NOT
# respect XDG_CACHE_HOME. This is a *different* JIT compiler than the
# TensorRT-LLM one above (assertion trace mentions deepgemm-src/jit/
# compiler.hpp). Reuse this user's existing, populated, SM90 (Hopper,
# matches these H200s) DeepGEMM cache from prior work instead of compiling
# from scratch -- any newly-compiled kernels get added to the same reusable
# cache for next time.
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-/import/ml-sc-scratch5/chenw/.cache/deep_gemm}"

# For kernel shapes not already in that cache, this DeepGEMM JIT compiler's
# generated source pulls in nvidia/cu13/include/cccl headers, which run a
# "CUDA compiler and CUDA toolkit headers are incompatible" self-check
# (cuda_toolkit.h) -- a false positive here (verified: our nvcc 13.3
# compiles a real DeepGEMM kernel cleanly once this check is bypassed).
# CCCL's own header comments this exact escape hatch as intended for
# exactly this "newer CTK than the compiler ships" situation. Inject it via
# nvcc's own flag-injection env var rather than patching DeepGEMM's JIT
# compiler internals to find where it builds its nvcc command line.
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

# This host's RoCE NICs (bnxt_re0-8) crash NCCL's IB/RoCE transport init
# (segfault in ncclNetPluginInit) -- not needed anyway since all ranks are
# on one node over 127.0.0.1; disable it rather than touch NVLink/P2P.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# Default to this checkout's own vllm/ as the sync target -- per AGENTS.md
# this repo is meant to be `uv pip install -e .`'d, so site-packages/vllm
# on a correctly set-up host IS this directory. Override if a different
# install is actually the one being served.
export VLLM_INSTALL_PATH="${VLLM_INSTALL_PATH:-$REPO_ROOT/vllm}"

# Model configuration
MODEL="${DEEPSEEK_V32_MODEL_PATH:-/import/ml-sc-scratch6/chenw/DeepSeek-V3.2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"
GPU_MEM="${GPU_MEM:-0.90}"
TP_SIZE="${TP_SIZE:-8}"  # V3.2 is a large sparse-MoE model; adjust to your GPU count.

# Benchmark configuration
PROMPTS_PER_DATASET="${PROMPTS_PER_DATASET:-100}"
CONCURRENCY="${CONCURRENCY:-4}"  # In-flight requests for continuous batching
SEED="${SEED:-42}"
# livecodebench and gpqa_diamond are excluded by default: gpqa_diamond is a
# gated HF dataset (needs an authenticated HF_TOKEN with accepted terms), and
# livecodebench/code_generation_lite uses a script-based HF loader that
# datasets>=4.0 no longer supports. Override to restore them once fixed.
EVAL_DATASETS="${EVAL_DATASETS:-aime}"

# Output configuration
RUN_ID="deepseek_v32_$(date +%Y%m%d_%H%M%S)"
LOGIT_DUMP_DIR="$PROJECT_ROOT/indexer_logits/$RUN_ID"
EVAL_LOG="$PROJECT_ROOT/logs/eval_${RUN_ID}.jsonl"
SERVER_LOG="$PROJECT_ROOT/logs/server_${RUN_ID}.log"

# Indexer logger configuration: attention-logit mode (the lightning indexer),
# NOT expert-routing mode. This matches the GLM-5.1 baseline configuration.
export INDEXER_LOGIT_DUMP_DIR="$LOGIT_DUMP_DIR"
export INDEXER_IS_EXPERT_ROUTING=0
export INDEXER_LOGIT_RANGE="-50.0,50.0"        # Raw attention logits
export INDEXER_LOGIT_BINS=4096
export INDEXER_LOGIT_SAMPLE_K=0                # Disable reservoir (not needed)

# Dump every 5000 record() calls (~1-2 min/worker at typical decode rates),
# not just at exit. vLLM's multiproc_executor tears down TP workers via a
# layered shutdown (natural queue-close, then SIGTERM, then SIGKILL after
# ~8s -- see MultiprocExecutor._ensure_worker_termination) driven by the
# *parent* closing a death-pipe; confirmed via a full 30-prompt/40-minute
# run that produced zero *.npz files and zero "[_indexer_logger]" dump or
# "WorkerProc shutting down" log lines -- none of the 8 worker processes got
# a clean interpreter exit before teardown, so the atexit/SIGTERM hook never
# fired and all accumulated histogram data was lost. Periodic dumping makes
# every run safe regardless of how shutdown actually goes; the exit-time
# dump (still registered) just becomes the final, most-complete write.
export INDEXER_LOGIT_DUMP_EVERY=5000
# Inert unless INDEXER_STREAMS includes D (see below):
export INDEXER_RADIX_BITS_SWEEP="4,6,8,12,16"
export INDEXER_RADIX_BKT_BINS=256
# Minimal capture: just the raw-logit histogram (B), per layer per phase.
# Skips stream A's and C's shared topk() call entirely (not just their
# output fields) and D's radix-bucket sort. Set INDEXER_STREAMS=A,B,C,D to
# restore the full four-stream capture.
export INDEXER_STREAMS="${INDEXER_STREAMS:-B}"

# Server configuration
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"

# =============================================================================
# Argument Parsing
# =============================================================================

DRY_RUN=0
SERVER_ONLY=0
EVAL_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --server-only) SERVER_ONLY=1 ;;
        --eval-only) EVAL_ONLY=1 ;;
        *) echo "ERROR: Unknown argument: $arg"; exit 1 ;;
    esac
done
if [[ $SERVER_ONLY -eq 1 && $EVAL_ONLY -eq 1 ]]; then
    echo "ERROR: --server-only and --eval-only are mutually exclusive"
    exit 1
fi
if [[ $DRY_RUN -eq 1 ]]; then
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
echo "Indexer streams:    $INDEXER_STREAMS"
echo "Cache base:         $CACHE_BASE"
echo "NCCL_IB_DISABLE:    $NCCL_IB_DISABLE"
echo "VLLM_INSTALL_PATH:  $VLLM_INSTALL_PATH"
echo "nvcc for JIT GEMM:  ${TRTLLM_DG_NVCC_COMPILER:-<not found, using PATH>}"
echo "DeepGEMM JIT cache: $DG_JIT_CACHE_DIR"
echo "NVCC_APPEND_FLAGS: $NVCC_APPEND_FLAGS"
echo ""
echo "Eval datasets:       $EVAL_DATASETS"
echo "Prompts per dataset: $PROMPTS_PER_DATASET (capped by each dataset's real size)"
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

# Check local checkpoint path (skip for HF hub model ids)
if [[ "$MODEL" == /* && ! -d "$MODEL" ]]; then
    echo "ERROR: Local checkpoint path not found: $MODEL"
    echo "Set DEEPSEEK_V32_MODEL_PATH to override."
    exit 1
fi

# Fail fast on a port conflict rather than let Phase 1 try to bind a second
# server and time out 300s later with a confusing error.
if [[ $EVAL_ONLY -eq 0 && $DRY_RUN -eq 0 ]] \
    && curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "ERROR: A server is already responding at http://localhost:$VLLM_PORT/health"
    echo "Use --eval-only to run the eval sweep against it, or stop it first"
    echo "(it was likely started by an earlier --server-only run)."
    exit 1
fi

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

if [[ $EVAL_ONLY -eq 1 ]]; then
    echo "--eval-only: skipping patch sync and server launch."
    echo "Checking for an already-running server at localhost:$VLLM_PORT..."
    if [[ $DRY_RUN -eq 0 ]]; then
        if ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            echo "ERROR: No server responding at http://localhost:$VLLM_PORT/health"
            echo "Start one first with --server-only, or drop --eval-only to launch fresh."
            exit 1
        fi
        echo "✓ Server is up"
    fi
    echo ""
else
    # Sync the indexer-logger patches (_indexer_logger.py + sparse_attn_indexer.py)
    # into the active vLLM install so the lightning indexer records logits.
    # VLLM_INSTALL_PATH always has a value (defaulted above), so this always
    # runs unless dry-run; override VLLM_INSTALL_PATH if a different install
    # is served.
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "Would sync indexer-logger patches (VLLM_INSTALL_PATH=$VLLM_INSTALL_PATH)"
    else
        bash "$SCRIPT_DIR/sync_to_env.sh"
    fi
    echo ""

    # =============================================================================
    # Phase 1: Launch vLLM Server
    # =============================================================================

    echo "=========================================="
    echo "Phase 1: Launching vLLM Server"
    echo "=========================================="

# --enforce-eager: _indexer_logger.record() does host-syncing ops (.item(),
# torch.histc, .cpu().numpy()) that CUDA graph capture forbids -- it crashes
# capture with cudaErrorStreamCaptureUnsupported. Worse, even if it didn't
# crash, a record() call baked into a captured graph would only ever fire
# once (at capture time) and then silently replay that stale call on every
# subsequent decode step, never actually recording new data. Since this is a
# data-collection experiment, not a latency benchmark, eager execution is the
# semantically correct choice, not just a workaround.
LAUNCH_CMD="python3 -m vllm.entrypoints.openai.api_server \
    --model '$MODEL' \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM \
    --tensor-parallel-size $TP_SIZE \
    --trust-remote-code \
    --enforce-eager \
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
    # Prevent a dropped/closed terminal from SIGHUP-ing this multi-hour job.
    disown "$SERVER_PID" 2>/dev/null || true
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
fi

if [[ $SERVER_ONLY -eq 1 ]]; then
    echo ""
    echo "=========================================="
    echo "--server-only: server is up, skipping eval sweep"
    echo "=========================================="
    echo "URL:        http://localhost:$VLLM_PORT"
    echo "Server PID: $SERVER_PID"
    echo "Server log: $SERVER_LOG"
    echo ""
    echo "Stop it with: kill $SERVER_PID"
    exit 0
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
    --datasets $EVAL_DATASETS \
    --n-per-dataset $PROMPTS_PER_DATASET \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --concurrency $CONCURRENCY \
    --seed $SEED \
    --out '$EVAL_LOG'"

echo "Datasets: $EVAL_DATASETS (n-per-dataset=$PROMPTS_PER_DATASET, actual"
echo "          count is capped by each dataset's real size, e.g. AIME=30)"
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

        # Quick stats. Python's default json.dumps puts a space after ':',
        # so the JSONL rows look like `"ok": true` -- match that exactly.
        # grep -c exits 1 (but still prints "0") when there are zero matches;
        # `|| echo "0"` would then run *too*, doubling the captured output
        # into an unparseable "0\n0" and breaking the arithmetic below. Use
        # `|| true` instead, which only normalizes the exit status.
        if [[ -f "$EVAL_LOG" ]]; then
            TOTAL=$(wc -l < "$EVAL_LOG")
            SUCCESS=$(grep -c '"ok": true' "$EVAL_LOG" || true)
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

if [[ $EVAL_ONLY -eq 1 ]]; then
    echo "--eval-only: leaving the server running (this script didn't start it)."
elif [[ $DRY_RUN -eq 1 ]]; then
    echo "Would kill server PID: ${SERVER_PID:-}"
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
