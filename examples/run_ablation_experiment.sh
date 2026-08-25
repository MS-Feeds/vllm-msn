#!/bin/bash
# Enhanced Experiment Runner for Ablation Study
# Supports all configuration options needed for systematic experiments

set -eo pipefail

# Pin cwd to this script's directory so all relative paths (OUTPUT_DIR,
# run_inference_configurable.py, etc.) resolve consistently no matter where
# the user invokes the script from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Always kill the GPU monitor on exit (even on error / Ctrl-C),
# so a failed experiment doesn't leave a zombie nvidia-smi loop.
MONITOR_PID=""
cleanup() {
    if [ -n "${MONITOR_PID}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# =============================================================================
# Configuration Parsing
# =============================================================================

# Defaults
EXPERIMENT_ID="E001"
BACKEND="FLASHINFER"
BATCH_SIZE=128
USE_FP8="true"
USE_CUDA_GRAPHS="false"
USE_MTP="false"
GPU_MEM_UTIL="0.75"
USE_TEXT_ONLY="false"
# Default KV cache dtype. Must be "auto" on A100 (sm_80):
#   - vLLM rejects fp8_e5m2 when loading an FP8-quantized checkpoint
#     (vllm/model_executor/layers/attention/attention.py:167).
#   - Triton's reshape_and_cache_flash kernel on sm_80 supports only
#     fp8e4b15 and fp8e5 — not fp8e4nv (= fp8_e4m3). A100 has no native
#     FP8 tensor cores; FP8 KV cache truly requires H100+ for fast paths.
# So the only working KV-cache-dtype on this hardware with FP8 weights is
# "auto", which keeps KV in BF16. E014's explicit --kv-cache-dtype fp8_e4m3
# will fail with the Triton sm_80 error — that's the legitimate "FP8 KV
# cache on A100" ablation result.
KV_CACHE_DTYPE="auto"
DRY_RUN="false"
MAX_BATCHED_TOKENS=""
# MAI dataset prompt distribution (1K-sample test set, approx tokens):
#   p50=3.4K  p90=16K  p95=22K  p99=48K  max=79K
# We're on a SINGLE A100 80GB; BF16 26B leaves ~25 GB after model load and
# CUDA overhead, so we can't be greedy with KV. 32K is the realistic ceiling:
#   - covers p95 (~22K) cleanly; drops ~2.8% of prompts (28/1000) at p99/tail
#   - KV per max-size request: ~336 MB (FP8) / ~670 MB (BF16) — leaves
#     comfortable batch headroom on both
#   - one max-length prompt fits in one prefill chunk at the default below
# Raise --max-model-len per experiment if you need to validate on the long
# tail; just be aware of the BF16 memory consequences.
MAX_MODEL_LEN="32768"
MAX_TOKENS="1024"       # Generation budget per request (overrideable). 16K matches
                        # the dataset's design ceiling but is too slow for ablation;
                        # 1024 amortizes prefill while keeping each experiment ~8-12 min.

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp|--experiment)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --batch|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --fp8)
            USE_FP8="true"
            shift
            ;;
        --no-fp8)
            USE_FP8="false"
            shift
            ;;
        --cuda-graphs)
            USE_CUDA_GRAPHS="true"
            shift
            ;;
        --no-cuda-graphs)
            USE_CUDA_GRAPHS="false"
            shift
            ;;
        --mtp)
            USE_MTP="true"
            shift
            ;;
        --no-mtp)
            USE_MTP="false"
            shift
            ;;
        --gpu-mem)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --text-only|--text-only-model)
            USE_TEXT_ONLY="true"
            shift
            ;;
        --kv-cache-dtype)
            KV_CACHE_DTYPE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --max-batched-tokens)
            MAX_BATCHED_TOKENS="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --exp, --experiment ID       Experiment ID (default: E001)"
            echo "  --backend TYPE               FLASHINFER|FLASH_ATTN (default: FLASHINFER)"
            echo "  --batch, --batch-size N      Batch size (default: 128)"
            echo "  --fp8                        Enable FP8 quantization (default)"
            echo "  --no-fp8                     Disable FP8 quantization"
            echo "  --cuda-graphs                Enable CUDA graphs"
            echo "  --no-cuda-graphs             Disable CUDA graphs (default)"
            echo "  --mtp                        Enable MTP speculative decoding"
            echo "  --no-mtp                     Disable MTP (default)"
            echo "  --gpu-mem RATIO              GPU memory utilization (default: 0.75)"
            echo "  --text-only                  Use text-only model variant"
            echo "  --kv-cache-dtype TYPE        fp8_e5m2|fp8_e4m3|auto (default: fp8_e5m2)"
            echo "  --max-batched-tokens N       Max batched tokens per scheduler step (default: max-model-len, i.e. one max-length sequence per chunk)"
            echo "  --max-model-len N            Max model context length (default: 32768; covers p95 of MAI dataset, drops ~2.8% of long-tail prompts)"
            echo "  --max-tokens N               Generation budget per request (default: 1024)"
            echo "  --dry-run                    Show config and exit"
            echo "  -h, --help                   Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --exp E002 --backend FLASHINFER --batch 128 --fp8"
            echo "  $0 --exp E005 --cuda-graphs --mtp --gpu-mem 0.75"
            echo "  $0 --exp E012 --backend FLASH_ATTN --no-fp8 --batch 64"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Default max_num_batched_tokens = max_model_len so a single max-length sequence
# (worst case in our dataset is ~79K tokens) prefills in one chunk. Going beyond
# 1× would only matter if we expected to prefill multiple max-length sequences
# per scheduler step, which doesn't happen with this dataset's distribution.
# Override via --max-batched-tokens if you specifically want larger prefill
# chunks for some experiment.
if [ -z "$MAX_BATCHED_TOKENS" ]; then
    MAX_BATCHED_TOKENS="${MAX_MODEL_LEN}"
fi

# Determine model path
if [ "$USE_TEXT_ONLY" = "true" ]; then
    MODEL_PATH="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only"
else
    MODEL_PATH="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it"
fi

ASSISTANT_PATH="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant"

# Determine quantization
if [ "$USE_FP8" = "true" ]; then
    DTYPE="bfloat16"
    QUANTIZATION="fp8"
else
    DTYPE="bfloat16"
    QUANTIZATION="none"
    # NOTE: previously this branch hard-forced KV_CACHE_DTYPE="auto" because
    # we assumed FP8 KV cache was only meaningful with FP8 weights. That's
    # wrong — fp8_e5m2 KV cache works *only* without FP8 checkpoints (vLLM
    # rejects fp8_e5m2 with FP8-quantized weights at engine init). So keep
    # whatever the user passed via --kv-cache-dtype (defaults to "auto" if
    # they didn't pass anything), and let vLLM enforce its own constraints.
fi

# =============================================================================
# Display Configuration
# =============================================================================

echo "=========================================="
echo "Gemma 4 MoE Ablation Experiment Runner"
echo "=========================================="
echo ""
echo "Experiment: ${EXPERIMENT_ID}"
echo ""
echo "Configuration:"
echo "  Model:                ${MODEL_PATH##*/}"
echo "  Backend:              ${BACKEND}"
echo "  Batch Size:           ${BATCH_SIZE}"
echo "  Max Batched Tokens:   ${MAX_BATCHED_TOKENS}"
echo "  Max Model Len:        ${MAX_MODEL_LEN}"
echo "  Max Tokens/Request:   ${MAX_TOKENS}"
echo "  FP8 Quantization:     ${USE_FP8}"
echo "  CUDA Graphs:          ${USE_CUDA_GRAPHS}"
echo "  MTP:                  ${USE_MTP}"
echo "  GPU Memory Util:      ${GPU_MEM_UTIL}"
echo "  KV Cache Dtype:       ${KV_CACHE_DTYPE}"
echo "  Text-Only Model:      ${USE_TEXT_ONLY}"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run mode - configuration displayed, exiting."
    exit 0
fi

# =============================================================================
# Setup
# =============================================================================

OUTPUT_DIR="${SCRIPT_DIR}/experiment_results/${EXPERIMENT_ID}"
mkdir -p "${OUTPUT_DIR}"
# Re-resolve via realpath so symlinks / trailing slashes are normalized; from
# here on every artifact reference is an absolute path immune to cwd changes.
OUTPUT_DIR="$(realpath "${OUTPUT_DIR}")"

echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Activate environment and verify it actually took effect (silent failures of
# `source ... activate` would otherwise leave us using system python with no
# vllm, and we'd only find out partway through the run).
source /root/miniconda3/bin/activate vllm-ablation
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "  ✗ Failed to activate vllm-ablation environment (python3 cannot import vllm)" >&2
    echo "    Check: conda env list  |  source /root/miniconda3/bin/activate vllm-ablation" >&2
    exit 1
fi

# Point FlashInfer's JIT compiler at a CUDA 12.x toolkit that has:
#   (a) an nvcc new enough to understand `--generate-dependencies-with-compile`
#       (FlashInfer 0.6.x emits this flag; the system /usr/bin/nvcc on this
#       box is v10.1 and rejects it)
#   (b) headers whose CUDA major version MATCHES the nvcc — FlashInfer ships
#       cccl/libcudacxx headers built for cu12; using cu13's nvcc against
#       them triggers "CUDA compiler and CUDA toolkit headers are incompatible"
#
# The `vila` env has a complete cu12.8 toolkit (bin/nvcc + include/ + lib/)
# under targets/x86_64-linux/. cu12.8 nvcc compiling for compute_80 is
# runtime-compatible with the cu12.6 driver on this box.
CUDA_HOME_OVERRIDE="/root/miniconda3/envs/vila/targets/x86_64-linux"
if [ -x "${CUDA_HOME_OVERRIDE}/bin/nvcc" ] && [ -f "${CUDA_HOME_OVERRIDE}/include/cuda_runtime.h" ]; then
    export CUDA_HOME="${CUDA_HOME_OVERRIDE}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "  ✓ CUDA_HOME=${CUDA_HOME} (nvcc $(nvcc --version 2>/dev/null | grep release | awk '{print $5,$6}' | tr -d ','))"
else
    echo "  ⚠ No usable cu12 nvcc toolkit; falling back to system nvcc (FlashInfer JIT may fail)" >&2
fi

# =============================================================================
# Environment Configuration
# =============================================================================

echo "Step 1: Configuring environment..."

# Backend
# NOTE: VLLM_USE_FLASHINFER_MOE_FP8 used to be set here for the FlashInfer
# attention path. We do NOT export it on this hardware: FlashInfer's FP8 MoE
# backend requires features (typically sm_90 hopper tensor cores) the A100
# doesn't have, and setting the env var raises
#   NotImplementedError: Found VLLM_USE_FLASHINFER_MOE_FP8=1, but no
#   FlashInfer FP8 MoE backend supports the configuration.
# Without the override, vLLM auto-selects the Marlin FP8 MoE backend, which
# does work on sm_80. FlashInfer is still used for ATTENTION via
# VLLM_ATTENTION_BACKEND=FLASHINFER below.
unset VLLM_USE_FLASHINFER_MOE_FP8

case "${BACKEND}" in
    FLASHINFER)
        export VLLM_ATTENTION_BACKEND=FLASHINFER
        echo "  ✓ FlashInfer backend configured (attention only; MoE FP8 falls back to Marlin on sm_80)"
        ;;
    FLASH_ATTN)
        export VLLM_ATTENTION_BACKEND=FLASH_ATTN
        echo "  ✓ Flash Attention 2 backend configured"
        ;;
    *)
        echo "  ✗ Unknown backend: ${BACKEND}"
        exit 1
        ;;
esac

# Logging
export VLLM_LOGGING_LEVEL=INFO

# Use PyTorch's native top-k/top-p sampler. FlashInfer's sampler kernel is
# JIT-compiled on first use, which requires a working CUDA toolkit on the host
# that matches FlashInfer's bundled cccl headers (cu12). The system /usr/bin/nvcc
# on this box is v10.1, the env's cu13 nvcc mismatches FlashInfer's headers,
# and the available cu12.8 nvcc is missing a flag emitted by FlashInfer.
# Falling back to the torch sampler avoids the whole compile dance — it's
# slightly slower than the FlashInfer sampler in absolute terms, but the
# overhead is a CONSTANT across all experiments, so relative comparisons in
# the ablation stay valid. (See vllm/v1/sample/ops/topk_topp_sampler.py.)
export VLLM_USE_FLASHINFER_SAMPLER=0

echo ""

# =============================================================================
# Log Environment Info
# =============================================================================

echo "Step 2: Logging environment information..."
{
    echo "# Experiment ${EXPERIMENT_ID} - Environment Info"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "## Configuration"
    echo "Model: ${MODEL_PATH}"
    echo "Backend: ${BACKEND}"
    echo "Batch Size: ${BATCH_SIZE}"
    echo "Max Batched Tokens: ${MAX_BATCHED_TOKENS}"
    echo "Max Model Len: ${MAX_MODEL_LEN}"
    echo "Max Tokens/Request: ${MAX_TOKENS}"
    echo "FP8: ${USE_FP8}"
    echo "Quantization: ${QUANTIZATION}"
    echo "KV Cache Dtype: ${KV_CACHE_DTYPE}"
    echo "CUDA Graphs: ${USE_CUDA_GRAPHS}"
    echo "MTP: ${USE_MTP}"
    echo "GPU Memory Util: ${GPU_MEM_UTIL}"
    echo "Text-Only: ${USE_TEXT_ONLY}"
    echo ""
    echo "## Hardware"
    # All experiments are pinned to GPU 0 via CUDA_VISIBLE_DEVICES=0.
    # Query the same GPU explicitly so multi-GPU machines don't mix rows.
    nvidia-smi -i 0 --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
    echo ""
    echo "## Software"
    python --version
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM: dev"
    echo "Attention Backend: ${VLLM_ATTENTION_BACKEND}"
    echo "FlashInfer MoE FP8: ${VLLM_USE_FLASHINFER_MOE_FP8:-disabled}"
    echo ""
    echo "## Git Commit"
    # Subshell so the cd does NOT leak into the parent script. The previous
    # (buggy) version cd'd here without a subshell, which silently broke every
    # subsequent relative-path operation in the script.
    (cd /nvmedata/chenw/vllm-ra && git log -1 --oneline && git rev-parse HEAD)
} > "${OUTPUT_DIR}/environment.txt"

echo "  ✓ Environment info saved"
echo ""

# =============================================================================
# Start Monitoring
# =============================================================================

echo "Step 3: Starting GPU monitor..."
# CRITICAL: -i 0 restricts the trace to GPU 0 (the one running the experiment).
# Without this on a multi-GPU box, the CSV has one row per GPU per timestamp
# with no GPU-index column, so peak/avg memory would include other workloads
# on GPUs 1-3 and corrupt every memory number in summary.md / metrics.json.
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu \
    --format=csv -l 1 > "${OUTPUT_DIR}/gpu_trace.csv" &
MONITOR_PID=$!  # picked up by the EXIT trap above
echo "  ✓ GPU monitor started (PID: ${MONITOR_PID})"
echo ""

# Initial state (snapshot of GPU 0 only).
nvidia-smi -i 0 > "${OUTPUT_DIR}/gpu_initial.txt"

# =============================================================================
# Build Python Command
# =============================================================================

echo "Step 4: Building inference command..."

PYTHON_CMD="python3 run_inference_configurable.py"

PYTHON_ARGS=(
    "--model_path" "${MODEL_PATH}"
    "--max_num_seqs" "${BATCH_SIZE}"
    "--max_num_batched_tokens" "${MAX_BATCHED_TOKENS}"
    "--max_model_len" "${MAX_MODEL_LEN}"
    "--max_tokens" "${MAX_TOKENS}"
    "--gpu_memory_utilization" "${GPU_MEM_UTIL}"
    "--dtype" "${DTYPE}"
    "--output_path" "${OUTPUT_DIR}/output.jsonl"
    "--input_path" "/nvmedata/data/layer1_delta_1k_test.txt"
    "--num_test_samples" "1000"
)

# Quantization + KV cache dtype.
# Always forward --kv_cache_dtype so a user override via --kv-cache-dtype
# reaches Python regardless of whether --fp8 is on. (Previously this branch
# only forwarded KV cache dtype with --fp8, which meant --no-fp8 runs always
# got the Python default — preventing experiments like "BF16 weights with
# FP8 KV cache".)
if [ "$USE_FP8" = "true" ]; then
    PYTHON_ARGS+=("--quantization" "fp8")
fi
PYTHON_ARGS+=("--kv_cache_dtype" "${KV_CACHE_DTYPE}")

# CUDA graphs
if [ "$USE_CUDA_GRAPHS" = "true" ]; then
    PYTHON_ARGS+=("--enable_cuda_graphs")
else
    PYTHON_ARGS+=("--enforce_eager")
fi

# MTP
if [ "$USE_MTP" = "true" ]; then
    PYTHON_ARGS+=("--speculative_model" "${ASSISTANT_PATH}")
    PYTHON_ARGS+=("--num_speculative_tokens" "5")
fi

echo "  Command: ${PYTHON_CMD}"
echo "  Arguments:"
for arg in "${PYTHON_ARGS[@]}"; do
    echo "    $arg"
done
echo ""

# =============================================================================
# Run Experiment
# =============================================================================

echo "Step 5: Running experiment..."
echo "  Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

START_TIME=$(date +%s)

cd /nvmedata/chenw/vllm-ra/examples

# Run the experiment. We want to:
#   - capture the Python exit code (NOT tee's, which is always 0)
#   - keep going through cleanup even on failure (so summary.md still gets written)
# So temporarily disable `set -e` and read PIPESTATUS[0].
set +e
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/nvmedata/chenw/vllm-ra \
time ${PYTHON_CMD} "${PYTHON_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/inference.log"
EXPERIMENT_STATUS=${PIPESTATUS[0]}
set -e

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Duration: ${DURATION} seconds ($((DURATION / 60)) minutes)"
echo "  Status: $([ ${EXPERIMENT_STATUS} -eq 0 ] && echo '✓ Success' || echo '✗ Failed')"
echo ""

# =============================================================================
# Stop Monitoring
# =============================================================================

echo "Step 6: Stopping monitor..."
kill ${MONITOR_PID} 2>/dev/null || true
echo "  ✓ GPU monitor stopped"
echo ""

# Final state
nvidia-smi -i 0 > "${OUTPUT_DIR}/gpu_final.txt"

# =============================================================================
# Analyze Results
# =============================================================================

echo "Step 7: Analyzing results..."

# The analysis block does a lot of best-effort log scraping (grep | sort | tail).
# Under `set -eo pipefail`, an empty grep result anywhere in those pipelines
# would abort the script and leave us without a summary — exactly the wrong
# outcome for a failed experiment, which is when summary.md matters most.
# So suspend errexit/pipefail just for this block.
set +eo pipefail

{
    echo "# Experiment ${EXPERIMENT_ID} - Results"
    echo ""
    echo "## Configuration"
    echo "Experiment ID: ${EXPERIMENT_ID}"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Status: $([ ${EXPERIMENT_STATUS} -eq 0 ] && echo 'Success' || echo 'Failed')"
    echo ""
    echo "Model: ${MODEL_PATH##*/}"
    echo "Backend: ${BACKEND}"
    echo "Batch Size: ${BATCH_SIZE}"
    echo "Max Batched Tokens: ${MAX_BATCHED_TOKENS}"
    echo "Max Model Len: ${MAX_MODEL_LEN}"
    echo "Max Tokens/Request: ${MAX_TOKENS}"
    echo "FP8: ${USE_FP8}"
    echo "CUDA Graphs: ${USE_CUDA_GRAPHS}"
    echo "MTP: ${USE_MTP}"
    echo "GPU Memory Util: ${GPU_MEM_UTIL}"
    echo "KV Cache Dtype: ${KV_CACHE_DTYPE}"
    echo ""
    echo "## Performance"
    echo "Duration: ${DURATION} seconds ($((DURATION / 60)) minutes)"
    echo ""

    if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
        # Pull structured metrics out of metrics.json using python (vllm-ablation env)
        python3 - <<PYEOF
import json, pathlib
m = json.loads(pathlib.Path("${OUTPUT_DIR}/metrics.json").read_text())
c, t, lat = m["counts"], m["throughput"], m["latency"]
ds = m.get("dataset", {})
if ds:
    print("### Dataset (post-filter)")
    print(f"- Input file:               {ds.get('input_path')}")
    print(f"- Chat template applied:    {ds.get('used_chat_template')}")
    print(f"- Token-length threshold:   {ds.get('max_input_tokens_threshold'):,} tokens (max_model_len - max_tokens)")
    print(f"- Total rows seen:          {ds.get('n_total_seen')}")
    print(f"- Loaded for inference:     {ds.get('n_loaded')}")
    print(f"- Skipped (bad format):     {ds.get('n_skipped_format')}")
    print(f"- Skipped (too long):       {ds.get('n_skipped_length')}")
    print()
print("### Counts")
print(f"- Total requests:        {c['total_requests']}")
print(f"- Finished:              {c['finished']}")
print(f"- Failed:                {c['failed']}")
print(f"- Prompt tokens:         {c['total_prompt_tokens']:,}")
print(f"- Output tokens:         {c['total_output_tokens']:,}")
print(f"- Cached tokens:         {c['total_cached_tokens']:,}")
print(f"- Prefix-cache hit rate: {c['prefix_cache_hit_rate'] * 100:.2f}%")
if c.get('finish_reasons'):
    print(f"- Finish reasons:        {c['finish_reasons']}")
if c.get('failed_errors'):
    print(f"- Failed-error types:    {c['failed_errors']}")
print()
print("### Throughput")
print(f"- QPS:                {t['qps']:.4f} req/s")
print(f"- Output tokens/sec:  {t['output_tokens_per_sec']:.2f}")
print(f"- Prompt tokens/sec:  {t['prompt_tokens_per_sec']:.2f}")
print(f"- Total tokens/sec:   {t['total_tokens_per_sec']:.2f}")
print()
print("### Latency distributions (ms)")
print()
print("NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's")
print("RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;")
print("it adds asyncio queue/scheduling slop on the consumer side and will")
print("read slightly higher under heavy concurrency.")
print()
def row(name, d):
    if d.get('count', 0) == 0:
        print(f"| {name} | n=0 | — | — | — | — | — | — |")
        return
    f = lambda k: f"{d[k] * 1000:.2f}"
    print(f"| {name} | n={d['count']} | {f('mean')} | {f('p50')} | {f('p90')} | {f('p95')} | {f('p99')} | {f('max')} |")
print("| metric | n | mean | p50 | p90 | p95 | p99 | max |")
print("|--------|---|------|-----|-----|-----|-----|-----|")
row("TTFT (engine)", lat['ttft_engine_s'])  # canonical
row("TPOT",          lat['tpot_s'])
row("E2E",           lat['e2e_s'])
row("TTFT (client)", lat['ttft_client_s'])  # sanity-check
PYEOF
    else
        echo "(metrics.json not found — the Python entrypoint likely crashed before writing metrics)"
    fi

    echo ""
    echo "## Engine-level snapshots (from vLLM periodic stats in inference.log)"
    if [ -f "${OUTPUT_DIR}/inference.log" ]; then
        # vLLM v1 stats line format (single line, comma-separated):
        #   "Avg prompt throughput: %.1f tokens/s, Avg generation throughput: %.1f tokens/s,
        #    Running: %d reqs, Waiting: %d reqs, ..., GPU KV cache usage: %.1f%%,
        #    Prefix cache hit rate: %.1f%%"
        # See vllm/v1/metrics/loggers.py:217-260.
        KV_PEAK=$(grep -oE "GPU KV cache usage: [0-9.]+%" "${OUTPUT_DIR}/inference.log" | grep -oE "[0-9.]+" | sort -n | tail -1)
        RUN_PEAK=$(grep -oE "Running: [0-9]+ reqs" "${OUTPUT_DIR}/inference.log" | grep -oE "[0-9]+" | sort -n | tail -1)
        WAIT_PEAK=$(grep -oE "Waiting: [0-9]+ reqs" "${OUTPUT_DIR}/inference.log" | grep -oE "[0-9]+" | sort -n | tail -1)
        LAST_STATS=$(grep -E "Avg prompt throughput.*Avg generation throughput" "${OUTPUT_DIR}/inference.log" | tail -1)
        echo "- Peak KV cache usage:    ${KV_PEAK:-N/A}%"
        echo "- Peak running requests:  ${RUN_PEAK:-N/A}"
        echo "- Peak waiting requests:  ${WAIT_PEAK:-N/A}"
        if [ -n "$LAST_STATS" ]; then
            echo "- Last engine stats line:"
            echo "${LAST_STATS}" | sed 's/^/    /'
        fi

        # Speculative-decoding stats (only present when --mtp is enabled).
        # vLLM emits per-window lines like:
        #   "SpecDecoding metrics: Mean acceptance length: 2.34, Accepted throughput: ...
        #    Avg Draft acceptance rate: 61.7%"
        # See vllm/v1/spec_decode/metrics.py:101-117. We summarize across all
        # emitted windows so a single experiment yields a stable summary number.
        if grep -q "SpecDecoding metrics:" "${OUTPUT_DIR}/inference.log"; then
            echo ""
            echo "### Speculative decoding (MTP)"
            # Mean of "Avg Draft acceptance rate" across windows = overall draft acceptance
            grep -oE "Avg Draft acceptance rate: [0-9.]+%" "${OUTPUT_DIR}/inference.log" \
                | grep -oE "[0-9.]+" \
                | awk '
                    { sum += $1; n += 1; if ($1 > max) max = $1; if (n == 1 || $1 < min) min = $1 }
                    END {
                        if (n > 0) printf("- Avg draft acceptance: mean=%.2f%% min=%.2f%% max=%.2f%% (windows=%d)\n", sum/n, min, max, n);
                    }'
            # Mean of "Mean acceptance length" — including bonus token (~ effective tokens/iter)
            grep -oE "Mean acceptance length: [0-9.]+" "${OUTPUT_DIR}/inference.log" \
                | grep -oE "[0-9.]+" \
                | awk '{ sum += $1; n += 1 } END { if (n>0) printf("- Mean acceptance length: %.2f (avg across %d windows)\n", sum/n, n) }'
            # Last reported throughputs
            LAST_SPEC=$(grep "SpecDecoding metrics:" "${OUTPUT_DIR}/inference.log" | tail -1)
            if [ -n "$LAST_SPEC" ]; then
                echo "- Last SpecDecoding line:"
                echo "${LAST_SPEC}" | sed 's/^/    /'
            fi
        fi
    fi

    echo ""
    echo "## GPU memory & utilization (GPU 0, sampled at 1 Hz)"
    if [ -f "${OUTPUT_DIR}/gpu_trace.csv" ]; then
        # Parse gpu_trace.csv (nvidia-smi --format=csv) and merge structured
        # numbers into metrics.json under a "gpu" subsection. Doing this here
        # rather than in the Python entrypoint keeps the entrypoint focused on
        # the inference workload — the trace is bash-owned (background process).
        python3 - <<PYEOF
import csv, json, pathlib

trace = pathlib.Path("${OUTPUT_DIR}/gpu_trace.csv")
metrics_path = pathlib.Path("${OUTPUT_DIR}/metrics.json")

used_mib, free_mib, util_gpu, util_mem = [], [], [], []
with trace.open() as f:
    # nvidia-smi puts a space after each comma in CSV; skipinitialspace strips it.
    reader = csv.reader(f, skipinitialspace=True)
    header = next(reader, None)
    if header is None:
        print("(gpu_trace.csv is empty)")
        raise SystemExit(0)
    # Expected columns (header includes units): timestamp, memory.used [MiB],
    # memory.free [MiB], utilization.gpu [%], utilization.memory [%], temperature.gpu
    def parse_int(s):
        # "12345 MiB" -> 12345 ;  "85 %" -> 85
        s = s.strip()
        if not s:
            return None
        return int(s.split()[0])
    for row in reader:
        if len(row) < 6:
            continue
        u = parse_int(row[1]); fr = parse_int(row[2])
        ug = parse_int(row[3]); um = parse_int(row[4])
        if u is not None:  used_mib.append(u)
        if fr is not None: free_mib.append(fr)
        if ug is not None: util_gpu.append(ug)
        if um is not None: util_mem.append(um)

if not used_mib:
    print("(no usable samples in gpu_trace.csv)")
    raise SystemExit(0)

total_mib = used_mib[0] + free_mib[0]  # used + free is constant for this GPU
peak_mib = max(used_mib)
avg_mib = sum(used_mib) / len(used_mib)
baseline_mib = used_mib[0]   # at trace start (typically just after monitor PID started, before model load)
end_mib = used_mib[-1]       # at trace end

gpu_section = {
    "samples": len(used_mib),
    "total_mib": total_mib,
    "total_gib": round(total_mib / 1024.0, 2),
    "baseline_mib": baseline_mib,
    "baseline_gib": round(baseline_mib / 1024.0, 2),
    "peak_mib": peak_mib,
    "peak_gib": round(peak_mib / 1024.0, 2),
    "peak_pct_of_total": round(100.0 * peak_mib / total_mib, 2),
    "avg_mib": round(avg_mib, 1),
    "avg_gib": round(avg_mib / 1024.0, 2),
    "min_mib": min(used_mib),
    "end_mib": end_mib,
    "end_gib": round(end_mib / 1024.0, 2),
    "delta_peak_minus_baseline_mib": peak_mib - baseline_mib,
    "delta_peak_minus_baseline_gib": round((peak_mib - baseline_mib) / 1024.0, 2),
    "util_gpu_peak_pct": max(util_gpu) if util_gpu else None,
    "util_gpu_avg_pct": round(sum(util_gpu) / len(util_gpu), 1) if util_gpu else None,
    "util_mem_peak_pct": max(util_mem) if util_mem else None,
    "util_mem_avg_pct": round(sum(util_mem) / len(util_mem), 1) if util_mem else None,
}

# Markdown for summary.md
print(f"- Total GPU memory:        {gpu_section['total_gib']:.2f} GiB ({total_mib:,} MiB)")
print(f"- Baseline (trace start):  {gpu_section['baseline_gib']:.2f} GiB")
print(f"- Peak used:               {gpu_section['peak_gib']:.2f} GiB  ({gpu_section['peak_pct_of_total']:.1f}% of total)")
print(f"- Average used:            {gpu_section['avg_gib']:.2f} GiB")
print(f"- End (trace end):         {gpu_section['end_gib']:.2f} GiB")
print(f"- Delta (peak - baseline): {gpu_section['delta_peak_minus_baseline_gib']:.2f} GiB  (model load + inference working set)")
print(f"- GPU compute util:        peak={gpu_section['util_gpu_peak_pct']}%  avg={gpu_section['util_gpu_avg_pct']}%")
print(f"- GPU memory-bw util:      peak={gpu_section['util_mem_peak_pct']}%  avg={gpu_section['util_mem_avg_pct']}%")

# Merge into metrics.json so downstream comparison scripts see GPU memory
# as a first-class ablation metric (not just text in summary.md).
if metrics_path.exists():
    m = json.loads(metrics_path.read_text())
    m["gpu"] = gpu_section
    metrics_path.write_text(json.dumps(m, indent=2))
PYEOF
    fi

    echo ""
    echo "## Files"
    echo "- Environment:           ${OUTPUT_DIR}/environment.txt"
    echo "- Inference log:         ${OUTPUT_DIR}/inference.log"
    echo "- GPU trace (1Hz):       ${OUTPUT_DIR}/gpu_trace.csv"
    echo "- Generations:           ${OUTPUT_DIR}/output.jsonl"
    echo "- Per-request metrics:   ${OUTPUT_DIR}/per_request_metrics.jsonl"
    echo "- Aggregate metrics:     ${OUTPUT_DIR}/metrics.json"
    echo "- GPU initial:           ${OUTPUT_DIR}/gpu_initial.txt"
    echo "- GPU final:             ${OUTPUT_DIR}/gpu_final.txt"
    echo ""
    echo "## Errors"
    ERROR_COUNT=$(grep -ic "error\|exception\|traceback" "${OUTPUT_DIR}/inference.log" 2>/dev/null || echo "0")
    if [ ${ERROR_COUNT} -gt 0 ]; then
        echo "⚠ Found ${ERROR_COUNT} error mentions"
        echo ""
        echo "Sample errors:"
        grep -i "error\|exception" "${OUTPUT_DIR}/inference.log" | head -10
    else
        echo "✓ No errors detected"
    fi

} > "${OUTPUT_DIR}/summary.md"

# Restore strict mode after the best-effort analysis block.
set -eo pipefail

cat "${OUTPUT_DIR}/summary.md"
echo ""

# =============================================================================
# Final Summary
# =============================================================================

echo "=========================================="
if [ ${EXPERIMENT_STATUS} -eq 0 ]; then
    echo "✓ Experiment ${EXPERIMENT_ID} Complete!"
else
    echo "✗ Experiment ${EXPERIMENT_ID} Failed"
fi
echo "=========================================="
echo ""
echo "Results: ${OUTPUT_DIR}/"
echo ""
echo "Quick commands:"
echo "  View summary:     cat ${OUTPUT_DIR}/summary.md"
echo "  View full log:    less ${OUTPUT_DIR}/inference.log"
echo "  Check errors:     grep -i error ${OUTPUT_DIR}/inference.log"
echo "  Plot GPU usage:   python3 plot_gpu_trace.py ${OUTPUT_DIR}/gpu_trace.csv"
echo ""

exit ${EXPERIMENT_STATUS}
