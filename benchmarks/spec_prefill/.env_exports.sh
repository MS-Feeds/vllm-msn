source /opt/conda/etc/profile.d/conda.sh
conda activate vllm-ablation
# Precompiled vLLM binaries need CUDA 13 runtime libs on LD_LIBRARY_PATH
# (see ../gemma4_moe_benchmarks/EXPERIMENT_PLAN.md's "Runtime environment"
# section -- that pipeline's run_experiments.sh sets this automatically,
# this one doesn't have an equivalent wrapper yet, so it's set here instead).
# $CONDA_PREFIX is set by `conda activate` above, so this must come after it.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export HF_HOME=/scratch/hf_cache
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN in shell before sourcing this file}
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

# Confirmed on real hardware (2026-07-28, via the Qwen3 port of this
# pipeline): vllm_patch/model_runner.py's SpecPrefillGPUModelRunner
# subclasses the legacy ("v1") GPUModelRunner at
# vllm/v1/worker/gpu_model_runner.py -- but this fork now defaults to a
# newer "v2" runner (vllm/v1/worker/gpu/model_runner.py) for dense,
# non-quantized architectures (vllm/config/vllm.py's use_v2_model_runner
# property excludes MoE models unconditionally). Gemma-4-26B-A4B-it is MoE,
# so this pipeline's target model was never actually at risk -- but forced
# here defensively too, in case the speculator (Gemma-4-E2B-it, not
# independently confirmed MoE or dense here) or a future model swap ever
# triggers the v2 default, which surfaces as an AttributeError
# (num_speculative_steps) at engine startup, not a subtle correctness bug --
# see spec_prefill_qwen/.env_exports.sh for the full failure mode.
export VLLM_USE_V2_MODEL_RUNNER=0

# Compile inductor kernels IN-PROCESS, not via a subprocess pool.
#
# Required whenever the target runs with tensor parallelism, and the failure
# is not obviously about compilation:
#
#   AssertionError: daemonic processes are not allowed to have children
#
# raised from `torch._inductor.async_compile.AsyncCompile.wakeup()` during
# `profile_run`. The chain: with TP > 1 vLLM uses `MultiprocExecutor`, which
# spawns its `WorkerProc`s as DAEMONIC processes; `VocabParallelEmbedding.
# forward` then calls `get_masked_input_and_mask`, which is decorated
# `@torch.compile(dynamic=True, ...)` and runs ONLY when `tp_size > 1`;
# inductor tries to start its own compile-worker pool from inside that
# daemonic process, and Python refuses.
#
# `enforce_eager=True` does NOT prevent this. That disables vLLM's own
# compilation and CUDA graphs, not a bare `@torch.compile` decorator inside a
# layer -- which is why this never appeared before TP was used here.
#
# Costs nothing in this pipeline: with enforce_eager on both engines that
# masking helper is about the only compiled function on the path, and it is a
# trivial elementwise op over token ids.
export TORCHINDUCTOR_COMPILE_THREADS=1

# Target model. This exact snapshot path was valid on a prior node
# ("node-0", per the 2026-07-14 rebuild this was originally copied from --
# see ../gemma4_moe_benchmarks/.env_exports.sh). On a different/fresh node,
# verify it's actually present before trusting this value:
#   ls -la /scratch/hf_cache/models--google--gemma-4-26B-A4B-it/snapshots/*/
#   du -sh /scratch/hf_cache/models--google--gemma-4-26B-A4B-it/   # expect ~49G
# If missing, download it (see REPRODUCE.md step 3) and update this path.
export GEMMA4_MODEL_PATH=/scratch/hf_cache/models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52

# Text-only variant of the target model (vision/audio/video weights + config
# stripped -- see ../../examples/create_text_only_model.py). Confirmed on
# real hardware (2026-07-24): the full multimodal checkpoint above loads as
# Gemma4ForConditionalGeneration, which reserves a multimodal "encoder cache"
# (sized off max_num_batched_tokens) even for pure-text workloads like
# LongBench v2 -- this ate ~12GB of GPU memory that would otherwise go to KV
# cache, exactly the difference between this pipeline's available KV cache
# and gemma4_moe_benchmarks' (which always uses model_variant="text_only").
# Without vision_config, this should resolve as plain Gemma4ForCausalLM
# instead and skip that reservation entirely -- confirm the startup log says
# "Resolved architecture: Gemma4ForCausalLM" (not
# "Gemma4ForConditionalGeneration") and no "Encoder cache" line appears.
# Not yet generated on this node -- run:
#   python3 ../../examples/create_text_only_model.py \
#       --model_path "$GEMMA4_MODEL_PATH" \
#       --output_path /scratch/hf_cache/gemma-4-26B-A4B-it-text-only
export GEMMA4_TEXT_ONLY_MODEL_PATH=/scratch/hf_cache/gemma-4-26B-A4B-it-text-only

# Speculator model (SpecPrefill-specific -- not used by gemma4_moe_benchmarks
# or evaluation_pipeline, see that pipeline's own .env_exports.sh for its NOTE
# on this). Already downloaded; this path was filled in when .env_exports.sh
# was moved into this directory. Same "no hub/ nesting" and "verify it's still
# present on this node" caveats as GEMMA4_MODEL_PATH above apply here too.
export GEMMA4_E2B_MODEL_PATH=/scratch/hf_cache/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7

# Gemma-4-31B (DENSE) -- the target for the multi-turn port's gate phase (see
# ../spec_prefill_multi_turn/, and the porting plan's "Decided scope"). Not
# used by THIS pipeline's own P001-P006 sweep, which stays on the 26B-A4B
# above; it lives here because this is where the gate script
# (verify_sliding_window_hypothesis.py) is run from.
#
# TODO: fill in with this node's real snapshot path once downloaded (gated HF
# repo -- request access and export HF_TOKEN first, see REPRODUCE.md step 3).
# Verify before trusting, same caveat as every path in this file:
#   ls -la /scratch/hf_cache/models--google--gemma-4-31B-it/snapshots/*/
#
# Why dense rather than the MoE 26B-A4B for the port: it keeps the FLOP
# accounting closer to the Llama model the multi-turn pipeline already has
# (no parallel dense-MLP + top-8-of-128 routing to model), while still
# exercising every property that actually blocks the port -- interleaved
# sliding-window attention, heterogeneous head dims, and cross-layer KV
# sharing are all still present.
#
# Checked so it does not get re-litigated: a DENSE target does NOT put this
# at risk of the newer "v2" model runner, even though that runner's gate
# mentions dense/non-quantized. `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`
# (vllm/config/vllm.py) is `{"Qwen3ForCausalLM"}` and
# `_is_default_v2_model_runner_model` returns False for anything outside it
# BEFORE reaching the dense/MoE check. VLLM_USE_V2_MODEL_RUNNER=0 above stays
# as a defensive pin, not a load-bearing one.
# Resolved from the snapshot directory rather than hardcoded.
#
# A hand-written hash is easy to malform, and the failure is opaque: a single
# missing slash produced ".../snapshots842da37...", which huggingface_hub then
# tried to read as a REPO ID and rejected with
# `HFValidationError: Repo id must be in the form 'repo_name' or
# 'namespace/repo_name'` -- a message that says nothing about the real problem.
# A stale hash after a repo update fails just as unhelpfully. Globbing the
# snapshots directory cannot get either wrong.
#
# Unset (not empty-and-silent) when the checkpoint is absent, so a run fails
# on a missing variable rather than on an empty path that looks like a
# relative filename.
_resolve_hf_snapshot() {
    local cache_dir="$1"
    local resolved
    resolved=$(ls -d "$cache_dir"/snapshots/*/ 2>/dev/null | head -1)
    if [ -z "$resolved" ]; then
        echo "[.env_exports] WARNING: no snapshot under $cache_dir/snapshots" >&2
        return 1
    fi
    # Strip the trailing slash: from_pretrained accepts either, but the bare
    # form is what every log line and error message in this pipeline prints.
    echo "${resolved%/}"
}

export GEMMA4_31B_MODEL_PATH=$(_resolve_hf_snapshot /scratch/hf_cache/models--google--gemma-4-31B-it)

# NOTE: no "hub/" in these paths -- `hf download --cache-dir X` places
# snapshots directly under X, not X/hub/ (that hub/ nesting is what
# HF_HOME-based lazy fetching via transformers/huggingface_hub's default
# cache would use instead). Re-verify this path if a model is ever
# re-downloaded with a different tool/flag combination.
