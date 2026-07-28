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

# Confirmed on real hardware (2026-07-28): vllm_patch/model_runner.py's
# SpecPrefillGPUModelRunner subclasses the legacy ("v1") GPUModelRunner at
# vllm/v1/worker/gpu_model_runner.py -- but this fork now defaults to a
# newer, refactored "v2" runner (vllm/v1/worker/gpu/model_runner.py) for
# dense, non-quantized architectures in DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES
# (vllm/config/vllm.py's use_v2_model_runner property), which Qwen3-8B is.
# vllm_patch/worker.py unconditionally swaps in our v1-based runner
# regardless, so the surrounding engine code (e.g. gpu_worker.py's
# warmup_kernels) ends up calling v2-only APIs (num_speculative_steps) on a
# v1 object -- AttributeError at engine startup. Gemma-4-26B-A4B-it never
# hit this because it's MoE, which the v2 default excludes unconditionally
# -- this is a genuine Qwen3-specific trigger, not a porting oversight.
# Force v1 until vllm_patch/ is ported to the v2 runner API (out of scope
# for now -- see validate_runner_integration.py's docstring history for
# what's already been validated against v1's behavior).
export VLLM_USE_V2_MODEL_RUNNER=0

# Target model (Qwen3-8B). TODO: fill in with this node's real downloaded
# snapshot path once available -- see REPRODUCE.md step 3 for download
# instructions. Verify it's actually present before trusting this value:
#   ls -la /scratch/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/*/
#   du -sh /scratch/hf_cache/hub/models--Qwen--Qwen3-8B/
export QWEN3_MODEL_PATH=/scratch/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 

# Speculator model (Qwen3-1.7B, SpecPrefill-specific -- not used by
# gemma4_moe_benchmarks or evaluation_pipeline).
# node" caveats as QWEN3_MODEL_PATH above apply here too.
export QWEN3_1_7B_MODEL_PATH=/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

# Unlike the sibling Gemma4 pipeline's .env_exports.sh, there is no
# QWEN3_TEXT_ONLY_MODEL_PATH here -- Qwen3-8B/1.7B have no multimodal
# checkpoint variant (no vision/audio encoder cache to strip), so that
# concern doesn't apply.
