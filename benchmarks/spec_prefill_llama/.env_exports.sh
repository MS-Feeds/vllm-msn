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

# vllm_patch/model_runner.py's SpecPrefillGPUModelRunner subclasses the
# legacy ("v1") GPUModelRunner at vllm/v1/worker/gpu_model_runner.py. This
# fork's use_v2_model_runner property (vllm/config/vllm.py) only
# auto-selects the newer "v2" runner for architectures listed in
# DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES -- currently just
# {"Qwen3ForCausalLM"} (confirmed by reading vllm/config/vllm.py:68 directly)
# -- so LlamaForCausalLM is not affected and already defaults to v1 without
# this override. Kept anyway as an explicit, harmless pin to the runner
# vllm_patch/ is actually built against, rather than relying on that
# architecture list never growing to include Llama.
export VLLM_USE_V2_MODEL_RUNNER=0

# Target model (Llama-3.1-8B-Instruct). TODO: fill in with this node's real
# downloaded snapshot path once available -- see REPRODUCE.md step 3 for
# download instructions (meta-llama/Llama-3.1-8B-Instruct is a gated repo on
# Hugging Face; request access and export HF_TOKEN before downloading).
# Verify it's actually present before trusting this value:
#   ls -la /scratch/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/
#   du -sh /scratch/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/
export LLAMA31_8B_MODEL_PATH=/scratch/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/REPLACE_WITH_REAL_SNAPSHOT_HASH

# Speculator model (Llama-3.2-1B-Instruct, SpecPrefill-specific -- not used
# by gemma4_moe_benchmarks or evaluation_pipeline). Same "fill in once
# downloaded on this node" caveats as LLAMA31_8B_MODEL_PATH above apply here
# too (also gated on Hugging Face).
export LLAMA32_1B_MODEL_PATH=/scratch/hf_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/REPLACE_WITH_REAL_SNAPSHOT_HASH

# Unlike the sibling Gemma4 pipeline's .env_exports.sh, there is no
# LLAMA_TEXT_ONLY_MODEL_PATH here -- Llama-3.1-8B/3.2-1B (text-only
# checkpoints, not the 3.2 vision variants) have no multimodal encoder cache
# to strip, so that concern doesn't apply.
