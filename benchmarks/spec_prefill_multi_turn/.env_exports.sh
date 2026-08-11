source /opt/conda/etc/profile.d/conda.sh
conda activate vllm-ablation
# Precompiled vLLM binaries need CUDA 13 runtime libs on LD_LIBRARY_PATH --
# same as ../spec_prefill_llama/.env_exports.sh, this pipeline needs its own
# copy since these exports aren't shared across benchmark directories.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export HF_HOME=/scratch/hf_cache
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN in shell before sourcing this file}
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

# Same reasoning as ../spec_prefill_llama/.env_exports.sh: LlamaForCausalLM
# is not in DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES (vllm/config/vllm.py),
# so this fork would not auto-select the newer "v2" runner for it anyway --
# kept as an explicit, harmless pin to the runner vllm_patch/model_runner.py
# and vllm_patch/speculator_worker.py are actually built against (both
# subclass the "v1" GPUModelRunner), rather than relying on that
# architecture list never growing to include Llama.
export VLLM_USE_V2_MODEL_RUNNER=0

# Target model (Llama-3.1-8B-Instruct). Reuse the same downloaded snapshot
# path as ../spec_prefill_llama/.env_exports.sh if you already have one --
# TODO: fill in with this node's real path (gated HF repo -- request access
# and export HF_TOKEN before downloading, see REPRODUCE.md step 2).
#   ls -la /scratch/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/
export LLAMA31_8B_MODEL_PATH=/scratch/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/REPLACE_ME

# Speculator model (Llama-3.2-1B-Instruct). Same "fill in once downloaded"
# caveat as above (also gated on Hugging Face).
export LLAMA32_1B_MODEL_PATH=/scratch/hf_cache/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/REPLACE_ME
