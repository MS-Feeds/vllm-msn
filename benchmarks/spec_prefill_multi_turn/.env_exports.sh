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
export LLAMA31_8B_MODEL_PATH=/scratch/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

# Speculator model (Llama-3.2-1B-Instruct). Same "fill in once downloaded"
# caveat as above (also gated on Hugging Face).
export LLAMA32_1B_MODEL_PATH=/scratch/hf_cache/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6

# Mid-size scorer (Llama-3.2-3B-Instruct) -- NOT part of the published
# experiment matrix. It exists for the scorer-capacity probe in
# ACCURACY_IMPROVEMENTS.md §1.6: ORACLE-k20 (the 8B scoring its own
# attention) recovers 17.0 of the 25.0-point scbench_kv degradation, and the
# §1.1/§1.2 sweep ruled out reaching that by changing how the 1B's attention
# is aggregated. Scoring with 3B instead answers whether the gap is a smooth
# function of scorer capacity -- i.e. whether the fix is "use a bigger draft
# model" (a FLOPs trade, priceable with flops_model.py) rather than better
# scoring math.
#
#   python3 predict_scbench.py --exp ORACLE-k20 --scbench-config scbench_kv --oracle-scorer-model $LLAMA32_3B_MODEL_PATH --scorer-prefill-chunk-tokens 32768
#
# Same gated-repo + "fill in the real snapshot hash once downloaded" caveat
# as the two above -- the hash below is a PLACEHOLDER, not this node's real
# path. Find it with:
#   ls -la /scratch/hf_cache/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/*/
export LLAMA32_3B_MODEL_PATH=/scratch/hf_cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
