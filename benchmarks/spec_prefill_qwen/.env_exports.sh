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

# Target model (Qwen3-8B). TODO: fill in with this node's real downloaded
# snapshot path once available -- see REPRODUCE.md step 3 for download
# instructions. Verify it's actually present before trusting this value:
#   ls -la /scratch/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/*/
#   du -sh /scratch/hf_cache/hub/models--Qwen--Qwen3-8B/
export QWEN3_MODEL_PATH=/scratch/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 

# Speculator model (Qwen3-1.7B, SpecPrefill-specific -- not used by
# gemma4_moe_benchmarks or evaluation_pipeline).
# step 3. Same "no hub/ nesting" and "verify it's still present on this
# node" caveats as QWEN3_MODEL_PATH above apply here too.
export QWEN3_1_7B_MODEL_PATH=/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

# NOTE: no "hub/" in these paths -- `hf download --cache-dir X` places
# snapshots directly under X, not X/hub/ (that hub/ nesting is what
# HF_HOME-based lazy fetching via transformers/huggingface_hub's default
# cache would use instead). Re-verify this path if a model is ever
# re-downloaded with a different tool/flag combination.

# Unlike the sibling Gemma4 pipeline's .env_exports.sh, there is no
# QWEN3_TEXT_ONLY_MODEL_PATH here -- Qwen3-8B/1.7B have no multimodal
# checkpoint variant (no vision/audio encoder cache to strip), so that
# concern doesn't apply.
