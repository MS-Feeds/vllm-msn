source /opt/conda/etc/profile.d/conda.sh
conda activate vllm-ablation
export HF_HOME=/scratch/hf_cache
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN in shell before sourcing this file}
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

# Target model. This exact snapshot path was valid on a prior node
# ("node-0", per the 2026-07-14 rebuild this was originally copied from --
# see ../gemma4_moe_benchmarks/.env_exports.sh). On a different/fresh node,
# verify it's actually present before trusting this value:
#   ls -la /scratch/hf_cache/models--google--gemma-4-26B-A4B-it/snapshots/*/
#   du -sh /scratch/hf_cache/models--google--gemma-4-26B-A4B-it/   # expect ~49G
# If missing, download it (see REPRODUCE.md step 3) and update this path.
export GEMMA4_MODEL_PATH=/scratch/hf_cache/models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52

# Speculator model (SpecPrefill-specific -- not used by gemma4_moe_benchmarks
# or evaluation_pipeline). Not yet downloaded as of this writing -- run the
# `hf download` command in REPRODUCE.md step 3, then uncomment and fill in
# the real snapshot path below.
export GEMMA4_E2B_MODEL_PATH=/scratch/hf_cache/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7

# NOTE: no "hub/" in these paths -- `hf download --cache-dir X` places
# snapshots directly under X, not X/hub/ (that hub/ nesting is what
# HF_HOME-based lazy fetching via transformers/huggingface_hub's default
# cache would use instead). Re-verify this path if a model is ever
# re-downloaded with a different tool/flag combination.
