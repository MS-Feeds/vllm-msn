source /opt/conda/etc/profile.d/conda.sh
conda activate vllm-ablation
export HF_HOME=/scratch/hf_cache
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN in shell before sourcing this file}
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
export GEMMA4_MODEL_PATH=/scratch/hf_cache/models--google--gemma-4-26B-A4B-it/snapshots/5305c1e72ea29c01f31a81230d52b375ba88b409
export GEMMA4_TEXT_ONLY_MODEL_PATH=/scratch/hf_cache/gemma-4-26B-A4B-it-text-only
export GEMMA4_ASSISTANT_MODEL_PATH=/scratch/hf_cache/models--google--gemma-4-26B-A4B-it-assistant/snapshots/d8888482e6d3cc4637fa8b25c1033cd3573b88ba
# NOTE: no "hub/" in these paths -- `hf download --cache-dir X` (the CLI
# used to fetch these after the 2026-07-14 node wipe) places snapshots
# directly under X, not X/hub/ (that hub/ nesting is what HF_HOME-based
# lazy fetching via transformers/huggingface_hub's default cache would
# use instead). Re-verify this path if the model is ever re-downloaded
# with a different tool/flag combination -- see chat history around
# 2026-07-14 for the debugging session that found this mismatch.