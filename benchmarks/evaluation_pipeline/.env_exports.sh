source /opt/conda/etc/profile.d/conda.sh
conda activate vllm-ablation
export HF_HOME=/scratch/hf_cache
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN in shell before sourcing this file}
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
export GEMMA4_MODEL_PATH=
export GEMMA4_TEXT_ONLY_MODEL_PATH=
export GEMMA4_ASSISTANT_MODEL_PATH=