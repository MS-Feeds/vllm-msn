source /opt/conda/etc/profile.d/conda.sh
conda activate vllm-ablation
# Precompiled vLLM binaries need CUDA 13 runtime libs on LD_LIBRARY_PATH
# (see ../gemma4_moe_benchmarks/EXPERIMENT_PLAN.md's "Runtime environment"
# section -- that pipeline's run_experiments.sh sets this automatically,
# this one doesn't have an equivalent wrapper yet, so it's set here instead).
# $CONDA_PREFIX is set by `conda activate` above, so this must come after it.
# NOT hardcoded to python3.10 (unlike the sibling Llama/dense-Qwen3 ports'
# copies of this line) -- this pipeline's conda env needs Python >=3.11
# (../rlm/pyproject.toml's requires-python, since run_arm.py/run_all_arms.py
# import the rlm package and vllm in the SAME process), so the real
# site-packages path varies. Resolved from the active interpreter instead.
_PYTHON_TAG="python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/$_PYTHON_TAG/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
unset _PYTHON_TAG
export HF_HOME=/scratch/hf_cache
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN in shell before sourcing this file}
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

# Unlike the sibling dense-Qwen3 pipeline (spec_prefill_qwen/.env_exports.sh),
# this override is NOT load-bearing here: both Qwen3-Coder-480B-A35B and
# Qwen3-Coder-30B-A3B are MoE checkpoints (Qwen3MoeForCausalLM), and
# vllm/config/vllm.py's use_v2_model_runner property requires `not
# model_config.is_moe` in addition to the architecture being in
# DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES ({"Qwen3ForCausalLM"} only, confirmed
# by reading vllm/config/vllm.py:68 directly) -- so both models already fall
# back to the v1 runner vllm_patch/model_runner.py is built against
# unconditionally, the same reason the sibling Gemma4/Llama pipelines never
# needed this override. Kept anyway as an explicit, harmless pin to the
# runner vllm_patch/ actually targets, consistent with spec_prefill_llama's
# own rationale for keeping a technically-redundant pin.
export VLLM_USE_V2_MODEL_RUNNER=0

# Target model (Qwen3-Coder-480B-A35B-Instruct, MoE -- ~480B total/~35B
# active params, 62 layers, 96 Q heads/8 KV heads, 160 experts/8 active,
# native context 262144). TODO: fill in with this node's real downloaded
# snapshot path once available -- see REPRODUCE.md step 3 for download
# instructions. Verify it's actually present before trusting this value:
#   ls -la /scratch/hf_cache/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct/snapshots/*/
#   du -sh /scratch/hf_cache/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct/
# At full BF16 (~960GB) this checkpoint does NOT fit on a single GPU, or
# even comfortably across most 8-GPU nodes -- the target engine must be
# constructed with tensor_parallel_size > 1 (see
# ../rlm_specprefill/target_stage/vllm_offline_engine.py's
# tensor_parallel_size parameter / --target-tensor-parallel-size CLI flag).
# For serving on 8x80GB GPUs, prefer a quantized checkpoint instead of BF16
# -- confirmed available: QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ
# (4-bit AWQ, ~236GB total, i.e. ~30GB/GPU shard at tensor_parallel_size=8).
# That model card documents (as of the version checked here) REQUIRING
# --enable-expert-parallel at tensor-parallel-size 8 ("otherwise, the expert
# tensors cannot be evenly split across tensor parallel ranks") -- pass
# --target-enable-expert-parallel to runner/run_arm.py or
# validate_runner_integration.py when using this checkpoint at TP>1 (96 Q
# heads / 8 KV heads means TP in {1,2,4,8} all divide evenly for this
# architecture; TP=8 is what the checkpoint's own card was tested against).
# Also documented on that card: vLLM >=0.9.2, transformers >=4.51.0,
# trust_remote_code=True (already always passed by this port), non-thinking
# mode only (matches predict_longbench_v2.py's enable_thinking=False
# assumption -- still worth re-verifying against the real chat template,
# see that file's render_chat docstring).
export QWEN3_CODER_480B_MODEL_PATH=/scratch/hf_cache/models--QuantTrio--Qwen3-Coder-480B-A35B-Instruct-AWQ/snapshots/9ce3eaa67fe88609afec235117e97eb03d9b3cda

# Starting tensor-parallel size for the target: 4, not the AWQ card's own
# tested 8 -- at TP=8 the target occupies every GPU on an 8-GPU node,
# leaving none free for the speculator (kept unquantized, ~60GB, per
# vllm_patch/proposer.py's design) without exceeding 80GB when sharing a
# GPU with a target shard. TP=4 uses 4 GPUs for the target (~59GB/GPU for
# the AWQ checkpoint, repo-size/4), leaves a dedicated GPU for the
# speculator among the remaining 4, and 3 GPUs spare -- see
# EXPERIMENT_PLAN.md's "Resource requirements" for the full trade-off.
# Both 4 and 8 are valid divisors of this model's 96 Q-heads/8 KV-heads
# (confirmed against Qwen3MoeAttention.__init__'s own assertions), so
# switching back to 8 later (e.g. once the speculator is also quantized to
# fit alongside a target shard) is just a matter of overriding this.
# Consumed by run_arm.py/run_all_arms.py/calibration scripts'
# --target-tensor-parallel-size CLI default, and by
# validate_runner_integration.py's flag of the same name.
export TARGET_TENSOR_PARALLEL_SIZE=4

# Speculator model (Qwen3-Coder-30B-A3B-Instruct, MoE -- ~30B total/~3B
# active params, SpecPrefill-specific -- not used by gemma4_moe_benchmarks or
# evaluation_pipeline). Same "fill in once downloaded on this node" caveats
# as QWEN3_CODER_480B_MODEL_PATH above apply here too. Unlike the target,
# this fits standalone on a single GPU in BF16 -- SpecPrefillProposer still
# only supports tensor_model_parallel_size=1 for the speculator (see
# vllm_patch/proposer.py's _ensure_distributed_environment docstring; TP/PP
# > 1 for the speculator remains out of scope for this pass).
#export QWEN3_CODER_30B_MODEL_PATH=/scratch/hf_cache/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120
export QWEN3_CODER_30B_MODEL_PATH=/scratch/hf_cache/models--cyankiwi--Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit/snapshots/4bd30395b72ea6045edd04806c4fea448d4467b3

# Unlike the sibling Gemma4 pipeline's .env_exports.sh, there is no
# QWEN3_CODER_TEXT_ONLY_MODEL_PATH here -- these are text-only checkpoints
# (no vision/audio encoder cache to strip), so that concern doesn't apply.
