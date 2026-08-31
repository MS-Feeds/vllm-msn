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

# Belt-and-braces: TORCHINDUCTOR_COMPILE_THREADS=1 alone was NOT sufficient on
# torch 2.13. Read the traceback carefully and the reason is visible --
# `AsyncCompile.wakeup()` calls `use_process_pool()`, and that function
# SPAWNS the pool (`cls.process_pool().submit(...)`, async_compile.py:328) as
# part of deciding whether to use one. So the thread-count knob never gets a
# chance to prevent the spawn that the daemonic worker forbids.
#
# Disabling dynamo stops the compiler being invoked at all, so the path is
# unreachable rather than merely reconfigured. Safe here because both engines
# already run `enforce_eager=True` -- vLLM compiles nothing, and the only
# affected function is a trivial elementwise mask over token ids. If anyone
# ever runs this pipeline WITHOUT enforce_eager, revisit this: it would then
# be disabling real model compilation too.
export TORCHDYNAMO_DISABLE=1

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

# ---------------------------------------------------------------------------
# Gemma 4 pair (target: Gemma-4-31B dense, speculator: Gemma-4-E2B-it).
#
# Added for the Gemma 4 port's GATE phase only. The published SCBench sweep in
# this directory is Llama-3.1-8B / Llama-3.2-1B and is unaffected: nothing
# reads these unless a run is launched with them explicitly.
#
# What is and is not ready, so a run does not get started on a false premise:
#
#   READY   - the speculator-side scoring path. The query-capture hook is now
#             architecture-generic (it hooks `Attention.forward` and reads the
#             post-RoPE query as an argument rather than recomputing any
#             model's forward body), and `scoring.LayerGeometry` supplies the
#             model's own per-layer attention scale, logit softcapping,
#             layer types and KV-sharing flags.
#
#   NOT READY - the target-side SPARSE path. `sparse_target_runner.py` writes
#             ONE gathered block_table into every layer's attention metadata,
#             and `speculator_worker.py` reads `input_batch.block_table[0]`.
#             Both assume a single KV-cache group, which is false for an
#             interleaved model and fails SILENTLY. See
#             `speculator_worker.py`'s "Known risk areas" #1 for the traced
#             chain and for the `--disable-hybrid-kv-cache-manager` escape
#             hatch that restores the assumption. Do not run a SPARSE-k* row
#             against these paths until that guard exists.
#
# TODO: fill in this node's real snapshot paths (gated HF repos -- request
# access and export HF_TOKEN first). Verify before trusting:
#   ls -la /scratch/hf_cache/models--google--gemma-4-31B-it/snapshots/*/
#   ls -la /scratch/hf_cache/models--google--gemma-4-E2B-it/snapshots/*/
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
export GEMMA4_E2B_MODEL_PATH=$(_resolve_hf_snapshot /scratch/hf_cache/models--google--gemma-4-E2B-it)

# Gemma-4-26B-A4B-it (MoE, ~3.8B active). The alternative target when only
# two GPUs are available: ~49GB of weights fits ONE 80GB card, so the target
# runs at TP=1 on cuda:0 and the speculator gets cuda:1 to itself -- no
# tensor parallelism, no card sharing, no memory tuning.
#
# The 31B needs TP=2, which on a 2-GPU node means the speculator must share a
# card with a target rank. Workable, but it stacks three tight budgets at
# once. For RANKING three scorers the target barely matters -- the comparison
# is relative -- so the simpler substrate is usually the better trade.
export GEMMA4_MODEL_PATH=$(_resolve_hf_snapshot /scratch/hf_cache/models--google--gemma-4-26B-A4B-it)
