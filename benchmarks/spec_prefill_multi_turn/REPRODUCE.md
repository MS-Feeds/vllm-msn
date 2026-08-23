# Reproduction Steps

Status: this pipeline is built on top of `../spec_prefill_llama/`'s
single-turn SpecPrefill port. Its architecture-agnostic pieces
(`vllm_patch/{scoring,kv_cache_utils,prefill_split,pruning_registry,
model_runner,worker}.py`) are carried over unchanged; the multi-turn
extension (`vllm_patch/conversation_state.py`, `vllm_patch/speculator_worker.py`,
and the rewritten `proposer.py`/`pruner.py`) is new and has NO single-turn
precedent to lean on — see `EXPERIMENT_PLAN.md`'s "Implementation status"
and each module's own "Known risk areas"/docstring notes. **None of this has
been run on real hardware yet.**

## 1. This fork's vLLM environment

Same conda env and install steps as `../spec_prefill_llama/REPRODUCE.md`
step 1 (itself following `../evaluation_pipeline/REPRODUCE.md`):

```bash
source /opt/conda/etc/profile.d/conda.sh
conda create -n vllm-ablation python=3.10 -y   # skip if it already exists
conda activate vllm-ablation
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/overwindows/vllm-msn
cd ~/vllm-msn
VLLM_USE_PRECOMPILED=1 pip install -e .
pip install pytest datasets
```

See `../evaluation_pipeline/REPRODUCE.md` steps 3-5 for the full gotchas (do
not `pip install vllm` from PyPI; OS-level multimodal deps; etc.) — not
duplicated here.

Once the env exists, this directory has its own `.env_exports.sh` — local to
`spec_prefill_multi_turn/`, not `../spec_prefill_llama/`'s (same paths would
work, since both target Llama-3.1-8B/3.2-1B, but keep this pipeline's own
copy as the source of truth for it, consistent with how every `spec_prefill*`
pipeline in this repo keeps its own):

```bash
export HF_TOKEN=<your token>
source benchmarks/spec_prefill_multi_turn/.env_exports.sh
```

## 2. Model checkpoints

Same two gated Hugging Face checkpoints as `../spec_prefill_llama/`
(`meta-llama/Llama-3.1-8B-Instruct` target, `meta-llama/Llama-3.2-1B-Instruct`
speculator) — see that directory's `REPRODUCE.md` step 3 for the full
download/verification instructions (request access on each model's HF page
first; verify `*.safetensors` actually landed, not just tokenizer/config
files). If you already downloaded these for `../spec_prefill_llama/`, reuse
the same snapshot paths in this pipeline's `.env_exports.sh`.

**Optional third checkpoint**: `meta-llama/Llama-3.2-3B-Instruct`
(`LLAMA32_3B_MODEL_PATH`), gated the same way. Not needed for any row of the
experiment matrix — it is only the mid-size scorer for
`ACCURACY_IMPROVEMENTS.md` §1.6's capacity probe, run via
`--oracle-scorer-model`. Skip it unless you are running that probe. Download
it the same way as the other two, then fill in the real snapshot hash:

```bash
hf download meta-llama/Llama-3.2-3B-Instruct --exclude "original/*"
```

```bash
ls -la /scratch/hf_cache/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/*/
```

## 3. SCBench dataset

`datasets/prep_scbench.py` fetches `microsoft/SCBench`'s 3 MVP configs
(`scbench_qa_eng`, `scbench_kv`, `scbench_summary`) from Hugging Face and
writes `datasets/scbench_samples.jsonl` (one row per conversation/context,
NOT per question — see that script's own docstring for why):

```bash
cd benchmarks/spec_prefill_multi_turn
python3 datasets/prep_scbench.py --max-keep-per-config -1
```

`grade_scbench.py` scores a predictions file (JSONL of
`{"conversation_id", "turn_idx", "config", "pred"}` rows) against that
samples file:

```bash
python3 grade_scbench.py \
    --samples datasets/scbench_samples.jsonl \
    --predictions results/<exp_id>_predictions.jsonl \
    --output results/scbench_result.json
```

## 4. Validating the Algorithm pieces built so far (`vllm_patch/`)

Three checks, in order — each depends on the previous passing. **None have
been run yet**:

1. **Without a GPU or model weights** — the engine-agnostic pieces
   (scoring math, KV-cache layout logic, `pruning_registry` lifecycle,
   PLUS `conversation_state.py`'s KEEP/DISCARD candidate-pool logic, the
   one genuinely new piece with no single-turn analog). This IS runnable
   in any Python 3.10+/torch/PyYAML environment (confirmed: ran clean, 19/19
   tests passed, in a CPU-only environment while writing this pipeline —
   see the test file's own output for what was actually exercised):
   ```bash
   cd benchmarks/spec_prefill_multi_turn
   python3 test_vllm_patch.py
   ```
2. **On the GPU node, once `LLAMA32_1B_MODEL_PATH` is set** — constructs
   the PERSISTENT speculator engine (a genuinely new integration pattern
   for this pipeline, see `vllm_patch/speculator_worker.py`'s "Known risk
   areas"), runs a 2-turn synthetic conversation, and directly checks
   whether turn 2 actually gets a prefix-cache hit for turn 1's content
   (the measured version of "each turn only prefills the new query tokens
   against its own cache") and whether K for turn-1-computed positions can
   still be retrieved from turn 2:
   ```bash
   source .env_exports.sh   # this directory's local copy, see step 1
   python3 validate_proposer.py --model $LLAMA32_1B_MODEL_PATH
   ```
3. **On the GPU node, once both checkpoints are set** — loads the target
   model through `worker_cls=vllm_patch.worker.SpecPrefillWorker`, confirms
   normal (non-pruned) generation still works, then runs a 2-turn synthetic
   conversation through the FULL conversation-aware pruning path
   (`compute_pruned_turn`/`prune_and_add_turn`) and directly checks — via a
   diagnostic hook on the target's own attention layers — that a
   MULTI-TURN pruned request's absolute (not turn-local) positions actually
   reach the model's real forward pass:
   ```bash
   python3 validate_runner_integration.py \
       --target-model $LLAMA31_8B_MODEL_PATH \
       --speculator-model $LLAMA32_1B_MODEL_PATH
   ```

**Known unverified assumption in step 3**: this script guesses the internal
attribute path to reach the target model instance for its diagnostic hook
(`llm.llm_engine.model_executor...`) — it prints a clear WARNING and skips
the position-verification assertion (rather than silently reporting a false
PASS) if that path doesn't resolve on this fork's actual `LLMEngine`
structure. Fix the path in the script if you hit that warning, using
whatever this fork's real internals turn out to be.

## 5. Running an experiment

`predict_scbench.py` runs the M000/M-k*-g*/ORACLE-k* matrix (see
`EXPERIMENT_PLAN.md`'s experiment matrix — note the ORACLE rows are not yet
wired up, see that file's "Implementation status" #4), with SpecPrefill
pruning, and writes a per-turn predictions JSONL per experiment that
`grade_scbench.py` (step 3) scores:

```bash
cd benchmarks/spec_prefill_multi_turn
python3 predict_scbench.py --list                                  # print the experiment matrix
python3 predict_scbench.py --exp M000 --max-conversations 2        # smoke test first
python3 predict_scbench.py --exp M000,M-k80-g32,M-k20-gtoken
python3 grade_scbench.py --samples datasets/scbench_samples.jsonl \
    --predictions results/M-k80-g32_predictions.jsonl
```

**Blocked on**: validation results from step 4 above (both scripts), plus
this pipeline's own real-hardware confirmation that the multi-turn-specific
mechanisms (persistent speculator prefix-cache reuse, absolute-position RoPE
restoration across turns) actually behave as designed — this is a
meaningfully bigger jump from "unvalidated" to "validated" than the
single-turn pipeline's own step 5 was, since there's no prior multi-turn run
(here or in any sibling pipeline) to lean on for confidence.

## Expected runtime / hardware

**2x A100 80GB**, per the protocol document (see `EXPERIMENT_PLAN.md`'s
"Resource requirements" for why this pipeline follows the protocol's stated
figure rather than the single-turn sibling's smaller "likely fits on one
GPU" estimate — the multi-turn speculator's long-lived, growing KV cache is
a new memory variable that estimate never had to account for).

**ETA**: TBD
