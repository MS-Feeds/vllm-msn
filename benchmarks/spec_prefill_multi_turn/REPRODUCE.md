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

### Gemma 4 only: pin transformers to exactly 5.14.1

```bash
pip install "transformers==5.14.1"
```

Exactly that version — not a floor, not a range. Both neighbours fail, and
neither failure names the version as the cause:

- **Older** (no `gemma4` in the registry) dies at the first `AutoConfig`
  call with *"The checkpoint you are trying to load has model type `gemma4`
  but Transformers does not recognize this architecture"*. It surfaces from
  `model_structure.native_context_length`, before any GPU work, and reads
  like a corrupt checkpoint.
- **5.15+** raises `AmbiguousGlobalPerLayerAttributeError` on Gemma 4's
  per-layer attributes (`head_dim` vs `global_head_dim` and friends).

**Order matters:** the `pip install pytest datasets` line above silently
DOWNGRADES transformers. Install datasets first, then re-pin:

```bash
pip install pytest datasets && pip install "transformers==5.14.1"
```

`.env_exports.sh` warns at source time if the installed version is wrong, so
sourcing it on a fresh node reports this immediately instead of two minutes
into a run. The Llama rows do not need the pin.

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

> **To reproduce the paper you need different checkpoints than this section
> downloads.** The paper's two pairs are `meta-llama/Llama-3.1-70B-Instruct` +
> `meta-llama/Llama-3.2-1B-Instruct`, and `google/gemma-4-31B-it` +
> `google/gemma-4-E2B-it`. `.env_exports.sh` already carries
> `GEMMA4_31B_MODEL_PATH` / `GEMMA4_E2B_MODEL_PATH`; add a 70B path alongside
> `LLAMA31_8B_MODEL_PATH`. The 8B instructions below are what the earlier
> SCBench work used and are kept because the rest of this file's validation
> steps reference them — they will not reproduce the paper's numbers.

Same two gated Hugging Face checkpoints as `../spec_prefill_llama/`
(`meta-llama/Llama-3.1-8B-Instruct` target, `meta-llama/Llama-3.2-1B-Instruct`
speculator). **Request access on each model's Hugging Face page first**
(approval is usually near-instant for an account in good standing) and export
`HF_TOKEN` before downloading. If you already pulled these for
`../spec_prefill_llama/`, reuse the same snapshot paths in this pipeline's
`.env_exports.sh` rather than downloading again.

```bash
export HF_TOKEN=<your token>
hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
hf download meta-llama/Llama-3.2-1B-Instruct --exclude "original/*"
```

Then fill in the real snapshot paths in `.env_exports.sh`'s
`LLAMA31_8B_MODEL_PATH` / `LLAMA32_1B_MODEL_PATH` (both ship as
placeholders):

```bash
ls -la $HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/
ls -la $HF_HOME/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*/
du -sh $HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct/
```

**Verify `*.safetensors` and `model.safetensors.index.json` actually landed**,
not just the config/tokenizer files. `AutoTokenizer.from_pretrained()` and
`AutoConfig.from_pretrained()` both succeed against a weightless snapshot, so
a partial download surfaces much later as a confusing load failure rather
than as a missing-file error (this exact gotcha is in
`../evaluation_pipeline/REPRODUCE.md`'s troubleshooting table). `du -sh`
should read roughly 15 GB for the 8B and 2.5 GB for the 1B.

`--exclude "original/*"` skips the duplicate `original/consolidated.*`
weights, which vLLM never reads — about 15 GB saved on the 8B alone.

The `hub/` path segment comes from `HF_HOME` (see the note at the end of this
section); on nodes where these were downloaded with an explicit
`--cache-dir /scratch/hf_cache` they sit one level up, WITHOUT `hub/`. Check
`echo $HF_HOME` if a path does not resolve.

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

**Gemma 4 checkpoints** (GATE-phase only — see "Gemma 4 only" pin above; the
published SCBench sweep in this directory does not need these). Three gated
repos, matching `.env_exports.sh`'s `GEMMA4_31B_MODEL_PATH`,
`GEMMA4_E2B_MODEL_PATH`, `GEMMA4_MODEL_PATH` (the 26B-A4B MoE, the two-GPU
alternative to the 31B — see that file's comment for when to use which):

```bash
export HF_TOKEN=<your token>
hf download google/gemma-4-31B-it --exclude "original/*"
hf download google/gemma-4-E2B-it --exclude "original/*"
hf download google/gemma-4-26B-A4B-it --exclude "original/*"
```

```bash
ls -la /scratch/hf_cache/hub/models--google--gemma-4-31B-it/snapshots/*/
ls -la /scratch/hf_cache/hub/models--google--gemma-4-E2B-it/snapshots/*/
ls -la /scratch/hf_cache/hub/models--google--gemma-4-26B-A4B-it/snapshots/*/
```

The `hub/` segment comes from `HF_HOME=/scratch/hf_cache` (the `hf` CLI
downloads under `$HF_HOME/hub/`) — check `echo $HF_HOME` if paths don't
resolve; older downloads on some nodes landed one level up, without `hub/`
(see `LLAMA31_8B_MODEL_PATH`/`LLAMA32_1B_MODEL_PATH` in `.env_exports.sh` vs
`LLAMA32_3B_MODEL_PATH`).

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

## 3b. SWE-bench agent trajectories (proposed extension, never run)

> **Not part of reproducing the paper.** The paper's results come from
> LongBench-v2-MC (§3a equivalent — `datasets/prep_longbench_v2_multiturn.py`),
> not from anything in this section. The code here is written and unit-tested
> but has never been executed on hardware. Skip this section entirely if you
> are reproducing published numbers.

The code-agent workload. Two phases that answer two different claims — see
`EXPERIMENT_PLAN.md`'s Benchmark section for why neither substitutes for the
other. Phase 1 needs no Docker; Phase 2 needs x86_64 Linux, Docker, and the
`swebench` package.

### 3b.1 Record (once, needs Docker)

Serve the **target** dense through stock vLLM, then drive
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) against it.
Record with the target and not a frontier model: the recorded action becomes
the grading reference, and a reference the target could never have produced
collapses `M000`'s dynamic range so a 5% criterion sits inside the noise.

```bash
vllm serve "$TARGET_MODEL_PATH" --port 8000
```

Point mini-swe-agent's `swebench.yaml` at it (`model_name:
"hosted_vllm/<path>"`, `model_kwargs.api_base: "http://localhost:8000/v1"`,
plus a `registry.json` for its cost tracking), then:

```bash
mini-extra swebench --subset verified --split test --workers 4 -o ./sweb_out
```

### 3b.2 Normalize and pack (no Docker)

```bash
python3 datasets/normalize_swebench_trajs.py --traj-dir ./sweb_out
```

Read its "trajectories with at least T steps" table before choosing
`--turns-per-conv`: a trajectory shorter than `T` is dropped whole.

```bash
python3 datasets/prep_swebench_agent_replay.py \
    --tokenizer "$TARGET_MODEL_PATH" \
    --turns-per-conv 16 --max-tokens 512 \
    --max-observation-tokens 4000 \
    --target-max-num-batched-tokens 130560 \
    --speculator-max-num-batched-tokens 131063 \
    --seed 42 \
    --output datasets/swebench_agent_replay.jsonl
```

### 3b.3 Phase 1 — replay sweep (paired, gradable)

```bash
python3 predict_scbench.py \
    --exp M000,SPARSE-k80-g32,SPARSE-k40-g32,SPARSE-k20-g32 \
    --samples datasets/swebench_agent_replay.jsonl \
    --output-suffix=-sweagent
```

```bash
python3 grade_scbench.py --batch --samples datasets/swebench_agent_replay.jsonl
```

**Gate before spending the matrix:** run `M000` alone on ~20 conversations
first and read `overall`. Below ~0.5 the dataset is not discriminative enough
to resolve a 5% effect — switch the headline metric to a softer one, or
re-record so the recorded and replayed prompts match exactly.

What to read: `overall_turn0` must **not** move between arms (turn 0's prefill
is dense under both scopes, so a difference there is a bug); the
`(config, turn_idx)` breakdown is the multi-turn signal; and
`num_skipped_too_large` must be **0** — a non-zero means the prep-time and
run-time budget checks disagree and every affected row is incomparable.

### 3b.4 Phase 2 — live agentic (needs Docker)

Same samples file; turn 0's issue statement is read from it and every later
turn comes from the container.

```bash
python3 predict_scbench.py --exp M000 --agentic \
    --samples datasets/swebench_agent_replay.jsonl \
    --max-turns 30 --max-observation-tokens 2000 \
    --max-conversations 1 --output-suffix=-agentsmoke
```

Then the three arms (`M000` plus the keep rates Phase 1 selected). Results
land in `results/<exp_id>_agentic.csv` and `results/<exp_id>_preds.json`.

Two things differ from replay and are **not** bugs:

- `num_skipped_too_large` is expected to be non-zero. Observation lengths are
  not knowable in advance, so the driver's own pre-flight retires an
  overlong conversation cleanly at the turn it would have overrun.
- `grade_scbench.py` should **not** be run on the output. Arms diverge, so
  there is no shared per-turn reference; read the `_agentic.csv` endpoints
  instead — wall clock, exit-reason mix and steps first, resolve rate last.

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

**4 GPUs.** Target and speculator both at tensor parallelism 4 on the same
four devices, eager mode, asynchronous scheduling disabled — for both model
pairs (Llama-3.1-70B + Llama-3.2-1B, Gemma-4-31B + Gemma-4-E2B). This is what
the paper's experiments ran on.

The count is set by the targets: a 70B or 31B model in BF16 will not
co-reside with a speculator and a ~130K-token persistent KV cache on fewer
devices. The speculators are small enough to share those same four rather
than needing their own.

Keep `--batch-conversations` at 1. The LongBench-v2 prep sizes every
conversation to fill `--target-max-num-batched-tokens`, so a second concurrent
conversation needs a second full copy of that KV, and the paper's rows were
all measured serially.

An earlier version of this section said "2x A100 80GB, per the protocol
document" against a Llama-3.1-8B target. That was the original plan, not the
experiment; see `EXPERIMENT_PLAN.md`'s header table for the authoritative
configuration.
