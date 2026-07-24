# Reproduction Steps

Status: the V1 port (`vllm_patch/`) is code-complete. Step 5.2 (speculator loading,
`validate_proposer.py`) **passes on real hardware as of 2026-07-21** — model loads,
all 35 attention layers hook correctly, heterogeneous head-dim confirmed (see step
5 below). Step 5.3 (`validate_runner_integration.py`, the riskiest assumption in
the design) and step 6 (an actual experiment) remain unvalidated on real
hardware. The multi-step lookahead gap this section used to list is resolved
(fixed 2026-07-23, see `vllm_patch/proposer.py`'s `run_lookahead_steps`
docstring). The LongBench v2 dataset gap is closed too: step 4 below has a
prep + grading script, and step 6 has the prediction-generation driver
(`predict_longbench_v2.py`) — see `EXPERIMENT_PLAN.md`'s "Implementation
status" #3. What's left is real-hardware validation of the full pipeline
end-to-end, not missing code.

## 1. This fork's vLLM environment (for the target/speculator serving side)

Same conda env and install steps as `../evaluation_pipeline/REPRODUCE.md`:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda create -n vllm-ablation python=3.10 -y   # skip if it already exists
conda activate vllm-ablation
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/overwindows/vllm-msn
cd ~/vllm-msn
VLLM_USE_PRECOMPILED=1 pip install -e .
pip install pytest   # vllm_patch/proposer.py's build_lookahead_metadata reuses
                      # this fork's own tests/v1/attention/utils.py (a proven
                      # attention-metadata-construction pattern -- see its
                      # docstring for why), which imports pytest for an
                      # unrelated fallback path never actually hit here; a real
                      # runtime dependency for this benchmark, confirmed on
                      # real hardware (ModuleNotFoundError without it).
pip install datasets  # Hugging Face `datasets` library (distinct from
                      # `transformers`) -- datasets/prep_longbench_v2.py's
                      # load_dataset() call needs it; confirmed on real
                      # hardware (ModuleNotFoundError without it) -- not
                      # pulled in by anything else in this env.
```

See `../evaluation_pipeline/REPRODUCE.md` steps 3-5 for the full gotchas (do not
`pip install vllm` from PyPI; OS-level multimodal deps; etc.) — not duplicated here.

Once the env exists, this directory has its own `.env_exports.sh` (conda
activation + `HF_TOKEN`/`HF_HOME` + `GEMMA4_MODEL_PATH`/`GEMMA4_E2B_MODEL_PATH`)
— **local to `spec_prefill/`, not the shared one in `../gemma4_moe_benchmarks/`**
(that file is still the source of truth for `gemma4_moe_benchmarks`/
`evaluation_pipeline` themselves, but doesn't carry the speculator path this
pipeline needs, so SpecPrefill work should source this local copy instead):

```bash
export HF_TOKEN=<your token>
source benchmarks/spec_prefill/.env_exports.sh
```

## 2. The cloned reference implementation's environment (separate env)

`speculative_prefill/` pins `vllm==0.6.3.post1`, `torch==2.4.0`,
`transformers==4.50.2` — these conflict with this fork's stack (`vllm-ablation`
above), so **do not** install it into the same env. Per its own README:

```bash
conda create -n sp python=3.10.15 -y
conda activate sp
cd benchmarks/spec_prefill/speculative_prefill
pip3 install -r requirements.txt
pip3 install -e .
```

This env is useful today only for reading/reference and for running the *original*
paper's experiments against Llama models — it does not run against Gemma 4 or this
fork's vLLM until the port (Implementation status #1-#2) is done.

## 3. Model checkpoints

- **Gemma-4-26B-A4B-it (target)**: `.env_exports.sh`'s `GEMMA4_MODEL_PATH` carries
  a specific snapshot path that was valid on a prior node ("node-0") — **verify
  it's actually present on whatever node you're using now** before trusting it,
  since a fresh/different node may not have it:
  ```bash
  ls -la /scratch/hf_cache/models--google--gemma-4-26B-A4B-it/snapshots/*/
  du -sh /scratch/hf_cache/models--google--gemma-4-26B-A4B-it/   # expect ~49G
  ```
  If missing, download it the same way as the speculator below
  (`hf download google/gemma-4-26B-A4B-it --cache-dir /scratch/hf_cache`) and
  update `GEMMA4_MODEL_PATH` in `.env_exports.sh`.
- **Gemma-4-E2B-it (speculator)**: `.env_exports.sh`'s `GEMMA4_E2B_MODEL_PATH`
  already carries a specific, already-downloaded snapshot path (filled in when
  `.env_exports.sh` was moved into this directory) — same as `GEMMA4_MODEL_PATH`
  above, **verify it's actually present on whatever node you're using now**
  before trusting it:
  ```bash
  ls -la /scratch/hf_cache/models--google--gemma-4-E2B-it/snapshots/*/
  du -sh /scratch/hf_cache/models--google--gemma-4-E2B-it/
  ```
  Confirm `*.safetensors` + `model.safetensors.index.json` are present (not just
  config/tokenizer files — `AutoTokenizer.from_pretrained()` can succeed even when
  the weight shards are missing; this exact gotcha is in
  `../evaluation_pipeline/REPRODUCE.md`'s troubleshooting table). If missing,
  download it the same way as the target checkpoint above
  (`hf download google/gemma-4-E2B-it --cache-dir /scratch/hf_cache`) and update
  `GEMMA4_E2B_MODEL_PATH` in **this directory's** `.env_exports.sh` (not the
  shared one in `../gemma4_moe_benchmarks/`).

## 4. LongBench v2 dataset

`datasets/prep_longbench_v2.py` fetches `THUDM/LongBench-v2` from Hugging Face,
filters to the "short" (<32k word) subset, and writes
`datasets/longbench_v2_samples.jsonl`:

```bash
cd benchmarks/spec_prefill
python3 datasets/prep_longbench_v2.py --max-keep -1
```

`grade_longbench_v2.py` scores a predictions file (JSONL of `{"id", "pred"}` rows)
against that samples file:

```bash
python3 grade_longbench_v2.py \
    --samples datasets/longbench_v2_samples.jsonl \
    --predictions <path-to-predictions.jsonl> \
    --output results/longbench_v2_result.json
```

**Still missing**: a prediction-*generation* script (the vLLM-driving counterpart
to the cloned reference repo's `eval/long_bench/pred_vllm.py`) to actually produce
that predictions file for a given keep-rate config — see
`EXPERIMENT_PLAN.md`'s "Implementation status" #3.

## 5. Validating the Algorithm 1 pieces built so far (`vllm_patch/`)

Three checks, in order — each depends on the previous one passing:

1. **Without a GPU or model weights** — the engine-agnostic pieces (scoring
   math, `Request.is_prefill_chunk` batch split, KV-cache layout logic,
   `pruning_registry` lifecycle):
   ```bash
   cd benchmarks/spec_prefill
   python3 test_vllm_patch.py
   ```
2. **On the GPU node, once step 3's `GEMMA4_E2B_MODEL_PATH` is set** — loads
   the real speculator standalone, installs the query-capture hook, and
   reports per-layer `head_dim`. **Confirmed passing (2026-07-21)**: 35
   layers found, `head_dim` heterogeneous (256 sliding / 512 full-attention,
   `num_kv_heads=1` throughout) — same pattern as the 26B target, so Step B
   (the forward-pass smoke test, which only runs for a uniform layout) is
   expected to self-skip:
   ```bash
   source .env_exports.sh   # this directory's local copy, see step 1
   python3 validate_proposer.py --model $GEMMA4_E2B_MODEL_PATH
   ```
3. **On the GPU node, once both checkpoints are set** — loads the *target*
   model through `worker_cls=vllm_patch.worker.SpecPrefillWorker`, confirms
   normal (non-pruned) generation still works, then directly checks the
   riskiest assumption in the whole design: does a pruned request's
   *original* token positions actually reach the model's real forward pass,
   or does the override silently fail to take effect?
   ```bash
   python3 validate_runner_integration.py \
       --target-model $GEMMA4_MODEL_PATH \
       --speculator-model $GEMMA4_E2B_MODEL_PATH
   ```

All three have been carefully reasoned through against this fork's verified V1
APIs but **not executed** (no GPU on the machine they were written on) — see
each script's own docstring ("Known risk areas" / "residual risk") for what's
most likely to need a fix on the first real run.

## 6. Running an experiment

`predict_longbench_v2.py` runs the P001–P006 keep-rate sweep (see
`EXPERIMENT_PLAN.md`'s experiment matrix), with or without SpecPrefill
pruning, and writes a predictions JSONL per experiment that `grade_longbench_v2.py`
(step 4) scores:

```bash
cd benchmarks/spec_prefill
python3 predict_longbench_v2.py --list           # print the experiment matrix
python3 predict_longbench_v2.py --exp P001 --max-keep 2   # smoke test first
python3 predict_longbench_v2.py --exp P001,P002,P003,P004,P005,P006
python3 grade_longbench_v2.py --samples datasets/longbench_v2_samples.jsonl \
    --predictions results/P002_predictions.jsonl
```

The multi-step lookahead limitation this section previously listed here is
resolved — `vllm_patch/proposer.py`'s `run_lookahead_steps` had a real bug
that made `look_ahead_cnt > 1` unreliable, fixed 2026-07-23 (see that
function's own docstring); `EXPERIMENT_PLAN.md`'s `look_ahead_cnt: 8` default
is safe to use now.

**Still blocked on**: validation results from step 5 above (Step B/B2 in
`validate_runner_integration.py`), plus `predict_longbench_v2.py`'s own
real-hardware validation — see that script's docstring ("Multi-request
engine-driving loop" section) for what's confirmed vs. reasoned-through-but-
not-yet-executed (no GPU on the machine it was written on, same as every
other script here).

## Expected runtime / hardware

TBD — no experiment has run yet. Target hardware per `EXPERIMENT_PLAN.md`: 2x A100,
160GB total GPU HBM.
