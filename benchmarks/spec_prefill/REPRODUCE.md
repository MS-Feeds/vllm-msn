# Reproduction Steps

Status: the V1 port (`vllm_patch/`) is code-complete. Step 5.2 (speculator loading,
`validate_proposer.py`) **passes on real hardware as of 2026-07-21** — model loads,
all 35 attention layers hook correctly, heterogeneous head-dim confirmed (see step
5 below). Step 5.3 (`validate_runner_integration.py`, the riskiest assumption in
the design) and step 6 (an actual experiment) remain unvalidated/blocked — on
step 5.3's results plus two known gaps (the LongBench v2 dataset, and multi-step
lookahead — see `EXPERIMENT_PLAN.md`'s "Implementation status" and
`vllm_patch/proposer.py`'s `build_lookahead_metadata` docstring).

## 1. This fork's vLLM environment (for the target/speculator serving side)

Same conda env and install steps as `../evaluation_pipeline/REPRODUCE.md`:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda create -n vllm-ablation python=3.10 -y   # skip if it already exists
conda activate vllm-ablation
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
cd ~/vllm-msn
VLLM_USE_PRECOMPILED=1 pip install -e .
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
- **Gemma-4-E2B-it (speculator)**: **not yet downloaded.** `.env_exports.sh`
  has a commented-out `GEMMA4_E2B_MODEL_PATH` placeholder ready to fill in. Following
  the same `hf download` pattern used for the target checkpoint above
  (`../evaluation_pipeline/REPRODUCE.md` step 6):

  ```bash
  export HF_TOKEN=<your token>
  df -h /scratch   # confirm space before a multi-GB download

  hf download google/gemma-4-E2B-it --cache-dir /scratch/hf_cache
  ```

  Then find the real snapshot path (same "no `hub/` nesting" gotcha as the other
  checkpoints — see `.env_exports.sh`'s own NOTE comment):

  ```bash
  ls -la /scratch/hf_cache/models--google--gemma-4-E2B-it/snapshots/*/
  du -sh /scratch/hf_cache/models--google--gemma-4-E2B-it/
  ```

  Confirm `*.safetensors` + `model.safetensors.index.json` are present (not just
  config/tokenizer files — `AutoTokenizer.from_pretrained()` can succeed even when
  the weight shards are missing; this exact gotcha is in
  `../evaluation_pipeline/REPRODUCE.md`'s troubleshooting table), then uncomment
  and fill in the real path in **this directory's** `.env_exports.sh` (not the
  shared one in `../gemma4_moe_benchmarks/`):

  ```bash
  export GEMMA4_E2B_MODEL_PATH=/scratch/hf_cache/models--google--gemma-4-E2B-it/snapshots/<actual-hash>
  ```

## 4. LongBench v2 dataset

**TBD** — no prep script exists yet (Implementation status #3). Will need to fetch
`THUDM/LongBench-v2` from Hugging Face and filter to the "short" (<32k word) subset,
analogous in shape to `../evaluation_pipeline/datasets/prep_aime.py`.

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

**TBD** — blocked on: (a) validation results from step 5 above, (b) the
multi-step lookahead limitation (`EXPERIMENT_PLAN.md`'s default
`look_ahead_cnt: 8` isn't reliable yet — see `vllm_patch/proposer.py`'s
`build_lookahead_metadata` docstring), and (c) the LongBench v2 dataset (step
4, not started). Once available, the entrypoint is expected to resemble the
reference repo's own pattern:

```bash
# reference repo's own pattern (Llama, v0 engine) — NOT directly valid against
# this fork; vllm_patch/pruner.py's prune_and_add_request + vllm_patch/worker.py's
# SpecPrefillWorker are this fork's equivalent entry points, still need a driver
# script to actually run a full benchmark sweep like this one does:
cd speculative_prefill
SPEC_CONFIG_PATH=../configs/config_p1_full_lah8.yaml python eval/long_bench/pred_vllm.py \
    --model "$GEMMA4_MODEL_PATH" \
    --spec-model "$GEMMA4_E2B_MODEL_PATH" \
    --spec-prefill \
    --exp spec_p1_full_lah8
```

## Expected runtime / hardware

TBD — no experiment has run yet. Target hardware per `EXPERIMENT_PLAN.md`: 2x A100,
160GB total GPU HBM.
