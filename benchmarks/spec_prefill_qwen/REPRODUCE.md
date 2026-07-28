# Reproduction Steps

Status: this is a Qwen3-8B (target) / Qwen3-1.7B (speculator) port of
`../spec_prefill/`'s Gemma4 pipeline (see that directory's own
`REPRODUCE.md`/`EXPERIMENT_PLAN.md` for the validation state of the
pipeline this was ported from). `vllm_patch/`'s architecture-agnostic
pieces (`scoring.py`, `kv_cache_utils.py`, `prefill_split.py`,
`pruning_registry.py`, `pruner.py`, `model_runner.py`, `worker.py`) are
carried over unchanged; `proposer.py`'s query-capture hook and
attention-layer discovery were rewritten against Qwen3's attention
implementation (see `EXPERIMENT_PLAN.md`'s "Implementation status" for what
changed and why). **None of this has been run on real hardware yet** — no
GPU on the machine this port was written on, and none of the Gemma4
pipeline's real-hardware findings (e.g. head_dim/num_kv_heads values,
whether `enable_chunked_prefill=False` is viable) automatically carry over,
since they were confirmed against Gemma4-26B-A4B-it/Gemma-4-E2B-it
specifically. Step 5 below is where that validation happens for Qwen3.

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
                      # real hardware (for the sibling Gemma4 pipeline;
                      # ModuleNotFoundError without it).
pip install datasets  # Hugging Face `datasets` library (distinct from
                      # `transformers`) -- datasets/prep_longbench_v2.py's
                      # load_dataset() call needs it; confirmed on real
                      # hardware for the sibling Gemma4 pipeline
                      # (ModuleNotFoundError without it) -- not pulled in by
                      # anything else in this env.
```

See `../evaluation_pipeline/REPRODUCE.md` steps 3-5 for the full gotchas (do not
`pip install vllm` from PyPI; OS-level multimodal deps; etc.) — not duplicated here.

Once the env exists, this directory has its own `.env_exports.sh` (conda
activation + `HF_TOKEN`/`HF_HOME` + `QWEN3_MODEL_PATH`/`QWEN3_1_7B_MODEL_PATH`)
— **local to `spec_prefill_qwen/`, not the shared one in
`../gemma4_moe_benchmarks/`, nor `../spec_prefill/`'s own** (those files are
still the source of truth for their own pipelines, but don't carry the
Qwen3 paths this pipeline needs, so SpecPrefill-for-Qwen3 work should source
this local copy instead):

```bash
export HF_TOKEN=<your token>
source benchmarks/spec_prefill_qwen/.env_exports.sh
```

## 2. The cloned reference implementation's environment (separate env)

The original SpecPrefill paper's reference repo is cloned at
`../spec_prefill/speculative_prefill/` (not duplicated here — it's a
gitignored external clone with its own `.git`, and this directory doesn't
need its own copy). It pins `vllm==0.6.3.post1`, `torch==2.4.0`,
`transformers==4.50.2` (conflicts with this fork's stack) and, per its own
README, only supports Llama as base/speculator — it does not run against
Qwen3 or Gemma4, or against this fork's vLLM, as-is. Useful only for
reading/reference; see `../spec_prefill/REPRODUCE.md` step 2 if you need to
set up its env.

## 3. Model checkpoints

- **Qwen3-8B (target)**: `.env_exports.sh`'s `QWEN3_MODEL_PATH` is a
  placeholder (`/path/to/qwen3-8b`) — fill in with this node's real
  downloaded snapshot path. If not yet downloaded:
  ```bash
  hf download Qwen/Qwen3-8B --cache-dir /scratch/hf_cache
  ```
  then verify it landed where expected and update `QWEN3_MODEL_PATH` in
  `.env_exports.sh`:
  ```bash
  ls -la /scratch/hf_cache/models--Qwen--Qwen3-8B/snapshots/*/
  du -sh /scratch/hf_cache/models--Qwen--Qwen3-8B/
  ```
- **Qwen3-1.7B (speculator)**: `.env_exports.sh`'s `QWEN3_1_7B_MODEL_PATH`
  is likewise a placeholder — fill in the real snapshot path. If not yet
  downloaded:
  ```bash
  hf download Qwen/Qwen3-1.7B --cache-dir /scratch/hf_cache
  ```
  Confirm `*.safetensors` + `model.safetensors.index.json` are present (not
  just config/tokenizer files — `AutoTokenizer.from_pretrained()` can
  succeed even when the weight shards are missing; this exact gotcha is in
  `../evaluation_pipeline/REPRODUCE.md`'s troubleshooting table):
  ```bash
  ls -la /scratch/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/*/
  du -sh /scratch/hf_cache/models--Qwen--Qwen3-1.7B/
  ```

Unlike the sibling Gemma4 pipeline, there is no text-only checkpoint
variant to generate here — Qwen3-8B/1.7B have no multimodal encoder cache
to strip (see `.env_exports.sh`'s own note).

## 4. LongBench v2 dataset

`datasets/prep_longbench_v2.py` fetches `THUDM/LongBench-v2` from Hugging Face,
filters to the "short" (<32k word) subset, and writes
`datasets/longbench_v2_samples.jsonl`:

```bash
cd benchmarks/spec_prefill_qwen
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

Both scripts are ported unchanged from `../spec_prefill/` — pure LongBench-v2
dataset/grading logic, no model coupling.

## 5. Validating the Algorithm 1 pieces built so far (`vllm_patch/`)

Three checks, in order — each depends on the previous one passing. **None
have been run for Qwen3 yet** (unlike the sibling Gemma4 pipeline, where
step 5.2 has passed on real hardware):

1. **Without a GPU or model weights** — the engine-agnostic pieces (scoring
   math, `Request.is_prefill_chunk` batch split, KV-cache layout logic,
   `pruning_registry` lifecycle), unchanged from `../spec_prefill/`:
   ```bash
   cd benchmarks/spec_prefill_qwen
   python3 test_vllm_patch.py
   ```
2. **On the GPU node, once step 3's `QWEN3_1_7B_MODEL_PATH` is set** — loads
   the real speculator standalone, installs the query-capture hook, and
   reports per-layer `head_dim`/`num_heads`/`num_kv_heads`. Expected (per
   `Qwen3Attention`'s implementation, not yet confirmed on hardware): a
   uniform head_dim across all layers, unlike Gemma-4-E2B-it's confirmed
   heterogeneous 256/512 split — if Step A instead finds heterogeneous
   head_dims, treat that as a surprise worth investigating before trusting
   anything downstream, not as expected behavior:
   ```bash
   source .env_exports.sh   # this directory's local copy, see step 1
   python3 validate_proposer.py --model $QWEN3_1_7B_MODEL_PATH
   ```
3. **On the GPU node, once both checkpoints are set** — loads the *target*
   model through `worker_cls=vllm_patch.worker.SpecPrefillWorker`, confirms
   normal (non-pruned) generation still works, then directly checks the
   riskiest assumption in the whole design: does a pruned request's
   *original* token positions actually reach the model's real forward pass,
   or does the override silently fail to take effect?
   ```bash
   python3 validate_runner_integration.py \
       --target-model $QWEN3_MODEL_PATH \
       --speculator-model $QWEN3_1_7B_MODEL_PATH
   ```

All three have been carefully reasoned through against this fork's verified
V1 APIs and against the sibling Gemma4 pipeline's own (real-hardware-passing)
version of these scripts, but **not executed for Qwen3** — see each script's
own docstring ("Known risk areas" / "residual risk") for what's most likely
to need a fix on the first real run, and in particular for what parts of
the Gemma4 pipeline's findings do vs. don't carry over (e.g. the
multimodal-driven `max_num_batched_tokens` floor and the "vLLM warns against
disabling chunked prefill" finding are not expected to apply to Qwen3, but
haven't been re-checked).

## 6. Running an experiment

`predict_longbench_v2.py` runs the P001–P006 keep-rate sweep (see
`EXPERIMENT_PLAN.md`'s experiment matrix), with or without SpecPrefill
pruning, and writes a predictions JSONL per experiment that `grade_longbench_v2.py`
(step 4) scores:

```bash
cd benchmarks/spec_prefill_qwen
python3 predict_longbench_v2.py --list           # print the experiment matrix
python3 predict_longbench_v2.py --exp P001 --max-keep 2   # smoke test first
python3 predict_longbench_v2.py --exp P001,P002,P003,P004,P005,P006
python3 grade_longbench_v2.py --samples datasets/longbench_v2_samples.jsonl \
    --predictions results/P002_predictions.jsonl
```

**Blocked on**: validation results from step 5 above (Step B/B2 in
`validate_runner_integration.py`), plus `predict_longbench_v2.py`'s own
real-hardware validation for Qwen3 — see that script's docstring for what's
carried over from the sibling Gemma4 pipeline's confirmed findings vs. what
still needs independent re-confirmation here.

## Expected runtime / hardware

TBD — no experiment has run yet, and no hardware sizing has been derived
for Qwen3-8B/1.7B (see `EXPERIMENT_PLAN.md`'s "Resource requirements").
Expect a far smaller footprint than the sibling Gemma4 pipeline's 2x A100 /
160GB, since Qwen3-8B+1.7B (dense, ~10B combined parameters) is much
smaller than Gemma-4-26B-A4B-it+E2B (MoE, tens of billions of parameters) —
likely fits on a single GPU, but not yet confirmed.
