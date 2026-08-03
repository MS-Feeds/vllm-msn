# Reproduction Steps

Status: this is a Qwen3-Coder-480B-A35B-Instruct (target) /
Qwen3-Coder-30B-A3B-Instruct (speculator) port of
`../spec_prefill_qwen/`'s dense-Qwen3 (Qwen3-8B/1.7B) pipeline, itself a
port of `../spec_prefill/`'s Gemma4 pipeline (see those directories' own
`REPRODUCE.md`/`EXPERIMENT_PLAN.md` for the validation state of the
pipelines this was ported from). `vllm_patch/`'s pieces (`scoring.py`,
`kv_cache_utils.py`, `prefill_split.py`, `pruning_registry.py`, `pruner.py`,
`model_runner.py`, `worker.py`, and — unlike the dense-Qwen3 port's own
rewrite of the Gemma4 hooks — `proposer.py`'s query-capture hook and
attention-layer discovery too) are all carried over **unchanged**: confirmed
directly against this fork's `vllm/model_executor/models/qwen3_moe.py` that
`Qwen3MoeAttention.forward` is byte-identical to dense `Qwen3Attention.
forward`, so the MoE target/speculator here needed no `vllm_patch/` logic
changes (see `EXPERIMENT_PLAN.md`'s "Implementation status" for the full
verification). **None of this has been run on real hardware yet** — no
GPU on the machine this port was written on, and none of the Gemma4 or
dense-Qwen3 pipelines' real-hardware findings (e.g. head_dim/num_kv_heads
values, whether `enable_chunked_prefill=False` is viable, whether the
target fits alongside the speculator on shared hardware) automatically
carry over, since they were confirmed against different checkpoints at a
much smaller scale. Step 5 below is where that validation happens for
Qwen3-Coder.

**The one qualitatively new requirement versus every other port here**:
Qwen3-Coder-480B-A35B does not fit on a single GPU. Everywhere below that
assumes single-GPU target loading, expect to add
`tensor_parallel_size`/`--target-tensor-parallel-size` (see
`../rlm_specprefill/target_stage/vllm_offline_engine.py` and
`validate_runner_integration.py`'s own flag of the same purpose).

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
activation + `HF_TOKEN`/`HF_HOME` +
`QWEN3_CODER_480B_MODEL_PATH`/`QWEN3_CODER_30B_MODEL_PATH`) — **local to
`spec_prefill_qwen_coder/`, not the shared one in
`../gemma4_moe_benchmarks/`, nor `../spec_prefill/`'s or
`../spec_prefill_qwen/`'s own** (those files are still the source of truth
for their own pipelines, but don't carry the Qwen3-Coder paths this
pipeline needs, so SpecPrefill-for-Qwen3-Coder work should source this
local copy instead):

```bash
export HF_TOKEN=<your token>
source benchmarks/spec_prefill_qwen_coder/.env_exports.sh
```

## 2. The cloned reference implementation's environment (separate env)

The original SpecPrefill paper's reference repo is cloned at
`../spec_prefill/speculative_prefill/` (not duplicated here — it's a
gitignored external clone with its own `.git`, and this directory doesn't
need its own copy). It pins `vllm==0.6.3.post1`, `torch==2.4.0`,
`transformers==4.50.2` (conflicts with this fork's stack) and, per its own
README, only supports Llama as base/speculator — it does not run against
Qwen3(-Coder) or Gemma4, or against this fork's vLLM, as-is. Useful only for
reading/reference; see `../spec_prefill/REPRODUCE.md` step 2 if you need to
set up its env.

## 3. Model checkpoints

- **Qwen3-Coder-480B-A35B-Instruct (target)**: `.env_exports.sh`'s
  `QWEN3_CODER_480B_MODEL_PATH` is a placeholder — fill in with this node's
  real downloaded snapshot path. **At full BF16 (~960GB) this does not fit
  on a single GPU, or comfortably across most 8-GPU nodes** — confirm
  available disk/HF cache space before downloading, and strongly consider
  the quantized option below instead for serving:
  ```bash
  hf download Qwen/Qwen3-Coder-480B-A35B-Instruct --cache-dir /scratch/hf_cache
  ```
  **4-bit AWQ option (recommended for easier serving, e.g. on 8x A100
  80GB)**: `QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ` (~236GB total,
  confirmed via its own model card) has a real `FusedMoE`-compatible
  AWQ-Marlin quant path in this fork — no code changes needed, vLLM
  auto-detects the quantization from the checkpoint's `config.json`:
  ```bash
  hf download QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ --cache-dir /scratch/hf_cache
  ```
  That card's own tested config is `tensor_parallel_size=8` **with
  `--enable-expert-parallel` REQUIRED** ("otherwise, the expert tensors
  cannot be evenly split across tensor parallel ranks") — pass
  `--target-enable-expert-parallel` to `runner/run_arm.py` /
  `validate_runner_integration.py` when using it at TP>1. It also documents
  vLLM ≥0.9.2, transformers ≥4.51.0, and non-thinking-only generation.
  Either way, after downloading:
  ```bash
  ls -la /scratch/hf_cache/hub/models--<org>--<repo>/snapshots/*/
  du -sh /scratch/hf_cache/hub/models--<org>--<repo>/
  ```
  and update `QWEN3_CODER_480B_MODEL_PATH` in `.env_exports.sh` to point at
  whichever checkpoint (BF16 or AWQ) you're actually serving. Determine the
  number of GPUs needed for `tensor_parallel_size` from this node's actual
  GPU memory and the checkpoint's real on-disk size (`du -sh` above) before
  attempting to load it — do not assume a specific TP degree from parameter
  count alone (see `EXPERIMENT_PLAN.md`'s "Resource requirements" for the
  full GPU-budget table and the speculator-placement trade-off this
  creates at TP=8).
- **Qwen3-Coder-30B-A3B-Instruct (speculator)**: `.env_exports.sh`'s
  `QWEN3_CODER_30B_MODEL_PATH` is likewise a placeholder — fill in the real
  snapshot path. If not yet downloaded:
  ```bash
  hf download Qwen/Qwen3-Coder-30B-A3B-Instruct --cache-dir /scratch/hf_cache
  ```
  Confirm `*.safetensors` + `model.safetensors.index.json` are present (not
  just config/tokenizer files — `AutoTokenizer.from_pretrained()` can
  succeed even when the weight shards are missing; this exact gotcha is in
  `../evaluation_pipeline/REPRODUCE.md`'s troubleshooting table):
  ```bash
  ls -la /scratch/hf_cache/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/*/
  du -sh /scratch/hf_cache/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/
  ```
  Unlike the target, this fits standalone on a single GPU
  (`tensor_model_parallel_size=1`, matching every other port's speculator
  scope) — place it on a GPU outside whatever range the target's TP group
  consumes (see `.env_exports.sh`'s own note and
  `vllm_offline_engine.py`'s speculator-device selection).

Unlike the sibling Gemma4 pipeline, there is no text-only checkpoint
variant to generate here — these are text-only checkpoints with no
multimodal encoder cache to strip (see `.env_exports.sh`'s own note).

Also verify, once these checkpoints are downloaded (not knowable from this
machine): whether Qwen3-Coder's chat template accepts an `enable_thinking`
kwarg at all (see `predict_longbench_v2.py`'s `render_chat` docstring —
these are reportedly non-thinking-only Instruct models, so the template may
simply not define that Jinja variable), and the real
`max_position_embeddings` for the 480B-A35B checkpoint (see that same
file's budget-clamping comment — not assumed to match the dense Qwen3-8B
port's 40960 value, likely much larger given Gemma4's precedent at this
MoE scale).

## 4. LongBench v2 dataset

`datasets/prep_longbench_v2.py` fetches `THUDM/LongBench-v2` from Hugging Face,
filters to the "short" (<32k word) subset, and writes
`datasets/longbench_v2_samples.jsonl`:

```bash
cd benchmarks/spec_prefill_qwen_coder
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
have been run for Qwen3-Coder yet** (unlike the sibling Gemma4 pipeline,
where step 5.2 has passed on real hardware):

1. **Without a GPU or model weights** — the engine-agnostic pieces (scoring
   math, `Request.is_prefill_chunk` batch split, KV-cache layout logic,
   `pruning_registry` lifecycle), unchanged from `../spec_prefill/`:
   ```bash
   cd benchmarks/spec_prefill_qwen_coder
   python3 test_vllm_patch.py
   ```
2. **On the GPU node, once step 3's `QWEN3_CODER_30B_MODEL_PATH` is set** —
   loads the real speculator standalone, installs the query-capture hook,
   and reports per-layer `head_dim`/`num_heads`/`num_kv_heads`. Expected
   (per `Qwen3MoeAttention`'s implementation, not yet confirmed on
   hardware): a uniform head_dim across all layers, unlike Gemma-4-E2B-it's
   confirmed heterogeneous 256/512 split — if Step A instead finds
   heterogeneous head_dims, treat that as a surprise worth investigating
   before trusting anything downstream, not as expected behavior:
   ```bash
   source .env_exports.sh   # this directory's local copy, see step 1
   python3 validate_proposer.py --model $QWEN3_CODER_30B_MODEL_PATH
   ```
3. **On the GPU node, once both checkpoints are set** — loads the *target*
   model through `worker_cls=vllm_patch.worker.SpecPrefillWorker` **with a
   real `--target-tensor-parallel-size` for the 480B-A35B checkpoint**
   (determine this from step 3's disk-size check and this node's GPU
   memory), confirms normal (non-pruned) generation still works, then
   directly checks the riskiest assumption in the whole design: does a
   pruned request's *original* token positions actually reach the model's
   real forward pass, or does the override silently fail to take effect?
   ```bash
   source .env_exports.sh   # this directory's local copy, sets
                             # TARGET_TENSOR_PARALLEL_SIZE=4 (see step 1
                             # and that file's own comment for why 4, not
                             # the AWQ checkpoint's own tested 8, is the
                             # current starting point)
   python3 validate_runner_integration.py \
       --target-model $QWEN3_CODER_480B_MODEL_PATH \
       --speculator-model $QWEN3_CODER_30B_MODEL_PATH
       # --target-tensor-parallel-size not needed -- defaults to 4 from
       # $TARGET_TENSOR_PARALLEL_SIZE above. --target-enable-expert-parallel
       # is NOT known to be required at TP=4 (unlike the AWQ checkpoint's
       # own tested TP=8, where its model card documents it as REQUIRED) --
       # add it if you hit an "expert tensors don't split evenly" error.
       # To try the checkpoint's own tested TP=8 config instead (once the
       # speculator's GPU placement at TP=8 is resolved, see
       # EXPERIMENT_PLAN.md's "Resource requirements"):
       #   --target-tensor-parallel-size 8 --target-enable-expert-parallel
   ```

All three have been carefully reasoned through against this fork's verified
V1 APIs and against the sibling Gemma4/dense-Qwen3 pipelines' own
(real-hardware-passing, for Gemma4; still unvalidated, for dense Qwen3)
versions of these scripts, but **not executed for Qwen3-Coder** — see each
script's own docstring ("Known risk areas" / "residual risk") for what's
most likely to need a fix on the first real run, and in particular for what
parts of the Gemma4/dense-Qwen3 pipelines' findings do vs. don't carry over
(e.g. whether the "vLLM warns against disabling chunked prefill" finding
applies here is a genuinely open question for this MoE target, unlike for
the dense Qwen3-8B port — see `validate_runner_integration.py`'s module
docstring).

## 6. Running an experiment

`predict_longbench_v2.py` runs the P001–P006 keep-rate sweep (see
`EXPERIMENT_PLAN.md`'s experiment matrix), with or without SpecPrefill
pruning, and writes a predictions JSONL per experiment that `grade_longbench_v2.py`
(step 4) scores:

```bash
cd benchmarks/spec_prefill_qwen_coder
python3 predict_longbench_v2.py --list           # print the experiment matrix
python3 predict_longbench_v2.py --exp P001 --max-keep 2   # smoke test first
python3 predict_longbench_v2.py --exp P001,P002,P003,P004,P005,P006
python3 grade_longbench_v2.py --samples datasets/longbench_v2_samples.jsonl \
    --predictions results/P002_predictions.jsonl
```

**Blocked on**: validation results from step 5 above (Step B/B2 in
`validate_runner_integration.py`), plus `predict_longbench_v2.py`'s own
real-hardware validation for Qwen3-Coder — see that script's docstring for
what's carried over from the sibling Gemma4/dense-Qwen3 pipelines'
confirmed findings vs. what still needs independent re-confirmation here.

## Expected runtime / hardware

No experiment has run yet on real hardware, but GPU sizing at 4-bit is now
derived (see `EXPERIMENT_PLAN.md`'s "Resource requirements" for the full
table): **Unlike the dense-Qwen3 port** (Qwen3-8B+1.7B, ~10B combined
parameters, expected to fit on a single GPU), this pipeline's target alone
(480B total/~35B active params, MoE) is far larger than the sibling Gemma4
pipeline's Gemma-4-26B-A4B-it+E2B (which needed 2x A100 / 160GB total HBM)
at BF16 or FP8 — but a 4-bit AWQ quant (~236GB, confirmed:
`QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ`) fits comfortably across
**8x A100 80GB** at `tensor_parallel_size=8` with
`--target-enable-expert-parallel` (that checkpoint's own tested config,
~30GB/GPU for target weights). The unresolved part is where the speculator
(kept unquantized, ~60GB) fits once the target's TP already occupies all 8
GPUs — see `EXPERIMENT_PLAN.md`'s two options (drop to `tensor_parallel_
size=4`, or quantize the speculator too) before running a real sweep.
Confirm all of this via step 5's validation scripts before trusting it.
