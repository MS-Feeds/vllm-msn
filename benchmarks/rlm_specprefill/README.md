# RLM + SpecPrefill Ablation

Does SpecPrefill (attention-transfer prefill token pruning) speed up an RLM
(Recursive Language Model) pipeline on massive (>131K-token) contexts, once
every other pipeline variable is held fixed?

- **Experimental design** (what to measure, arms A/B/C, the latency model,
  scope controls): [`../rlm/rlm_specprefill_ablation_plan.md`](../rlm/rlm_specprefill_ablation_plan.md).
- **Implementation plan** (how this directory is built, what's done vs. not,
  design decisions and why): [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).
- **Reproduction runbook** (GPU-node steps to actually run the sweep):
  [`REPRODUCE.md`](REPRODUCE.md).

This directory orchestrates two things that otherwise have zero code
coupling, without owning either:

- [`../rlm/`](../rlm/) — the RLM orchestration framework (root model + REPL +
  recursive sub-calls). A vendored dependency; not edited in place.
- A self-hosted SpecPrefill target/speculator pairing, one of:
  - [`../spec_prefill_llama/`](../spec_prefill_llama/) — Llama-3.1-8B
    (target) + Llama-3.2-1B (speculator), dense. Code-complete, not yet run
    on real GPU hardware. The **default** pairing (`.env_exports.sh`,
    `configs/arms.yaml`) — every example below assumes this pairing unless
    stated otherwise.
  - [`../spec_prefill_qwen_coder/`](../spec_prefill_qwen_coder/) —
    Qwen3-Coder-480B-A35B-Instruct (target) + Qwen3-Coder-30B-A3B-Instruct
    (speculator), MoE. Code-complete, not yet run on real GPU hardware; the
    480B-A35B target additionally requires `tensor_parallel_size > 1` (it
    doesn't fit on one GPU), unlike every other pairing here. Select it via
    `.env_exports_qwen_coder.sh` + `configs/arms_qwen_coder.yaml` +
    `runner/run_arm.py`'s `--spec-prefill-dir-env=SPEC_PREFILL_QWEN_CODER_DIR
    --target-tensor-parallel-size=<N>` flags — see "Running with a
    different target/speculator pairing" below.

Root model for this ablation is hosted Claude via the Anthropic API (root
doesn't need local attention access). Target + speculator are self-hosted
via vLLM's offline `LLM` class (SpecPrefill needs local attention access,
which only a self-hosted model can give it) — see above for which pairing.

## Layout

```
configs/       guardrails, SpecConfig, N_min, per-arm generation params
prompts/       the "RLM as retrieval front-end" system prompt + target-answer template
eval_data/     >131K-token eval-set prep (LongBench v2 "long" bucket + synthetic NIAH)
rlm_stage/     runs RLM to produce cached "candidate evidence" per query
target_stage/  drives vLLM's offline engine (plain and SpecPrefill-pruned) over that evidence
runner/        CLI entry points that run one arm / all arms end-to-end
calibration/   N_min crossover sweep + RLM-format transferability check
analysis/      aggregates per-arm results into the ablation doc's reported metrics
tests/         unit tests, GPU-independent where possible
results/       per-arm predictions + timing, and the evidence cache
```

## Status

All 11 build-order steps in `IMPLEMENTATION_PLAN.md` are done: scaffolding,
`eval_data/` prep, the evidence-extraction prompt + RLM stage, the evidence
cache/replay layer, the RLM-trajectory timing decomposition, `target_stage/`
(vLLM offline engine, gate, query routing), `runner/` (CLI entry points),
`calibration/` (N_min sweep, transferability check), and `analysis/`
(metrics aggregation). **85/85 unit tests pass**, plus real Anthropic API
smoke tests (`rlm_stage/evidence_rlm.py --smoke-test`, `runner/smoke_test.py`)
and a real end-to-end `run_arm.py --arm A --dry-run` CLI run confirming the
evidence-cache confound-control mechanism works in practice (cache miss on
first pass, cache hit on second). **Nothing past the evidence-collection
stage has been run against a GPU or vLLM** — this was written on a machine
with neither — see `REPRODUCE.md` for the GPU-node validation steps that
still need to happen before Arms A/B/C can be trusted end-to-end. See
`IMPLEMENTATION_PLAN.md`'s Verification section for the exact list of
what's been checked vs. not, and its load-bearing findings for real bugs
discovered along the way (in a vendored dependency, and a gap in
`predict_longbench_v2.py`'s own instrumentation pattern that had to be
fixed for `f` to be computable per-sample).

## Running with a different target/speculator pairing

`target_stage/vllm_offline_engine.py`, `runner/run_arm.py`,
`runner/run_all_arms.py`, and `calibration/{sweep_n_min,
transferability_check}.py` all default to the original Llama-3.1-8B/3.2-1B
pairing (unchanged behavior, zero new flags required) but accept two
optional arguments to target a different SpecPrefill port instead:

- `--spec-prefill-dir-env` — which env var (set by that port's own
  `.env_exports*.sh`) points at the sibling SpecPrefill port directory
  whose `vllm_patch/` to import. Default `SPEC_PREFILL_LLAMA_DIR`.
- `--target-tensor-parallel-size` — passed straight through to vLLM's
  `LLM(...)`. Falls back to `1` (every existing pairing's actual
  requirement) unless `$TARGET_TENSOR_PARALLEL_SIZE` is set --
  `.env_exports_qwen_coder.sh` sets it to `4` (see that file's own comment
  / `../spec_prefill_qwen_coder/EXPERIMENT_PLAN.md`'s "Resource
  requirements" for why 4, not the AWQ checkpoint's own tested 8, is the
  current starting point: at TP=8 the target occupies every GPU on an
  8-GPU node, leaving none for the speculator).
- `--target-enable-expert-parallel` — passed straight through to vLLM's
  `LLM(...)` as `enable_expert_parallel`. Default off; required by at
  least some quantized MoE checkpoints at TP>1 (e.g. QuantTrio's AWQ quant
  of Qwen3-Coder-480B-A35B-Instruct documents this as REQUIRED at
  `tensor_parallel_size=8`, "otherwise the expert tensors cannot be evenly
  split across tensor parallel ranks" — their model card's own wording).
  Not known to be required at TP=4 (the current default) — add it if you
  hit that same "expert tensors don't split evenly" error.

For the Qwen3-Coder-480B-A35B/30B-A3B pairing (4-bit AWQ target, e.g. on
8x A100 80GB — see `../spec_prefill_qwen_coder/EXPERIMENT_PLAN.md`'s
"Resource requirements" for the GPU-budget table and the
speculator-placement trade-off at `tensor_parallel_size=8`):

```bash
source .env_exports_qwen_coder.sh   # sets SPEC_PREFILL_QWEN_CODER_DIR,
                                     # QWEN3_CODER_480B_MODEL_PATH,
                                     # TARGET_TENSOR_PARALLEL_SIZE=4, etc.
python3 runner/run_arm.py --arm A \
    --dataset eval_data/<your-dataset>.jsonl \
    --target-model "$QWEN3_CODER_480B_MODEL_PATH" \
    --speculator-model "$QWEN3_CODER_30B_MODEL_PATH" \
    --spec-prefill-dir-env SPEC_PREFILL_QWEN_CODER_DIR
    # --target-tensor-parallel-size not needed -- defaults to 4 from
    # $TARGET_TENSOR_PARALLEL_SIZE above. Override with an explicit flag
    # (e.g. --target-tensor-parallel-size 8 --target-enable-expert-parallel,
    # the AWQ checkpoint's own tested config) once the speculator's GPU
    # placement at TP=8 is resolved -- see the "Resource requirements" link
    # above.
```

`configs/arms_qwen_coder.yaml` documents this pairing's per-arm settings
(mirrors `configs/arms.yaml`); Arm B/C load
`configs/spec_config_always_on_qwen_coder.yaml` (chunk_size=64, matching
that port's own predict script) rather than the Llama pairing's
`spec_config_always_on.yaml` (chunk_size=32) — these two SpecConfig files
are deliberately different, not a discrepancy to "fix". See
`../spec_prefill_qwen_coder/README.md`/`REPRODUCE.md` for that port's own
setup and unresolved validation items before trusting Arm B/C results.
