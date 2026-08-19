# Top-K KV Cache Selection for Multi-Turn Conversation

Evaluates whether **SpecPrefill** (draft-model-based prefill token
preselection) generalizes from single-shot long-context QA to long, growing
**multi-turn conversations**, where context/prompt size is the serving
bottleneck across every turn, not just once. Target: Llama-3.1-8B-Instruct.
Speculator: Llama-3.2-1B-Instruct. Dataset: SCBench (3 configs:
`scbench_qa_eng`/`scbench_kv`/`scbench_summary`). Grid: keep-rate x
KV-entry-granularity, plus an oracle upper bound, at the protocol's `KEEP`
history-retention setting first.

Status: **code-complete but unvalidated on real hardware** (no GPU on the
machine this was written on). Built directly on top of
`../spec_prefill_llama/`'s single-turn SpecPrefill port -- reuses its
architecture-agnostic pieces verbatim (`scoring.py`, `kv_cache_utils.py`,
`prefill_split.py`, `pruning_registry.py`, `model_runner.py`, `worker.py`)
and adds the multi-turn-specific pieces on top: a per-conversation
absolute-position ledger (`vllm_patch/conversation_state.py`) and a
persistent speculator engine (`vllm_patch/speculator_worker.py`) -- see
`EXPERIMENT_PLAN.md`'s "Key architectural decisions" for the full reasoning,
and `REPRODUCE.md`'s validation steps for what to run first on a GPU node.
The multi-turn extension has NO precedent in any sibling `spec_prefill*`
pipeline (all single-turn) -- treat it as less validated than even the
already-unvalidated single-turn baseline it's built from.

Sibling to `../spec_prefill_llama/` (the single-turn Llama pipeline this was
built from), `../spec_prefill/`/`../spec_prefill_qwen/`/
`../spec_prefill_qwen_coder/` (other single-turn SpecPrefill ports); follows
the same protocol-markdown + `datasets/`/`results/` structure. Has its own
`.env_exports.sh`, separate from those other pipelines'.

## Contents

- `EXPERIMENT_PLAN.md` — the full protocol: motivation, architectural
  decisions (golden-context mode, persistent speculator engine,
  KEEP/DISCARD candidate pools, oracle upper bound), experiment matrix
  (`M000`/`M-k*-g*`/`ORACLE-k*`), success criteria, implementation status.
- `REPRODUCE.md` — environment setup, checkpoint downloads, validation steps.
- `.env_exports.sh` — local env config (model paths, HF token).
- `vllm_patch/` — the multi-turn Algorithm 1 implementation. See its own
  `__init__.py` module map for exactly which files are copied verbatim from
  `../spec_prefill_llama/vllm_patch/`, which are new (`conversation_state.py`,
  `speculator_worker.py`), and which are rewritten (`proposer.py`,
  `pruner.py`, `config.py`).
- `test_vllm_patch.py` — unit tests for the engine-agnostic pieces, including
  `conversation_state.py`'s KEEP/DISCARD logic (no GPU needed -- confirmed
  passing, 19/19, in a CPU-only environment).
- `validate_proposer.py` / `validate_runner_integration.py` — GPU-node
  validation scripts, see `REPRODUCE.md` step 4.
- `datasets/prep_scbench.py` — downloads `microsoft/SCBench`'s 3 MVP
  configs; see `REPRODUCE.md` step 3.
- `predict_scbench.py` — runs the experiment matrix (per-conversation,
  per-turn sequential driving loop -- see its own docstring for why this
  differs structurally from the single-turn pipelines' flat, all-samples-
  at-once driving loop); writes a per-turn predictions JSONL per experiment.
- `grade_scbench.py` — scores predictions against `prep_scbench.py`'s
  samples, with per-SCBench-config metrics ported from the official
  `microsoft/MInference/scbench/eval_utils.py`.
- `flops_model.py` — analytic FLOP model for the combined speculator +
  target system, driven by per-turn token counts measured during the run.
  Answers what wall-clock can't: whether the compute the speculator adds is
  smaller than the compute the target saves. Wired into `predict_scbench.py`
  (per-turn `flops` in the predictions JSONL, per-stage TFLOP columns in
  `all_runs.csv`); pure Python, no GPU needed.
- `validate_flops_model.py` — GPU-node validation of that model, without
  `ncu`: `torch.profiler(with_flops=True)` for the dense term, a
  cross-check of the sparse gather's attended lengths against
  `sparse_decode_microbench.py`'s independent block counts for the
  attention term, and an above-peak falsification bound.
- `datasets/` — SCBench prompt sets (gitignored).
- `results/` — gitignored output directory.

Not implemented in this pass (confirmed out of scope with the user): the
protocol's baseline comparison methods (H2O, StreamingLLM, Quest, KVzip,
HeadKV) and head-level (rather than token-level) selection -- see
`EXPERIMENT_PLAN.md`'s decision #7 for the seam left for adding them later.
Also not yet wired up: the oracle-upper-bound row's driving loop (the
scoring core is ready, `vllm_patch/pruner.py::compute_oracle_kept_pairs`;
`predict_scbench.py`'s oracle branch raises `NotImplementedError` with
details on what's missing).
