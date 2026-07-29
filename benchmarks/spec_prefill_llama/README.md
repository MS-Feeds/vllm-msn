# SpecPrefill Evaluation (Llama)

Evaluates **SpecPrefill** (draft-model-based prefill token preselection) against
Llama-3.1-8B, using Llama-3.2-1B as speculator, across a token keep-rate
sweep (10/30/50/70/90%), on LongBench v2's "short" (<32k word) question subset —
measuring both accuracy retention (≤5% drop vs. baseline) and QPS improvement.

Status: **a Llama port of `../spec_prefill_qwen/`'s Qwen3 pipeline (itself a
port of `../spec_prefill/`'s Gemma4 pipeline) — `vllm_patch/`'s Algorithm 1
implementation is code-complete but unvalidated on real hardware** (written
on a machine with no GPU, same caveat as the pipelines this was ported
from). See `EXPERIMENT_PLAN.md`'s "Implementation status" for what carried
over unchanged vs. what was rewritten for Llama's attention architecture,
and `REPRODUCE.md` step 5 for the validation scripts to run first on the GPU
node. The original paper's reference implementation is cloned at
`../spec_prefill/speculative_prefill/` (not duplicated here — see that
directory's README for details; targets vLLM v0 + Llama models, but is not
directly runnable against this fork's V1 engine or this `worker_cls`-based
integration as-is).

Sibling to `../spec_prefill/` (the Gemma4 version of this same pipeline),
`../spec_prefill_qwen/` (the Qwen3 version), `../gemma4_moe_benchmarks/`, and
`../evaluation_pipeline/`; follows the same protocol-markdown +
`datasets/`/`results/` structure. Has its own `.env_exports.sh`, separate
from the ones those other pipelines use.

## Contents

- `EXPERIMENT_PLAN.md` — the full protocol: motivation, algorithm reference, keep-rate
  sweep matrix (`P001`–`P006`), success criteria, and the implementation status
  (including what changed relative to `../spec_prefill_qwen/`'s Qwen3 version).
- `REPRODUCE.md` — environment setup, checkpoint downloads, and validation steps.
- `.env_exports.sh` — local env config (model paths, HF token).
- `vllm_patch/` — the Algorithm 1 implementation (speculator loading, lookahead
  loop, scoring, pruning, and the `worker_cls`-based runner integration).
  Mostly unchanged from `../spec_prefill_qwen/`'s version — only `proposer.py`'s
  query-capture hook and attention-layer discovery were rewritten against
  Llama's attention implementation, which is simpler still than Qwen3's (no
  q_norm/k_norm step).
- `test_vllm_patch.py` / `validate_proposer.py` / `validate_runner_integration.py`
  — tests and GPU-node validation scripts, see `REPRODUCE.md` step 5.
- `datasets/prep_longbench_v2.py` — downloads `THUDM/LongBench-v2` and filters to
  the "short" (<32k word) subset; see `REPRODUCE.md` step 4. Ported unchanged.
- `predict_longbench_v2.py` — runs the P001–P006 keep-rate sweep (with/without
  SpecPrefill pruning) against those samples and writes a predictions JSONL per
  experiment; see `REPRODUCE.md` step 6.
- `grade_longbench_v2.py` — scores a predictions file against `prep_longbench_v2.py`'s
  samples. Ported unchanged (no model-specific code).
- `datasets/` — LongBench v2 prompt sets (`longbench_v2_samples.jsonl`).
- `results/` — gitignored output directory.

Not ported from `../spec_prefill/`/`../spec_prefill_qwen/`: `diag_worker.py`,
`repro_chunked_prefill_bug.py`, and `verify_sliding_window_hypothesis.py` —
these investigate bugs premised on Gemma4-specific architecture (a
multimodal chunked-prefill token floor; a sliding-window-vs-full-attention
scoring-corruption hypothesis) that don't structurally apply to Llama
(dense, non-multimodal, uniform full attention across all layers). See
`EXPERIMENT_PLAN.md`'s "Implementation status" #6.
