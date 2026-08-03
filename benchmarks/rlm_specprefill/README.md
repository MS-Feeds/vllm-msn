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
- [`../spec_prefill_llama/`](../spec_prefill_llama/) — the SpecPrefill port for
  vLLM's V1 engine, targeting Llama-3.1-8B (target) + Llama-3.2-1B (speculator).
  Code-complete, not yet run on real GPU hardware.

Root model for this ablation is hosted Claude via the Anthropic API (root
doesn't need local attention access). Target + speculator are self-hosted
Llama via vLLM's offline `LLM` class (SpecPrefill needs local attention
access, which only a self-hosted model can give it).

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
