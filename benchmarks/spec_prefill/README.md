# SpecPrefill Evaluation

Evaluates **SpecPrefill** (draft-model-based prefill token preselection) against
Gemma-4-26B-A4B-it, using Gemma-4-E2B-it as speculator, across a token keep-rate
sweep (10/30/50/70/90%), on LongBench v2's "short" (<32k word) question subset —
measuring both accuracy retention (≤5% drop vs. baseline) and QPS improvement.

Status: **protocol only — SpecPrefill mechanism not yet implemented in this vLLM
fork.** See `EXPERIMENT_PLAN.md`'s "Implementation status" section for the porting
checklist. A functional reference implementation from the paper's authors has been
cloned to `speculative_prefill/` for reference (targets vLLM v0 + Llama models —
not directly runnable against this fork's V1 engine / Gemma 4 as-is).

Sibling to `../gemma4_moe_benchmarks/` and `../evaluation_pipeline/`; follows the
same protocol-markdown + `datasets/`/`results/` structure.

## Contents

- `EXPERIMENT_PLAN.md` — the full protocol: motivation, algorithm reference, keep-rate
  sweep matrix (`P001`–`P006`), success criteria, and the implementation porting
  checklist.
- `REPRODUCE.md` — environment setup and reproduction steps.
- `speculative_prefill/` — cloned reference implementation (gitignored).
- `datasets/` — LongBench v2 prompt sets (empty — prep script is future work).
- `results/` — gitignored output directory.
