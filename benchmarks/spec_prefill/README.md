# SpecPrefill Evaluation

Evaluates **SpecPrefill** (draft-model-based prefill token preselection) against
Gemma-4-26B-A4B-it, using Gemma-4-E2B-it as speculator, across a token keep-rate
sweep (10/30/50/70/90%), on LongBench v2's "short" (<32k word) question subset —
measuring both accuracy retention (≤5% drop vs. baseline) and QPS improvement.

Status: **`vllm_patch/`'s Algorithm 1 implementation is code-complete but
unvalidated on real hardware** (written on a machine with no GPU). See
`EXPERIMENT_PLAN.md`'s "Implementation status" for what's confirmed vs. still
open, and `REPRODUCE.md` step 5 for the validation scripts to run first on the
GPU node. A functional reference implementation from the paper's authors has
been cloned to `speculative_prefill/` for reference (targets vLLM v0 + Llama
models — not directly runnable against this fork's V1 engine / Gemma 4 as-is;
`vllm_patch/` is the from-scratch V1/Gemma4 port, not an adaptation of it).

Sibling to `../gemma4_moe_benchmarks/` and `../evaluation_pipeline/`; follows the
same protocol-markdown + `datasets/`/`results/` structure. Has its own
`.env_exports.sh`, separate from the one those two pipelines share.

## Contents

- `EXPERIMENT_PLAN.md` — the full protocol: motivation, algorithm reference, keep-rate
  sweep matrix (`P001`–`P006`), success criteria, and the implementation status.
- `REPRODUCE.md` — environment setup, checkpoint downloads, and validation steps.
- `.env_exports.sh` — local env config (model paths, HF token).
- `vllm_patch/` — the Algorithm 1 implementation (speculator loading, lookahead
  loop, scoring, pruning, and the `worker_cls`-based runner integration).
- `test_vllm_patch.py` / `validate_proposer.py` / `validate_runner_integration.py`
  — tests and GPU-node validation scripts, see `REPRODUCE.md` step 5.
- `speculative_prefill/` — cloned reference implementation (gitignored).
- `datasets/` — LongBench v2 prompt sets (empty — prep script is future work).
- `results/` — gitignored output directory.
