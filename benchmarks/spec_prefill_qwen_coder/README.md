# SpecPrefill Evaluation (Qwen3-Coder)

Evaluates **SpecPrefill** (draft-model-based prefill token preselection) against
Qwen3-Coder-480B-A35B-Instruct (MoE), using Qwen3-Coder-30B-A3B-Instruct (MoE)
as speculator, across a token keep-rate sweep (10/30/50/70/90%), on LongBench
v2's "short" (<32k word) question subset — measuring both accuracy retention
(≤5% drop vs. baseline) and QPS improvement.

Status: **a Qwen3-Coder port of `../spec_prefill_qwen/`'s dense-Qwen3 pipeline
(itself a port of `../spec_prefill/`'s Gemma4 pipeline) —
`vllm_patch/`'s Algorithm 1 implementation is code-complete but
unvalidated on real hardware** (written on a machine with no GPU, same
caveat as the pipelines this was ported from). Unlike the dense-Qwen3 port,
target and speculator here are MoE (`Qwen3MoeForCausalLM`) — confirmed this
required **zero `vllm_patch/` logic changes** (see `EXPERIMENT_PLAN.md`'s
"Implementation status" #1), only a genuinely new resource requirement: the
480B-A35B target does not fit on a single GPU at BF16 and needs real tensor
parallelism. For easier serving, a confirmed 4-bit AWQ quant exists
(`QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ`, ~236GB) that fits
comfortably on 8x A100 80GB at `tensor_parallel_size=8` — see
`EXPERIMENT_PLAN.md`'s "Resource requirements" for the full GPU-budget
table, the `--enable-expert-parallel` requirement that checkpoint's card
documents, and the speculator-GPU-placement trade-off it creates. See
`EXPERIMENT_PLAN.md`'s "Implementation status" for the full list of what
carried over unchanged vs. what's new for this port, and `REPRODUCE.md`
step 5 for the validation scripts to run first on the GPU node. The
original paper's reference implementation is cloned at
`../spec_prefill/speculative_prefill/` (not duplicated here — see that
directory's README for details; targets vLLM v0 + Llama models, not
directly runnable against this fork's V1 engine or Qwen3(-Coder) as-is).

Sibling to `../spec_prefill/` (Gemma4), `../spec_prefill_llama/` (Llama),
`../spec_prefill_qwen/` (dense Qwen3-8B/1.7B — the pipeline this one was
directly ported from), `../gemma4_moe_benchmarks/`, and
`../evaluation_pipeline/`; follows the same protocol-markdown +
`datasets/`/`results/` structure. Has its own `.env_exports.sh`, separate
from the ones those other pipelines use. Also used by
`../rlm_specprefill/` as the target/speculator backend for its Qwen-Coder
arm (see that directory's `.env_exports_qwen_coder.sh` /
`configs/arms_qwen_coder.yaml`).

## Contents

- `EXPERIMENT_PLAN.md` — the full protocol: motivation, algorithm reference, keep-rate
  sweep matrix (`P001`–`P006`), success criteria, resource requirements
  (multi-GPU target), and the implementation status (including what changed
  relative to `../spec_prefill_qwen/`'s dense-Qwen3 version).
- `REPRODUCE.md` — environment setup, checkpoint downloads, and validation steps.
- `.env_exports.sh` — local env config (model paths, HF token).
- `vllm_patch/` — the Algorithm 1 implementation (speculator loading, lookahead
  loop, scoring, pruning, and the `worker_cls`-based runner integration).
  Unchanged from `../spec_prefill_qwen/`'s version, including `proposer.py`'s
  query-capture hook and attention-layer discovery — confirmed
  `Qwen3MoeAttention.forward` is byte-identical to dense `Qwen3Attention.
  forward`, so no rewrite was needed for the MoE speculator here (see that
  file's module docstring for the full verification).
- `test_vllm_patch.py` / `validate_proposer.py` / `validate_runner_integration.py`
  — tests and GPU-node validation scripts, see `REPRODUCE.md` step 5.
  `validate_runner_integration.py` additionally gained
  `--target-tensor-parallel-size` and `--target-enable-expert-parallel`
  flags not present in the dense-Qwen3 port, since the 480B-A35B target
  requires the former and at least the QuantTrio AWQ checkpoint requires
  the latter at TP>1.
- `datasets/prep_longbench_v2.py` — downloads `THUDM/LongBench-v2` and filters to
  the "short" (<32k word) subset; see `REPRODUCE.md` step 4. Ported unchanged.
- `predict_longbench_v2.py` — runs the P001–P006 keep-rate sweep (with/without
  SpecPrefill pruning) against those samples and writes a predictions JSONL per
  experiment; see `REPRODUCE.md` step 6.
- `grade_longbench_v2.py` — scores a predictions file against `prep_longbench_v2.py`'s
  samples. Ported unchanged (no model-specific code).
- `datasets/` — LongBench v2 prompt sets (`longbench_v2_samples.jsonl`).
- `results/` — gitignored output directory.

Not ported from `../spec_prefill/`: `diag_worker.py`,
`repro_chunked_prefill_bug.py`, and `verify_sliding_window_hypothesis.py` —
these investigate bugs premised on Gemma4-specific architecture (a
multimodal chunked-prefill token floor; a sliding-window-vs-full-attention
scoring-corruption hypothesis premised on Gemma4's heterogeneous
attention-layer split) that don't structurally apply to Qwen3-Coder
(uniform full attention across all layers — MoE routing affects only the
`mlp`, not `self_attn`). See `EXPERIMENT_PLAN.md`'s "Implementation status" #7.
