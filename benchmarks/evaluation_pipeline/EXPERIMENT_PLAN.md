# Evaluation Pipeline — Experiment Plan

Status: stub. To be filled in alongside `run_pipeline.py`'s `EXPERIMENTS` dict.

## Sweep matrix (planned)

- **Dataset** x **{aime, livecodebench, gpqa_diamond}**
- **Speculative decoding** x **{off, on}** (on = `spec_model=GEMMA4_ASSISTANT_MODEL_PATH`)
- **MTP draft length k** x **{1, 3, 5, 8}** (only applies when spec_decode=on)
- **Samples per dataset**: ~200-300 (exact count TBD, see README open questions)

## Metrics captured per run

No accuracy/correctness scoring -- throughput and spec-decode behavior
only:

- Requests per second (QPS)
- End-to-end wall time
- Draft token acceptance rate (`metrics.py`)
- Mean draft acceptance length (`metrics.py`)

## Expected signal

Higher k should increase draft length but generally lower per-token
acceptance rate; net QPS effect at each k, per dataset, is the primary
thing this pipeline measures. The three datasets are included to see
whether that tradeoff differs by prompt/response shape (e.g. long
reasoning chains for AIME vs. code generation vs. short multiple-choice
answers for GPQA), not to grade correctness.

## Relationship to `../gemma4_moe_benchmarks/EXPERIMENT_PLAN.md`

That plan's E001-E018 IDs cover throughput-only ablations (FP8, CUDA
graphs, batch size, gpu-mem). This plan is scoped narrowly to the
spec-decode/k/dataset axes relevant to throughput + acceptance rate and
does not reuse those experiment IDs.
