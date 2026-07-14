# Evaluation Pipeline — Experiment Plan

Status: stub. To be filled in alongside `run_pipeline.py`'s `EXPERIMENTS` dict.

## Sweep matrix (planned)

- **Dataset** x **{aime, livecodebench, gpqa_diamond}**
- **Speculative decoding** x **{off, on}** (on = `spec_model=GEMMA4_ASSISTANT_MODEL_PATH`)
- **MTP draft length k** x **{1, 2, 3, 4, 5}** (only applies when spec_decode=on)
- **Samples per dataset**: ~200-300 (exact count TBD, see README open questions)

## Metrics captured per run

No accuracy/correctness scoring -- throughput and spec-decode behavior
only:

- Requests per second (QPS)
- End-to-end wall time
- Draft token acceptance rate (`metrics.py`)
- Mean draft acceptance length (`metrics.py`)
- Model FLOP Utilization / Model Bandwidth Utilization (`hardware_metrics.py`)
  -- "n/a" on unrecognized GPUs (see `hardware_metrics.GPU_SPECS`); for
  spec_decode=True rows these are a lower bound, since the MTP draft
  model's own forward pass isn't counted (see module docstring)

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

## Relationship to `EXPERIMENT_PLAN_MAX_NUM_SEQS.md` and `EXPERIMENT_PLAN_MNS_SPEC_CROSS.md`

This is the `spec` suite (`S0xx`) in `run_pipeline.py`'s three-suite
`--suite {spec,batch,cross}` split. The sibling `batch` suite (`B0xx`,
see `EXPERIMENT_PLAN_MAX_NUM_SEQS.md`) sweeps `max_num_seqs` instead,
with spec decode held off, on the same driver/datasets/metrics. The
`cross` suite (`X0xx`, see `EXPERIMENT_PLAN_MNS_SPEC_CROSS.md`) crosses
both axes at once (`max_num_seqs` x MTP `k`, spec decode on). All three
are mutually exclusive per invocation — pick a suite per invocation, no
`--all` ever spans more than one.
