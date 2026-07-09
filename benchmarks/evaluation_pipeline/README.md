# Gemma4 MoE Speculative-Decoding Evaluation Pipeline

Measures task accuracy, QPS, end-to-end time, and speculative-decoding
(MTP) draft token acceptance rate/length for Gemma 4 26B-A4B-it, with and
without speculative decoding, across a draft-length (k) sweep.

Sibling to `../gemma4_moe_benchmarks/` (throughput-only ablation study);
this pipeline reuses its `LLM(...)` construction and env var conventions
but adds correctness scoring and spec-decode metric collection on top.

Status: scaffolding only — see file headers for what each component will
do. No implementation logic yet.

## Contents

- `EXPERIMENT_PLAN.md` — the dataset x spec-decode x k sweep matrix.
- `datasets/` — per-dataset sample preparation (AIME, LiveCodeBench, GPQA Diamond).
- `scorers/` — per-dataset correctness scoring.
- `metrics.py` — spec-decode acceptance rate / draft length collection.
- `run_pipeline.py` — the driver: initialize -> evaluate -> present.
- `analyze_results.py` — aggregates `results/*.json` into `results/summary.md`.
- `run_experiments.sh` — env setup + entrypoint wrapper.
- `REPRODUCE.md` — step-by-step reproduction instructions.
- `results/` — gitignored output directory.

## Usage (once implemented)

```bash
./run_experiments.sh --all
python3 analyze_results.py
```
