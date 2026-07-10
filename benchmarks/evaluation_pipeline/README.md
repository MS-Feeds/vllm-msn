# Gemma4 MoE Speculative-Decoding Evaluation Pipeline

Measures QPS, end-to-end time, speculative-decoding (MTP) draft token
acceptance rate/length, and Model FLOP/Bandwidth Utilization (MFU/MBU)
for Gemma 4 26B-A4B-it, with and without speculative decoding, across a
draft-length (k) sweep. **Throughput and spec-decode behavior only -- no
accuracy/correctness scoring.** AIME/LiveCodeBench/GPQA Diamond prompts
(see `datasets/`) are used purely as a realistic, varied-length prompt
mix to drive that measurement, not for grading model output.

Sibling to `../gemma4_moe_benchmarks/` (its own throughput-only ablation
study); this pipeline reuses its `LLM(...)` construction and env var
conventions, adding spec-decode metric collection (acceptance
rate/draft length) and hardware-utilization metrics (MFU/MBU) on top.

Status: fully implemented and validated end-to-end on real A100 80GB
hardware (real experiment runs, not just local unit tests) -- see file
headers for what each component does.

## Contents

- `EXPERIMENT_PLAN.md` — the dataset x spec-decode x k sweep matrix.
- `datasets/` — per-dataset prompt-set preparation (AIME, LiveCodeBench, GPQA Diamond).
- `metrics.py` — spec-decode acceptance rate / draft length collection.
- `hardware_metrics.py` — MFU/MBU, derived from the real Gemma4 MoE architecture.
- `run_pipeline.py` — the driver: initialize -> evaluate -> present.
- `analyze_results.py` — aggregates `results/*.json` into `results/summary.md`.
- `run_experiments.sh` — env setup + entrypoint wrapper.
- `REPRODUCE.md` — step-by-step reproduction instructions.
- `results/` — gitignored output directory.

## Usage

```bash
source ../gemma4_moe_benchmarks/.env_exports.sh
./run_experiments.sh --all
python3 analyze_results.py
```
