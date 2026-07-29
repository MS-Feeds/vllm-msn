# Gemma 4 MoE FP8 — Ablation Experiment Plan

**Hardware target**: A100 80 GB (sm_80)  
**Dataset**: `datasets/sc1_delta_v2.jsonl` (sc1 only)  
**Driver**: `bench_experiment.py` via `run_experiments.sh`  
**Results**: default `results/all_runs.csv` + per-run JSON; completed 40G mock run at `results_A100_40G_mock/all_runs.csv`  
**Analysis**: `python3 analyze_results.py` → `results/summary.md`

---

## Operational guardrails (must-pass preflight)

- Use the intended Python environment explicitly. This benchmark shell wrapper now
    checks `vllm`, `numpy`, and `scipy` imports before running any experiment.
- On this A100 setup, a known compatible stack is `vllm==0.22.0`, `numpy<2`,
    and `scipy>=1.14,<2`.
- If `GEMMA4_TEXT_ONLY_MODEL_PATH` equals `GEMMA4_MODEL_PATH`, text-only
    experiments silently become full-model runs. Build a dedicated text-only
    checkpoint first (for example via `examples/create_text_only_model.py`).
- Treat shell "PASSED" as valid only when `bench_experiment.py` exits 0. The
    driver now returns non-zero when any experiment throws.

---

## Fixed scenario parameters (sc1)

| Parameter | Value | Source |
|---|---|---|
| Dataset | `datasets/sc1_delta_v2.jsonl` | `prep_dataset.py --max-keep 1000` |
| `num_prompts` | 1 000 | ablation-sized subset of full 10 000 |
| `output_len` (max_tokens) | 8 192 | matches REPRODUCE_PRODSHAPE |
| `max_model_len` | 24 576 | matches REPRODUCE_PRODSHAPE |
| `max_num_batched_tokens` | 16 384 | matches REPRODUCE_PRODSHAPE |
| Sampling | temp=0.7, top_p=0.95, ignore_eos=False | consistent with bench_offline |
| Reps per experiment | 2 | mean ± σ across reps |
| Attention backend | **TRITON_ATTN** (forced by vLLM) | Gemma 4 heterogeneous head dims |

> **Why TRITON_ATTN is forced**: Gemma 4 has heterogeneous attention head dimensions
> (256 and 512 in different layers). vLLM cannot route a mixed-head-dim model through
> FLASH_ATTN or FLASHINFER, so it falls back to TRITON_ATTN unconditionally regardless
> of the `VLLM_ATTENTION_BACKEND` env var. Setting that var is a no-op for this model.

---

## Branch-level model optimizations (already in this branch)

These are code-level optimizations in the Gemma4 model path and are active
for all experiments in this plan. They are not toggled per experiment ID.

| Optimization | Where | Why it matters |
|---|---|---|
| Fused dual-RMSNorm Triton kernel (with residual/scalar fusion) | `vllm/model_executor/layers/gemma4_fused_ops.py` | Reduces launch overhead and memory traffic in MoE normalization path. |
| Fused path coverage guard + 3-D support | `vllm/model_executor/models/gemma4.py` | Applies fused RMSNorm path for both 2-D and 3-D tensors (flatten/reshape), while requiring hidden-dim contiguous layout for correctness. |
| Remove redundant V projection on `k_eq_v` full-attention layers | `vllm/model_executor/models/gemma4.py` | Avoids unnecessary QKV work/weights when V is derived from K-equivalent path. |
| Gemma4 multimodal guardrails for text-only checkpoints | `vllm/model_executor/models/gemma4_mm.py` | Prevents text-only runs from triggering vision/video initialization failures. |

Interpretation note:
- The ablation matrix below measures runtime config deltas (FP8, CUDA graphs,
  MTP, mns, gpu mem, text-only). The above kernel/model-path optimizations are
  a shared baseline for all rows.

---

## Dataset preparation

The source file is `/nvmedata/data/layer1_delta_20260501.txt` (859,988 JSONL rows,
each with `{"messages": [{"role":"system",...}, {"role":"user",...}]}`).

> **Note**: `prep_dataset.py` filters on `_export_prompt: true` and will yield
> 0 records from the raw `.txt` file. Use the simpler direct-conversion command
> below instead:

```bash
cd benchmarks/gemma4_moe_benchmarks

python3 - <<'EOF'
import json, sys
from pathlib import Path
from transformers import AutoTokenizer

src  = "/nvmedata/data/layer1_delta_20260501.txt"
dst  = "datasets/sc1_delta_v2.jsonl"
model = "/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it"
max_tokens = 16384   # max_model_len(24576) - output_len(8192)
max_keep   = 1000

tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
Path(dst).parent.mkdir(exist_ok=True)
kept, skipped = 0, 0
with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except: continue
        msgs = d.get("messages", [])
        if not msgs: continue
        # fold system + user into a single user turn
        parts = []
        for m in msgs:
            c = m.get("content","")
            if isinstance(c, list): c = "".join(p.get("text","") for p in c if isinstance(p,dict))
            if m.get("role") == "system": parts.append(f"[SYSTEM]\n{c}")
            else: parts.append(c)
        text = "\n\n".join(parts)
        rendered = tok.apply_chat_template([{"role":"user","content":text}],
                                           add_generation_prompt=True, tokenize=False)
        n = len(tok(rendered, add_special_tokens=False).input_ids)
        if n > max_tokens: skipped += 1; continue
        fout.write(json.dumps({"prompt": text}, ensure_ascii=False) + "\n")
        kept += 1
        if kept >= max_keep: break
print(f"kept={kept}  skipped_too_long={skipped}")
EOF
```

This is equivalent to `prep_dataset.py` but reads the raw format directly.
Result: `datasets/sc1_delta_v2.jsonl` with 1000 prompts, all ≤ 16384 tokens rendered.

---

## Experiment matrix

### Group A — Reproduce REPRODUCE_PRODSHAPE baseline

Goal: verify the sc1 numbers from REPRODUCE_PRODSHAPE.md land in the right ballpark on A100
(H100 NVL reference: bf16=1870 out tok/s, FP8=2056 out tok/s — A100 will be lower).

| ID | Label | quant | KV dtype | eager | MTP | mns | gpu_mem | model |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E001** | BF16 baseline — matches REPRODUCE_PRODSHAPE sc1 | bf16 | auto | ✓ | ✗ | 128 | 0.90 | full |
| **E002** | +FP8 weights (KV cache stays auto/bf16) | fp8 | auto | ✓ | ✗ | 128 | 0.90 | full |
| **E003** | +FP8 KV cache (fp8_e4m3) — **FAIL expected on A100** | fp8 | fp8_e4m3 | ✓ | ✗ | 128 | 0.90 | full |

**E003 note**: `fp8_e4m3` KV cache requires Triton `fp8e4nv` which is not supported on
sm_80. The expected result is a hard error. On H100 (sm_90) this is the FP8 run from
REPRODUCE_PRODSHAPE. Recording the failure on A100 IS the result.

---

### Group B — Incremental optimizations (stack-up)

Each experiment adds one technique on top of the previous best.

| ID | Label | quant | KV dtype | eager | MTP | mns | gpu_mem | model | Builds on |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E004** | +CUDA graphs | fp8 | auto | ✗ | ✗ | 128 | 0.90 | full | E002 |
| **E005** | +MTP speculative decoding (k=5) | fp8 | auto | ✗ | ✓ k=5 | 128 | 0.90 | full | E004 |
| **E006** | +text-only model (vision tower stripped) | fp8 | auto | ✗ | ✓ k=5 | 128 | 0.90 | text_only | E005 |

**E006 is the "best-so-far" config** that Groups C, D, E branch from.

---

### Group C — Batch size (`max_num_seqs`) sweep

Base config: E006 (FP8, CUDA graphs, MTP k=5, text-only, gpu_mem=0.90).  
Question: is mns=128 optimal, or does A100 saturate earlier/later?

| ID | Label | mns | All other params |
|---|---|:---:|---|
| **E007** | batch sweep: mns=64 | 64 | same as E006 |
| E006 | *(control, mns=128)* | 128 | — |
| **E008** | batch sweep: mns=192 | 192 | same as E006 |
| **E009** | batch sweep: mns=256 | 256 | same as E006 |

---

### Group D — GPU memory utilization sweep

Base config: E006 (mns=128).  
Question: does giving vLLM more KV cache headroom help on A100?

| ID | Label | gpu_mem | All other params |
|---|---|:---:|---|
| **E010** | gpu_mem sweep: 0.80 | 0.80 | same as E006 |
| E006 | *(control, gpu_mem=0.90)* | 0.90 | — |
| **E011** | gpu_mem sweep: 0.95 | 0.95 | same as E006 |

---

### Group E — Isolation (ablate single contributions)

Base config: E006. Each experiment turns off exactly one optimization to measure its
isolated contribution.

| ID | Label | What is turned off | quant | eager | MTP | mns | gpu_mem | model |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E012** | no MTP at optimal | MTP disabled | fp8 | ✗ | ✗ | 128 | 0.90 | text_only |
| **E013** | no CUDA graphs at optimal | CUDA graphs disabled | fp8 | ✓ | ✓ k=5 | 128 | 0.90 | text_only |
| **E014** | BF16 weights at optimal | FP8 weights removed | bf16 | ✗ | ✓ k=5 | 128 | 0.90 | text_only |
| **E015** | BF16 reference (text-only, no opts) | All opts off | bf16 | ✓ | ✗ | 128 | 0.90 | text_only |
| **E016** | BF16 + CUDA graphs only | CUDA graphs only | bf16 | ✗ | ✗ | 128 | 0.90 | text_only |

**Isolation pairs** (E006 is the "on" state, column is the "off" state):

| Contribution measured | ON | OFF | Expected sign |
|---|:---:|:---:|:---:|
| MTP k=5 | E006 | E012 | E006 > E012 |
| CUDA graphs (on FP8+MTP) | E006 | E013 | E006 ≥ E013 (may regress on heterogeneous batch) |
| CUDA graphs (on BF16 only) | E016 | E015 | E016 ≥ E015 |
| FP8 weights | E006 | E014 | E006 > E014 |
| text-only model vs full | E006 | E005 | E006 > E005 |

---

## Complete config table

| ID | Group | quant | KV dtype | eager | MTP k | mns | gpu_mem | model |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E001 | A | bf16 | auto | ✓ | — | 128 | 0.90 | full |
| E002 | A | fp8 | auto | ✓ | — | 128 | 0.90 | full |
| E003 | A | fp8 | fp8_e4m3 | ✓ | — | 128 | 0.90 | full |
| E004 | B | fp8 | auto | ✗ | — | 128 | 0.90 | full |
| E005 | B | fp8 | auto | ✗ | 5 | 128 | 0.90 | full |
| E006 | B | fp8 | auto | ✗ | 5 | 128 | 0.90 | text_only |
| E007 | C | fp8 | auto | ✗ | 5 | **64** | 0.90 | text_only |
| E008 | C | fp8 | auto | ✗ | 5 | **192** | 0.90 | text_only |
| E009 | C | fp8 | auto | ✗ | 5 | **256** | 0.90 | text_only |
| E010 | D | fp8 | auto | ✗ | 5 | 128 | **0.80** | text_only |
| E011 | D | fp8 | auto | ✗ | 5 | 128 | **0.95** | text_only |
| E012 | E | fp8 | auto | ✗ | **—** | 128 | 0.90 | text_only |
| E013 | E | fp8 | auto | **✓** | 5 | 128 | 0.90 | text_only |
| E014 | E | **bf16** | auto | ✗ | 5 | 128 | 0.90 | text_only |
| E015 | E | **bf16** | auto | **✓** | **—** | 128 | 0.90 | text_only |
| E016 | E | **bf16** | auto | ✗ | **—** | 128 | 0.90 | text_only |

Bold = the parameter(s) that differ from E006.

---

## Current execution status — A100 40G mock

This branch now has a completed **A100 40G mock** result set in:

- `benchmarks/gemma4_moe_benchmarks/results_A100_40G_mock/all_runs.csv`
- `benchmarks/gemma4_moe_benchmarks/results_A100_40G_mock/all_runs.md`

These runs were launched with `--mock-a100-40g`, which applies `GPU_MEM_SCALE=0.5`.
As a result, the effective `gpu_memory_utilization` values in the recorded runs are:

- nominal `0.90` → effective `0.45`
- nominal `0.95` → effective `0.475`

Completed experiments in this folder:

- E002
- E004
- E005
- E006
- E007
- E008
- E009
- E011
- E012
- E013

Not present in the current `results_A100_40G_mock` folder:

- E001, E003
- E010
- E014, E015, E016

### Aggregated results (mean over 2 reps)

| ID | Label | Mean req/s | Mean output tok/s | Mean total tok/s | Mean elapsed (s) |
|---|---|---:|---:|---:|---:|
| E002 | +FP8 weights (kv cache stays BF16 / auto) | 0.2026 | 258.04 | 1141.01 | 4937.003 |
| E004 | +CUDA graphs | 0.5164 | 661.79 | 2912.43 | 1936.470 |
| E005 | +MTP speculative decoding (k=5) | 0.7851 | 999.45 | 4421.12 | 1273.890 |
| E006 | +text-only model (vision stripped) | 0.9010 | 1167.49 | 5094.31 | 1110.276 |
| E007 | batch sweep: mns=64 | 0.9607 | 1245.32 | 5432.67 | 1040.925 |
| E008 | batch sweep: mns=192 | 0.9140 | 1200.18 | 5183.51 | 1094.206 |
| E009 | batch sweep: mns=256 | 0.8940 | 1142.20 | 5038.42 | 1118.604 |
| E011 | gpu_mem sweep: 0.95 | **1.0901** | **1412.13** | **6163.29** | **917.365** |
| E012 | no MTP at optimal | 0.5459 | 696.60 | 3075.76 | 1831.988 |
| E013 | no CUDA graphs at optimal | 0.6253 | 805.12 | 3530.41 | 1599.209 |

### Main findings from the completed 40G mock sweep

| Finding | Evidence |
|---|---|
| Best completed config is **E011** | Highest req/s (`1.0901`), output tok/s (`1412.13`), and total tok/s (`6163.29`) among completed runs. |
| End-to-end stackup over E002 is large | E011 vs E002: `+438.1%` req/s, `+447.3%` output tok/s, `+440.2%` total tok/s. |
| MTP is a major contributor | E006 vs E012: `+65.0%` req/s with MTP enabled. |
| CUDA graphs are also material | E006 vs E013: `+44.1%` req/s with CUDA graphs enabled. |
| Text-only conversion helps further | E006 vs E005: `+14.8%` req/s after stripping vision path overhead. |
| For `max_num_seqs`, **64** beat 128/192/256 in this 40G mock run | E007 outperformed E006/E008/E009 on req/s and total tok/s. |

Interpretation note:

- The current best result is **within the completed subset only**. Since E010 and the BF16 isolation runs are absent from this folder, this is not yet a mathematically complete 16-experiment comparison set.

---

## Running the experiments

### Prepare dataset (once)

See the **Dataset preparation** section above for the full inline Python command.
Quick reference:
```bash
# From benchmarks/gemma4_moe_benchmarks/
# Run the python3 here-doc in the "Dataset preparation" section.
# Source: /nvmedata/data/layer1_delta_20260501.txt
# Output: datasets/sc1_delta_v2.jsonl  (1000 prompts, all ≤ 16384 tokens)
```

### Run all experiments
```bash
./run_experiments.sh --all --scenario sc1 --reps 2
```

### Run a single experiment
```bash
./run_experiments.sh E001 --scenario sc1 --reps 2
```

### Run a group
```bash
./run_experiments.sh E001,E002,E003 --scenario sc1 --reps 2   # Group A
./run_experiments.sh E004,E005,E006 --scenario sc1 --reps 2   # Group B
```

### Analyze results
```bash
python3 analyze_results.py
# Output: results/summary.md
```

---

## Environment variables (set automatically by run_experiments.sh)

| Variable | Value | Notes |
|---|---|---|
| `VLLM_ATTENTION_BACKEND` | `FLASH_ATTN` | No-op for Gemma 4 (TRITON_ATTN forced) |
| `VLLM_USE_FLASHINFER_MOE_FP8` | `0` on A100, `1` on H100 | Auto-detected via compute cap |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | Avoids JIT failures with old nvcc on A100 |

---

## Model paths

| Variable | Default | Used by |
|---|---|---|
| `GEMMA4_MODEL_PATH` | `/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it` | E001–E005 (full model) |
| `GEMMA4_TEXT_ONLY_MODEL_PATH` | `$GEMMA4_MODEL_PATH-text-only` | E006–E015 (vision tower stripped) |
| `GEMMA4_ASSISTANT_MODEL_PATH` | `/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant` | All MTP experiments (E005–E011, E013, E014) |

Override before running:
```bash
export GEMMA4_MODEL_PATH=/your/local/path/gemma-4-26B-A4B-it
./run_experiments.sh --all
```

---

## Python environment setup

### Creating the environment

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create conda environment for vLLM ablation study
conda create -n vllm-ablation python=3.10 -y
conda activate vllm-ablation

# Install PyTorch 2.11.0 with CUDA 12.6 support
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install vLLM in editable mode (use VLLM_USE_PRECOMPILED=1 to avoid C++ build issues)
# This allows testing Python-only changes (like Triton kernels) without recompiling C++/CUDA code
cd /nvmedata/chenw/vllm-ra
VLLM_USE_PRECOMPILED=1 pip install -e .

# Install additional dependencies
pip install transformers datasets
```

### System requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.10.20 | via conda |
| PyTorch | 2.11.0+cu126 | Requires CUDA 12.0+ |
| CUDA Toolkit | 12.9 (or 11.8+) | System: /usr/local/cuda-12.9 |
| CUDA Driver | 560.35.03 | Supports CUDA 12.6 |
| GPU | A100 80GB PCIe | sm_80 compute capability |
| CMake | 3.28.3 (system) | Avoid cmake-4.x Python package |
| g++ | 9.4.0 | C++17 support required |

### Known build issues and workarounds

**Issue**: vLLM C++/CUDA extensions fail to build with CMake 4.x or mismatched CUDA versions.

**Workaround**: Use `VLLM_USE_PRECOMPILED=1` for editable installs. This:
- Skips recompiling C++/CUDA extensions
- Still allows testing Python code changes (including Triton kernels)
- Requires matching PyTorch CUDA version with system CUDA (both 12.x)

**Alternative**: If full source build is needed:
```bash
# Remove cmake Python package (use system cmake 3.28.3)
pip uninstall -y cmake

# Ensure CUDA 12.x is in PATH (PyTorch 2.11 requires CUDA 12.0+)
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH

# Build from source
pip install -e . --no-build-isolation
```

### Runtime environment

The precompiled vLLM binaries require CUDA 13 runtime libraries. Add to your shell profile or export before running:

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/vllm-ablation/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

This is automatically set by `run_experiments.sh`.

### Verifying the installation

```bash
conda activate vllm-ablation
export LD_LIBRARY_PATH=/root/miniconda3/envs/vllm-ablation/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "from vllm.model_executor.layers.gemma4_fused_ops import gemma_dual_rmsnorm_residual_scalar; print('Gemma4 fused ops OK')"
```

Expected output:
```
vLLM version: 0.21.1rc1.dev269+ge0959bd61.d20260526
PyTorch: 2.11.0+cu126, CUDA: 12.6
Gemma4 fused ops loaded: gemma_dual_rmsnorm_residual_scalar
```

---

## Expected outcomes (A100 80 GB)

H100 NVL reference numbers (from REPRODUCE_PRODSHAPE.md §6):

| Config | H100 out tok/s |
|---|---:|
| BF16 baseline (E001 equivalent) | 1 870 ± 14 |
| FP8 weights + FP8 KV (E003 equivalent, H100 only) | 2 056 ± 21 |

A100 will produce lower absolute numbers (~30–50% lower than H100 NVL). The
**ratio** between experiments (e.g. FP8 vs BF16, MTP on vs off) is the portable signal.

From the prior A100 ablation study (`examples/EXPERIMENT_PLAN_ABLATION_STUDY.md`,
different dataset/output_len but same techniques):

| Technique | A100 contribution |
|---|---:|
| FP8 weights | +5.7% |
| CUDA graphs | −2.6% (regressed on heterogeneous batch) |
| MTP k=5 | **+26.8%** (single biggest win) |
| text-only model | −1.8% (marginal on text-only workload) |

These are rough indicators — the new 8192-token output length and larger batch size
may shift these numbers significantly. The isolation experiments (Group E) will give
the definitive answers for this workload.

---

## Files in this directory

| File | Purpose |
|---|---|
| `bench_experiment.py` | Main ablation driver — EXPERIMENTS dict, run logic |
| `run_experiments.sh` | Shell wrapper — sets env vars, calls bench_experiment.py |
| `analyze_results.py` | Post-run analysis → results/summary.md |
| `prep_dataset.py` | One-time dataset generation from raw delta_prompts/ |
| `bench_offline.py` | Original prod-shape benchmark (REPRODUCE_PRODSHAPE) |
| `REPRODUCE_PRODSHAPE.md` | How to reproduce H100 baseline numbers |
| `EXPERIMENT_PLAN.md` | This file |
| `BENCHMARK_LOG.md` | Run log — record results here after each experiment |
| `datasets/sc1_delta_v2.jsonl` | Ablation dataset (generated by prep_dataset.py) |
| `results/` | Output directory — all_runs.csv + per-run JSON |
