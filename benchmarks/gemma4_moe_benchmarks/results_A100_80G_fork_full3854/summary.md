# Gemma 4 MoE FP8 — Ablation Benchmark Summary (Fork Full3854)

## A100 80GB Results (LLM engine)

_Current ablation study using `vllm.LLM` (benchmarks/gemma4_moe_benchmarks/bench_experiment.py)_

### Scenario: sc1

| Exp | Label | out tok/s (A100 80GB) | ±σ | vs E001 | Backend | eager | MTP | seqs | mem% |
|-----|-------|:---:|:---:|:---:|---------|-------|-----|------|------|
| E001 | BF16 baseline — matches REPRODUCE_PRODSHAPE sc1 | 750.6 | 3.6 | 1.000× | FLASH_ATTN | ✓ | ✗ | 128 | 0.9 |
| E002 | +FP8 weights (kv cache stays BF16 / auto) | 1242.5 | 11.4 | 1.655× | FLASH_ATTN | ✓ | ✗ | 128 | 0.9 |
| E004 | +CUDA graphs (enforce_eager=False) | 1512.2 | 6.7 | 2.015× | FLASH_ATTN | ✗ (CG) | ✗ | 128 | 0.9 |
| E005 | +MTP speculative decoding (k=5) | 2025.5 | 7.8 | 2.699× | FLASH_ATTN | ✗ (CG) | ✓ k=5 | 128 | 0.9 |
| E006 | +text-only model (vision stripped) | 2023.2 | 7.1 | 2.696× | FLASH_ATTN | ✗ (CG) | ✓ k=5 | 128 | 0.9 |
| E007 | batch sweep: mns=64 | 1899.8 | 0.4 | 2.531× | FLASH_ATTN | ✗ (CG) | ✓ k=5 | 64 | 0.9 |
| E008 | batch sweep: mns=192 | 2011.8 | 3.9 | 2.680× | FLASH_ATTN | ✗ (CG) | ✓ k=5 | 192 | 0.9 |
| E009 | batch sweep: mns=256 | 2008.2 | 5.9 | 2.676× | FLASH_ATTN | ✗ (CG) | ✓ k=5 | 256 | 0.9 |
| E010 | gpu_mem sweep: 0.80 | 1981.7 | 5.9 | 2.640× | FLASH_ATTN | ✗ (CG) | ✓ k=5 | 128 | 0.8 |
| E011 | gpu_mem sweep: 0.95 | 2071.0 | 3.6 | 2.759× | FLASH_ATTN | ✗ (CG) | ✓ k=5 | 128 | 0.95 |
| E012 | no MTP at optimal (isolates MTP contribution) | 1508.7 | 3.9 | 2.010× | FLASH_ATTN | ✗ (CG) | ✗ | 128 | 0.9 |
| E013 | no CUDA graphs at optimal (isolates CG contribution) | 1961.6 | 21.7 | 2.613× | FLASH_ATTN | ✓ | ✓ k=5 | 128 | 0.9 |
| E014 | BF16 weights at optimal config (isolates FP8 weight ... | 1842.0 | 6.4 | 2.454× | FLASH_ATTN | ✗ (CG) | ✓ k=5 | 128 | 0.9 |
| E015 | BF16 reference (text-only, no opts) | 741.6 | 1.3 | 0.988× | FLASH_ATTN | ✓ | ✗ | 128 | 0.9 |
| E016 | BF16 + CUDA graphs only (no MTP, no FP8) | 1047.9 | 3.9 | 1.396× | FLASH_ATTN | ✗ (CG) | ✗ | 128 | 0.9 |
| E017 | low-mns sweep: mns=16 | 1262.3 | 1.8 | 1.682× | FLASH_ATTN | ✗ (CG) | ✓ k=5 | 16 | 0.9 |
| E018 | low-mns sweep: mns=32 | 1590.5 | 1.2 | 2.119× | FLASH_ATTN | ✗ (CG) | ✓ k=5 | 32 | 0.9 |

**Best A100 80GB result**: E011 — 2071.0 output tok/s
  Overall A100 80GB speedup vs BF16 baseline: 2.759×

---
## Key configuration contribution (A100 80GB, mean across reps)

Estimated contribution of each optimization layer, from ablation pairs:

**sc1:**
  FP8 weights vs BF16 (E002-E001)          : +492.0 tok/s (+65.5%)
  CUDA graphs vs eager (E004-E002)         : +269.7 tok/s (+21.7%)
  MTP k=5 (E005-E004)                      : +513.3 tok/s (+33.9%)
  text-only model (E006-E005)              : -2.3 tok/s (-0.1%)
  batch mns=64 vs 128 (E007-E006)          : -123.5 tok/s (-6.1%)
  batch mns=192 vs 128 (E008-E006)         : -11.4 tok/s (-0.6%)
  batch mns=256 vs 128 (E009-E006)         : -15.1 tok/s (-0.7%)
  gpu_mem=0.80 vs 0.90 (E010-E006)         : -41.5 tok/s (-2.1%)
  gpu_mem=0.95 vs 0.90 (E011-E006)         : +47.8 tok/s (+2.4%)
  disable MTP at optimal (E012-E006) — negative expected : -514.5 tok/s (-25.4%)
  disable CUDA graphs at optimal (E013-E006) — negative expected : -61.7 tok/s (-3.0%)
  BF16 weights at optimal (E014-E006) — isolates FP8 : -181.3 tok/s (-9.0%)
