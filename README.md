<!-- markdownlint-disable MD001 MD041 -->

# vllm-ra — vLLM fork: Relay Attention + Gemma 4 MoE FP8

This repository is a fork of [vllm-project/vllm](https://github.com/vllm-project/vllm)
maintained by **rayleizhu**. It hosts two lines of work on top of upstream
vLLM:

1. **Relay Attention** — a system-prompt KV-cache-reuse mechanism originally
   prototyped on vLLM 0.2.6, now being ported to vLLM 0.9.1 / 0.19.x.
2. **Gemma 4 26B-A4B MoE FP8 benchmarking** — prod-shape throughput on
   H100 NVL and a 15-experiment ablation campaign on A100 80 GB PCIe.

Upstream vLLM features (PagedAttention, continuous batching, FP8/INT8
quantization, speculative decoding, OpenAI-compatible server, etc.) are
preserved as-is. Everything below describes only the fork-specific work.

> Branch: `feat/gemma4-moe-opt-a100` &nbsp;·&nbsp; vLLM base: 0.19.1.dev6
> (V1 engine, `VLLM_COMPILE` mode 3).

---

## 1. Relay Attention

**Idea.** For workloads with a long shared system prompt, recompute the
system-prompt attention once into a static KV buffer, then *fuse* it with
the per-request user attention using a log-sum-exp combination instead of
re-attending over the system tokens for every request. This removes the
shared-prefix cost from both prefill and decode.

**Status.**

| Component                                              | State |
|---|---|
| Original implementation on vLLM 0.2.6                   | ✓ working (eager + CUDA graphs) |
| Triton relay-fusion kernel                              | ✓ |
| Paged-attention kernel returning log-softmax-exp        | ✓ |
| Standalone latency / memory benchmarks (teaser)         | ✓ |
| Non-interactive throughput benchmarks (synthetic + ShareGPT) | ✓ |
| Interactive benchmarks (TTFT / TPOT on ShareGPT)        | ✓ |
| MQA / GQA via native FlashAttention                     | ☐ |
| Window-attention + `seq_len > window` adaptation        | ☐ |
| ALiBi support                                           | ☐ |
| Port to vLLM 0.9.1 / 0.19.x                              | in progress |

**Where to look.**

- Design notes and TODO list: [relay_attention.md](relay_attention.md)
- v0.2.6 → v0.9.1 porting plan: [RELAY_ATTENTION_PORTING_PLAN.md](RELAY_ATTENTION_PORTING_PLAN.md)
- Ported sources, side-by-side v0.2.6 / v0.9.1 files, integration example,
  and verification script: [relay_attention_port/](relay_attention_port/)
- Benchmark drivers and plotting notebooks: [_scripts/](_scripts/) and
  [_cluster/](_cluster/) (SLURM wrappers).

The fork keeps the v0.2.6 and v0.9.1 variants of each touched file
side-by-side under `relay_attention_port/` (e.g.
[attention_v026.py](relay_attention_port/attention_v026.py) vs
[attention_v091.py](relay_attention_port/attention_v091.py)) so the diffs
between the two engine generations stay reviewable.

---

## 2. Gemma 4 26B-A4B MoE FP8 benchmarking

All scripts, datasets, configs, and result CSVs live under
[benchmarks/gemma4_moe_fp8/](benchmarks/gemma4_moe_fp8/). Three independent
campaigns are documented there:

1. **Prod-shape benchmark** (H100 NVL, 96 GB) — 10 000-prompt offline runs,
   bf16 vs FP8.
2. **Sweep v1 / v2** (H100 NVL) — `max_num_seqs` sweep over the two
   prod-shape scenarios.
3. **A100 80 GB ablation** — 15-experiment stack-up isolating FP8 weights,
   CUDA graphs, MTP speculative decoding, text-only model variant, batch
   and `gpu_memory_utilization` sweeps.

Hardware numbers are not portable across H100 / A100; the per-technique
*ratios* are. Gemma 4's heterogeneous attention head dims (256 / 512)
force the `TRITON_ATTN` backend on both H100 and A100 — setting
`VLLM_ATTENTION_BACKEND` is a no-op for this model.

### 2.1 A100 80 GB ablation — headline result

Best A100 80 GB result on the sc1 scenario (1000 prompts of the
`sc1_delta_v2.jsonl` dataset, `output_len_cap=8192`,
`max_model_len=24576`, 2 reps):

> **E011 — 1771.5 ± 31.2 output tok/s**
> = FP8 weights + CUDA graphs + MTP k=5 + text-only model at
> `gpu_memory_utilization=0.95`
> **2.184× the BF16 baseline** (E001 = 811.1 tok/s).

Aggregated mean ± σ across 2 reps, derived from
[benchmarks/gemma4_moe_fp8/ablation_results/all_runs.csv](benchmarks/gemma4_moe_fp8/ablation_results/all_runs.csv)
via
[benchmarks/gemma4_moe_fp8/analyze_ablation.py](benchmarks/gemma4_moe_fp8/analyze_ablation.py):

| Exp  | Label                                              | out tok/s | ±σ  | vs E001 |
|------|----------------------------------------------------|----:|----:|--------:|
| E001 | BF16 baseline                                      |  811.1 | 63.6 | 1.000× |
| E002 | + FP8 weights                                      | 1149.0 |  8.4 | 1.417× |
| E004 | + CUDA graphs                                      | 1291.5 |  3.3 | 1.592× |
| E005 | + MTP speculative decoding (k=5)                   | 1699.9 |  8.0 | 2.096× |
| E006 | + text-only model (vision stripped)                | 1748.3 |  8.0 | 2.156× |
| E007 | batch sweep: max_num_seqs = 64                     | 1656.3 | 14.0 | 2.042× |
| E008 | batch sweep: max_num_seqs = 192                    | 1742.5 | 15.8 | 2.148× |
| E009 | batch sweep: max_num_seqs = 256                    | 1747.0 |  0.1 | 2.154× |
| E010 | gpu_mem sweep: 0.80                                | 1716.9 |  7.4 | 2.117× |
| E011 | gpu_mem sweep: 0.95 **← best**                     | 1771.5 | 31.2 | 2.184× |
| E012 | optimal − MTP (isolates MTP)                       | 1291.6 | 14.8 | 1.592× |
| E013 | optimal − CUDA graphs (isolates CG)                | 1606.8 | 24.9 | 1.981× |
| E014 | optimal w/ BF16 weights (isolates FP8 weights)     | 1589.8 | 23.1 | 1.960× |
| E015 | BF16 reference (text-only, no opts)                |  832.9 | 31.6 | 1.027× |

(E003 — FP8 KV cache `fp8_e4m3` — is absent: it fails on A100 as expected.
See §4 of [benchmarks/gemma4_moe_fp8/README.md](benchmarks/gemma4_moe_fp8/README.md)
for the full matrix, per-rep table, narrative, and old-A100 reference
comparison.)

### 2.2 Per-technique contribution (mean across reps, sc1)

| Pair                                            | Δ out tok/s | Δ %     |
|---|---:|---:|
| FP8 weights vs BF16 (E002 − E001)               | +338.0 | +41.7 % |
| CUDA graphs vs eager (E004 − E002)              | +142.4 | +12.4 % |
| MTP k=5 (E005 − E004)                           | +408.4 | +31.6 % |
| text-only model (E006 − E005)                   |  +48.4 |  +2.8 % |
| gpu_mem = 0.95 vs 0.90 (E011 − E006)            |  +23.1 |  +1.3 % |
| disable MTP at optimal (E012 − E006)            | −456.7 | −26.1 % |
| disable CUDA graphs at optimal (E013 − E006)    | −141.5 |  −8.1 % |
| BF16 weights at optimal (E014 − E006)           | −158.5 |  −9.1 % |

**Reading.** MTP is the single biggest win on A100 (+31.6 % stack-up gain;
disabling it at the optimum costs 26.1 %). FP8 weights are the runner-up
(+41.7 % over the BF16 baseline; isolated cost of removing them is 9.1 %).
CUDA graphs add ~12 % at the stack-up step and account for ~8 % at the
optimum. Text-only stripping and `gpu_memory_utilization` are second-order
(≤ 3 %). The batch-size sweep is flat from 128–256 and slightly hurts at 64.

### 2.3 Reproducing the A100 campaign

```bash
# Env (precompiled-kernel install; no source build needed)
source /opt/conda/etc/profile.d/conda.sh
conda create -n vllm-ablation python=3.11 pip -y
conda activate vllm-ablation
cd vllm-msn
export VLLM_USE_PRECOMPILED=1
pip install -e .

# All 15 experiments, sc1 scenario, 2 reps each
cd benchmarks/gemma4_moe_fp8
chmod +x run_ablation.sh
./run_ablation.sh --all --scenario sc1 --reps 2

# Aggregate -> ablation_results/summary.md
python3 analyze_ablation.py
```

Raw per-rep results land in
[`ablation_results/all_runs.csv`](benchmarks/gemma4_moe_fp8/ablation_results/all_runs.csv)
(append-only) plus per-run JSON
(`ablation_results/<exp>_<scenario>_rep<N>.json`). Full run log:
[`run_sc1_delta_v2_20260526_213939.log`](benchmarks/gemma4_moe_fp8/ablation_results/run_sc1_delta_v2_20260526_213939.log).

See [benchmarks/gemma4_moe_fp8/README.md](benchmarks/gemma4_moe_fp8/README.md)
for the prod-shape (H100 NVL) and `max_num_seqs` sweep campaigns, the
dataset preparation pipeline, and the appendix on env-setup gotchas.

---

## 3. Repository layout (fork-specific)

| Path | Purpose |
|---|---|
| [relay_attention.md](relay_attention.md) | Relay Attention design notes & TODO |
| [RELAY_ATTENTION_PORTING_PLAN.md](RELAY_ATTENTION_PORTING_PLAN.md) | 0.2.6 → 0.9.1 port plan |
| [relay_attention_port/](relay_attention_port/) | Side-by-side v0.2.6 / v0.9.1 sources, integration example, verifier |
| [benchmarks/gemma4_moe_fp8/](benchmarks/gemma4_moe_fp8/) | Gemma 4 MoE FP8 campaigns (prod-shape, sweeps, A100 ablation) |
| [_scripts/](_scripts/) | Relay-attention benchmark drivers & plotting notebooks |
| [_cluster/](_cluster/) | SLURM wrappers for the relay-attention benchmarks |
| [examples/gemma4/](examples/gemma4/) | Earlier AsyncEngine Gemma 4 ablation (kept as historical reference) |

Everything else mirrors upstream vLLM. Upstream layout and contribution
guidelines still apply for changes outside the paths above — see
[AGENTS.md](AGENTS.md) and the upstream
[Contributing](https://docs.vllm.ai/en/latest/contributing/index.html)
guide.

---

## 4. Citation

If you build on the **relay attention** work in this fork, please cite the
original vLLM paper as well as this repository:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

---

## 5. Upstream vLLM

Upstream documentation, model support, deployment guides, and community
links live at the canonical project sites:

- Docs: <https://docs.vllm.ai>
- Repo: <https://github.com/vllm-project/vllm>
- Blog: <https://blog.vllm.ai>
- Forum: <https://discuss.vllm.ai>
- Slack: <https://slack.vllm.ai>

This fork inherits upstream's Apache-2.0 license; see [LICENSE](LICENSE).
