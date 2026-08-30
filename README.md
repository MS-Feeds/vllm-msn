<!-- markdownlint-disable MD001 MD041 -->

# MS-Feeds/vllm-msn

This repository is a fork of [vllm-project/vllm](https://github.com/vllm-project/vllm)
with fork-specific work on top of the upstream codebase. The current focus is
threefold:

1. **Relay Attention** experiments and porting artifacts.
2. **Gemma 4 26B-A4B MoE FP8 benchmarking** campaigns and analysis.
3. **A substantial Rust frontend** under [rust/](rust/) as an experimental
  drop-in alternative to the Python frontend.

The fork is currently synchronized with upstream vLLM v0.28.0 while preserving
fork-specific runtime and CI behavior. Some benchmark artifacts, especially the
Relay Attention and Gemma 4 materials, were originally documented against
specific experimental environments (for example vLLM 0.19.1.dev6 where
explicitly noted); the repository keeps those results as historical context and
repro guidance.

> Current upstream baseline: this fork tracks upstream vLLM v0.28.0.
> General vLLM docs, model support, deployment guidance, and upstream
> contribution workflows remain upstream-owned and are documented by the
> canonical project.

---

## Experimental Rust frontend

The code in [rust/](rust/) is an experimental Rust frontend for vLLM. It aims
to reimplement the northbound serving layer in Rust while continuing to talk to
Python-managed vLLM engine processes through the existing ZMQ/engine boundary.
The project is intentionally not presented as production-stable or
feature-complete; it is a focused experimental path that is still being
extended.

For the full design and current details, see [rust/README.md](rust/README.md).

### Architecture at a glance

The workspace is organized by crate, bottom-up:

- `vllm-engine-core-client` — ZMQ transport + MessagePack protocol to the
 headless vLLM engine
- `vllm-llm` — thin token-in/token-out facade over the engine client
- `vllm-text` — tokenizer and incremental detokenizer
- `vllm-chat` — chat-completion templating and structured assistant events
- `vllm-server` — OpenAI-compatible HTTP API (Axum)
- `vllm-cmd` / `vllm-rs` — CLI entrypoints for serving and render modes

### Python-supervised startup

The Rust frontend can be launched as a Python-supervised worker via the existing
vLLM entrypoints:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B
```

This keeps Python in charge of process startup while the Rust worker handles
front-end serving logic.

### Standalone external-engine mode

The frontend can also run standalone when the Python engine is started elsewhere
and only the Rust frontend is expected to serve the request layer:

```bash
vllm serve Qwen/Qwen3-0.6B \
 --headless \
 --data-parallel-address 127.0.0.1 \
 --data-parallel-rpc-port 62100 \
 --data-parallel-size 1 \
 --data-parallel-size-local 1
```

Then launch the Rust frontend-only server:

```bash
vllm-rs serve Qwen/Qwen3-0.6B \
 --data-parallel-address 127.0.0.1 \
 --data-parallel-rpc-port 62100 \
 --data-parallel-size 1 \
 --data-parallel-size-local 0
```

### Engine-free render mode

The frontend includes a render-only mode that prepares text requests without
starting or connecting to a Python inference engine:

```bash
cargo run --manifest-path rust/Cargo.toml -p vllm-cmd --release -- \
 render Qwen/Qwen3-32B \
 --host 127.0.0.1 --max-model-len 32768
```

This exposes render endpoints that prepare chat/completion payloads without
loading model weights or requiring a Python runtime. It is designed for token
preparation and request validation paths.

### OpenAI-compatible request

After either frontend startup path, regular OpenAI-compatible clients can talk
through the inference endpoints:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{
   "model": "Qwen/Qwen3-0.6B",
   "messages": [{"role": "user", "content": "What is the capital of France?"}],
   "stream": true
 }'
```

See [rust/README.md](rust/README.md) for the complete status, architecture, and
usage details.

---

## Rust benchmark client (`vllm-bench`)

The Rust benchmark client lives in [rust/src/bench/](rust/src/bench/) and is a
standalone, no-Python-runtime benchmark binary for serving and throughput
workloads. It is intended as a fast drop-in benchmark client for vLLM serving
endpoints and can be built as a standalone binary without Python dependencies.
See [rust/src/bench/README.md](rust/src/bench/README.md) for the full CLI and
usage guide.

Example:

```bash
vllm-bench --backend vllm --base-url http://127.0.0.1:8000 \
 --model <model> --dataset-name random \
 --random-input-len 1024 --random-output-len 128 \
 --num-prompts 1000 --max-concurrency 200
```

---

## 1. Relay Attention

**Idea.** For workloads with a long shared system prompt, recompute the
system-prompt attention once into a static KV buffer, then *fuse* it with the
per-request user attention using a log-sum-exp combination instead of
re-attending over the system tokens for every request. This removes the
shared-prefix cost from both prefill and decode.

**Status.**

| Component | State |
| --- | --- |
| Original implementation on vLLM 0.2.6 | ✓ working (eager + CUDA graphs) |
| Triton relay-fusion kernel | ✓ |
| Paged-attention kernel returning log-softmax-exp | ✓ |
| Standalone latency / memory benchmarks (teaser) | ✓ |
| Non-interactive throughput benchmarks (synthetic + ShareGPT) | ✓ |
| Interactive benchmarks (TTFT / TPOT on ShareGPT) | ✓ |
| MQA / GQA via native FlashAttention | ☐ |
| Window-attention + `seq_len > window` adaptation | ☐ |
| ALiBi support | ☐ |
| Port to vLLM 0.9.1 / 0.19.x | historical / in progress |

**Where to look.**

- Design notes and TODO list: [relay_attention.md](relay_attention.md)
- Porting plan: [RELAY_ATTENTION_PORTING_PLAN.md](RELAY_ATTENTION_PORTING_PLAN.md)
- Side-by-side legacy and ported sources: [relay_attention_port/](relay_attention_port/)
- Benchmark drivers and plotting notebooks: [_scripts/](_scripts/) and
 [_cluster/](_cluster/)

The fork keeps the v0.2.6 and v0.9.1 variants of each touched file side-by-side
under `relay_attention_port/` (for example
[attention_v026.py](relay_attention_port/attention_v026.py) vs
[attention_v091.py](relay_attention_port/attention_v091.py)) so the diffs
between engine generations remain reviewable.

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
  CUDA graphs, MTP speculative decoding, text-only model variant, batch and
  `gpu_memory_utilization` sweeps.

Hardware numbers are not portable across H100 / A100; the per-technique
*ratios* are. Gemma 4's heterogeneous attention head dims (256 / 512) force the
`TRITON_ATTN` backend on both H100 and A100; setting
`VLLM_ATTENTION_BACKEND` is effectively a no-op for this model.

### 2.1 A100 80 GB ablation — headline result

Best A100 80 GB result on the sc1 scenario (1000 prompts of the
`sc1_delta_v2.jsonl` dataset, `output_len_cap=8192`,
`max_model_len=24576`, 2 reps):

> **E011 — 1771.5 ± 31.2 output tok/s**
> = FP8 weights + CUDA graphs + MTP k=5 + text-only model at
> `gpu_memory_utilization=0.95`
> **2.184× the BF16 baseline** (E001 = 811.1 tok/s).

Aggregated mean ± σ across 2 reps, summarized in
[benchmarks/gemma4_moe_fp8/ablation_results/summary.md](benchmarks/gemma4_moe_fp8/ablation_results/summary.md)
and reproduced via
[benchmarks/gemma4_moe_fp8/analyze_ablation.py](benchmarks/gemma4_moe_fp8/analyze_ablation.py):

| Exp | Label | out tok/s | ±σ | vs E001 |
| --- | --- | ---: | ---: | ---: |
| E001 | BF16 baseline | 811.1 | 63.6 | 1.000× |
| E002 | + FP8 weights | 1149.0 | 8.4 | 1.417× |
| E004 | + CUDA graphs | 1291.5 | 3.3 | 1.592× |
| E005 | + MTP speculative decoding (k=5) | 1699.9 | 8.0 | 2.096× |
| E006 | + text-only model (vision stripped) | 1748.3 | 8.0 | 2.156× |
| E007 | batch sweep: max_num_seqs = 64 | 1656.3 | 14.0 | 2.042× |
| E008 | batch sweep: max_num_seqs = 192 | 1742.5 | 15.8 | 2.148× |
| E009 | batch sweep: max_num_seqs = 256 | 1747.0 | 0.1 | 2.154× |
| E010 | gpu_mem sweep: 0.80 | 1716.9 | 7.4 | 2.117× |
| E011 | gpu_mem sweep: 0.95 **← best** | 1771.5 | 31.2 | 2.184× |
| E012 | optimal − MTP (isolates MTP) | 1291.6 | 14.8 | 1.592× |
| E013 | optimal − CUDA graphs (isolates CG) | 1606.8 | 24.9 | 1.981× |
| E014 | optimal w/ BF16 weights (isolates FP8 weights) | 1589.8 | 23.1 | 1.960× |
| E015 | BF16 reference (text-only, no opts) | 832.9 | 31.6 | 1.027× |

(E003 — FP8 KV cache `fp8_e4m3` — is absent: it fails on A100 as expected.
See §4 of [benchmarks/gemma4_moe_fp8/README.md](benchmarks/gemma4_moe_fp8/README.md)
for the full matrix, per-rep table, narrative, and old-A100 reference
comparison.)

### 2.2 Per-technique contribution (mean across reps, sc1)

| Pair | Δ out tok/s | Δ % |
| --- | ---: | ---: |
| FP8 weights vs BF16 (E002 − E001) | +338.0 | +41.7 % |
| CUDA graphs vs eager (E004 − E002) | +142.4 | +12.4 % |
| MTP k=5 (E005 − E004) | +408.4 | +31.6 % |
| text-only model (E006 − E005) | +48.4 | +2.8 % |
| gpu_mem = 0.95 vs 0.90 (E011 − E006) | +23.1 | +1.3 % |
| disable MTP at optimal (E012 − E006) | −456.7 | −26.1 % |
| disable CUDA graphs at optimal (E013 − E006) | −141.5 | −8.1 % |
| BF16 weights at optimal (E014 − E006) | −158.5 | −9.1 % |

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

The aggregated campaign summary is recorded in
[benchmarks/gemma4_moe_fp8/ablation_results/summary.md](benchmarks/gemma4_moe_fp8/ablation_results/summary.md).
See [benchmarks/gemma4_moe_fp8/README.md](benchmarks/gemma4_moe_fp8/README.md)
for the prod-shape (H100 NVL) and `max_num_seqs` sweep campaigns, the dataset
preparation pipeline, and the appendix on env-setup gotchas.

---

## 3. Repository layout (fork-specific)

| Path | Purpose |
| --- | --- |
| [rust/](rust/) | Experimental Rust frontend workspaces and CLI/server crates |
| [rust/README.md](rust/README.md) | Overview of the Rust frontend architecture and examples |
| [rust/src/bench/](rust/src/bench/) | Standalone Rust benchmark client (`vllm-bench`) |
| [rust/src/bench/README.md](rust/src/bench/README.md) | `vllm-bench` usage, workloads, and CLI docs |
| [relay_attention.md](relay_attention.md) | Relay Attention design notes and TODO |
| [RELAY_ATTENTION_PORTING_PLAN.md](RELAY_ATTENTION_PORTING_PLAN.md) | v0.2.6 → v0.9.1 / v0.19.x port plan |
| [relay_attention_port/](relay_attention_port/) | Side-by-side legacy / ported sources and verification scripts |
| [benchmarks/gemma4_moe_fp8/](benchmarks/gemma4_moe_fp8/) | Gemma 4 MoE FP8 benchmark campaigns and results |
| [_scripts/](_scripts/) | Relay-attention benchmark drivers and plotting notebooks |
| [_cluster/](_cluster/) | SLURM wrappers for Relay Attention experiments |
| [examples/gemma4/](examples/gemma4/) | Earlier AsyncEngine Gemma 4 ablation references |

Everything else mirrors upstream vLLM. Upstream layout and contribution
practices still apply for changes outside the paths above; see [AGENTS.md](AGENTS.md)
and the upstream [Contributing](https://docs.vllm.ai/en/latest/contributing/index.html)
guide.

---

## 4. Citation

If you build on the **Relay Attention** work in this fork, please cite the
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

Upstream documentation, model support, deployment guidance, and community links
live at the canonical project sites:

- Docs: <https://docs.vllm.ai>
- Repo: <https://github.com/vllm-project/vllm>
- Blog: <https://blog.vllm.ai>
- Forum: <https://discuss.vllm.ai>
- Slack: <https://slack.vllm.ai>

This fork inherits upstream's Apache-2.0 license; see [LICENSE](LICENSE).
