# Relay Attention: Design, Related Work, and Integration Plan

## Overview

Relay Attention is an optimization for LLM serving workloads that involve a **long shared system prompt** (e.g., tool definitions, few-shot examples, RAG documents, instruction schemas). Instead of re-reading the system prompt's KV pairs from GPU DRAM for every request in a batch, RelayAttention reads them **exactly once per batch** and mathematically fuses the system and user attention outputs.

The core principle: standard causal attention can be reformulated so that:

$$\text{Attn}(Q, K, V) = \text{Fuse}\!\left(\text{Attn}(Q, K_{\text{sys}}, V_{\text{sys}}),\ \text{Attn}(Q, K_{\text{usr}}, V_{\text{usr}})\right)$$

where `Fuse` is a log-sum-exp (LSE) weighted combination:

$$\alpha = \frac{1}{1 + \exp(\text{LSE}_{\text{usr}} - \text{LSE}_{\text{sys}})}, \quad \text{out} = \alpha \cdot \text{out}_{\text{sys}} + (1 - \alpha) \cdot \text{out}_{\text{usr}}$$

This reformulation is **exact** (lossless) and requires **no model retraining**.

---

## Related Work Survey

### Foundational (the basis of this implementation)

| Paper | Venue | arXiv | Key Idea |
|---|---|---|---|
| **RelayAttention for Efficient LLM Serving with Long System Prompts** — Zhu et al. | ACL 2024 | [2402.14808](https://arxiv.org/abs/2402.14808) | Reads system-prompt KV from DRAM once per batch; LSE-based fusion of system + user attention. Lossless, no retraining. Integrated with vLLM 0.2.6; this repo ports it to vLLM 0.9.1+. |

### Contemporary (same Feb 2024 wave — shared-prefix family)

| Paper | Venue | arXiv | Key Idea | Relation |
|---|---|---|---|---|
| **Hydragen: High-Throughput LLM Inference with Shared Prefixes** — Juravsky et al. | — | [2402.05099](https://arxiv.org/abs/2402.05099) | Decomposes attention into shared-prefix + unique-suffix parts; batches prefix queries together enabling matrix-matrix (not matrix-vector) ops. Up to 32× throughput on CodeLlama-13B. | Very similar goal to RelayAttention. Hydragen focuses on batching prefix queries; RelayAttention focuses on reading KV once. Complementary decomposition strategies — could be combined. |
| **ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition** — Ye et al. | ACL 2024 | [2402.15220](https://arxiv.org/abs/2402.15220) | Breaks KV into chunks stored in a prefix tree; two-phase partition improves data locality. 3.2–4.8× speedup on attention kernel (system prompt 1K–4K tokens). | Orthogonal: improves memory layout rather than reformulating attention math. Could be layered on top of RelayAttention's prefix KV cache. |
| **DeFT: Decoding with Flash Tree-Attention** — Yao et al. | ICLR 2025 | [2404.00242](https://arxiv.org/abs/2404.00242) | Flash Tree-Attention for tree-structured inference (few-shot, speculative decode, multi-step reasoning). KV-Guided Grouping reduces redundant KV cache IO by 73–99%. Up to 2.23/3.59× speedup in end-to-end/attention latency. | Generalizes shared-prefix to tree structures. The natural next evolution once linear prefix sharing works — enables multi-turn / speculative decoding scenarios. |

### Newer Work (2025–2026) — directly relevant to this implementation

| Paper | Venue | arXiv | Key Idea | Why Relevant |
|---|---|---|---|---|
| **APE: Adaptive Parallel Encoding** — Yang et al. | ICLR 2025 | [2502.05431](https://arxiv.org/abs/2502.05431) | Independently pre-encodes multiple contexts (RAG docs, ICL examples) in parallel; corrects attention distribution mismatch via shared prefix, temperature, and scaling factor. 98% / 93% of sequential-encoding quality; 4.5× end-to-end speedup, scales to 128K-length context. | **Direct upgrade path for RelayAttention.** APE generalizes the fusion beyond a single system prompt to *multiple independent contexts*. Its distribution correction (temperature + scaling) addresses the quality gap that naive parallel encoding causes. Applicable to Gemma 4's multi-document tool schemas. |
| **MiniPIC: Flexible Position-Independent Caching in <100LOC** — Ordonez & Parnell | 2026 | [2606.13126](https://arxiv.org/abs/2606.13126) | Position-Independent Caching in vLLM: stores *unrotated* K vectors in cache, applies RoPE inside the attention kernel per-request. Implements Block-Attention, EPIC, Prompt Cache within one vLLM instance. <100 LOC core engine changes. 49% prefill throughput improvement; up to 2 orders of magnitude TTFT reduction for cached spans. | **Critical for correct RoPE handling.** An open TODO in this repo's `relay_attention.md` is handling position encodings for prefix KV. MiniPIC solves exactly this. Their `unrotated-K + per-request RoPE` approach should be adopted in the relay backend. |
| **PackInfer: Compute- and I/O-Efficient Attention for Batched LLM Inference** — Ning et al. | 2026 | [2602.06072](https://arxiv.org/abs/2602.06072) | Groups shared-prefix requests and reorganizes KV into contiguous layouts; load-balanced kernel packs multiple requests into unified kernel launches. 13–20% latency reduction, 20% throughput gain vs FlashAttention. | **Complementary kernel optimization.** PackInfer's contiguous KV grouping can be combined with RelayAttention's DRAM-once-per-batch read for compounding gains on large shared-prefix batches. |

---

## Existing Implementation in This Repo

The port from vLLM 0.2.6 → 0.9.1 is largely complete in `relay_attention_port/`:

| File | Purpose |
|---|---|
| `relayattn_ops_v091.py` | Core relay fusion: native PyTorch + Triton kernel (LSE-weighted combination) |
| `relay_attention_backend.py` | Sketch of vLLM 0.9.1 `AttentionBackend` / `AttentionImpl` integration |
| `relay_config.py` | `RelayConfig`, `RelayInfo`, `RelayMetadata` dataclasses |
| `integration_example.py` | Usage examples |

**Key discovery**: `vllm/v1/attention/ops/merge_attn_states.py` already implements LSE-based partial attention merging (used for chunked prefill). RelayAttention's fusion can reuse this directly, avoiding a separate Triton kernel for the merge step.

---

## Integration Plan (vLLM v1 Engine)

### Phase 1: Backend registration
- Add `RELAY_ATTN` entry to `vllm/v1/attention/backends/registry.py`
- Create `vllm/v1/attention/backends/relay_attn.py` extending `FlashAttentionBackend`

### Phase 2: Config
- Add `enable_relay_attention: bool` and `relay_system_prompt: Optional[str]` to `vllm/config/`
- Thread config through `AttentionSelectorConfig` in `vllm/v1/attention/selector.py`

### Phase 3: RoPE-correct prefix KV cache
- Adapt MiniPIC's unrotated-K approach: store K without RoPE in prefix cache, apply RoPE inside the relay backend's forward pass

### Phase 4: Gemma 4 MoE integration & benchmarking
- Use `benchmarks/gemma4_moe_benchmarks/` scripts to validate throughput/latency improvements

### Phase 5 (future): APE-style multi-context fusion
- Extend single system-prompt relay to multiple parallel contexts (RAG, tool docs)
- Apply APE's temperature + scaling correction to maintain generation quality

---

## Speculative Decoding Compatibility

Relay Attention is compatible with all speculative decoding strategies in vLLM:

| Strategy | Compatible | Notes |
|---|---|---|
| Standard chain (ngram, MLP, draft model) | ✅ | `merge_attn_states` is per-token; `1+K` query tokens per sequence work transparently |
| EAGLE / padded batch | ✅ | Inherited from `FlashAttentionBackend.supports_batch_invariance()` |
| Tree-based (EAGLE3) | ✅ + bonus | All tree branches share system-prompt KV — relay delivers an amplified DRAM saving vs. non-speculative batches. Aligns with DeFT (arXiv:2404.00242). |
| CUDAGraph + spec decode | ✅ | Inherited from `FlashAttentionBackend` |

**Important**: the **draft model** should use a standard backend such as `FLASH_ATTN`. Configure via:

```python
SpeculativeConfig(attention_backend="FLASH_ATTN", ...)
```

The draft model has no long system prompt to share, so relay adds overhead with no benefit there.

---

## Open TODOs (from original implementation)

From `relay_attention.md`:
- [ ] Adaptations for window attention when sequence length > window size
- [ ] Adaptations to support ALiBi
- [ ] RoPE handling for prefix KV cache (→ use MiniPIC approach)
- [ ] GQA/MQA support via native flash attention kernel
