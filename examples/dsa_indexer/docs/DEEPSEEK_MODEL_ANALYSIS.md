# DeepSeek V2/V3 Model Analysis — Instrumentation Points

**Date**: 2026-06-01  
**Phase**: 1 — Model Support Verification  
**Status**: ✅ Complete

---

## Executive Summary

**DeepSeek V2/V3 support in vLLM**: ✅ **CONFIRMED**

- **Model file**: `vllm/model_executor/models/deepseek_v2.py`
- **Config support**: `DeepseekV2Config`, `DeepseekV3Config` from `transformers`
- **MoE architecture**: Routed experts with gating network
- **Top-K location**: FusedMoE layer via `fused_topk()` function
- **Instrumentation target**: Post-activation scores (softmax/sigmoid output)

---

## Architecture Overview

### Model Hierarchy

```
DeepseekV2ForCausalLM
├── DeepseekV2Model
│   └── DeepseekV2DecoderLayer  (per layer)
│       ├── DeepseekV2Attention (MLA - Multi-Latent Attention)
│       └── DeepseekV2MoE ⭐ TARGET
│           ├── GateLinear (hidden → expert logits)
│           ├── FusedMoE (expert execution)
│           │   ├── FusedMoERouter (Top-K selection)
│           │   └── Expert weights (w1, w2, w3)
│           └── SharedExperts (optional)
└── ParallelLMHead (language modeling)
```

### Key Configuration Parameters

From `DeepseekV2Config`:

```python
{
    "n_routed_experts": 160,        # Total number of experts (V2)
    "n_shared_experts": 2,          # Shared experts (always active)
    "num_experts_per_tok": 6,       # Top-K value (usually 6-8)
    "scoring_func": "softmax",      # Activation function for gating
    "norm_topk_prob": True,         # Renormalize top-k weights
    "routed_scaling_factor": 1.0,   # Output scaling
}
```

**Note**: DeepSeek V3 may use different expert counts. V4 specs TBD.

---

## MoE Forward Pass Flow

### File: `vllm/model_executor/models/deepseek_v2.py`

**Class**: `DeepseekV2MoE` (line 245-389)

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # [batch*seq_len, hidden_dim]
    
    # Step 1: Compute router logits (gate projection)
    if self.experts.is_internal_router:
        # Some backends do routing internally
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=hidden_states  # Placeholder
        )
    else:
        # Standard routing: GateLinear projects to expert scores
        router_logits, _ = self.gate(hidden_states)  # [M, num_experts]
        
        # ⭐ INSTRUMENTATION POINT 1: Raw router logits
        # Shape: [num_tokens, n_routed_experts]
        # Values: Raw gate outputs (unbounded, can be negative)
        
        # Step 2: FusedMoE does Top-K + expert execution
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits
        )
    
    return final_hidden_states
```

### File: `vllm/model_executor/layers/fused_moe/layer.py`

**Class**: `FusedMoE.forward()` (line 1305-1315)

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    # Delegates to self.runner.forward()
    # Runner orchestrates: routing → permutation → expert execution → aggregation
    return self.runner.forward(hidden_states, router_logits, input_ids)
```

The runner internally calls the **FusedMoERouter** for Top-K selection.

### File: `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`

**Function**: `fused_topk()` (line 69-114)

```python
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,      # = router_logits from gate
    topk: int,                         # = num_experts_per_tok (e.g., 6)
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",    # or "sigmoid"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    M, _ = hidden_states.size()       # M = num_tokens
    
    # Allocate output tensors
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=...)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=...)
    token_expert_indices = torch.empty(M, topk, dtype=torch.int32, device=...)
    
    if scoring_func == "softmax":
        # ⭐ INSTRUMENTATION POINT 2: Post-softmax scores (Joseph's target)
        # This calls ops.topk_softmax() which:
        #   1. Applies softmax(gating_output) → [M, num_experts]
        #   2. Selects top-k → (topk_weights, topk_ids)
        #   3. Optionally renormalizes top-k weights
        
        topk_func = dispatch_topk_softmax_func()
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices,
            gating_output,  # Raw logits input
            renormalize
        )
        # After this: topk_weights contains post-softmax scores for top-k experts
        
        return topk_weights, topk_ids, token_expert_indices
    
    elif scoring_func == "sigmoid":
        # Alternative: sigmoid activation instead of softmax
        topk_func = dispatch_topk_sigmoid_func()
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices,
            gating_output,
            renormalize
        )
        return topk_weights, topk_ids, token_expert_indices
```

**Key insight**: The activation (softmax/sigmoid) is **fused inside** the custom op `ops.topk_softmax()` or `ops.topk_sigmoid()`. We cannot directly capture post-activation scores without either:
1. Modifying the custom op (complex, requires CUDA kernel changes)
2. Capturing raw logits and recomputing activation externally (feasible)

---

## Instrumentation Strategy

### Option A: Capture Raw Logits + External Activation (Recommended)

**Where**: In `DeepseekV2MoE.forward()` after `self.gate(hidden_states)`

**Advantages**:
- No custom op modifications needed
- Can compute post-softmax/sigmoid scores for logging
- Minimal performance impact (activation already computed in forward pass)

**Implementation**:

```python
# File: vllm/model_executor/models/deepseek_v2.py
# In DeepseekV2MoE.forward()

def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    
    if self.is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)
    
    if self.experts.is_internal_router:
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )
    else:
        router_logits, _ = self.gate(hidden_states)  # [M, n_routed_experts]
        
        # ===== INSTRUMENTATION START =====
        if _INDEXER_ENABLED:
            # Apply same activation as fused_topk will use
            if self.experts.runner.router.scoring_func == "softmax":
                # Post-softmax scores (Joseph's request)
                router_scores = torch.softmax(router_logits.float(), dim=-1)
            elif self.experts.runner.router.scoring_func == "sigmoid":
                # Post-sigmoid scores
                router_scores = torch.sigmoid(router_logits.float())
            else:
                router_scores = router_logits  # Fallback: raw logits
            
            # Log to indexer
            _log_indexer(
                k_cache_prefix=f"{self.prefix}.gate",  # e.g., "model.layers.0.mlp.gate"
                phase="decode" if router_scores.shape[0] <= 128 else "prefill",  # Heuristic
                logits=router_scores,  # [M, n_routed_experts]
                index_topk=self.experts.top_k,  # num_experts_per_tok (6-8)
                key_valid_starts=torch.zeros(router_scores.shape[0], dtype=torch.long, device=router_scores.device),
                key_valid_ends=torch.full((router_scores.shape[0],), router_scores.shape[1], dtype=torch.long, device=router_scores.device),
            )
        # ===== INSTRUMENTATION END =====
        
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
    
    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(
            final_hidden_states, 0
        )
        final_hidden_states = final_hidden_states[:num_tokens]
    
    return final_hidden_states.view(num_tokens, hidden_dim)
```

### Option B: Capture Inside Custom Op (Alternative)

**Where**: Modify `csrc/moe/topk_softmax.cu` (or equivalent)

**Advantages**:
- Captures exact post-activation scores (no recomputation)
- Can log before Top-K selection (more accurate)

**Disadvantages**:
- Requires CUDA kernel modification
- More complex implementation
- Harder to maintain across vLLM versions

**Decision**: Use **Option A** for Phase 2 implementation.

---

## Semantic Differences vs GLM-5.1

| Aspect | GLM-5.1 (Attention) | DeepSeek (MoE) |
|--------|---------------------|----------------|
| **Operation** | Sparse attention Top-K | Expert routing Top-K |
| **Input semantics** | Attention logits (Q·K^T) | Expert gate logits |
| **Value range** | Typically [-50, +50] | Unbounded (pre-activation) |
| **Post-activation** | Softmax → probabilities [0,1] | Softmax → probabilities [0,1] |
| **Sparsity** | Variable-length sequences | Fixed expert count |
| **Masking** | Needs key_valid_starts/ends | All experts valid |
| **K value** | 256-2048 (tokens) | 6-8 (experts) |
| **Frequency** | 2× per layer (prefill/decode) | 1× per layer |

**Key adjustment for logger**:
- GLM-5.1: Records attention logits (can be negative, needs masking)
- DeepSeek: Records post-softmax scores (always ≥ 0, no masking needed)

**Histogram range**:
- GLM-5.1: `INDEXER_LOGIT_RANGE="-50.0,50.0"`
- DeepSeek: `INDEXER_LOGIT_RANGE="0.0,1.0"` (post-softmax probabilities)

---

## Layer Naming Convention

DeepSeek V2/V3 uses:

```
model.layers.{L}.mlp.gate                  # GateLinear
model.layers.{L}.mlp.experts               # FusedMoE
model.layers.{L}.mlp.shared_experts        # SharedExperts (optional)
```

Where `L` ∈ [0, num_hidden_layers-1].

For instrumentation, use:
- `k_cache_prefix = f"model.layers.{layer_idx}.mlp.gate"`
- This matches the pattern in `_indexer_logger._parse_layer()`:
  ```python
  _LAYER_RE = re.compile(r"layers\.(\d+)\.")
  ```

---

## Configuration Check

### Verify DeepSeek V3 Model Access

```bash
# Check if model is available on HuggingFace
huggingface-cli download deepseek-ai/DeepSeek-V3 --repo-type model

# Inspect config
python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('deepseek-ai/DeepSeek-V3', trust_remote_code=True)
print(f'Experts: {config.n_routed_experts}')
print(f'Shared: {config.n_shared_experts}')
print(f'Top-K: {config.num_experts_per_tok}')
print(f'Scoring: {getattr(config, \"scoring_func\", \"softmax\")}')
"
```

### Test vLLM Loading

```python
from vllm import LLM

# Minimal smoke test
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    trust_remote_code=True,
    max_model_len=4096,       # Start small
    gpu_memory_utilization=0.9,
    tensor_parallel_size=4,   # Adjust based on GPU count
)

# Test generation
output = llm.generate(
    "Hello, world!",
    sampling_params=SamplingParams(max_tokens=10)
)
print(output[0].outputs[0].text)
```

---

## Next Steps (Phase 2)

1. **Patch DeepseekV2MoE.forward()**:
   - Add import for `_indexer_logger`
   - Insert instrumentation after `self.gate(hidden_states)`
   - Compute post-softmax/sigmoid scores
   - Call `_log_indexer()`

2. **Update `_indexer_logger.py` configuration**:
   - Add environment variable: `INDEXER_IS_EXPERT_ROUTING=1`
   - Adjust default histogram range: `[0.0, 1.0]` for post-softmax
   - Update documentation strings

3. **Sync patches to environment**:
   ```bash
   cd examples/dsa_indexer
   bash sync_to_env.sh
   ```

4. **Smoke test**:
   - Run single-prompt inference with `INDEXER_LOGIT_DUMP_DIR` set
   - Verify `.npz` files are created
   - Check `layer_ids`, `count_prefill`, `count_decode`

5. **Proceed to Phase 3** (Benchmark configuration)

---

## Appendix: File Map

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `vllm/model_executor/models/deepseek_v2.py` | Model definition | 1711 | ⭐ **Patch target** |
| `vllm/model_executor/layers/fused_moe/layer.py` | FusedMoE layer | 1411 | Read-only (reference) |
| `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py` | Top-K routing | 166 | Read-only (reference) |
| `vllm/model_executor/layers/fused_moe/router/base_router.py` | Router base class | ? | Not inspected |
| `vllm/_custom_ops` | Custom CUDA ops | ? | Not needed (Option A) |
| `examples/dsa_indexer/vllm/_indexer_logger.py` | Logger (existing) | 918 | Minor config updates |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| DeepSeek V4 not yet supported in vLLM | Medium | High | Start with V3, update when V4 lands |
| Post-softmax recomputation overhead | Low | Low | Softmax is cheap; already computed in forward pass |
| Histogram range mismatch (post-activation [0,1]) | High | Medium | Update `INDEXER_LOGIT_RANGE` config |
| Layer naming regex fails | Low | Medium | Test `_parse_layer()` with DeepSeek prefix |
| Internal router path (no external gate) | Low | High | Check `is_internal_router` flag, handle both paths |

---

## Summary

✅ **Phase 1 Complete**: DeepSeek V2/V3 support confirmed in vLLM.

**Key findings**:
1. MoE architecture uses `DeepseekV2MoE` → `GateLinear` → `FusedMoE` → `fused_topk()`
2. Top-K selection: 6-8 experts per token (vs 256-2048 tokens for attention)
3. Activation: Softmax (default) or Sigmoid, fused into custom op
4. Instrumentation: Capture raw logits + recompute activation (Option A)
5. Histogram range: [0.0, 1.0] for post-softmax probabilities

**Next**: Proceed to Phase 2 (Code Instrumentation).
