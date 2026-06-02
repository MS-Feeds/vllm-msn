# DeepSeek V4 Flash Instrumentation

**Date**: 2026-06-01  
**Model**: DeepSeek V4 Flash (MXFP4 quantized)  
**Status**: ✅ Complete

---

## Summary

Successfully instrumented **DeepSeek V4 Flash** MoE expert routing for Top-K logit distribution analysis. V4 uses a separate implementation from V2/V3, so required dedicated patches.

---

## Files Patched

### `vllm/models/deepseek_v4/nvidia/model.py`

**Changes**:
1. ✅ Added `_indexer_logger` import with fallback (lines ~10-19)
2. ✅ Instrumented `DeepseekV4MoE.forward()` (main path, lines ~775-832)
3. ✅ Instrumented `DeepseekV4MoE._forward_fused_moe()` (alternative path, lines ~869-896)

**Instrumentation locations**:
```python
# Main forward path (mega_moe=True)
normed_x, router_logits = self.norm_gate(hidden_states)
# ===== INSTRUMENTATION HERE (line 777) =====
topk_weights, topk_ids = fused_topk_bias(...)

# Alternative path (mega_moe=False)  
normed_x, router_logits = self.norm_gate(hidden_states)
# ===== INSTRUMENTATION HERE (line 871) =====
final_hidden_states = self.experts(...)
```

---

## V4 vs V2/V3 Differences

| Aspect | V2/V3 | V4 Flash |
|--------|-------|----------|
| **File location** | `model_executor/models/deepseek_v2.py` | `models/deepseek_v4/nvidia/model.py` |
| **Gate module** | `GateLinear` | `NormGateLinear` |
| **Expert routing** | `fused_topk()` | `fused_topk_bias()` |
| **Hash routing** | No | Optional (tid2eid table) |
| **Quantization** | BF16/FP8 | MXFP4/FP8 |
| **MoE variants** | Single path | Two paths (mega_moe flag) |
| **Shared experts** | Optional | Optional |

---

## Deployment Instructions

### Step 1: Sync V4 Patches

```bash
cd /nvmedata/chenw/vllm-ra/examples/dsa_indexer

# Deploy V4 patch to vllm-ablation environment
VLLM_INSTALL_PATH="/root/miniconda3/envs/vllm-ablation/lib/python3.10/site-packages/vllm" \
    bash sync_to_env.sh
```

**Note**: The sync script now includes V4 files:
- `models/deepseek_v4/nvidia/model.py` (new)
- `model_executor/models/deepseek_v2.py` (existing)
- `_indexer_logger.py` (existing)

### Step 2: Verify Installation

```bash
conda activate vllm-ablation

python -c "
from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM
print('✓ DeepSeek V4 model imports successfully')
"
```

### Step 3: Launch Server

```bash
# Set up indexer logging
export INDEXER_LOGIT_DUMP_DIR="./indexer_logits/deepseek_v4_flash_$(date +%Y%m%d_%H%M%S)"
export INDEXER_IS_EXPERT_ROUTING=1
export INDEXER_LOGIT_RANGE="0.0,1.0"

# Launch V4 Flash server
bash launch_server.sh deepseek-ai/DeepSeek-V4-Flash 8000

# OR for V4 Flash-Base (FP8):
# bash launch_server.sh deepseek-ai/DeepSeek-V4-Flash-Base 8000
```

---

## Configuration

### V4 Flash Specific Settings

```bash
# Model
export DEEPSEEK_MODEL="deepseek-ai/DeepSeek-V4-Flash"

# Hardware (V4 Flash is smaller, needs fewer GPUs)
export TP_SIZE=2              # V4 Flash: 2-4 GPUs
export GPU_MEM=0.90
export MAX_MODEL_LEN=8192

# Indexer (same as V2/V3)
export INDEXER_LOGIT_DUMP_DIR="./indexer_logits/v4_flash_run"
export INDEXER_IS_EXPERT_ROUTING=1
export INDEXER_LOGIT_RANGE="0.0,1.0"
```

---

## GPU Requirements

### Memory Estimates

| Model | Quant | GPUs | Memory/GPU | Notes |
|-------|-------|------|------------|-------|
| **V4-Flash** | MXFP4 | 2-4× A100 | 40-60 GB | Smallest, fastest |
| **V4-Flash-Base** | FP8 | 4× A100 | 60-70 GB | Medium |
| **V3** | BF16 | 8× A100 | 70-80 GB | Largest |
| **V3.2** (DSA) | BF16 | 8× A100 | 70-80 GB | With sparse attn |

---

## Testing

### Quick Smoke Test

```bash
# After server starts, test with single request
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "DeepSeek-V4-Flash",
        "messages": [{"role": "user", "content": "Hello, test!"}],
        "max_tokens": 50
    }' | jq .

# Check logit capture
ls -lh $INDEXER_LOGIT_DUMP_DIR/
# Should see: indexer_logits_rank*.npz files
```

### Verify Instrumentation Active

```bash
# Check server log for indexer messages
grep "indexer_logger\|INDEXER" logs/server_*.log | head -5

# Expected output:
# [_indexer_logger] rank=0 dump_path=.../indexer_logits_rank0.npz
```

---

## V4 Architecture Notes

### MoE Routing in V4

DeepSeek V4 has **two MoE routing paths**:

#### Path 1: Mega MoE (main)
```python
# Line 765-862
def forward(self, hidden_states, input_ids=None):
    normed_x, router_logits = self.norm_gate(hidden_states)
    # ← Instrumented here
    topk_weights, topk_ids = fused_topk_bias(
        gating_output=router_logits,
        scoring_func=self.scoring_func,  # "softmax" or "sigmoid"
        topk=self.n_activated_experts,
        ...
    )
    final_hidden_states = self.experts(normed_x, topk_weights, topk_ids)
```

**Features**:
- Uses `fused_topk_bias()` with bias correction
- Supports hash-based routing (optional `tid2eid` table)
- Explicit Top-K selection before expert execution

#### Path 2: Fused MoE (fallback)
```python
# Line 864-876
def _forward_fused_moe(self, hidden_states, input_ids=None):
    normed_x, router_logits = self.norm_gate(hidden_states)
    # ← Instrumented here
    final_hidden_states = self.experts(
        hidden_states=normed_x,
        router_logits=router_logits,
        input_ids=input_ids,
    )
```

**Features**:
- Top-K selection happens inside `self.experts()`
- Simpler code path
- Used when `use_mega_moe=False`

**Both paths are instrumented** to ensure complete coverage.

---

## Differences from Phase 2 (V2/V3)

### New for V4

1. **Platform-specific implementation**: V4 has separate `nvidia/` and `amd/` subdirectories
2. **NormGateLinear**: V4 uses normalized gating (vs plain GateLinear in V2/V3)
3. **Hash-based routing**: V4 supports optional token→expert hash tables
4. **Dual forward paths**: Mega MoE vs Fused MoE (V2/V3 had one path)
5. **MXFP4 quantization**: 4-bit expert weights (V2/V3 used BF16/FP8)

### Unchanged

1. **Indexer logger API**: Same `record()` function signature
2. **Activation functions**: Still softmax/sigmoid for expert scores
3. **Histogram range**: Still `[0, 1]` for post-activation probabilities
4. **Analysis tools**: Same `plot_indexer_logits.py` and `threshold_analysis.py`

---

## Known Issues

### Issue: Platform Detection

V4 uses runtime platform detection to choose nvidia vs amd implementation:

```python
# vllm/models/deepseek_v4/__init__.py
if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia.model import DeepseekV4ForCausalLM
else:
    from .amd.model import DeepseekV4ForCausalLM
```

**Impact**: Our patches only cover NVIDIA path. AMD users would need separate patches to `amd/model.py`.

**Workaround**: Current instrumentation works for NVIDIA GPUs (A100, H100). AMD support can be added later if needed.

---

## Validation Checklist

Before running full experiment:

- [ ] V4 patches synced to vllm-ablation environment
- [ ] `DeepseekV4ForCausalLM` imports without error
- [ ] Server launches successfully
- [ ] Single request creates `.npz` files
- [ ] Server log shows `[_indexer_logger] rank=X dump_path=...`
- [ ] Logit values in `[0, 1]` range (post-softmax)

---

## Next Steps

1. ✅ V4 instrumentation complete
2. → Run experiment (see `QUICKSTART_DEEPSEEK.md`)
3. → Compare V4 Flash vs V3 results
4. → Analyze: is V4 Flash suitable for production given threshold constraints?

---

## Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `vllm/models/deepseek_v4/nvidia/model.py` | V4 MoE instrumentation | ✅ Patched |
| `vllm/_indexer_logger.py` | Logger (shared V2/V3/V4) | ✅ Ready |
| `docs/DEEPSEEK_V4_INSTRUMENTATION.md` | This document | ✅ Complete |

**Total lines added to V4**: ~110 lines (2 instrumentation blocks + imports)
