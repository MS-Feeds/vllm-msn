# DeepSeek Top-K Logit Distribution Analysis Plan

**Date**: 2026-06-01  
**Author**: Chen Wu  
**Collaborator**: Joseph Rogers (Stockholm office)  
**Goal**: Analyze post-RELU score distributions for DeepSeek V3/V4 Top-K expert routing to enable threshold-based pruning and radix sort optimization

---

## Executive Summary

Joseph Rogers (Stockholm) requested histogram analysis of Top-K input distributions for DeepSeek models to explore threshold-based pruning optimizations (e.g., "never consider values under 0.5"). This aligns with Chen Wu's ongoing radix sort bit-reduction work for DSA Top-K acceleration.

**Deliverables**:
- Histogram of post-RELU scores across all layers
- Per-layer distribution statistics (mean, std, percentiles)
- Cross-benchmark comparison (Math, Code, QA)
- Threshold recommendation table
- Bit-width analysis for radix sort optimization

**Timeline**: 1 week (pending GPU availability)

---

## Background

### Slack Conversation Summary (2026-05-28 to 2026-05-31)

**Joseph Rogers' Request**:
> I'm interested in seeing a histogram of some kind which shows the spread of all of the scalar values coming out of the RELU function. This could (in theory) allow us to make some assumptions to optimize the Top-K engine (e.g., never consider values under 0.5).

**Chen Wu's Response**:
> I am currently working on accelerating the GLM-5.1 model, and I am looking into Top-K selection for DSA as well. Right now, our baseline relies on a bitonic sort, which performs a full sort and feels somewhat wasteful just to extract the Top-K elements. To optimize this, I've been exploring alternative engines like radix sort or counting sort.
>
> I am in the middle of conducting experiments on the range and distribution analysis of the logits used for the Top-K calculation. My goal is to see if we can reduce the required bits for the exponent/mantissa, which would allow a radix sort to run with fewer scan passes and smaller buckets.

**Agreed Plan**:
- Chen Wu will run experiments on logits distributions
- Top-K is computed for each token → compute mean average for each layer
- Test across math, code, and QA benchmarks to see if patterns vary
- Timeline: ~1 week after GPU access

---

## Current Infrastructure (GLM-5.1 Baseline)

### Existing Tools in `dsa_indexer/`

#### 1. **`vllm/_indexer_logger.py`** (918 lines)
**Purpose**: Per-layer, per-phase logit recorder with 4 data streams.

**Stream A - Relative Gap**:
- Captures: `(logit[k] - logit[k+1]) / |logit[k]|`
- Purpose: Determines precision requirements (BF16 vs FP8 vs FP16)
- Output: Histogram with format-epsilon vertical lines

**Stream B - Raw Logit Range** ⭐ **Joseph's Primary Need**:
- Captures: All valid logit values (masked by valid positions)
- Purpose: Value distribution for threshold analysis
- Output: Per-layer histograms, min/max/mean/std stats

**Stream C - Top-K Winners**:
- Captures: The actual top-k selected values
- Purpose: Winner dynamic range (saturation analysis)
- Output: Moments, min/max per layer

**Stream D - Radix Boundary-Bucket** ⭐ **Chen's Bit-Reduction Work**:
- Captures: Bucket sizes at radix sort boundaries for d ∈ {4,6,8,12,16} bits
- Purpose: Determine minimum bits for single-pass radix select
- Output: Cascade success rate plots, bucket size distributions

**Configuration (Environment Variables)**:
```bash
export INDEXER_LOGIT_DUMP_DIR="/path/to/output"
export INDEXER_LOGIT_RANGE="-50.0,50.0"           # Histogram range
export INDEXER_LOGIT_BINS=4096                    # Histogram bins
export INDEXER_RADIX_BITS_SWEEP="4,6,8,12,16"    # Radix bit widths to test
export INDEXER_RADIX_BKT_BINS=256                 # Bucket size histogram bins
```

#### 2. **`tools/plot_indexer_logits.py`** (1,125 lines)
**Purpose**: Comprehensive visualization suite.

**Outputs**:
- `stream_a_panels_{phase}.png` - Gap histograms with format-epsilon lines
- `stream_a_p1_by_layer.png` - Per-layer p1 (1st percentile) vs format safety
- `safety_heatmap_{phase}.png` - Layer × format heatmap (p1/ε ratio)
- `stream_b_panels_{phase}.png` - Raw logit histograms ⭐ **Joseph needs this**
- `stream_b_summary.png` - Mean ± std, min/max per layer
- `stream_b_saturation.png` - Fraction outside histogram range
- `stream_c_winner_range.png` - Top-k winner range vs format max
- `stream_d_cascade_rate.png` - One-pass success rate vs d (bits)
- `stream_d_summary.png` - Boundary-bucket size by layer × d
- `stream_d_panels_{phase}.png` - Heatmap: layer × bucket size
- `summary.txt` - Text report with all key statistics

#### 3. **`tools/run_eval_prompts.py`** (321 lines)
**Purpose**: Evaluation harness for driving inference with diverse benchmarks.

**Supported Benchmarks**:
- **LiveCodeBench** (`livecodebench/code_generation_lite`): Competitive programming
- **AIME** (`Maxwell-Jia/AIME_2024`): Math competition problems
- **GPQA Diamond** (`Idavidrein/gpqa`): PhD-level science questions

**Features**:
- Concurrent request support (continuous batching)
- Resume capability (skip already-completed prompts)
- JSONL output with latency, usage, completion tracking

---

## Adaptation Requirements

### Model Support Matrix

| Model | Priority | vLLM Version | Top-K Location | Notes |
|---|:---:|:---:|---|---|
| **GLM-5.1** | ✅ DONE | Current | `sparse_attn_indexer.py` | Baseline (working) |
| **DeepSeekV4** | 🔴 HIGH | TBD | MoE expert routing | Joseph's primary target |
| **DeepSeekV3.2** | 🟡 MEDIUM | TBD | MoE expert routing | Vinh/Mingran exploring DSA enablement |

### Architecture Differences

#### GLM-5.1 (Current):
```python
# File: vllm/model_executor/layers/sparse_attn_indexer.py
# Two Top-K calls per layer:
1. Prefill: fp8_mha_mqa_logits(...) → topk(K_total=2048)
2. Decode:  fp8_mha_mqa_logits(...) → topk(K_window=256)

# Instrumentation:
_indexer_logger.record(
    k_cache_prefix=self.prefix,  # e.g. "model.layers.0.self_attn"
    phase="prefill",              # or "decode"
    logits=attention_logits,      # [batch, num_queries, num_keys]
    index_topk=2048,              # K value
    key_valid_starts=starts,      # Masking
    key_valid_ends=ends
)
```

#### DeepSeek V3/V4 (Target):
```python
# File: vllm/model_executor/layers/fused_moe/*.py (TBD - need to locate)
# MoE expert routing:
1. Gate projection: hidden → expert_logits [batch, seq_len, num_experts]
2. Activation: expert_scores = ReLU(expert_logits)  ← Joseph wants THIS
3. Top-K: topk(expert_scores, k=top_k_experts)
4. Expert execution: route tokens to selected experts

# Target instrumentation (pseudo-code):
_indexer_logger.record(
    k_cache_prefix=f"model.layers.{layer_id}.mlp.gate",
    phase="decode",              # Infer from input shape
    logits=expert_scores,        # POST-RELU ← Key difference!
    index_topk=top_k_experts,    # Usually 6-8 for DeepSeek
    key_valid_starts=torch.zeros(batch_size),  # All experts valid
    key_valid_ends=torch.full((batch_size,), num_experts)
)
```

**Key Adaptation Points**:
1. **Value semantics**: GLM-5.1 records attention logits (can be negative), DeepSeek records post-RELU scores (≥0)
2. **Sparsity**: GLM-5.1 has variable-length key sequences (need masking), DeepSeek has fixed expert count (all valid)
3. **Histogram range**: GLM-5.1 uses `[-50, 50]`, DeepSeek should use `[0, 100]` or similar
4. **Layer naming**: GLM-5.1 uses `sparse_attn_indexer`, DeepSeek uses `fused_moe` or similar

---

## Implementation Plan

### Phase 1: Model Support Verification (Day 1)

**Goal**: Confirm DeepSeek V3/V4 works in current vLLM.

**Steps**:
1. Check vLLM model registry:
   ```bash
   grep -r "deepseek" vllm/model_executor/models/
   ```

2. Locate expert routing code:
   ```bash
   find vllm -name "*.py" -exec grep -l "expert.*gate\|expert.*router" {} \;
   ```

3. Test model loading:
   ```python
   from vllm import LLM
   llm = LLM(
       model="deepseek-ai/DeepSeek-V3",  # or V4
       trust_remote_code=True,
       max_model_len=8192,
       gpu_memory_utilization=0.9,
   )
   ```

4. Verify Top-K parameters:
   - How many experts? (DeepSeek V2 has 160 experts, top-6 routing)
   - What activation? (ReLU, SoftMax, Gumbel?)
   - Pre or post activation for Top-K?

**Deliverable**: `docs/DEEPSEEK_MODEL_ANALYSIS.md` with:
- Model architecture summary
- Top-K implementation location
- Instrumentation insertion points

---

### Phase 2: Code Instrumentation (Day 2)

**Goal**: Adapt `_indexer_logger.py` recorder for DeepSeek.

**Changes Required**:

#### 2.1 Update logger to handle expert routing semantics

**File**: `vllm/_indexer_logger.py`

**Modify `record()` signature** (OPTIONAL - maintain backward compatibility):
```python
def record(
    k_cache_prefix: str,
    phase: str,
    logits: torch.Tensor,
    index_topk: int,
    key_valid_starts: torch.Tensor,
    key_valid_ends: torch.Tensor,
    # NEW: Optional flag to indicate this is expert routing (not attention)
    is_expert_routing: bool = False,
) -> None:
```

**Adjust histogram range based on context**:
```python
# Inside _Config.from_env():
# For expert routing (post-RELU), default to [0, 100]
# For attention logits, keep [-50, 50]
if "INDEXER_IS_EXPERT_ROUTING" in os.environ:
    default_range = (0.0, 100.0)
else:
    default_range = (-50.0, 50.0)
```

#### 2.2 Patch DeepSeek model file

**File**: `vllm/model_executor/layers/fused_moe/*.py` (exact path TBD from Phase 1)

**Insertion point** (pseudo-code, adjust based on actual code):
```python
# At top of file
try:
    from vllm._indexer_logger import record as _log_indexer
    _INDEXER_ENABLED = True
except ImportError:
    _INDEXER_ENABLED = False

# In expert router forward():
def forward(self, hidden_states: torch.Tensor, ...):
    # ... existing code ...
    
    # Compute expert logits
    router_logits = self.gate(hidden_states)  # [batch, seq_len, num_experts]
    
    # Apply activation (ReLU or similar)
    router_scores = self.activation(router_logits)  # POST-ACTIVATION
    
    # ===== INSTRUMENTATION START =====
    if _INDEXER_ENABLED:
        batch_size, seq_len, num_experts = router_scores.shape
        # Flatten to [batch*seq_len, num_experts] for logger
        flat_scores = router_scores.reshape(-1, num_experts)
        
        # Infer phase from shape (heuristic: prefill has seq_len > 1)
        phase = "prefill" if seq_len > 1 else "decode"
        
        # All experts are valid (no masking needed)
        starts = torch.zeros(flat_scores.shape[0], dtype=torch.long, device=router_scores.device)
        ends = torch.full((flat_scores.shape[0],), num_experts, dtype=torch.long, device=router_scores.device)
        
        _log_indexer(
            k_cache_prefix=self.prefix,  # e.g., "model.layers.0.mlp.gate"
            phase=phase,
            logits=flat_scores,
            index_topk=self.top_k,
            key_valid_starts=starts,
            key_valid_ends=ends,
        )
    # ===== INSTRUMENTATION END =====
    
    # Top-K selection
    topk_weights, topk_ids = torch.topk(router_scores, k=self.top_k, dim=-1)
    
    # ... rest of routing logic ...
```

**Sync to environment**:
```bash
bash sync_to_env.sh
```

**Deliverable**: 
- Patched `vllm/_indexer_logger.py`
- Patched `vllm/model_executor/layers/fused_moe/*.py`
- Updated `sync_to_env.sh` to include new files

---

### Phase 3: Benchmark Configuration (Day 3)

**Goal**: Define the 3-benchmark experiment matching your commitment to Joseph.

**Dataset Selection**:

| Benchmark | Type | Prompts | Rationale |
|---|:---:|:---:|---|
| **AIME 2024** | Math | 100 | Competition math → reasoning-heavy |
| **LiveCodeBench** | Code | 100 | Competitive programming → long context |
| **GPQA Diamond** | QA | 100 | PhD-level science → knowledge-heavy |
| **Total** | | **300** | Balanced coverage |

**Experiment Configuration**:

**File**: `configs/deepseek_topk_experiment.sh`

```bash
#!/usr/bin/env bash
# DeepSeek Top-K Distribution Experiment
# 3 benchmarks × 100 prompts = 300 total

set -euo pipefail

# Model configuration
export MODEL="deepseek-ai/DeepSeek-V3"  # Or V4 when available
export MAX_MODEL_LEN=8192
export MAX_TOKENS=2048
export TEMPERATURE=0.7
export GPU_MEM=0.90

# Indexer configuration
export INDEXER_LOGIT_DUMP_DIR="indexer_logits/deepseek_v3_$(date +%Y%m%d_%H%M%S)"
export INDEXER_LOGIT_RANGE="0.0,100.0"        # Post-RELU scores ≥ 0
export INDEXER_LOGIT_BINS=4096
export INDEXER_RADIX_BITS_SWEEP="4,6,8,12,16"
export INDEXER_RADIX_BKT_BINS=256
export INDEXER_IS_EXPERT_ROUTING=1             # Signal to logger

# Launch vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --trust-remote-code \
    --port 8000 &

SERVER_PID=$!
echo "vLLM server started (PID=$SERVER_PID)"

# Wait for server ready
sleep 30

# Run evaluation
python tools/run_eval_prompts.py \
    --url http://localhost:8000 \
    --model "$MODEL" \
    --datasets aime livecodebench gpqa_diamond \
    --n-per-dataset 100 \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --concurrency 4 \
    --out "logs/deepseek_eval_$(date +%Y%m%d_%H%M%S).jsonl"

# Shutdown server
kill $SERVER_PID
wait

echo "Experiment complete. Logits dumped to: $INDEXER_LOGIT_DUMP_DIR"
```

**Deliverable**: `configs/deepseek_topk_experiment.sh`

---

### Phase 4: Execution (Day 4-5)

**Goal**: Run the 300-prompt experiment and capture distributions.

**Resource Requirements**:
- **GPU**: 4-8× A100 80GB (DeepSeek V3 is ~685B parameters)
- **Time**: ~6-10 hours for 300 prompts (estimate: 1-2 min/prompt)
- **Storage**: ~50 GB for checkpoints + 5 GB for logit captures

**Execution Steps**:

1. **Pre-flight checks**:
   ```bash
   # Verify model downloads
   huggingface-cli download deepseek-ai/DeepSeek-V3
   
   # Verify datasets accessible
   python -c "from datasets import load_dataset; load_dataset('Maxwell-Jia/AIME_2024', split='train')"
   ```

2. **Run experiment**:
   ```bash
   cd examples/dsa_indexer
   bash configs/deepseek_topk_experiment.sh
   ```

3. **Monitor progress**:
   ```bash
   # Watch logit captures accumulate
   watch -n 60 "ls -lh $INDEXER_LOGIT_DUMP_DIR/*.npz"
   
   # Monitor eval progress
   tail -f logs/deepseek_eval_*.jsonl | jq -r '.ok, .latency'
   ```

4. **Verify captures**:
   ```python
   import numpy as np
   d = np.load("indexer_logits/.../indexer_logits_rank0.npz")
   print(f"Layers captured: {d['layer_ids']}")
   print(f"Prefill rows: {d['count_prefill']}")
   print(f"Decode rows: {d['count_decode']}")
   ```

**Deliverable**: 
- `indexer_logits/deepseek_v3_*/indexer_logits_rank*.npz` (raw captures)
- `logs/deepseek_eval_*.jsonl` (eval results)

---

### Phase 5: Analysis & Visualization (Day 6)

**Goal**: Generate all plots and statistics for Joseph's report.

**Analysis Script**:

```bash
cd examples/dsa_indexer

# Generate all visualizations
python tools/plot_indexer_logits.py \
    --inputs indexer_logits/deepseek_v3_*/indexer_logits_rank*.npz \
    --out figures/deepseek_v3_run01 \
    --aggregate

# Output files:
# - figures/deepseek_v3_run01/summary.txt               ← Quick reference
# - figures/deepseek_v3_run01/stream_b_panels_*.png    ← Joseph's histograms
# - figures/deepseek_v3_run01/stream_b_summary.png     ← Mean/std per layer
# - figures/deepseek_v3_run01/stream_d_cascade_rate.png ← Radix optimization
# - ... (20+ plots)
```

**Additional Analysis for Joseph**:

**File**: `tools/threshold_analysis.py`

```python
#!/usr/bin/env python3
"""Generate threshold recommendation table for DeepSeek Top-K pruning.

Computes P(score < threshold) for a range of threshold values, per layer.
Answers Joseph's question: "Can we prune values under 0.5?"
"""
import numpy as np
from pathlib import Path

def threshold_analysis(npz_path: Path):
    d = np.load(npz_path)
    layers = d['layer_ids']
    bins = int(d['bins'])
    lo, hi = float(d['range_lo']), float(d['range_hi'])
    
    # Thresholds to test (Joseph mentioned 0.5)
    thresholds = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0]
    
    print("# DeepSeek Top-K Threshold Analysis")
    print()
    print("P(score < threshold) — percentage of values below threshold")
    print()
    
    for phase in ('prefill', 'decode'):
        hist = d[f'hist_{phase}']  # [num_layers, bins]
        count = d[f'count_{phase}']
        
        if count.sum() == 0:
            continue
        
        print(f"## {phase.upper()}")
        print()
        print(f"| Layer | " + " | ".join(f"< {t}" for t in thresholds) + " | Recommendation |")
        print("|-------|" + "|".join("-----:" for _ in thresholds) + "|----------------|")
        
        for i, layer_id in enumerate(layers):
            if count[i] == 0:
                continue
            
            # Compute CDF
            edges = np.linspace(lo, hi, bins + 1)
            h = hist[i].astype(float)
            cum = np.cumsum(h)
            total = cum[-1]
            
            row = [f"L{layer_id:02d}"]
            for thresh in thresholds:
                # Find bin index for threshold
                bin_idx = int(np.searchsorted(edges, thresh))
                if bin_idx >= bins:
                    p = 1.0
                else:
                    p = cum[bin_idx] / total if total > 0 else 0.0
                row.append(f"{p*100:.1f}%")
            
            # Recommendation: find lowest threshold capturing < 1% of values
            rec_thresh = None
            for thresh in thresholds:
                bin_idx = int(np.searchsorted(edges, thresh))
                p = (cum[bin_idx] / total) if (bin_idx < bins and total > 0) else 1.0
                if p < 0.01:  # Less than 1% below threshold → safe to prune
                    rec_thresh = thresh
                    break
            
            if rec_thresh:
                row.append(f"Prune < {rec_thresh}")
            else:
                row.append("No safe threshold")
            
            print("| " + " | ".join(row) + " |")
        
        print()

if __name__ == "__main__":
    import sys
    threshold_analysis(Path(sys.argv[1]))
```

**Run threshold analysis**:
```bash
python tools/threshold_analysis.py indexer_logits/deepseek_v3_*/indexer_logits_rank0.npz > figures/deepseek_v3_run01/threshold_recommendations.txt
```

**Deliverable**: 
- All plots from `plot_indexer_logits.py`
- `threshold_recommendations.txt` (Joseph's pruning guide)

---

### Phase 6: Report Writing (Day 7)

**Goal**: Synthesize findings into actionable report for Joseph.

**File**: `docs/DEEPSEEK_TOPK_EXPERIMENT_RESULTS.md`

**Structure** (mirror `INDEXER_LOGIT_EXPERIMENT.md`):

```markdown
# DeepSeek Top-K Distribution Experiment — Results

**Date**: 2026-06-XX  
**Model**: DeepSeek-V3  
**Prompts**: 300 (100 math, 100 code, 100 QA)  
**Requestor**: Joseph Rogers (Stockholm office)

---

## TL;DR

**Goal**: Measure post-RELU score distributions for Top-K expert routing to enable 
threshold-based pruning.

**Headline Findings**:
1. **Threshold Viability**: XX% of scores < 0.5 across all layers (see §3)
2. **Cross-Benchmark Variance**: Math/Code/QA show [similar/different] patterns (§4)
3. **Bit-Width Recommendation**: d=12 bits sufficient for single-pass radix (§5)
4. **Layer Heterogeneity**: Layers 0-10 concentrate differently than 20-30 (§6)

**RDU Recommendation**: [Threshold = X.X] OR [No threshold viable] (see §7)

---

## §1. Experiment Setup

[Dataset details, model config, capture methodology]

---

## §2. Histogram Overview

**Figure 2.1**: Per-layer post-RELU score distributions (prefill)

![Stream B Panels - Prefill](../figures/deepseek_v3_run01/stream_b_panels_prefill.png)

Key observations:
- Range: [0, XX.X]
- Mean: X.X ± Y.Y
- Outliers: ZZ% > 10.0

---

## §3. Threshold Analysis (Joseph's Question)

**Joseph's Question**: "Can we prune values under 0.5?"

**Answer**: [YES/NO/CONDITIONAL]

| Layer | P(score < 0.1) | P(score < 0.5) | P(score < 1.0) | Recommendation |
|-------|----------------|----------------|----------------|----------------|
| L00   | 2.3%           | 18.7%          | 45.2%          | Prune < 0.1    |
| L01   | 1.9%           | 16.3%          | 42.8%          | Prune < 0.1    |
| ...   | ...            | ...            | ...            | ...            |

**Interpretation**:
- Layers 0-15: Safe to prune < 0.1 (< 5% loss)
- Layers 16-30: Higher concentration near zero, pruning risky
- **Conservative recommendation**: Threshold = 0.05 (99.8% recall)

---

## §4. Cross-Benchmark Comparison

**Question**: "Does the pattern vary across different benchmarks?"

**Answer**: [YES/NO] — see Figure 4.1

![Benchmark Overlay](../figures/deepseek_v3_run01/benchmark_comparison.png)

- Math (AIME): [description]
- Code (LiveCodeBench): [description]
- QA (GPQA): [description]

**Conclusion**: [One threshold for all OR per-benchmark tuning needed]

---

## §5. Radix Sort Optimization (Chen's Goal)

**Current RDU Baseline**: Bitonic sort (full sort for Top-K)

**Radix Analysis**:

| d (bits) | Buckets | One-pass fit | Cascade depth | Memory | Verdict |
|----------|---------|--------------|---------------|--------|---------|
| 4        | 16      | 12%          | ~4 passes     | 64 B   | Too slow |
| 8        | 256     | 68%          | ~2 passes     | 1 KB   | Viable |
| 12       | 4,096   | **99.7%**    | **1 pass**    | 16 KB  | **Recommended** |
| 16       | 65,536  | 100%         | 1 pass        | 256 KB | Overkill |

**Recommendation for RDU**: **d=12 bits, 4,096 buckets, 1-pass radix select**

Trade-off: 16 KB on-chip memory per row vs. ~4× fewer passes than d=4.

---

## §6. Layer Heterogeneity

[Analysis of per-layer variance — do early layers behave differently than late layers?]

---

## §7. Recommendations

**For Joseph (Threshold Pruning)**:
1. Conservative: Prune values < 0.05 (expected <1% accuracy loss)
2. Aggressive: Prune values < 0.5 (expect 5-15% accuracy loss, needs validation)
3. Benchmark-specific tuning: Not needed (patterns are consistent)

**For Chen (Radix Sort)**:
1. Use top-12 bits (bits 20-31 of sortable uint32)
2. 4,096 buckets, single-pass select
3. ~16 KB per-row SMEM budget

**For DSA Team (RDU Implementation)**:
1. [Specific recommendations for RDU hardware]

---

## §8. Appendix

- Full capture stats: `summary.txt`
- All plots: `figures/deepseek_v3_run01/`
- Raw data: `indexer_logits/deepseek_v3_*/`

---

## Acknowledgements

- Joseph Rogers (Stockholm): Problem definition, threshold pruning idea
- Nasim Farahini, Romy Tsoupidi (Stockholm): DeepSeek V4 Top-K work
- Tuowen Zhao, Vinh Nguyen, Mingran (DSA team): DeepSeek V3.2 DSA enablement
- Chen Wu: Experiment execution, radix sort analysis
```

**Deliverable**: `docs/DEEPSEEK_TOPK_EXPERIMENT_RESULTS.md`

---

## Success Criteria

### Minimum Viable Deliverable (for Joseph)

✅ **Histogram plots**: Post-RELU score distributions per layer  
✅ **Threshold table**: P(score < X) for X ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0}  
✅ **Cross-benchmark comparison**: Math vs Code vs QA overlay  
✅ **Recommendation**: "Can we prune < 0.5?" → Yes/No + rationale

### Extended Deliverable (for DSA team)

✅ **Radix optimization**: Bit-width recommendation + cascade analysis  
✅ **Memory budget**: On-chip memory requirements per configuration  
✅ **Format analysis**: BF16 vs FP8 vs FP16 safety assessment

### Stretch Goals (if time permits)

- Compare DeepSeek V3 vs V4 