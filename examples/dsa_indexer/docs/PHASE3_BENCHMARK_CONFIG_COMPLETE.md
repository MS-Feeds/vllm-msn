# Phase 3: Benchmark Configuration — Complete

**Date**: 2026-06-01  
**Status**: ✅ Complete  
**Next**: Ready for Experiment Execution (Phase 4)

---

## Summary

Phase 3 successfully created all experiment configuration files, benchmark scripts, and analysis tools. The full 300-prompt evaluation is ready to run.

---

## Deliverables Created

### 1. **Main Experiment Script** ✅
**Path**: `configs/deepseek_topk_experiment.sh`

**Features**:
- ✅ End-to-end automation (server launch → evaluation → analysis)
- ✅ Configurable via environment variables
- ✅ 4-phase execution:
  1. Launch vLLM server
  2. Run 300-prompt evaluation
  3. Shutdown server
  4. Verify output files
- ✅ Dry-run mode for validation
- ✅ Comprehensive error handling and pre-flight checks
- ✅ Progress tracking and logging

**Usage**:
```bash
# Dry run (verify config without executing)
bash configs/deepseek_topk_experiment.sh --dry-run

# Full run
bash configs/deepseek_topk_experiment.sh
```

**Configuration**:
```bash
# Model
export DEEPSEEK_MODEL="deepseek-ai/DeepSeek-V3"
export MAX_MODEL_LEN=8192
export MAX_TOKENS=2048
export TEMPERATURE=0.7

# Hardware
export TP_SIZE=4              # Tensor parallel size
export GPU_MEM=0.90           # GPU memory utilization
export VLLM_PORT=8000

# Workload
export PROMPTS_PER_DATASET=100  # 100 × 3 = 300 total
export CONCURRENCY=4            # In-flight requests
export SEED=42

# Indexer
export INDEXER_LOGIT_DUMP_DIR="./indexer_logits/deepseek_run01"
export INDEXER_IS_EXPERT_ROUTING=1
```

---

### 2. **Threshold Analysis Tool** ✅
**Path**: `tools/threshold_analysis.py`

**Purpose**: Answer Joseph's question: "Can we prune values under 0.5?"

**Features**:
- ✅ Computes P(score < threshold) per layer
- ✅ Generates pruning recommendation table
- ✅ Finds safe thresholds (< 1% loss)
- ✅ Markdown output for easy sharing
- ✅ Aggregate statistics across layers

**Usage**:
```bash
python3 tools/threshold_analysis.py \
    indexer_logits/run01/indexer_logits_rank0.npz \
    > figures/run01/threshold_recommendations.txt
```

**Output Format**:
```markdown
## PREFILL

P(score < threshold) — percentage of expert scores below threshold

| Layer | < 0.01 | < 0.05 | < 0.10 | < 0.30 | < 0.50 | < 0.70 | < 1.00 | Recommendation |
|-------|--------|--------|--------|--------|--------|--------|--------|----------------|
| L00   |  2.3%  |  8.7%  | 18.2%  | 45.3%  | 72.1%  | 89.4%  | 100%   | Prune < 0.015  |
| L01   |  1.9%  |  7.2%  | 16.8%  | 43.1%  | 70.5%  | 88.2%  | 100%   | Prune < 0.012  |
...

### Prefill Summary

**Mean P(score < threshold) across all layers:**
- `< 0.01`: 2.1% (range: 1.5% – 3.2%)
- `< 0.05`: 8.0% (range: 6.1% – 10.3%)
- `< 0.50`: 71.3% (range: 68.2% – 75.8%)

**Recommended thresholds (< 1.0% loss):**
- Minimum (most conservative): 0.008
- Median: 0.013
- Maximum (most aggressive): 0.019

**Suggestion**: Use threshold = 0.008 to be safe across all layers.

## Interpretation

**For Joseph's question: "Can we prune values under 0.5?"**

✗ **NO**: Pruning < 0.5 would discard 71% of scores on average (up to 76% in worst layer). 
This is likely too aggressive.
```

---

### 3. **Quick Start Guide** ✅
**Path**: `QUICKSTART_DEEPSEEK.md`

**Sections**:
1. ✅ Prerequisites (GPU, software, disk space)
2. ✅ 7-step walkthrough:
   - Deploy patches
   - Test instrumentation
   - Configure experiment
   - Download model
   - Run experiment (6-10 hours)
   - Analyze results
   - Share with collaborators
3. ✅ Troubleshooting guide
4. ✅ Advanced options (custom benchmarks, resume, comparisons)
5. ✅ Timeline reference
6. ✅ Email templates for deliverables

**Target audience**: Someone reproducing the experiment from scratch.

---

## Experiment Design

### Benchmark Selection

| Benchmark | Type | Prompts | Dataset | Rationale |
|-----------|------|---------|---------|-----------|
| **AIME** | Math | 100 | `Maxwell-Jia/AIME_2024` | Competition math problems → reasoning-heavy |
| **LiveCodeBench** | Code | 100 | `livecodebench/code_generation_lite` | Competitive programming → long context |
| **GPQA Diamond** | QA | 100 | `Idavidrein/gpqa` | PhD-level science → knowledge-intensive |
| **Total** | — | **300** | — | Balanced coverage across domains |

**Why these three?**
- **Math (AIME)**: Tests reasoning chains, step-by-step logic
- **Code (LiveCodeBench)**: Tests algorithmic thinking, problem decomposition
- **QA (GPQA)**: Tests knowledge retrieval, domain expertise
- **Coverage**: Matches Chen's commitment to Joseph (math, code, QA)

### Sampling Strategy

- **100 prompts per benchmark**: Statistical significance while keeping runtime manageable
- **Random seed=42**: Reproducible sampling
- **Shuffled order**: Prevents benchmark-specific warm-up effects

### Expected Outputs

**Per run**:
1. **Logit captures**: `indexer_logits/deepseek_v3_YYYYMMDD_HHMMSS/indexer_logits_rank*.npz`
   - ~500 MB per rank (4 ranks × 500 MB = 2 GB total typical)
   - Contains: histograms, counts, min/max, radix analysis
   
2. **Eval log**: `logs/eval_deepseek_v3_YYYYMMDD_HHMMSS.jsonl`
   - ~2 MB for 300 prompts
   - Contains: prompt text, completion, latency, usage stats
   
3. **Server log**: `logs/server_deepseek_v3_YYYYMMDD_HHMMSS.log`
   - ~10-50 MB depending on verbosity
   - Contains: vLLM startup, throughput stats, errors

---

## Analysis Pipeline

### Step 1: Visualization (10 min)

```bash
python3 tools/plot_indexer_logits.py \
    --inputs indexer_logits/deepseek_v3_*/indexer_logits_rank*.npz \
    --out figures/deepseek_v3_run01 \
    --aggregate
```

**Outputs** (20+ plots):
- `stream_b_panels_*.png` — Per-layer score histograms (Joseph's request)
- `stream_b_summary.png` — Mean/std per layer
- `stream_b_saturation.png` — Fraction outside range
- `stream_c_winner_range.png` — Top-K winner dynamic range
- `stream_d_cascade_rate.png` — Radix one-pass success rate
- `stream_d_summary.png` — Boundary-bucket size by layer
- `summary.txt` — Text report with key stats

### Step 2: Threshold Analysis (1 min)

```bash
python3 tools/threshold_analysis.py \
    indexer_logits/deepseek_v3_*/indexer_logits_rank0.npz \
    > figures/deepseek_v3_run01/threshold_recommendations.txt
```

**Outputs**:
- Markdown table: P(score < threshold) per layer
- Recommended safe thresholds (< 1% loss)
- Answer to Joseph's pruning question

### Step 3: Report Writing (1-2 hours)

Template provided in `docs/DEEPSEEK_TOPK_ANALYSIS_PLAN.md` (Phase 6).

**Key sections**:
1. TL;DR (headline findings)
2. Experiment setup
3. Histogram overview (Joseph's plots)
4. Threshold analysis (Joseph's question)
5. Cross-benchmark comparison
6. Radix sort optimization (Chen's goal)
7. Recommendations (for Joseph + DSA team)

---

## Resource Planning

### GPU Requirements

**DeepSeek V3** (685B parameters):
- **Minimum**: 4× A100 80GB (FP16)
- **Recommended**: 8× A100 80GB (headroom for large batches)
- **Memory per GPU**: ~60-70 GB during inference

**DeepSeek V2** (236B parameters):
- **Minimum**: 2× A100 80GB
- **Recommended**: 4× A100 80GB

### Time Estimates

| Phase | Duration | Notes |
|-------|----------|-------|
| Model download | 10-20 min | One-time, ~40-50 GB |
| Server startup | 2-5 min | Model loading |
| Evaluation | 6-10 hours | 300 prompts @ ~1-2 min/prompt |
| Server shutdown | < 1 min | Graceful |
| Visualization | 10 min | 20+ plots |
| Threshold analysis | 1 min | Single npz file |
| **Total** | **7-11 hours** | Mostly evaluation |

**Throughput assumptions**:
- Prefill: 500-1000 tokens/s
- Decode: 20-40 tokens/s (per sequence)
- Concurrency: 4 (continuous batching)
- Average prompt: 200 tokens input, 500 tokens output

### Disk Space

| Item | Size | Notes |
|------|------|-------|
| Model checkpoint | 40-50 GB | V3, FP16 |
| Logit captures | 2-5 GB | All ranks combined |
| Eval log | 2 MB | 300 prompts |
| Server log | 10-50 MB | Depends on verbosity |
| Plots | 50 MB | 20+ PNGs |
| **Total** | **50-60 GB** | Per experiment run |

---

## Validation Checklist

Before running the full experiment:

- [ ] **Patches deployed**: `./sync_to_env.sh` completed
- [ ] **Smoke test passed**: `python3 test_deepseek_instrumentation.py`
- [ ] **Model downloaded**: `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/`
- [ ] **GPU available**: `nvidia-smi` shows 4+ GPUs
- [ ] **Config verified**: `bash configs/deepseek_topk_experiment.sh --dry-run`
- [ ] **Disk space**: `df -h` shows 60+ GB free
- [ ] **Environment vars**: `INDEXER_LOGIT_DUMP_DIR` and `INDEXER_IS_EXPERT_ROUTING=1`

---

## Risk Mitigation

### Risk 1: Server OOM

**Likelihood**: Medium (if model too large for GPU count)

**Mitigation**:
- Start with `MAX_MODEL_LEN=4096` (down from 8192)
- Reduce `GPU_MEM=0.85` (down from 0.90)
- Increase `TP_SIZE` if more GPUs available
- Use quantization (FP8) if supported

**Detection**:
```bash
# Monitor GPU memory during startup
watch -n 1 nvidia-smi
```

### Risk 2: Evaluation Hangs

**Likelihood**: Low (network issues or server crash)

**Mitigation**:
- `--resume` flag allows restarting from last checkpoint
- `CONCURRENCY=2` reduces load on server
- `--timeout 600` (10 min) per request prevents infinite hangs

**Recovery**:
```bash
# Kill server
pkill -f vllm.entrypoints.openai.api_server

# Resume evaluation
python3 tools/run_eval_prompts.py ... --resume
```

### Risk 3: Logit Captures Missing

**Likelihood**: Low (if env vars not propagated to server)

**Detection**:
```bash
# After 5 min of evaluation, check:
ls indexer_logits/deepseek_v3_*/
# Should see indexer_logits_rank*.npz files
```

**Mitigation**:
- Verify env vars before server launch: `printenv | grep INDEXER`
- Check server log for "[_indexer_logger] rank=X dump_path=..."
- Run quick 1-prompt test first

---

## Advanced Experiment Variations

### Variation 1: Larger Sample (1,000 prompts)

```bash
export PROMPTS_PER_DATASET=333  # 999 total
bash configs/deepseek_topk_experiment.sh
# Estimated time: 20-30 hours
```

### Variation 2: Single Benchmark Deep-Dive

```bash
# Math-only (500 prompts)
export PROMPTS_PER_DATASET=500
python3 tools/run_eval_prompts.py \
    --url http://localhost:8000 \
    --model DeepSeek-V3 \
    --datasets aime \
    --n-per-dataset 500 \
    --out logs/deepseek_math_deep.jsonl
```

### Variation 3: Compare V2 vs V3

```bash
# Run V2
export DEEPSEEK_MODEL="deepseek-ai/DeepSeek-V2"
bash configs/deepseek_topk_experiment.sh

# Run V3
export DEEPSEEK_MODEL="deepseek-ai/DeepSeek-V3"
bash configs/deepseek_topk_experiment.sh

# Compare histograms side-by-side
```

### Variation 4: Temperature Sweep

```bash
for temp in 0.0 0.5 1.0; do
    export TEMPERATURE=$temp
    bash configs/deepseek_topk_experiment.sh
done
# Check if routing distributions change with temperature
```

---

## Success Criteria for Phase 3

✅ **All criteria met**:

1. ✅ Experiment script created and tested (dry-run mode)
2. ✅ Benchmark configuration finalized (3 datasets, 100 prompts each)
3. ✅ Threshold analysis tool implemented
4. ✅ Quick-start guide written for reproducibility
5. ✅ Resource requirements documented
6. ✅ Validation checklist provided
7. ✅ Risk mitigation strategies defined
8. ✅ Analysis pipeline documented

**Phase 3 is complete. Ready for Phase 4 (Execution).**

---

## Files Summary

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `configs/deepseek_topk_experiment.sh` | 350 | Main experiment script | ✅ Ready |
| `tools/threshold_analysis.py` | 350 | Pruning threshold analysis | ✅ Ready |
| `QUICKSTART_DEEPSEEK.md` | 600 | User guide | ✅ Ready |
| `docs/PHASE3_BENCHMARK_CONFIG_COMPLETE.md` | (this) | Phase summary | ✅ Complete |

**Total**: ~1,300 lines of new code + documentation for Phase 3.

---

## Next Steps (Phase 4: Execution)

**When ready to run**:

```bash
cd /nvmedata/chenw/vllm-ra/examples/dsa_indexer

# 1. Final pre-flight check
bash configs/deepseek_topk_experiment.sh --dry-run

# 2. Launch experiment (background, with logging)
nohup bash configs/deepseek_topk_experiment.sh > experiment_output.log 2>&1 &

# 3. Monitor progress
tail -f logs/eval_deepseek_v3_*.jsonl | jq -r '.source, .latency'

# 4. After completion (~6-10 hours), analyze
RUN_DIR=$(ls -td indexer_logits/deepseek_v3_* | head -1)
python3 tools/plot_indexer_logits.py --inputs "$RUN_DIR"/*.npz --out "figures/$(basename $RUN_DIR)" --aggregate
python3 tools/threshold_analysis.py "$RUN_DIR/indexer_logits_rank0.npz" > "figures/$(basename $RUN_DIR)/threshold_recommendations.txt"

# 5. Share results (follow QUICKSTART_DEEPSEEK.md § Step 7)
```

**User confirmation needed before Phase 4**:
- GPU resources available?
- Ready to commit 6-10 hours of GPU time?
- Results destination prepared (for sharing with Joseph)?

---

## Appendix: Experiment Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Server Launch (2-5 min)                                │
│   • Load DeepSeek V3 checkpoint                                  │
│   • Initialize TP=4 workers                                      │
│   • Start OpenAI-compatible API server                           │
│   • Wait for /health endpoint                                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Evaluation (6-10 hours)                                │
│   ┌─────────────────┬─────────────────┬─────────────────────┐  │
│   │ AIME (Math)     │ LiveCodeBench   │ GPQA Diamond (QA)   │  │
│   │ 100 prompts     │ (Code)          │ 100 prompts         │  │
│   │                 │ 100 prompts     │                     │  │
│   └────────┬────────┴────────┬────────┴──────────┬──────────┘  │
│            │                  │                    │             │
│            └──────────────────┴────────────────────┘             │
│                              │                                   │
│                              ▼                                   │
│         ┌─────────────────────────────────────────┐             │
│         │ For each prompt:                        │             │
│         │  1. POST /v1/chat/completions           │             │
│         │  2. Server routes through MoE layers    │             │
│         │  3. Indexer logger captures scores      │             │
│         │  4. Return completion                   │             │
│         │  5. Log to eval_*.jsonl                 │             │
│         └─────────────────────────────────────────┘             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Shutdown (< 1 min)                                     │
│   • Trigger indexer logger dump (atexit)                         │
│   • Write indexer_logits_rank*.npz files                         │
│   • Graceful server shutdown                                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: Analysis (10-15 min)                                   │
│   • plot_indexer_logits.py  → 20+ plots                         │
│   • threshold_analysis.py   → threshold recommendations          │
│   • summary.txt             → text report                        │
└─────────────────────────────────────────────────────────────────┘
```
