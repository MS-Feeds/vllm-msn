# Quick Start: DeepSeek Top-K Experiment

**Goal**: Capture and analyze post-softmax expert routing scores for DeepSeek V2/V3 MoE models across math, code, and QA benchmarks.

**Time estimate**: 
- Setup: 30 minutes
- Experiment run: 6-10 hours (300 prompts)
- Analysis: 1-2 hours

---

## Prerequisites

- **GPU**: 4-8× A100 80GB (or equivalent for DeepSeek V3)
- **vLLM**: Installed and working
- **Python**: 3.10+
- **Disk space**: ~60 GB (model + logit captures)

---

## Step 1: Deploy Patches (5 min)

```bash
cd /nvmedata/chenw/vllm-ra/examples/dsa_indexer

# Sync instrumented code to vLLM installation
./sync_to_env.sh

# Verify
python3 -c "from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE; print('✓ OK')"
```

---

## Step 2: Test Instrumentation (5 min)

```bash
# Run smoke test
python3 test_deepseek_instrumentation.py

# Expected output:
#   ✓ PASS    Import
#   ✓ PASS    Config
#   ✓ PASS    Recording
#   ✓ PASS    Model Import
```

---

## Step 3: Configure Experiment (10 min)

Edit `configs/deepseek_topk_experiment.sh` if needed:

```bash
# Model (default: DeepSeek-V3)
export DEEPSEEK_MODEL="deepseek-ai/DeepSeek-V3"

# Hardware (adjust to your GPU count)
export TP_SIZE=4              # Tensor parallel size
export GPU_MEM=0.90           # GPU memory utilization

# Workload (300 prompts = 100 per benchmark)
export PROMPTS_PER_DATASET=100
export MAX_TOKENS=2048
export TEMPERATURE=0.7
export CONCURRENCY=4
```

**Dry run** to verify configuration:

```bash
bash configs/deepseek_topk_experiment.sh --dry-run
```

---

## Step 4: Download Model (10 min, one-time)

```bash
# Download DeepSeek V3 checkpoint
huggingface-cli download deepseek-ai/DeepSeek-V3 --repo-type model

# Verify size
du -sh ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3
# Expected: ~40-50 GB for V3
```

---

## Step 5: Run Experiment (6-10 hours)

```bash
# Launch full experiment
bash configs/deepseek_topk_experiment.sh

# Monitor progress (in another terminal)
tail -f logs/eval_deepseek_v3_*.jsonl | jq -r '.source, .latency'

# OR watch logit captures
watch -n 60 'ls -lh indexer_logits/deepseek_v3_*/indexer_logits_rank*.npz'
```

The script will:
1. Launch vLLM server with DeepSeek V3
2. Run 300 prompts (100 each: AIME, LiveCodeBench, GPQA Diamond)
3. Capture expert routing scores to `indexer_logits/deepseek_v3_*/`
4. Shutdown server gracefully
5. Verify output files

**What's captured**:
- Post-softmax expert scores (probabilities in [0, 1])
- Per-layer histograms (prefill + decode phases)
- Radix boundary-bucket analysis for Top-K optimization

---

## Step 6: Analyze Results (1-2 hours)

### 6.1 Generate Visualizations

```bash
# Find your run directory
RUN_DIR=$(ls -td indexer_logits/deepseek_v3_* | head -1)

# Generate all plots
python3 tools/plot_indexer_logits.py \
    --inputs "$RUN_DIR"/*.npz \
    --out "figures/$(basename $RUN_DIR)" \
    --aggregate

# Output: 20+ plots in figures/deepseek_v3_*/
```

**Key plots**:
- `stream_b_panels_*.png` — Per-layer score distributions (Joseph's histograms)
- `stream_b_summary.png` — Mean ± std per layer
- `stream_d_cascade_rate.png` — Radix sort optimization analysis
- `summary.txt` — Text summary with key statistics

### 6.2 Threshold Analysis

```bash
# Answer: "Can we prune values < 0.5?"
python3 tools/threshold_analysis.py \
    "$RUN_DIR/indexer_logits_rank0.npz" \
    > "figures/$(basename $RUN_DIR)/threshold_recommendations.txt"

# View recommendations
cat "figures/$(basename $RUN_DIR)/threshold_recommendations.txt"
```

**Output**:
- P(score < threshold) table for each layer
- Recommended safe thresholds (< 1% loss)
- Answer to Joseph's pruning question

### 6.3 Review Summary

```bash
# Quick stats
cat "figures/$(basename $RUN_DIR)/summary.txt"

# Key sections:
#   - Prefill/decode row counts
#   - Per-layer p1 (1st percentile) of scores
#   - Cascade dial: one-pass radix success rate
```

---

## Step 7: Share Results (30 min)

### For Joseph Rogers (Stockholm)

**Deliverables**:
1. **Histograms**: `stream_b_panels_prefill.png`, `stream_b_panels_decode.png`
2. **Threshold table**: `threshold_recommendations.txt`
3. **Summary**: First few sections of `summary.txt`

**Email template**:

```
Subject: DeepSeek Top-K Distribution Analysis — Results

Hi Joseph,

Completed the logit distribution analysis for DeepSeek V3 across 300 prompts
(100 math, 100 code, 100 QA). Here are the key findings:

**Your Question: "Can we prune values < 0.5?"**

[Copy answer from threshold_recommendations.txt]

**Histograms**: Attached (per-layer post-softmax scores)
  - Prefill phase: [attach stream_b_panels_prefill.png]
  - Decode phase: [attach stream_b_panels_decode.png]

**Recommended Threshold**: X.XX (conservative, < 1% loss across all layers)

**Radix Sort Optimization** (for Chen's work):
  - Top-12 bits: 99.7% one-pass success
  - 16 KB per-row memory budget
  - See figures/deepseek_v3_*/stream_d_cascade_rate.png

Full results and plots: [link to shared directory or attach ZIP]

Let me know if you need any other analysis!

Best,
Chen
```

### For DSA Team

**Deliverables**:
1. **Cascade analysis**: `stream_d_cascade_rate.png`, `stream_d_summary.png`
2. **Bit-width recommendations**: Section from `summary.txt`
3. **Implementation guide**: `docs/DEEPSEEK_MODEL_ANALYSIS.md`

---

## Troubleshooting

### Server fails to start

```bash
# Check CUDA/GPU
nvidia-smi

# Check vLLM installation
python3 -c "import vllm; print(vllm.__version__)"

# Check server log
tail -100 logs/server_deepseek_v3_*.log
```

### No logit files created

```bash
# Verify environment
echo $INDEXER_LOGIT_DUMP_DIR
echo $INDEXER_IS_EXPERT_ROUTING

# Check if logger is enabled
python3 -c "
import sys; sys.path.insert(0, 'vllm')
from _indexer_logger import is_enabled
print(f'Enabled: {is_enabled()}')
"
```

### Evaluation hangs/fails

```bash
# Check server health
curl http://localhost:8000/health

# Test single prompt
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'
```

### Out of memory

```bash
# Reduce max_model_len
export MAX_MODEL_LEN=4096  # Down from 8192

# Reduce GPU memory utilization
export GPU_MEM=0.80  # Down from 0.90

# Reduce concurrency
export CONCURRENCY=2  # Down from 4
```

---

## Advanced Options

### Custom benchmark mix

```bash
# Use only math benchmarks
export PROMPTS_PER_DATASET=200
python3 tools/run_eval_prompts.py \
    --url http://localhost:8000 \
    --model DeepSeek-V3 \
    --datasets aime \
    --n-per-dataset 200 \
    --out logs/deepseek_math_only.jsonl
```

### Resume interrupted run

```bash
# run_eval_prompts.py supports --resume flag
python3 tools/run_eval_prompts.py \
    --url http://localhost:8000 \
    --model DeepSeek-V3 \
    --datasets aime livecodebench gpqa_diamond \
    --n-per-dataset 100 \
    --out logs/eval_deepseek_v3_20260601_120000.jsonl \
    --resume  # Skip already-completed prompts
```

### Compare DeepSeek V2 vs V3

```bash
# Run V2
export DEEPSEEK_MODEL="deepseek-ai/DeepSeek-V2"
bash configs/deepseek_topk_experiment.sh

# Run V3
export DEEPSEEK_MODEL="deepseek-ai/DeepSeek-V3"
bash configs/deepseek_topk_experiment.sh

# Compare results
python3 tools/plot_indexer_logits.py \
    --inputs indexer_logits/deepseek_v2_*/indexer_logits_rank0.npz \
    --out figures/v2_vs_v3/v2
python3 tools/plot_indexer_logits.py \
    --inputs indexer_logits/deepseek_v3_*/indexer_logits_rank0.npz \
    --out figures/v2_vs_v3/v3
```

---

## Timeline Reference

Based on typical A100 80GB × 4 setup:

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup + patches | 30 min | One-time |
| Model download | 10 min | One-time, cached |
| Server startup | 2-5 min | Per run |
| Evaluation (300 prompts) | 6-10 hours | ~1-2 min/prompt |
| Server shutdown | 10 sec | Per run |
| Plot generation | 10 min | Per run |
| Threshold analysis | 1 min | Per run |
| **Total (first run)** | **7-11 hours** | Including setup |
| **Total (subsequent runs)** | **6-10 hours** | Setup cached |

---

## Next Steps After Results

1. **Share with Joseph**: Email histograms + threshold recommendations
2. **Write full report**: Use template in `docs/DEEPSEEK_TOPK_EXPERIMENT_RESULTS.md`
3. **Validate thresholds**: Run accuracy benchmarks with pruning enabled
4. **Optimize RDU kernel**: Implement recommended bit-width (top-12 bits)
5. **Extend analysis**: Compare across different model sizes or configurations

---

## Questions?

- **Instrumentation issues**: Check `docs/PHASE2_INSTRUMENTATION_COMPLETE.md`
- **Model architecture**: See `docs/DEEPSEEK_MODEL_ANALYSIS.md`
- **Full plan**: Read `docs/DEEPSEEK_TOPK_ANALYSIS_PLAN.md`
- **GLM-5.1 reference**: See `INDEXER_LOGIT_EXPERIMENT.md`
