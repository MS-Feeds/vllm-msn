# Experiment Quickstart Guide

Quick guide to running and documenting experiments with Gemma 4 MoE.

## Quick Run

### Option 1: Automated Experiment Runner

```bash
cd /nvmedata/chenw/vllm-ra/examples

# Run with FlashInfer (recommended)
./experiment_runner.sh E001 FLASHINFER 128

# Run with Flash Attention 2
./experiment_runner.sh E002 FLASH_ATTN 128

# Run with different batch size
./experiment_runner.sh E003 FLASHINFER 64
```

Results will be saved to `./experiments/E001/`, `./experiments/E002/`, etc.

### Option 2: Manual Run

```bash
cd /nvmedata/chenw/vllm-ra/examples

# Start memory monitor (terminal 1)
watch -n 1 nvidia-smi

# Run experiment (terminal 2)
conda activate vllm
export VLLM_ATTENTION_BACKEND=FLASHINFER
time ./vllm_gemma4_moe_fp8_mtp.sh 2>&1 | tee experiment_e001.log
```

## Common Experiment Scenarios

### Scenario 1: Compare FlashInfer vs Flash Attention 2

```bash
# Test 1: FlashInfer
./experiment_runner.sh E001 FLASHINFER 128

# Test 2: Flash Attention 2
./experiment_runner.sh E002 FLASH_ATTN 128

# Compare results
diff experiments/E001/summary.txt experiments/E002/summary.txt
```

### Scenario 2: Test MTP Impact

```bash
# With MTP (speculative decoding)
./experiment_runner.sh E003 FLASHINFER 128
# Uses: llm_analyzer_gemma4_moe_fp8_mtp.py

# Without MTP
export VLLM_ATTENTION_BACKEND=FLASHINFER
time python3 llm_analyzer_gemma4_moe_fp8.py \
  --input_path /nvmedata/chenw/genz/genz_users_20k_format.tsv \
  --output_path experiments/E004/output.jsonl \
  --batch_size 128 \
  2>&1 | tee experiments/E004/inference.log
```

### Scenario 3: Memory Optimization

```bash
# Baseline (gpu_memory_utilization=0.75)
./experiment_runner.sh E005 FLASHINFER 128

# Higher utilization (modify script: 0.75 → 0.80)
# Edit llm_analyzer_gemma4_moe_fp8_mtp.py first
./experiment_runner.sh E006 FLASHINFER 128

# Compare peak memory usage
grep "Peak memory" experiments/E00*/summary.txt
```

### Scenario 4: Batch Size Sweep

```bash
# Test different batch sizes
for BATCH in 32 64 128 256; do
  ./experiment_runner.sh E00${BATCH} FLASHINFER ${BATCH}
done

# Compare throughput
grep "Throughput" experiments/E00*/summary.txt
```

## Analyzing Results

### Quick Stats

```bash
# Throughput comparison
grep "Throughput" experiments/*/summary.txt

# Memory comparison
grep "Peak memory" experiments/*/summary.txt

# Duration comparison
grep "Duration" experiments/*/summary.txt

# Error check
grep -i "error" experiments/*/inference.log | wc -l
```

### Memory Analysis

```bash
# Plot memory usage over time (requires Python)
python3 << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('experiments/E001/memory_trace.csv')
df['memory_gb'] = df[' memory.used [MiB]'] / 1024

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['memory_gb'])
plt.xlabel('Time (seconds)')
plt.ylabel('Memory Usage (GB)')
plt.title('GPU Memory Usage - Experiment E001')
plt.grid(True)
plt.savefig('experiments/E001/memory_plot.png')
print("Plot saved to experiments/E001/memory_plot.png")
EOF
```

### Detailed Comparison

```bash
# Create comparison table
cat > compare_experiments.sh << 'SCRIPT'
#!/bin/bash
echo "Experiment Comparison"
echo "===================="
echo ""
printf "%-10s %-15s %-15s %-15s %-15s\n" "Exp ID" "Backend" "Duration (s)" "Throughput" "Peak Mem (MB)"
echo "------------------------------------------------------------------------"
for dir in experiments/E*/; do
  if [ -f "${dir}/summary.txt" ]; then
    EXP=$(basename "$dir")
    BACKEND=$(grep "Backend:" "${dir}/summary.txt" | cut -d' ' -f2)
    DURATION=$(grep "Duration:" "${dir}/summary.txt" | head -1 | awk '{print $2}')
    THROUGHPUT=$(grep "Throughput:" "${dir}/summary.txt" | awk '{print $2}')
    MEMORY=$(grep "Peak memory:" "${dir}/summary.txt" | awk '{print $3}')
    printf "%-10s %-15s %-15s %-15s %-15s\n" "$EXP" "$BACKEND" "$DURATION" "$THROUGHPUT" "$MEMORY"
  fi
done
SCRIPT
chmod +x compare_experiments.sh
./compare_experiments.sh
```

## Documentation Workflow

### After Each Experiment:

1. **Review automated summary**:
   ```bash
   cat experiments/E001/summary.txt
   ```

2. **Copy experiment template**:
   ```bash
   cp experiment_log_template.md experiments/E001/EXPERIMENT_LOG.md
   ```

3. **Fill in template** with results from `summary.txt` and observations

4. **Add to experiment index**:
   ```bash
   echo "| E001 | $(date +%Y-%m-%d) | FP8+MTP+FlashInfer | FlashInfer | 85 req/s, 35GB | Completed |" >> EXPERIMENT_INDEX.md
   ```

## Common Issues & Fixes

### Issue: OOM During Experiment

**Solution 1**: Reduce batch size
```bash
./experiment_runner.sh E001 FLASHINFER 64  # Instead of 128
```

**Solution 2**: Reduce gpu_memory_utilization
```python
# In llm_analyzer_gemma4_moe_fp8_mtp.py
gpu_memory_utilization=0.70  # Instead of 0.75
```

### Issue: Low Throughput

**Check 1**: GPU utilization
```bash
# Should be >70%
grep "utilization.gpu" experiments/E001/memory_trace.csv
```

**Check 2**: Batch size too small
```bash
# Try larger batch
./experiment_runner.sh E001 FLASHINFER 256
```

**Check 3**: Wrong backend
```bash
# Verify backend in use
grep "attention" experiments/E001/inference.log
```

### Issue: High P99 Latency

**Possible causes**:
1. CUDA graph recompilation (check logs for "graph" mentions)
2. Wrong attention backend (try FlashInfer instead of Flash Attn)
3. Memory swapping (check if memory usage >95%)

## Experiment Checklist

Before running:
- [ ] GPU is free (check `nvidia-smi`)
- [ ] Environment activated (`conda activate vllm`)
- [ ] Model files exist
- [ ] Input data exists
- [ ] Sufficient disk space for output

During run:
- [ ] Monitor memory usage
- [ ] Watch for errors in logs
- [ ] Check GPU temperature (<85°C)

After run:
- [ ] Review summary.txt
- [ ] Check for errors in logs
- [ ] Verify output file generated
- [ ] Document findings
- [ ] Update experiment index

## Quick Reference

### File Locations

- **Scripts**: `/nvmedata/chenw/vllm-ra/examples/`
- **Models**: `/nvmedata/hf_checkpoints/`
- **Data**: `/nvmedata/chenw/genz/`
- **Results**: `/nvmedata/chenw/vllm-ra/examples/experiments/`

### Important Files

- `experiment_runner.sh` - Automated experiment runner
- `experiment_log_template.md` - Template for documenting experiments
- `ATTENTION_BACKEND_ANALYSIS.md` - Backend comparison analysis
- `README_MTP_MEMORY.md` - Memory configuration guide

### Useful Commands

```bash
# Check GPU
nvidia-smi

# Activate environment
conda activate vllm

# Run experiment
./experiment_runner.sh E001 FLASHINFER 128

# Compare experiments
./compare_experiments.sh

# Check logs
tail -f experiments/E001/inference.log

# Monitor memory
watch -n 1 'nvidia-smi | grep python'
```

## Next Steps

1. Run baseline experiment: `./experiment_runner.sh E001 FLASHINFER 128`
2. Review results: `cat experiments/E001/summary.txt`
3. Document findings: Fill in `experiments/E001/EXPERIMENT_LOG.md`
4. Compare configurations as needed
5. Update main documentation with findings
