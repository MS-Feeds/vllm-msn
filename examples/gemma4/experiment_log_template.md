# Gemma 4 26B MoE Experiment Log

## Experiment Tracking Template

Use this template to document experiments with different configurations, attention backends, and optimizations.

---

## Experiment Index

| ID | Date | Configuration | Attention Backend | Key Finding | Status |
|----|------|---------------|-------------------|-------------|---------|
| E001 | YYYY-MM-DD | FP8 + MTP + FlashInfer | FlashInfer | TBD | Pending |
| E002 | YYYY-MM-DD | FP8 + MTP + Flash-Attn | Flash Attn 2 | TBD | Pending |
| E003 | YYYY-MM-DD | FP8 (no MTP) + FlashInfer | FlashInfer | TBD | Pending |

---

# Experiment E001: [Brief Description]

## Metadata

- **Date**: YYYY-MM-DD
- **Experiment ID**: E001
- **Operator**: [Your Name]
- **Objective**: [What are you testing? e.g., "Compare FlashInfer vs Flash Attention 2 performance"]
- **Status**: [ ] Pending / [ ] Running / [ ] Completed / [ ] Failed

## Hardware Configuration

- **GPU**: A100 40GB (CUDA Compute Capability 8.0)
- **GPU Count**: 1
- **CUDA Version**: 12.6
- **Driver Version**: [Check with `nvidia-smi`]
- **CPU**: [Optional]
- **RAM**: [Optional]
- **Storage**: NVMe SSD

## Software Environment

- **vLLM Version**: dev (commit: [git rev-parse HEAD])
- **PyTorch Version**: 2.7.1
- **Python Version**: 3.10
- **Conda Environment**: vllm
- **Flash Attention Version**: [2.8.3 / Not installed]
- **FlashInfer Version**: [0.2.12 / Not installed]
- **xformers Version**: [0.0.31 / Not installed]

## Model Configuration

- **Model**: google/gemma-4-26B-A4B-it
- **Model Type**: Gemma 4 26B MoE (128 experts, top-8 routing)
- **Model Path**: `/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it`
- **Assistant Model**: google/gemma-4-26B-A4B-it-assistant (for MTP)
- **Assistant Path**: `/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant`
- **Quantization**: fp8 (weights + KV cache)
- **KV Cache dtype**: fp8_e5m2

## Inference Configuration

### Engine Settings

```python
# From llm_analyzer_gemma4_moe_fp8_mtp.py
tensor_parallel_size = 1
dtype = "bfloat16"
quantization = "fp8"
kv_cache_dtype = "fp8_e5m2"
gpu_memory_utilization = 0.75
max_num_batched_tokens = 6144
max_num_seqs = 128
max_model_len = 8192
enforce_eager = False  # CUDA graphs enabled
enable_prefix_caching = True
```

### MTP (Speculative Decoding) Settings

```python
# Multi-Token Prediction
speculative_model = "/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant"
num_speculative_tokens = 5
```

### Attention Backend Configuration

```bash
# Environment variables
export VLLM_ATTENTION_BACKEND=FLASHINFER  # or FLASH_ATTN
export VLLM_USE_FLASHINFER_MOE_FP8=1      # For FlashInfer
export VLLM_MOE_BACKEND=auto
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # For Flash Attention
export TORCH_CUDA_ARCH_LIST="8.0"
```

**Selected Backend**: [FlashInfer / Flash Attention 2 / xformers / auto]

### Workload Configuration

```python
# From shell script
input_path = "/nvmedata/chenw/genz/genz_users_20k_format.tsv"
output_path = "/nvmedata/chenw/genz/genz_users_interests_gemma4_fp8_mtp.jsonl"
batch_size = 128
```

- **Dataset**: GenZ users (20k samples)
- **Task**: User interest analysis (JSON generation)
- **Average sequence length**: [Estimate, e.g., 500-800 tokens]
- **Max sequence length**: 8192 tokens

## Commands

### Pre-Experiment Checks

```bash
# Check GPU availability
nvidia-smi

# Check free memory
nvidia-smi --query-gpu=memory.free --format=csv

# Verify model files exist
ls -lh /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it/
ls -lh /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant/

# Check attention backend
conda activate vllm
python -c "import flash_attn; print(f'Flash Attn: {flash_attn.__version__}')" 2>/dev/null || echo "Not installed"
python -c "import flashinfer; print(f'FlashInfer: {flashinfer.__version__}')" 2>/dev/null || echo "Not installed"
```

### Run Experiment

```bash
# Activate environment
conda activate vllm
cd /nvmedata/chenw/vllm-ra/examples

# Start memory monitoring (in separate terminal)
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader'

# Run experiment
time ./vllm_gemma4_moe_fp8_mtp.sh 2>&1 | tee experiment_e001.log

# Or with explicit backend:
# export VLLM_ATTENTION_BACKEND=FLASHINFER
# time python3 llm_analyzer_gemma4_moe_fp8_mtp.py \
#   --input_path /nvmedata/chenw/genz/genz_users_20k_format.tsv \
#   --output_path /nvmedata/chenw/genz/genz_users_interests_e001.jsonl \
#   --model_path /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it \
#   --speculative_model /nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-assistant \
#   --num_speculative_tokens 5 \
#   --batch_size 128
```

### Memory Profiling (Optional)

```bash
# Detailed memory trace (run in background)
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv -l 1 > experiment_e001_memory.csv &

# Kill after experiment completes
# kill %1
```

## Results

### Performance Metrics

**Execution Time:**
- Start time: [HH:MM:SS]
- End time: [HH:MM:SS]
- **Total duration**: [X minutes Y seconds]

**Throughput:**
- Total samples processed: 20,000
- **Throughput**: [X.X samples/sec]
- **Throughput**: [X.X requests/sec]
- **Tokens processed**: [Estimate total tokens]
- **Token throughput**: [X.X tokens/sec]

**Latency (from logs if available):**
- **P50 latency**: [X.X ms]
- **P90 latency**: [X.X ms]
- **P95 latency**: [X.X ms]
- **P99 latency**: [X.X ms]
- **Max latency**: [X.X ms]

### Memory Usage

**Peak Memory:**
- Model loading: [X.X GB]
- During inference: [X.X GB]
- **Peak total**: [X.X GB]

**Memory Breakdown (estimated):**
```
Component               Memory
────────────────────────────────
Main model (FP8)        XX.X GB
Assistant model         XX.X GB
KV cache                XX.X GB
CUDA graphs             XX.X GB
Other overhead          XX.X GB
────────────────────────────────
Total                   XX.X GB / 40 GB (XX%)
```

**Memory Stability:**
- OOM errors: [Yes/No]
- Memory warnings: [Any warnings in logs?]
- Memory growth over time: [Stable / Growing / Fluctuating]

### GPU Utilization

- **Average GPU utilization**: [X%]
- **Peak GPU utilization**: [X%]
- **Average GPU temperature**: [X°C]
- **Peak GPU temperature**: [X°C]

### Model-Specific Metrics

**MTP (Speculative Decoding) Statistics:**
- Speculative tokens per iteration: 5
- **Acceptance rate**: [X.X% if logged]
- **Average tokens accepted**: [X.X if logged]
- **MTP speedup vs baseline**: [X.Xx if known]

**Attention Backend Statistics:**
- Backend used: [FlashInfer / Flash Attention 2 / xformers]
- Attention kernel time: [X.X ms if logged]
- Cache hits (prefix caching): [X if logged]

### Output Quality

- Output file: `/nvmedata/chenw/genz/genz_users_interests_e001.jsonl`
- Output file size: [X.X MB]
- **Samples completed**: [X / 20000]
- **Success rate**: [XX.X%]
- Average response length: [X tokens]

**Sample Output Check:**
```bash
# Check first few outputs
head -n 3 /nvmedata/chenw/genz/genz_users_interests_e001.jsonl

# Check for errors
grep -i "error\|fail\|exception" experiment_e001.log | head -10

# Count valid JSON outputs
cat /nvmedata/chenw/genz/genz_users_interests_e001.jsonl | wc -l
```

## Observations

### What Went Well
- [List positive observations]
- Example: "Model loaded without OOM"
- Example: "Consistent memory usage throughout run"
- Example: "No CUDA errors"

### Issues Encountered
- [List any problems]
- Example: "Initial warmup took 2 minutes"
- Example: "Occasional CUDA graph recompilation warnings"
- Example: "High P99 latency variance"

### Unexpected Findings
- [List surprises]
- Example: "Memory usage lower than expected (32GB vs 34GB)"
- Example: "MTP acceptance rate higher than anticipated"

### Bottlenecks Identified
- [What's limiting performance?]
- [ ] GPU compute
- [ ] Memory bandwidth
- [ ] KV cache size
- [ ] Batch scheduling
- [ ] MoE routing overhead
- [ ] Other: [describe]

## Log Excerpts

### Model Loading

```
[Paste relevant log lines showing model loading]
```

### Inference Stats

```
[Paste any throughput/latency stats from logs]
```

### Errors/Warnings

```
[Paste any errors or warnings]
```

## Comparison with Previous Experiments

| Metric | E001 (Current) | E002 (Previous) | Δ | Notes |
|--------|----------------|-----------------|---|-------|
| Duration | X min | Y min | ±Z% | [Better/Worse] |
| Throughput | X req/s | Y req/s | ±Z% | [Better/Worse] |
| P99 Latency | X ms | Y ms | ±Z% | [Better/Worse] |
| Peak Memory | X GB | Y GB | ±Z GB | [Better/Worse] |
| GPU Util | X% | Y% | ±Z% | [Better/Worse] |

## Analysis

### Performance Summary

[Write 2-3 sentences summarizing performance]

Example:
> The experiment achieved 85 requests/sec with FlashInfer backend, using 35GB peak memory. P99 latency was stable at 180ms, indicating good predictability for online serving. No OOM errors occurred during the 4-hour run.

### Backend Comparison (if applicable)

[Compare FlashInfer vs Flash Attention 2 if testing both]

| Aspect | FlashInfer | Flash Attn 2 | Winner |
|--------|-----------|--------------|---------|
| Throughput | X req/s | Y req/s | [Backend] |
| P99 Latency | X ms | Y ms | [Backend] |
| Memory | X GB | Y GB | [Backend] |
| Stability | [Rating] | [Rating] | [Backend] |

### Recommendations

Based on this experiment:
- [ ] Use this configuration for production
- [ ] Test with different batch size: [X]
- [ ] Test with different backend: [Backend]
- [ ] Optimize memory utilization: [suggestion]
- [ ] Investigate: [specific issue]

## Next Steps

1. [What to test next?]
   - Example: "Test with batch_size=64 to reduce memory"
   - Example: "Compare with MTP disabled"
   - Example: "Try Flash Attention 2 backend"

2. [Configuration changes to try]
   - Example: "Increase gpu_memory_utilization to 0.80"
   - Example: "Reduce max_num_seqs to 96"

3. [Open questions]
   - Example: "Why is P99 latency higher than expected?"
   - Example: "Can we increase batch size without OOM?"

## Files Generated

- **Log file**: `experiment_e001.log`
- **Output file**: `/nvmedata/chenw/genz/genz_users_interests_e001.jsonl`
- **Memory trace**: `experiment_e001_memory.csv`
- **Screenshots**: [Any nvidia-smi screenshots]

## References

- Configuration file: `vllm_gemma4_moe_fp8_mtp.sh`
- Python script: `llm_analyzer_gemma4_moe_fp8_mtp.py`
- Analysis doc: `ATTENTION_BACKEND_ANALYSIS.md`
- Memory guide: `README_MTP_MEMORY.md`

---

## Sign-off

**Completed by**: [Your Name]
**Date**: YYYY-MM-DD
**Approved for**: [ ] Production / [ ] Further Testing
**Confidence**: [ ] High / [ ] Medium / [ ] Low

