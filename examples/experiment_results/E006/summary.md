# Experiment E006 - Results

## Configuration
Experiment ID: E006
Date: 2026-05-21 02:07:34
Status: Success

Model: gemma-4-26B-A4B-it
Backend: FLASHINFER
Batch Size: 128
Max Batched Tokens: 32768
Max Model Len: 32768
Max Tokens/Request: 1024
FP8: true
CUDA Graphs: true
MTP: true
GPU Memory Util: 0.75
KV Cache Dtype: auto

## Performance
Duration: 938 seconds (15 minutes)

### Dataset (post-filter)
- Input file:               /nvmedata/data/layer1_delta_1k_test.txt
- Chat template applied:    True
- Token-length threshold:   31,744 tokens (max_model_len - max_tokens)
- Total rows seen:          1000
- Loaded for inference:     969
- Skipped (bad format):     0
- Skipped (too long):       31

### Counts
- Total requests:        969
- Finished:              969
- Failed:                0
- Prompt tokens:         5,582,695
- Output tokens:         809,635
- Cached tokens:         751,200
- Prefix-cache hit rate: 13.46%
- Finish reasons:        {'length': 589, 'stop': 380}

### Throughput
- QPS:                1.1664 req/s
- Output tokens/sec:  974.59
- Prompt tokens/sec:  6720.14
- Total tokens/sec:   7694.73

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 409392.11 | 407799.23 | 742449.77 | 782105.79 | 800614.20 | 805180.56 |
| TPOT | n=969 | 72.87 | 70.31 | 90.95 | 116.50 | 209.74 | 525.87 |
| E2E | n=969 | 470889.63 | 481422.78 | 803493.89 | 821515.41 | 828656.39 | 830699.26 |
| TTFT (client) | n=969 | 413891.31 | 412249.04 | 750473.82 | 790524.58 | 809211.12 | 813844.36 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               62.88 GiB  (79.3% of total)
- Average used:            58.88 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 62.88 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=90.4%
- GPU memory-bw util:      peak=54%  avg=25.3%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E006/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E006/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E006/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E006/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E006/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E006/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E006/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E006/gpu_final.txt

## Errors
✓ No errors detected
