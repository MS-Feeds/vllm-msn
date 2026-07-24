# Experiment E005 - Results

## Configuration
Experiment ID: E005
Date: 2026-05-21 01:16:35
Status: Success

Model: gemma-4-26B-A4B-it
Backend: FLASHINFER
Batch Size: 128
Max Batched Tokens: 32768
Max Model Len: 32768
Max Tokens/Request: 1024
FP8: true
CUDA Graphs: true
MTP: false
GPU Memory Util: 0.75
KV Cache Dtype: auto

## Performance
Duration: 1208 seconds (20 minutes)

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
- Output tokens:         803,319
- Cached tokens:         807,840
- Prefix-cache hit rate: 14.47%
- Finish reasons:        {'length': 571, 'stop': 398}

### Throughput
- QPS:                0.9269 req/s
- Output tokens/sec:  768.43
- Prompt tokens/sec:  5340.24
- Total tokens/sec:   6108.68

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 503104.48 | 503579.41 | 924125.85 | 962255.93 | 1002894.44 | 1004990.96 |
| TPOT | n=969 | 98.72 | 97.20 | 109.32 | 122.81 | 249.61 | 558.91 |
| E2E | n=969 | 586638.62 | 600350.38 | 1006002.67 | 1028687.86 | 1041761.22 | 1045358.61 |
| TTFT (client) | n=969 | 507671.46 | 508104.19 | 932269.13 | 970788.67 | 1011635.42 | 1013782.31 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               61.19 GiB  (77.2% of total)
- Average used:            56.94 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 61.19 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=87.9%
- GPU memory-bw util:      peak=94%  avg=34.9%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E005/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E005/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E005/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E005/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E005/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E005/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E005/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E005/gpu_final.txt

## Errors
✓ No errors detected
