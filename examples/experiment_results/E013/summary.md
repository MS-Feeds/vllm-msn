# Experiment E013 - Results

## Configuration
Experiment ID: E013
Date: 2026-05-21 04:15:12
Status: Success

Model: gemma-4-26B-A4B-it-text-only
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
Duration: 1155 seconds (19 minutes)

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
- Output tokens:         804,322
- Cached tokens:         807,840
- Prefix-cache hit rate: 14.47%
- Finish reasons:        {'length': 587, 'stop': 382}

### Throughput
- QPS:                0.9414 req/s
- Output tokens/sec:  781.39
- Prompt tokens/sec:  5423.55
- Total tokens/sec:   6204.94

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 493297.05 | 486297.30 | 914196.12 | 940405.97 | 986336.31 | 989932.31 |
| TPOT | n=969 | 107.99 | 103.80 | 124.96 | 162.81 | 326.21 | 889.55 |
| E2E | n=969 | 581600.07 | 599668.13 | 995909.96 | 1014547.99 | 1025988.70 | 1029302.30 |
| TTFT (client) | n=969 | 497767.46 | 490762.17 | 922090.19 | 948711.09 | 994848.50 | 998521.00 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               61.71 GiB  (77.9% of total)
- Average used:            58.24 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 61.71 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=89.5%
- GPU memory-bw util:      peak=86%  avg=35.9%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E013/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E013/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E013/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E013/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E013/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E013/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E013/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E013/gpu_final.txt

## Errors
✓ No errors detected
