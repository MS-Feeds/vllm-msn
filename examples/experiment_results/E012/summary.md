# Experiment E012 - Results

## Configuration
Experiment ID: E012
Date: 2026-05-21 03:55:19
Status: Success

Model: gemma-4-26B-A4B-it-text-only
Backend: FLASH_ATTN
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
Duration: 894 seconds (14 minutes)

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
- Output tokens:         805,638
- Cached tokens:         752,800
- Prefix-cache hit rate: 13.48%
- Finish reasons:        {'length': 581, 'stop': 388}

### Throughput
- QPS:                1.1710 req/s
- Output tokens/sec:  973.58
- Prompt tokens/sec:  6746.44
- Total tokens/sec:   7720.02

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 407625.23 | 404175.91 | 744497.29 | 777275.68 | 800737.73 | 804674.79 |
| TPOT | n=969 | 77.08 | 75.14 | 92.68 | 116.19 | 215.59 | 578.83 |
| E2E | n=969 | 472470.05 | 491669.68 | 803522.43 | 818910.22 | 825402.87 | 827460.04 |
| TTFT (client) | n=969 | 411927.06 | 408435.57 | 752152.34 | 785283.74 | 808956.95 | 812956.08 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               63.28 GiB  (79.8% of total)
- Average used:            60.84 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 63.28 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=93.6%
- GPU memory-bw util:      peak=54%  avg=25.9%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E012/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E012/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E012/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E012/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E012/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E012/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E012/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E012/gpu_final.txt

## Errors
✓ No errors detected
