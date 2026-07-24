# Experiment E009 - Results

## Configuration
Experiment ID: E009
Date: 2026-05-21 03:08:31
Status: Success

Model: gemma-4-26B-A4B-it-text-only
Backend: FLASHINFER
Batch Size: 256
Max Batched Tokens: 32768
Max Model Len: 32768
Max Tokens/Request: 1024
FP8: true
CUDA Graphs: true
MTP: true
GPU Memory Util: 0.75
KV Cache Dtype: auto

## Performance
Duration: 897 seconds (14 minutes)

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
- Output tokens:         806,020
- Cached tokens:         749,600
- Prefix-cache hit rate: 13.43%
- Finish reasons:        {'length': 578, 'stop': 391}

### Throughput
- QPS:                1.1672 req/s
- Output tokens/sec:  970.87
- Prompt tokens/sec:  6724.49
- Total tokens/sec:   7695.36

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 409401.98 | 406852.25 | 745990.75 | 780789.57 | 803099.02 | 807224.19 |
| TPOT | n=969 | 76.44 | 74.84 | 92.42 | 106.89 | 196.77 | 582.90 |
| E2E | n=969 | 474476.70 | 488836.36 | 806492.65 | 821023.93 | 827781.61 | 830162.88 |
| TTFT (client) | n=969 | 413698.56 | 411108.54 | 753638.13 | 788825.10 | 811310.50 | 815509.53 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               62.59 GiB  (79.0% of total)
- Average used:            60.14 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 62.59 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=93.8%
- GPU memory-bw util:      peak=53%  avg=26.0%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E009/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E009/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E009/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E009/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E009/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E009/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E009/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E009/gpu_final.txt

## Errors
✓ No errors detected
