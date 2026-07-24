# Experiment E011 - Results

## Configuration
Experiment ID: E011
Date: 2026-05-21 03:39:47
Status: Success

Model: gemma-4-26B-A4B-it-text-only
Backend: FLASHINFER
Batch Size: 128
Max Batched Tokens: 32768
Max Model Len: 32768
Max Tokens/Request: 1024
FP8: true
CUDA Graphs: true
MTP: true
GPU Memory Util: 0.80
KV Cache Dtype: auto

## Performance
Duration: 887 seconds (14 minutes)

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
- Output tokens:         806,894
- Cached tokens:         757,600
- Prefix-cache hit rate: 13.57%
- Finish reasons:        {'length': 576, 'stop': 393}

### Throughput
- QPS:                1.1814 req/s
- Output tokens/sec:  983.72
- Prompt tokens/sec:  6806.12
- Total tokens/sec:   7789.85

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 402736.64 | 401721.45 | 732153.30 | 770694.97 | 787735.29 | 793956.12 |
| TPOT | n=969 | 90.22 | 86.31 | 111.75 | 149.46 | 330.66 | 591.10 |
| E2E | n=969 | 476695.99 | 499143.06 | 803651.37 | 814916.57 | 818390.15 | 820205.74 |
| TTFT (client) | n=969 | 407417.37 | 406404.37 | 740319.40 | 779264.78 | 796478.01 | 802767.02 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               70.61 GiB  (89.1% of total)
- Average used:            67.23 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 70.61 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=93.4%
- GPU memory-bw util:      peak=54%  avg=25.9%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E011/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E011/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E011/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E011/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E011/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E011/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E011/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E011/gpu_final.txt

## Errors
✓ No errors detected
