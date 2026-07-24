# Experiment E004 - Results

## Configuration
Experiment ID: E004
Date: 2026-05-21 00:55:49
Status: Success

Model: gemma-4-26B-A4B-it
Backend: FLASHINFER
Batch Size: 128
Max Batched Tokens: 32768
Max Model Len: 32768
Max Tokens/Request: 1024
FP8: true
CUDA Graphs: false
MTP: false
GPU Memory Util: 0.85
KV Cache Dtype: auto

## Performance
Duration: 1104 seconds (18 minutes)

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
- Output tokens:         806,917
- Cached tokens:         813,024
- Prefix-cache hit rate: 14.56%
- Finish reasons:        {'length': 581, 'stop': 388}

### Throughput
- QPS:                0.9476 req/s
- Output tokens/sec:  789.09
- Prompt tokens/sec:  5459.34
- Total tokens/sec:   6248.43

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 479659.83 | 454753.41 | 870683.47 | 939347.45 | 959080.44 | 961101.67 |
| TPOT | n=969 | 128.70 | 126.16 | 145.12 | 192.72 | 325.05 | 1495.85 |
| E2E | n=969 | 585338.08 | 594428.81 | 978577.34 | 1006559.70 | 1020703.93 | 1022550.37 |
| TTFT (client) | n=969 | 484178.62 | 459231.75 | 878637.17 | 947740.06 | 967651.58 | 969736.18 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               68.64 GiB  (86.6% of total)
- Average used:            65.63 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 68.63 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=93.1%
- GPU memory-bw util:      peak=66%  avg=36.4%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E004/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E004/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E004/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E004/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E004/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E004/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E004/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E004/gpu_final.txt

## Errors
✓ No errors detected
