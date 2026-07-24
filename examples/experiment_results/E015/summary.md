# Experiment E015 - Results

## Configuration
Experiment ID: E015
Date: 2026-05-21 06:56:04
Status: Success

Model: gemma-4-26B-A4B-it-text-only
Backend: FLASHINFER
Batch Size: 32
Max Batched Tokens: 32768
Max Model Len: 32768
Max Tokens/Request: 1024
FP8: false
CUDA Graphs: false
MTP: false
GPU Memory Util: 0.95
KV Cache Dtype: auto

## Performance
Duration: 1741 seconds (29 minutes)

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
- Output tokens:         809,840
- Cached tokens:         836,352
- Prefix-cache hit rate: 14.98%
- Finish reasons:        {'length': 596, 'stop': 373}

### Throughput
- QPS:                0.5708 req/s
- Output tokens/sec:  477.06
- Prompt tokens/sec:  3288.64
- Total tokens/sec:   3765.70

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 818405.15 | 813256.21 | 1492273.85 | 1571502.18 | 1628905.26 | 1639448.27 |
| TPOT | n=969 | 64.85 | 64.07 | 73.98 | 78.27 | 87.55 | 134.31 |
| E2E | n=969 | 876882.95 | 871766.18 | 1557020.19 | 1623527.46 | 1682069.31 | 1697527.14 |
| TTFT (client) | n=969 | 822700.61 | 817506.37 | 1499934.50 | 1579527.40 | 1637132.80 | 1647745.54 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               77.22 GiB  (97.4% of total)
- Average used:            76.17 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 77.22 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=89.1%
- GPU memory-bw util:      peak=63%  avg=43.6%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E015/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E015/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E015/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E015/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E015/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E015/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E015/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E015/gpu_final.txt

## Errors
✓ No errors detected
