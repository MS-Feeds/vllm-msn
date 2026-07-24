# Experiment E008 - Results

## Configuration
Experiment ID: E008
Date: 2026-05-21 02:52:56
Status: Success

Model: gemma-4-26B-A4B-it-text-only
Backend: FLASHINFER
Batch Size: 192
Max Batched Tokens: 32768
Max Model Len: 32768
Max Tokens/Request: 1024
FP8: true
CUDA Graphs: true
MTP: true
GPU Memory Util: 0.75
KV Cache Dtype: auto

## Performance
Duration: 899 seconds (14 minutes)

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
- Output tokens:         805,033
- Cached tokens:         752,096
- Prefix-cache hit rate: 13.47%
- Finish reasons:        {'length': 576, 'stop': 393}

### Throughput
- QPS:                1.1657 req/s
- Output tokens/sec:  968.44
- Prompt tokens/sec:  6715.91
- Total tokens/sec:   7684.35

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 409375.98 | 409324.74 | 744768.52 | 778457.60 | 804274.93 | 806582.77 |
| TPOT | n=969 | 76.23 | 75.07 | 92.52 | 107.75 | 208.71 | 664.64 |
| E2E | n=969 | 474280.32 | 488031.49 | 805993.16 | 821180.09 | 828224.17 | 831224.01 |
| TTFT (client) | n=969 | 413693.22 | 413599.39 | 752448.33 | 786529.71 | 812534.54 | 814891.23 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               63.51 GiB  (80.1% of total)
- Average used:            61.02 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 63.51 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=93.3%
- GPU memory-bw util:      peak=53%  avg=25.9%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E008/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E008/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E008/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E008/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E008/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E008/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E008/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E008/gpu_final.txt

## Errors
✓ No errors detected
