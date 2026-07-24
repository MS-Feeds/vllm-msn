# Experiment E003 - Results

## Configuration
Experiment ID: E003
Date: 2026-05-21 00:36:47
Status: Success

Model: gemma-4-26B-A4B-it
Backend: FLASHINFER
Batch Size: 64
Max Batched Tokens: 32768
Max Model Len: 32768
Max Tokens/Request: 1024
FP8: true
CUDA Graphs: false
MTP: false
GPU Memory Util: 0.85
KV Cache Dtype: auto

## Performance
Duration: 1234 seconds (20 minutes)

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
- Output tokens:         805,689
- Cached tokens:         836,352
- Prefix-cache hit rate: 14.98%
- Finish reasons:        {'length': 570, 'stop': 399}

### Throughput
- QPS:                0.8377 req/s
- Output tokens/sec:  696.48
- Prompt tokens/sec:  4825.95
- Total tokens/sec:   5522.43

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 549385.90 | 544669.35 | 1005026.57 | 1057026.19 | 1090266.70 | 1100724.43 |
| TPOT | n=969 | 87.73 | 87.05 | 102.51 | 108.47 | 157.59 | 357.77 |
| E2E | n=969 | 625873.55 | 648429.82 | 1080125.84 | 1116135.81 | 1146353.33 | 1156766.32 |
| TTFT (client) | n=969 | 553692.19 | 548929.24 | 1012705.17 | 1065078.62 | 1098516.67 | 1109044.32 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               68.87 GiB  (86.9% of total)
- Average used:            66.28 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 68.87 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=92.6%
- GPU memory-bw util:      peak=60%  avg=36.4%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E003/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E003/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E003/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E003/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E003/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E003/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E003/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E003/gpu_final.txt

## Errors
✓ No errors detected
