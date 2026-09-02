# Experiment E002 - Results

## Configuration
Experiment ID: E002
Date: 2026-05-21 00:15:35
Status: Success

Model: gemma-4-26B-A4B-it
Backend: FLASH_ATTN
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
Duration: 1249 seconds (20 minutes)

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
- Output tokens:         806,659
- Cached tokens:         836,352
- Prefix-cache hit rate: 14.98%
- Finish reasons:        {'length': 582, 'stop': 387}

### Throughput
- QPS:                0.8314 req/s
- Output tokens/sec:  692.13
- Prompt tokens/sec:  4790.09
- Total tokens/sec:   5482.22

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 553112.28 | 548591.53 | 1012511.03 | 1063312.34 | 1098358.44 | 1109262.16 |
| TPOT | n=969 | 88.28 | 87.31 | 102.74 | 108.78 | 164.72 | 358.88 |
| E2E | n=969 | 630125.83 | 653083.34 | 1088157.41 | 1123799.43 | 1154673.27 | 1165425.42 |
| TTFT (client) | n=969 | 557424.69 | 552875.39 | 1020198.62 | 1071367.40 | 1106617.55 | 1117591.54 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               68.86 GiB  (86.9% of total)
- Average used:            66.08 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 68.86 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=92.1%
- GPU memory-bw util:      peak=61%  avg=36.1%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E002/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E002/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E002/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E002/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E002/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E002/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E002/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E002/gpu_final.txt

## Errors
✓ No errors detected
