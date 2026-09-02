# Experiment E001 - Results

## Configuration
Experiment ID: E001
Date: 2026-05-20 23:54:08
Status: Success

Model: gemma-4-26B-A4B-it
Backend: FLASH_ATTN
Batch Size: 64
Max Batched Tokens: 32768
Max Model Len: 32768
Max Tokens/Request: 1024
FP8: false
CUDA Graphs: false
MTP: false
GPU Memory Util: 0.95
KV Cache Dtype: auto

## Performance
Duration: 1313 seconds (21 minutes)

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
- Output tokens:         808,654
- Cached tokens:         813,248
- Prefix-cache hit rate: 14.57%
- Finish reasons:        {'length': 599, 'stop': 370}

### Throughput
- QPS:                0.7848 req/s
- Output tokens/sec:  654.91
- Prompt tokens/sec:  4521.28
- Total tokens/sec:   5176.19

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 589811.75 | 577503.57 | 1085241.80 | 1137355.63 | 1167069.12 | 1177159.86 |
| TPOT | n=969 | 82.76 | 81.12 | 95.57 | 103.29 | 137.42 | 315.98 |
| E2E | n=969 | 661856.26 | 658582.92 | 1146099.85 | 1191589.62 | 1228313.00 | 1234716.44 |
| TTFT (client) | n=969 | 594126.49 | 581779.17 | 1092923.64 | 1145427.62 | 1175323.77 | 1185480.44 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               76.77 GiB  (96.9% of total)
- Average used:            74.67 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 76.77 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=93.5%
- GPU memory-bw util:      peak=73%  avg=44.9%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E001/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E001/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E001/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E001/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E001/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E001/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E001/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E001/gpu_final.txt

## Errors
✓ No errors detected
