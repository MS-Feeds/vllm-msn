# Experiment E010 - Results

## Configuration
Experiment ID: E010
Date: 2026-05-21 03:24:22
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
GPU Memory Util: 0.70
KV Cache Dtype: auto

## Performance
Duration: 913 seconds (15 minutes)

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
- Output tokens:         806,623
- Cached tokens:         747,200
- Prefix-cache hit rate: 13.38%
- Finish reasons:        {'length': 580, 'stop': 389}

### Throughput
- QPS:                1.1474 req/s
- Output tokens/sec:  955.11
- Prompt tokens/sec:  6610.38
- Total tokens/sec:   7565.49

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 418670.90 | 416836.16 | 758145.23 | 796674.84 | 816992.99 | 819439.70 |
| TPOT | n=969 | 64.62 | 63.75 | 79.51 | 91.25 | 156.26 | 435.56 |
| E2E | n=969 | 475377.63 | 486451.83 | 816506.15 | 836112.42 | 843537.12 | 844493.66 |
| TTFT (client) | n=969 | 423206.65 | 421322.70 | 766230.53 | 805169.14 | 825681.30 | 828190.48 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               59.44 GiB  (75.0% of total)
- Average used:            57.15 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 59.44 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=93.4%
- GPU memory-bw util:      peak=53%  avg=26.3%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E010/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E010/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E010/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E010/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E010/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E010/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E010/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E010/gpu_final.txt

## Errors
✓ No errors detected
