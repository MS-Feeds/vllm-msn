# Experiment E007 - Results

## Configuration
Experiment ID: E007
Date: 2026-05-21 02:37:20
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
GPU Memory Util: 0.75
KV Cache Dtype: auto

## Performance
Duration: 984 seconds (16 minutes)

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
- Output tokens:         805,582
- Cached tokens:         751,200
- Prefix-cache hit rate: 13.46%
- Finish reasons:        {'length': 578, 'stop': 391}

### Throughput
- QPS:                1.1515 req/s
- Output tokens/sec:  957.32
- Prompt tokens/sec:  6634.25
- Total tokens/sec:   7591.57

### Latency distributions (ms)

NOTE: TTFT (engine) is the canonical TTFT — read from vLLM's
RequestStateStats.first_token_latency. TTFT (client) is a sanity-check;
it adds asyncio queue/scheduling slop on the consumer side and will
read slightly higher under heavy concurrency.

| metric | n | mean | p50 | p90 | p95 | p99 | max |
|--------|---|------|-----|-----|-----|-----|-----|
| TTFT (engine) | n=969 | 417756.13 | 415347.89 | 755293.03 | 788597.74 | 812192.67 | 816750.88 |
| TPOT | n=969 | 77.08 | 75.38 | 94.37 | 111.12 | 225.23 | 523.26 |
| E2E | n=969 | 483112.77 | 497292.56 | 816053.80 | 831879.01 | 839270.61 | 841454.69 |
| TTFT (client) | n=969 | 422190.22 | 419748.88 | 763163.86 | 796859.25 | 820628.67 | 825264.08 |

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               63.40 GiB  (80.0% of total)
- Average used:            58.41 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 63.39 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=86.0%
- GPU memory-bw util:      peak=87%  avg=24.3%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E007/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E007/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E007/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E007/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E007/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E007/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E007/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E007/gpu_final.txt

## Errors
✓ No errors detected
