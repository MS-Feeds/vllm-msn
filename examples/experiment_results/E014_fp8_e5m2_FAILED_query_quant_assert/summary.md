# Experiment E014 - Results

## Configuration
Experiment ID: E014
Date: 2026-05-21 07:09:59
Status: Failed

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
KV Cache Dtype: fp8_e5m2

## Performance
Duration: 28 seconds (0 minutes)

(metrics.json not found — the Python entrypoint likely crashed before writing metrics)

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               50.78 GiB  (64.1% of total)
- Average used:            17.49 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 50.78 GiB  (model load + inference working set)
- GPU compute util:        peak=100%  avg=11.9%
- GPU memory-bw util:      peak=3%  avg=0.2%

## Files
- Environment:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E014/environment.txt
- Inference log:         /nvmedata/chenw/vllm-ra/examples/experiment_results/E014/inference.log
- GPU trace (1Hz):       /nvmedata/chenw/vllm-ra/examples/experiment_results/E014/gpu_trace.csv
- Generations:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E014/output.jsonl
- Per-request metrics:   /nvmedata/chenw/vllm-ra/examples/experiment_results/E014/per_request_metrics.jsonl
- Aggregate metrics:     /nvmedata/chenw/vllm-ra/examples/experiment_results/E014/metrics.json
- GPU initial:           /nvmedata/chenw/vllm-ra/examples/experiment_results/E014/gpu_initial.txt
- GPU final:             /nvmedata/chenw/vllm-ra/examples/experiment_results/E014/gpu_final.txt

## Errors
⚠ Found 65 error mentions

Sample errors:
(EngineCore pid=1289840) ERROR 05-21 07:09:57 [core.py:1159] EngineCore failed to start.
(EngineCore pid=1289840) ERROR 05-21 07:09:57 [core.py:1159] Traceback (most recent call last):
(EngineCore pid=1289840) ERROR 05-21 07:09:57 [core.py:1159]   File "/nvmedata/chenw/vllm-ra/vllm/v1/engine/core.py", line 1133, in run_engine_core
(EngineCore pid=1289840) ERROR 05-21 07:09:57 [core.py:1159]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=1289840) ERROR 05-21 07:09:57 [core.py:1159]   File "/nvmedata/chenw/vllm-ra/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=1289840) ERROR 05-21 07:09:57 [core.py:1159]     return func(*args, **kwargs)
(EngineCore pid=1289840) ERROR 05-21 07:09:57 [core.py:1159]   File "/nvmedata/chenw/vllm-ra/vllm/v1/engine/core.py", line 899, in __init__
(EngineCore pid=1289840) ERROR 05-21 07:09:57 [core.py:1159]     super().__init__(
(EngineCore pid=1289840) ERROR 05-21 07:09:57 [core.py:1159]   File "/nvmedata/chenw/vllm-ra/vllm/v1/engine/core.py", line 128, in __init__
(EngineCore pid=1289840) ERROR 05-21 07:09:57 [core.py:1159]     kv_cache_config = self._initialize_kv_caches(vllm_config)
