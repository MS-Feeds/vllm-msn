# Experiment E014 - Results

## Configuration
Experiment ID: E014
Date: 2026-05-21 06:26:24
Status: Failed

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
KV Cache Dtype: fp8_e4m3

## Performance
Duration: 46 seconds (0 minutes)

(metrics.json not found — the Python entrypoint likely crashed before writing metrics)

## Engine-level snapshots (from vLLM periodic stats in inference.log)
- Peak KV cache usage:    N/A%
- Peak running requests:  N/A
- Peak waiting requests:  N/A

## GPU memory & utilization (GPU 0, sampled at 1 Hz)
- Total GPU memory:        79.25 GiB (81,157 MiB)
- Baseline (trace start):  0.00 GiB
- Peak used:               56.26 GiB  (71.0% of total)
- Average used:            17.22 GiB
- End (trace end):         0.00 GiB
- Delta (peak - baseline): 56.26 GiB  (model load + inference working set)
- GPU compute util:        peak=44%  avg=7.8%
- GPU memory-bw util:      peak=2%  avg=0.3%

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
⚠ Found 123 error mentions

Sample errors:
(EngineCore pid=1265650) [rank0]:E0521 06:26:22.322000 1265650 site-packages/torch/_inductor/runtime/triton_heuristics.py:874] triton.compiler.errors.CompilationError: at 1:0:
(EngineCore pid=1265650) [rank0]:E0521 06:26:22.322000 1265650 site-packages/torch/_inductor/runtime/triton_heuristics.py:874] ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
(EngineCore pid=1265650) [rank0]:E0521 06:26:22.945000 1265650 site-packages/torch/_inductor/runtime/triton_heuristics.py:874] triton.compiler.errors.CompilationError: at 1:0:
(EngineCore pid=1265650) [rank0]:E0521 06:26:22.945000 1265650 site-packages/torch/_inductor/runtime/triton_heuristics.py:874] ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
(EngineCore pid=1265650) ERROR 05-21 06:26:22 [core.py:1159] EngineCore failed to start.
(EngineCore pid=1265650) ERROR 05-21 06:26:22 [core.py:1159] Traceback (most recent call last):
(EngineCore pid=1265650) ERROR 05-21 06:26:22 [core.py:1159]   File "/nvmedata/chenw/vllm-ra/vllm/v1/engine/core.py", line 1133, in run_engine_core
(EngineCore pid=1265650) ERROR 05-21 06:26:22 [core.py:1159]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=1265650) ERROR 05-21 06:26:22 [core.py:1159]   File "/nvmedata/chenw/vllm-ra/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=1265650) ERROR 05-21 06:26:22 [core.py:1159]     return func(*args, **kwargs)
