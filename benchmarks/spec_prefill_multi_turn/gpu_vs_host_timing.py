#!/usr/bin/env python3
"""Decomposes the measured decode-step wall-clock time (`sparse_decode_
microbench.py`'s ~22ms/tok at 4k context, keep_rate=1.0) into GPU execution
time vs. host-side time, and reports the actual resolved CUDA-graph config --
answers "where does the 22ms go" before guessing further.

## Hook point: `_model_forward`, not `execute_model`

`GPUModelRunner._model_forward` (vllm/v1/worker/gpu_model_runner.py:3653) is
a small, deliberately-overridable wrapper around the real `self.model(...)`
call -- its own docstring says exactly this: "This method can be overridden
by subclasses for model execution... We can inspect only this method versus
the whole execute_model, which has additional logic." That's the correct,
minimal hook: wrap ONLY the actual forward pass in `torch.cuda.Event` pairs,
without needing to reimplement or reason about `execute_model`'s much larger
surrounding logic (metadata building, sampling, KV-connector handling, etc).

`torch.cuda.Event(enable_timing=True)` pairs are used (not
`torch.cuda.synchronize()` around each step) so recording itself doesn't
force a sync on every step -- events are recorded async, and only
`.elapsed_time()` (called once, in `pop_gpu_forward_times_ms`, after a
single `torch.cuda.synchronize()`) actually blocks. This measures genuine
GPU execution time for the forward pass, decoupled from whatever the host
was doing at the same time -- the gap between this and the wall-clock
per-step time already measured by `sparse_decode_microbench.py`'s
`run_decode_only` (which DOES `torch.cuda.synchronize()` around the whole
step) is host-side time: Python dispatch, kernel launch overhead under
`enforce_eager=True` (no CUDA-graph batching to amortize it), the sparse
metadata-patch itself, sampling, etc.

## CUDA-graph config

Also reports, once per engine construction, the ACTUAL resolved
`enforce_eager`/`cudagraph_mode`/`cudagraph_capture_sizes` read directly
from `vllm_config` -- not inferred from what flags were passed, in case
something downstream (e.g. `vllm/config/vllm.py`'s own enforce_eager
handling, which forces `cudagraph_capture_sizes = []` when enforce_eager is
set) changes what actually gets resolved. `cudagraph_capture_sizes`
defaults (see `vllm/config/vllm.py`) are small integers keyed by BATCH SIZE
(`[1, 2, 4, 8, 16, ...]`), not context length -- consistent with captured
decode graphs being replayable at any context length via pre-allocated,
in-place-written block_table/seq_lens buffers, which is relevant to
whether the sparse gather's own in-place tensor writes
(`block_table[req_idx, :] = ...`, `seq_lens[req_idx] = ...` in
`sparse_target_runner.py`) are structurally graph-capture-compatible or not
-- this script doesn't test that directly (this run is always
enforce_eager=True, matching every other script in this set so far), it
just reports what's actually configured so that question isn't answered by
assumption.

Usage:
    python3 gpu_vs_host_timing.py --model $LLAMA31_8B_MODEL_PATH \\
        --context-tokens 4000 --decode-tokens 64 --reps 3 --kv-granularity 16
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

from vllm.v1.worker.gpu_worker import Worker
from sparse_decode_microbench import (
    InstrumentedSparseTargetGPUModelRunner,
    InstrumentedSparseTargetWorker,
    build_synthetic_context,
    compute_keep_positions,
    run_decode_only,
)


class GpuTimedSparseTargetGPUModelRunner(InstrumentedSparseTargetGPUModelRunner):
    def _gpu_forward_events_accum(self) -> Dict[str, List[Tuple[object, object]]]:
        if not hasattr(self, "_gpu_forward_events"):
            self._gpu_forward_events: Dict[str, List[Tuple[object, object]]] = {}
        return self._gpu_forward_events

    def _model_forward(self, *args, **kwargs):
        # Only time genuine decode steps -- same prefill-exclusion boundary
        # used everywhere else in this set of scripts (num_computed >=
        # num_prompt). Prefill's forward shape/cost is a different question
        # not being asked here.
        is_decode = False
        req_id = None
        if self.input_batch.num_reqs > 0:
            req_id = self.input_batch.req_ids[0]
            num_computed = int(self.input_batch.num_computed_tokens_cpu_tensor[0])
            num_prompt = int(self.input_batch.num_prompt_tokens_cpu_tensor[0])
            is_decode = num_computed >= num_prompt

        if not is_decode:
            return super()._model_forward(*args, **kwargs)

        import torch
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = super()._model_forward(*args, **kwargs)
        end.record()
        self._gpu_forward_events_accum().setdefault(req_id, []).append((start, end))
        return out

    def pop_gpu_forward_times_ms(self, request_id: str) -> List[float]:
        events = self._gpu_forward_events_accum().pop(request_id, [])
        if not events:
            return []
        import torch
        # Single sync here, not one per step -- see module docstring on why
        # events (not per-step synchronize()) are used for the recording
        # side; this is the only blocking point, and it's after the fact.
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in events]


class GpuTimedSparseTargetWorker(InstrumentedSparseTargetWorker):
    def init_device(self) -> None:
        # Same "call Worker.init_device directly, construct only the final
        # runner once" reasoning as InstrumentedSparseTargetWorker's own
        # override.
        Worker.init_device(self)
        self.model_runner = GpuTimedSparseTargetGPUModelRunner(self.vllm_config, self.device)

    def pop_gpu_forward_times_ms(self, request_id: str) -> List[float]:
        return self.model_runner.pop_gpu_forward_times_ms(request_id)

    def report_cudagraph_config(self) -> dict:
        cc = self.vllm_config.compilation_config
        return dict(
            enforce_eager=self.vllm_config.model_config.enforce_eager,
            cudagraph_mode=str(cc.cudagraph_mode),
            cudagraph_capture_sizes=list(cc.cudagraph_capture_sizes) if cc.cudagraph_capture_sizes else [],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Defaults to $LLAMA31_8B_MODEL_PATH.")
    parser.add_argument("--context-tokens", type=int, default=4000)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--keep-rate", type=float, default=1.0)
    parser.add_argument("--reps", type=int, default=3, help="rep 0 is discarded as warmup.")
    parser.add_argument("--kv-granularity", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_path = args.model or os.environ.get("LLAMA31_8B_MODEL_PATH")
    if not model_path:
        parser.error("--model or $LLAMA31_8B_MODEL_PATH is required")
    if args.reps < 2:
        parser.error("--reps must be >= 2 (rep 0 is discarded as warmup).")

    from transformers import AutoConfig
    from vllm import LLM

    sys.modules.setdefault("gpu_vs_host_timing", sys.modules[__name__])

    native_max_model_len = AutoConfig.from_pretrained(model_path, trust_remote_code=True).max_position_embeddings
    max_model_len = args.context_tokens + args.decode_tokens + 64
    if native_max_model_len is not None:
        max_model_len = min(max_model_len, int(native_max_model_len))

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.kv_granularity,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=max_model_len,
        worker_cls="gpu_vs_host_timing.GpuTimedSparseTargetWorker",
        enable_prefix_caching=True,
        async_scheduling=False,
    )
    llm_engine = llm.llm_engine
    tok = llm.get_tokenizer()
    actual_block_size = llm_engine.vllm_config.cache_config.block_size

    cg_config = llm_engine.collective_rpc("report_cudagraph_config")[0]
    print(f"[gpu_vs_host_timing] CUDA-graph config actually resolved: {json.dumps(cg_config)}")
    print(f"[gpu_vs_host_timing] block_size={actual_block_size} (requested {args.kv_granularity}) "
          f"context_tokens={args.context_tokens} keep_rate={args.keep_rate}")

    context_ids = build_synthetic_context(tok, args.context_tokens, args.seed)
    positions = compute_keep_positions(len(context_ids), actual_block_size, args.keep_rate)

    rep_wall_ms: List[List[float]] = []
    rep_gpu_ms: List[List[float]] = []
    for rep in range(args.reps):
        request_id = f"gputiming-{args.context_tokens}-{args.keep_rate}-{rep}"
        t_rep = time.time()
        wall_latencies = run_decode_only(
            llm_engine, request_id, context_ids, positions, args.decode_tokens,
        )
        gpu_ms = llm_engine.collective_rpc("pop_gpu_forward_times_ms", args=(request_id,))[0]
        print(f"[gpu_vs_host_timing] rep={rep}: {len(wall_latencies)} decode steps in "
              f"{time.time() - t_rep:.1f}s wall ({'WARMUP, discarded' if rep == 0 else 'kept'})")
        if rep == 0:
            continue
        rep_wall_ms.append([s * 1000.0 for s in wall_latencies])
        rep_gpu_ms.append(gpu_ms)

    pooled_wall_ms = [v for rep in rep_wall_ms for v in rep]
    pooled_gpu_ms = [v for rep in rep_gpu_ms for v in rep]

    if len(pooled_wall_ms) != len(pooled_gpu_ms):
        print(f"[gpu_vs_host_timing] WARNING: {len(pooled_wall_ms)} wall-clock samples vs "
              f"{len(pooled_gpu_ms)} GPU-event samples -- counts should match exactly (one "
              f"event pair per timed decode step); a mismatch means the pairing below is "
              f"comparing medians of possibly-misaligned populations, not the same steps.")

    wall_median = statistics.median(pooled_wall_ms)
    gpu_median = statistics.median(pooled_gpu_ms) if pooled_gpu_ms else float("nan")
    host_gap_ms = wall_median - gpu_median

    print(f"\n{'=' * 70}")
    print(f"wall-clock ms/tok (median, host-synchronized): {wall_median:.3f}")
    print(f"GPU forward-pass ms/tok (median, CUDA-event):  {gpu_median:.3f}")
    print(f"host-side gap (wall - GPU):                    {host_gap_ms:+.3f} ms")
    print(f"{'=' * 70}")
    if gpu_median > 0 and host_gap_ms / wall_median > 0.3:
        print(
            f"\n[HOST-BOUND] {host_gap_ms:.3f} ms of the {wall_median:.3f} ms/tok wall-clock total "
            f"({host_gap_ms / wall_median * 100:.0f}%) is NOT GPU execution time -- consistent with "
            f"enforce_eager=True's per-step Python dispatch/kernel-launch overhead (no CUDA-graph "
            f"batching to amortize it) dominating over actual GPU work. cudagraph_mode was reported "
            f"as {cg_config['cudagraph_mode']!r} (enforce_eager={cg_config['enforce_eager']}) for this "
            f"run -- see stock_vllm_control.py for whether enabling graphs (on a build WITHOUT the "
            f"sparse override) recovers this gap, which would tell you whether the fix is making the "
            f"override itself capture-compatible, or whether the overhead is elsewhere."
        )
    else:
        print(
            f"\n[GPU-BOUND] host-side gap ({host_gap_ms:.3f} ms) is a modest fraction of total wall-clock "
            f"time -- the {wall_median:.3f} ms/tok figure is mostly genuine GPU execution time, not host "
            f"dispatch overhead. If that's still far above the bandwidth roofline, the GPU kernels "
            f"themselves are inefficient, not the host loop -- worth an nsys/ncu kernel-level look next, "
            f"not more host-side profiling."
        )

    results = dict(
        args=vars(args),
        cudagraph_config=cg_config,
        wall_ms_median=wall_median,
        gpu_ms_median=gpu_median,
        host_gap_ms=host_gap_ms,
        raw_wall_ms=pooled_wall_ms,
        raw_gpu_ms=pooled_gpu_ms,
    )
    out_path = Path(f"gpu_vs_host_timing_{args.context_tokens}_{args.keep_rate}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[gpu_vs_host_timing] full results written to {out_path}")


if __name__ == "__main__":
    main()
