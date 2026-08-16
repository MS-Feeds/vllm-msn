#!/usr/bin/env python3
"""Minimal process for `ncu --metrics dram__bytes_read.sum` to attach to --
answers the ground-truth question the Python-level instrumentation in
`sparse_decode_microbench.py` can't: does the attention kernel's actual HBM
traffic match the metadata-derived KV-bytes estimate, now that
`sparse_target_runner.py`'s `max_seq_len` fix has landed?

Deliberately does ONE keep_rate per process (not both, like
`sparse_decode_microbench.py`) -- makes the resulting ncu report
unambiguous: every attention-kernel launch it captured belongs to that one
config, no need to disambiguate by launch index.

Deliberately does very few decode steps (`--decode-steps`, default 5) -- ncu
profiling overhead per captured kernel launch is high (each layer's
attention call is captured separately, so a handful of decode steps is
already dozens of launches at Llama-3.1-8B's 32 layers); a handful of
steady-state decode steps is enough to read a stable per-kernel DRAM byte
count, this doesn't need the full 512-token decode run.

## NVTX must be pushed inside the EngineCore subprocess, not the driver

First version of this script pushed `torch.cuda.nvtx.range_push` from the
driver's own step()-loop -- and `ncu --nvtx-include` matched ZERO kernels
("No kernels were profiled"), confirmed on real hardware. Root cause: vLLM
runs the actual CUDA work in a SEPARATE `EngineCore` subprocess (visible as
`(EngineCore pid=...)` in logs) -- the driver process's `llm_engine.step()`
just sends an RPC and blocks on the reply; it never launches a kernel
itself. NVTX ranges are per-process, so a range pushed in the driver is
invisible to anything happening in EngineCore.

Fix: `NvtxScopedSparseTargetGPUModelRunner` below pushes the range itself,
from inside `_build_attention_metadata` -- which genuinely executes in the
EngineCore subprocess (same place `SparseTargetGPUModelRunner`'s own
override already runs, confirmed by its logging showing up under the
`(EngineCore pid=...)` prefix). It pushes once, on the first genuine decode
step (same prefill-exclusion boundary used elsewhere), and the driver pops
it via an RPC call (`pop_nvtx_range`) after the request finishes -- that RPC
executes synchronously in the SAME subprocess/thread, so it closes the range
only after every decode-step kernel launch for this request has already
been enqueued.

Uses the REAL, un-instrumented gather logic from `SparseTargetGPUModelRunner`
(only the NVTX push/pop is added) -- no Python-level byte counting needed
here, ncu measures hardware truth directly.

## NVTX range naming -- avoid "/" in the range name itself

An earlier version of this script named the range `f"decode/{request_id}"`
-- Nsight Compute's `--nvtx-include` filter treats "/" as a RANGE-NESTING
separator (`"A/B"` means "kernels launched while nested inside range B,
itself nested inside range A"), not a literal character allowed in a single
range's own name. That collided with this script's actual (single,
unnested) range and produced "No kernels were profiled" even after the
process-placement fix below was already correct. The range name is now
`f"decode_kv_probe_{request_id}"` (underscores only) to avoid any ambiguity
with ncu's own filter syntax.

If you STILL see "No kernels were profiled" after that, debug in stages
rather than guessing further -- first confirm ncu can see ANY kernels at
all in the right process, decoupled from NVTX filtering entirely:

    ncu --target-processes all --metrics dram__bytes_read.sum \\
        -o ncu_debug_unfiltered \\
        python3 ncu_kv_bytes_probe.py --model $LLAMA31_8B_MODEL_PATH \\
            --context-tokens 4096 --decode-steps 3 --keep-rate 1.0 --kv-granularity 16

(small context/few steps -- with NO --nvtx and NO --kernel-name filter,
this captures every kernel in the whole run, so keep it cheap.) Then:

    ncu --import ncu_debug_unfiltered.ncu-rep --print-summary per-kernel

If that lists kernels, ncu is attaching to the EngineCore subprocess fine,
and the real attention kernel's exact name is now known -- use it (not a
guessed "flash"/"attn" substring) to build an accurate `--kernel-name`
regex, and add `--nvtx --nvtx-include "decode_kv_probe_ncu-probe-1.0/"`
back in. If it lists NOTHING, the problem is process attachment itself
(check `--target-processes all` took effect, and that ncu has the
perf-counter permissions needed to profile at all -- see the ERR_NVGPUCTRPERM
note from before).

Usage (run ONCE per keep rate, each under its own `ncu` invocation -- the
NVTX label is `f"decode_kv_probe_{request_id}"`, and `request_id` is
`f"ncu-probe-{keep_rate}"`, so e.g. for --keep-rate 1.0 the label is
"decode_kv_probe_ncu-probe-1.0"):

    ncu --nvtx --nvtx-include "decode_kv_probe_ncu-probe-1.0/" \\
        --kernel-name regex:".*[Ff]lash.*|.*[Aa]ttn.*" --kernel-name-base demangled \\
        --metrics dram__bytes_read.sum,dram__bytes_write.sum \\
        --target-processes all -o ncu_keep_1.0 \\
        python3 ncu_kv_bytes_probe.py --model $LLAMA31_8B_MODEL_PATH \\
            --context-tokens 77000 --decode-steps 5 --keep-rate 1.0 --kv-granularity 16

    ncu --nvtx --nvtx-include "decode_kv_probe_ncu-probe-0.2/" \\
        --kernel-name regex:".*[Ff]lash.*|.*[Aa]ttn.*" --kernel-name-base demangled \\
        --metrics dram__bytes_read.sum,dram__bytes_write.sum \\
        --target-processes all -o ncu_keep_0.2 \\
        python3 ncu_kv_bytes_probe.py --model $LLAMA31_8B_MODEL_PATH \\
            --context-tokens 77000 --decode-steps 5 --keep-rate 0.2 --kv-granularity 16

--target-processes all is still required -- ncu also needs to find and
attach to the EngineCore subprocess in the first place, on top of the NVTX
fix above.

Then, per captured kernel launch, compare against the PER-LAYER expected
bytes (not the per-step total across all 32 layers -- ncu's report is a
flat list of individual kernel launches, one per layer per step, so compare
apples to apples at that granularity):
    expected_bytes_per_layer = blocks_loaded * block_size
        * (2 * num_key_value_heads * head_dim * dtype_size)
(the same `bytes_per_token_kv` computed in `sparse_decode_microbench.py`,
divided by num_hidden_layers -- or just take that script's own printed
`kv_bytes_per_step` for this config and divide by 32).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Set

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

# Top-level, not deferred -- worker_cls resolution needs these as real
# module attributes at import time in the EngineCore subprocess (see
# sparse_decode_microbench.py's own comment on this exact failure mode).
from vllm.v1.worker.gpu_worker import Worker
from vllm_patch.sparse_target_runner import SparseTargetGPUModelRunner, SparseTargetWorker

# Reuse the already-verified context/selection builders rather than
# reimplementing them -- see sparse_decode_microbench.py.
from sparse_decode_microbench import build_synthetic_context, compute_keep_positions


class NvtxScopedSparseTargetGPUModelRunner(SparseTargetGPUModelRunner):
    def _nvtx_pushed_reqs(self) -> Set[str]:
        if not hasattr(self, "_nvtx_pushed_req_ids"):
            self._nvtx_pushed_req_ids: Set[str] = set()
        return self._nvtx_pushed_req_ids

    def _build_attention_metadata(self, *args, **kwargs):
        attn_metadata, spec_decode_meta = super()._build_attention_metadata(*args, **kwargs)
        self._maybe_push_nvtx(attn_metadata)
        return attn_metadata, spec_decode_meta

    def _maybe_push_nvtx(self, attn_metadata) -> None:
        # See module docstring -- this runs inside the EngineCore
        # subprocess, unlike a range pushed from the driver's step()-loop.
        if not attn_metadata or self.input_batch.num_reqs == 0:
            return
        req_id = self.input_batch.req_ids[0]
        num_computed = int(self.input_batch.num_computed_tokens_cpu_tensor[0])
        num_prompt = int(self.input_batch.num_prompt_tokens_cpu_tensor[0])
        if num_computed < num_prompt:
            return  # prefill, not decode -- excluded, same boundary as elsewhere
        pushed = self._nvtx_pushed_reqs()
        if req_id not in pushed:
            import torch
            torch.cuda.nvtx.range_push(f"decode_kv_probe_{req_id}")
            pushed.add(req_id)

    def pop_nvtx_range(self, request_id: str) -> None:
        pushed = self._nvtx_pushed_reqs()
        if request_id in pushed:
            import torch
            torch.cuda.nvtx.range_pop()
            pushed.discard(request_id)


class NvtxScopedSparseTargetWorker(SparseTargetWorker):
    def init_device(self) -> None:
        Worker.init_device(self)
        self.model_runner = NvtxScopedSparseTargetGPUModelRunner(self.vllm_config, self.device)

    def pop_nvtx_range(self, request_id: str) -> None:
        self.model_runner.pop_nvtx_range(request_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Defaults to $LLAMA31_8B_MODEL_PATH.")
    parser.add_argument("--context-tokens", type=int, default=77000)
    parser.add_argument("--decode-steps", type=int, default=5,
                         help="Small on purpose -- see module docstring on ncu overhead per launch.")
    parser.add_argument("--keep-rate", type=float, required=True,
                         help="Single value -- run this script (and ncu) once per keep rate.")
    parser.add_argument("--kv-granularity", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_path = args.model or os.environ.get("LLAMA31_8B_MODEL_PATH")
    if not model_path:
        parser.error("--model or $LLAMA31_8B_MODEL_PATH is required")

    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    sys.modules.setdefault("ncu_kv_bytes_probe", sys.modules[__name__])

    native_max_model_len = AutoConfig.from_pretrained(model_path, trust_remote_code=True).max_position_embeddings
    max_model_len = args.context_tokens + args.decode_steps + 64
    if native_max_model_len is not None:
        max_model_len = min(max_model_len, int(native_max_model_len))

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.kv_granularity,
        max_model_len=max_model_len,
        worker_cls="ncu_kv_bytes_probe.NvtxScopedSparseTargetWorker",
        enable_prefix_caching=True,
        async_scheduling=False,
    )
    llm_engine = llm.llm_engine
    tok = llm.get_tokenizer()
    actual_block_size = llm_engine.vllm_config.cache_config.block_size
    print(f"[ncu_probe] block_size={actual_block_size} (requested {args.kv_granularity}) "
          f"keep_rate={args.keep_rate} context_tokens={args.context_tokens} "
          f"decode_steps={args.decode_steps}")

    context_ids = build_synthetic_context(tok, args.context_tokens, args.seed)
    positions = compute_keep_positions(len(context_ids), actual_block_size, args.keep_rate)
    print(f"[ncu_probe] {len(positions)} blocks registered")

    request_id = f"ncu-probe-{args.keep_rate}"
    print(f"[ncu_probe] NVTX label will be 'decode_kv_probe_{request_id}' -- use that in --nvtx-include")
    llm_engine.collective_rpc("register_sparse_selection", args=(request_id, positions))

    sampling_params = SamplingParams(
        max_tokens=args.decode_steps, min_tokens=args.decode_steps, ignore_eos=True, temperature=0.0
    )
    prompt = TokensPrompt(prompt_token_ids=context_ids)
    real_id = llm_engine.add_request(request_id, prompt, sampling_params)
    assert real_id == request_id, (
        f"expected request_id={request_id!r} verbatim, got {real_id!r} -- "
        f"VLLM_DISABLE_REQUEST_ID_RANDOMIZATION must be set."
    )

    while llm_engine.has_unfinished_requests():
        llm_engine.step()

    # Pops the NVTX range from INSIDE the EngineCore subprocess (same
    # reasoning as the push -- see module docstring) -- runs after every
    # decode-step kernel launch for this request has already been enqueued.
    llm_engine.collective_rpc("pop_nvtx_range", args=(request_id,))
    llm_engine.collective_rpc("discard_sparse_selection", args=(request_id,))
    print("[ncu_probe] done")


if __name__ == "__main__":
    main()
