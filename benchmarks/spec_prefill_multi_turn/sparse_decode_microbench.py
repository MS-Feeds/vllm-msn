#!/usr/bin/env python3
"""Standalone microbenchmark isolating DECODE-TIME cost of the sparse-attention
block-table-gather mechanism (`vllm_patch/sparse_target_runner.py`) -- no
speculator, no scoring, no multi-turn conversation harness. This is not part of
the SCBench eval pipeline (`predict_scbench.py`); it exists purely to answer:
given a fixed keep-rate and KV-cache block granularity, how many KV blocks does
the attention kernel actually get asked to read per decode step, and does that
translate into lower measured per-token latency versus dense (keep=1.0)?

## What this reuses vs. what it bypasses

Reuses the REAL production mechanism unmodified: `SparseTargetGPUModelRunner`/
`SparseTargetWorker`'s `_build_attention_metadata` override, which patches
`AttentionMetadata.block_table`/`.seq_lens` before the stock, unmodified
attention kernel runs (see that module's own docstring for the full
call-chain tracing). This script drives it exactly the way
`validate_sparse_attention.py` does: `register_sparse_selection` via
`collective_rpc` BEFORE `add_request`, one ordinary one-shot request (no
resumable session -- irrelevant here, this is a single turn).

Bypasses the speculator/proposer/scoring machinery entirely (`proposer.py`,
`scoring.py`) -- this script computes its OWN deterministic, block-aligned
position list for a given `--keep-rate` (uniform stride over block indices),
standing in for whatever a real speculator would have selected. That's a
legitimate substitution here because this benchmark's question is about the
MECHANISM's per-block-count cost, not about selection quality.

## `--kv-granularity` == the engine's real KV-cache `block_size`

The gather mechanism operates on `cache_config.block_size`, not on
`scoring.py`'s `chunk_size` (that only matters to a real speculator's
selection, which this script doesn't run) -- see `EXPERIMENT_PLAN.md`'s own
admission that for the SPARSE pipeline, "granularity" reduces to block_size.
So `--kv-granularity` is plumbed straight through as `LLM(block_size=...)`.
Default is 16 (this fork's own default block size).

## `--keep-mode` is inert here, on purpose

`SpecConfig.keep_mode` ("keep" vs "discard", `vllm_patch/config.py:48`) only
affects candidate-pool construction ACROSS TURNS (`conversation_state.py`).
This benchmark is one fixed prefill + one decode phase -- there is no second
turn for it to apply to. `--keep-mode` is accepted and echoed into the output
purely for label parity with the real sweep's config naming; it has no effect
on anything this script measures.

## Instrumentation: block counts are read directly off what the kernel is
## told to read, not inferred from `--keep-rate`

`InstrumentedSparseTargetGPUModelRunner` below overrides only
`_compute_gathered_view` (called exactly once per request per decode step,
before the per-layer metadata-patch loop -- see the base class for the call
site) to additionally record how many KV blocks this step's gathered
`seq_len` implies. Every block in a gathered view is either fully occupied
(`block_size` tokens) or -- for at most one boundary/force-keep block -- the
turn's own historically-growing tail (see `kv_cache_utils.py`'s
`BaseGatherView` docstring for exactly why only one block can be partial);
that structural guarantee is what makes `ceil(gathered_seq_len / block_size)`
an EXACT block count, not an approximation, and it's also literally the same
arithmetic a paged-attention kernel does internally to know how many blocks
to loop over from `seq_lens` -- so this is reading the same number the
kernel itself will read, not reverse-engineering it from the keep rate.

`req_id` is not a parameter of `_compute_gathered_view` (the base class's
call site doesn't pass it -- see `_apply_sparse_attention_overrides`), so
this override falls back to `self.input_batch.req_ids[0]`, relying on the
same "one in-flight request at a time" assumption the base module's own
docstring already documents as load-bearing elsewhere (Known risk area #2).
This script never has more than one request in flight, so that's safe here.

## Timing methodology

Wall-clock ms/decode-step is measured with `torch.cuda.synchronize()`
immediately before starting the timer and immediately after `engine.step()`
returns, `time.perf_counter()` in between. The step that produces the FIRST
generated token (i.e. the step after which `output.outputs[0].token_ids`
first becomes non-empty) is prefill/TTFT and is excluded entirely -- every
step after that produces exactly one new token (single request, batch=1) and
is a genuine decode step.

Attention-vs-everything-else wall-clock split and the profiler/nsys trace are
measured in a SEPARATE, short pass per config (`--profile-decode-tokens`,
default 64), run AFTER the timed reps finish -- wrapping every one of the
(reps * decode_tokens) timed steps in `torch.profiler` would itself add
Python-side overhead into the very "everything else" bucket being measured,
contaminating the timed numbers. The profiled pass's own `torch.profiler`
`key_averages()` are bucketed into attention vs. other by CUDA kernel name
substring match (`--attention-kernel-patterns`) and a Chrome/Perfetto trace
is written per config.

**KNOWN BROKEN, not yet fixed**: `run_profiled_decode`'s `torch.profiler`
context currently wraps the DRIVER process's `step()`-loop -- but vLLM runs
the actual CUDA work in a separate EngineCore subprocess (confirmed via the
identical NVTX bug below, and directly evidenced by this producing
`attn_ms_per_step`/`other_ms_per_step` of exactly `0.000` for every config
on real hardware: the driver process launches no kernels of its own for
`torch.profiler` to see). Treat `attn_ms_per_step`/`other_ms_per_step` in
the output table as not-yet-trustworthy until this is fixed the same way
the NVTX push below was -- profiling needs to be started/stopped from
INSIDE the worker process (e.g. via a `collective_rpc`-triggered
`torch.profiler` session, or vLLM's own built-in start/stop-profile RPC
surface if this fork exposes one), not from the driver.

The NVTX range used for `nsys` diffing IS fixed: it's pushed/popped inside
`InstrumentedSparseTargetGPUModelRunner` (worker-process side, keyed by
`request_id`, label `f"decode_kv_probe_{request_id}"`), not from this
script's own step()-loop -- an earlier version pushed driver-side and `ncu
--nvtx-include` confirmed on real hardware that it matched ZERO kernels
("No kernels were profiled"), since NVTX ranges are per-process and the
driver's own `step()` calls never launch a kernel themselves. The range
name deliberately avoids "/" -- Nsight Compute's `--nvtx-include` treats
"/" as a range-NESTING separator, not a literal character, and an earlier
label (`f"decode/{request_id}"`) collided with that and silently continued
to match nothing even after the process-placement fix. Running this whole
script under `nsys profile --trace=cuda,nvtx -o <file> python3
sparse_decode_microbench.py ...` produces a trace where each
(keep_rate, rep)'s decode phase is demarcated by its own
`decode_kv_probe_microbench-<keep_rate>-<rep>` range -- this script never
shells out to `nsys` itself.

## Roofline

`roofline_ms = (weights_gib * 2^30 + kv_bytes_measured) / (bandwidth_tbps *
1e12) * 1000` -- GiB for memory sizes, decimal TB/s for bandwidth, matching
the unit convention implied by reverse-checking a stated ~12.9/~8.9 ms/tok
target against ~15GiB weights / ~9.5GiB KV / ~2.0e12 B/s. `kv_bytes_measured`
is THIS config's own measured bytes/step (from the direct block-count
instrumentation above), not a value derived from `--keep-rate`.

Usage:
    python3 sparse_decode_microbench.py --model $LLAMA31_8B_MODEL_PATH \\
        --context-tokens 77000 --decode-tokens 512 --keep-rates 1.0,0.2 \\
        --kv-granularity 16 --reps 3

    # under nsys, to diff the two configs' attention kernel behavior:
    nsys profile --trace=cuda,nvtx -o sparse_decode_trace \\
        python3 sparse_decode_microbench.py --model $LLAMA31_8B_MODEL_PATH ...
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

# Must be set before add_request() -> InputProcessor.assign_request_id first
# reads it -- same requirement as validate_sparse_attention.py/
# validate_resumable_session.py's identical default. Without this, a random
# suffix gets appended to request_id.
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

# Top-level (not deferred into a function) is required: `worker_cls`
# resolution (`resolve_obj_by_qualname`, vllm/utils/import_utils.py) imports
# this module fresh in the EngineCore subprocess and does a plain
# `getattr(module, "InstrumentedSparseTargetWorker")` -- a class only
# defined inside a function body is never an attribute of the module at
# all, regardless of fork vs. spawn. (An earlier version of this script
# defined these inside a function and hit exactly that:
# `AttributeError: module '__main__' has no attribute
# 'InstrumentedSparseTargetWorker'` from the EngineCore subprocess.)
from vllm.v1.worker.gpu_worker import Worker
from vllm_patch.sparse_target_runner import (
    SparseTargetGPUModelRunner,
    SparseTargetWorker,
)


# --------------------------------------------------------------------------
# Instrumented worker/runner -- subclasses only, no vllm_patch changes.
# --------------------------------------------------------------------------

class InstrumentedSparseTargetGPUModelRunner(SparseTargetGPUModelRunner):
    def _block_counts_accum(self) -> Dict[str, List[int]]:
        if not hasattr(self, "_microbench_block_counts"):
            self._microbench_block_counts: Dict[str, List[int]] = {}
        return self._microbench_block_counts

    def _nvtx_pushed_reqs(self):
        if not hasattr(self, "_nvtx_pushed_req_ids"):
            self._nvtx_pushed_req_ids = set()
        return self._nvtx_pushed_req_ids

    def _build_attention_metadata(self, *args, **kwargs):
        attn_metadata, spec_decode_meta = super()._build_attention_metadata(*args, **kwargs)
        self._maybe_push_nvtx(attn_metadata)
        return attn_metadata, spec_decode_meta

    def _maybe_push_nvtx(self, attn_metadata) -> None:
        # Pushed HERE (inside the EngineCore subprocess where the actual
        # attention kernels launch), NOT from the driver's step()-loop --
        # NVTX ranges are per-process, and vLLM's real CUDA work happens in
        # a separate EngineCore subprocess from the driver. An earlier
        # version of this pushed driver-side, which `ncu --nvtx-include`
        # confirmed sees ZERO kernels ("No kernels were profiled") since
        # the driver's own step() calls never launch a kernel themselves.
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

    def _compute_gathered_view(
        self, base_view, full_block_table_row, block_size, num_prompt, seq_len
    ):
        # `seq_len`, not `num_computed` -- renamed in lockstep with
        # `sparse_target_runner.py`/`kv_cache_utils.py` when the gather's
        # length parameter was corrected from `num_computed_tokens` to the
        # step's real `num_computed_tokens + num_scheduled_tokens` (see
        # `compute_sparse_gather_view`'s `seq_len` Args entry). This
        # override must keep the base method's exact signature or the
        # base's own keyword call raises TypeError.
        gathered = super()._compute_gathered_view(
            base_view=base_view,
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            num_prompt=num_prompt,
            seq_len=seq_len,
        )
        if gathered is None:
            # Degenerate no-op case (selection covers every resident
            # block already, e.g. keep_rate=1.0) -- the dense fallback
            # block count is exactly the number of blocks allocated so
            # far, same ceil-div the base module's own range checks use.
            blocks_loaded = -(-seq_len // block_size)
        else:
            _, gathered_seq_len = gathered
            blocks_loaded = -(-gathered_seq_len // block_size)
        # Single-in-flight-request assumption -- see module docstring.
        req_id = self.input_batch.req_ids[0]
        self._block_counts_accum().setdefault(req_id, []).append(blocks_loaded)
        return gathered

    def pop_block_counts(self, request_id: str) -> List[int]:
        return self._block_counts_accum().pop(request_id, [])


class InstrumentedSparseTargetWorker(SparseTargetWorker):
    def init_device(self) -> None:
        # Deliberately call Worker.init_device directly (not
        # super().init_device(), which is SparseTargetWorker's own
        # override) to avoid constructing the non-instrumented
        # SparseTargetGPUModelRunner first only to immediately replace
        # it -- same device-init side effects, one runner instead of two.
        Worker.init_device(self)
        self.model_runner = InstrumentedSparseTargetGPUModelRunner(self.vllm_config, self.device)

    def pop_block_counts(self, request_id: str) -> List[int]:
        return self.model_runner.pop_block_counts(request_id)

    def pop_nvtx_range(self, request_id: str) -> None:
        self.model_runner.pop_nvtx_range(request_id)


# --------------------------------------------------------------------------
# Context / selection construction
# --------------------------------------------------------------------------

def build_synthetic_context(tok, num_tokens: int, seed: int) -> List[int]:
    """Deterministic synthetic token ids, avoiding the tokenizer's special
    ids so the prompt can't accidentally contain e.g. an EOS/BOS mid-stream."""
    import numpy as np

    special_ids = set(tok.all_special_ids or [])
    vocab_size = tok.vocab_size
    allowed = np.array([i for i in range(vocab_size) if i not in special_ids], dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(allowed), size=num_tokens)
    return allowed[idx].tolist()


def compute_keep_positions(total_tokens: int, block_size: int, keep_rate: float) -> List[int]:
    """One representative position per kept block (block-index granularity
    is all the gather mechanism cares about -- see
    `block_indices_from_positions`), evenly spread across the context via
    `linspace` over block indices so the selection isn't concentrated in one
    region (a more representative stand-in for a real speculator's selection
    than e.g. "keep the first N blocks")."""
    import numpy as np

    total_blocks = -(-total_tokens // block_size)
    num_keep = max(1, round(total_blocks * keep_rate))
    if num_keep >= total_blocks:
        keep_block_indices = list(range(total_blocks))
    else:
        keep_block_indices = sorted(set(np.linspace(0, total_blocks - 1, num_keep).round().astype(int).tolist()))
    return [b * block_size for b in keep_block_indices]


# --------------------------------------------------------------------------
# Model-derived constants
# --------------------------------------------------------------------------

def bytes_per_token_kv(hf_config, dtype) -> int:
    import torch

    num_layers = hf_config.num_hidden_layers
    num_kv_heads = getattr(hf_config, "num_key_value_heads", None) or hf_config.num_attention_heads
    head_dim = getattr(hf_config, "head_dim", None) or (hf_config.hidden_size // hf_config.num_attention_heads)
    dtype_size = torch.empty(0, dtype=dtype).element_size()
    return 2 * num_layers * num_kv_heads * head_dim * dtype_size  # 2 == K and V


# --------------------------------------------------------------------------
# Decode-only driving loop
# --------------------------------------------------------------------------

def run_decode_only(
    llm_engine,
    request_id: str,
    context_ids: List[int],
    positions: List[int],
    decode_tokens: int,
) -> List[float]:
    """Registers `positions` as the sparse selection, submits one one-shot
    request for `context_ids` + `decode_tokens` forced generated tokens, and
    returns the wall-clock seconds for each genuine decode step (excludes
    the prefill/first-token step entirely). See module docstring for the
    boundary-detection and synchronization methodology.

    NVTX range for `nsys` diffing is pushed/popped INSIDE
    `InstrumentedSparseTargetGPUModelRunner` (worker-process side, keyed by
    `request_id`), not here -- see that class's own comment for why a
    driver-side push is invisible to `ncu`/`nsys` (vLLM's real CUDA work
    runs in a separate EngineCore subprocess)."""
    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    llm_engine.collective_rpc("register_sparse_selection", args=(request_id, positions))

    sampling_params = SamplingParams(
        max_tokens=decode_tokens, min_tokens=decode_tokens, ignore_eos=True, temperature=0.0
    )
    prompt = TokensPrompt(prompt_token_ids=context_ids)
    real_id = llm_engine.add_request(request_id, prompt, sampling_params)
    assert real_id == request_id, (
        f"expected request_id={request_id!r} verbatim, got {real_id!r} -- "
        f"VLLM_DISABLE_REQUEST_ID_RANDOMIZATION must be set."
    )

    step_latencies: List[float] = []
    first_token_seen = False
    while llm_engine.has_unfinished_requests():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm_engine.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        out = next((o for o in outputs if o.request_id == request_id), None)
        if out is None:
            continue
        has_token = len(out.outputs[0].token_ids) > 0
        if not first_token_seen:
            # This step is prefill (+ the first generated token) --
            # excluded entirely, per "excluding prefill/TTFT entirely".
            if has_token:
                first_token_seen = True
            continue
        # Every step after the first-token step is a pure decode step:
        # exactly one new token, single request, batch=1.
        step_latencies.append(t1 - t0)
        if out.finished:
            break

    # Pops the NVTX range from inside the worker process -- runs after
    # every decode-step kernel launch for this request has been enqueued.
    llm_engine.collective_rpc("pop_nvtx_range", args=(request_id,))
    llm_engine.collective_rpc("discard_sparse_selection", args=(request_id,))
    return step_latencies


def run_profiled_decode(
    llm_engine, request_id: str, context_ids: List[int], positions: List[int],
    decode_tokens: int, attention_patterns: List[str], trace_path: Path,
) -> Tuple[float, float]:
    """Same driving loop as `run_decode_only`, wrapped in `torch.profiler`
    for a short, separate pass (see module docstring for why this isn't
    combined with the timed reps). Returns (attn_ms_per_step, other_ms_per_step)
    averaged over the profiled decode steps, and writes a Chrome/Perfetto
    trace to `trace_path`."""
    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from torch.profiler import profile, ProfilerActivity

    llm_engine.collective_rpc("register_sparse_selection", args=(request_id, positions))
    sampling_params = SamplingParams(
        max_tokens=decode_tokens, min_tokens=decode_tokens, ignore_eos=True, temperature=0.0
    )
    prompt = TokensPrompt(prompt_token_ids=context_ids)
    real_id = llm_engine.add_request(request_id, prompt, sampling_params)
    assert real_id == request_id

    first_token_seen = False
    num_decode_steps = 0
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        while llm_engine.has_unfinished_requests():
            outputs = llm_engine.step()
            out = next((o for o in outputs if o.request_id == request_id), None)
            if out is None:
                continue
            has_token = len(out.outputs[0].token_ids) > 0
            if not first_token_seen:
                if has_token:
                    first_token_seen = True
                continue
            num_decode_steps += 1
            if out.finished:
                break

    llm_engine.collective_rpc("discard_sparse_selection", args=(request_id,))
    prof.export_chrome_trace(str(trace_path))

    attn_us = 0.0
    other_us = 0.0
    for evt in prof.key_averages():
        cuda_time = getattr(evt, "self_cuda_time_total", 0) or getattr(evt, "self_device_time_total", 0)
        name_lower = evt.key.lower()
        if any(p in name_lower for p in attention_patterns):
            attn_us += cuda_time
        else:
            other_us += cuda_time

    denom = max(num_decode_steps, 1)
    attn_ms_per_step = (attn_us / 1000.0) / denom
    other_ms_per_step = (other_us / 1000.0) / denom
    return attn_ms_per_step, other_ms_per_step


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Defaults to $LLAMA31_8B_MODEL_PATH.")
    parser.add_argument("--context-tokens", type=int, default=77000)
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--keep-rates", default="1.0,0.2", help="Comma-separated keep rates.")
    parser.add_argument("--enforce-eager", type=_str2bool, default=True,
                         help="true (default, matches every prior run in this set) or false -- runs the "
                              "EXISTING, unmodified sparse override with CUDA graphs on. block_table/seq_lens "
                              "are already written in-place into vLLM's own persistent graph-safe buffers (see "
                              "the approved plan's Phase 2 finding), so this tests whether that alone is enough "
                              "before any further allocation-reduction changes to vllm_patch/. UNVERIFIED risk, "
                              "not yet confirmed against a live capture: vLLM's capture-time dummy run has no "
                              "request registered in sparse_selection_registry, so _apply_sparse_attention_"
                              "overrides skips entirely during capture -- the graph gets captured in its dense "
                              "shape. Re-run diagnose_h1_metadata.py-style correctness checks (or better, "
                              "validate_sparse_attention.py's needle test) before trusting any timing numbers "
                              "this produces with --enforce-eager false.")
    parser.add_argument("--kv-granularity", type=int, default=16, help="Engine KV-cache block_size.")
    parser.add_argument("--keep-mode", default="keep", choices=["keep", "discard"],
                         help="Echoed into output only -- inert for this single-turn benchmark, see module docstring.")
    parser.add_argument("--reps", type=int, default=3, help="rep 0 is discarded as warmup.")
    parser.add_argument("--profile-decode-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--weights-gib", type=float, default=14.96)
    parser.add_argument("--bandwidth-tbps", type=float, default=2.0)
    parser.add_argument("--flat-threshold-pct", type=float, default=5.0)
    parser.add_argument("--attention-kernel-patterns", default="flash,attn,paged_attention",
                         help="Comma-separated, case-insensitive substrings used to bucket profiler kernels as attention.")
    parser.add_argument("--output-dir", default="./sparse_decode_microbench_out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Shard the model across this many GPUs. Required for a checkpoint "
             "whose weights do not fit one card: Gemma-4-31B is ~62GB in bf16, "
             "which leaves nothing for KV on an 80GB card and fails in "
             "`_check_enough_kv_cache_memory` with 'No available memory for the "
             "cache blocks' before any measurement is taken.")
    args = parser.parse_args()

    model_path = args.model or os.environ.get("LLAMA31_8B_MODEL_PATH")
    if not model_path:
        parser.error("--model or $LLAMA31_8B_MODEL_PATH is required")

    keep_rates = [float(x) for x in args.keep_rates.split(",") if x.strip()]
    attention_patterns = [p.strip().lower() for p in args.attention_kernel_patterns.split(",") if p.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reps < 2:
        parser.error("--reps must be >= 2 (rep 0 is discarded as warmup, at least 1 rep must remain).")

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    # Make the dotted worker_cls path resolvable -- register this module
    # under a stable name importable by vLLM's worker bootstrap, mirroring
    # how `vllm_patch.sparse_target_runner.SparseTargetWorker` is resolved.
    # (Belt-and-suspenders: the classes are already real top-level
    # attributes of whatever module this file loaded as -- see the classes'
    # own definition-site comment -- this additionally covers the case
    # where the EngineCore subprocess looks up "sparse_decode_microbench"
    # by name rather than inheriting sys.modules via fork.)
    sys.modules.setdefault("sparse_decode_microbench", sys.modules[__name__])

    # Text config, not the wrapper -- a natively multimodal checkpoint
    # (Gemma 4) has no `max_position_embeddings` at the top level, and
    # reading it raises AttributeError. See
    # `vllm_patch.model_structure.native_context_length`.
    from vllm_patch.model_structure import (
        has_multimodal_tower,
        native_context_length,
    )

    native_max_model_len = native_context_length(model_path)
    max_model_len = args.context_tokens + args.decode_tokens + 64
    if native_max_model_len is not None:
        max_model_len = min(max_model_len, int(native_max_model_len))

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.kv_granularity,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=max_model_len,
        worker_cls="sparse_decode_microbench.InstrumentedSparseTargetWorker",
        enable_prefix_caching=True,
        async_scheduling=False,
    )
    print(f"[sparse_decode_microbench] constructing engine: {json.dumps({k: v for k, v in llm_kwargs.items() if k != 'model'}, default=str)}")
    t_engine = time.time()

    # Text-only workload, but a natively multimodal checkpoint still reserves
    # an encoder cache and PROFILES it (on Gemma-4-31B: "profiled with 3 video
    # items of the maximum feature size"), eating KV budget for a modality
    # this script never uses. Same treatment predict_scbench.py applies.
    if has_multimodal_tower(model_path):
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}
        print("[sparse_decode_microbench] multimodal checkpoint; zeroing modality limits "
              "(text-only workload)")

    llm = LLM(**llm_kwargs)
    print(f"[sparse_decode_microbench] engine constructed in {time.time() - t_engine:.1f}s")

    llm_engine = llm.llm_engine
    tok = llm.get_tokenizer()
    vllm_config = llm_engine.vllm_config
    cc = vllm_config.compilation_config
    print(f"[sparse_decode_microbench] resolved: enforce_eager={vllm_config.model_config.enforce_eager} "
          f"cudagraph_mode={cc.cudagraph_mode} "
          f"cudagraph_capture_sizes={list(cc.cudagraph_capture_sizes) if cc.cudagraph_capture_sizes else []}")
    actual_block_size = vllm_config.cache_config.block_size
    print(f"[sparse_decode_microbench] requested block_size={args.kv_granularity}, "
          f"engine actually using block_size={actual_block_size}")
    if actual_block_size != args.kv_granularity:
        print("[sparse_decode_microbench] WARNING: engine did not honor the requested "
              "block_size -- kv_granularity results below reflect the ACTUAL block_size above.")

    kv_bytes_per_token = bytes_per_token_kv(vllm_config.model_config.hf_config, vllm_config.model_config.dtype)
    print(f"[sparse_decode_microbench] derived bytes_per_token_kv={kv_bytes_per_token} "
          f"(from hf_config, not hardcoded)")

    context_ids = build_synthetic_context(tok, args.context_tokens, args.seed)
    print(f"[sparse_decode_microbench] built fixed synthetic context: {len(context_ids)} tokens "
          f"(reused via prefix caching for every run below)")

    results = []
    for keep_rate in keep_rates:
        positions = compute_keep_positions(len(context_ids), actual_block_size, keep_rate)
        print(f"\n[sparse_decode_microbench] === keep_rate={keep_rate} "
              f"({len(positions)} blocks registered) ===")

        rep_step_latencies: List[List[float]] = []
        rep_block_counts: List[List[int]] = []
        for rep in range(args.reps):
            request_id = f"microbench-{keep_rate}-{rep}"
            t_rep = time.time()
            latencies = run_decode_only(
                llm_engine, request_id, context_ids, positions, args.decode_tokens,
            )
            block_counts = llm_engine.collective_rpc("pop_block_counts", args=(request_id,))[0]
            print(f"[sparse_decode_microbench] keep={keep_rate} rep={rep}: "
                  f"{len(latencies)} decode steps in {time.time() - t_rep:.1f}s wall "
                  f"({'WARMUP, discarded' if rep == 0 else 'kept'})")
            if rep == 0:
                continue  # warmup, discarded from stats
            rep_step_latencies.append(latencies)
            rep_block_counts.append(block_counts)

        pooled_latencies_ms = [s * 1000.0 for rep_lat in rep_step_latencies for s in rep_lat]
        pooled_block_counts = [b for rep_bc in rep_block_counts for b in rep_bc]
        if not pooled_block_counts:
            print(f"[sparse_decode_microbench] WARNING: no block counts recorded for keep_rate="
                  f"{keep_rate} (degenerate/no-gather path every step?) -- falling back to the "
                  f"dense block count for byte accounting.")
            total_blocks = -(-len(context_ids) // actual_block_size)
            pooled_block_counts = [total_blocks]

        median_block_count = statistics.median(pooled_block_counts)
        bytes_per_step = median_block_count * actual_block_size * kv_bytes_per_token
        roofline_ms = (args.weights_gib * (2 ** 30) + bytes_per_step) / (args.bandwidth_tbps * 1e12) * 1000

        trace_path = output_dir / f"trace_keep_{keep_rate}.json"
        attn_ms, other_ms = run_profiled_decode(
            llm_engine, f"microbench-profile-{keep_rate}", context_ids, positions,
            args.profile_decode_tokens, attention_patterns, trace_path,
        )
        print(f"[sparse_decode_microbench] profiler pass keep={keep_rate}: "
              f"attn={attn_ms:.3f}ms/step other={other_ms:.3f}ms/step "
              f"(patterns={attention_patterns}) -> {trace_path}")

        results.append(dict(
            keep_rate=keep_rate,
            keep_mode=args.keep_mode,  # inert, echoed only -- see module docstring
            kv_granularity_requested=args.kv_granularity,
            kv_granularity_actual=actual_block_size,
            ms_per_tok_median=statistics.median(pooled_latencies_ms),
            ms_per_tok_p90=(statistics.quantiles(pooled_latencies_ms, n=10)[8]
                            if len(pooled_latencies_ms) >= 10 else max(pooled_latencies_ms)),
            kv_bytes_per_step=bytes_per_step,
            median_blocks_loaded=median_block_count,
            attn_ms_per_step=attn_ms,
            other_ms_per_step=other_ms,
            roofline_ms_per_tok=roofline_ms,
            measured_vs_roofline_ratio=statistics.median(pooled_latencies_ms) / roofline_ms,
            trace_path=str(trace_path),
            raw_step_latencies_ms=pooled_latencies_ms,
            raw_block_counts=pooled_block_counts,
        ))

    # ---- flatness check (decision #11) ----
    if len(results) >= 2:
        lo = min(results, key=lambda r: r["keep_rate"])
        hi = max(results, key=lambda r: r["keep_rate"])
        triggers = []
        if hi["kv_bytes_per_step"] > 0:
            byte_delta_pct = abs(lo["kv_bytes_per_step"] - hi["kv_bytes_per_step"]) / hi["kv_bytes_per_step"] * 100
            if byte_delta_pct < args.flat_threshold_pct:
                triggers.append(f"KV bytes/step changed only {byte_delta_pct:.1f}% between "
                                 f"keep={lo['keep_rate']} and keep={hi['keep_rate']}")
        if hi["ms_per_tok_median"] > 0:
            ms_delta_pct = abs(lo["ms_per_tok_median"] - hi["ms_per_tok_median"]) / hi["ms_per_tok_median"] * 100
            if ms_delta_pct < args.flat_threshold_pct:
                triggers.append(f"ms/tok changed only {ms_delta_pct:.1f}% between "
                                 f"keep={lo['keep_rate']} and keep={hi['keep_rate']}")
        if triggers:
            print(f"\n[sparse_decode_microbench] [FLAT -- EVICTION NOT REDUCING KV TRAFFIC] "
                  f"below --flat-threshold-pct={args.flat_threshold_pct}%: " + "; ".join(triggers) +
                  ". This means the kernel is very likely loading the full cache regardless of "
                  "the registered selection -- the block-table gather isn't actually skipping "
                  "blocks. Do not trust a real sweep's speedup numbers until this is understood.")

    # ---- final table ----
    print("\n" + "=" * 100)
    header = f"{'keep_rate':>9} | {'ms/tok(med)':>11} | {'ms/tok(p90)':>11} | {'KV bytes/step':>14} | " \
             f"{'attn ms/step':>12} | {'other ms/step':>13} | {'roofline ms/tok':>16} | {'meas/roofline':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['keep_rate']:>9} | {r['ms_per_tok_median']:>11.3f} | {r['ms_per_tok_p90']:>11.3f} | "
              f"{r['kv_bytes_per_step']:>14,.0f} | {r['attn_ms_per_step']:>12.3f} | "
              f"{r['other_ms_per_step']:>13.3f} | {r['roofline_ms_per_tok']:>16.3f} | "
              f"{r['measured_vs_roofline_ratio']:>13.2f}")
    print("=" * 100)
    print(f"keep_mode={args.keep_mode!r} (echoed only, inert for this single-turn benchmark)")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(dict(args=vars(args), results=results), f, indent=2)
    print(f"\n[sparse_decode_microbench] full results written to {results_path}")


if __name__ == "__main__":
    main()
