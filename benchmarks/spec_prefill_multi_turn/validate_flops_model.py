#!/usr/bin/env python3
"""Validates `flops_model.py` on real hardware, WITHOUT `ncu`.

`ncu` is deliberately not used anywhere here. It has repeatedly failed to
attach on this node (`ncu_kv_bytes_probe.py`'s docstring is a record of that
fight), and the FLOP model doesn't need it: the model has exactly two parts,
and each has a cheaper, more direct check.

Run this once per meaningful config change to `flops_model.py` or to
`sparse_target_runner.py`'s gather. It is NOT part of the experiment matrix
-- `predict_scbench.py` emits the FLOP columns on every run on its own.

## Check A -- the dense term, via `torch.profiler(with_flops=True)`

Wraps `_model_forward` (the same small, deliberately-overridable hook
`gpu_vs_host_timing.py` establishes as the right minimal wrap point, and
which genuinely runs inside the `EngineCore` subprocess) in a profiler
window for a bounded number of forwards, and compares PyTorch's own
`aten::mm`/`addmm`/`bmm` FLOP estimate against
`linear_flops_per_token x tokens_in_that_forward`.

Two things are deliberately NOT in this comparison, and both would look like
a model bug if you forgot them:

- **Attention.** vLLM runs it as a custom op (`torch.ops.vllm.
  unified_attention`), not an aten matmul, so `with_flops` cannot see it and
  reports zero for it. That's why attention is validated separately, by
  Check B, rather than here.
- **`lm_head`.** It lives in `compute_logits`, which `execute_model` calls
  AFTER `_model_forward` returns -- outside this window entirely.

So the expected ratio here is `measured / (tokens x linear_flops_per_token)
~= 1.0`. A ratio well below 1 means the profiler didn't recognize some GEMM
(print the op table before blaming the model); well above 1 means the model
is under-counting the transformer body.

## Check B -- the attention term, by cross-checking its driver

Attention FLOPs are `(a constant pinned by unit test) x (attended length per
decode step)`. So validating the attention term reduces to validating
`pop_attended_lens` -- and that quantity is independently measurable a
second way, by `sparse_decode_microbench.py`'s `pop_block_counts`, which
this script inherits rather than reimplements.

Two independently-written accumulators reading the same gather must satisfy
`block_counts[i] == ceil(attended_lens[i] / block_size)` step for step. An
off-by-one, a missed `gathered is None` dense-fallback step, or a
prefill/decode boundary error breaks it immediately. This is the check that
actually matters: it pins the only input to the whole FLOP model that isn't
either a config constant or a token count the driver already holds.

## Check C -- the falsification bound

Model FLOPs divided by MEASURED GPU forward time (via the `torch.cuda.Event`
pairs inherited from `gpu_vs_host_timing.py`, not wall clock) must come out
below the device's peak. Above-peak throughput is arithmetically impossible,
so it proves the model over-counts -- the whole transposed-dimension /
duplicated-`num_layers` class of bug, caught for free.

Note this uses GPU forward time, so unlike `predict_scbench.py`'s
`achieved_tflops_per_s` (deliberately full-pipeline wall clock, matching
that CSV's other rate columns) the number here is a genuine kernel-level
efficiency figure and should land in a plausible MFU range, not near zero.

Usage:

    python3 validate_flops_model.py --model $LLAMA31_8B_MODEL_PATH \\
        --context-tokens 4000 --decode-tokens 32 --keep-rate 0.4 \\
        --peak-tflops 312
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

from vllm.v1.worker.gpu_worker import Worker

from flops_model import model_flop_config, target_decode_flops
from vllm_patch.model_structure import (
    has_multimodal_tower,
    native_context_length,
)
from gpu_vs_host_timing import GpuTimedSparseTargetGPUModelRunner
from sparse_decode_microbench import (
    InstrumentedSparseTargetWorker,
    build_synthetic_context,
    compute_keep_positions,
    run_decode_only,
)


class FlopValidationGPUModelRunner(GpuTimedSparseTargetGPUModelRunner):
    """Adds Check A's profiler window on top of the existing stack.

    Inherits, rather than reimplements: block counts (Check B's second
    opinion) come from `InstrumentedSparseTargetGPUModelRunner`, GPU forward
    times (Check C's denominator) from `GpuTimedSparseTargetGPUModelRunner`,
    and attended lengths (Check B's first opinion) from the real,
    un-instrumented `SparseTargetGPUModelRunner` -- which is the point: the
    quantity under test is the production one, not a copy of it.
    """

    #: How many forwards to profile. Profiler start/stop is expensive
    #: (tens of ms), so this stays small -- a handful of forwards is plenty
    #: to read a stable FLOPs-per-token ratio, the same "few steps is
    #: enough" reasoning `ncu_kv_bytes_probe.py` uses for its own budget.
    max_profiled_forwards = 6

    def _profiled_accum(self) -> List[Tuple[int, float, bool]]:
        if not hasattr(self, "_flop_profiled"):
            self._flop_profiled: List[Tuple[int, float, bool]] = []
        return self._flop_profiled

    def _model_forward(self, *args, **kwargs):
        accum = self._profiled_accum()
        if len(accum) >= self.max_profiled_forwards or self.input_batch.num_reqs == 0:
            return super()._model_forward(*args, **kwargs)

        import torch

        num_computed = int(self.input_batch.num_computed_tokens_cpu_tensor[0])
        num_prompt = int(self.input_batch.num_prompt_tokens_cpu_tensor[0])
        is_decode = num_computed >= num_prompt

        # Token count for THIS forward, taken from the positions tensor the
        # forward is actually given -- not from the scheduler's own count,
        # which can differ from what the GEMMs see once padding is applied.
        # The comparison has to be against what the matmuls processed.
        positions = kwargs.get("positions")
        if positions is None and len(args) >= 2:
            positions = args[1]
        num_tokens = int(positions.numel()) if positions is not None else 0
        if num_tokens == 0:
            return super()._model_forward(*args, **kwargs)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_flops=True,
        ) as prof:
            out = super()._model_forward(*args, **kwargs)
            torch.cuda.synchronize()

        measured = float(sum(getattr(e, "flops", 0) or 0 for e in prof.key_averages()))
        accum.append((num_tokens, measured, is_decode))
        return out

    def pop_profiled_forwards(self) -> List[Tuple[int, float, bool]]:
        accum = self._profiled_accum()
        popped = list(accum)
        accum.clear()
        return popped


class FlopValidationWorker(InstrumentedSparseTargetWorker):
    def init_device(self) -> None:
        # Same "call Worker.init_device directly, construct only the final
        # runner once" reasoning as InstrumentedSparseTargetWorker's own
        # override.
        Worker.init_device(self)
        self.model_runner = FlopValidationGPUModelRunner(self.vllm_config, self.device)

    def pop_gpu_forward_times_ms(self, request_id: str) -> List[float]:
        return self.model_runner.pop_gpu_forward_times_ms(request_id)

    def pop_attended_lens(self, request_id: str) -> List[int]:
        return self.model_runner.pop_attended_lens(request_id)

    def pop_profiled_forwards(self) -> List[Tuple[int, float, bool]]:
        return self.model_runner.pop_profiled_forwards()


def _report_check_a(profiled, cfg, tolerance_pct: float) -> bool:
    print("\n=== Check A: dense term vs. torch.profiler(with_flops=True) ===")
    if not profiled:
        print("  FAIL: no forwards were profiled at all -- the _model_forward "
              "override never ran (wrong worker_cls?).")
        return False
    print(f"  {'kind':>8} | {'tokens':>8} | {'measured GFLOP':>15} | "
          f"{'predicted GFLOP':>16} | {'ratio':>7}")
    ok = True
    for num_tokens, measured, is_decode in profiled:
        predicted = num_tokens * cfg.linear_flops_per_token
        ratio = measured / predicted if predicted else float("nan")
        print(f"  {'decode' if is_decode else 'prefill':>8} | {num_tokens:>8} | "
              f"{measured / 1e9:>15,.2f} | {predicted / 1e9:>16,.2f} | {ratio:>7.3f}")
        # Judge on the prefill forwards: a 1-token decode forward is a tiny
        # GEMV where profiler overhead and any fixed-size padding dominate
        # the ratio, so it's reported for visibility but not asserted on.
        if not is_decode and abs(ratio - 1.0) * 100 > tolerance_pct:
            ok = False
    if ok:
        print(f"  PASS: every prefill forward within {tolerance_pct:.0f}% of the model.")
    else:
        print(f"  FAIL: a prefill forward is off by more than {tolerance_pct:.0f}%.")
        print("  Before blaming flops_model.py, check WHICH ops the profiler "
              "attributed -- an unrecognized fused GEMM reads exactly like a "
              "model bug. Attention and lm_head are expected to be absent "
              "here; see this module's docstring.")
    return ok


def _report_check_b(attended_lens, block_counts, block_size) -> bool:
    print("\n=== Check B: attended lengths vs. independent block counts ===")
    if not attended_lens:
        print("  FAIL: pop_attended_lens returned nothing -- no decode step "
              "recorded an attended length.")
        return False
    # Both accumulators fire once per (step, request) around the SAME
    # `_compute_gathered_view` call -- the block counter inside it, the
    # attended-length counter immediately after it -- and both cover the
    # `gathered is None` dense-fallback branch. So they must be the same
    # length and index-aligned; anything else means one of them is
    # skipping steps, which is exactly the bug this check exists to catch.
    if len(attended_lens) != len(block_counts):
        print(f"  FAIL: {len(attended_lens)} attended-length samples vs. "
              f"{len(block_counts)} block-count samples -- they are recorded "
              f"around the same call and must match one-for-one.")
        return False
    mismatches = []
    for i, blocks in enumerate(block_counts):
        expected = math.ceil(attended_lens[i] / block_size)
        if blocks != expected:
            mismatches.append((i, attended_lens[i], blocks, expected))
    print(f"  {len(attended_lens)} decode steps, attended length "
          f"min={min(attended_lens)} max={max(attended_lens)} "
          f"mean={sum(attended_lens) / len(attended_lens):.1f} "
          f"(block_size={block_size})")
    if mismatches:
        print(f"  FAIL: {len(mismatches)} step(s) disagree. First few:")
        for i, alen, blocks, expected in mismatches[:5]:
            print(f"    step {i}: attended={alen} -> expected {expected} blocks, "
                  f"block accumulator said {blocks}")
        return False
    print("  PASS: block_counts == ceil(attended_lens / block_size) on every "
          "gathered step -- the FLOP model's only measured input is sound.")
    return True


def _report_check_c(attended_lens, cfg, gpu_ms, peak_tflops) -> bool:
    print("\n=== Check C: falsification bound (model FLOPs / measured GPU time) ===")
    if not attended_lens or not gpu_ms:
        print("  SKIP: need both attended lengths and GPU forward times.")
        return True
    steps = min(len(attended_lens), len(gpu_ms))
    decode_flops = target_decode_flops(cfg, attended_lens[:steps])
    gpu_seconds = sum(gpu_ms[:steps]) / 1000.0
    achieved = decode_flops / gpu_seconds / 1e12 if gpu_seconds > 0 else 0.0
    print(f"  {steps} decode steps: {decode_flops / 1e12:.3f} TFLOP in "
          f"{gpu_seconds * 1000:.1f}ms GPU forward time -> {achieved:.2f} TFLOP/s")
    if peak_tflops:
        print(f"  peak={peak_tflops:.0f} TFLOP/s -> MFU {achieved / peak_tflops:.2%}")
        if achieved > peak_tflops:
            print("  FAIL: above device peak -- arithmetically impossible, so "
                  "the FLOP model is over-counting.")
            return False
        print("  PASS: below device peak.")
        print("  (A LOW MFU here is expected and is not a model bug: decode is "
              "memory-bound, which is the entire premise of the sparse-"
              "attention row -- see sparse_decode_microbench.py's roofline.)")
    else:
        print("  --peak-tflops not given, so the bound can't be checked. "
              "Pass it (e.g. 312 for A100 BF16) to enable this check.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Defaults to $LLAMA31_8B_MODEL_PATH.")
    parser.add_argument("--context-tokens", type=int, default=4000)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--keep-rate", type=float, default=0.4,
                         help="Use a rate < 1.0 so the gather actually runs; "
                              "at 1.0 every step takes the dense-fallback path "
                              "and Check B has no gathered steps to compare.")
    parser.add_argument("--kv-granularity", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--peak-tflops", type=float, default=None,
                         help="Device peak dense TFLOP/s, enables Check C's bound.")
    parser.add_argument("--tolerance-pct", type=float, default=15.0,
                         help="Check A pass band around a ratio of 1.0.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_path = args.model or os.environ.get("LLAMA31_8B_MODEL_PATH")
    if not model_path:
        parser.error("--model or $LLAMA31_8B_MODEL_PATH is required")

    from transformers import AutoConfig
    from vllm import LLM

    sys.modules.setdefault("validate_flops_model", sys.modules[__name__])

    # Text config, not the wrapper -- a natively multimodal checkpoint
    # (Gemma 4) has no `max_position_embeddings` at the top level. See
    # `vllm_patch.model_structure.native_context_length`.
    native_max_model_len = native_context_length(model_path)
    max_model_len = args.context_tokens + args.decode_tokens + 64
    if native_max_model_len is not None:
        max_model_len = min(max_model_len, int(native_max_model_len))

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.kv_granularity,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=max_model_len,
        worker_cls="validate_flops_model.FlopValidationWorker",
        enable_prefix_caching=True,
        async_scheduling=False,
    )
    # Text-only check; a multimodal checkpoint would otherwise reserve and
    # profile an encoder cache out of this script's deliberately small
    # budget. Same treatment as predict_scbench.py / proposer.py.
    if has_multimodal_tower(model_path):
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}
        print("[validate_flops_model] multimodal checkpoint; zeroing modality "
              "limits")
    llm = LLM(**llm_kwargs)
    llm_engine = llm.llm_engine
    tok = llm.get_tokenizer()
    block_size = llm_engine.vllm_config.cache_config.block_size
    cfg = model_flop_config(llm_engine.vllm_config.model_config.hf_config)
    print(f"[validate_flops_model] {json.dumps(cfg.__dict__)}")
    print(f"[validate_flops_model] linear_flops_per_token="
          f"{cfg.linear_flops_per_token / 1e9:.2f} GFLOP, lm_head="
          f"{cfg.lm_head_flops / 1e9:.2f} GFLOP, block_size={block_size}")

    context_ids = build_synthetic_context(tok, args.context_tokens, args.seed)
    positions = compute_keep_positions(len(context_ids), block_size, args.keep_rate)
    print(f"[validate_flops_model] context={len(context_ids)} tokens, "
          f"keep_rate={args.keep_rate} -> {len(positions)} selected blocks")

    request_id = "flop-validation"
    run_decode_only(llm_engine, request_id, context_ids, positions, args.decode_tokens)

    attended_lens = llm_engine.collective_rpc("pop_attended_lens", args=(request_id,))[0]
    block_counts = llm_engine.collective_rpc("pop_block_counts", args=(request_id,))[0]
    gpu_ms = llm_engine.collective_rpc("pop_gpu_forward_times_ms", args=(request_id,))[0]
    profiled = llm_engine.collective_rpc("pop_profiled_forwards")[0]

    results = [
        _report_check_a(profiled, cfg, args.tolerance_pct),
        _report_check_b(attended_lens, block_counts, block_size),
        _report_check_c(attended_lens, cfg, gpu_ms, args.peak_tflops),
    ]
    print("\n=== summary ===")
    print(f"  {sum(results)}/{len(results)} checks passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
