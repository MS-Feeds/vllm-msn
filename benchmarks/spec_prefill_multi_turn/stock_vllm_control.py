#!/usr/bin/env python3
"""Stock-vLLM control: same context length and decode-token count as the
rest of this benchmark set, but through the COMPLETELY UNMODIFIED runner --
no `worker_cls` override, no `vllm_patch` import, no sparse-selection RPCs.
Answers: is the ~22ms/tok figure `sparse_decode_microbench.py` measured
specific to the sparse-attention override machinery (Python-side per-step
metadata patching preventing CUDA-graph capture, or adding its own
overhead), or does stock vLLM show the same number at this context length
regardless -- in which case the overhead is elsewhere entirely and worth an
nsys/py-spy look at the host thread, not further changes to
`sparse_target_runner.py`.

Deliberately does NOT reuse `sparse_decode_microbench.py`'s `run_decode_only`
-- that function calls `register_sparse_selection`/`discard_sparse_selection`/
`pop_nvtx_range` RPCs that only exist on `SparseTargetWorker`; a genuinely
stock run has none of that, so this script has its own minimal decode-only
loop (same prefill/TTFT-exclusion boundary and `torch.cuda.synchronize()`
methodology, just without any sparse-specific calls).

Run ONCE per `--enforce-eager` value (mirrors `ncu_kv_bytes_probe.py`'s
one-config-per-process convention -- keeps each run's engine construction
and CUDA-graph capture warmup unambiguous, and avoids constructing two full
engines' worth of GPU memory in one process):

    python3 stock_vllm_control.py --model $LLAMA31_8B_MODEL_PATH \\
        --context-tokens 77000 --decode-tokens 512 --reps 3 --enforce-eager true

    python3 stock_vllm_control.py --model $LLAMA31_8B_MODEL_PATH \\
        --context-tokens 77000 --decode-tokens 512 --reps 3 --enforce-eager false

Compare the two runs' printed ms/tok. If `--enforce-eager false` lands near
the ~10-13ms bandwidth roofline and `--enforce-eager true` lands near ~22ms
(matching `sparse_decode_microbench.py`'s own eager-mode number), the
overhead is specific to eager-mode dispatch, not to the sparse override --
and the fix path becomes making the override's per-step metadata patch
CUDA-graph-compatible (write into pre-allocated persistent tensors the
graph was captured against, rather than only relying on eager execution).
If `--enforce-eager false` is ALSO near ~22ms here, cudagraph capture isn't
actually engaging for this context length/config even in the fully stock
path, or the overhead is elsewhere -- worth an nsys/py-spy look at the host
thread next, not more changes to sparse_target_runner.py.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

# Only pure-utility, vllm_patch-independent helpers -- see module docstring
# on why this script otherwise avoids vllm_patch entirely.
from sparse_decode_microbench import build_synthetic_context


def run_decode_only_stock(llm_engine, request_id: str, context_ids: List[int], decode_tokens: int) -> List[float]:
    """Same boundary-detection/synchronization methodology as
    `sparse_decode_microbench.py`'s `run_decode_only`, minus every
    sparse-specific RPC call -- see module docstring for why."""
    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

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
            if has_token:
                first_token_seen = True
            continue
        step_latencies.append(t1 - t0)
        if out.finished:
            break

    return step_latencies


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
    parser.add_argument("--reps", type=int, default=3, help="rep 0 is discarded as warmup.")
    parser.add_argument("--enforce-eager", type=_str2bool, required=True,
                         help="true or false -- run this script once per value, see module docstring.")
    parser.add_argument("--kv-granularity", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cudagraph-mode", default="FULL_DECODE_ONLY",
        choices=["FULL_DECODE_ONLY", "FULL", "PIECEWISE",
                 "FULL_AND_PIECEWISE", "NONE"],
        help="Only used when --enforce-eager false. FULL_DECODE_ONLY and "
             "FULL work without Dynamo; the PIECEWISE variants need it and "
             "will fail under TORCHDYNAMO_DISABLE=1.")
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
    if args.reps < 2:
        parser.error("--reps must be >= 2 (rep 0 is discarded as warmup).")

    from transformers import AutoConfig
    from vllm import LLM

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
        # No worker_cls -- genuinely stock GPUModelRunner/Worker, no
        # vllm_patch involvement at all. No enable_prefix_caching/
        # async_scheduling overrides either -- irrelevant here (no
        # resumable sessions, no sparse mechanism), left at vLLM's own
        # defaults.
    )

    # `--enforce-eager false` ALONE is not enough to get CUDA graphs in this
    # pipeline's environment, and the failure is a hard crash rather than a
    # silent fallback: it takes `@support_torch_compile`'s AOT path
    # (`vllm/compilation/wrapper.py`), which needs Dynamo, and this pipeline
    # pins `TORCHDYNAMO_DISABLE=1` because Inductor's process pool cannot
    # spawn inside daemonic MultiprocExecutor workers. Observed on
    # Gemma-4-31B as `RuntimeError: aot_compile is not supported by the
    # current configuration`.
    #
    # The way out is stated by vLLM's own docs (`config/compilation.py`):
    # "the cudagraph logic is generally orthogonal to the compilation logic.
    # While piecewise cudagraphs require piecewise compilation ... full
    # [cudagraphs do not]". So pin BOTH: `mode=NONE` (no torch.compile at
    # all, hence no Dynamo) and an explicitly FULL cudagraph mode.
    #
    # `FULL_DECODE_ONLY` is the default here rather than `FULL` because only
    # decode batches are being measured, and it leaves prefill eager --
    # matching what the sparse path would need, since its prefill gather
    # varies per chunk and must not be captured. Verified reachable: the
    # empty-`splitting_ops` validation downgrades PIECEWISE and
    # FULL_AND_PIECEWISE but leaves FULL/FULL_DECODE_ONLY alone (and its own
    # warning text recommends exactly this), and TRITON_ATTN -- which
    # Gemma 4 forces -- declares `AttentionCGSupport.ALWAYS`.
    if not args.enforce_eager:
        from vllm.config.compilation import (
            CompilationConfig,
            CompilationMode,
            CUDAGraphMode,
        )

        llm_kwargs["compilation_config"] = CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode[args.cudagraph_mode],
        )
        print(f"[stock_vllm_control] cudagraph_mode={args.cudagraph_mode} with "
              f"compilation mode=NONE (no Dynamo -- see comment above)")

    # Text-only workload, but a natively multimodal checkpoint still reserves
    # an encoder cache and PROFILES it. On Gemma-4-31B the log reads
    # "profiled with 3 video items of the maximum feature size" and the
    # reservation eats KV budget for a modality this script never uses.
    # Same treatment predict_scbench.py applies for the same reason.
    if has_multimodal_tower(model_path):
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}
        print("[stock_vllm_control] multimodal checkpoint; zeroing modality "
              "limits (text-only workload)")
    print(f"[stock_vllm_control] constructing STOCK engine (no worker_cls): "
          f"{json.dumps({k: v for k, v in llm_kwargs.items() if k != 'model'})}")
    t_engine = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[stock_vllm_control] engine constructed in {time.time() - t_engine:.1f}s "
          f"(enforce_eager={args.enforce_eager} run includes any CUDA-graph capture warmup here)")

    llm_engine = llm.llm_engine
    tok = llm.get_tokenizer()
    vllm_config = llm_engine.vllm_config
    cc = vllm_config.compilation_config
    print(f"[stock_vllm_control] resolved: enforce_eager={vllm_config.model_config.enforce_eager} "
          f"cudagraph_mode={cc.cudagraph_mode} "
          f"cudagraph_capture_sizes={list(cc.cudagraph_capture_sizes) if cc.cudagraph_capture_sizes else []}")

    context_ids = build_synthetic_context(tok, args.context_tokens, args.seed)

    rep_latencies: List[List[float]] = []
    for rep in range(args.reps):
        request_id = f"stockcontrol-{args.enforce_eager}-{rep}"
        t_rep = time.time()
        latencies = run_decode_only_stock(llm_engine, request_id, context_ids, args.decode_tokens)
        print(f"[stock_vllm_control] rep={rep}: {len(latencies)} decode steps in "
              f"{time.time() - t_rep:.1f}s wall ({'WARMUP, discarded' if rep == 0 else 'kept'})")
        if rep == 0:
            continue
        rep_latencies.append([s * 1000.0 for s in latencies])

    pooled_ms = [v for rep in rep_latencies for v in rep]
    median_ms = statistics.median(pooled_ms)
    p90_ms = statistics.quantiles(pooled_ms, n=10)[8] if len(pooled_ms) >= 10 else max(pooled_ms)

    print(f"\n{'=' * 60}")
    print(f"enforce_eager={args.enforce_eager}  context_tokens={args.context_tokens}  "
          f"decode_tokens={args.decode_tokens}")
    print(f"ms/tok median: {median_ms:.3f}   p90: {p90_ms:.3f}")
    print(f"{'=' * 60}")

    out_path = Path(f"stock_vllm_control_eager{args.enforce_eager}.json")
    with open(out_path, "w") as f:
        json.dump(dict(args=vars(args), ms_per_tok_median=median_ms, ms_per_tok_p90=p90_ms,
                        raw_ms=pooled_ms), f, indent=2)
    print(f"[stock_vllm_control] full results written to {out_path}")


if __name__ == "__main__":
    main()
