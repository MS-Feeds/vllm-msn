#!/usr/bin/env python3
"""Part A -- dense (keep=1.0) context-length sweep. Tests the roofline
PREMISE itself, independent of the sparse-gather mechanism entirely (keep=1.0
is always the degenerate, no-op dense path -- see
`compute_sparse_gather_view_incremental`'s "covers everything" return-None
case) and independent of both bugs already fixed in
`vllm_patch/sparse_target_runner.py` (the missing `max_seq_len` patch, and
the 32x-redundant `.item()` sync -- neither one is exercised at keep=1.0).

## What this answers

If decode-time attention is genuinely memory-bandwidth-bound on KV-cache
reads, ms/tok should grow roughly linearly with context length (more
resident KV = more bytes read per decode step = more time, at a fixed
model/weights floor that doesn't change with context length). This sweeps
context length at a FIXED keep_rate=1.0 and 512 decode tokens, and compares
the measured ms/tok(max context) - ms/tok(min context) delta against the
KV-bandwidth-only delta a memory-bound attention kernel implies.

If the measured delta comes back near zero (or negative) despite the KV
byte delta between the smallest and largest context being large, that's
direct evidence attention time in THIS build isn't traffic-driven at all --
independent of any sparse/eviction mechanism, no implementation of KV
pruning could buy decode latency here, because decode latency isn't
KV-bandwidth-bound to begin with. That would make chasing further
sparse-attention-specific bugs (stale `scheduler_metadata` etc.) moot until
THIS is understood, since fixing the gather mechanism further can't fix a
problem that was never about the gather.

## Reused, not reimplemented

`build_synthetic_context`, `compute_keep_positions`, `bytes_per_token_kv`,
`run_decode_only`, and the `InstrumentedSparseTargetGPUModelRunner`/
`InstrumentedSparseTargetWorker` pair (for the same directly-measured,
not-inferred block-count -> KV-bytes accounting used elsewhere) are all
imported from `sparse_decode_microbench.py` rather than duplicated here --
same `worker_cls="sparse_decode_microbench.InstrumentedSparseTargetWorker"`
dotted path, which resolves correctly here because this script imports that
module under its own genuine name (not the `__main__`-aliasing trick
`sparse_decode_microbench.py` itself needs when run directly).

Context lengths are built as PREFIXES of one shared max-length synthetic
context (not independently regenerated per length) -- guarantees the
smaller contexts are byte-identical prefixes of the larger ones, so this
isn't confounded by different random content per config, and lets vLLM's
prefix cache make each successive (larger) run's prefill cheaper than a
from-scratch prefill would be (prefill/TTFT is excluded from all timing
either way, so this is a wall-clock convenience only, not a correctness
requirement).

Usage:
    python3 dense_context_sweep.py --model $LLAMA31_8B_MODEL_PATH \\
        --context-tokens-list 4000,16000,32000,64000,77000 \\
        --decode-tokens 512 --reps 3 --kv-granularity 16
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

from sparse_decode_microbench import (
    build_synthetic_context,
    bytes_per_token_kv,
    compute_keep_positions,
    run_decode_only,
)

# Fixed, not a CLI flag -- Part A is explicitly keep=1.0 only, testing the
# roofline premise itself, not the sparse-gather mechanism. See module
# docstring.
KEEP_RATE = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Defaults to $LLAMA31_8B_MODEL_PATH.")
    parser.add_argument("--context-tokens-list", default="4000,16000,32000,64000,77000")
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--reps", type=int, default=3, help="rep 0 is discarded as warmup.")
    parser.add_argument("--kv-granularity", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--bandwidth-tbps", type=float, default=2.0)
    parser.add_argument("--delta-threshold-ms", type=float, default=1.0,
                         help="If the measured max-vs-min-context ms/tok delta is below this, "
                              "attention is flagged as not traffic-driven in this build.")
    parser.add_argument("--output-dir", default="./dense_context_sweep_out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_path = args.model or os.environ.get("LLAMA31_8B_MODEL_PATH")
    if not model_path:
        parser.error("--model or $LLAMA31_8B_MODEL_PATH is required")

    if args.reps < 2:
        parser.error("--reps must be >= 2 (rep 0 is discarded as warmup, at least 1 rep must remain).")

    context_lengths = sorted(int(x) for x in args.context_tokens_list.split(",") if x.strip())
    max_context = context_lengths[-1]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    # Text config, not the wrapper -- a natively multimodal checkpoint
    # (Gemma 4) has no `max_position_embeddings` at the top level, and
    # reading it raises AttributeError. See
    # `vllm_patch.model_structure.native_context_length`.
    from vllm_patch.model_structure import native_context_length

    native_max_model_len = native_context_length(model_path)
    max_model_len = max_context + args.decode_tokens + 64
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
        worker_cls="sparse_decode_microbench.InstrumentedSparseTargetWorker",
        enable_prefix_caching=True,
        async_scheduling=False,
    )
    print(f"[dense_context_sweep] constructing engine: "
          f"{json.dumps({k: v for k, v in llm_kwargs.items() if k != 'model'})}")
    t_engine = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[dense_context_sweep] engine constructed in {time.time() - t_engine:.1f}s")

    llm_engine = llm.llm_engine
    tok = llm.get_tokenizer()
    vllm_config = llm_engine.vllm_config
    actual_block_size = vllm_config.cache_config.block_size
    print(f"[dense_context_sweep] block_size={actual_block_size} (requested {args.kv_granularity})")

    kv_bytes_per_token = bytes_per_token_kv(vllm_config.model_config.hf_config, vllm_config.model_config.dtype)
    print(f"[dense_context_sweep] derived bytes_per_token_kv={kv_bytes_per_token} (from hf_config)")

    # One shared max-length context, sliced into prefixes -- see module
    # docstring for why (not independently regenerated per length).
    base_context_ids = build_synthetic_context(tok, max_context, args.seed)

    results = []
    for context_tokens in context_lengths:
        context_ids = base_context_ids[:context_tokens]
        positions = compute_keep_positions(len(context_ids), actual_block_size, KEEP_RATE)
        print(f"\n[dense_context_sweep] === context_tokens={context_tokens} "
              f"({len(positions)} blocks registered, keep_rate={KEEP_RATE}) ===")

        rep_step_latencies: List[List[float]] = []
        rep_block_counts: List[List[int]] = []
        for rep in range(args.reps):
            request_id = f"densesweep-{context_tokens}-{rep}"
            t_rep = time.time()
            latencies = run_decode_only(
                llm_engine, request_id, context_ids, positions, args.decode_tokens,
            )
            block_counts = llm_engine.collective_rpc("pop_block_counts", args=(request_id,))[0]
            print(f"[dense_context_sweep] context={context_tokens} rep={rep}: "
                  f"{len(latencies)} decode steps in {time.time() - t_rep:.1f}s wall "
                  f"({'WARMUP, discarded' if rep == 0 else 'kept'})")
            if rep == 0:
                continue
            rep_step_latencies.append(latencies)
            rep_block_counts.append(block_counts)

        pooled_latencies_ms = [s * 1000.0 for rep_lat in rep_step_latencies for s in rep_lat]
        pooled_block_counts = [b for rep_bc in rep_block_counts for b in rep_bc]
        if not pooled_block_counts:
            # keep_rate=1.0 always hits the degenerate no-op path (see
            # module docstring) -- gathered is always None, so
            # InstrumentedSparseTargetGPUModelRunner's dense-fallback
            # ceil-div still records a value every step. Empty here would
            # mean something unexpected broke; fall back to the same
            # ceil-div rather than silently reporting zero.
            total_blocks = -(-len(context_ids) // actual_block_size)
            pooled_block_counts = [total_blocks]

        median_block_count = statistics.median(pooled_block_counts)
        kv_bytes_measured = median_block_count * actual_block_size * kv_bytes_per_token
        attn_floor_ms = kv_bytes_measured / (args.bandwidth_tbps * 1e12) * 1000

        results.append(dict(
            context_tokens=context_tokens,
            kv_bytes_measured=kv_bytes_measured,
            median_blocks_loaded=median_block_count,
            ms_per_tok_median=statistics.median(pooled_latencies_ms),
            ms_per_tok_p90=(statistics.quantiles(pooled_latencies_ms, n=10)[8]
                            if len(pooled_latencies_ms) >= 10 else max(pooled_latencies_ms)),
            attn_floor_ms=attn_floor_ms,
            raw_step_latencies_ms=pooled_latencies_ms,
        ))

    # ---- table ----
    print("\n" + "=" * 100)
    header = f"{'context':>9} | {'KV bytes (measured)':>20} | {'ms/tok(med)':>11} | " \
             f"{'ms/tok(p90)':>11} | {'attn floor ms':>13} | {'meas/floor':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        ratio = r["ms_per_tok_median"] / r["attn_floor_ms"] if r["attn_floor_ms"] > 0 else float("nan")
        print(f"{r['context_tokens']:>9} | {r['kv_bytes_measured']:>20,.0f} | "
              f"{r['ms_per_tok_median']:>11.3f} | {r['ms_per_tok_p90']:>11.3f} | "
              f"{r['attn_floor_ms']:>13.3f} | {ratio:>10.2f}")
    print("=" * 100)

    # ---- the one number that matters ----
    lo, hi = results[0], results[-1]
    measured_delta_ms = hi["ms_per_tok_median"] - lo["ms_per_tok_median"]
    expected_delta_ms = (hi["kv_bytes_measured"] - lo["kv_bytes_measured"]) / (args.bandwidth_tbps * 1e12) * 1000
    print(f"\nms/tok({hi['context_tokens']}) - ms/tok({lo['context_tokens']}) = "
          f"{hi['ms_per_tok_median']:.3f} - {lo['ms_per_tok_median']:.3f} = {measured_delta_ms:+.3f} ms")
    print(f"KV-bandwidth-only expected delta for the same two contexts: {expected_delta_ms:+.3f} ms "
          f"(at {args.bandwidth_tbps} TB/s)")

    if measured_delta_ms < args.delta_threshold_ms:
        print(
            f"\n[ATTENTION TIME NOT TRAFFIC-DRIVEN] measured delta ({measured_delta_ms:+.3f} ms) is below "
            f"--delta-threshold-ms={args.delta_threshold_ms} despite an expected KV-bandwidth-bound delta of "
            f"{expected_delta_ms:+.3f} ms across a {lo['kv_bytes_measured'] / (2**30):.2f}->"
            f"{hi['kv_bytes_measured'] / (2**30):.2f} GiB KV read-volume range. The roofline premise does not "
            f"hold in this build: decode-time attention latency is not scaling with KV bytes read, so no "
            f"KV-eviction/sparsity implementation can buy decode latency here regardless of correctness. "
            f"Chasing further sparse-attention-specific fixes (e.g. scheduler_metadata staleness) is moot "
            f"until THIS is understood -- something other than KV-cache memory bandwidth is dominating "
            f"decode-step wall-clock time in this build."
        )
    else:
        print(
            f"\n[TRAFFIC-DRIVEN] measured delta ({measured_delta_ms:+.3f} ms) is at or above "
            f"--delta-threshold-ms={args.delta_threshold_ms} and in the direction a bandwidth-bound attention "
            f"kernel predicts (expected {expected_delta_ms:+.3f} ms) -- the roofline premise holds in this "
            f"build. A real sparse-attention implementation should be able to buy decode latency here."
        )

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(dict(args=vars(args), results=results,
                        measured_delta_ms=measured_delta_ms, expected_delta_ms=expected_delta_ms), f, indent=2)
    print(f"\n[dense_context_sweep] full results written to {results_path}")


if __name__ == "__main__":
    main()
