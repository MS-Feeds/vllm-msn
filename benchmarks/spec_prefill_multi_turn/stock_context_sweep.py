#!/usr/bin/env python3
"""Stock-vLLM context-length sweep, graphs ON (`enforce_eager=False` fixed)
-- the missing half of the "one test that pins it" pairing with
`stock_vllm_control.py`. Same methodology as `dense_context_sweep.py`
(Part A: keep=1.0-equivalent dense decode, context 4k->77k, 512 decode
tokens, prefix-shared contexts, prefill/TTFT excluded from timing) but
through the COMPLETELY UNMODIFIED runner -- no `worker_cls`, no
`vllm_patch` import beyond two vllm_patch-independent utility functions --
with CUDA graphs enabled instead of `enforce_eager=True`.

## What this answers

`dense_context_sweep.py` already confirmed decode IS memory-bandwidth-driven
in this build under `enforce_eager=True` (ms/tok grows with context as the
KV-bandwidth roofline predicts). This script re-runs that same sweep with
graphs ON, on the genuinely stock path, to check the predicted slope still
holds (+4.69ms from 4k->77k per the user's own arithmetic, i.e. roughly
12.3->17.0 ms/tok) -- if it does, the roofline model and this whole
benchmark set's measurement methodology are validated independent of the
sparse-attention mechanism entirely, and any remaining sparse-vs-stock gap
is attributable to the sparse override itself, not a measurement artifact.

Deliberately reuses `stock_vllm_control.py`'s `run_decode_only_stock` (not
`sparse_decode_microbench.py`'s `run_decode_only`, which depends on
sparse-only RPCs that don't exist here) and `sparse_decode_microbench.py`'s
`build_synthetic_context`/`bytes_per_token_kv` (both vllm_patch-independent
pure utilities). KV bytes/step here are computed analytically
(`context_tokens * bytes_per_token_kv`), not measured via the
block-count-instrumentation RPC `dense_context_sweep.py` uses -- there is no
sparse gather here at all, so full-context KV read every step is exact by
construction, not something that needs measuring.

Usage:
    python3 stock_context_sweep.py --model $LLAMA31_8B_MODEL_PATH \\
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

from sparse_decode_microbench import build_synthetic_context, bytes_per_token_kv
from stock_vllm_control import run_decode_only_stock


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
                              "attention is flagged as not traffic-driven even in stock/graphs-on.")
    parser.add_argument("--output-dir", default="./stock_context_sweep_out")
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

    native_max_model_len = AutoConfig.from_pretrained(model_path, trust_remote_code=True).max_position_embeddings
    max_model_len = max_context + args.decode_tokens + 64
    if native_max_model_len is not None:
        max_model_len = min(max_model_len, int(native_max_model_len))

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=False,  # fixed -- graphs ON, see module docstring
        disable_log_stats=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.kv_granularity,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=max_model_len,
        # No worker_cls -- genuinely stock, same reasoning as
        # stock_vllm_control.py.
    )
    print(f"[stock_context_sweep] constructing STOCK engine (enforce_eager=False): "
          f"{json.dumps({k: v for k, v in llm_kwargs.items() if k != 'model'})}")
    t_engine = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[stock_context_sweep] engine constructed in {time.time() - t_engine:.1f}s "
          f"(includes CUDA-graph capture warmup)")

    llm_engine = llm.llm_engine
    tok = llm.get_tokenizer()
    vllm_config = llm_engine.vllm_config
    cc = vllm_config.compilation_config
    print(f"[stock_context_sweep] resolved: enforce_eager={vllm_config.model_config.enforce_eager} "
          f"cudagraph_mode={cc.cudagraph_mode} "
          f"cudagraph_capture_sizes={list(cc.cudagraph_capture_sizes) if cc.cudagraph_capture_sizes else []}")

    kv_bytes_per_token = bytes_per_token_kv(vllm_config.model_config.hf_config, vllm_config.model_config.dtype)
    print(f"[stock_context_sweep] derived bytes_per_token_kv={kv_bytes_per_token} (from hf_config)")

    # One shared max-length context, sliced into prefixes -- same
    # no-content-confound reasoning as dense_context_sweep.py.
    base_context_ids = build_synthetic_context(tok, max_context, args.seed)

    results = []
    for context_tokens in context_lengths:
        context_ids = base_context_ids[:context_tokens]
        print(f"\n[stock_context_sweep] === context_tokens={context_tokens} ===")

        rep_latencies: List[List[float]] = []
        for rep in range(args.reps):
            request_id = f"stocksweep-{context_tokens}-{rep}"
            t_rep = time.time()
            latencies = run_decode_only_stock(llm_engine, request_id, context_ids, args.decode_tokens)
            print(f"[stock_context_sweep] context={context_tokens} rep={rep}: "
                  f"{len(latencies)} decode steps in {time.time() - t_rep:.1f}s wall "
                  f"({'WARMUP, discarded' if rep == 0 else 'kept'})")
            if rep == 0:
                continue
            rep_latencies.append([s * 1000.0 for s in latencies])

        pooled_ms = [v for rep in rep_latencies for v in rep]
        kv_bytes = context_tokens * kv_bytes_per_token  # exact -- always full-context, no gather here
        attn_floor_ms = kv_bytes / (args.bandwidth_tbps * 1e12) * 1000

        results.append(dict(
            context_tokens=context_tokens,
            kv_bytes=kv_bytes,
            ms_per_tok_median=statistics.median(pooled_ms),
            ms_per_tok_p90=(statistics.quantiles(pooled_ms, n=10)[8] if len(pooled_ms) >= 10 else max(pooled_ms)),
            attn_floor_ms=attn_floor_ms,
            raw_ms=pooled_ms,
        ))

    # ---- table ----
    print("\n" + "=" * 90)
    header = f"{'context':>9} | {'KV bytes':>16} | {'ms/tok(med)':>11} | " \
             f"{'ms/tok(p90)':>11} | {'attn floor ms':>13} | {'meas/floor':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        ratio = r["ms_per_tok_median"] / r["attn_floor_ms"] if r["attn_floor_ms"] > 0 else float("nan")
        print(f"{r['context_tokens']:>9} | {r['kv_bytes']:>16,.0f} | {r['ms_per_tok_median']:>11.3f} | "
              f"{r['ms_per_tok_p90']:>11.3f} | {r['attn_floor_ms']:>13.3f} | {ratio:>10.2f}")
    print("=" * 90)

    lo, hi = results[0], results[-1]
    measured_delta_ms = hi["ms_per_tok_median"] - lo["ms_per_tok_median"]
    expected_delta_ms = (hi["kv_bytes"] - lo["kv_bytes"]) / (args.bandwidth_tbps * 1e12) * 1000
    print(f"\nms/tok({hi['context_tokens']}) - ms/tok({lo['context_tokens']}) = "
          f"{hi['ms_per_tok_median']:.3f} - {lo['ms_per_tok_median']:.3f} = {measured_delta_ms:+.3f} ms")
    print(f"KV-bandwidth-only expected delta for the same two contexts: {expected_delta_ms:+.3f} ms "
          f"(at {args.bandwidth_tbps} TB/s)")

    if measured_delta_ms < args.delta_threshold_ms:
        print(
            f"\n[ATTENTION TIME NOT TRAFFIC-DRIVEN, EVEN STOCK/GRAPHS-ON] measured delta "
            f"({measured_delta_ms:+.3f} ms) is below --delta-threshold-ms={args.delta_threshold_ms} despite an "
            f"expected KV-bandwidth-bound delta of {expected_delta_ms:+.3f} ms. If this is flat even here (no "
            f"sparse override, no vllm_patch, graphs on), the roofline premise or measurement methodology needs "
            f"re-examining independent of anything in vllm_patch/."
        )
    else:
        print(
            f"\n[TRAFFIC-DRIVEN] measured delta ({measured_delta_ms:+.3f} ms) is at or above "
            f"--delta-threshold-ms={args.delta_threshold_ms}, in the direction a bandwidth-bound attention kernel "
            f"predicts (expected {expected_delta_ms:+.3f} ms) -- roofline model and measurement methodology "
            f"validated on the stock/graphs-on path."
        )

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(dict(args=vars(args), results=results,
                        measured_delta_ms=measured_delta_ms, expected_delta_ms=expected_delta_ms), f, indent=2)
    print(f"\n[stock_context_sweep] full results written to {results_path}")


if __name__ == "__main__":
    main()
