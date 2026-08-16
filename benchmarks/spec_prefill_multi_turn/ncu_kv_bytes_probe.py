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

Uses the REAL, un-instrumented `SparseTargetWorker` (not
`sparse_decode_microbench.py`'s `InstrumentedSparseTargetWorker`) -- no
Python-level counting needed here, ncu measures hardware truth directly.

## Why NVTX scoping, not --launch-skip/--launch-count

Prefill for a 77k-token context is chunked (`max_num_batched_tokens`), so
the number of attention-kernel launches before the first genuine decode
step depends on chunk count and isn't worth computing by hand. Instead this
script pushes an NVTX range ONLY once the first generated token is observed
(i.e. only around genuine, single-query decode steps -- same
prefill/TTFT-exclusion boundary `sparse_decode_microbench.py`'s timed loop
uses), so `ncu --nvtx --nvtx-include "decode_kv_probe/keep=<rate>/"`
captures exactly the kernels launched during real decode steps and nothing
from prefill.

Usage (run ONCE per keep rate, each under its own `ncu` invocation):

    ncu --nvtx --nvtx-include "decode_kv_probe/keep=1.0/" \\
        --kernel-name regex:".*[Ff]lash.*|.*[Aa]ttn.*" \\
        --metrics dram__bytes_read.sum,dram__bytes_write.sum \\
        --target-processes all -o ncu_keep_1.0 \\
        python3 ncu_kv_bytes_probe.py --model $LLAMA31_8B_MODEL_PATH \\
            --context-tokens 77000 --decode-steps 5 --keep-rate 1.0 --kv-granularity 16

    ncu --nvtx --nvtx-include "decode_kv_probe/keep=0.2/" \\
        --kernel-name regex:".*[Ff]lash.*|.*[Aa]ttn.*" \\
        --metrics dram__bytes_read.sum,dram__bytes_write.sum \\
        --target-processes all -o ncu_keep_0.2 \\
        python3 ncu_kv_bytes_probe.py --model $LLAMA31_8B_MODEL_PATH \\
            --context-tokens 77000 --decode-steps 5 --keep-rate 0.2 --kv-granularity 16

--target-processes all is required -- vLLM's actual CUDA work runs in the
EngineCore subprocess, not this driver process.

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

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

# Reuse the already-verified context/selection builders rather than
# reimplementing them -- see sparse_decode_microbench.py.
from sparse_decode_microbench import build_synthetic_context, compute_keep_positions


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
    import torch

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
        worker_cls="vllm_patch.sparse_target_runner.SparseTargetWorker",
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

    nvtx_label = f"decode_kv_probe/keep={args.keep_rate}/"
    first_token_seen = False
    nvtx_pushed = False
    print("[ncu_probe] running prefill (NOT NVTX-scoped -- ncu should exclude this via --nvtx-include)")
    try:
        while llm_engine.has_unfinished_requests():
            outputs = llm_engine.step()
            out = next((o for o in outputs if o.request_id == request_id), None)
            if out is None:
                continue
            has_token = len(out.outputs[0].token_ids) > 0
            if not first_token_seen:
                if has_token:
                    first_token_seen = True
                    print(f"[ncu_probe] prefill done -- entering NVTX range {nvtx_label!r} for decode steps")
                    torch.cuda.nvtx.range_push(nvtx_label)
                    nvtx_pushed = True
                continue
            if out.finished:
                break
    finally:
        if nvtx_pushed:
            torch.cuda.nvtx.range_pop()

    llm_engine.collective_rpc("discard_sparse_selection", args=(request_id,))
    print("[ncu_probe] done")


if __name__ == "__main__":
    main()
