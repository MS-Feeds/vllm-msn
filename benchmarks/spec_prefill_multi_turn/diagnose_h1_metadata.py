#!/usr/bin/env python3
"""H1 diagnostic: "the kernel doesn't honor the shortened metadata -- block
table is short, but a separate max_seq_len/context_lens/seq_lens field still
says ~full-context and drives the loop bound."

Cheap check requested: dump `block_table.shape`, the count of distinct block
IDs actually present in the (single) request's block-table row, and every
length-like field on the per-layer `AttentionMetadata` object, for the first
few DECODE steps of both keep=1.0 and keep=0.2 -- taken right after
`SparseTargetGPUModelRunner._apply_sparse_attention_overrides` has already
patched `.block_table`/`.seq_lens` (see `sparse_target_runner.py`'s own
`_build_attention_metadata` override: it calls the override BEFORE
returning, so by the time this script's hook runs, `attn_metadata` already
reflects whatever the real forward pass is about to see). This is the
closest Python-level proxy available for "at the attention call site" --
the kernel itself is a compiled op, not independently instrumentable here.

If ANY length-like field other than `block_table`/`seq_lens` still reads
close to the full context length while `block_table`'s own distinct-id count
is small, that's H1 confirmed: the gather patches the wrong/incomplete set
of fields and some other field (found here) is what actually drives the
kernel's read volume.

Usage:
    python3 diagnose_h1_metadata.py --model $LLAMA31_8B_MODEL_PATH \\
        --context-tokens 4096 --decode-steps 3 --keep-rates 1.0,0.2 --kv-granularity 16
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

# Top-level, not deferred -- worker_cls resolution needs these as real
# module attributes at import time in the EngineCore subprocess (see
# sparse_decode_microbench.py's own comment on this for the failure mode
# when they're nested inside a function instead).
from vllm.v1.worker.gpu_worker import Worker
from vllm_patch.sparse_target_runner import SparseTargetGPUModelRunner, SparseTargetWorker

# Reuse the existing, already-verified synthetic-context/selection builders
# rather than re-implementing them.
from sparse_decode_microbench import build_synthetic_context, compute_keep_positions

# Length-like field names to surface -- deliberately broad (substring match)
# so this doesn't silently miss the offending field by guessing its exact
# name wrong.
_LENGTH_LIKE_SUBSTRINGS = (
    "seq_len", "context_len", "max_len", "max_seq", "num_", "cu_seq",
    "slot_mapping", "block_table", "query_len",
)
_MAX_DIAG_STEPS_PER_REQUEST = 3


class DiagnosticSparseTargetGPUModelRunner(SparseTargetGPUModelRunner):
    def _h1_step_counts(self) -> Dict[str, int]:
        if not hasattr(self, "_h1_step_counts_dict"):
            self._h1_step_counts_dict: Dict[str, int] = {}
        return self._h1_step_counts_dict

    def _build_attention_metadata(self, *args, **kwargs):
        # super() here is SparseTargetGPUModelRunner's own override, which
        # already calls _apply_sparse_attention_overrides internally before
        # returning -- so attn_metadata below is POST-patch, exactly what
        # the model forward / attention kernel is about to receive.
        attn_metadata, spec_decode_meta = super()._build_attention_metadata(*args, **kwargs)
        self._dump_h1_diagnostics(attn_metadata)
        return attn_metadata, spec_decode_meta

    def _dump_h1_diagnostics(self, attn_metadata) -> None:
        if not attn_metadata:
            return
        if self.input_batch.num_reqs == 0:
            return
        req_id = self.input_batch.req_ids[0]
        num_computed = int(self.input_batch.num_computed_tokens_cpu_tensor[0])
        num_prompt = int(self.input_batch.num_prompt_tokens_cpu_tensor[0])
        if num_computed < num_prompt:
            return  # prefill step, not decode -- nothing gathered here, skip

        count = self._h1_step_counts().get(req_id, 0)
        if count >= _MAX_DIAG_STEPS_PER_REQUEST:
            return
        self._h1_step_counts()[req_id] = count + 1

        import torch

        layer_name, layer_metadata = next(iter(attn_metadata.items()))
        print(f"\n[H1-diag] req={req_id} decode_step={count} layer={layer_name!r} "
              f"num_computed={num_computed} num_prompt={num_prompt}")

        block_table = getattr(layer_metadata, "block_table", None)
        if block_table is not None:
            row = block_table[0]
            unique_vals = torch.unique(row)
            print(f"  block_table.shape={tuple(block_table.shape)} row.shape={tuple(row.shape)} "
                  f"distinct_ids_in_row={unique_vals.numel()} "
                  f"(includes any padding/NULL_BLOCK_ID entry -- interpret as an upper bound "
                  f"on real occupied blocks)")
        else:
            print("  block_table: <field not present on this layer's metadata>")

        for f in dataclasses.fields(layer_metadata):
            name_l = f.name.lower()
            if f.name == "block_table" or not any(k in name_l for k in _LENGTH_LIKE_SUBSTRINGS):
                continue
            val = getattr(layer_metadata, f.name)
            if torch.is_tensor(val):
                flat = val.flatten()
                preview = flat[:8].tolist()
                suffix = "..." if flat.numel() > 8 else ""
                print(f"  {f.name}: tensor shape={tuple(val.shape)} dtype={val.dtype} "
                      f"first8={preview}{suffix}")
            else:
                print(f"  {f.name}: {val!r}")


class DiagnosticSparseTargetWorker(SparseTargetWorker):
    def init_device(self) -> None:
        Worker.init_device(self)
        self.model_runner = DiagnosticSparseTargetGPUModelRunner(self.vllm_config, self.device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Defaults to $LLAMA31_8B_MODEL_PATH.")
    parser.add_argument("--context-tokens", type=int, default=4096,
                         help="Deliberately small -- H1 doesn't need the full 77k context to reveal itself.")
    parser.add_argument("--decode-steps", type=int, default=_MAX_DIAG_STEPS_PER_REQUEST)
    parser.add_argument("--keep-rates", default="1.0,0.2")
    parser.add_argument("--kv-granularity", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_path = args.model or os.environ.get("LLAMA31_8B_MODEL_PATH")
    if not model_path:
        parser.error("--model or $LLAMA31_8B_MODEL_PATH is required")

    keep_rates = [float(x) for x in args.keep_rates.split(",") if x.strip()]

    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    sys.modules.setdefault("diagnose_h1_metadata", sys.modules[__name__])

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
        worker_cls="diagnose_h1_metadata.DiagnosticSparseTargetWorker",
        enable_prefix_caching=True,
        async_scheduling=False,
    )
    llm_engine = llm.llm_engine
    tok = llm.get_tokenizer()
    actual_block_size = llm_engine.vllm_config.cache_config.block_size
    print(f"[diagnose_h1] engine block_size={actual_block_size} (requested {args.kv_granularity})")

    context_ids = build_synthetic_context(tok, args.context_tokens, args.seed)

    for keep_rate in keep_rates:
        print(f"\n{'=' * 70}\n[diagnose_h1] keep_rate={keep_rate}\n{'=' * 70}")
        positions = compute_keep_positions(len(context_ids), actual_block_size, keep_rate)
        request_id = f"h1diag-{keep_rate}"
        llm_engine.collective_rpc("register_sparse_selection", args=(request_id, positions))
        sampling_params = SamplingParams(
            max_tokens=args.decode_steps, min_tokens=args.decode_steps, ignore_eos=True, temperature=0.0
        )
        prompt = TokensPrompt(prompt_token_ids=context_ids)
        real_id = llm_engine.add_request(request_id, prompt, sampling_params)
        assert real_id == request_id

        while llm_engine.has_unfinished_requests():
            llm_engine.step()

        llm_engine.collective_rpc("discard_sparse_selection", args=(request_id,))

    print("\n[diagnose_h1] done -- compare the printed fields above between keep_rate=1.0 and "
          "keep_rate=0.2. If block_table/distinct_ids_in_row shrinks for keep=0.2 but some OTHER "
          "length-like field does not, that field is the H1 bug.")


if __name__ == "__main__":
    main()
