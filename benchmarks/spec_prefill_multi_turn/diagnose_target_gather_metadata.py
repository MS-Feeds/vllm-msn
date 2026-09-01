#!/usr/bin/env python3
"""Target-side counterpart to `diagnose_speculator_selection.py`. That
script proved (on a real, reliably-failing conversation) that the
SPECULATOR's selection is coherent, sensible prose -- ruling out
`scoring.py`/`proposer.py`. This script takes that SAME selection (turn 0
only, for now -- see module docstring below), registers it against the
REAL target (`SparseTargetWorker`), decodes a few real tokens, and directly
compares the EXPECTED block count (computed independently, in pure Python,
from the translated positions) against what `sparse_target_runner.py`'s
gather actually reports patching -- closing the loop on whether the
corruption is introduced in position translation / the gather mechanism
itself, or somewhere even further downstream (the attention kernel call,
`scheduler_metadata` staleness flagged in that module's own docstring,
etc.) if the metadata already matches.

## Deliberately turn-0 only, for now

Same reasoning as `diagnose_speculator_selection.py`: turn 0's position
translation is a trivial constant shift (`LedgerToTargetPositionMap` has
only its initial breakpoint, no `add_wrapper` calls yet), so it's the
simplest possible case to test the target-side gather against a KNOWN-GOOD
selection, with the fewest moving parts. Extending to turn > 0 would need
the same real-prior-turn-output replay `diagnose_speculator_selection.py`
supports via `--predictions-file`, plus actually resuming a session rather
than a one-shot request -- not built here to keep this diagnostic focused;
add it the same way if turn 0 turns out clean and a later turn is needed.

## What "expected block count" means here

Computed directly in this script, independently of anything
`sparse_target_runner.py` does: `{p // block_size for p in
translated_positions}`, i.e. the same trivial computation
`block_indices_from_positions` performs. If the ACTUAL count
`InstrumentedSparseTargetGPUModelRunner` reports (reusing the exact same
instrumentation `sparse_decode_microbench.py` already uses and this
session already trusts) matches this expected count, the gather mechanism
is faithfully patching what it was told to -- the bug, if the target's
output is still garbage, is further downstream. If it doesn't match,
that's the bug, located precisely.

Usage:
    python3 diagnose_target_gather_metadata.py \\
        --conversation-id scbench_qa_eng-53 \\
        --keep-percentage 0.8 --granularity 32 --decode-tokens 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

# Reused, not reimplemented -- see sparse_decode_microbench.py for what
# this instrumentation actually does (block count derived from the
# gathered seq_len, correctly excludes the max_seq_len/32x-sync bugs
# already fixed this session). Only the Worker needs importing here --
# worker_cls resolution needs it as a real top-level attribute of this
# module (see that module's own comment on why), and it references
# InstrumentedSparseTargetGPUModelRunner internally already.
from sparse_decode_microbench import InstrumentedSparseTargetWorker  # noqa: F401 -- used via worker_cls string below


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=Path(__file__).parent / "datasets" / "scbench_samples.jsonl")
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--keep-percentage", type=float, default=0.8)
    parser.add_argument("--granularity", default="32", choices=["token", "16", "32", "64"])
    parser.add_argument("--keep-mode", default="keep", choices=["keep", "discard"])
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--target-model", default=None, help="Defaults to $LLAMA31_8B_MODEL_PATH.")
    parser.add_argument("--speculator-model", default=None, help="Defaults to $LLAMA32_1B_MODEL_PATH.")
    parser.add_argument("--speculator-device", default="cuda:1")
    parser.add_argument("--speculator-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--speculator-max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--target-gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--target-max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--kv-granularity", type=int, default=16, help="Engine KV-cache block_size.")
    args = parser.parse_args()

    target_model = args.target_model or os.environ.get("LLAMA31_8B_MODEL_PATH")
    speculator_model = args.speculator_model or os.environ.get("LLAMA32_1B_MODEL_PATH")
    if not target_model:
        parser.error("--target-model or $LLAMA31_8B_MODEL_PATH is required")
    if not speculator_model:
        parser.error("--speculator-model or $LLAMA32_1B_MODEL_PATH is required")

    import torch
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from predict_scbench import (
        GRANULARITIES,
        LOOK_AHEAD_CNT,
        POOL_KERNEL_SIZE,
        chat_wrapper_pieces,
        load_conversations,
        render_turn_query,
    )
    from vllm_patch.config import SpecConfig
    from vllm_patch.model_structure import native_context_length
    from vllm_patch.conversation_state import ConversationState
    from vllm_patch.pruner import compute_pruned_turn
    from vllm_patch.proposer import SpecPrefillProposer

    sys.modules.setdefault("diagnose_target_gather_metadata", sys.modules[__name__])

    # ---- Step 1: reproduce the speculator's selection exactly (same code
    # as diagnose_speculator_selection.py) ----
    conversations = load_conversations(args.samples, max_keep=-1)
    matches = [c for c in conversations if c["id"] == args.conversation_id]
    if not matches:
        parser.error(f"conversation_id {args.conversation_id!r} not found in {args.samples}")
    conv = matches[0]

    spec_tok = AutoTokenizer.from_pretrained(speculator_model, trust_remote_code=True)
    gran_kwargs = GRANULARITIES[args.granularity]
    spec_config = SpecConfig(
        keep_strategy="percentage",
        keep_kwargs={**gran_kwargs, "percentage": args.keep_percentage},
        look_ahead_cnt=LOOK_AHEAD_CNT,
        pool_kernel_size=POOL_KERNEL_SIZE,
        keep_mode=args.keep_mode,
    )

    # Text config, not the wrapper -- a natively multimodal checkpoint
    # (Gemma 4) has no `max_position_embeddings` on its top-level config.
    # See `model_structure.native_context_length`.
    speculator_native_max_model_len = native_context_length(speculator_model)
    speculator_max_num_batched_tokens = args.speculator_max_num_batched_tokens
    if speculator_native_max_model_len is not None and speculator_max_num_batched_tokens > speculator_native_max_model_len:
        speculator_max_num_batched_tokens = int(speculator_native_max_model_len)
    speculator_max_model_len = min(
        speculator_max_num_batched_tokens + 1 + LOOK_AHEAD_CNT,
        int(speculator_native_max_model_len) if speculator_native_max_model_len is not None
        else speculator_max_num_batched_tokens + 1 + LOOK_AHEAD_CNT,
    )

    print(f"[diagnose_target_gather] constructing speculator ({speculator_model})...")
    proposer = SpecPrefillProposer(
        speculator_model_path=speculator_model,
        device=torch.device(args.speculator_device),
        gpu_memory_utilization=args.speculator_gpu_memory_utilization,
        max_num_batched_tokens=speculator_max_num_batched_tokens,
        max_model_len=speculator_max_model_len,
    )

    context_ids = spec_tok.encode(conv["context"], add_special_tokens=False)
    state = ConversationState(conv["id"], context_ids, args.keep_mode)
    query_ids = render_turn_query(spec_tok, 0, conv["turns"][0])
    result = compute_pruned_turn(proposer, spec_config, state, query_ids)
    print(f"[diagnose_target_gather] speculator selection: kept {len(result.kept_positions)}/"
          f"{result.orig_len} ledger positions ({len(result.kept_positions) / result.orig_len * 100:.1f}%)")

    # ---- Step 2: translate to target-stream positions -- turn 0, so this
    # is just the trivial constant-offset case (no add_wrapper calls yet) ----
    tgt_tok = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    chat_before, chat_after = chat_wrapper_pieces(tgt_tok)
    chat_before_ids = tgt_tok.encode(chat_before, add_special_tokens=False)
    chat_after_ids = tgt_tok.encode(chat_after, add_special_tokens=False)
    # NOTE: context/query were tokenized with the SPECULATOR's tokenizer
    # above (matching predict_scbench.py's own real behavior -- the
    # speculator and target are re-tokenized independently from the same
    # source TEXT in that pipeline, not sharing token ids across models).
    # Re-tokenize with the TARGET's own tokenizer here for what actually
    # gets submitted to it, and translate the speculator's POSITIONS
    # (index-based, not token-id-based) directly -- positions are only
    # meaningful within one consistent tokenization, so this assumes (as
    # predict_scbench.py itself does) that both tokenizers segment this
    # text into the SAME number of tokens. Flagged here explicitly since
    # it's an assumption, not verified independently by this script.
    target_context_ids = tgt_tok.encode(conv["context"], add_special_tokens=False)
    target_query_ids = render_turn_query(tgt_tok, 0, conv["turns"][0])
    if len(target_context_ids) != len(context_ids) or len(target_query_ids) != len(query_ids):
        print(f"[diagnose_target_gather] WARNING: speculator/target tokenization length mismatch -- "
              f"context {len(context_ids)} vs {len(target_context_ids)}, query {len(query_ids)} vs "
              f"{len(target_query_ids)}. Position translation below assumes these match; if they "
              f"don't, that mismatch alone could explain corrupted attention -- note this and treat "
              f"results below with caution.")

    initial_offset = len(chat_before_ids)
    # Union with the chat-template wrapper spans, exactly as
    # `predict_scbench.py::run_sparse_attention` now does (see
    # `LedgerToTargetPositionMap.wrapper_target_spans`'s docstring): a bare
    # `p + initial_offset` can never reach target positions
    # `[0, len(chat_before_ids))`, so registering only those left the
    # attention sink out of the selection -- a confirmed cause of the
    # repetition-loop output this script exists to diagnose. Turn 0 only
    # here (this script is single-turn by construction), so the wrapper
    # spans are exactly chat_before at the front and chat_after at the end.
    wrapper_positions = (
        list(range(initial_offset))
        + list(range(initial_offset + result.orig_len,
                     initial_offset + result.orig_len + len(chat_after_ids)))
    )
    translated_positions = sorted(
        set(p + initial_offset for p in result.kept_positions) | set(wrapper_positions)
    )
    block_size = args.kv_granularity
    expected_block_indices = {p // block_size for p in translated_positions}
    print(f"[diagnose_target_gather] translated {len(translated_positions)} positions to target-stream "
          f"numbering (turn 0 constant offset +{initial_offset}); expected distinct blocks at "
          f"block_size={block_size}: {len(expected_block_indices)}")

    delta_ids = chat_before_ids + target_context_ids + target_query_ids + chat_after_ids
    print(f"[diagnose_target_gather] delta_ids (full turn-0 prompt) length: {len(delta_ids)}")

    # ---- Step 3: construct the REAL target and register this exact
    # selection ----
    native_max_model_len = native_context_length(target_model)
    max_model_len = len(delta_ids) + args.decode_tokens + 64
    if native_max_model_len is not None:
        max_model_len = min(max_model_len, int(native_max_model_len))

    print(f"[diagnose_target_gather] constructing target ({target_model})...")
    llm = LLM(
        model=target_model,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=False,
        gpu_memory_utilization=args.target_gpu_memory_utilization,
        block_size=block_size,
        max_num_batched_tokens=args.target_max_num_batched_tokens,
        max_model_len=max_model_len,
        worker_cls="diagnose_target_gather_metadata.InstrumentedSparseTargetWorker",
        enable_prefix_caching=True,
        async_scheduling=False,
    )
    llm_engine = llm.llm_engine
    actual_block_size = llm_engine.vllm_config.cache_config.block_size
    if actual_block_size != block_size:
        print(f"[diagnose_target_gather] WARNING: engine block_size={actual_block_size} != requested "
              f"{block_size} -- expected_block_indices above was computed against the REQUESTED size "
              f"and is now invalid; re-run with --kv-granularity {actual_block_size}.")

    request_id = f"diag-{args.conversation_id}-turn0"
    llm_engine.collective_rpc("register_sparse_selection", args=(request_id, translated_positions))

    sampling_params = SamplingParams(
        max_tokens=args.decode_tokens, min_tokens=args.decode_tokens, ignore_eos=True, temperature=0.0
    )
    prompt = TokensPrompt(prompt_token_ids=delta_ids)
    real_id = llm_engine.add_request(request_id, prompt, sampling_params)
    assert real_id == request_id

    # ---- Step 4: drive decode, inspect actual gathered block counts ----
    first_token_seen = False
    last_output = None
    while llm_engine.has_unfinished_requests():
        outputs = llm_engine.step()
        out = next((o for o in outputs if o.request_id == request_id), None)
        if out is None:
            continue
        last_output = out
        if len(out.outputs[0].token_ids) > 0:
            first_token_seen = True
        if out.finished:
            break

    block_counts = llm_engine.collective_rpc("pop_block_counts", args=(request_id,))[0]
    llm_engine.collective_rpc("discard_sparse_selection", args=(request_id,))

    print(f"\n{'=' * 90}")
    print(f"expected distinct blocks (computed independently, pure Python): {len(expected_block_indices)}")
    print(f"actual blocks gathered per decode step (from the real kernel-facing metadata): {block_counts}")
    print(f"{'=' * 90}")

    expected = len(expected_block_indices)
    # The force-keep tail makes the count GROW during a turn, by design: each
    # decode step must be able to attend to the tokens this turn has already
    # generated, so `tail_blocks = range(boundary_block, dynamic_end)` picks
    # up another block every time generation crosses a block boundary. A
    # prompt ending k tokens short of a boundary therefore gives k steps at
    # `expected`, then `expected + 1`, and so on.
    #
    # An earlier version of this check required every step to equal
    # `expected` exactly and reported that growth as "[MISMATCH] ... this is
    # the bug", which is precisely backwards -- without the growth the model
    # could not see its own output. What actually indicates a bug is the
    # FIRST step disagreeing (the selection itself is wrong), the count
    # DECREASING (blocks vanishing mid-turn), or growth faster than decode
    # can cross boundaries.
    max_tail_growth = -(-args.decode_tokens // block_size) + 1
    first_ok = bool(block_counts) and block_counts[0] == expected
    monotonic = all(b <= c for b, c in zip(block_counts, block_counts[1:]))
    growth = (max(block_counts) - block_counts[0]) if block_counts else 0

    if first_ok and monotonic and growth <= max_tail_growth:
        print(f"\n[MATCH] the gather patched exactly the expected {expected} blocks on the "
              f"first decode step, growing to {max(block_counts)} as the force-keep tail "
              f"picked up newly generated tokens (at most {max_tail_growth} block(s) for "
              f"{args.decode_tokens} decode tokens at block_size={block_size} -- expected, "
              f"not drift).\n"
              f"So the corruption, if the generated text below is still garbage, is NOT in "
              f"position translation or block selection. Look further downstream: the "
              f"attention kernel call itself, or backend-specific scheduling state that the "
              f"gather does not invalidate (sparse_target_runner.py flags this as unverified "
              f"for any backend other than FlashAttention).")
    else:
        reason = (
            f"first step gathered {block_counts[0] if block_counts else 'nothing'}, "
            f"expected {expected}" if not first_ok else
            "block count DECREASED during the turn" if not monotonic else
            f"count grew by {growth}, more than the {max_tail_growth} block(s) the "
            f"force-keep tail can account for"
        )
        print(f"\n[MISMATCH] {reason}. Actual per-step counts: {block_counts}. "
              f"This points at position translation or block selection, not further "
              f"downstream.")

    if last_output is not None:
        generated_text = tgt_tok.decode(last_output.outputs[0].token_ids, skip_special_tokens=True)
        print(f"\ngenerated text for these {args.decode_tokens} forced tokens: {generated_text!r}")


if __name__ == "__main__":
    main()
