#!/usr/bin/env python3
"""The sliding-window gate: are token selections being decided by layers that
could never have attended to the tokens they picked?

Migrated here from `../spec_prefill/verify_sliding_window_hypothesis.py`, which
was written against the single-turn pipeline and never executed. Same question,
measured through THIS pipeline's machinery instead.

## The hypothesis

`scoring.compute_attention_score` computes `Q @ K^T` over the ENTIRE context
for EVERY layer. On an interleaved sliding-window model most layers can never
attend beyond a 512-1024 token window during real inference, so their score
for a distant position is a number the model never computes. Because
`aggregate_attention_score` collapses with `max` over (layer, head), ONE such
layer is enough to decide a token's importance by itself.

Gemma-4-E2B-it is 35 layers, 28 of them sliding -- so 4 in 5 votes come from
layers that can only see 512 tokens back.

## What it reports

The share of winning (layer, head) votes cast by a sliding layer scoring a
position further away than its own window, computed separately for:

  - the positions the selection actually KEPT, and
  - a random sample of the positions it PRUNED AWAY.

A bare rate means little; the comparison is the evidence. If KEPT is
meaningfully higher than PRUNED-AWAY, selection is being driven by scores the
model itself never computes, and restricting scoring to the full-attention
layers (`--score-layers global_only`) is the fix. If both are similar, the
hypothesis does not hold for this workload regardless of whether the
underlying gap is real in general.

## Why it lives here now, and what changed in the move

1. **It shares production's aggregation instead of copying it.** The original
   reimplemented the softmax -> pool -> max chain so it could keep
   `torch.max`'s discarded `.indices`. A copy drifts from what it copied, and
   a gate that silently measures something other than production is worse than
   no gate. `aggregate_attention_score` now takes a `winning_layers` out-list,
   so there is exactly one implementation of those steps.

2. **It reads a real engine's KV cache.** The single-turn pipeline builds a
   hand-rolled dummy cache, one `torch.randn` tensor per layer. A
   cross-layer-KV-sharing layer never writes its own cache, so on E2B -- where
   20 of 35 layers are KV-shared -- more than half the layers would have been
   scored against uninitialized memory. Noise winning under `max` is a second,
   unrelated source of exactly the signal this gate attributes to sliding
   layers, which would have made the measurement unreadable.

3. **It runs on SCBench, not LongBench-v2** -- this pipeline's own multi-turn
   workload, at the context lengths its rows are actually measured at.

4. **Per-layer windows, read off the live `Attention` modules.** The original
   inferred "is this a sliding layer" from `head_dim < max(head_dims)`,
   exploiting Gemma-4-26B-A4B's 256/512 split. That proxy reports every layer
   as full-attention on any checkpoint with uniform head dims -- a 0% rate for
   both samples, reading as REFUTED when the hypothesis was never tested.

Turn 0 only: the gate is about which layer wins a vote, and turn 0 is where
the context's KV is first computed, so it needs no history replay and no prior
predictions file (unlike `diagnose_retrieval_heads.py`).

Usage:
    python3 diagnose_sliding_window_votes.py \
        --speculator-model "$GEMMA4_E2B_MODEL_PATH" \
        --keep-percentage 0.3 --config scbench_kv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _pct(phantom: int, total: int) -> str:
    return f"{phantom / total:.1%}" if total else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=Path,
                        default=Path(__file__).parent / "datasets" / "scbench_samples.jsonl")
    parser.add_argument("--config", default="scbench_kv",
                        help="SCBench config to draw conversations from. kv is "
                             "where the SPARSE degradation actually lives, so it "
                             "is the default here too.")
    parser.add_argument("--max-conversations", type=int, default=10)
    parser.add_argument("--keep-percentage", type=float, default=0.3,
                        help="The rate to measure at. Lower rates make the "
                             "selection more discriminating and so make a "
                             "phantom-vote effect easier to see; 0.3 matches "
                             "the single-turn run this gate was written for.")
    parser.add_argument("--granularity", default="32", choices=["token", "16", "32", "64"])
    parser.add_argument("--score-layers", default=None,
                        help="Pass 'global_only' to measure the FIX rather than "
                             "the problem: it should drive the phantom rate to "
                             "0%% by construction, which is a useful check that "
                             "this harness measures what it claims.")
    parser.add_argument("--mask-sliding-window", action="store_true",
                        help="Mask each sliding layer to its own window before "
                             "the scoring softmax, instead of dropping those "
                             "layers (--score-layers global_only) or letting "
                             "them score the whole context unmasked (the "
                             "default). The third mode: it keeps a sliding "
                             "layer's real opinion about what is inside its "
                             "window. Expect a ~0%% phantom rate, same as "
                             "global_only -- masked positions cannot be won -- "
                             "so the gate cannot separate these two. Grading "
                             "is what tells them apart. Not EXACTLY 0: "
                             "avg_pool1d runs after the softmax, so in-window "
                             "mass smears up to pool_kernel_size//2 positions "
                             "past the boundary and can win there. ~0.015%% on "
                             "E2B at pool 13; exactly 0 with pooling off.")
    parser.add_argument("--compare-sample-size", type=int, default=200,
                        help="How many pruned-away positions to sample for the "
                             "comparison baseline (all of them would dominate "
                             "the runtime at SCBench context lengths).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speculator-model", default=None)
    parser.add_argument("--speculator-device", default="cuda:0")
    # 0.5, not the 0.3 the rest of this pipeline uses for its speculator.
    # That default was set for Llama-3.2-1B; Gemma-4-E2B-it loads its full
    # 5.1B parameters (the "2.3B effective" figure counts activated
    # parameters, not resident weights) plus vision/audio towers, so it needs
    # roughly 5x the weight memory before any KV cache exists.
    parser.add_argument("--speculator-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--speculator-max-num-batched-tokens", type=int, default=131072,
                        help="Conversation-length ceiling: a turn longer than "
                             "this is SKIPPED rather than served. Kept high on "
                             "purpose -- lowering it silently drops the longest "
                             "conversations instead of measuring them.")
    parser.add_argument("--speculator-prefill-chunk-tokens", type=int, default=32768,
                        help="Per-step prefill batch size, set SEPARATELY from "
                             "the ceiling above -- the same split "
                             "`--scorer-prefill-chunk-tokens` makes in "
                             "diagnose_retrieval_heads.py, and for the same "
                             "reason. vLLM sizes its KV pool from a PROFILED "
                             "activation peak, and that peak scales with the "
                             "per-step batch: at 131072 tokens a single MLP "
                             "intermediate is gigabytes, with several live at "
                             "once, which is what drives 'Available KV cache "
                             "memory' negative. Costs nothing in accuracy here "
                             "-- `end_capture` filters captured queries by "
                             "shape, so any number of leading prefill chunks "
                             "is harmless.")
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    from predict_scbench import (
        GRANULARITIES,
        LOOK_AHEAD_CNT,
        POOL_KERNEL_SIZE,
        render_turn_query,
    )
    from vllm_patch.config import SpecConfig
    from vllm_patch.conversation_state import ConversationState
    from vllm_patch.proposer import SpecPrefillProposer

    speculator_model = (
        args.speculator_model
        or os.environ.get("GEMMA4_E2B_MODEL_PATH")
        or os.environ.get("LLAMA32_1B_MODEL_PATH")
    )
    if not speculator_model:
        parser.error(
            "--speculator-model, $GEMMA4_E2B_MODEL_PATH or "
            "$LLAMA32_1B_MODEL_PATH is required"
        )

    conversations = []
    with open(args.samples, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") == args.config:
                conversations.append(row)
            if len(conversations) >= args.max_conversations:
                break
    if not conversations:
        parser.error(f"no {args.config!r} conversations in {args.samples}")

    tok = AutoTokenizer.from_pretrained(speculator_model, trust_remote_code=True)
    spec_config = SpecConfig(
        keep_strategy="percentage",
        keep_kwargs={**GRANULARITIES[args.granularity],
                     "percentage": args.keep_percentage},
        look_ahead_cnt=LOOK_AHEAD_CNT,
        pool_kernel_size=POOL_KERNEL_SIZE,
        score_layers=args.score_layers,
        keep_mode="keep",
    )

    hf_config = AutoConfig.from_pretrained(speculator_model, trust_remote_code=True)
    native_len = hf_config.get_text_config().max_position_embeddings
    max_batched = args.speculator_max_num_batched_tokens
    if native_len is not None:
        max_batched = min(max_batched, int(native_len))
    max_model_len = max_batched + 1 + LOOK_AHEAD_CNT
    if native_len is not None:
        max_model_len = min(max_model_len, int(native_len))

    import torch

    chunk_tokens = min(args.speculator_prefill_chunk_tokens, max_batched)
    print(f"[gate] speculator: {speculator_model}")
    print(f"[gate] gpu_memory_utilization={args.speculator_gpu_memory_utilization}, "
          f"prefill chunk={chunk_tokens}, max_position_embeddings={native_len}, "
          f"max_model_len={max_model_len}")
    proposer = SpecPrefillProposer(
        speculator_model_path=speculator_model,
        device=torch.device(args.speculator_device),
        gpu_memory_utilization=args.speculator_gpu_memory_utilization,
        max_num_batched_tokens=chunk_tokens,
        enable_chunked_prefill=True,
        max_model_len=max_model_len,
    )

    # What one request can actually occupy, leaving room for the lookahead
    # tokens this turn will generate on top of its prompt.
    servable = max_model_len - spec_config.look_ahead_cnt - 1
    print(f"[gate] one turn can occupy {servable} tokens; longer contexts are "
          f"truncated to fit, not skipped")

    totals = {"kept_phantom": 0, "kept_total": 0, "kept_expected": 0.0,
              "pruned_phantom": 0, "pruned_total": 0, "pruned_expected": 0.0}
    measured = 0
    skipped = 0
    truncated = 0
    layer_windows = None

    for conv in conversations:
        conv_id = str(conv["id"])
        context_ids = tok.encode(conv["context"], add_special_tokens=False)
        state = ConversationState(conv["id"], context_ids, "keep")
        query_ids = render_turn_query(tok, 0, conv["turns"][0])
        candidate_pool, force_keep_query = state.begin_turn(query_ids)
        full_ids = [tid for tid, _ in candidate_pool + force_keep_query]

        # TRUNCATE rather than skip. This gate asks which LAYER won each
        # vote, not whether the task was answered, and that is visible in any
        # context long enough to put positions outside a 512-1024 token
        # window. Refusing every conversation that exceeds the speculator's
        # own servable length -- SCBench contexts reach ~124k, and a small
        # scorer's `max_position_embeddings` can be well below that --
        # measures nothing at all, which is strictly worse than measuring a
        # prefix. Reported per conversation so a truncated run is never
        # mistaken for a full-length one.
        if len(full_ids) > servable:
            room = servable - len(query_ids)
            if room <= 0:
                print(f"[gate] {conv_id}: skipped, this turn's own query is "
                      f"{len(query_ids)} tokens and the scorer can serve only "
                      f"{servable}")
                skipped += 1
                continue
            # Keep the PREFIX: it preserves the attention-sink tokens at
            # position 0, whose absence is its own well-documented source of
            # degeneration in this pipeline, and keeps the far-from-query
            # positions the measurement depends on.
            state = ConversationState(conv["id"], context_ids[:room], "keep")
            candidate_pool, force_keep_query = state.begin_turn(query_ids)
            full_ids = [tid for tid, _ in candidate_pool + force_keep_query]
            truncated += 1
            print(f"[gate] {conv_id}: context truncated to {room} tokens "
                  f"(scorer serves {servable})")

        diag = proposer.run_turn_and_sliding_window_diagnostics(
            conversation_salt=conv_id,
            turn_idx=0,
            full_sequence_token_ids=full_ids,
            look_ahead_cnt=spec_config.look_ahead_cnt,
            pool_kernel_size=spec_config.pool_kernel_size,
            keep_kwargs=spec_config.keep_kwargs,
            score_layers=spec_config.score_layers,
            compare_sample_size=args.compare_sample_size,
            seed=args.seed,
            mask_sliding_window=args.mask_sliding_window,
        )
        proposer.discard_conversation(conv_id)

        if diag is None:
            print(f"[gate] {conv_id}: skipped, no lookahead step completed "
                  f"(EOS on the speculator's first candidate token)")
            skipped += 1
            continue

        measured += 1
        layer_windows = diag["layer_windows"]
        for k in totals:
            totals[k] += diag[k]
        print(f"[gate] {conv_id}: {diag['orig_len']} tokens, kept "
              f"{diag['num_kept']} ({diag['num_kept'] / diag['orig_len']:.1%}), "
              f"{diag['look_ahead_cnt']} lookahead steps -- phantom KEPT "
              f"{_pct(diag['kept_phantom'], diag['kept_total'])} vs PRUNED "
              f"{_pct(diag['pruned_phantom'], diag['pruned_total'])}")

    if not measured:
        print("[gate] nothing measured -- no conversation produced a scoreable turn.")
        return

    windows = sorted({w for w in (layer_windows or []) if w is not None})
    num_sliding = sum(1 for w in (layer_windows or []) if w is not None)
    print("\n" + "=" * 72)
    print(f"[gate] {measured} conversation(s) measured, {truncated} truncated, "
          f"{skipped} skipped")
    print(f"[gate] scorer: {len(layer_windows or [])} layers, {num_sliding} "
          f"sliding (windows {windows or 'none'}), "
          f"score_layers={args.score_layers!r}, "
          f"mask_to_window={args.mask_sliding_window}, "
          f"keep={args.keep_percentage}")
    for label, key in (("KEPT positions", "kept"), ("PRUNED-AWAY positions", "pruned")):
        phantom = totals[f"{key}_phantom"]
        total = totals[f"{key}_total"]
        null = totals[f"{key}_expected"]
        excess = (phantom - null) / total if total else 0.0
        print(f"[gate] {label:<22} {_pct(phantom, total):>6}  "
              f"(null {_pct(null, total):>6}, excess {excess:+.1%})  "
              f"{phantom}/{total} votes")
    print("=" * 72)
    print("[gate] 'null' is what the rate would be if the winning layer were "
          "chosen at random: on a mostly-sliding model most wins land on a "
          "sliding layer by composition alone, so the EXCESS over null is the "
          "signal, not the headline rate.")

    if num_sliding == 0:
        print("[gate] NOTE: this scorer has no sliding-window layer, so the "
              "hypothesis cannot apply to it. A 0% rate here says nothing.")
    else:
        print("[gate] Read this as: KEPT meaningfully ABOVE PRUNED-AWAY means "
              "selection was decided by layers scoring positions they can "
              "never reach -- run again with --score-layers global_only to "
              "confirm it drops to 0%, then treat that as the fix. Similar "
              "rates mean the hypothesis does not hold for this workload.")


if __name__ == "__main__":
    main()
