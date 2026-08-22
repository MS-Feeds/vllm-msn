#!/usr/bin/env python3
"""Measures, per TURN INDEX, whether the gold answer's own text survives the
speculator's selection -- the check that separates the two live explanations
for `SPARSE-k*`'s turn-0 accuracy deficit.

## The observation this exists to explain

A real graded sweep (`scbench_kv`, 100 conversations x 5 turns) showed every
SPARSE keep rate scoring far worse at turn 0 than at turns 1-4:

    SPARSE-k40:  47 / 69 / 69 / 72 / 70
    SPARSE-k60:  54 / 77 / 77 / 79 / 77
    SPARSE-k80:  64 / 81 / 81 / 84 / 86
    M000 (dense, old flattened rendering):  70 / 74 / 61 / 59 / 51

Three properties of that pattern constrain the explanation, and rule most
candidates out:

  * **Turn-0-specific.** Turns 1-4 are flat-to-slightly-rising; only turn 0
    drops.
  * **Sparsity-scaling.** Dense M000 shows no such deficit at all (its turn 0
    is among its BEST turns), so whatever this is, it acts through the
    gather.
  * **Binary, not a gradient.** The whole drop happens between turn 0 and
    turn 1, then flattens. A smooth positional or context-length effect
    would show a gradient across all five turns instead.

## The two explanations those constraints leave, and how this tells them apart

  A. **Selection.** The answer's tokens are dropped from the selection more
     often at turn 0 than later. Turn 0 is the only turn whose candidate
     pool is pure context, with no prior-turn material adjacent to the query
     for an attention-based, recency-biased scorer to latch onto -- so if
     the scorer is finding the answer region by proximity rather than by
     content, turn 0 is where that fails.
  B. **Generation.** The selection contains the answer just as often at
     turn 0, but the model doesn't emit it cleanly. Turn 0 is the only turn
     with no completed assistant message in context to imitate for
     terseness, so it is the turn most likely to open with a preamble and
     lose the tail of a long value to `--max-tokens` (default 64). Note
     `grade_scbench.py`'s `in_match` is `ground_truth in prediction`, so
     verbosity itself is free -- only truncation or corruption scores 0.

This script measures A directly. It replays the exact selection a real run
registered (same `SpecConfig`/`SpecPrefillProposer`/`compute_pruned_turn` as
`predict_scbench.py::run_sparse_attention`, and the same prior-turn output
replay `diagnose_speculator_selection.py` established is required for
turn > 0), then asks one question per turn: **is the gold answer still
present in the text the selection kept?**

    gold survival flat across turns  -> A is ruled out; the deficit is B,
                                        and a larger --max-tokens should
                                        close most of it.
    gold survival low at turn 0      -> A confirmed; the scorer is the
                                        thing to fix, not the gather.

## Why this needs only the speculator

Same scope cut as `diagnose_speculator_selection.py`: `compute_pruned_turn`
is byte-identical to what a real run registers, and touches only the 1B
speculator -- no target model, no decode loop, no `SparseTargetWorker`.

## Known approximation

The target attends at KV-BLOCK granularity plus the chat-template wrapper
spans, so it sees slightly MORE than `kept_positions` alone. This script
reports survival against the raw selection, which is therefore a mild
UNDER-estimate -- uniformly across turns, so per-turn comparison (the only
thing being asked here) is unaffected.

Usage:
    python3 diagnose_gold_survival.py \\
        --predictions-file results/SPARSE-k40-g32_predictions.jsonl \\
        --keep-percentage 0.4 --granularity 32 --max-conversations 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")


def _group_consecutive_runs(positions: List[int]) -> List[tuple]:
    """`[1,2,3,7,8]` -> `[(1,3),(7,8)]`. Same helper as
    `diagnose_speculator_selection.py`, duplicated rather than imported so
    that script stays independently runnable."""
    if not positions:
        return []
    ordered = sorted(positions)
    runs = [(ordered[0], ordered[0])]
    for pos in ordered[1:]:
        start, end = runs[-1]
        if pos == end + 1:
            runs[-1] = (start, pos)
        else:
            runs.append((pos, pos))
    return runs


def main() -> None:
    # Imported here rather than with the other deferred imports below: these
    # feed argparse `choices`, which is built before any of that runs.
    # Cheap -- `vllm_patch.config` pulls in neither vLLM nor torch.
    from vllm_patch.config import SCORE_AGGREGATIONS, SCORE_LAYER_SELECTIONS

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=Path,
                        default=Path(__file__).parent / "datasets" / "scbench_samples.jsonl")
    parser.add_argument("--predictions-file", type=Path, required=True,
                        help="A real run's <exp_id>_predictions.jsonl -- supplies each "
                             "conversation's prior-turn outputs, without which the replayed "
                             "ledger diverges from what the run actually had resident (see "
                             "diagnose_speculator_selection.py's docstring).")
    parser.add_argument("--keep-percentage", type=float, required=True)
    parser.add_argument("--granularity", default="32", choices=["token", "16", "32", "64"])
    parser.add_argument("--keep-mode", default="keep", choices=["keep", "discard"])
    # The fast inner loop for ACCURACY_IMPROVEMENTS.md §1: gold-answer
    # survival is what `in_match` is downstream of, and measuring it needs
    # only the SPECULATOR (no target engine, no generation, no grading), so
    # a scoring variant can be judged in minutes instead of a full run.
    # Defaults reproduce the reference scoring exactly.
    parser.add_argument("--score-aggregation", default="max",
                        choices=sorted(SCORE_AGGREGATIONS),
                        help="How per-(layer, head) attention collapses into one "
                             "score per token -- see SpecConfig.")
    parser.add_argument("--score-layers", default=None,
                        choices=[x for x in sorted(SCORE_LAYER_SELECTIONS, key=str) if x],
                        help="Restrict which layers vote (default: all).")
    parser.add_argument("--config", default="scbench_kv",
                        help="Restrict to one SCBench config (answers must be exact strings "
                             "for the survival check to mean anything -- retrieval configs "
                             "only; summarization answers won't match verbatim).")
    parser.add_argument("--max-conversations", type=int, default=20)
    parser.add_argument("--speculator-model", default=None)
    parser.add_argument("--speculator-device", default="cuda:1")
    parser.add_argument("--speculator-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--speculator-max-num-batched-tokens", type=int, default=131072)
    args = parser.parse_args()

    speculator_model = args.speculator_model or os.environ.get("LLAMA32_1B_MODEL_PATH")
    if not speculator_model:
        parser.error("--speculator-model or $LLAMA32_1B_MODEL_PATH is required")

    import torch
    from transformers import AutoConfig, AutoTokenizer

    from predict_scbench import (
        GRANULARITIES,
        LOOK_AHEAD_CNT,
        POOL_KERNEL_SIZE,
        load_conversations,
        render_turn_query,
    )
    from vllm_patch.config import SpecConfig
    from vllm_patch.conversation_state import ConversationState
    from vllm_patch.proposer import SpecPrefillProposer
    from vllm_patch.pruner import compute_pruned_turn

    prior_outputs: dict[str, dict[int, str]] = defaultdict(dict)
    with args.predictions_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prior_outputs[str(row["conversation_id"])][int(row["turn_idx"])] = row["pred"]

    conversations = [
        c for c in load_conversations(args.samples, max_keep=-1)
        if c["config"] == args.config and str(c["id"]) in prior_outputs
    ][: args.max_conversations]
    if not conversations:
        parser.error(f"no {args.config!r} conversations found in both {args.samples} "
                     f"and {args.predictions_file}")

    tok = AutoTokenizer.from_pretrained(speculator_model, trust_remote_code=True)
    spec_config = SpecConfig(
        keep_strategy="percentage",
        keep_kwargs={**GRANULARITIES[args.granularity], "percentage": args.keep_percentage},
        look_ahead_cnt=LOOK_AHEAD_CNT,
        pool_kernel_size=POOL_KERNEL_SIZE,
        keep_mode=args.keep_mode,
        score_aggregation=args.score_aggregation,
        score_layers=args.score_layers,
    )
    if args.score_aggregation != "max" or args.score_layers:
        print(f"[diagnose_gold_survival] scoring variant: "
              f"score_aggregation={args.score_aggregation!r} "
              f"score_layers={args.score_layers!r}")

    native_len = AutoConfig.from_pretrained(
        speculator_model, trust_remote_code=True).max_position_embeddings
    max_batched = args.speculator_max_num_batched_tokens
    if native_len is not None and max_batched > native_len:
        max_batched = int(native_len)
    max_model_len = min(max_batched + 1 + LOOK_AHEAD_CNT,
                        int(native_len) if native_len is not None
                        else max_batched + 1 + LOOK_AHEAD_CNT)
    proposer = SpecPrefillProposer(
        speculator_model_path=speculator_model,
        device=torch.device(args.speculator_device),
        gpu_memory_utilization=args.speculator_gpu_memory_utilization,
        max_num_batched_tokens=max_batched,
        max_model_len=max_model_len,
    )

    per_turn: dict[int, dict] = defaultdict(
        lambda: {"n": 0, "survived": 0, "survived_intact": 0,
                 "gold_rel_pos": [], "keep_rates": []})

    for conv in conversations:
        conv_id = str(conv["id"])
        outputs = prior_outputs[conv_id]
        context_ids = tok.encode(conv["context"], add_special_tokens=False)
        state = ConversationState(conv["id"], context_ids, args.keep_mode)
        # Tracked in parallel with `state` so kept_positions can be decoded
        # back to text -- same deliberate reach-around
        # diagnose_speculator_selection.py documents.
        ledger_ids: List[int] = list(context_ids)

        for turn_idx, turn in enumerate(conv["turns"]):
            if turn_idx > 0 and (turn_idx - 1) not in outputs:
                break  # can't replay history past a turn the run never produced
            query_ids = render_turn_query(tok, turn_idx, turn)
            result = compute_pruned_turn(proposer, spec_config, state, query_ids)
            ledger_ids.extend(query_ids)

            gold = str(turn["answer"])
            runs = _group_consecutive_runs(result.kept_positions)
            run_texts = [tok.decode(ledger_ids[s:e + 1], skip_special_tokens=True)
                         for s, e in runs]
            kept_text = "".join(run_texts)
            full_text = tok.decode(ledger_ids[:result.orig_len], skip_special_tokens=True)

            b = per_turn[turn_idx]
            b["n"] += 1
            if gold in kept_text:
                b["survived"] += 1
            # Stricter: intact inside ONE contiguous run, so a match can't be
            # an artifact of two kept fragments abutting at a seam.
            if any(gold in t for t in run_texts):
                b["survived_intact"] += 1
            if result.orig_len:
                b["keep_rates"].append(len(result.kept_positions) / result.orig_len)
            where = full_text.find(gold)
            if where >= 0 and len(full_text):
                b["gold_rel_pos"].append(where / len(full_text))

            if turn_idx in outputs:
                output_ids = tok.encode(f" {outputs[turn_idx]}", add_special_tokens=False)
                state.complete_turn(result.kept_history_pairs, output_ids)
                ledger_ids.extend(output_ids)

        print(f"[diagnose_gold_survival] {conv_id} done", flush=True)

    print(f"\n=== gold-answer survival by turn index "
          f"(keep={args.keep_percentage}, g{args.granularity}, {args.config}) ===")
    print(f"{'turn':>4} {'n':>4} {'survived':>9} {'intact':>8} "
          f"{'actual keep':>12} {'gold pos in ctx':>16}")
    for turn_idx in sorted(per_turn):
        b = per_turn[turn_idx]
        n = b["n"] or 1
        keep = statistics.mean(b["keep_rates"]) if b["keep_rates"] else float("nan")
        pos = statistics.mean(b["gold_rel_pos"]) if b["gold_rel_pos"] else float("nan")
        print(f"{turn_idx:>4} {b['n']:>4} {100.0 * b['survived'] / n:>8.1f}% "
              f"{100.0 * b['survived_intact'] / n:>7.1f}% {100.0 * keep:>11.1f}% "
              f"{pos:>15.2f}")

    print("\nReading it:")
    print("  survival flat across turns   -> the selection is NOT the cause. The turn-0")
    print("                                  deficit is generation-side; re-run turn 0 with a")
    print("                                  larger --max-tokens and see if the gap closes.")
    print("  survival low at turn 0       -> the scorer is dropping the answer region when")
    print("                                  there's no prior-turn material near the query.")
    print("  'gold pos in ctx' low at 0   -> turn 0's answers sit earlier in the context, so a")
    print("                                  recency-biased scorer would miss them by position")
    print("                                  alone -- a dataset property, not a code bug.")


if __name__ == "__main__":
    main()
