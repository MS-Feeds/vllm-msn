#!/usr/bin/env python3
"""Decodes and prints the ACTUAL TEXT the speculator selected to keep for
one real SCBench conversation's turn, at a given keep rate/granularity --
built to isolate two possible fault domains for the pervasive repetition-
loop garbage found in real `SPARSE-k*-g*` SCBench runs:

  A. The SPECULATOR's own selection is already garbled/scrambled/nonsensical
     -- a bug in `scoring.py`/`proposer.py` (the lookahead-decode scoring
     math), independent of anything in `sparse_target_runner.py`.
  B. The selection LOOKS sensible (coherent, if fragmented, prose) when
     decoded back to text, but the TARGET still produces garbage -- points
     at the target-side gather/attention mechanism instead
     (`sparse_target_runner.py`/`kv_cache_utils.py`), or at how positions
     get translated/registered.

Confirmed NOT explained by the id()-reuse cache bug already fixed in
`sparse_target_runner.py` (that fix landed, output is now deterministic,
but the garbage persists) -- and confirmed to happen even at `k80`
(losing only 20% of context) while `M000` (0% pruned) is clean on the same
conversations, which is a much larger effect than "20% less context"
should plausibly cause on its own.

## Why this only needs the speculator, not the target

`compute_pruned_turn` is exactly what `predict_scbench.py`'s own
`run_experiment(mode="sparse")` calls -- same `SpecConfig`, same
`SpecPrefillProposer` construction, byte-identical selection to what a
real run would register via `register_sparse_selection`. It only touches
the SPECULATOR (Llama-3.2-1B, a separate device) -- no `SparseTargetWorker`,
no 8B target model, no decode loop. That's a deliberate scope cut: it
answers "is the SELECTION itself already broken" without paying for (or
risking confusion from) the target-side machinery at all.

## Turn > 0 needs prior turns' actual generated text

This pipeline uses SELF-GENERATED history (see EXPERIMENT_PLAN.md /
`sparse_target_runner.py`'s module docstring) -- turn N's candidate pool
in KEEP mode is the full ledger including every PRIOR turn's own actual
output, not the golden answer. Diagnosing turn 0 needs nothing extra (the
ledger is just `context + turn 0's query`). Diagnosing turn N>0 needs the
real target output for turns `0..N-1` of this SAME conversation, supplied
via `--predictions-file` (an existing `results/<exp_id>_predictions.jsonl`
from a real run of this conversation) -- without it, the reconstructed
ledger would silently diverge from what the real run actually had
resident, making the diagnosis meaningless for turn > 0.

Usage:
    # turn 0 -- no predictions file needed
    python3 diagnose_speculator_selection.py \\
        --conversation-id scbench_qa_eng-53 --turn 0 \\
        --keep-percentage 0.8 --granularity 32

    # turn 2 -- needs turns 0-1's real generated output to replay history
    python3 diagnose_speculator_selection.py \\
        --conversation-id scbench_qa_eng-53 --turn 2 \\
        --keep-percentage 0.8 --granularity 32 \\
        --predictions-file results/SPARSE-k80-g32_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")


def _group_consecutive_runs(positions: List[int]) -> List[Tuple[int, int]]:
    """Groups a sorted list of positions into (start, end_inclusive) runs
    of consecutive integers -- used to print the selection as alternating
    kept-text / [...skipped N tokens...] segments, so gaps (and how often
    they cut through a sentence) are visible, not just a flat token soup."""
    if not positions:
        return []
    runs = []
    run_start = positions[0]
    run_end = positions[0]
    for p in positions[1:]:
        if p == run_end + 1:
            run_end = p
        else:
            runs.append((run_start, run_end))
            run_start = p
            run_end = p
    runs.append((run_start, run_end))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=Path(__file__).parent / "datasets" / "scbench_samples.jsonl")
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--turn", type=int, default=0)
    parser.add_argument("--keep-percentage", type=float, default=0.8)
    parser.add_argument("--granularity", default="32", choices=["token", "16", "32", "64"])
    parser.add_argument("--keep-mode", default="keep", choices=["keep", "discard"])
    parser.add_argument("--predictions-file", type=Path, default=None,
                         help="Required if --turn > 0 -- see module docstring.")
    parser.add_argument("--speculator-model", default=None, help="Defaults to $LLAMA32_1B_MODEL_PATH.")
    parser.add_argument("--speculator-device", default="cuda:1")
    parser.add_argument("--speculator-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--speculator-max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--context-chars", type=int, default=400,
                         help="How much of the surrounding context to print around each gap, for readability.")
    args = parser.parse_args()

    if args.turn > 0 and args.predictions_file is None:
        parser.error("--predictions-file is required when --turn > 0 -- see module docstring on why.")

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
    from vllm_patch.pruner import compute_pruned_turn
    from vllm_patch.proposer import SpecPrefillProposer

    conversations = load_conversations(args.samples, max_keep=-1)
    matches = [c for c in conversations if c["id"] == args.conversation_id]
    if not matches:
        parser.error(f"conversation_id {args.conversation_id!r} not found in {args.samples}")
    conv = matches[0]
    if args.turn >= len(conv["turns"]):
        parser.error(f"--turn {args.turn} out of range -- conversation has {len(conv['turns'])} turns")

    prior_outputs: Dict[int, str] = {}
    if args.predictions_file is not None:
        with open(args.predictions_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row["conversation_id"] == args.conversation_id:
                    prior_outputs[row["turn_idx"]] = row["pred"]
        missing = [i for i in range(args.turn) if i not in prior_outputs]
        if missing:
            parser.error(
                f"--predictions-file is missing turn_idx {missing} for conversation "
                f"{args.conversation_id!r} -- can't correctly replay the ledger up to --turn "
                f"{args.turn} without every prior turn's real output."
            )

    tok = AutoTokenizer.from_pretrained(speculator_model, trust_remote_code=True)

    gran_kwargs = GRANULARITIES[args.granularity]
    spec_config = SpecConfig(
        keep_strategy="percentage",
        keep_kwargs={**gran_kwargs, "percentage": args.keep_percentage},
        look_ahead_cnt=LOOK_AHEAD_CNT,
        pool_kernel_size=POOL_KERNEL_SIZE,
        keep_mode=args.keep_mode,
    )
    print(f"[diagnose_speculator_selection] conversation={args.conversation_id!r} turn={args.turn} "
          f"keep_percentage={args.keep_percentage} granularity={args.granularity!r} keep_mode={args.keep_mode!r}")

    # Text config, not the wrapper -- a natively multimodal checkpoint
    # (Gemma 4) has no `max_position_embeddings` at the top level, and
    # reading it raises AttributeError. See
    # `vllm_patch.model_structure.native_context_length`.
    from vllm_patch.model_structure import native_context_length

    speculator_native_max_model_len = native_context_length(
        speculator_model)
    speculator_max_num_batched_tokens = args.speculator_max_num_batched_tokens
    if speculator_native_max_model_len is not None and speculator_max_num_batched_tokens > speculator_native_max_model_len:
        speculator_max_num_batched_tokens = int(speculator_native_max_model_len)
    speculator_max_model_len = min(
        speculator_max_num_batched_tokens + 1 + LOOK_AHEAD_CNT,
        int(speculator_native_max_model_len) if speculator_native_max_model_len is not None
        else speculator_max_num_batched_tokens + 1 + LOOK_AHEAD_CNT,
    )

    proposer = SpecPrefillProposer(
        speculator_model_path=speculator_model,
        device=torch.device(args.speculator_device),
        gpu_memory_utilization=args.speculator_gpu_memory_utilization,
        max_num_batched_tokens=speculator_max_num_batched_tokens,
        max_model_len=speculator_max_model_len,
    )

    context_ids = tok.encode(conv["context"], add_special_tokens=False)
    state = ConversationState(conv["id"], context_ids, args.keep_mode)

    # Ledger token ids, tracked in parallel with `state` -- needed to
    # decode `result.kept_positions` back to text, since ConversationState
    # doesn't expose its internal ledger publicly (by design, see that
    # module's docstring -- this script reaches around that deliberately,
    # for diagnostic purposes only).
    ledger_ids: List[int] = list(context_ids)

    result = None
    for turn_idx in range(args.turn + 1):
        query_ids = render_turn_query(tok, turn_idx, conv["turns"][turn_idx])
        result = compute_pruned_turn(proposer, spec_config, state, query_ids)
        ledger_ids.extend(query_ids)

        if turn_idx < args.turn:
            # Replay this prior turn's REAL generated output into both the
            # ledger and conversation_state, exactly as run_sparse_attention
            # does -- see module docstring on why this must be the real
            # output, not golden or empty.
            output_text = prior_outputs[turn_idx]
            output_ids = tok.encode(f" {output_text}", add_special_tokens=False)
            state.complete_turn(result.kept_history_pairs, output_ids)
            ledger_ids.extend(output_ids)
        else:
            # Target turn -- don't complete it, we only want its selection.
            pass

    assert result is not None
    total_ledger_len = len(ledger_ids)
    kept = result.kept_positions
    print(f"\n[diagnose_speculator_selection] ledger length at this turn's query end: "
          f"{result.orig_len} (== len(ledger_ids) as tracked here: {len(ledger_ids)})")
    print(f"[diagnose_speculator_selection] kept {len(kept)}/{result.orig_len} positions "
          f"({len(kept) / result.orig_len * 100:.1f}%, requested keep_percentage={args.keep_percentage})")

    runs = _group_consecutive_runs(kept)
    print(f"[diagnose_speculator_selection] {len(runs)} contiguous kept run(s) "
          f"(fewer, longer runs = less fragmented; this many runs over {result.orig_len} "
          f"positions gives a rough fragmentation sense)")

    print("\n" + "=" * 100)
    print("RECONSTRUCTED KEPT CONTENT (in original order, gaps marked) --")
    print("this is notionally what the target's decode-step attention can see for this turn:")
    print("=" * 100)
    for i, (start, end) in enumerate(runs):
        run_text = tok.decode(ledger_ids[start:end + 1], skip_special_tokens=True)
        print(f"\n--- run {i} [ledger positions {start}-{end}, {end - start + 1} tokens] ---")
        print(run_text)
        if i < len(runs) - 1:
            next_start = runs[i + 1][0]
            gap = next_start - end - 1
            print(f"\n    ...[{gap} tokens skipped]...")
    print("\n" + "=" * 100)

    full_text = tok.decode(ledger_ids[:result.orig_len], skip_special_tokens=True)
    print(f"\n[diagnose_speculator_selection] for reference, the FULL (unrestricted) ledger text "
          f"up to this turn's query end follows -- compare against the reconstructed kept content "
          f"above to judge whether the selection reads as a sensible (if fragmented) subset, or as "
          f"scrambled/duplicated nonsense:")
    print("-" * 100)
    print(full_text[-4000:] if len(full_text) > 4000 else full_text)


if __name__ == "__main__":
    main()
