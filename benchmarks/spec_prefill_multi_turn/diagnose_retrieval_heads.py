#!/usr/bin/env python3
"""The §1.3 gate: measures the UPPER BOUND on retrieval-head filtering
before anyone builds it (ACCURACY_IMPROVEMENTS.md §1.3).

## What this answers, and why it comes first

The §1.1/§1.2 sweep found that `max` over (layer, head) beats `mean` by 12.4
points and `zmean` by 17.2 -- i.e. the useful signal is concentrated in a few
heads, and averaging lets the ~500 uninformative ones outvote them. That is
the retrieval-head hypothesis, measured accidentally. The obvious follow-up is
to identify the retrieval heads explicitly and aggregate only within them.

That build is multi-day work (offline needle-probe identification, a head-set
config surface, a graded sweep). This script answers, in one speculator-only
pass, whether it could possibly pay -- by selecting heads USING THE GOLD
ANSWER'S OWN POSITION. That is cheating, so whatever it reports is a ceiling
no honest head-identification method can exceed:

- ceiling near the all-head baseline  -> the 1B does not localize the needle
  at all, no head selection can help, §1.3 is dead for the cost of one run.
- ceiling far above it                -> the information IS present and being
  drowned out; §1.3 is a readout fix and worth building.

Same "measure the ceiling before optimizing toward it" move as the ORACLE-k*
row, which has already killed two multi-day ideas cheaply.

## The second question, which is just as decisive

A high ceiling is necessary but not sufficient. Retrieval heads are supposed
to be a FIXED property of a checkpoint -- if the heads that matter are
different heads on every conversation, then no static head list can capture
them and §1.3 dies anyway, ceiling or no ceiling. So this also reports how
stable the per-turn oracle head sets are against each other and against a
single global ranking. Both answers come from the same pass.

## Scope

Speculator only -- no target engine, no decode loop, no grading. Reuses
`compute_pruned_turn`'s exact submission path via
`SpecPrefillProposer.run_turn_and_head_diagnostics`, so the captured queries
are byte-identical to what a real scoring turn would produce.

Retrieval configs only (`scbench_kv` by default): the gold answer has to be
an exact string present in the context for "which positions are the gold
span" to be a well-posed question at all.

Usage:
    python3 diagnose_retrieval_heads.py \\
        --predictions-file results/SPARSE-k20-g32_predictions.jsonl \\
        --keep-percentage 0.2 --granularity 32 --max-conversations 20

Also the gate for predict_scbench.py's EARLY-k*-g32-L<n> family (scoring
with the TARGET's own first n layers instead of a separate speculator) --
point --speculator-model at the target and ask for layer prefixes:

    python3 diagnose_retrieval_heads.py \\
        --speculator-model "$LLAMA31_8B_MODEL_PATH" \\
        --predictions-file results/SPARSE-k20-g32_predictions.jsonl \\
        --keep-percentage 0.2 --granularity 32 --max-conversations 20 \\
        --layer-prefix-budgets 1,2,3,4,5,6,7,8,16,32
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

DEFAULT_TOP_N = [1, 2, 4, 8, 16, 32, 64, 128]


def gold_token_positions(tok, context: str, gold: str) -> List[int]:
    """Ledger positions of the gold answer's tokens inside the context.

    Located by CHARACTER offset and mapped through the tokenizer's own
    `offset_mapping`, not by searching for the gold's standalone token ids in
    the context's ids. The latter is the obvious approach and is wrong: a
    string tokenizes differently in isolation than it does mid-context
    (leading-space merges, digit grouping), so the subsequence often is not
    there to find, and when it is, it can be found in the wrong place.

    The offsets come from the SAME single `tok(context)` call that produces
    the ids the ledger's context region is built from, so the indices line up
    exactly with the positions the attention scores are indexed by -- no
    re-tokenization seam to drift across.

    Returns `[]` if the gold string does not appear verbatim in the context,
    which is a legitimate outcome for a non-retrieval config rather than an
    error (the caller skips those turns).
    """
    where = context.find(gold)
    if where < 0:
        return []
    encoded = tok(context, add_special_tokens=False, return_offsets_mapping=True)
    end = where + len(gold)
    return [
        i for i, (start, stop) in enumerate(encoded["offset_mapping"])
        if stop > where and start < end
    ]


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def main() -> None:
    from vllm_patch.config import SCORE_AGGREGATIONS  # noqa: F401  (parity check)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=Path,
                        default=Path(__file__).parent / "datasets" / "scbench_samples.jsonl")
    parser.add_argument("--predictions-file", type=Path, required=True,
                        help="A completed run's predictions, used ONLY to replay the "
                             "same self-generated history this diagnostic's turns "
                             "were produced under -- same reason "
                             "diagnose_gold_survival.py needs one.")
    parser.add_argument("--keep-percentage", type=float, required=True)
    parser.add_argument("--granularity", default="32", choices=["token", "16", "32", "64"])
    parser.add_argument("--config", default="scbench_kv",
                        help="Retrieval configs only -- the gold answer must appear "
                             "verbatim in the context.")
    parser.add_argument("--max-conversations", type=int, default=20)
    parser.add_argument("--skip-conversations", type=int, default=0,
                        help="Skip the first N conversations. Exists for ONE reason: "
                             "a head set ranked on conversations 1..N and then scored "
                             "on conversations 1..M is scored partly on the turns it "
                             "was fitted to, and 2 heads chosen from 512 can memorize "
                             "a little even at that ratio. Rank with "
                             "--max-conversations N, then score with "
                             "--skip-conversations N for a clean out-of-sample "
                             "number.")
    parser.add_argument("--top-n", default=",".join(str(n) for n in DEFAULT_TOP_N),
                        help="Comma-separated head-budget sizes to evaluate the "
                             "cheating selection at.")
    parser.add_argument("--speculator-model", default=None)
    parser.add_argument("--speculator-device", default="cuda:1")
    parser.add_argument("--speculator-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--speculator-max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--scorer-prefill-chunk-tokens", type=int, default=None,
                        help="Per-step batch size for the speculator (see "
                             "predict_scbench.py's flag of the same name). Lower it "
                             "(e.g. 32768) if a long context OOMs during prefill.")
    parser.add_argument("--fixed-heads-from", type=Path, default=None,
                        help="A --head-mass-out JSON from a PREVIOUS run. Evaluates "
                             "survival under the global top-N head sets from that "
                             "file -- heads chosen WITHOUT seeing this turn's gold, "
                             "i.e. the honest section 1.3 number rather than the "
                             "ceiling. Run once without it to produce the file, then "
                             "again with it (ideally over a different "
                             "--max-conversations slice, so the fixed set is scored "
                             "out-of-sample).")
    parser.add_argument("--fixed-head-budgets", default="1,2,4,8,16,32",
                        help="Head-set sizes to evaluate from --fixed-heads-from.")
    parser.add_argument("--layer-prefix-budgets", default="",
                        help="Comma-separated layer counts n (e.g. "
                             "'1,2,3,4,5,6,7,8,16,32'). Evaluates gold "
                             "survival when ONLY the first n layers' heads "
                             "vote -- the cheap gate for predict_scbench.py's "
                             "EARLY-k*-g32-L<n> family (scoring with the "
                             "target's own early layers instead of a separate "
                             "speculator). Point --speculator-model at the "
                             "TARGET checkpoint so the layer axis is the "
                             "target's own. Every n is measured in ONE pass, "
                             "since a layer prefix is just the first "
                             "n*num_heads rows of the flattened layer*head "
                             "axis the fixed-head-set machinery already "
                             "takes. UPPER BOUND, not the EARLY row's own "
                             "number: the lookahead tokens here are still "
                             "generated by the FULL model, so this isolates "
                             "'is early-layer attention informative' from "
                             "'can a truncated model produce usable lookahead "
                             "queries'.")
    parser.add_argument("--head-mass-out", type=Path, default=None,
                        help="Optional JSON dump of the accumulated per-head gold "
                             "mass -- the input a real §1.3 head list would be built "
                             "from, if the gate says build it.")
    args = parser.parse_args()

    top_n_list = [int(x) for x in args.top_n.split(",") if x.strip()]

    from transformers import AutoConfig, AutoTokenizer

    from predict_scbench import GRANULARITIES, LOOK_AHEAD_CNT, POOL_KERNEL_SIZE, render_turn_query
    from vllm_patch.config import SpecConfig
    from vllm_patch.conversation_state import ConversationState
    from vllm_patch.proposer import SpecPrefillProposer

    speculator_model = args.speculator_model or os.environ.get("LLAMA32_1B_MODEL_PATH")
    if not speculator_model:
        parser.error("--speculator-model or $LLAMA32_1B_MODEL_PATH is required")

    samples = {}
    with open(args.samples, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("config") == args.config:
                    samples[str(row["id"])] = row

    prior_outputs: dict = defaultdict(dict)
    with open(args.predictions_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                prior_outputs[str(row["conversation_id"])][row["turn_idx"]] = row["pred"]

    conversations = [
        samples[cid] for cid in samples if cid in prior_outputs
    ][args.skip_conversations: args.skip_conversations + args.max_conversations]
    if not conversations:
        parser.error(f"no {args.config!r} conversations present in both "
                     f"{args.samples} and {args.predictions_file}")

    tok = AutoTokenizer.from_pretrained(speculator_model, trust_remote_code=True)
    spec_config = SpecConfig(
        keep_strategy="percentage",
        keep_kwargs={**GRANULARITIES[args.granularity], "percentage": args.keep_percentage},
        look_ahead_cnt=LOOK_AHEAD_CNT,
        pool_kernel_size=POOL_KERNEL_SIZE,
        # KEEP only, hardcoded: this script drives `begin_turn`/`complete_turn`
        # itself (it needs the raw sequence, not `compute_pruned_turn`'s
        # selection) and passes `[]` as the kept-history pairs, which DISCARD
        # mode would read as "last turn kept nothing" and silently shrink the
        # candidate pool to nothing. The SPARSE architecture is keep-only
        # anyway -- it never evicts, so DISCARD has no meaning there.
        keep_mode="keep",
    )

    scorer_hf_config = AutoConfig.from_pretrained(
        speculator_model, trust_remote_code=True)
    native_len = scorer_hf_config.max_position_embeddings
    max_batched = args.speculator_max_num_batched_tokens
    if native_len is not None and max_batched > native_len:
        max_batched = int(native_len)
    max_model_len = min(max_batched + 1 + LOOK_AHEAD_CNT,
                        int(native_len) if native_len is not None else max_batched + 1 + LOOK_AHEAD_CNT)

    import torch

    proposer = SpecPrefillProposer(
        speculator_model_path=speculator_model,
        device=torch.device(args.speculator_device),
        gpu_memory_utilization=args.speculator_gpu_memory_utilization,
        max_num_batched_tokens=args.scorer_prefill_chunk_tokens or max_batched,
        enable_chunked_prefill=True,
        max_model_len=max_model_len,
    )

    baseline_survived = 0
    oracle_survived = {n: 0 for n in top_n_list}
    turns_measured = 0
    turns_skipped_no_gold = 0
    accumulated_mass: List[float] = []
    fixed_survived: dict = {}

    # A list both sources append to, rather than one owned by
    # --fixed-heads-from: --layer-prefix-budgets is an independent way to name
    # a head subset and must work without a prior ranking file.
    fixed_head_sets: list = []
    if args.fixed_heads_from:
        prior = json.loads(args.fixed_heads_from.read_text(encoding="utf-8"))
        ranking = prior["global_rank_desc"]
        fixed_head_sets += [
            [f"global top-{n}", ranking[:n]]
            for n in (int(x) for x in args.fixed_head_budgets.split(",") if x.strip())
            if n <= len(ranking)
        ]
        print(f"[diagnose_retrieval_heads] evaluating {len(fixed_head_sets)} fixed "
              f"head set(s) from {args.fixed_heads_from}")

    # Layer prefixes. `end_capture_and_head_diagnostics` indexes the FLATTENED
    # layer*head axis (`attn.flatten(0, 1)`), where head `h` of layer `l` is
    # row `l * num_heads + h` -- so "the first n layers" is exactly
    # `range(n * num_heads)`, and no new worker-side code is needed for it.
    layer_prefix_ns = [
        int(x) for x in args.layer_prefix_budgets.split(",") if x.strip()
    ]
    if layer_prefix_ns:
        scorer_heads = scorer_hf_config.num_attention_heads
        scorer_layers = scorer_hf_config.num_hidden_layers
        bad = [n for n in layer_prefix_ns if not 1 <= n <= scorer_layers]
        if bad:
            parser.error(
                f"--layer-prefix-budgets {bad} out of range: "
                f"{speculator_model} has {scorer_layers} layers"
            )
        fixed_head_sets += [
            [f"first-{n}-layers", list(range(n * scorer_heads))]
            for n in layer_prefix_ns
        ]
        print(f"[diagnose_retrieval_heads] evaluating layer prefixes "
              f"{layer_prefix_ns} of {speculator_model} "
              f"({scorer_layers} layers x {scorer_heads} heads)")
    fixed_head_sets = fixed_head_sets or None
    num_heads = None
    # Stability is reported at SEVERAL set sizes, not one. Reporting only
    # top-16 is actively misleading when the ceiling peaks at top-1: if a
    # couple of heads are stable and ranks 3-16 are noise, top-16 Jaccard
    # lands near 0.2 and reads as "no stable heads" when the opposite is
    # true. The budget where the ceiling peaks is the budget whose stability
    # decides anything.
    STABILITY_NS = [1, 2, 4, 8, 16]
    per_turn_top_sets_by_n = {n: [] for n in STABILITY_NS}

    for conv in conversations:
        conv_id = str(conv["id"])
        outputs = prior_outputs[conv_id]
        context = conv["context"]
        context_ids = tok.encode(context, add_special_tokens=False)
        state = ConversationState(conv["id"], context_ids, "keep")

        for turn_idx, turn in enumerate(conv["turns"]):
            if turn_idx > 0 and (turn_idx - 1) not in outputs:
                break  # cannot replay history past a turn the run never produced
            query_ids = render_turn_query(tok, turn_idx, turn)
            gold_positions = gold_token_positions(tok, context, str(turn["answer"]))

            candidate_pool, force_keep_query = state.begin_turn(query_ids)
            full_ids = [tid for tid, _ in candidate_pool + force_keep_query]

            if not gold_positions:
                turns_skipped_no_gold += 1
                diagnostics = None
            else:
                diagnostics = proposer.run_turn_and_head_diagnostics(
                    conversation_salt=conv_id,
                    turn_idx=turn_idx,
                    full_sequence_token_ids=full_ids,
                    look_ahead_cnt=spec_config.look_ahead_cnt,
                    pool_kernel_size=spec_config.pool_kernel_size,
                    keep_kwargs=spec_config.keep_kwargs,
                    gold_positions=gold_positions,
                    top_n_list=top_n_list,
                    fixed_head_sets=fixed_head_sets,
                    ignore_eos=spec_config.ignore_eos,
                )

            if diagnostics is not None:
                turns_measured += 1
                num_heads = diagnostics["num_heads"]
                baseline_survived += int(diagnostics["baseline_survived"])
                for n, ok in diagnostics["oracle_survived"].items():
                    oracle_survived[int(n)] = oracle_survived.get(int(n), 0) + int(ok)
                mass = diagnostics["gold_mass"]
                if not accumulated_mass:
                    accumulated_mass = [0.0] * len(mass)
                for i, m in enumerate(mass):
                    accumulated_mass[i] += m
                ranked = sorted(range(len(mass)), key=lambda i: mass[i], reverse=True)
                for n in STABILITY_NS:
                    per_turn_top_sets_by_n[n].append(set(ranked[:n]))
                for label, ok in (diagnostics.get("fixed_survived") or {}).items():
                    fixed_survived[label] = fixed_survived.get(label, 0) + int(ok)

            # Advance the ledger with the ACTUAL run's output, so turn N+1
            # sees the same history the graded run did.
            output_ids = (
                tok.encode(f" {outputs[turn_idx]}", add_special_tokens=False)
                if turn_idx in outputs else []
            )
            state.complete_turn([], output_ids)

        proposer.discard_conversation(conv_id)
        print(f"[diagnose_retrieval_heads] {conv_id} done", flush=True)

    if not turns_measured:
        print("no turns measured -- every turn's gold answer was absent from its "
              "context, or no lookahead step was ever captured")
        return

    slice_note = (
        f", conversations {args.skip_conversations + 1}.."
        f"{args.skip_conversations + len(conversations)}"
        if args.skip_conversations else ""
    )
    print(f"\n=== retrieval-head ceiling (keep={args.keep_percentage}, "
          f"g{args.granularity}, {args.config}, {turns_measured} turns, "
          f"{num_heads} heads{slice_note}) ===")
    if turns_skipped_no_gold:
        print(f"({turns_skipped_no_gold} turns skipped: gold answer not verbatim "
              f"in the context)")
    base_pct = 100.0 * baseline_survived / turns_measured
    print(f"\n{'head budget':>12} {'gold survived':>14} {'vs. all-head max':>18}")
    print(f"{'all (max)':>12} {base_pct:13.1f}% {'—':>18}")
    for n in sorted(oracle_survived):
        pct = 100.0 * oracle_survived[n] / turns_measured
        print(f"{n:>12} {pct:13.1f}% {pct - base_pct:+17.1f}")

    # Two kinds of fixed set share one accumulator (the worker just returns
    # whatever labels it was given), but they answer different questions and
    # are reported separately -- interleaving "global top-4" with
    # "first-4-layers" in one table sorted by N would read as a single sweep
    # of one variable, which it is not.
    global_sets = {l: v for l, v in fixed_survived.items() if l.startswith("global top-")}
    layer_sets = {l: v for l, v in fixed_survived.items() if l.startswith("first-")}

    # Fixed head sets, if a previous run's ranking was supplied: the honest
    # §1.3 number. Printed FIRST because it settles what the ceiling and the
    # stability numbers can only bracket.
    if global_sets:
        print("\n=== fixed global head sets (chosen without this turn's gold) ===")
        print(f"{'head set':>16} {'gold survived':>14} {'vs. all-head max':>18}")
        for label in sorted(global_sets, key=lambda l: int(l.split("-")[-1])):
            pct = 100.0 * global_sets[label] / turns_measured
            print(f"{label:>16} {pct:13.1f}% {pct - base_pct:+17.1f}")
        print("\nThis is what §1.3 would actually deliver. The ceiling above "
              "is what it could\ndeliver with per-input clairvoyance; the gap "
              "between them is the cost of\nhaving to choose the heads in advance.")

    if layer_sets:
        print("\n=== layer prefixes: only the first n layers' heads vote ===")
        print(f"{'layers':>16} {'gold survived':>14} {'vs. all-head max':>18}")
        for label in sorted(layer_sets, key=lambda l: int(l.split("-")[1])):
            pct = 100.0 * layer_sets[label] / turns_measured
            print(f"{label:>16} {pct:13.1f}% {pct - base_pct:+17.1f}")
        print(
            "\nThe gate for predict_scbench.py's EARLY-k*-g32-L<n> family: scoring with the "
            "target's own early layers instead of a separate speculator."
        )
        print(
            "An UPPER BOUND on what those rows can score, not their own "
            "number -- the lookahead"
        )
        print(
            "tokens here still come from the FULL model, while a real truncated "
            "scorer generates"
        )
        print(
            "them from layer-n hidden states through an untuned lm_head. A prefix "
            "at or near the"
        )
        print(
            "all-head baseline says early-layer attention carries the signal and "
            "the family is worth"
        )
        print(
            "running; every prefix far below it kills the family for the price of "
            "this one pass."
        )

    # Head stability: can a FIXED head list exist at all?
    global_rank = sorted(range(len(accumulated_mass)),
                         key=lambda i: accumulated_mass[i], reverse=True)
    print("\n=== head stability ===")
    print(f"{'set size':>9} {'vs. global':>11} {'consecutive':>13} {'distinct heads used':>21}")
    for n in STABILITY_NS:
        sets_n = per_turn_top_sets_by_n[n]
        if not sets_n:
            continue
        global_top = set(global_rank[:n])
        overlaps = [jaccard(s, global_top) for s in sets_n]
        pairwise = [jaccard(sets_n[i], sets_n[i + 1]) for i in range(len(sets_n) - 1)]
        distinct = len(set().union(*sets_n))
        pair_str = f"{statistics.mean(pairwise):.2f}" if pairwise else "n/a"
        print(f"{n:>9} {statistics.mean(overlaps):11.2f} {pair_str:>13} "
              f"{f'{distinct} of {num_heads}':>21}")
    print("\nRead the row whose set size matches where the CEILING peaks -- "
          "that is the budget\n§1.3 would actually use. High overlap there "
          "means a fixed head list can exist; low\noverlap at every size means "
          "the useful heads are chosen per input, which is not\nwhat a retrieval "
          "head is, and §1.3 cannot be built as a static set.")

    if args.head_mass_out:
        args.head_mass_out.parent.mkdir(parents=True, exist_ok=True)
        args.head_mass_out.write_text(json.dumps({
            "speculator_model": speculator_model,
            "config": args.config,
            "turns_measured": turns_measured,
            "num_heads": num_heads,
            "mean_gold_mass_per_head": [m / turns_measured for m in accumulated_mass],
            "global_rank_desc": global_rank,
        }, indent=2), encoding="utf-8")
        print(f"\n[diagnose_retrieval_heads] wrote per-head gold mass -> "
              f"{args.head_mass_out}")


if __name__ == "__main__":
    main()
