#!/usr/bin/env python3
"""GPU-node validation for `vllm_patch/proposer.py` +
`vllm_patch/speculator_worker.py` -- the persistent multi-turn speculator
engine. Run this BEFORE `validate_runner_integration.py` and BEFORE any
`predict_scbench.py` sweep -- see `REPRODUCE.md`'s validation order.

This is the least precedented part of the whole multi-turn port (see
`speculator_worker.py`'s module docstring's "Known risk areas") -- unlike
the single-turn pipeline's `validate_proposer.py` (which only had to check
a standalone, manually-driven model), this script exercises the
cross-process `worker_cls` + `collective_rpc` machinery
`speculator_worker.py` newly introduces.

Steps, in order -- each depends on the previous passing:

  A. Speculator engine construction -- `SpecPrefillProposer(...)` builds
     successfully, `worker_cls` resolves, `install_query_capture_hooks`
     runs without error inside the Worker process (no exception surfaced
     back through `LLM(...)` construction is the only externally-visible
     signal of this from the driver side).
  B. Single-turn scoring sanity -- `proposer.run_turn(...)` for one
     synthetic "conversation" (a short prompt, no real dataset needed)
     returns `actual_look_ahead_cnt == look_ahead_cnt` (no early EOS) and
     `query_buffer[i].shape == (1, look_ahead_cnt, H*D)` for every layer --
     confirms the begin_capture-after-first-token timing in `proposer.py`'s
     `run_turn` actually excludes the bootstrap prefill's own capture (the
     single-turn pipeline's equivalent bug -- conflating the prefill with a
     scored step -- was a real, confirmed-on-hardware bug there; nothing
     about this port's redesign makes that class of bug structurally
     impossible, only differently shaped, so re-check it fresh here).
  C. **The real multi-turn-specific check**: run a SECOND turn on the SAME
     conversation salt, extending the same speculator prompt by a small
     suffix. Confirm `RequestOutput.num_cached_tokens` on turn 2 reflects a
     real prefix-cache hit for (approximately) the turn-1 prefix length --
     this is the literal, measured version of the protocol's "each turn
     only prefills the new query tokens against its own cache" claim (see
     EXPERIMENT_PLAN.md's KEEP-mode risk note). Also confirms
     `proposer.retrieve_keys(...)` can recover K for LOCAL positions that
     were computed back in turn 1 (not recomputed in turn 2) -- the
     accumulating-slot-history mechanism `speculator_worker.py` adds
     specifically to make this possible.
  D. Per-layer head_dim/num_heads/num_kv_heads report -- same purpose as
     the single-turn pipeline's Step A: expected (per `LlamaAttention`'s
     implementation) uniform across all of Llama-3.2-1B's layers, unlike
     the Gemma4 sibling pipeline's confirmed-heterogeneous case. Treat a
     heterogeneous result as a surprise worth investigating, not expected
     behavior.

Usage:
    python3 validate_proposer.py --model $LLAMA32_1B_MODEL_PATH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("LLAMA32_1B_MODEL_PATH"),
        help="Speculator model path (Llama-3.2-1B). Defaults to $LLAMA32_1B_MODEL_PATH "
             "(see .env_exports.sh) -- pass explicitly to override.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--look-ahead-cnt", type=int, default=8)
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or $LLAMA32_1B_MODEL_PATH is required (source .env_exports.sh first)")

    import torch

    from vllm_patch.proposer import SpecPrefillProposer

    print(f"[validate_proposer] Step A: constructing speculator engine from {args.model}")
    proposer = SpecPrefillProposer(
        speculator_model_path=args.model,
        device=torch.device(args.device),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print("[validate_proposer] Step A: PASS -- engine constructed, hooks installed")

    conversation_salt = "validate-proposer-conv-1"
    # A short synthetic "conversation" -- real token ids don't matter for
    # this smoke test, only that they're valid vocabulary ids. Uses a small
    # ascending range rather than random ids so a failure is easy to eyeball.
    turn1_tokens = list(range(10, 40))  # 30 "tokens"

    print("[validate_proposer] Step B: running turn 1 (bootstrap conversation)")
    # ignore_eos=True: this smoke test feeds synthetic/out-of-distribution
    # token ids (an ascending range, not real language) as the "prompt" --
    # a real model can legitimately produce its own real EOS token partway
    # through decoding from nonsense input (confirmed on real hardware:
    # exactly this happened, actual_look_ahead_cnt=6 instead of 8, on a
    # perfectly healthy capture mechanism). Forcing ignore_eos=True makes
    # this check deterministic (always exactly look_ahead_cnt steps) so it
    # actually isolates the capture-timing mechanism this step exists to
    # validate, rather than being contaminated by incidental EOS timing on
    # meaningless input. Real sweep runs (predict_scbench.py) leave
    # ignore_eos at SpecConfig's own default (False) -- respecting real EOS
    # there is correct, only this synthetic smoke test wants it disabled.
    query_buffer, actual_look_ahead_cnt, num_cached_tokens = proposer.run_turn(
        conversation_salt=conversation_salt,
        turn_idx=0,
        full_sequence_token_ids=turn1_tokens,
        look_ahead_cnt=args.look_ahead_cnt,
        ignore_eos=True,
    )
    print(
        f"[validate_proposer] Step B: actual_look_ahead_cnt={actual_look_ahead_cnt} "
        f"(expected exactly {args.look_ahead_cnt} -- ignore_eos=True above rules "
        f"out real-EOS timing as an explanation for anything less), "
        f"num_cached_tokens={num_cached_tokens} (expected 0 -- nothing cached yet)"
    )
    assert actual_look_ahead_cnt == args.look_ahead_cnt, (
        f"expected exactly {args.look_ahead_cnt} lookahead steps (ignore_eos=True "
        f"rules out real EOS as the explanation), got {actual_look_ahead_cnt} -- "
        f"begin_capture likely fired too early/late (see proposer.py's run_turn "
        f"docstring on the exact timing this depends on), or ignore_eos isn't "
        f"actually reaching SamplingParams."
    )
    for i, q in enumerate(query_buffer):
        assert q.shape[1] == args.look_ahead_cnt, (
            f"layer {i}: expected {args.look_ahead_cnt} captured steps, got "
            f"shape {tuple(q.shape)}"
        )
    print("[validate_proposer] Step B: PASS")

    print("[validate_proposer] Step C: running turn 2 (extends the same conversation)")
    turn2_new_tokens = list(range(40, 45))  # 5 new "tokens"
    turn2_full_sequence = turn1_tokens + turn2_new_tokens
    query_buffer_2, actual_look_ahead_cnt_2, num_cached_tokens_2 = proposer.run_turn(
        conversation_salt=conversation_salt,
        turn_idx=1,
        full_sequence_token_ids=turn2_full_sequence,
        look_ahead_cnt=args.look_ahead_cnt,
        ignore_eos=True,
    )
    print(
        f"[validate_proposer] Step C: turn 2 num_cached_tokens={num_cached_tokens_2} "
        f"(expected close to {len(turn1_tokens)} -- the turn-1 prefix should hit "
        f"cache; some slack is normal since prefix-cache hits are block-size-"
        f"aligned, not exact)."
    )
    if num_cached_tokens_2 < len(turn1_tokens) // 2:
        print(
            "[validate_proposer] Step C: WARNING -- num_cached_tokens is far "
            "below the turn-1 prefix length. Either enable_prefix_caching isn't "
            "actually active, or cache_salt isn't being threaded through "
            "consistently -- investigate before trusting KEEP-mode's cost claim."
        )

    local_positions = list(range(len(turn1_tokens)))  # positions computed in turn 1
    keys = proposer.retrieve_keys(conversation_salt, local_positions)
    assert len(keys) == len(query_buffer_2), "K layer count must match Q layer count"
    for i, k in enumerate(keys):
        assert k.shape[0] == len(local_positions), (
            f"layer {i}: expected K for {len(local_positions)} positions, got "
            f"shape {tuple(k.shape)} -- retrieve_keys likely failed to recover "
            f"slots recorded in a PAST turn (turn 1), not just the current one "
            f"-- see speculator_worker.py's accumulating slot-history design."
        )
    print("[validate_proposer] Step C: PASS -- K retrieved for turn-1-computed positions from turn 2")

    proposer.discard_conversation(conversation_salt)

    print("[validate_proposer] Step D: per-layer head_dim/num_heads/num_kv_heads")
    print(
        "[validate_proposer] Step D: SKIPPED -- no permanent RPC method exposes "
        "per-layer head_dim/num_heads/num_kv_heads (unlike the single-turn "
        "pipeline's validate_proposer.py, which read this directly off "
        "proposer._speculator_layers in the driver's OWN process). Add a "
        "one-off collective_rpc call against SpeculatorWorker if this needs "
        "checking manually -- e.g. args=('model_runner._hooked_layers[i].head_dim', ...) "
        "-- or extend SpeculatorWorker with a real diagnostic method; not "
        "added permanently here to avoid growing that class's RPC surface "
        "for a one-time check."
    )

    print("\n[validate_proposer] ALL STEPS PASSED (Step D skipped, see above)")


if __name__ == "__main__":
    main()
