#!/usr/bin/env python3
"""Verifies (or refutes) the sliding-window scoring hypothesis for one
specific LongBench v2 sample -- see the conversation this was written for:
sample 66f37eb9821e116aacb2d295 degenerated into a hard repetition loop
("liberty liberty liberty...") at keep_percentage=0.3, after the
never-resumes-past-chunk and wrong-position-on-later-chunk hypotheses were
both structurally ruled out (the V1 scheduler here processes one new
prefill request at a time, to full completion -- neither bug's
precondition ever occurs), and after confirming the SAME sample does NOT
degenerate in the P001 (no pruning) baseline -- so the cause is located in
SpecPrefill's pruning decision for this document specifically, not the
runner-integration plumbing.

**Status: NOT executed here -- no GPU on this machine, same caveat as
every other script in this directory.**

Hypothesis under test: Gemma-4-26B-A4B-it's 28 sliding-window attention
layers (head_dim=256, confirmed real-hardware fact from
validate_proposer.py) have a sliding window of 1024 tokens (confirmed via
web search against Hugging Face/Ollama's published config -- see this
repo's conversation history for sources; NOT independently confirmed
against the actual loaded model's own hf_config here, see the
cross-check this script prints). vllm_patch/'s scoring pipeline
(kv_cache_utils.py's key retrieval, scoring.py's compute_attention_score)
has no window-masking logic anywhere (confirmed by grep -- see that
investigation) -- it computes unmasked Q@K^T for every layer against the
ENTIRE document, including the 28 sliding layers, even though those
layers can never actually attend beyond their 1024-token window during
real generation. Since aggregate_attention_score's aggregation is MAX
over (layer, head), any context position beyond 1024 tokens from the
query has its "importance" score decided by whichever of the 35
layers happens to produce the largest dot product there -- for the 28
sliding layers, that's an untrained-for-this-input, essentially arbitrary
number, since those layers never see anything that far away in real
inference.

This script does NOT modify vllm_patch/scoring.py to test this --
instead it calls the SAME production pipeline up through
compute_attention_score (unmodified, so results reflect exactly what
production computed), then does its own PARALLEL aggregation replicating
aggregate_attention_score's exact steps (softmax -> flatten -> pool ->
max -> mean) but ALSO capturing which (layer, head) won the max at each
(lookahead_step, context_position) pair -- torch.max returns indices
too, aggregate_attention_score just discards them (`.max(0)[0]`). Both
the production path (real kept positions, via chunk_select_from_
smoothed_attention, unmodified) and this diagnostic path consume the
SAME attn_scores tensor, so the speculator only runs once.

Reports, separately for the KEPT positions (what actually got sent to
the target model) and a random comparison sample of PRUNED-AWAY
positions: what fraction of the winning (layer, head) votes across all
look_ahead_cnt steps came from a sliding-window layer scoring a position
MORE than 1024 tokens from that lookahead step's query position -- i.e.
scoring something that layer could never have actually attended to. If
KEPT positions show a meaningfully higher phantom-vote rate than the
pruned-away comparison sample, that's direct evidence this document's
selection was corrupted by exactly the noise the hypothesis predicts. If
the rates are similar (or low for both), the hypothesis isn't what
happened for this document, regardless of whether the underlying gap is
real in general.

Usage:
    python3 verify_sliding_window_hypothesis.py \
        --target-model $GEMMA4_MODEL_PATH \
        --speculator-model $GEMMA4_E2B_MODEL_PATH \
        --sample-id 66f37eb9821e116aacb2d295
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # for `vllm_patch` imports

from predict_longbench_v2 import (  # noqa: E402
    CHUNK_SIZE,
    DEFAULT_SAMPLES,
    LOOK_AHEAD_CNT,
    POOL_KERNEL_SIZE,
    chat_wrapper_pieces,
    load_samples,
)

# Mirrors predict_longbench_v2.py's submit_pruned_requests EXACTLY (must
# stay in sync with it -- same requirement that function's own docstring
# states relative to prep_longbench_v2.py) -- this script needs to
# reconstruct the identical full_prompt_token_ids that function built for
# this sample, since the whole point is analyzing THAT exact scoring run,
# not a slightly different one.
_PREFIX_TEXT = "Please read the following long text and answer the question below.\n\n"
_SUFFIX_TEMPLATE = (
    "\n\nQuestion: {question}\n(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}"
    "\n(D) {choice_D}\n\nAnswer with a single letter (A, B, C, or D) "
    "corresponding to the correct choice."
)

# Best available evidence (web search against HF/Ollama's published config
# for google/gemma-4-26B-A4B-it), NOT hardcoded blindly -- this script also
# tries to read the value directly from the loaded model's own hf_config
# and prints both, flagging a mismatch rather than silently trusting one
# source. See this script's module docstring for why this number matters.
ASSUMED_SLIDING_WINDOW = 1024


def find_sample(samples: list[dict], sample_id: str) -> dict:
    for s in samples:
        if s["id"] == sample_id:
            return s
    raise ValueError(f"Sample id {sample_id!r} not found in loaded samples "
                     f"(loaded {len(samples)} total). Check --samples path.")


def instrumented_aggregate(
    attn_scores, pool_kernel_size: int, num_heads_per_layer: list[int]
):
    """Reimplements scoring.aggregate_attention_score's exact steps for ONE
    sample, but additionally returns which flat (layer, head) index won the
    max at every (lookahead_step, context_position) pair -- production's
    own `attn.max(0)[0]` throws this away by only keeping torch.max's
    `.values`, not its `.indices`.

    Args:
        attn_scores: [num_layer, num_head, look_ahead_cnt, context_len] for
            ONE sample (attn_scores[0] from scoring.compute_attention_score's
            per-sample list).
        num_heads_per_layer: heads per layer, used to convert a flat
            (layer*head) index back into (layer_idx, head_idx). Uniform
            across layers here (Gemma4-E2B), so this is really just
            [num_head] * num_layer, but kept as a list for clarity/
            generality rather than assuming.

    Returns:
        (token_importance, winning_layer_idx) -- token_importance is
        IDENTICAL to what scoring.aggregate_attention_score would return
        for this sample (verify this if in doubt: the value computation is
        copied verbatim). winning_layer_idx is [look_ahead_cnt, context_len]
        int array of which LAYER (not layer*head -- collapsed via the
        head's own layer membership) won the max at each position.
    """
    import torch

    num_layer, num_head, look_ahead_cnt, context_len = attn_scores.shape
    assert num_heads_per_layer == [num_head] * num_layer, (
        "This script assumes uniform heads-per-layer (true for Gemma4-E2B, "
        "confirmed via validate_proposer.py) -- if that's no longer true, "
        "the flat-index -> layer_idx division below needs to change."
    )

    original_dtype = attn_scores.dtype
    attn = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
        original_dtype
    )
    attn = attn.flatten(0, 1)  # [num_layer*num_head, look_ahead_cnt, context_len]

    if pool_kernel_size:
        attn = torch.nn.functional.avg_pool1d(
            attn, kernel_size=pool_kernel_size, padding=pool_kernel_size // 2, stride=1
        )

    max_vals, max_flat_idx = attn.max(0)  # both [look_ahead_cnt, context_len]
    winning_layer_idx = (max_flat_idx // num_head).cpu().numpy()

    token_importance = max_vals.mean(0)  # [context_len], matches production exactly
    return token_importance, winning_layer_idx


def analyze_positions(
    positions: list[int],
    winning_layer_idx,  # [look_ahead_cnt, context_len] numpy array
    layer_head_dims: list[int],
    orig_len: int,
    look_ahead_cnt: int,
    sliding_window: int,
    label: str,
) -> None:
    phantom_votes = 0
    total_votes = 0
    for p in positions:
        for step in range(look_ahead_cnt):
            layer_idx = int(winning_layer_idx[step, p])
            head_dim = layer_head_dims[layer_idx]
            is_sliding_layer = head_dim < max(layer_head_dims)  # 256 < 512 for Gemma4-E2B
            query_pos = orig_len + step
            distance = query_pos - p
            out_of_window = distance > sliding_window
            total_votes += 1
            if is_sliding_layer and out_of_window:
                phantom_votes += 1

    rate = phantom_votes / total_votes if total_votes else 0.0
    print(f"[verify] {label}: {len(positions)} position(s), {total_votes} "
          f"(position, lookahead-step) votes, {phantom_votes} ({rate:.1%}) "
          f"won by a sliding-window layer scoring OUTSIDE its real "
          f"{sliding_window}-token window.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--sample-id", default="66f37eb9821e116aacb2d295")
    parser.add_argument("--target-model", default=os.environ.get("GEMMA4_MODEL_PATH"))
    parser.add_argument("--speculator-model", default=os.environ.get("GEMMA4_E2B_MODEL_PATH"))
    parser.add_argument("--keep-percentage", type=float, default=0.3,
                         help="Must match the run that actually produced the "
                              "degenerate output for a faithful repro -- 0.3 "
                              "was this investigation's default.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speculator-device", default=None)
    parser.add_argument("--compare-sample-size", type=int, default=200,
                         help="How many random PRUNED-AWAY positions to sample "
                              "for the comparison baseline (all of them would "
                              "be slow for a 10k-90k token document).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.target_model or not args.speculator_model:
        parser.error("--target-model/--speculator-model (or $GEMMA4_MODEL_PATH/"
                      "$GEMMA4_E2B_MODEL_PATH) are required.")

    random.seed(args.seed)

    import torch
    from transformers import AutoTokenizer
    from vllm.config import ModelConfig, VllmConfig
    from vllm_patch.config import SpecConfig
    from vllm_patch.proposer import SpecPrefillProposer
    from vllm_patch.scoring import chunk_select_from_smoothed_attention, compute_attention_score

    samples = load_samples(args.samples, max_keep=-1)
    sample = find_sample(samples, args.sample_id)
    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    chat_before, chat_after = chat_wrapper_pieces(tok)
    prefix_text = chat_before + _PREFIX_TEXT
    context_text = sample["context"]
    suffix_text = _SUFFIX_TEMPLATE.format(
        question=sample["question"],
        choice_A=sample["choices"][0],
        choice_B=sample["choices"][1],
        choice_C=sample["choices"][2],
        choice_D=sample["choices"][3],
    ) + chat_after

    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    context_ids = tok.encode(context_text, add_special_tokens=False)
    suffix_ids = tok.encode(suffix_text, add_special_tokens=False)
    full_prompt_token_ids = prefix_ids + context_ids + suffix_ids
    orig_len = len(full_prompt_token_ids)
    print(f"[verify] sample id={sample['id']!r}: full prompt length={orig_len} "
          f"(prefix={len(prefix_ids)}, context={len(context_ids)}, suffix={len(suffix_ids)})")

    if args.speculator_device is not None:
        speculator_device = torch.device(args.speculator_device)
    elif torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        speculator_device = torch.device("cuda:1")
    else:
        speculator_device = torch.device(args.device)
        print(f"WARNING: only {torch.cuda.device_count()} GPU(s) visible -- "
              f"speculator will share the target's GPU.")
    if speculator_device.type == "cuda":
        torch.cuda.set_device(speculator_device)

    speculator_model_config = ModelConfig(
        model=args.speculator_model, trust_remote_code=True, dtype="bfloat16"
    )
    base_vllm_config = VllmConfig(model_config=speculator_model_config)
    proposer = SpecPrefillProposer(
        base_vllm_config=base_vllm_config,
        speculator_model_config=speculator_model_config,
        device=speculator_device,
    )

    layer_head_dims = [layer.head_dim for layer in proposer._speculator_layers]
    num_heads_per_layer = [layer.num_heads for layer in proposer._speculator_layers]
    print(f"[verify] {len(layer_head_dims)} speculator layers, head_dims: "
          f"{sorted(set(layer_head_dims))} "
          f"(expect {{256, 512}} per validate_proposer.py's confirmed finding)")

    # Best-effort direct read from the loaded model's own config, as a
    # cross-check against ASSUMED_SLIDING_WINDOW (see that constant's
    # comment) -- print both regardless of whether they match, since a
    # silent mismatch would make every distance/window judgment below wrong.
    sliding_window = ASSUMED_SLIDING_WINDOW
    try:
        hf_config = proposer.model.config
        live_value = getattr(hf_config, "sliding_window", None)
        if live_value is None and hasattr(hf_config, "text_config"):
            live_value = getattr(hf_config.text_config, "sliding_window", None)
        if live_value is not None:
            print(f"[verify] sliding_window from the loaded model's own config: "
                  f"{live_value} (assumed value: {ASSUMED_SLIDING_WINDOW}, "
                  f"{'MATCH' if live_value == ASSUMED_SLIDING_WINDOW else 'MISMATCH -- using live value'})")
            sliding_window = int(live_value)
        else:
            print(f"[verify] could not find sliding_window on the loaded model's "
                  f"config -- falling back to assumed value {ASSUMED_SLIDING_WINDOW} "
                  f"(web-search-sourced, not independently confirmed here).")
    except Exception as e:
        print(f"[verify] error reading live sliding_window ({e}) -- falling back "
              f"to assumed value {ASSUMED_SLIDING_WINDOW}.")

    head_dim = layer_head_dims[0] if len(set(layer_head_dims)) == 1 else layer_head_dims[0]
    look_ahead_cnt = LOOK_AHEAD_CNT

    lookahead_meta = proposer.build_lookahead_metadata(orig_len, look_ahead_cnt, head_dim)
    input_ids = torch.tensor(full_prompt_token_ids, dtype=torch.int32, device=speculator_device)
    positions_t = torch.arange(orig_len, dtype=torch.int64, device=speculator_device)

    def _last_token_only(sampled):
        return sampled[-1:].int()

    def _next_positions_fn(positions, step):
        return torch.tensor([orig_len + step], dtype=torch.int64, device=positions.device)

    query_buffer = proposer.run_lookahead_steps(
        initial_input_ids=input_ids,
        initial_positions=positions_t,
        look_ahead_cnt=look_ahead_cnt,
        prefill_attn_metadata=lookahead_meta.prefill_attn_metadata,
        prefill_slot_mapping=lookahead_meta.prefill_slot_mapping,
        per_step_attn_metadata=lookahead_meta.per_step_attn_metadata,
        per_step_slot_mapping=lookahead_meta.per_step_slot_mapping,
        next_input_fn=_last_token_only,
        next_positions_fn=_next_positions_fn,
        eos_token_id=tok.eos_token_id,
    )
    actual_look_ahead_cnts = [query_buffer[0].shape[1]]
    if actual_look_ahead_cnts[0] == 0:
        print("[verify] 0 lookahead steps completed -- can't analyze, matches "
              "compute_pruned_prompt's own early-return guard for this case.")
        return

    gathered_qk = proposer.tp_gather_qk(query_buffer)
    per_sample_slot_mapping = [lookahead_meta.slot_mapping]
    _, key_buffer = proposer.retrieve_qk(
        gathered_qk, per_sample_slot_mapping, lookahead_meta.block_size,
        lookahead_meta.num_kv_heads, head_dim,
    )

    # Unmodified production call -- attn_scores is exactly what
    # aggregate_attention_score/the real run consumed.
    attn_scores = compute_attention_score(gathered_qk, key_buffer, actual_look_ahead_cnts)

    spec_config = SpecConfig(
        keep_strategy="percentage",
        keep_kwargs={"chunk": True, "chunk_size": CHUNK_SIZE, "percentage": args.keep_percentage},
        look_ahead_cnt=look_ahead_cnt,
        pool_kernel_size=POOL_KERNEL_SIZE,
    )

    # Production aggregation (unmodified) -- real kept positions, exactly
    # what actually got sent to the target model in the run that degenerated.
    from vllm_patch.scoring import aggregate_attention_score
    real_token_importance = aggregate_attention_score(attn_scores, spec_config)
    real_kept_indices = chunk_select_from_smoothed_attention(real_token_importance, spec_config)[0]
    kept_positions = set(real_kept_indices.tolist())
    print(f"[verify] production pruning kept {len(kept_positions)} of {orig_len} "
          f"positions ({len(kept_positions) / orig_len:.1%}) -- this should match "
          f"the pruned length observed in the actual run for this sample.")

    # This script's parallel, instrumented aggregation -- consumes the SAME
    # attn_scores, so the speculator only ran once. token_importance here
    # should be numerically identical to real_token_importance (both copy
    # aggregate_attention_score's math verbatim); not asserted equal only
    # because of possible float nondeterminism across two separate softmax/
    # pool calls, not because the logic differs.
    _, winning_layer_idx = instrumented_aggregate(
        attn_scores[0], POOL_KERNEL_SIZE, num_heads_per_layer
    )

    analyze_positions(
        sorted(kept_positions), winning_layer_idx, layer_head_dims, orig_len,
        actual_look_ahead_cnts[0], sliding_window, "KEPT positions (sent to target model)"
    )

    pruned_away = [p for p in range(orig_len) if p not in kept_positions]
    comparison_sample = random.sample(
        pruned_away, min(args.compare_sample_size, len(pruned_away))
    )
    analyze_positions(
        comparison_sample, winning_layer_idx, layer_head_dims, orig_len,
        actual_look_ahead_cnts[0], sliding_window,
        f"PRUNED-AWAY positions (random sample of {len(comparison_sample)})"
    )

    print(
        "\n[verify] Interpretation: if KEPT's phantom-vote rate is meaningfully "
        "higher than PRUNED-AWAY's, this document's selection was measurably "
        "corrupted by sliding-window layers scoring content outside their real "
        "window -- direct evidence for the hypothesis on THIS document. If the "
        "rates are similar, the hypothesis isn't what happened here, regardless "
        "of whether the underlying scoring gap is real in general."
    )


if __name__ == "__main__":
    main()
