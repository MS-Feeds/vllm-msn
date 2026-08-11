#!/usr/bin/env python3
"""Prediction-generation driver for the multi-turn Top-K KV Cache Selection
sweep (see EXPERIMENT_PLAN.md's experiment matrix: keep-rate x KV-entry-
granularity, a baseline row, and an oracle-upper-bound row, all against
SCBench's `scbench_qa_eng`/`scbench_kv`/`scbench_summary` configs). Writes a
per-turn predictions JSONL (`{"conversation_id", "turn_idx", "config",
"pred"}`) that `grade_scbench.py` scores.

**Structurally different from `../spec_prefill_llama/predict_longbench_v2.py`**
in the ways EXPERIMENT_PLAN.md's architectural decisions require:

1. **Driven per-conversation, then per-turn, sequentially within a
   conversation** -- not a flat "submit all N independent samples, drive
   the engine to completion once" loop. A conversation's turn N+1 prompt
   depends on turn N's own scoring/pruning decision (DISCARD mode) and on
   `conversation_state.py`'s absolute-position ledger (both modes), so
   turns cannot be computed out of order. Different CONVERSATIONS don't
   depend on each other and could in principle be interleaved for target-
   side throughput -- **not attempted in this pass** (this driver runs one
   conversation fully to completion before starting the next, mirroring
   `proposer.py`'s own declared "one in-flight speculator request at a
   time" MVP scope) -- flagged as a real, deliberate simplification, not an
   oversight; interleaving conversations for throughput is a natural
   follow-up once correctness is confirmed on real hardware.
2. **Golden-context mode only** (EXPERIMENT_PLAN.md's decision #1): each
   turn's target generation is what gets GRADED, but the token sequence fed
   forward into FUTURE turns' history/ledger is the dataset's own reference
   answer, not the model's own output. This is what makes the whole
   per-turn token sequence for a conversation knowable statically up front
   -- see that decision's docstring for why this matters for tractability.
3. **Turn/answer text is tokenized as separate pieces and concatenated**,
   never produced by a single `apply_chat_template` call over an
   accumulating multi-role messages list. A "real" multi-turn chat
   rendering (alternating user/assistant messages, one per turn) would be
   more faithful to how these models are actually instruction-tuned for
   conversation, but makes it much harder to recover exact token-level
   turn/answer boundaries after the fact (chat templates insert
   role-header/eot wrapper tokens between messages that would need to be
   separately, carefully accounted for in `conversation_state.py`'s
   ledger). Instead, this driver renders the WHOLE conversation as ONE
   user/assistant exchange (a single chat-template wrapper, applied once
   per turn's growing prompt), with turn structure expressed in plain text
   ("Question 1: ... Answer 1: ... Question 2: ...") inside that one
   message -- see `render_turn_query`/`render_golden_answer` below.

   **Not an ad hoc shortcut** -- confirmed (not assumed) to match SCBench's
   own official reference harness's DEFAULT evaluation mode
   (`use_chat_template=False` in
   `microsoft/MInference/scbench/eval_utils.py`'s `create_multiturn_prompt`;
   their `follow_up_template` differs in exact wording but is the same
   structural shape: one open exchange, no role tags between turns), so
   results here should be comparable to published SCBench numbers using
   this same mode.

   **Real-run evidence this has a real cost, not just a theoretical one**
   (M000, 2026-08-11): turns 1-4 of a 5-turn `scbench_qa_eng` conversation
   produced direct, on-topic answers, but turn 5 showed the model
   misreading the accumulated `Question N: ... Answer N: ...` history as
   one compound question needing re-enumeration, rather than a sequence of
   already-closed exchanges -- plausibly connected to the lack of
   `<|eot_id|>` turn-closing signals in this rendering (not yet confirmed
   systematic across more conversations/turn positions; worth checking
   before trusting later-turn accuracy numbers from a full sweep). Two
   settings deferred to a later pass if this proves systematic, both
   real redesigns, not flag flips -- see EXPERIMENT_PLAN.md's "Key
   architectural decisions" #1 for the full reasoning on each:
     - **Chat-template rendering** -- SCBench's own `use_chat_template=True`
       alternative (real per-turn role messages), which would need
       `conversation_state.py` to track per-turn wrapper-token boundaries,
       not just the single constant wrapper this pass tracks.
     - **Self-generated history** -- SCBench's own `disable_golden_context=
       True` alternative (build future turns' history from the model's own
       completions instead of golden answers), which would remove the
       "every turn's tokens known up front" property this driver's
       tractable, batchable design (see decision #1 above) depends on.
4. **`conversation_state.ConversationState` is used uniformly across
   baseline, SpecPrefill, and oracle runs** -- even the baseline (no
   pruning) path calls `begin_turn`/`complete_turn` to keep the ledger
   bookkeeping identical across all three, it just never scores/prunes
   anything (keeps 100% of `candidate_pool` unconditionally).

Usage:
    python3 predict_scbench.py --list
    python3 predict_scbench.py --exp M000 --max-conversations 2   # smoke test
    python3 predict_scbench.py --exp M000,M-k80-g32,M-k20-gtoken
    python3 predict_scbench.py --exp specprefill   # all 16 M-k*-g* rows, no baseline
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))  # for `vllm_patch` imports

try:
    from tqdm import tqdm
except ImportError:
    # tqdm is a transitive dependency of vLLM itself (used for its own
    # model-loading progress bars), so this should always resolve in this
    # pipeline's real environment -- fallback only so a missing tqdm
    # degrades to "no progress bar" instead of crashing a run that could
    # otherwise take over an hour (a real baseline sweep against SCBench's
    # up-to-~3M-character contexts). Must duck-type .set_postfix()/.write()
    # too (not just be iterable), since run_baseline/run_specprefill call
    # both on whatever this returns -- a bare `return iterable` here would
    # crash the very degraded-mode fallback this exists to keep working.
    class _NoOpProgress:
        def __init__(self, iterable):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, **kwargs):
            pass

        def write(self, message):
            print(message)

    def tqdm(iterable, **kwargs):
        return _NoOpProgress(iterable)

OUT_DIR = Path(os.environ.get("BENCH_RESULTS_DIR", "results"))
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "all_runs.csv"
DEFAULT_SAMPLES = Path(__file__).parent / "datasets" / "scbench_samples.jsonl"

CSV_FIELDS = [
    "ts", "exp_id", "label", "mode", "keep_mode",
    "keep_percentage", "kv_granularity", "chunk_size",
    "look_ahead_cnt", "pool_kernel_size",
    "target_gpu_memory_utilization", "speculator_gpu_memory_utilization",
    "target_max_num_batched_tokens",
    "rep", "seed", "max_tokens",
    "num_conversations", "num_turns", "num_skipped_too_large",
    "elapsed_time", "turns_per_second",
    "actual_keep_rate_mean",
    "ttft_mean_ms", "ttft_p50_ms", "ttft_p90_ms",
    "num_cached_tokens_speculator_mean",
    "out_len_mean", "out_len_stdev",
    "finish_stop", "finish_length", "finish_other",
]

# See EXPERIMENT_PLAN.md's "SpecPrefill settings" -- algorithm hyperparameters
# shared across the whole keep-rate/granularity sweep, not swept themselves
# in this MVP pass.
LOOK_AHEAD_CNT = 8
POOL_KERNEL_SIZE = 13

# Protocol's evaluation grid (EXPERIMENT_PLAN.md): K% in {100,80,60,40,20},
# oracle upper bound, KV entry size in {token,16,32,64}. 100% == the
# baseline row (M000), handled separately (bypasses pruning entirely, same
# reasoning as the single-turn pipeline's P001 not paying for
# SpecPrefillWorker when nothing is pruned).
KEEP_RATES = [0.8, 0.6, 0.4, 0.2]
GRANULARITIES = {
    "token": {"chunk": False},
    "16": {"chunk": True, "chunk_size": 16},
    "32": {"chunk": True, "chunk_size": 32},
    "64": {"chunk": True, "chunk_size": 64},
}
# Oracle rows pair with ONE representative granularity (32), not the full
# granularity cross -- an oracle row needs an extra teacher-forced target
# forward pass per turn on top of the full unpruned baseline generation, so
# crossing it with all 4 granularities would multiply that cost 4x for a
# reference/ceiling metric, not the main throughput comparison. Revisit if
# granularity turns out to matter for the oracle ceiling too.
ORACLE_GRANULARITY = "32"


def _build_experiments() -> dict:
    experiments = {
        "M000": {
            "label": "Baseline (no pruning, full growing context every turn)",
            "mode": "baseline", "keep_mode": "keep",
            "keep_percentage": None, "granularity": None,
        },
    }
    for rate in KEEP_RATES:
        for gran_name in GRANULARITIES:
            exp_id = f"M-k{int(rate * 100)}-g{gran_name}"
            experiments[exp_id] = {
                "label": f"SpecPrefill keep={int(rate * 100)}% granularity={gran_name} keep_mode=keep",
                "mode": "specprefill", "keep_mode": "keep",
                "keep_percentage": rate, "granularity": gran_name,
            }
    for rate in KEEP_RATES:
        exp_id = f"ORACLE-k{int(rate * 100)}"
        experiments[exp_id] = {
            "label": f"Oracle upper bound keep={int(rate * 100)}% granularity={ORACLE_GRANULARITY} keep_mode=keep",
            "mode": "oracle", "keep_mode": "keep",
            "keep_percentage": rate, "granularity": ORACLE_GRANULARITY,
        }
    return experiments


EXPERIMENTS = _build_experiments()


def ensure_csv_header() -> None:
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(row: dict) -> None:
    with CSV_PATH.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def percentile(sorted_vals: list, q: float):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def load_conversations(path: Path, max_keep: int) -> list[dict]:
    conversations = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conversations.append(json.loads(line))
            if max_keep >= 0 and len(conversations) >= max_keep:
                break
    if not conversations:
        raise FileNotFoundError(
            f"Samples file empty or not found: {path}\nRun datasets/prep_scbench.py first."
        )
    return conversations


_CHAT_WRAPPER_PLACEHOLDER = "@@SPEC_PREFILL_MT_PLACEHOLDER@@"


def chat_wrapper_pieces(tok) -> tuple[str, str]:
    """Same technique as the single-turn pipeline's
    `predict_longbench_v2.py::chat_wrapper_pieces` -- returns (before, after)
    text surrounding a single user turn's content, tokenized separately from
    the (large, growing) conversation content itself."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": _CHAT_WRAPPER_PLACEHOLDER}],
        add_generation_prompt=True,
        tokenize=False,
    )
    before, after = rendered.split(_CHAT_WRAPPER_PLACEHOLDER)
    return before, after


def build_conversation_state(context: str, tok, keep_mode: str, conversation_id: str):
    from vllm_patch.conversation_state import ConversationState

    context_ids = tok.encode(context, add_special_tokens=False)
    return ConversationState(conversation_id, context_ids, keep_mode)


def render_turn_query(tok, turn_idx: int, turn: dict) -> list[int]:
    """This turn's own force-kept region -- see module docstring #3 for why
    the whole conversation is rendered as plain text inside one chat
    message rather than real alternating chat-template turns."""
    question_text = f"\n\nQuestion {turn_idx + 1}: {turn['input']}\nAnswer {turn_idx + 1}:"
    return tok.encode(question_text, add_special_tokens=False)


def render_golden_answer(tok, turn: dict) -> list[int]:
    return tok.encode(f" {turn['answer']}", add_special_tokens=False)


def drive_single_request_to_completion(llm_engine, request_id: str):
    """Same discipline as predict_longbench_v2.py's drive_engine_to_completion
    (never break early, never abort) -- specialized to exactly one in-flight
    request, per module docstring #1's MVP scope."""
    latest_output = None
    while llm_engine.has_unfinished_requests():
        for output in llm_engine.step():
            if output.request_id == request_id:
                latest_output = output
    return latest_output


def run_baseline(
    llm, tok, conversations, max_tokens, keep_mode, target_max_num_batched_tokens
) -> tuple[list[dict], dict]:
    """M000: plain add_request per turn, no worker_cls/proposer/pruning --
    still uses ConversationState for ledger bookkeeping (module docstring
    #4) but keeps every candidate token unconditionally.

    **Pre-flight length check, skip-and-report (not crash).** Ported from
    the single-turn pipeline's `predict_longbench_v2.py::submit_baseline_
    requests` -- a real failure this fixes, not a defensive guess: a real
    run against SCBench (contexts up to ~3M characters) hit vLLM's own
    hard `ValueError` for a prompt exceeding `max_model_len`, crashing the
    ENTIRE experiment on the first oversized conversation rather than
    skipping just that one. Baseline is EXPECTED to skip more than
    SpecPrefill experiments (it needs the full, unpruned conversation to
    fit; pruning exists specifically to make oversized contexts fit) -- see
    `grade_scbench.py`'s "missing" (excluded from every score denominator,
    not counted as wrong) vs. matched distinction, mirroring the same
    baseline-skips-more asymmetry the single-turn pipeline's own grading
    already accounts for.

    Checked BEFORE `state.begin_turn` mutates the ledger (using
    `state.total_len` -- the ledger length as of right before this turn's
    query, valid to check without begin_turn's side effects) so an
    oversized turn can be skipped cleanly, with `state` simply abandoned
    (no partial-turn cleanup needed) -- since context only grows turn over
    turn in baseline mode (no pruning ever shrinks it), once one turn is
    too large every later turn in the same conversation will be too;
    skipping the rest of that conversation rather than checking turn by
    turn avoids paying for tokenization + a doomed size check repeatedly.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    chat_before, chat_after = chat_wrapper_pieces(tok)
    chat_before_ids = tok.encode(chat_before, add_special_tokens=False)
    chat_after_ids = tok.encode(chat_after, add_special_tokens=False)
    wrapper_len = len(chat_before_ids) + len(chat_after_ids)

    predictions = []
    stats = {"ttfts": [], "out_lens": [], "finish": {"stop": 0, "length": 0, "other": 0},
              "actual_keep_rates": [], "num_cached_tokens_speculator": [],
              "num_skipped_too_large": 0}

    progress = tqdm(conversations, desc="M000 baseline", unit="conv")
    for conv in progress:
        progress.set_postfix(turns=len(predictions), skipped=stats["num_skipped_too_large"])
        state = build_conversation_state(conv["context"], tok, keep_mode, conv["id"])
        for turn_idx, turn in enumerate(conv["turns"]):
            query_ids = render_turn_query(tok, turn_idx, turn)

            prospective_len = state.total_len + len(query_ids) + wrapper_len
            if prospective_len > target_max_num_batched_tokens:
                progress.write(
                    f"[predict_scbench] SKIP conversation id={conv['id']!r} "
                    f"from turn {turn_idx} onward: full (unpruned) prompt "
                    f"length {prospective_len} exceeds "
                    f"--target-max-num-batched-tokens="
                    f"{target_max_num_batched_tokens} -- baseline needs the "
                    f"whole conversation to fit, no pruning to shrink it."
                )
                stats["num_skipped_too_large"] += 1
                break

            candidate_pool, force_keep_query = state.begin_turn(query_ids)
            full_ids = [tid for tid, _ in candidate_pool] + [tid for tid, _ in force_keep_query]
            prompt_ids = chat_before_ids + full_ids + chat_after_ids

            request_id = f"{conv['id']}::turn{turn_idx}"
            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
            prompt = TokensPrompt(prompt_token_ids=prompt_ids)
            llm.llm_engine.add_request(request_id, prompt, sampling_params)
            output = drive_single_request_to_completion(llm.llm_engine, request_id)

            if output is not None:
                completion = output.outputs[0]
                stats["out_lens"].append(len(completion.token_ids))
                stats["finish"][completion.finish_reason if completion.finish_reason in
                                 stats["finish"] else "other"] += 1
                if output.metrics is not None and output.metrics.first_token_latency:
                    stats["ttfts"].append(output.metrics.first_token_latency * 1000)
                stats["actual_keep_rates"].append(1.0)
                predictions.append({
                    "conversation_id": conv["id"], "turn_idx": turn_idx,
                    "config": conv["config"], "pred": completion.text,
                })

            golden_answer_ids = render_golden_answer(tok, turn)
            state.complete_turn(candidate_pool, golden_answer_ids)

    return predictions, stats


def run_specprefill(
    llm,
    tok,
    proposer,
    spec_config,
    conversations,
    max_tokens,
    keep_mode,
    speculator_max_num_batched_tokens,
    target_max_num_batched_tokens,
) -> tuple[list[dict], dict]:
    """M-k*-g*: SpecPrefill pruning via the speculator, per turn.

    **Two-stage length check, skip-and-report (not crash)** -- same real
    failure `run_baseline`'s docstring describes, mirrored here with the
    single-turn pipeline's OWN two-stage check
    (`predict_longbench_v2.py::submit_pruned_requests`, which separately
    checks the full unpruned prompt against the speculator's budget before
    scoring, then the pruned result against the target's budget before
    generating):

    1. **Before `compute_pruned_turn`** (which submits the FULL candidate
       pool + query to the speculator for scoring -- see `proposer.py`'s
       module docstring): check `state.total_len + len(query_ids)` (the
       KEEP-mode candidate pool size -- see note below for DISCARD) against
       `speculator_max_num_batched_tokens`. This is the check that actually
       matters for SCBench's huge contexts: pruning only shrinks what the
       TARGET sees, never what the SPECULATOR must process to decide what
       to prune in the first place.
    2. **After pruning**, a lightweight safety net: the pruned result
       (usually much smaller than the speculator's own budget, since it's a
       strict subset) is checked against `target_max_num_batched_tokens`
       too, in case that budget happens to be configured smaller than the
       speculator's.

    Either failing skips the rest of THIS conversation (same reasoning as
    `run_baseline`: context/candidate-pool size is non-decreasing turn over
    turn under KEEP mode, so once one turn is too large, so is every later
    one in the same conversation).

    **DISCARD-mode note**: check 1 uses `state.total_len + len(query_ids)`,
    which is exactly the KEEP-mode candidate pool size but an OVER-estimate
    for DISCARD mode (whose actual candidate pool is a shrunk subset) --
    conservative/safe (might skip a turn DISCARD could have actually
    handled) rather than wrong in the unsafe direction. Not tightened here
    since DISCARD isn't part of this MVP's default experiment matrix (see
    EXPERIMENT_PLAN.md).
    """
    from vllm import SamplingParams
    from vllm_patch.pruner import PrunedTurnResult, compute_pruned_turn, prune_and_add_turn

    chat_before, chat_after = chat_wrapper_pieces(tok)
    chat_before_ids = tok.encode(chat_before, add_special_tokens=False)
    chat_after_ids = tok.encode(chat_after, add_special_tokens=False)
    wrapper_len = len(chat_before_ids) + len(chat_after_ids)

    predictions = []
    stats = {"ttfts": [], "out_lens": [], "finish": {"stop": 0, "length": 0, "other": 0},
              "actual_keep_rates": [], "num_cached_tokens_speculator": [],
              "num_skipped_too_large": 0}

    keep_pct = spec_config.keep_kwargs.get("percentage")
    desc = f"SpecPrefill keep={keep_pct}"
    progress = tqdm(conversations, desc=desc, unit="conv")
    for conv in progress:
        progress.set_postfix(turns=len(predictions), skipped=stats["num_skipped_too_large"])
        state = build_conversation_state(conv["context"], tok, keep_mode, conv["id"])
        for turn_idx, turn in enumerate(conv["turns"]):
            query_ids = render_turn_query(tok, turn_idx, turn)

            prospective_speculator_len = state.total_len + len(query_ids)
            if prospective_speculator_len > speculator_max_num_batched_tokens:
                progress.write(
                    f"[predict_scbench] SKIP conversation id={conv['id']!r} "
                    f"from turn {turn_idx} onward: full (unpruned) "
                    f"candidate-pool length {prospective_speculator_len} "
                    f"exceeds --speculator-max-num-batched-tokens="
                    f"{speculator_max_num_batched_tokens} -- the speculator "
                    f"must process the whole thing to score it, before any "
                    f"pruning happens."
                )
                stats["num_skipped_too_large"] += 1
                break

            result = compute_pruned_turn(proposer, spec_config, state, query_ids)
            prompt_ids = chat_before_ids + result.pruned_token_ids + chat_after_ids

            if len(prompt_ids) > target_max_num_batched_tokens:
                progress.write(
                    f"[predict_scbench] SKIP conversation id={conv['id']!r} "
                    f"from turn {turn_idx} onward: pruned prompt length "
                    f"{len(prompt_ids)} still exceeds "
                    f"--target-max-num-batched-tokens="
                    f"{target_max_num_batched_tokens} (after keeping "
                    f"{len(result.kept_positions)}/{result.orig_len} "
                    f"tokens) -- would hit the same upstream failure this "
                    f"check exists to avoid."
                )
                stats["num_skipped_too_large"] += 1
                break

            request_id = f"{conv['id']}::turn{turn_idx}"
            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
            # PruneRecord positions must cover the FULL prompt actually sent
            # (including the constant chat wrapper pieces), not just the
            # scored candidate+query span -- offset kept_positions/orig_len
            # by len(chat_before_ids) and treat the wrapper pieces as
            # trivially "kept at their own identity position" by extending
            # both ends. Simplest correct approach: the wrapper tokens are
            # never scored/pruned, so their RoPE positions should just be
            # their own contiguous slot in the final prompt -- shift the
            # scored region's absolute positions up by len(chat_before_ids)
            # and let the wrapper tokens occupy [0, len(chat_before_ids))
            # and the trailing len(chat_after_ids) directly, matching
            # ordinary (unshifted) positions there since they're always
            # present, in the same place, every turn.
            offset = len(chat_before_ids)
            full_kept_positions = (
                list(range(len(chat_before_ids)))
                + [p + offset for p in result.kept_positions]
                + list(range(
                    offset + result.orig_len,
                    offset + result.orig_len + len(chat_after_ids),
                ))
            )
            full_orig_len = offset + result.orig_len + len(chat_after_ids)
            full_result = PrunedTurnResult(
                pruned_token_ids=prompt_ids,
                kept_positions=full_kept_positions,
                orig_len=full_orig_len,
                kept_history_pairs=result.kept_history_pairs,
                actual_look_ahead_cnt=result.actual_look_ahead_cnt,
                num_cached_tokens=result.num_cached_tokens,
            )
            prune_and_add_turn(llm.llm_engine, request_id, full_result, sampling_params)
            output = drive_single_request_to_completion(llm.llm_engine, request_id)

            if output is not None:
                completion = output.outputs[0]
                stats["out_lens"].append(len(completion.token_ids))
                stats["finish"][completion.finish_reason if completion.finish_reason in
                                 stats["finish"] else "other"] += 1
                if output.metrics is not None and output.metrics.first_token_latency:
                    stats["ttfts"].append(output.metrics.first_token_latency * 1000)
                if result.orig_len > 0:
                    stats["actual_keep_rates"].append(len(result.kept_positions) / result.orig_len)
                stats["num_cached_tokens_speculator"].append(result.num_cached_tokens)
                predictions.append({
                    "conversation_id": conv["id"], "turn_idx": turn_idx,
                    "config": conv["config"], "pred": completion.text,
                })

            golden_answer_ids = render_golden_answer(tok, turn)
            state.complete_turn(result.kept_history_pairs, golden_answer_ids)

        proposer.discard_conversation(conv["id"])

    return predictions, stats


def run_experiment(exp_id: str, exp_cfg: dict, args) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM
    from vllm_patch.config import SpecConfig

    label = exp_cfg["label"]
    mode = exp_cfg["mode"]
    keep_mode = exp_cfg["keep_mode"]
    keep_percentage = exp_cfg["keep_percentage"]
    granularity = exp_cfg["granularity"]

    print(f"\n{'=' * 70}\n{exp_id}: {label}\n{'=' * 70}")

    conversations = load_conversations(args.samples, args.max_conversations)
    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    # Clamp both max_num_batched_tokens and max_model_len to the checkpoint's
    # own native max_position_embeddings -- ported from the single-turn
    # pipeline's predict_longbench_v2.py (a real failure this fixes, not a
    # defensive guess): vLLM's own ModelConfig validation rejects a
    # user-specified max_model_len that exceeds the checkpoint's derived
    # ceiling outright (confirmed: --target-max-num-batched-tokens's default
    # 131072 + --max-tokens's default 64 = 131136, 64 over Llama-3.1-8B's
    # native 131072, on the very first M000 smoke test).
    # VLLM_ALLOW_LONG_MAX_MODEL_LEN would bypass this, but is NOT safe for a
    # RoPE model (positions beyond the trained range produce NaN, per vLLM's
    # own error message) -- not used here, same reasoning as the single-turn
    # pipeline. A single request can never need a batched-token budget larger
    # than the model's own max context anyway, so clamping only
    # max_model_len while leaving max_num_batched_tokens larger would be
    # internally inconsistent, not a real fix -- both are clamped together.
    native_max_model_len = AutoConfig.from_pretrained(
        args.target_model, trust_remote_code=True
    ).max_position_embeddings
    target_max_num_batched_tokens = args.target_max_num_batched_tokens
    if native_max_model_len is not None and target_max_num_batched_tokens > native_max_model_len:
        print(
            f"[predict_scbench] --target-max-num-batched-tokens "
            f"({target_max_num_batched_tokens}) exceeds {args.target_model}'s "
            f"native max_position_embeddings ({native_max_model_len}) -- "
            f"clamping down to it."
        )
        target_max_num_batched_tokens = int(native_max_model_len)

    max_model_len = target_max_num_batched_tokens + args.max_tokens
    if native_max_model_len is not None:
        max_model_len = min(max_model_len, int(native_max_model_len))

    llm_kwargs = dict(
        model=args.target_model,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=False,
        gpu_memory_utilization=args.target_gpu_memory_utilization,
        max_num_batched_tokens=target_max_num_batched_tokens,
        max_model_len=max_model_len,
    )
    if mode != "baseline":
        llm_kwargs["worker_cls"] = "vllm_patch.worker.SpecPrefillWorker"

    t_engine = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[predict_scbench] target engine constructed in {time.time() - t_engine:.1f}s")

    proposer = None
    spec_config = None
    speculator_max_num_batched_tokens = None
    if mode == "specprefill":
        import torch
        from vllm_patch.proposer import SpecPrefillProposer

        speculator_device = torch.device(args.speculator_device or "cuda:1")
        gran_kwargs = GRANULARITIES[granularity]
        spec_config = SpecConfig(
            keep_strategy="percentage",
            keep_kwargs={**gran_kwargs, "percentage": keep_percentage},
            look_ahead_cnt=LOOK_AHEAD_CNT,
            pool_kernel_size=POOL_KERNEL_SIZE,
            keep_mode=keep_mode,
        )

        # Same clamp-to-native-ceiling reasoning as the target above --
        # independently derived (not assumed identical to the target's, even
        # though both are Llama-3.x checkpoints here) since a different
        # speculator model could have a different native context window.
        # The speculator's OWN prompt is the FULL (never-pruned) candidate
        # pool + query (see proposer.py's module docstring) -- this is what
        # can genuinely exceed the budget for a huge SCBench context, well
        # before pruning ever gets a chance to shrink anything, so this
        # budget (not the target's) is the one that determines whether a
        # turn's SCORING pass is even possible at all.
        speculator_native_max_model_len = AutoConfig.from_pretrained(
            args.speculator_model, trust_remote_code=True
        ).max_position_embeddings
        speculator_max_num_batched_tokens = args.speculator_max_num_batched_tokens
        if (
            speculator_native_max_model_len is not None
            and speculator_max_num_batched_tokens > speculator_native_max_model_len
        ):
            print(
                f"[predict_scbench] --speculator-max-num-batched-tokens "
                f"({speculator_max_num_batched_tokens}) exceeds "
                f"{args.speculator_model}'s native max_position_embeddings "
                f"({speculator_native_max_model_len}) -- clamping down to it."
            )
            speculator_max_num_batched_tokens = int(speculator_native_max_model_len)

        # + LOOK_AHEAD_CNT + 1: proposer.py's run_turn submits the full
        # candidate_pool+query as this budget's own prompt, THEN generates
        # 1 + look_ahead_cnt more tokens (bootstrap + lookahead decode) from
        # it INSIDE the speculator's own engine -- that generation also
        # needs room within max_model_len, or run_turn itself would hit the
        # same "prompt + max_tokens exceeds max_model_len" error this whole
        # fix is about, just one level deeper (inside proposer.py instead of
        # here).
        speculator_max_model_len = min(
            speculator_max_num_batched_tokens + 1 + LOOK_AHEAD_CNT,
            int(speculator_native_max_model_len)
            if speculator_native_max_model_len is not None
            else speculator_max_num_batched_tokens + 1 + LOOK_AHEAD_CNT,
        )

        proposer = SpecPrefillProposer(
            speculator_model_path=args.speculator_model,
            device=speculator_device,
            gpu_memory_utilization=args.speculator_gpu_memory_utilization,
            max_num_batched_tokens=speculator_max_num_batched_tokens,
            max_model_len=speculator_max_model_len,
        )
    elif mode == "oracle":
        raise NotImplementedError(
            "Oracle-mode driving (teacher-forced target-side hook + "
            "pruner.compute_oracle_kept_pairs) is not wired up in this "
            "driver yet -- vllm_patch.pruner.compute_oracle_kept_pairs is "
            "ready to consume a (query_buffer, key_buffer_per_layer, "
            "actual_look_ahead_cnt) triple, but installing the query-"
            "capture hook on the TARGET's own attention layers (mirroring "
            "vllm_patch/speculator_worker.py's hook, but on "
            "vllm_patch/worker.py's SpecPrefillWorker instead) and driving "
            "the teacher-forced next-turn forward pass is real, not-yet-"
            "built plumbing -- see EXPERIMENT_PLAN.md's Oracle section. "
            "Run M000/M-k*-g* experiments first."
        )

    try:
        for rep in range(1, args.reps + 1):
            t0 = time.time()
            if mode == "baseline":
                predictions, stats = run_baseline(
                    llm, tok, conversations, args.max_tokens, keep_mode,
                    target_max_num_batched_tokens,
                )
            else:
                predictions, stats = run_specprefill(
                    llm, tok, proposer, spec_config, conversations, args.max_tokens, keep_mode,
                    speculator_max_num_batched_tokens, target_max_num_batched_tokens,
                )
            elapsed = time.time() - t0

            if rep == args.reps:
                pred_path = OUT_DIR / f"{exp_id}_predictions.jsonl"
                with open(pred_path, "w", encoding="utf-8") as f:
                    for row in predictions:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[predict_scbench] wrote {len(predictions)} predictions -> {pred_path}")

            ttfts_sorted = sorted(stats["ttfts"])
            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "exp_id": exp_id, "label": label, "mode": mode, "keep_mode": keep_mode,
                "keep_percentage": keep_percentage, "kv_granularity": granularity,
                "chunk_size": GRANULARITIES.get(granularity, {}).get("chunk_size") if granularity else None,
                "look_ahead_cnt": LOOK_AHEAD_CNT if mode != "baseline" else None,
                "pool_kernel_size": POOL_KERNEL_SIZE if mode != "baseline" else None,
                "target_gpu_memory_utilization": args.target_gpu_memory_utilization,
                "speculator_gpu_memory_utilization": args.speculator_gpu_memory_utilization if mode != "baseline" else None,
                "target_max_num_batched_tokens": target_max_num_batched_tokens,
                "rep": rep, "seed": 0, "max_tokens": args.max_tokens,
                "num_conversations": len(conversations),
                "num_turns": len(predictions),
                "num_skipped_too_large": stats["num_skipped_too_large"],
                "elapsed_time": elapsed,
                "turns_per_second": len(predictions) / elapsed if elapsed > 0 else None,
                "actual_keep_rate_mean": statistics.mean(stats["actual_keep_rates"]) if stats["actual_keep_rates"] else None,
                "ttft_mean_ms": statistics.mean(stats["ttfts"]) if stats["ttfts"] else None,
                "ttft_p50_ms": percentile(ttfts_sorted, 0.5),
                "ttft_p90_ms": percentile(ttfts_sorted, 0.9),
                "num_cached_tokens_speculator_mean": (
                    statistics.mean(stats["num_cached_tokens_speculator"])
                    if stats["num_cached_tokens_speculator"] else None
                ),
                "out_len_mean": statistics.mean(stats["out_lens"]) if stats["out_lens"] else None,
                "out_len_stdev": statistics.stdev(stats["out_lens"]) if len(stats["out_lens"]) > 1 else 0.0,
                "finish_stop": stats["finish"]["stop"],
                "finish_length": stats["finish"]["length"],
                "finish_other": stats["finish"]["other"],
            }
            append_csv_row(row)
            print(
                f"[predict_scbench] rep {rep}/{args.reps}: {row['num_turns']} turns "
                f"across {row['num_conversations']} conversations in {elapsed:.1f}s "
                f"({row['turns_per_second']:.2f} turns/s), "
                f"ttft_mean={row['ttft_mean_ms']}, "
                f"actual_keep_rate_mean={row['actual_keep_rate_mean']}, "
                f"num_cached_tokens_speculator_mean={row['num_cached_tokens_speculator_mean']}"
            )
    finally:
        del llm
        if proposer is not None:
            del proposer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp",
        help="Comma-separated experiment IDs (see --list), or one of the "
             "group keywords 'specprefill' (all M-k*-g* rows, i.e. "
             "everything except M000 baseline -- ORACLE-k* rows are NOT "
             "included, since that mode isn't wired up yet) or 'all' "
             "(every defined experiment, including the not-yet-implemented "
             "ORACLE-k* rows, which will fail and be reported, not silently "
             "skipped).",
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--target-model", default=os.environ.get("LLAMA31_8B_MODEL_PATH"))
    parser.add_argument("--speculator-model", default=os.environ.get("LLAMA32_1B_MODEL_PATH"))
    parser.add_argument("--speculator-device", default=None)
    parser.add_argument("--target-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--speculator-gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--target-max-num-batched-tokens", type=int, default=131072)
    parser.add_argument(
        "--speculator-max-num-batched-tokens", type=int, default=131072,
        help="Clamped down to the speculator checkpoint's own native "
             "max_position_embeddings if larger -- see run_experiment's "
             "specprefill branch. This is the budget that actually "
             "determines whether a huge SCBench conversation's turn can be "
             "SCORED at all, before any pruning shrinks what the target "
             "sees.",
    )
    parser.add_argument("--max-tokens", type=int, default=64,
                         help="Generation cap per turn.")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--max-conversations", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for exp_id, cfg in EXPERIMENTS.items():
            print(f"{exp_id}: {cfg['label']}")
        return

    if args.output_dir is not None:
        global OUT_DIR, CSV_PATH
        OUT_DIR = args.output_dir
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CSV_PATH = OUT_DIR / "all_runs.csv"

    if not args.exp:
        parser.error("--exp is required (or use --list)")
    if not args.target_model:
        parser.error("--target-model or $LLAMA31_8B_MODEL_PATH is required")

    ensure_csv_header()
    # Two convenience group keywords, alongside the normal comma-separated
    # explicit-id list -- "specprefill" is what you want for "everything
    # except baseline" in practice: it's every implemented, non-baseline
    # experiment (the 16 M-k*-g* rows). "all" includes the ORACLE-k* rows
    # too, which are NOT wired up yet (predict_scbench.py's oracle branch
    # raises NotImplementedError) -- included for completeness, not because
    # it's expected to succeed; each failure is caught per-experiment (see
    # the try/except below) and reported at the end, not fatal to the rest
    # of the sweep.
    exp_arg = args.exp.strip().lower()
    if exp_arg == "specprefill":
        exp_ids = [eid for eid, cfg in EXPERIMENTS.items() if cfg["mode"] == "specprefill"]
    elif exp_arg == "all":
        exp_ids = list(EXPERIMENTS.keys())
    else:
        exp_ids = [x.strip() for x in args.exp.split(",")]
    failed_exp_ids = []
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"Unknown experiment ID: {exp_id!r} (see --list)")
            sys.exit(1)
        exp_cfg = EXPERIMENTS[exp_id]
        if exp_cfg["mode"] != "baseline" and not args.speculator_model:
            parser.error(f"{exp_id} requires --speculator-model or $LLAMA32_1B_MODEL_PATH")
        try:
            run_experiment(exp_id, exp_cfg, args)
        except Exception as e:
            import traceback
            print(f"!!! experiment {exp_id} FAILED: {e}")
            traceback.print_exc()
            failed_exp_ids.append(exp_id)

    if failed_exp_ids:
        print(f"\n{len(failed_exp_ids)} experiment(s) failed: {failed_exp_ids}")
        sys.exit(1)


if __name__ == "__main__":
    main()
