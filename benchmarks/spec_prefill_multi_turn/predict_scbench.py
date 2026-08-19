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
2. **Golden-context mode for `M-k*-g*`/oracle, NOT for `M000` baseline**
   (EXPERIMENT_PLAN.md's decision #1, since narrowed): for `run_specprefill`,
   each turn's target generation is what gets GRADED, but the token
   sequence fed forward into FUTURE turns' history/ledger is the dataset's
   own reference answer, not the model's own output -- this is what makes
   the whole per-turn token sequence for a conversation knowable statically
   up front for THAT path, see that decision's docstring for why this
   matters for tractability. `run_baseline` was changed to use
   self-generated history instead (its own actual completions, INCLUDING
   whatever special/EOS tokens the model generated -- see that function's
   own docstring), for the same reason `run_sparse_attention` already had
   to: it's what a real conversation actually looks like, and it removes a
   real confound when comparing `M000` against `SPARSE-k*-g*` (both now
   self-generated, so context-length growth rate no longer differs between
   them for reasons unrelated to either architecture -- previously `M000`'s
   golden answers were typically much shorter than `SPARSE`'s own
   `max_tokens`-bounded completions, making `SPARSE`'s conversations grow
   faster turn over turn independent of its sparse-attention design).
   `M-k*-g*`/oracle still need golden-context's "every turn's tokens known
   up front" property (DISCARD mode's candidate-pool bookkeeping and the
   pre-flight speculator-budget check both depend on it), so they keep it.
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

   **This now describes `run_specprefill` (M-k*-g*/ORACLE) ONLY.**
   `run_sparse_attention` always used genuine `<|eot_id|>`-delimited chat
   turns (it doesn't need the ledger to track wrapper boundaries -- see
   `chat_turn_boundary_pieces`), and `run_baseline` (M000) was changed to
   match it after a real graded run showed the rendering difference
   dominating the M000-vs-SPARSE comparison at every turn index except 0
   (see `run_baseline`'s own docstring for the numbers and the reasoning).
   The consequence, stated plainly because it's a real cost: M000 and
   SPARSE are no longer in SCBench's DEFAULT `use_chat_template=False`
   mode, so their absolute scores are no longer directly comparable to
   published SCBench numbers -- they are comparable to EACH OTHER, which
   is what those two rows exist for. M-k*-g*/ORACLE stay in the default
   mode and stay comparable to published numbers.

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
    python3 predict_scbench.py --exp SPARSE-k80-g32 --scbench-config scbench_kv
        # restrict to just one of the 3 bundled SCBench configs
    python3 predict_scbench.py --exp sparse --chunk-size 32
        # just the granularity=32 rows across every SPARSE-k*-g32 keep rate
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

from flops_model import (
    FlopBreakdown,
    dense_decode_attended_lens,
    model_flop_config,
    speculator_turn_flops,
    target_decode_flops,
    target_prefill_flops,
)

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
    "num_conversations_loaded", "num_conversations", "num_turns", "num_skipped_too_large",
    "elapsed_time", "turns_per_second", "seconds_per_conversation",
    "seconds_per_turn_mean", "seconds_per_turn_excl_turn0_mean",
    "actual_keep_rate_mean",
    "ttft_mean_ms", "ttft_p50_ms", "ttft_p90_ms",
    "num_cached_tokens_speculator_mean",
    "out_len_mean", "out_len_stdev", "out_tokens_per_second",
    "finish_stop", "finish_length", "finish_other",
    # Analytic FLOP accounting -- see flops_model.py's module docstring for
    # what is and isn't counted. Per-turn means (not totals) for the stage
    # breakdown, so rows with different turn counts stay comparable;
    # total_tflops is the whole experiment, matching elapsed_time's scope.
    "spec_prefill_tflops_per_turn_mean", "spec_lookahead_tflops_per_turn_mean",
    "spec_scoring_tflops_per_turn_mean", "target_prefill_tflops_per_turn_mean",
    "target_decode_tflops_per_turn_mean", "total_tflops_per_turn_mean",
    "total_tflops", "speculator_flops_fraction", "achieved_tflops_per_s", "mfu",
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
    # Persistent-KV-cache + speculator-guided sparse attention -- the
    # OTHER architecture (see EXPERIMENT_PLAN.md's separate section for it),
    # run alongside M-k*-g*/ORACLE-k* rather than replacing them, per the
    # user's own decision to keep the physically-pruned sweep as a
    # comparison baseline. Same keep-rate x granularity grid as M-k*-g*, but
    # "token" granularity is NOT offered here -- the block-table-gather
    # mechanism this architecture depends on is block-granular by
    # construction (see sparse_target_runner.py's module docstring), no
    # token-level path exists.
    for rate in KEEP_RATES:
        for gran_name in GRANULARITIES:
            if gran_name == "token":
                continue
            exp_id = f"SPARSE-k{int(rate * 100)}-g{gran_name}"
            experiments[exp_id] = {
                "label": f"Sparse attention (persistent cache) keep={int(rate * 100)}% "
                         f"granularity={gran_name}",
                "mode": "sparse", "keep_mode": "keep",
                "keep_percentage": rate, "granularity": gran_name,
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


def load_conversations(
    path: Path, max_keep: int, config_filter: Optional[set] = None
) -> list[dict]:
    """`config_filter`, if given, keeps only rows whose `"config"` field is
    in the set -- e.g. `{"scbench_kv"}` to benchmark just one of the 3
    MVP configs `scbench_samples.jsonl` bundles together, without needing
    to re-run `prep_scbench.py` (which would re-hit the HF Hub) just to
    get a single-config file. Filtering happens BEFORE `max_keep` is
    applied, not after -- so `--max-conversations N` means "N
    conversations of the config(s) actually being benchmarked", not "N
    conversations from the front of the mixed file, some of which might
    then get filtered away leaving fewer than N or even zero"."""
    conversations = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if config_filter is not None and row.get("config") not in config_filter:
                continue
            conversations.append(row)
            if max_keep >= 0 and len(conversations) >= max_keep:
                break
    if not conversations:
        if config_filter is not None:
            raise FileNotFoundError(
                f"No rows matching config(s) {sorted(config_filter)!r} found "
                f"in {path} -- check --scbench-config against the actual "
                f"\"config\" values present in that file (e.g. `cut -d'\"' "
                f"-f4 {path} | sort -u` lists them), or that the file isn't "
                f"simply empty/missing (run datasets/prep_scbench.py first)."
            )
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


_ASSISTANT_BOUNDARY_PLACEHOLDER = "@@SPEC_PREFILL_MT_ASSISTANT_PLACEHOLDER@@"
_USER_BOUNDARY_PLACEHOLDER = "@@SPEC_PREFILL_MT_USER_PLACEHOLDER@@"


def chat_turn_boundary_pieces(tok) -> str:
    """The text that closes an assistant turn and opens the next user turn
    (e.g. Llama's `<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n`)
    -- derived from the tokenizer's own chat template via placeholder
    substitution (same technique as `chat_wrapper_pieces` above), not
    hardcoded, so this isn't tied to Llama's specific special tokens.

    Only used by the sparse-attention experiment path (`run_sparse_
    attention` below) -- that path uses GENUINE per-turn chat-template
    boundaries (real `<|eot_id|>`-delimited turns), unlike M-k*-g*/M000's
    flattened-text-block rendering (module docstring #3's documented
    simplification, and its real-hardware-observed turn-5 confusion cost --
    see that docstring section). The sparse-attention path can afford real
    chat-template turns because it doesn't need `conversation_state.py` to
    track per-turn wrapper-token boundaries the way physical pruning would
    (see EXPERIMENT_PLAN.md's sparse-attention section: nothing is ever
    pruned from the target's own resident cache in that architecture, so
    there's no candidate-pool bookkeeping that the wrapper tokens would
    need to be excluded from)."""
    rendered = tok.apply_chat_template(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": _ASSISTANT_BOUNDARY_PLACEHOLDER},
            {"role": "user", "content": _USER_BOUNDARY_PLACEHOLDER},
        ],
        add_generation_prompt=False,
        tokenize=False,
    )
    after_assistant = rendered.split(_ASSISTANT_BOUNDARY_PLACEHOLDER, 1)[1]
    boundary = after_assistant.split(_USER_BOUNDARY_PLACEHOLDER, 1)[0]
    return boundary


def build_conversation_state(context: str, tok, keep_mode: str, conversation_id: str):
    from vllm_patch.conversation_state import ConversationState

    context_ids = tok.encode(context, add_special_tokens=False)
    return ConversationState(conversation_id, context_ids, keep_mode)


def render_turn_query(tok, turn_idx: int, turn: dict) -> list[int]:
    """This turn's own force-kept region. The `Question N:`/`Answer N:`
    plain-text framing is kept for EVERY path, including the two that now
    use real chat-template turn boundaries (`run_baseline`,
    `run_sparse_attention`) -- it's SCBench's own question framing, not a
    substitute for turn structure, and dropping it would change the task
    text itself rather than just the rendering. See module docstring #3 for
    which paths wrap it in real `<|eot_id|>` turns and which keep the whole
    conversation inside one flattened chat message."""
    question_text = f"\n\nQuestion {turn_idx + 1}: {turn['input']}\nAnswer {turn_idx + 1}:"
    return tok.encode(question_text, add_special_tokens=False)


def render_golden_answer(tok, turn: dict) -> list[int]:
    return tok.encode(f" {turn['answer']}", add_special_tokens=False)


def build_turn_delta_ids(
    turn_idx: int,
    query_ids: list[int],
    chat_before_ids: list[int],
    context_ids: list[int],
    chat_after_ids: list[int],
    turn_boundary_ids: list[int],
) -> list[int]:
    """The token delta one turn contributes to a genuine
    `<|eot_id|>`-delimited chat stream: turn 0 opens the conversation with
    `chat_before + context + query + chat_after`, every later turn closes
    the previous assistant turn and opens a new user turn with
    `turn_boundary + query + chat_after`. The model's own generated tokens
    land between one turn's delta and the next.

    **Shared by `run_baseline` (M000) and `run_sparse_attention`
    (SPARSE-k*) so the two cannot drift.** They arrive at the same stream
    by different mechanics -- baseline resubmits the whole accumulated
    prompt as a fresh one-shot request each turn, while the sparse path
    submits only this delta into a persistent, resumable session -- and
    the point of M000 as a control is that both nonetheless see the same
    tokens (see `run_baseline`'s docstring for the rendering confound this
    removes, and `test_vllm_patch.py`'s `test_baseline_and_sparse_build_
    the_same_chat_stream` for the property checked directly).

    `context_ids`/`chat_before_ids` are ignored for `turn_idx > 0`, since
    the conversation is already open by then -- passed unconditionally so
    callers don't have to branch before calling."""
    if turn_idx == 0:
        return list(chat_before_ids) + list(context_ids) + list(query_ids) + list(chat_after_ids)
    return list(turn_boundary_ids) + list(query_ids) + list(chat_after_ids)


def drive_single_request_to_completion(llm_engine, request_id: str):
    """Same discipline as predict_longbench_v2.py's drive_engine_to_completion
    (never break early, never abort) -- specialized to exactly one in-flight
    request, per module docstring #1's MVP scope.

    Logs the prefill/decode split -- added specifically to let
    baseline/specprefill/sparse's target-side "weight-loading is a shared,
    roughly-constant per-decode-step floor; sparse additionally pays for
    per-layer metadata-patch bookkeeping" theory (discussed at length
    interactively) be checked against real numbers instead of taken on
    faith: baseline and specprefill's DECODE-only time here should be
    directly comparable to sparse's own decode-only time (see
    `drive_one_turn_of_session` below) if that theory is right, since both
    run through the exact same target model/layer count -- if sparse's is
    noticeably higher, that's the metadata-patch overhead (also now
    directly measured, see `sparse_target_runner.py::pop_override_timing`)
    showing up on top of the same shared floor."""
    latest_output = None
    t_start = time.time()
    t_prefill_done = None
    while llm_engine.has_unfinished_requests():
        for output in llm_engine.step():
            if output.request_id == request_id:
                if latest_output is None:
                    t_prefill_done = time.time()
                    print(
                        f"[predict_scbench] {request_id!r}: target prefill "
                        f"done in {t_prefill_done - t_start:.2f}s"
                    )
                latest_output = output
    if t_prefill_done is not None and latest_output is not None:
        print(
            f"[predict_scbench] {request_id!r}: target decode done in "
            f"{time.time() - t_prefill_done:.2f}s "
            f"({len(latest_output.outputs[0].token_ids)} tokens generated)"
        )
    return latest_output


def drive_one_turn_of_session(llm_engine, request_id: str):
    """Same helper as `validate_resumable_session.py`'s own
    `drive_one_turn_of_session` -- confirmed on real hardware there that a
    resumable request's own turn-level stop does NOT make
    `has_unfinished_requests()` go False (it parks in `RequestStatus.
    WAITING_FOR_STREAMING_REQ` and re-enqueues, not finishes), so this
    watches `output.outputs[0].finish_reason` directly instead of relying
    on `drive_single_request_to_completion`'s "loop until nothing's left"
    shape, which would spin forever here.

    Logs the prefill/decode split -- "prefill" here means the DELTA
    (this turn's new tokens) only, since everything earlier is already
    resident in the session's own persistent cache, not a full-context
    prefill -- see `drive_single_request_to_completion`'s own docstring
    for why this split exists (checking the shared-decode-floor /
    sparse-specific-overhead theory against real numbers)."""
    last_output = None
    t_start = time.time()
    t_prefill_done = None
    while llm_engine.has_unfinished_requests():
        for output in llm_engine.step():
            if output.request_id != request_id:
                continue
            if last_output is None:
                t_prefill_done = time.time()
                print(
                    f"[predict_scbench] {request_id!r}: target delta-prefill "
                    f"done in {t_prefill_done - t_start:.2f}s"
                )
            last_output = output
            if output.outputs[0].finish_reason is not None:
                if t_prefill_done is not None:
                    print(
                        f"[predict_scbench] {request_id!r}: target decode "
                        f"done in {time.time() - t_prefill_done:.2f}s"
                    )
                return last_output
    return last_output


def build_sparse_session_request(llm_engine, request_id, prompt_token_ids, sampling_params, resumable=True):
    """Same technique as `validate_resumable_session.py`'s own
    `build_resumable_request` -- `LLMEngine.add_request()` has no
    `resumable=` kwarg (confirmed by reading its real signature), so this
    constructs an `EngineCoreRequest` directly and passes it AS the
    `prompt` argument, which `add_request()` accepts verbatim via its
    `isinstance(prompt, EngineCoreRequest)` branch (`request = prompt`,
    `llm_engine.py:233` -- confirmed by reading the real source, not
    assumed).

    **Real bug this fixes, confirmed via real-hardware evidence (not
    theoretical)**: that verbatim-acceptance branch means
    `self.input_processor.process_inputs(...)` -- the NORMAL path's own
    request construction -- never runs for a directly-built
    `EngineCoreRequest`. That method is the ONLY place `SamplingParams.
    update_from_generation_config(...)` gets called
    (`input_processor.py:315-318`), which is what populates a request's
    real stop-token set from the model's own `generation_config.json`
    (for a Llama-3.x-Instruct model, this includes the CHAT-TEMPLATE's own
    turn-ending token, e.g. `<|eot_id|>`, IN ADDITION to the base
    tokenizer's `eos_token_id` -- confirmed by reading `SamplingParams.
    update_from_generation_config`'s real body: without this call,
    `_eos_token_id` is never set and `stop_token_ids` never gets the
    chat-specific ids added). A bare `SamplingParams(max_tokens=...,
    temperature=...)`, as constructed in `run_sparse_attention`, has NONE
    of this -- so the target's own real turn-ending token was never
    recognized as a stop condition for ANY sparse-session turn.

    Real symptom this produced (from an actual predictions file): the
    model would generate a correct, concise answer, then -- having no
    recognized way to stop -- continue sampling from a now
    completely-off-distribution continuation, degenerating into repeating
    the literal word "assistant"/"Assistant" (or, in one turn, a
    different repetition loop) until hitting `max_tokens`. This also fully
    explains the earlier "sparse decode consistently takes ~1.5-1.6s /
    ~63 steps" finding -- it was never really about weight-loading or
    per-layer overhead, it was hitting the token cap on effectively every
    turn because it structurally could never stop early.

    Fixed by reusing vLLM's own exact logic (not reimplementing it,
    which could silently drift from whatever a future vLLM version
    changes here) via the already-constructed engine's own
    `input_processor` -- the same `generation_config_fields`/`renderer`/
    `tokenizer` the NORMAL `add_request()` path would have used for this
    exact model."""
    import time as _time

    from vllm.v1.engine import EngineCoreRequest

    input_processor = llm_engine.input_processor
    sampling_params.update_from_generation_config(
        input_processor.generation_config_fields,
        input_processor.renderer.get_eos_token_id(),
    )
    if input_processor.tokenizer is not None:
        sampling_params.update_from_tokenizer(input_processor.tokenizer)

    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=_time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
    )


class LedgerToTargetPositionMap:
    """Translates `conversation_state.py`'s pure-CONTENT ledger positions
    (context + turn queries + turn outputs only -- see that module's
    docstring) into the sparse-attention target session's own real
    token-stream positions, which additionally contain chat-template
    wrapper tokens (`chat_before`/`chat_after`/`turn_boundary`, from
    `chat_wrapper_pieces`/`chat_turn_boundary_pieces`) interspersed between
    turns. `sparse_target_runner.py`'s gather logic reads positions in the
    TARGET's own numbering (`self.input_batch.num_computed_tokens_cpu`
    counts every real submitted token, wrapper included) -- a
    speculator-selected CONTENT position must be translated before being
    registered via `sparse_selection_registry`, or it would point at the
    wrong physical block once enough wrapper tokens have accumulated.

    Tracks a monotonic step function: the offset (target position - ledger
    position) only ever increases, and only at the exact ledger positions
    where a wrapper insertion happens -- recorded via `add_wrapper` each
    time this driver actually submits one."""

    def __init__(self, initial_offset: int):
        # initial_offset = len(chat_before_ids) -- ledger position 0
        # (context's first token) maps to target position
        # len(chat_before_ids), since chat_before precedes everything.
        self._breakpoints = [(0, initial_offset)]

    def add_wrapper(self, ledger_position: int, wrapper_len: int) -> None:
        """Call once, right after submitting `wrapper_len` target-only
        tokens that logically sit at `ledger_position` (i.e. every ledger
        position >= `ledger_position` from now on is shifted by an
        additional `wrapper_len`)."""
        current_offset = self._breakpoints[-1][1]
        self._breakpoints.append((ledger_position, current_offset + wrapper_len))

    def translate(self, ledger_position: int) -> int:
        offset = self._breakpoints[0][1]
        for threshold, off in self._breakpoints:
            if ledger_position >= threshold:
                offset = off
            else:
                break
        return ledger_position + offset

    def wrapper_target_spans(self) -> list[tuple[int, int]]:
        """Every `[start, end)` span of TARGET positions occupied by
        wrapper tokens recorded so far, in ascending order -- the exact
        complement of what `translate` can ever produce.

        **Why this exists -- a real, confirmed correctness bug, not
        completeness for its own sake.** `translate(p) = p + offset` with
        `offset >= initial_offset = len(chat_before_ids)`, so the smallest
        target position any translated ledger position can EVER reach is
        `len(chat_before_ids)`. Target positions `[0, len(chat_before_ids))`
        -- `<|begin_of_text|>` and the whole system header, ~40 tokens for
        Llama-3.1, i.e. block 0 and its neighbours -- were therefore
        unreachable by `run_sparse_attention`'s registered selection at
        EVERY keep rate, and so were dropped from decode-time attention
        whenever the gather fired at all. Dropping the initial sink tokens
        from what a decoder may attend to is the canonical trigger for
        fluent-start-then-repetition-loop degeneration (StreamingLLM's
        attention-sink result), and it matched the observed symptom
        exactly: clean `M000` (which never gathers), repetition loops on
        every `SPARSE-k*` row. The same reachability argument applies to
        every mid-stream `chat_after_ids`/`turn_boundary_ids` insertion --
        those are target-only tokens with no ledger position at all, so
        the turn structure itself (`<|eot_id|>`, the assistant generation
        header) was equally unselectable.

        `run_specprefill` never had this bug because it builds its own
        `full_kept_positions` with `list(range(len(chat_before_ids)))`
        prepended explicitly (see its call site); `run_sparse_attention`
        registered bare `translate`d positions instead. This method is the
        sparse path's equivalent, derived from the SAME breakpoint
        bookkeeping `translate` uses so the two can't drift apart.

        Derivation: a breakpoint `(ledger_pos, offset_i)` following
        `(_, offset_prev)` means ledger position `ledger_pos - 1` sits at
        target `ledger_pos - 1 + offset_prev` while `ledger_pos` sits at
        `ledger_pos + offset_i`, so the wrapper tokens submitted between
        them occupy `[ledger_pos + offset_prev, ledger_pos + offset_i)` --
        exactly `offset_i - offset_prev == wrapper_len` positions.
        """
        spans = [(0, self._breakpoints[0][1])]
        for (_, prev_offset), (ledger_pos, offset) in zip(
            self._breakpoints, self._breakpoints[1:]
        ):
            spans.append((ledger_pos + prev_offset, ledger_pos + offset))
        return [(start, end) for start, end in spans if end > start]

    def wrapper_target_positions(self) -> list[int]:
        """Flattened `wrapper_target_spans()` -- the positions to union
        into a registered sparse selection so the chat-template scaffolding
        (attention sink included) is always attendable. See
        `wrapper_target_spans`'s docstring for why this is load-bearing."""
        return [p for start, end in self.wrapper_target_spans() for p in range(start, end)]


def _progress_postfix(
    predictions: list, stats: dict, t_loop_start: float, conversations_processed: int
) -> dict:
    """Shared `tqdm.set_postfix` payload for `run_baseline`/`run_specprefill`/
    `run_sparse_attention`'s per-conversation progress bars. `s/conv`
    deliberately does NOT use tqdm's own built-in rate (`n / elapsed`,
    where `n` advances once per LOADED conversation, including ones that
    get skipped and complete almost instantly -- exactly the same "total
    vs. processed" distortion the CSV's `seconds_per_conversation` field
    exists to avoid, see `run_experiment`'s own comment on that field) --
    computed here from `conversations_processed` (conversations that
    contributed >=1 turn to `predictions`) instead, so it reads as real
    per-conversation cost even while a run is skipping a lot of oversized
    conversations (baseline especially, see `run_baseline`'s docstring).
    `None` (omitted) until the first conversation actually completes, to
    avoid a misleadingly tiny number from a single fast conversation."""
    postfix = {"turns": len(predictions), "skipped": stats["num_skipped_too_large"]}
    if conversations_processed > 0:
        postfix["s/conv"] = f"{(time.time() - t_loop_start) / conversations_processed:.1f}"
    return postfix


_TFLOP = 1e12


def _flop_summary_fields(turn_flops, elapsed, peak_tflops) -> dict:
    """Rolls per-turn `FlopBreakdown`s into this experiment's CSV columns.

    Stage columns are per-turn MEANS so rows with different turn counts (a
    skipped oversized conversation costs turns) stay comparable;
    `total_tflops` is the whole experiment, matching `elapsed_time`'s own
    scope.

    `achieved_tflops_per_s` divides by the FULL experiment elapsed time --
    the same "over the whole experiment's elapsed time" convention
    `turns_per_second`/`out_tokens_per_second` already use, including
    tokenization, scoring, and every host-side gap. It is therefore a
    PIPELINE-level throughput number, not a claim about kernel efficiency,
    and will read well below peak for that reason. Its real job is the
    falsification bound: if it ever exceeds the device's peak, the FLOP
    model is provably wrong (a transposed dimension, a duplicated
    num_layers factor) -- no profiler needed to catch that class of bug.
    """
    if not turn_flops:
        return {f: None for f in (
            "spec_prefill_tflops_per_turn_mean", "spec_lookahead_tflops_per_turn_mean",
            "spec_scoring_tflops_per_turn_mean", "target_prefill_tflops_per_turn_mean",
            "target_decode_tflops_per_turn_mean", "total_tflops_per_turn_mean",
            "total_tflops", "speculator_flops_fraction", "achieved_tflops_per_s", "mfu",
        )}

    total = FlopBreakdown()
    for bd in turn_flops:
        total += bd
    n = len(turn_flops)
    total_tflops = total.total / _TFLOP
    achieved = total_tflops / elapsed if elapsed > 0 else None
    return {
        "spec_prefill_tflops_per_turn_mean": total.spec_prefill / n / _TFLOP,
        "spec_lookahead_tflops_per_turn_mean": total.spec_lookahead / n / _TFLOP,
        "spec_scoring_tflops_per_turn_mean": total.spec_scoring / n / _TFLOP,
        "target_prefill_tflops_per_turn_mean": total.target_prefill / n / _TFLOP,
        "target_decode_tflops_per_turn_mean": total.target_decode / n / _TFLOP,
        "total_tflops_per_turn_mean": total.total / n / _TFLOP,
        "total_tflops": total_tflops,
        "speculator_flops_fraction": total.speculator_fraction,
        "achieved_tflops_per_s": achieved,
        "mfu": (achieved / peak_tflops) if (achieved and peak_tflops) else None,
    }


def _target_flop_config(llm):
    """Target model's `ModelFlopConfig`, from the engine's own resolved
    config -- same `llm_engine.vllm_config.model_config.hf_config` access
    path `sparse_decode_microbench.py` uses for `bytes_per_token_kv`, so the
    two derived quantities can never disagree about the model's shape."""
    return model_flop_config(llm.llm_engine.vllm_config.model_config.hf_config)


def _speculator_flop_config(proposer):
    """Speculator's `ModelFlopConfig`, read from its own persistent engine
    (`proposer.llm_engine`, see `vllm_patch/proposer.py`) rather than
    re-loading the checkpoint's config from disk."""
    return model_flop_config(proposer.llm_engine.vllm_config.model_config.hf_config)


def _record_turn_flops(stats, breakdown: FlopBreakdown, **flop_inputs) -> dict:
    """Accumulates one turn's FLOP breakdown into `stats` and returns the
    fields to merge into that turn's prediction record.

    `flop_inputs` (the raw token counts the breakdown was computed from) is
    carried into the JSONL alongside the FLOPs deliberately: the model is
    analytic, so a surprising FLOP number is only debuggable if the inputs
    that produced it are visible next to it. Cheap -- a handful of ints per
    turn."""
    stats["flops"].append(breakdown)
    return {"flops": breakdown.as_dict(), "flop_inputs": flop_inputs}


def _num_decode_steps(out_len: int) -> int:
    """Decode steps for `out_len` generated tokens.

    One less: the first output token is sampled from the prefill's own
    logits row, which `target_prefill_flops` already charges."""
    return max(out_len - 1, 0)


def run_baseline(
    llm, tok, conversations, max_tokens, target_max_num_batched_tokens
) -> tuple[list[dict], dict]:
    """M000: plain add_request per turn, no worker_cls/proposer/pruning --
    keeps every token of the conversation unconditionally.

    **Renders GENUINE per-turn chat-template boundaries, identical to
    `run_sparse_attention`'s own submitted token stream** -- not the
    flattened "whole conversation as plain text inside one user message"
    form this function used to build (module docstring #3's documented
    simplification). Changed deliberately, to remove a confound that made
    the M000-vs-SPARSE comparison unreadable at every turn index except 0.

    Why it mattered: at turn 0 the two paths already produced byte-identical
    prompts (`chat_before + context + query + chat_after`), so turn 0 was
    the only apples-to-apples row in the sweep. From turn 1 onward the old
    flattened rendering diverged from SPARSE's real `<|eot_id|>`-delimited
    turns, and a real graded run showed M000 decaying across turn index
    (70/74/61/59/51 on `scbench_kv`) while every SPARSE row stayed flat and
    high (e.g. k80: 64/81/81/84/86) -- i.e. SPARSE appeared to BEAT dense
    attention by 20-35 points at turns 1-4, which sparse attention cannot
    do on the same prompt. That gap was the rendering, not the attention
    (consistent with module docstring #3's own separately-observed "turn-5
    confusion cost" for the flattened form). With both paths on real chat
    turns, a per-turn-index difference between M000 and SPARSE is
    attributable to the attention gather again.

    Note this leaves `run_specprefill` (M-k*-g*/ORACLE) still on the
    flattened rendering AND still on golden-answer history (module
    docstring #2) -- those rows are comparable to each other, not to M000
    or SPARSE, and were not touched here.

    **The ledger (`ConversationState`) is no longer used by this path.**
    Its only jobs here were producing the flattened prompt and estimating
    prompt length; the real submitted stream (`chat_ids` below) now does
    both, exactly rather than approximately. Keeping a parallel ledger that
    nothing reads would be exactly the two-places-tracking-the-same-thing
    drift hazard `LedgerToTargetPositionMap`'s docstring warns about.
    `run_specprefill`/`run_sparse_attention` still use it, since both need
    the pure-content position numbering the speculator scores against.

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

    Computed from `chat_ids` (the real, wrapper-inclusive stream submitted
    so far) plus this turn's own not-yet-appended delta, so it counts
    exactly what `add_request` will receive -- no `wrapper_overhead`
    estimate to drift, which is the same class of bug `run_sparse_
    attention`'s own docstring records having to fix (a single constant
    increasingly undercounted a growing conversation and let a real run
    overrun the budget deep into a sweep instead of skipping cleanly).
    Since nothing ever shrinks the stream in baseline mode, once one turn
    is too large every later turn in the same conversation will be too;
    skipping the rest of that conversation rather than checking turn by
    turn avoids paying for tokenization + a doomed size check repeatedly.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    chat_before, chat_after = chat_wrapper_pieces(tok)
    chat_before_ids = tok.encode(chat_before, add_special_tokens=False)
    chat_after_ids = tok.encode(chat_after, add_special_tokens=False)
    # Same pieces, from the same helpers, in the same order as
    # `run_sparse_attention` -- the whole point of this path's rendering
    # change is that the two produce the same token stream.
    turn_boundary_ids = tok.encode(chat_turn_boundary_pieces(tok), add_special_tokens=False)

    predictions = []
    stats = {"ttfts": [], "out_lens": [], "finish": {"stop": 0, "length": 0, "other": 0},
              "actual_keep_rates": [], "num_cached_tokens_speculator": [],
              "num_skipped_too_large": 0, "turn_elapsed": [], "flops": []}
    target_flop_cfg = _target_flop_config(llm)

    t_loop_start = time.time()
    conversations_processed = 0
    progress = tqdm(conversations, desc="M000 baseline", unit="conv")
    for conv in progress:
        progress.set_postfix(_progress_postfix(predictions, stats, t_loop_start, conversations_processed))
        turns_before = len(predictions)
        # The real submitted token stream, grown in place exactly as
        # `run_sparse_attention` grows its persistent session: turn 0
        # contributes `chat_before + context + query + chat_after`, every
        # later turn contributes `turn_boundary + query + chat_after`, and
        # each turn's own generated tokens land in between. This IS the
        # prompt -- there is no separate ledger to keep in sync (see this
        # function's docstring).
        context_ids = tok.encode(conv["context"], add_special_tokens=False)
        chat_ids: list[int] = []
        for turn_idx, turn in enumerate(conv["turns"]):
            t_turn_start = time.time()
            query_ids = render_turn_query(tok, turn_idx, turn)

            delta_ids = build_turn_delta_ids(
                turn_idx=turn_idx, query_ids=query_ids,
                chat_before_ids=chat_before_ids, context_ids=context_ids,
                chat_after_ids=chat_after_ids, turn_boundary_ids=turn_boundary_ids,
            )
            prospective_len = len(chat_ids) + len(delta_ids)
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

            chat_ids = chat_ids + delta_ids
            prompt_ids = chat_ids

            request_id = f"{conv['id']}::turn{turn_idx}"
            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
            prompt = TokensPrompt(prompt_token_ids=prompt_ids)
            t_gen_start = time.time()
            llm.llm_engine.add_request(request_id, prompt, sampling_params)
            output = drive_single_request_to_completion(llm.llm_engine, request_id)
            progress.write(
                f"[predict_scbench] {conv['id']!r} turn {turn_idx}: target "
                f"generation done in {time.time() - t_gen_start:.2f}s "
                f"({len(prompt_ids)} prompt tokens)"
            )

            actual_output_ids: list[int] = []
            if output is not None:
                completion = output.outputs[0]
                # Real model generation, NOT golden_answer_ids -- unlike
                # run_specprefill/the oracle path (still golden-context, see
                # module docstring #2), baseline feeds its own actual
                # output forward into future turns' ledger/context, same
                # self-generated-history principle run_sparse_attention
                # already uses (see that function's own docstring) and for
                # the same reason: it's what a REAL multi-turn conversation
                # actually looks like, not a golden-context simplification.
                # `completion.token_ids` is used AS-IS, not re-tokenized
                # from `.text` -- includes the model's own EOS/special
                # tokens verbatim when generation stopped on one (vLLM's
                # token_ids always reflects the raw sampled sequence;
                # `skip_special_tokens` only affects `.text` decoding, never
                # `.token_ids`), since a real conversation's own history
                # would include whatever the model actually produced,
                # special tokens included, not a cleaned-up version of it.
                # This is a ONE-SHOT (non-resumable) request per turn here
                # (unlike run_sparse_attention's persistent session), so
                # `completion.token_ids` is this turn's own output only --
                # no cumulative-output slicing needed (see that function's
                # own docstring for why THAT pipeline needs it and this one
                # doesn't).
                new_output_ids = list(completion.token_ids)
                stats["out_lens"].append(len(new_output_ids))
                # Drop the last generated token before appending to the
                # conversation, matching `run_sparse_attention` exactly.
                # There it's forced: `_update_request_as_session` discards
                # the last sampled token (its KV was never computed), so
                # the sparse target's real resident history simply does not
                # contain it. Mirrored here for two reasons, one of them
                # load-bearing rather than cosmetic:
                #   1. Comparability -- if M000's history carried one extra
                #      token per turn that SPARSE's physically cannot, this
                #      change would trade the rendering confound it exists
                #      to remove for a smaller one in the other direction.
                #   2. Well-formedness -- when a turn stops on the model's
                #      own `<|eot_id|>`, that token is the one dropped, and
                #      `turn_boundary_ids` supplies its own `<|eot_id|>`
                #      immediately after. Keeping both would emit a DOUBLE
                #      end-of-turn marker into the chat stream every turn.
                # Applied unconditionally (not just when the last token is
                # an EOS) precisely because the sparse path drops it
                # unconditionally: on a `length`-capped turn both paths
                # then lose the same single token, instead of diverging on
                # exactly the turns most likely to be scored differently.
                actual_output_ids = new_output_ids[:-1]
                stats["finish"][completion.finish_reason if completion.finish_reason in
                                 stats["finish"] else "other"] += 1
                if output.metrics is not None and output.metrics.first_token_latency:
                    stats["ttfts"].append(output.metrics.first_token_latency * 1000)
                stats["actual_keep_rates"].append(1.0)
                stats["turn_elapsed"].append((turn_idx, time.time() - t_turn_start))

                # M000 pays no speculator cost at all -- every spec_* stage
                # stays zero, which is exactly what makes this row the
                # denominator every other row's speculator_flops_fraction
                # is judged against.
                #
                # `num_cached_tokens` is READ, not assumed: this path never
                # sets enable_prefix_caching explicitly, but vLLM v1
                # defaults it on, and each turn resubmits the whole
                # cumulative conversation -- so most of a later turn's
                # prompt is a cache hit and charging it as fresh prefill
                # would wildly over-count the baseline it's the reference
                # for.
                num_cached = output.num_cached_tokens or 0
                out_len = len(new_output_ids)
                bd = FlopBreakdown(
                    target_prefill=target_prefill_flops(
                        target_flop_cfg, len(prompt_ids), num_cached),
                    target_decode=target_decode_flops(
                        target_flop_cfg,
                        dense_decode_attended_lens(
                            len(prompt_ids), _num_decode_steps(out_len)),
                    ),
                )
                flop_fields = _record_turn_flops(
                    stats, bd,
                    target_prompt_len=len(prompt_ids),
                    target_cached_tokens=num_cached,
                    decode_steps=_num_decode_steps(out_len),
                )
                predictions.append({
                    "conversation_id": conv["id"], "turn_idx": turn_idx,
                    "config": conv["config"], "pred": completion.text,
                    **flop_fields,
                })

            # Append this turn's own output to the conversation, so the
            # next turn's `turn_boundary_ids` closes a real assistant turn.
            # Empty when `output is None` (nothing generated), which just
            # leaves the stream where it was.
            chat_ids = chat_ids + actual_output_ids

        if len(predictions) > turns_before:
            conversations_processed += 1

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
              "num_skipped_too_large": 0, "turn_elapsed": [], "flops": []}

    target_flop_cfg = _target_flop_config(llm)
    spec_flop_cfg = _speculator_flop_config(proposer)

    keep_pct = spec_config.keep_kwargs.get("percentage")
    desc = f"SpecPrefill keep={keep_pct}"
    t_loop_start = time.time()
    conversations_processed = 0
    progress = tqdm(conversations, desc=desc, unit="conv")
    for conv in progress:
        progress.set_postfix(_progress_postfix(predictions, stats, t_loop_start, conversations_processed))
        turns_before = len(predictions)
        state = build_conversation_state(conv["context"], tok, keep_mode, conv["id"])
        for turn_idx, turn in enumerate(conv["turns"]):
            t_turn_start = time.time()
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

            t_scoring_start = time.time()
            result = compute_pruned_turn(proposer, spec_config, state, query_ids)
            progress.write(
                f"[predict_scbench] {conv['id']!r} turn {turn_idx}: speculator "
                f"scoring done in {time.time() - t_scoring_start:.2f}s "
                f"(kept {len(result.kept_positions)}/{result.orig_len})"
            )
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
            t_gen_start = time.time()
            prune_and_add_turn(llm.llm_engine, request_id, full_result, sampling_params)
            output = drive_single_request_to_completion(llm.llm_engine, request_id)
            progress.write(
                f"[predict_scbench] {conv['id']!r} turn {turn_idx}: target "
                f"generation done in {time.time() - t_gen_start:.2f}s"
            )

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
                stats["turn_elapsed"].append((turn_idx, time.time() - t_turn_start))

                # The whole point of this row: the speculator pays a FULL
                # prefill over the unpruned candidate pool (pruning shrinks
                # what the TARGET sees, never what the speculator must
                # process to decide the pruning -- the same asymmetry the
                # two-stage length check above exists for), and the target
                # pays a SHRUNK dense prefill over `prompt_ids`. Whether
                # that trade wins at this keep rate is exactly the
                # arithmetic these two numbers settle.
                num_cached = output.num_cached_tokens or 0
                out_len = len(completion.token_ids)
                bd = speculator_turn_flops(
                    spec_flop_cfg,
                    pool_len=result.orig_len,
                    num_cached=result.num_cached_tokens,
                    look_ahead=result.actual_look_ahead_cnt,
                )
                bd.target_prefill = target_prefill_flops(
                    target_flop_cfg, len(prompt_ids), num_cached)
                bd.target_decode = target_decode_flops(
                    target_flop_cfg,
                    dense_decode_attended_lens(
                        len(prompt_ids), _num_decode_steps(out_len)),
                )
                flop_fields = _record_turn_flops(
                    stats, bd,
                    spec_pool_len=result.orig_len,
                    spec_cached_tokens=result.num_cached_tokens,
                    spec_look_ahead=result.actual_look_ahead_cnt,
                    target_prompt_len=len(prompt_ids),
                    target_cached_tokens=num_cached,
                    decode_steps=_num_decode_steps(out_len),
                )
                predictions.append({
                    "conversation_id": conv["id"], "turn_idx": turn_idx,
                    "config": conv["config"], "pred": completion.text,
                    **flop_fields,
                })

            golden_answer_ids = render_golden_answer(tok, turn)
            state.complete_turn(result.kept_history_pairs, golden_answer_ids)

        proposer.discard_conversation(conv["id"])
        if len(predictions) > turns_before:
            conversations_processed += 1

    return predictions, stats


def run_sparse_attention(
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
    """SPARSE-k*-g*: persistent full-KV-cache target session, speculator-
    selected sparse attention over it during decode (see
    `vllm_patch/sparse_target_runner.py`'s module docstring for the
    mechanism, validated end-to-end on real hardware by
    `validate_resumable_session.py`/`validate_sparse_attention.py`).
    Structurally different from `run_specprefill` in three ways:

    1. **One target session (one resumable `request_id`) per conversation,
       not one `add_request` per turn.** Turn 2+ delivers only the NEW
       content (this turn's wrapper/query tokens) as a session update --
       everything from earlier turns is already resident in the target's
       own KV cache, never resubmitted.
    2. **The target always receives the FULL, unpruned content** (context
       on turn 0, this turn's query every turn, joined by real
       chat-template turn boundaries via `chat_turn_boundary_pieces` --
       NOT the flattened-text-block rendering `run_baseline`/
       `run_specprefill` use, see that function's own docstring for why
       this pipeline can afford real per-turn boundaries).
       `compute_pruned_turn`'s `pruned_token_ids` (the physically-shrunk
       subset the OTHER pipeline sends to the target) is not used for the
       target's own prompt here -- only `kept_positions` (which tokens the
       speculator judged important) is used, to build the decode-time
       attention selection registered via `register_sparse_selection`.
    3. **Self-generated history**: `state.complete_turn` is fed the
       target's own actual generated output for this turn, not the
       dataset's golden answer -- this pipeline's target session already
       auto-appends its own output tokens to its real KV cache (that's
       what "persistent" means here), so `conversation_state`'s ledger
       must mirror the SAME tokens to stay in lockstep with
       `LedgerToTargetPositionMap`'s bookkeeping. Golden answers are only
       used for grading (`grade_scbench.py` reads the dataset directly),
       never fed into this driver's own state.

    A `LedgerToTargetPositionMap` per conversation translates
    `conversation_state`'s pure-content ledger positions into the target
    session's real (wrapper-inclusive) token-stream positions before every
    `register_sparse_selection` call -- see that class's docstring. The
    map's own wrapper breakpoints are recorded in the same order those
    wrapper tokens are actually submitted to the target: `chat_before_ids`
    is folded into the map's `initial_offset` (always present, before
    anything else); `turn_boundary_ids` (turn 1+) is recorded right before
    translating that turn's own selection, since the selection includes
    this turn's own force-kept query, which sits AFTER the boundary in the
    target's real stream; `chat_after_ids` is recorded right after, since
    it only affects positions belonging to LATER turns (this turn's own
    generated output, appended once decoding finishes).

    **Two-stage length check, same reasoning as `run_specprefill`'s own
    docstring** -- check 1 (speculator budget) is identical. Check 2
    differs from `run_specprefill`'s: there is no pruned/shrunk prompt to
    fall back on here, so it checks the FULL prospective target session
    length (same shape as `run_baseline`'s single check), since this
    pipeline's target session is never pruned, only its ATTENTION is
    restricted -- it needs the whole conversation to fit physically, same
    as the baseline.

    **Real bug this fixes, confirmed via a real crash** (`ValueError:
    could not broadcast input array from shape (131084,) into shape
    (131072,)` inside `gpu_input_batch.py::add_request`, i.e. the target
    session's real length overran `--target-max-num-batched-tokens`
    despite this check supposedly guarding against exactly that): an
    earlier version of this check used a single constant `wrapper_overhead
    = len(chat_before_ids) + len(chat_after_ids) + len(turn_boundary_ids)`
    (one of EACH wrapper piece), but the REAL target session accumulates
    one `chat_after_ids` per turn AND one `turn_boundary_ids` per turn
    (from turn 1 onward) -- see this function's own turn loop, which
    submits `turn_boundary_ids + query_ids + chat_after_ids` as `delta_ids`
    EVERY turn, not once. A single constant therefore increasingly
    UNDERcounted the true resident length as a conversation progressed
    (by roughly `turn_idx * (chat_after_len + turn_boundary_len)` tokens),
    letting a conversation slip past this guard and crash the engine deep
    into a run instead of being cleanly skipped up front. Fixed by
    computing the check from `position_map` itself -- the SAME
    wrapper-accumulation bookkeeping already used for translating
    `kept_positions` below, so there's only one place tracking "how many
    wrapper tokens are actually resident so far," not two that can drift
    apart.
    """
    from vllm import SamplingParams
    from vllm_patch.conversation_state import ConversationState
    from vllm_patch.pruner import compute_pruned_turn

    chat_before, chat_after = chat_wrapper_pieces(tok)
    chat_before_ids = tok.encode(chat_before, add_special_tokens=False)
    chat_after_ids = tok.encode(chat_after, add_special_tokens=False)
    turn_boundary_ids = tok.encode(chat_turn_boundary_pieces(tok), add_special_tokens=False)

    predictions = []
    stats = {"ttfts": [], "out_lens": [], "finish": {"stop": 0, "length": 0, "other": 0},
              "actual_keep_rates": [], "num_cached_tokens_speculator": [],
              "num_skipped_too_large": 0, "turn_elapsed": [], "flops": []}

    target_flop_cfg = _target_flop_config(llm)
    spec_flop_cfg = _speculator_flop_config(proposer)

    keep_pct = spec_config.keep_kwargs.get("percentage")
    desc = f"Sparse attention keep={keep_pct}"
    t_loop_start = time.time()
    conversations_processed = 0
    progress = tqdm(conversations, desc=desc, unit="conv")
    for conv in progress:
        progress.set_postfix(_progress_postfix(predictions, stats, t_loop_start, conversations_processed))
        turns_before = len(predictions)

        context_ids = tok.encode(conv["context"], add_special_tokens=False)
        state = ConversationState(conv["id"], context_ids, keep_mode)
        position_map = LedgerToTargetPositionMap(initial_offset=len(chat_before_ids))
        target_request_id = f"{conv['id']}::sparse-session"
        session_started = False
        # RequestOutput.outputs[0].token_ids/.text are CUMULATIVE for the
        # WHOLE request's lifetime under this engine's default (non-DELTA)
        # output_kind -- confirmed by reading output_processor.py's
        # RequestState/_new_completion_output (token_ids = self.detokenizer.
        # output_token_ids when not delta) and detokenizer.py's
        # IncrementalDetokenizer (self.output_text += ...) -- NEITHER is
        # reset by apply_streaming_update/`_update_request_as_session` on a
        # session resumption, so turn 2's `output` already contains turn 1's
        # tokens/text prepended, turn 3's contains turns 1+2's, etc. Track
        # how many output tokens existed before THIS turn so each turn's
        # genuinely NEW tokens can be sliced out -- real hardware evidence
        # this matters (not a theoretical concern): feeding the raw
        # cumulative slice into state.complete_turn re-appended every prior
        # turn's output to the ledger each turn, a compounding drift that
        # crashed sparse_target_runner.py's block-index bounds check a few
        # turns into a real SCBench conversation (kept_positions translated
        # to a target position far beyond anything actually computed).
        prev_cumulative_output_len = 0
        # Target-side resident KV length, for FLOP accounting only. Tracked
        # explicitly rather than read from `RequestOutput.num_cached_tokens`
        # because this path is a resumable SESSION, not a fresh request per
        # turn: the prior history is already resident in the session's own
        # KV, which is a different thing from a prefix-cache hit, and the
        # two are not interchangeable in `num_cached_tokens`. This is the
        # `n_cached` that `delta_ids`' prefill attention runs against.
        target_resident_len = 0

        for turn_idx, turn in enumerate(conv["turns"]):
            t_turn_start = time.time()
            query_ids = render_turn_query(tok, turn_idx, turn)

            prospective_speculator_len = state.total_len + len(query_ids)
            if prospective_speculator_len > speculator_max_num_batched_tokens:
                progress.write(
                    f"[predict_scbench] SKIP conversation id={conv['id']!r} "
                    f"from turn {turn_idx} onward: full candidate-pool length "
                    f"{prospective_speculator_len} exceeds "
                    f"--speculator-max-num-batched-tokens="
                    f"{speculator_max_num_batched_tokens} -- the speculator "
                    f"must process the whole thing to score it."
                )
                stats["num_skipped_too_large"] += 1
                break
            # position_map.translate(state.total_len) gives the REAL target-
            # stream position of "everything resident so far" (context +
            # every prior turn's query/output, PLUS every prior turn's own
            # chat_after_ids/turn_boundary_ids wrapper insertions already
            # accounted for by position_map's breakpoints) -- see this
            # function's own docstring for why a single constant
            # `wrapper_overhead` used to undercount this. What's added on
            # top here is exactly this turn's own not-yet-submitted delta:
            # turn_boundary_ids (turn_idx > 0 only) + this turn's query +
            # chat_after_ids, mirroring delta_ids's own construction below.
            prospective_target_len = (
                position_map.translate(state.total_len)
                + (len(turn_boundary_ids) if turn_idx > 0 else 0)
                + len(query_ids)
                + len(chat_after_ids)
            )
            if prospective_target_len > target_max_num_batched_tokens:
                progress.write(
                    f"[predict_scbench] SKIP conversation id={conv['id']!r} "
                    f"from turn {turn_idx} onward: full target session length "
                    f"{prospective_target_len} would exceed "
                    f"--target-max-num-batched-tokens="
                    f"{target_max_num_batched_tokens} -- this pipeline's "
                    f"target session is never pruned, only its ATTENTION is "
                    f"restricted, so it needs the whole conversation to fit "
                    f"physically, same as the baseline."
                )
                stats["num_skipped_too_large"] += 1
                break

            t_scoring_start = time.time()
            result = compute_pruned_turn(proposer, spec_config, state, query_ids)
            progress.write(
                f"[predict_scbench] {conv['id']!r} turn {turn_idx}: speculator "
                f"scoring done in {time.time() - t_scoring_start:.2f}s "
                f"(kept {len(result.kept_positions)}/{result.orig_len})"
            )
            query_start_ledger_pos = result.orig_len - len(query_ids)

            if turn_idx > 0:
                position_map.add_wrapper(query_start_ledger_pos, len(turn_boundary_ids))

            # Record THIS turn's own chat_after_ids BEFORE translating, not
            # after. Safe -- a breakpoint at `result.orig_len` only shifts
            # ledger positions `>= result.orig_len`, and every entry in
            # `result.kept_positions` is `< result.orig_len` by
            # construction (`orig_len == force_keep_query[-1][1] + 1`, see
            # `pruner.py::_positions_from_kept_indices`), so no translated
            # position changes. Required, because those tokens ARE resident
            # for this turn's decode (they're part of `delta_ids` below,
            # submitted before generation starts), so they must appear in
            # `wrapper_target_positions()` when the selection is registered
            # a few lines down -- registering after would leave this turn's
            # own assistant generation header unattendable.
            position_map.add_wrapper(result.orig_len, len(chat_after_ids))

            # Union the speculator's own selection with every chat-template
            # wrapper span -- see `LedgerToTargetPositionMap.wrapper_target_
            # spans`'s docstring for the confirmed repetition-loop bug this
            # fixes (the attention sink at target position 0 was previously
            # unreachable by `translate`, so it was dropped from decode
            # attention at every keep rate). Mirrors `run_specprefill`'s own
            # `full_kept_positions`, which has always done this explicitly.
            translated_positions = sorted(
                set(position_map.translate(p) for p in result.kept_positions)
                | set(position_map.wrapper_target_positions())
            )
            llm.llm_engine.collective_rpc(
                "register_sparse_selection",
                args=(target_request_id, translated_positions),
            )

            # Shared with `run_baseline` so M000 and SPARSE submit the same
            # token stream -- see `build_turn_delta_ids`'s docstring.
            delta_ids = build_turn_delta_ids(
                turn_idx=turn_idx, query_ids=query_ids,
                chat_before_ids=chat_before_ids, context_ids=context_ids,
                chat_after_ids=chat_after_ids, turn_boundary_ids=turn_boundary_ids,
            )

            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
            prompt = build_sparse_session_request(
                llm.llm_engine, target_request_id, delta_ids, sampling_params, resumable=True,
            )
            t_gen_start = time.time()
            real_id = llm.llm_engine.add_request(target_request_id, prompt, sampling_params)
            assert real_id == target_request_id, (
                f"expected request_id={target_request_id!r} verbatim, got "
                f"{real_id!r} -- VLLM_DISABLE_REQUEST_ID_RANDOMIZATION must "
                f"be set (proposer.py sets this at import time)."
            )
            session_started = True
            output = drive_one_turn_of_session(llm.llm_engine, target_request_id)
            progress.write(
                f"[predict_scbench] {conv['id']!r} turn {turn_idx}: target "
                f"generation done in {time.time() - t_gen_start:.2f}s "
                f"({len(delta_ids)} new delta tokens submitted)"
            )
            # Direct measurement of the per-layer metadata-patch bookkeeping
            # cost hypothesized (and discussed at length interactively) as
            # sparse's own extra decode-step overhead on top of the
            # weight-loading floor it shares with baseline/specprefill --
            # see sparse_target_runner.py::pop_override_timing's docstring
            # for exactly what this measures (CPU-side dispatch time, not
            # confirmed GPU execution time) and why.
            override_total, override_steps = llm.llm_engine.collective_rpc(
                "pop_override_timing", args=(target_request_id,),
            )[0]
            if override_steps > 0:
                override_msg = (
                    f"{override_total:.3f}s across {override_steps} decode "
                    f"steps ({override_total / override_steps * 1000:.2f}ms/step avg)"
                )
            else:
                override_msg = "no decode steps recorded"
            progress.write(
                f"[predict_scbench] {conv['id']!r} turn {turn_idx}: sparse "
                f"metadata-patch overhead: {override_msg}"
            )
            # Per-decode-step attended KV length -- the one FLOP input that
            # can't be derived driver-side on this path, since it's decided
            # per step by how the selection maps onto block boundaries. Must
            # be popped BEFORE discard_sparse_selection below only for
            # symmetry with pop_override_timing; the accumulator itself is
            # keyed by request_id and independent of the registry. See
            # sparse_target_runner.py::pop_attended_lens.
            attended_lens = llm.llm_engine.collective_rpc(
                "pop_attended_lens", args=(target_request_id,),
            )[0]
            # attended_lens is recorded on a strict SUPERSET of the steps
            # override timing is: it's appended before the `gathered is
            # None` early-out (dense-fallback steps are real attention
            # work), while the timing accumulator only fires on steps that
            # actually patched metadata. So `>=` is the invariant, not
            # equality -- at keep_rate=1.0 every step is a dense fallback
            # and override_steps is legitimately 0 while attended_lens is
            # full. Fewer attended samples than patched steps, on the other
            # hand, means the accumulator is missing steps it should have
            # seen, and the decode FLOPs below are under-counted.
            if len(attended_lens) < override_steps:
                progress.write(
                    f"[predict_scbench] WARNING {conv['id']!r} turn "
                    f"{turn_idx}: attended-length samples ({len(attended_lens)}) "
                    f"< metadata-patch steps ({override_steps}) -- decode "
                    f"FLOPs for this turn are under-counted."
                )
            llm.llm_engine.collective_rpc(
                "discard_sparse_selection", args=(target_request_id,),
            )

            actual_output_ids: list[int] = []
            if output is not None:
                completion = output.outputs[0]
                # Slice out just THIS turn's new tokens from the cumulative
                # list -- see prev_cumulative_output_len's own comment above
                # for why the raw list can't be used directly.
                cumulative_output_ids = list(completion.token_ids)
                new_output_ids = cumulative_output_ids[prev_cumulative_output_len:]
                prev_cumulative_output_len = len(cumulative_output_ids)

                # Drop the last generated token before feeding the ledger --
                # `_update_request_as_session` (vllm/v1/core/sched/
                # scheduler.py) does the same when a session resumes
                # ("Discards the last sampled output token from the prior
                # input chunk"): that token was only ever SAMPLED, never fed
                # back into the model, so its own KV was never computed and
                # it is not physically part of the target's retained
                # context going forward. The ledger must mirror that or
                # LedgerToTargetPositionMap drifts by one token per turn.
                actual_output_ids = new_output_ids[:-1]

                stats["out_lens"].append(len(new_output_ids))
                stats["finish"][completion.finish_reason if completion.finish_reason in
                                 stats["finish"] else "other"] += 1
                if output.metrics is not None and output.metrics.first_token_latency:
                    stats["ttfts"].append(output.metrics.first_token_latency * 1000)
                if result.orig_len > 0:
                    stats["actual_keep_rates"].append(len(result.kept_positions) / result.orig_len)
                stats["num_cached_tokens_speculator"].append(result.num_cached_tokens)
                stats["turn_elapsed"].append((turn_idx, time.time() - t_turn_start))
                # completion.text is ALSO cumulative for the same reason --
                # re-decode just this turn's own new tokens rather than use
                # it directly.
                pred_text = tok.decode(new_output_ids, skip_special_tokens=True)

                # The structural asymmetry this row exists to expose: the
                # speculator is paid for IN FULL (same cost as the
                # SpecPrefill row), the target's PREFILL is fully dense
                # (the gather is decode-only -- sparse_target_runner.py
                # `continue`s while num_computed < num_prompt), and only
                # decode attention shrinks. Reading these three stages
                # against each other is what tells you whether the decode
                # saving can ever pay for the speculator at a given
                # context length.
                bd = speculator_turn_flops(
                    spec_flop_cfg,
                    pool_len=result.orig_len,
                    num_cached=result.num_cached_tokens,
                    look_ahead=result.actual_look_ahead_cnt,
                )
                bd.target_prefill = target_prefill_flops(
                    target_flop_cfg,
                    prompt_len=target_resident_len + len(delta_ids),
                    num_cached=target_resident_len,
                )
                # Measured per step, not derived -- these are the
                # block-padded lengths the kernel was actually handed.
                bd.target_decode = target_decode_flops(target_flop_cfg, attended_lens)
                flop_fields = _record_turn_flops(
                    stats, bd,
                    spec_pool_len=result.orig_len,
                    spec_cached_tokens=result.num_cached_tokens,
                    spec_look_ahead=result.actual_look_ahead_cnt,
                    target_resident_len=target_resident_len,
                    target_delta_len=len(delta_ids),
                    decode_steps=len(attended_lens),
                    decode_attended_len_mean=(
                        sum(attended_lens) / len(attended_lens) if attended_lens else 0
                    ),
                )
                predictions.append({
                    "conversation_id": conv["id"], "turn_idx": turn_idx,
                    "config": conv["config"], "pred": pred_text,
                    **flop_fields,
                })

            # Advance the resident-length tracker by exactly what this turn
            # added to the session's KV: the submitted delta, plus every
            # generated token whose KV was actually computed. That's
            # `actual_output_ids`, NOT `new_output_ids` -- the final sampled
            # token was never fed back through the model, so it has no KV,
            # which is the same reason it's dropped before feeding the
            # ledger a few lines above.
            target_resident_len += len(delta_ids) + len(actual_output_ids)

            state.complete_turn(result.kept_history_pairs, actual_output_ids)

        proposer.discard_conversation(conv["id"])
        if session_started:
            llm.llm_engine.abort_request([target_request_id])
        if len(predictions) > turns_before:
            conversations_processed += 1

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

    config_filter = (
        {c.strip() for c in args.scbench_config.split(",") if c.strip()}
        if args.scbench_config else None
    )
    conversations = load_conversations(args.samples, args.max_conversations, config_filter)
    if config_filter is not None:
        print(f"[predict_scbench] filtered to config(s) {sorted(config_filter)!r}: "
              f"{len(conversations)} conversations")
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
    if mode == "sparse":
        # SparseTargetWorker, not SpecPrefillWorker -- this path never
        # physically shrinks the prompt (no RoPE-position-override
        # machinery needed at all, see sparse_target_runner.py's module
        # docstring), it restricts decode-step attention over a session
        # that must stay resident turn over turn.
        # enable_prefix_caching=True mirrors validate_resumable_session.py/
        # validate_sparse_attention.py's own construction -- required for
        # the resumable-session mechanism this path depends on.
        llm_kwargs["worker_cls"] = "vllm_patch.sparse_target_runner.SparseTargetWorker"
        llm_kwargs["enable_prefix_caching"] = True
        # async_scheduling=False -- a real crash this fixes, not a
        # defensive guess. Real hardware hit `RuntimeError: Invalid
        # request status: RUNNING` inside Scheduler.schedule() on a
        # session's first resumption AFTER the EOS/stop-token fix started
        # letting turns stop naturally instead of always hitting
        # max_tokens. Root cause, confirmed by reading vLLM source:
        # UniProcExecutor.max_concurrent_batches returns 2 whenever
        # SchedulerConfig.async_scheduling is on (the default), which
        # routes EngineCore through step_with_batch_queue -- a PIPELINED
        # mode where AsyncScheduler._update_after_schedule optimistically
        # schedules a request's NEXT step before the CURRENT step's real
        # output (and thus whether it actually stopped) is known. For an
        # ordinary request that's fine (nothing else touches it once it's
        # truly finished). For a RESUMABLE session, `_handle_stopped_
        # request` immediately re-parks/re-enqueues the SAME request the
        # instant a real stop is observed -- while a "phantom" already-
        # pipelined step for that same request can still be in flight,
        # leading the scheduler to later find it already RUNNING when it
        # expected WAITING/PREEMPTED. This combination (async pipelining +
        # resumable sessions + a stop that can happen mid-pipeline, i.e.
        # a genuine EOS match rather than always hitting max_tokens) was
        # never exercised before the EOS fix, since every turn previously
        # ran to the token cap. Disabling async scheduling forces the
        # plain, non-pipelined `step()` path (`batch_queue` stays None),
        # sidestepping the race entirely -- a real perf cost (no overlap
        # between scheduling and execution) but a correctness requirement
        # for this pipeline's own resumable-session mechanism until/unless
        # this is fixed further upstream.
        llm_kwargs["async_scheduling"] = False
    elif mode != "baseline":
        llm_kwargs["worker_cls"] = "vllm_patch.worker.SpecPrefillWorker"

    t_engine = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[predict_scbench] target engine constructed in {time.time() - t_engine:.1f}s")

    proposer = None
    spec_config = None
    speculator_max_num_batched_tokens = None
    if mode in ("specprefill", "sparse"):
        # Same speculator construction for both -- the speculator's own
        # job (score importance via lookahead decode) is identical either
        # way, only how the TARGET consumes the resulting selection
        # differs (see run_sparse_attention's module docstring).
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
                    llm, tok, conversations, args.max_tokens,
                    target_max_num_batched_tokens,
                )
            elif mode == "sparse":
                predictions, stats = run_sparse_attention(
                    llm, tok, proposer, spec_config, conversations, args.max_tokens, keep_mode,
                    speculator_max_num_batched_tokens, target_max_num_batched_tokens,
                )
            else:
                predictions, stats = run_specprefill(
                    llm, tok, proposer, spec_config, conversations, args.max_tokens, keep_mode,
                    speculator_max_num_batched_tokens, target_max_num_batched_tokens,
                )
            elapsed = time.time() - t0
            # "Processed" == actually contributed >=1 turn to `predictions`
            # (not skipped from turn 0 onward by one of run_*'s pre-flight
            # length checks) -- rate/average metrics below (turns_per_second
            # already did this implicitly, since a prediction is only ever
            # appended for a turn that wasn't skipped) use ONLY this count,
            # not `len(conversations)` (the raw number loaded from the
            # samples file, kept separately as num_conversations_loaded for
            # context/skip-rate visibility) -- a conversation skipped
            # immediately (e.g. baseline against an oversized SCBench
            # context) costs near-zero wall time, so counting it in a
            # per-conversation denominator would understate real per-
            # conversation cost.
            num_conversations_processed = len({p["conversation_id"] for p in predictions})

            if rep == args.reps:
                pred_path = OUT_DIR / f"{exp_id}_predictions.jsonl"
                with open(pred_path, "w", encoding="utf-8") as f:
                    for row in predictions:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[predict_scbench] wrote {len(predictions)} predictions -> {pred_path}")

            ttfts_sorted = sorted(stats["ttfts"])
            # turn_idx == 0 pays each conversation's own "cold start" cost
            # (the full context's first prefill -- into the target directly
            # for baseline/specprefill, or into both the target session AND
            # the speculator's own growing cache for sparse) -- a
            # fundamentally different, much larger cost than any later
            # turn's incremental one. Averaging it in with every other turn
            # would make "time per turn" mostly reflect how big turn 0 was,
            # not the STEADY-STATE per-turn cost this metric exists to
            # show -- exclude it, per turn_idx, for every conversation (not
            # just the first one processed).
            turn_times_excl_first = [t for idx, t in stats["turn_elapsed"] if idx > 0]
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
                "num_conversations_loaded": len(conversations),
                "num_conversations": num_conversations_processed,
                "num_turns": len(predictions),
                "num_skipped_too_large": stats["num_skipped_too_large"],
                "elapsed_time": elapsed,
                "turns_per_second": len(predictions) / elapsed if elapsed > 0 else None,
                "seconds_per_conversation": (
                    elapsed / num_conversations_processed
                    if num_conversations_processed > 0 else None
                ),
                "seconds_per_turn_mean": (
                    statistics.mean(t for _, t in stats["turn_elapsed"])
                    if stats["turn_elapsed"] else None
                ),
                "seconds_per_turn_excl_turn0_mean": (
                    statistics.mean(turn_times_excl_first) if turn_times_excl_first else None
                ),
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
                # Total OUTPUT tokens generated / total wall time -- same
                # "over the whole experiment's elapsed time" convention as
                # turns_per_second (not decode-only: elapsed includes
                # prefill, speculator scoring for specprefill/sparse, etc.
                # -- this is throughput of the pipeline as actually run,
                # not a claim about raw decode speed in isolation).
                "out_tokens_per_second": (
                    sum(stats["out_lens"]) / elapsed if elapsed > 0 and stats["out_lens"] else None
                ),
                "finish_stop": stats["finish"]["stop"],
                "finish_length": stats["finish"]["length"],
                "finish_other": stats["finish"]["other"],
                **_flop_summary_fields(stats["flops"], elapsed, args.peak_tflops),
            }
            append_csv_row(row)
            spc = row["seconds_per_conversation"]
            spc_str = f"{spc:.1f}s/conversation" if spc is not None else "n/a s/conversation"
            spt = row["seconds_per_turn_mean"]
            spt_excl = row["seconds_per_turn_excl_turn0_mean"]
            spt_str = f"{spt:.2f}s/turn (all)" if spt is not None else "n/a s/turn"
            spt_excl_str = (
                f"{spt_excl:.2f}s/turn (excl. turn0)" if spt_excl is not None else "n/a s/turn (excl. turn0)"
            )
            otps = row["out_tokens_per_second"]
            otps_str = f"{otps:.1f} out tok/s" if otps is not None else "n/a out tok/s"
            print(
                f"[predict_scbench] rep {rep}/{args.reps}: {row['num_turns']} turns "
                f"across {row['num_conversations']} processed conversations "
                f"({row['num_conversations_loaded']} loaded, "
                f"{row['num_skipped_too_large']} skipped) in {elapsed:.1f}s "
                f"({row['turns_per_second']:.2f} turns/s, {spc_str}, {spt_str}, "
                f"{spt_excl_str}, {otps_str}), "
                f"ttft_mean={row['ttft_mean_ms']}, "
                f"actual_keep_rate_mean={row['actual_keep_rate_mean']}, "
                f"num_cached_tokens_speculator_mean={row['num_cached_tokens_speculator_mean']}"
            )
            if row["total_tflops_per_turn_mean"] is not None:
                print(
                    f"[predict_scbench] rep {rep}/{args.reps} FLOPs/turn (TFLOP): "
                    f"spec_prefill={row['spec_prefill_tflops_per_turn_mean']:.2f} "
                    f"spec_lookahead={row['spec_lookahead_tflops_per_turn_mean']:.2f} "
                    f"spec_scoring={row['spec_scoring_tflops_per_turn_mean']:.2f} "
                    f"target_prefill={row['target_prefill_tflops_per_turn_mean']:.2f} "
                    f"target_decode={row['target_decode_tflops_per_turn_mean']:.2f} "
                    f"| total={row['total_tflops_per_turn_mean']:.2f} "
                    f"speculator_share={row['speculator_flops_fraction']:.1%} "
                    f"achieved={row['achieved_tflops_per_s']:.2f} TFLOP/s"
                    + (f" mfu={row['mfu']:.1%}" if row["mfu"] is not None else "")
                )
                if args.peak_tflops and row["achieved_tflops_per_s"] > args.peak_tflops:
                    # §4c falsification bound: above-peak throughput is
                    # arithmetically impossible, so the model (not the
                    # hardware) is wrong. Costs nothing and catches the
                    # whole transposed-dimension class of bug without a
                    # profiler.
                    print(
                        f"[predict_scbench] WARNING: achieved_tflops_per_s="
                        f"{row['achieved_tflops_per_s']:.2f} EXCEEDS --peak-tflops="
                        f"{args.peak_tflops} -- the FLOP model is over-counting; "
                        f"treat every FLOP column in this row as wrong."
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
             "included, since that mode isn't wired up yet), 'sparse' (all "
             "SPARSE-k*-g* rows, the persistent-cache + sparse-attention "
             "architecture), or 'all' (every defined experiment, including "
             "the not-yet-implemented ORACLE-k* rows, which will fail and "
             "be reported, not silently skipped).",
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--scbench-config", default=None,
        help="Comma-separated SCBench config name(s) to restrict this run "
             "to (e.g. 'scbench_kv', or 'scbench_kv,scbench_summary') -- "
             "filters scbench_samples.jsonl's own \"config\" field, since "
             "that file bundles all 3 MVP configs (scbench_qa_eng/"
             "scbench_kv/scbench_summary) together by default. Omit to "
             "benchmark all configs present in the samples file (previous "
             "behavior, unchanged).",
    )
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
    parser.add_argument(
        "--peak-tflops", type=float, default=None,
        help="Device peak dense throughput in TFLOP/s, used only to emit the "
             "`mfu` column (e.g. 312 for A100 BF16). Left unset by default "
             "rather than guessed from the device name -- an MFU against the "
             "wrong peak is worse than no MFU. Does not affect any FLOP count.",
    )
    parser.add_argument("--max-conversations", type=int, default=-1)
    parser.add_argument(
        "--chunk-size", default=None,
        help="Restrict to experiment rows with this KV-entry granularity: "
             "'16', '32', '64', or 'token' (matches the g16/g32/g64/gtoken "
             "suffix in each experiment id). Combines with --exp's group "
             "keywords ('specprefill', 'sparse', 'all') to select e.g. "
             "just the granularity=32 rows across every keep rate, without "
             "hand-listing each M-k*-g32/SPARSE-k*-g32 id. M000 (no "
             "granularity at all) is excluded whenever this is set. Has no "
             "effect combined with an explicit --exp id list beyond "
             "filtering out any listed ids that don't match.",
    )
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
    elif exp_arg == "sparse":
        exp_ids = [eid for eid, cfg in EXPERIMENTS.items() if cfg["mode"] == "sparse"]
    elif exp_arg == "all":
        exp_ids = list(EXPERIMENTS.keys())
    else:
        exp_ids = [x.strip() for x in args.exp.split(",")]

    if args.chunk_size:
        # Unknown ids are deliberately kept here (not filtered away) so the
        # existing per-id "Unknown experiment ID" check below still catches
        # typos -- only ids that ARE recognized get judged against
        # --chunk-size, via each row's own "granularity" field (None for
        # M000, which this therefore always excludes).
        before = exp_ids
        exp_ids = [
            eid for eid in before
            if eid not in EXPERIMENTS or EXPERIMENTS[eid].get("granularity") == args.chunk_size
        ]
        print(f"[predict_scbench] --chunk-size {args.chunk_size!r}: "
              f"{len(exp_ids)}/{len(before)} experiment id(s) kept")
        if not exp_ids:
            parser.error(
                f"--chunk-size {args.chunk_size!r} matched no experiments "
                f"in the --exp selection {args.exp!r} -- valid granularity "
                f"values are '16', '32', '64', 'token' (see --list for "
                f"which ids have which)."
            )

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
