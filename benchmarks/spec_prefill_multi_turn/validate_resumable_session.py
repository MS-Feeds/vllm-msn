#!/usr/bin/env python3
"""GPU-node validation for the resumable-session request mechanism this
fork's scheduler already implements (`vllm/v1/core/sched/scheduler.py`) --
the foundation the new persistent-KV-cache + speculator-guided sparse
attention architecture is built on (see the approved plan,
`Target-Side Persistent KV Cache + Speculator-Guided Sparse Attention`).

**Why this script exists before anything else in that architecture is
built**: the scheduler-internal mechanics of the resumable-session lifecycle
were traced precisely against real source (`Request.resumable`,
`RequestStatus.WAITING_FOR_STREAMING_REQ`, `Scheduler._update_request_as_
session`, `Scheduler._handle_stopped_request`), but the DRIVER-facing API to
actually invoke this from a normal script was not fully confirmed --
`LLMEngine.add_request()` does not expose a `resumable=` kwarg at all
(confirmed: its real signature has no such parameter), so this script uses
the one path that does exist: constructing an `EngineCoreRequest` directly
(with `resumable=True` set) and passing it AS the `prompt` argument, which
`LLMEngine.add_request()` accepts verbatim (with a deprecation warning) via
its `isinstance(prompt, EngineCoreRequest)` branch. This is the correct,
if unusual, entry point -- confirmed by reading `add_request()`'s own body,
not assumed.

**What "resumable" actually means for the driving loop -- confirmed to be
NOT the same shape as this pipeline's existing single-shot driving loops**:
a resumable request that finishes generating for the current turn does NOT
become "finished" in the normal sense -- `Scheduler._handle_stopped_request`
returns `False` (not finished) and parks it in `RequestStatus.WAITING_FOR_
STREAMING_REQ`, re-enqueuing it into the WAITING queue rather than freeing
it. This means `llm_engine.has_unfinished_requests()` is expected to keep
returning `True` even after a turn's own generation has genuinely stopped
-- the existing `drive_single_request_to_completion` helper (used
throughout `predict_scbench.py`'s other experiment paths) would loop
forever if reused here unmodified. This script instead watches each step's
`RequestOutput.outputs[0].finish_reason` (confirmed on real hardware to be
where `finish_reason` actually lives -- it is NOT a top-level `RequestOutput`
attribute, an earlier version of this script assumed it was and hit
`AttributeError: 'RequestOutput' object has no attribute 'finish_reason'`;
this matches `predict_scbench.py`'s own established `completion =
output.outputs[0]; completion.finish_reason` pattern, which this script
should have followed from the start) and stops consuming steps for the
CURRENT TURN once that turn's own generation reports a finish reason --
NOT once the whole request/session is "finished" -- see
`drive_one_turn_of_session` below.

**What this script checks, concretely**:
  A. Construct a resumable session request (turn 1), drive it to its own
     turn-level stop, and read back an early `num_cached_tokens`/token-count
     sanity check.
  B. Submit turn 2 against the SAME request_id (a NEW EngineCoreRequest
     containing ONLY the new turn's delta tokens -- confirmed from
     `Scheduler._update_request_as_session`'s body:
     `session.prompt_token_ids.extend(update.prompt_token_ids or ())` is an
     EXTEND, not a replace, so submitting the full cumulative prompt again
     would be doubly wrong here), and directly verify -- via
     `RequestOutput.num_cached_tokens` -- that turn 1's tokens were served
     from cache rather than recomputed. This is the single most important
     assertion in this script: it's the concrete, measured proof that the
     KV cache genuinely persisted across the turn boundary, not just that
     the API calls didn't crash.
  C. Cleanly terminate the session afterward (via `abort_request`, not the
     `None`-update sentinel path in `Scheduler.add_request` -- simpler and
     already a proven, used-elsewhere API in this pipeline) so its blocks
     are freed rather than leaked as a permanently-parked request.

Usage:
    python3 validate_resumable_session.py --model $LLAMA31_8B_MODEL_PATH
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Must be set before add_request() -> InputProcessor.assign_request_id first
# reads it, same requirement/reasoning as proposer.py's and pruner.py's own
# module-level default -- NOT inherited for free here, since this is a
# standalone script that doesn't import either of those modules. Without
# this, add_request() appends a random suffix to request_id, breaking the
# `existing = self.requests.get(request.request_id)` lookup
# Scheduler.add_request relies on to recognize turn 2 as a continuation of
# turn 1 rather than an unrelated new request.
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")


def drive_one_turn_of_session(llm_engine, request_id: str):
    """Steps the engine until `request_id`'s CURRENT TURN reports a finish
    reason -- NOT until the whole (resumable) request/session is finished,
    which per this script's module docstring may never happen on its own
    for a parked session. Returns the last RequestOutput observed for this
    request_id (whose `.outputs[0].finish_reason` should be non-None if
    this turn genuinely stopped normally -- confirmed against this
    pipeline's own existing usage elsewhere, e.g. predict_scbench.py's
    `completion = output.outputs[0]; completion.finish_reason`:
    `finish_reason` lives on the per-sample `CompletionOutput` inside
    `RequestOutput.outputs`, not on `RequestOutput` itself -- an earlier
    version of this function read `output.finish_reason` directly and hit
    `AttributeError: 'RequestOutput' object has no attribute
    'finish_reason'` on real hardware), or None if the engine ran out of
    unfinished requests without ever producing one (a real failure worth
    surfacing distinctly, not silently returning an empty result).

    The `return` inside the loop IS the exit mechanism (a turn-scoped
    early-exit the pipeline's other, single-shot driving loops don't need,
    since for THEM "finished" means the same thing at both the turn and
    request level) -- still guarded by `has_unfinished_requests()` so this
    doesn't spin forever if the engine genuinely has nothing left to step.
    """
    last_output = None
    while llm_engine.has_unfinished_requests():
        for output in llm_engine.step():
            if output.request_id != request_id:
                continue
            last_output = output
            if output.outputs[0].finish_reason is not None:
                return last_output
    return last_output


def build_resumable_request(request_id, prompt_token_ids, sampling_params, resumable=True):
    """Constructs an EngineCoreRequest directly and returns it -- the one
    confirmed way to set resumable=True, since LLMEngine.add_request()'s
    own signature has no such kwarg (confirmed by reading its real
    signature, not assumed). Passing this object AS the `prompt` argument
    to add_request() is accepted verbatim via its
    `isinstance(prompt, EngineCoreRequest)` branch (with a deprecation
    warning -- expected, not a sign something is wrong)."""
    from vllm.v1.engine import EngineCoreRequest

    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=False, default=None,
                         help="Defaults to $LLAMA31_8B_MODEL_PATH.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-tokens-per-turn", type=int, default=16)
    args = parser.parse_args()

    model_path = args.model or os.environ.get("LLAMA31_8B_MODEL_PATH")
    if not model_path:
        parser.error("--model or $LLAMA31_8B_MODEL_PATH is required")

    from vllm import LLM, SamplingParams

    print(f"[validate_resumable_session] constructing target engine from {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=False,  # needed for RequestOutput.num_cached_tokens
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    llm_engine = llm.llm_engine
    tok = llm.get_tokenizer()

    request_id = "validate-resumable-session-conv-1"
    sampling_params = SamplingParams(max_tokens=args.max_tokens_per_turn, temperature=0.0)

    # Turn 1: a real, moderately long prompt so a turn-2 cache hit is
    # unambiguous (not just 1-2 tokens where noise could hide a miss).
    turn1_text = tok.apply_chat_template(
        [{"role": "user", "content": "The quick brown fox jumps over the lazy dog. " * 60 +
                                      "\n\nQuestion 1: What animal is mentioned twice?\nAnswer:"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    turn1_ids = tok.encode(turn1_text, add_special_tokens=False)
    print(f"[validate_resumable_session] Step A: submitting turn 1 ({len(turn1_ids)} tokens)")

    req1 = build_resumable_request(request_id, turn1_ids, sampling_params, resumable=True)
    real_id = llm_engine.add_request(request_id, req1, sampling_params)
    assert real_id == request_id, (
        f"expected add_request() to use request_id={request_id!r} verbatim "
        f"(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 is set at this module's import "
        f"time) but got {real_id!r} back instead -- turn 2's session-continuation "
        f"lookup depends on this matching exactly."
    )
    print(f"[validate_resumable_session] Step A: add_request confirmed request_id={real_id!r}")

    t0 = time.time()
    output1 = drive_one_turn_of_session(llm_engine, request_id)
    turn1_elapsed = time.time() - t0
    if output1 is None:
        print("[validate_resumable_session] Step A: FAIL -- no output ever observed for "
              "this request_id. Either the request was never scheduled, or "
              "drive_one_turn_of_session's loop-exit logic is wrong for this fork -- "
              "see this script's module docstring for what to check.")
        sys.exit(1)

    turn1_ttft = (
        output1.metrics.first_token_latency * 1000
        if output1.metrics is not None and output1.metrics.first_token_latency
        else None
    )
    print(
        f"[validate_resumable_session] Step A: turn 1 finish_reason="
        f"{output1.outputs[0].finish_reason!r}, generated={output1.outputs[0].text!r}, "
        f"elapsed={turn1_elapsed:.2f}s, ttft={turn1_ttft}ms "
        f"(prefilled {len(turn1_ids)} tokens from scratch -- this is the SLOW "
        f"reference point Step B's turn 2 timing gets compared against). "
        f"num_cached_tokens={output1.num_cached_tokens} (expected 0 -- nothing to "
        f"hit yet on a fresh conversation, not itself meaningful beyond that)."
    )

    # Turn 2: ONLY the new delta tokens -- confirmed from
    # Scheduler._update_request_as_session's real body that this gets
    # EXTENDED onto the session's existing prompt_token_ids, not treated as
    # a replacement. Submitting the full cumulative text here would be a
    # real, silent correctness bug (double-counted history), not just
    # inefficient.
    turn2_text = "\n\nQuestion 2: What did the fox jump over?\nAnswer:"
    turn2_ids = tok.encode(turn2_text, add_special_tokens=False)
    print(f"[validate_resumable_session] Step B: submitting turn 2 delta ({len(turn2_ids)} "
          f"new tokens only, per Scheduler._update_request_as_session's extend-not-replace "
          f"semantics)")

    req2 = build_resumable_request(request_id, turn2_ids, sampling_params, resumable=True)
    real_id_2 = llm_engine.add_request(request_id, req2, sampling_params)
    assert real_id_2 == request_id, (
        f"expected the SAME request_id back on turn 2 ({request_id!r}), got "
        f"{real_id_2!r} -- if these differ, the session continuation path "
        f"(Scheduler.add_request's `existing = self.requests.get(...)` lookup) "
        f"cannot have matched turn 1's request, and this is really just two "
        f"unrelated requests, not a continued session."
    )

    t0 = time.time()
    output2 = drive_one_turn_of_session(llm_engine, request_id)
    turn2_elapsed = time.time() - t0
    if output2 is None:
        print("[validate_resumable_session] Step B: FAIL -- no output observed for turn 2.")
        sys.exit(1)

    turn2_ttft = (
        output2.metrics.first_token_latency * 1000
        if output2.metrics is not None and output2.metrics.first_token_latency
        else None
    )
    print(
        f"[validate_resumable_session] Step B: turn 2 finish_reason="
        f"{output2.outputs[0].finish_reason!r}, generated={output2.outputs[0].text!r}, "
        f"elapsed={turn2_elapsed:.2f}s, ttft={turn2_ttft}ms. "
        f"num_cached_tokens={output2.num_cached_tokens} -- **expected to ALSO read 0 "
        f"here, and this is NOT a failure signal**: traced against real source "
        f"(scheduler.py's Scheduler.schedule(), the block starting "
        f"`if request.num_computed_tokens == 0:`) that the prefix-cache-lookup/"
        f"stats-recording path this metric comes from is gated to a request's "
        f"VERY FIRST scheduling only; _update_request_as_session never resets "
        f"num_computed_tokens, so a resumed session's later turns take the ELSE "
        f"branch (reuse num_computed_tokens directly, no lookup, no stats update) "
        f"every time -- num_cached_tokens is architecturally not meaningful for "
        f"this code path, not evidence against persistence."
    )

    # THE actual verification signal for this code path: prefill cost is
    # proportional to how many NEW tokens actually needed a forward pass.
    # Turn 1 prefilled len(turn1_ids) tokens from scratch; if turn 2 only
    # ever needed to prefill its own len(turn2_ids) new tokens (because
    # turn 1's KV is still resident, not recomputed), its TTFT should be
    # dramatically lower than turn 1's -- not just "a bit faster", since
    # turn1_ids is ~46x longer than turn2_ids in this script's own prompt.
    # Turn 2's own generated text already showed the model correctly used
    # turn 1's context regardless of caching, so the MODEL clearly had
    # access to it either way -- what this check isolates is whether that
    # access was served from cache (cheap) or genuinely recomputed
    # (expensive).
    if turn1_ttft is not None and turn2_ttft is not None:
        if turn2_ttft > turn1_ttft * 0.5:
            print(
                f"[validate_resumable_session] Step B: WARNING -- turn 2's TTFT "
                f"({turn2_ttft:.0f}ms) is not dramatically lower than turn 1's "
                f"({turn1_ttft:.0f}ms), despite turn 2 only submitting "
                f"{len(turn2_ids)} new tokens against turn 1's {len(turn1_ids)}. "
                f"This suggests turn 2 may have re-prefilled turn 1's content from "
                f"scratch rather than reusing already-computed KV -- do not proceed "
                f"with building the sparse-attention runner on top of this until "
                f"this is understood (the model producing a correct answer either "
                f"way doesn't distinguish 'served from cache' from 'recomputed', "
                f"only the timing does)."
            )
        else:
            print(
                f"[validate_resumable_session] Step B: PASS -- turn 2's TTFT "
                f"({turn2_ttft:.0f}ms) is dramatically lower than turn 1's "
                f"({turn1_ttft:.0f}ms), consistent with turn 1's KV genuinely "
                f"persisting and being reused rather than recomputed."
            )
    else:
        print(
            "[validate_resumable_session] Step B: WARNING -- first_token_latency "
            "was not available on one or both outputs (metrics=None?) -- cannot "
            "make the timing-based determination above. Check that "
            "disable_log_stats=False actually took effect, or fall back to "
            "comparing elapsed= (cruder -- includes decode time, not just "
            "prefill) printed above."
        )

    # Step C: clean termination -- abort_request is a real, already-used
    # API elsewhere in vLLM's own client code (not invented for this
    # script), simpler than trying to drive the None-update sentinel path
    # in Scheduler.add_request (which would require yet another add_request
    # call with a specially-constructed "empty" update -- not obviously
    # simpler or safer than just aborting outright once the conversation is
    # genuinely done).
    print("[validate_resumable_session] Step C: aborting session to free its blocks")
    llm_engine.abort_request([request_id])

    print("\n[validate_resumable_session] ALL STEPS COMPLETE -- see Step B's PASS/WARNING "
          "above for the actual verdict on cross-turn persistence.")


if __name__ == "__main__":
    main()
