#!/usr/bin/env python3
"""Speculative-decoding metric collection for the evaluation pipeline.

vLLM exposes spec-decode stats as aggregate, engine-level Prometheus
counters -- there is no per-request field for this (see
vllm/v1/spec_decode/metrics.py: SpecDecodingProm). The three relevant
counters, confirmed against that source:

    vllm:spec_decode_num_drafts          -- number of speculation rounds
    vllm:spec_decode_num_draft_tokens    -- total tokens proposed
    vllm:spec_decode_num_accepted_tokens -- total tokens accepted

These are only *registered* when a SpeculativeConfig is present
(SpecDecodingProm.__init__ returns early otherwise) -- so their absence
from llm.get_metrics() is the actual signal that speculative decoding is
off, not an error to guard against separately.

snapshot_spec_decode_counters(llm) reads the current cumulative totals.
diff_spec_decode_counters(before, after) turns two snapshots into the
rates for the window between them, using the same formulas documented in
SpecDecodingProm's docstring and SpecDecodingLogging.log():

    acceptance_rate     = accepted_tokens / draft_tokens
    mean_accept_length  = 1 + (accepted_tokens / num_drafts)   # includes
                            the bonus (guaranteed) token, matching vLLM's
                            own convention.
"""

from __future__ import annotations

_COUNTER_NAMES = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)


def snapshot_spec_decode_counters(llm) -> dict[str, int] | None:
    """Returns {"num_drafts", "num_draft_tokens", "num_accepted_tokens"}
    summed across all engines/labels, or None if speculative decoding
    isn't configured on this LLM instance (the counters simply won't be
    present in get_metrics() in that case)."""
    from vllm.v1.metrics.reader import Counter

    totals = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
    }
    found = False
    for metric in llm.get_metrics():
        if not isinstance(metric, Counter) or metric.name not in _COUNTER_NAMES:
            continue
        found = True
        key = metric.name.removeprefix("vllm:spec_decode_")
        totals[key] += metric.value

    return totals if found else None


def decode_window_seconds(outputs) -> float | None:
    """Wall-clock duration of the decode phase for a batch of
    llm.generate() outputs, derived from each request's own first/last
    output-token timestamps rather than the whole call's elapsed time
    (which also includes prefill -- MBU is a decode-phase, memory-bound
    metric and should not be diluted by compute-bound prefill time).

    RequestOutput.metrics is a RequestStateStats (vllm/v1/metrics/stats.py),
    populated whenever log_stats is on (the default -- see
    EngineArgs.disable_log_stats). first_token_ts/last_token_ts are
    monotonic engine-core timestamps for when each request's first and
    last output tokens were produced; output_processor.py's own tracing
    code computes decode_time the same way (last_token_ts - first_token_ts)
    for a single request. Here we take the earliest first-token across the
    whole batch (roughly: when the batch's prefill finished and decode
    steps began) to the latest last-token (when the slowest request
    finished decoding), so the window covers exactly the span where decode
    steps were running.

    Returns None if per-request metrics aren't available (log_stats
    disabled) or the window collapses to zero (e.g. every request's
    generation was a single bonus token with no real decode step) --
    callers should report MBU as n/a rather than divide by zero."""
    starts = []
    ends = []
    for o in outputs:
        m = o.metrics
        if m is None or m.first_token_ts <= 0 or m.last_token_ts <= 0:
            return None
        starts.append(m.first_token_ts)
        ends.append(m.last_token_ts)

    if not starts:
        return None

    window = max(ends) - min(starts)
    return window if window > 0 else None


def diff_spec_decode_counters(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, float] | None:
    """Diffs two snapshots into acceptance_rate / mean_accept_length for
    the window between them (e.g. before/after one llm.generate() call).

    Returns None if speculative decoding wasn't active (either snapshot
    is None) or if no speculation rounds occurred in the window (e.g. a
    zero-token generation) -- both are "nothing to report" cases, not
    errors."""
    if before is None or after is None:
        return None

    draft_tokens = after["num_draft_tokens"] - before["num_draft_tokens"]
    accepted_tokens = after["num_accepted_tokens"] - before["num_accepted_tokens"]
    num_drafts = after["num_drafts"] - before["num_drafts"]

    if draft_tokens <= 0 or num_drafts <= 0:
        return None

    return {
        "acceptance_rate": accepted_tokens / draft_tokens,
        "mean_accept_length": 1 + (accepted_tokens / num_drafts),
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted_tokens,
        "num_drafts": num_drafts,
    }
