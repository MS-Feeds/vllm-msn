"""Process-local SpecPrefill pruning registry.

Populated by `pruner.py` before `LLMEngine.add_request()`; consumed by
`model_runner.py`'s `SpecPrefillGPUModelRunner` to override RoPE positions for
pruned requests (see algorithm lines 18-19 -- "restore_pos_ids"/"merge_requests").

A plain process-local dict, not a distributed/shared-memory structure -- this
only works when the driver and the vLLM worker run in the same OS process
(true for the default single-GPU `UniProcExecutor`, see
`vllm/v1/executor/uniproc_executor.py:45`). Explicitly out of scope for
multiprocess/Ray executors, see EXPERIMENT_PLAN.md's "Implementation status"
and the approved plan's "Scope for this pass".

Lifecycle: `register()` when a pruned request is created (by `pruner.py`,
before `add_request`), `get()` read once per step by the runner subclass to
find that request's token range within `self.positions`, `discard_finished()`
called once per step by the runner subclass with
`SchedulerOutput.finished_req_ids` to avoid unbounded growth across a long
benchmark run (see the approved plan's risk #3).
"""

import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class PruneRecord:
    """T (kept_positions) and M (orig_len) for one pruned request.

    kept_positions: the original, scattered position ids of the N surviving
        (kept) prompt tokens, in the same order they were written into the
        pruned prompt_token_ids passed to add_request -- i.e.
        kept_positions[i] is the true original position of the token now at
        contiguous index i in the pruned prompt.
    orig_len: M, the original (unpruned) prompt length -- used to compute the
        decode-position offset (M - N) once prefill is done, per the
        approved plan's "Position override mechanism".
    """

    kept_positions: List[int]
    orig_len: int

    def __post_init__(self) -> None:
        if len(self.kept_positions) > self.orig_len:
            raise ValueError(
                f"kept_positions has {len(self.kept_positions)} entries, "
                f"more than orig_len={self.orig_len} -- cannot keep more "
                f"tokens than the original prompt had."
            )
        if any(p < 0 or p >= self.orig_len for p in self.kept_positions):
            raise ValueError(
                f"kept_positions must all be in [0, orig_len={self.orig_len}), "
                f"got {self.kept_positions}."
            )

    @property
    def num_kept(self) -> int:
        return len(self.kept_positions)

    @property
    def decode_offset(self) -> int:
        """M - N: added to stock (N-based) positions for decode-step tokens
        so they continue the *original* prompt's position numbering."""
        return self.orig_len - self.num_kept

    def positions_for_step(
        self, num_computed_before: int, num_scheduled_this_step: int
    ) -> Optional[List[int]]:
        """Original positions for this step's scheduled token range, or
        `None` if this step is pure decode (caller should add
        `decode_offset` instead).

        Handles the first prefill chunk AND any later prefill-continuation
        chunk with the same slicing logic -- chunked prefill splits a long
        pruned prompt's N kept tokens across multiple engine steps once N
        exceeds the scheduler's per-step token budget. Per vLLM's own
        scheduling invariant (confirmed against this fork's
        vllm/v1/core/sched/scheduler.py, 2026-07-23: a request's per-step
        scheduled token count during prefill is always
        min(remaining_prompt_tokens, token_budget)), a step's scheduled
        range for one request can reach but never straddle the pruned
        prefill/decode boundary (num_kept) -- so the result is always
        entirely within the kept-prefill region or entirely absent from it,
        never a mix.

        Returns None when num_computed_before >= num_kept -- covers both the
        first decode step (num_computed_before == num_kept exactly) and
        every later one. Preemption is not a special case: the scheduler
        resets a preempted request's num_computed_tokens to 0 on
        resumption, so a replayed prefill just re-enters this method from
        num_computed_before=0 against the same (unmodified) PruneRecord.
        """
        if num_computed_before >= self.num_kept:
            return None
        end = num_computed_before + num_scheduled_this_step
        if end > self.num_kept:
            raise ValueError(
                f"Pruned prefill chunk [{num_computed_before}, {end}) "
                f"overruns num_kept={self.num_kept} -- a step's scheduled "
                f"range should never straddle the pruned prefill/decode "
                f"boundary (see this method's docstring for the scheduler "
                f"invariant this assumes). Either that invariant no longer "
                f"holds, or num_computed_before was computed incorrectly by "
                f"the caller."
            )
        return self.kept_positions[num_computed_before:end]


_registry: Dict[str, PruneRecord] = {}
_lock = threading.Lock()


def register(request_id: str, kept_positions: List[int], orig_len: int) -> None:
    """Record a pruning decision for `request_id`. Must be called before the
    corresponding `LLMEngine.add_request(request_id, ...)` call."""
    record = PruneRecord(kept_positions=list(kept_positions), orig_len=orig_len)
    with _lock:
        _registry[request_id] = record


def get(request_id: str) -> Optional[PruneRecord]:
    """Returns None for any request that was never pruned (the common case)."""
    with _lock:
        return _registry.get(request_id)


def discard(request_id: str) -> None:
    with _lock:
        _registry.pop(request_id, None)


def discard_finished(finished_req_ids: Iterable[str]) -> None:
    with _lock:
        for request_id in finished_req_ids:
            _registry.pop(request_id, None)


def clear() -> None:
    """For tests only -- resets the whole registry."""
    with _lock:
        _registry.clear()
