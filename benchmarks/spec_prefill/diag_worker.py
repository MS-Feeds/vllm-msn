"""Diagnostic worker_cls for repro_chunked_prefill_bug.py.

Purpose: find out whether concurrently-submitted pruned requests that get
their prefill split across >1 scheduler step (because they lost the shared
token-budget race to other concurrently-running requests, NOT because any
single one exceeds max_num_batched_tokens on its own) trigger the
already-confirmed-on-stock-vLLM "prefill never resumes past its first
chunk" bug documented in validate_runner_integration.py's module docstring
(finding #2, 2026-07-23).

Not something vllm_patch/ itself should carry permanently -- this is a
one-off diagnostic, kept as a separate worker_cls/runner subclass so
production code (vllm_patch/model_runner.py, worker.py) stays untouched.
Points repro_chunked_prefill_bug.py's `worker_cls` at
`diag_worker.DiagnosticSpecPrefillWorker` instead of
`vllm_patch.worker.SpecPrefillWorker` for this run only.

**Must live in its own importable module file, not be defined inline in a
driver script.** `parallel_config.worker_cls` is a dotted path resolved
via `resolve_obj_by_qualname` *inside the Worker's own separate OS
process* (see vllm_patch/worker.py's module docstring -- confirmed on real
hardware that EngineCore/Worker always runs out-of-process from the
driver, even under the default single-GPU UniProcExecutor). A class
defined only in the driver's `__main__` script would not be importable by
dotted path from that other process. Living next to vllm_patch/ in
`benchmarks/spec_prefill/` (already on sys.path for every script here, per
worker.py's own requirement) makes `diag_worker.DiagnosticSpecPrefillWorker`
resolve the same way `vllm_patch.worker.SpecPrefillWorker` already does.

Logging mechanism: writes one JSON line per (request, scheduler step) to
a file, whose path comes from the `SPEC_PREFILL_DIAG_LOG` env var --
NOT an in-memory list read back via collective_rpc, and NOT a plain
module-level dict a la pruning_registry.py. A file survives the same
process boundary pruning_registry.py's own docstring warns about, without
needing a second collective_rpc round trip just to retrieve results (the
driver reads the file directly once the run finishes) -- simpler for a
throwaway diagnostic than replicating pruning_registry's RPC-registration
machinery for read-back too. Caller (repro_chunked_prefill_bug.py) is
responsible for setting the env var before constructing the LLM (env vars
set in the driver process before spawning propagate to the Worker's
process) and for truncating any stale log from a previous run first.
"""

import json
import os
import time

from vllm_patch import pruning_registry
from vllm_patch.model_runner import SpecPrefillGPUModelRunner
from vllm_patch.worker import SpecPrefillWorker


class DiagnosticSpecPrefillGPUModelRunner(SpecPrefillGPUModelRunner):
    """Adds per-step, per-request logging on top of SpecPrefillGPUModelRunner's
    unmodified position-override behavior -- this class changes nothing about
    what the model actually sees or computes, only what gets observed."""

    _diag_step_counter = 0

    def _prepare_inputs(self, scheduler_output, *args, **kwargs):
        # Unmodified: runs the stock computation AND
        # SpecPrefillGPUModelRunner's own position-override logic exactly as
        # production does -- this class must not change model-visible
        # behavior, only add observation after the fact.
        result = super()._prepare_inputs(scheduler_output, *args, **kwargs)
        self._log_diagnostic_step(scheduler_output)
        return result

    def _log_diagnostic_step(self, scheduler_output) -> None:
        log_path = os.environ.get("SPEC_PREFILL_DIAG_LOG")
        if not log_path:
            return  # diagnostic logging not requested -- no-op, zero overhead

        self.__class__._diag_step_counter += 1
        step = self.__class__._diag_step_counter

        num_reqs = self.input_batch.num_reqs
        query_start_loc_np = self.query_start_loc.np[: num_reqs + 1]

        # Direct cross-check against the SCHEDULER's own raw decision, not
        # inferred through self.input_batch/query_start_loc (which are the
        # runner's own derived state, several layers downstream of
        # scheduler_output). scheduler_output.num_scheduled_tokens is set by
        # scheduler.py's `num_scheduled_tokens[request_id] = num_new_tokens`
        # (the exact token-budget-capped, possibly-partial value) -- reading
        # it here directly rules out any bug in this diagnostic's own
        # downstream inference chain being the reason no chunking was
        # observed across two real-hardware runs despite the scheduler
        # source (vllm/v1/core/sched/scheduler.py's
        # `num_new_tokens = min(num_new_tokens, token_budget)`, confirmed
        # unconditional when chunked prefill is enabled) suggesting it
        # should happen under this repro's forced budget contention.
        new_req_ids_this_step = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        sched_num_scheduled_tokens = dict(scheduler_output.num_scheduled_tokens)

        # discard_request_mask / num_computed_tokens_cpu are both populated by
        # the super()._prepare_inputs() call just above (gpu_model_runner.py
        # ~lines 1970-1997/1885) -- reading them here observes exactly what
        # the real forward pass and sampler will see this step, same timing
        # guarantee vllm_patch/model_runner.py's own override already relies
        # on (see that module's docstring).
        discard_mask_np = self.discard_request_mask.np[:num_reqs]

        rows = []
        for req_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[req_idx]
            num_computed_before = int(self.input_batch.num_computed_tokens_cpu[req_idx])
            start = int(query_start_loc_np[req_idx])
            end = int(query_start_loc_np[req_idx + 1])
            num_scheduled_this_step = end - start
            if num_scheduled_this_step <= 0:
                continue

            # Ground truth for "how long is this request's actual prompt":
            # PruneRecord.num_kept (this worker process's own registry, the
            # exact value model_runner.py's position-override already keys
            # off of) for pruned requests -- NOT self.requests[req_id]
            # .num_tokens, whose exact semantics near the prefill/decode
            # boundary (does it already include appended output tokens?)
            # aren't independently verified here. Falls back to
            # .num_tokens only for a request with no PruneRecord (shouldn't
            # occur in this repro -- every submitted request is pruned --
            # but kept generic rather than assuming).
            record = pruning_registry.get(req_id)
            full_target_len = (
                record.num_kept if record is not None else int(self.requests[req_id].num_tokens)
            )
            num_computed_after = num_computed_before + num_scheduled_this_step
            prefill_complete_after_this_step = num_computed_after >= full_target_len

            # Direct identity check for model_runner.py's untested multi-chunk
            # PruneRecord.positions_for_step branch (see this repro's own
            # module docstring, hypothesis #2): expected_positions is the
            # EXACT list of original positions this step's tokens should carry
            # if this is a prefill chunk (any chunk -- first or a later
            # continuation), or None if this step is decode (positions_for_step
            # already makes that distinction, see pruning_registry.py). Compare
            # against self.positions[start:end] -- the actual, real values
            # about to be fed into the model's RoPE this step, read via the
            # exact same "self.positions is a persistent, already-overridden
            # instance buffer" property model_runner.py's own docstring
            # documents and validate_runner_integration.py's Step B/B2 already
            # relied on for the single-chunk case. is_prefill_chunk_step=True
            # + positions_correct=False on ANY row is the smoking gun for
            # hypothesis #2; is_prefill_chunk_step=True +
            # positions_correct=True on EVERY row (including 2nd/3rd/...
            # continuation chunks, which Step B/B2 never exercised) rules it
            # out with the same rigor hypothesis #1 was ruled out with.
            is_prefill_chunk_step = False
            positions_correct = None
            if record is not None:
                expected_positions = record.positions_for_step(
                    num_computed_before, num_scheduled_this_step
                )
                if expected_positions is not None:
                    is_prefill_chunk_step = True
                    actual_positions = (
                        self.positions[start:end].detach().to("cpu").tolist()
                    )
                    positions_correct = actual_positions == expected_positions

            rows.append(
                {
                    "step": step,
                    "ts": time.time(),
                    "req_id": req_id,
                    "has_prune_record": record is not None,
                    "num_computed_before": num_computed_before,
                    "num_scheduled_this_step": num_scheduled_this_step,
                    "num_computed_after": num_computed_after,
                    "full_target_len": full_target_len,
                    "prefill_complete_after_this_step": prefill_complete_after_this_step,
                    # discard_request_mask is vLLM's OWN computation
                    # (gpu_model_runner.py ~lines 1989-1997), reused
                    # unmodified here, not re-derived -- so this diagnostic
                    # can never disagree with what the real sampler saw.
                    "discard_request_mask": bool(discard_mask_np[req_idx]),
                    "is_prefill_chunk_step": is_prefill_chunk_step,
                    "positions_correct": positions_correct,
                    # Direct scheduler cross-check (see comment above where
                    # these are read). sched_num_scheduled_tokens should
                    # exactly equal num_scheduled_this_step -- if it doesn't,
                    # the bug is in THIS diagnostic's query_start_loc-based
                    # derivation, not in vllm_patch/ or the scheduler.
                    # is_new_admission_this_step distinguishes "just entered
                    # self.running for the first time" from "continuing from
                    # an earlier step" -- multiple True rows in the SAME step
                    # across different requests directly proves multiple
                    # requests were considered for admission in that one
                    # step (informative even when none of them end up
                    # partially chunked).
                    "sched_num_scheduled_tokens": sched_num_scheduled_tokens.get(req_id),
                    "is_new_admission_this_step": req_id in new_req_ids_this_step,
                    "num_new_admissions_this_step": len(new_req_ids_this_step),
                }
            )

        if rows:
            with open(log_path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")


class DiagnosticSpecPrefillWorker(SpecPrefillWorker):
    """Same as vllm_patch.worker.SpecPrefillWorker, but swaps in the logging
    runner subclass above instead of the production SpecPrefillGPUModelRunner.
    register_prune_record/discard_prune_record are inherited unchanged --
    this only touches which runner class gets constructed."""

    def init_device(self) -> None:
        # SpecPrefillWorker.init_device() already does
        # super().init_device() (stock runner) then replaces
        # self.model_runner with the production SpecPrefillGPUModelRunner --
        # replacing it again here with the diagnostic subclass is one more
        # throwaway allocation on top of that same already-accepted
        # process-startup-only pattern (see vllm_patch/worker.py's
        # docstring), simpler and less error-prone than reimplementing
        # init_device()'s ~90 lines of CUDA/distributed setup a third way.
        super().init_device()
        self.model_runner = DiagnosticSpecPrefillGPUModelRunner(self.vllm_config, self.device)
