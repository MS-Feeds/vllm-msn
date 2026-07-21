"""SpecPrefillGPUModelRunner — Algorithm 1 lines 18-19 consumption side
("restore_pos_ids"/"merge_requests").

Subclasses vLLM's real `GPUModelRunner` (`vllm/v1/worker/gpu_model_runner.py`
-- confirmed the active runner for Gemma4/MoE models, see EXPERIMENT_PLAN.md
and the approved plan) rather than editing it. Overrides only
`_prepare_inputs()`, and even that override is a thin wrapper, not a
reimplementation -- see the approved plan's "Position override mechanism"
section for the full reasoning (verified directly against
`gpu_model_runner.py`'s real source, not assumed) for why this is safe:

- Token-buffer gather (lines 1899-1905 in the base class) and KV-cache slot
  mapping (2081-2085) are unaffected -- because pruning happens *before* the
  `Request` is constructed (see `pruner.py`), the pruned prompt's `N` tokens
  already occupy contiguous columns `0..N-1`, exactly like any normal
  `N`-token prompt. Nothing here needs to change.
- Only the RoPE angle (embedded into `CommonAttentionMetadata.positions` at
  line 2280, a plain *view* over the persistent `self.positions` buffer,
  not a copy) needs to diverge, to reflect each pruned token's *original*
  position instead of its contiguous storage index. Since `self.positions`
  is mutated in place by the base class (not reallocated), we can call the
  stock implementation unmodified via `super()`, then patch just that view
  afterward for pruned requests only.

**Residual risk, not yet verified on real hardware** (see the approved
plan's risk #1): this assumes nothing in `_prepare_inputs()` after line 2280
touches `self.positions` again, and that the attention backend's metadata
builder doesn't `.clone()`/`.contiguous()` the positions field before the
model actually reads it (which would break the view-aliasing this depends
on). If either assumption is wrong, positions patched here won't reach the
model, and this needs to move earlier (which would require a much larger
override, see the plan).
"""

from typing import TYPE_CHECKING

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from . import pruning_registry

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class SpecPrefillGPUModelRunner(GPUModelRunner):
    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        result = super()._prepare_inputs(scheduler_output)
        self._apply_spec_prefill_position_overrides(scheduler_output)
        return result

    def _apply_spec_prefill_position_overrides(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Patch `self.positions` (in place, via the persistent instance
        buffer -- see module docstring) for any request in this step's batch
        that has a `PruneRecord` in `pruning_registry`.

        Two cases, distinguished by whether this is the request's first
        step (prefill) or a later one (decode):

        - **Prefill** (`num_computed_tokens_cpu[req_idx] == 0`): per scope
          (single-step, non-chunked pruned prefill -- see
          EXPERIMENT_PLAN.md), the *entire* pruned prompt must be scheduled
          in this one step. Replace this request's slice of `self.positions`
          with `PruneRecord.kept_positions` directly (T, the original
          scattered positions of the surviving tokens).
        - **Decode** (`num_computed_tokens_cpu[req_idx] > 0`): the stock
          formula already gives correct *relative* spacing (each step
          advances by the right amount), it's just anchored to the pruned
          length `N` instead of the original length `M`. Add the constant
          `PruneRecord.decode_offset` (`M - N`) to continue the *original*
          prompt's numbering.
        """
        num_reqs = self.input_batch.num_reqs
        # Populated by the stock super() call just above, still valid here --
        # see module docstring for why these persistent instance buffers
        # remain readable after _prepare_inputs returns.
        query_start_loc_np = self.query_start_loc.np[: num_reqs + 1]

        for req_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[req_idx]
            record = pruning_registry.get(req_id)
            if record is None:
                continue  # not a pruned request -- stock positions stand.

            start = int(query_start_loc_np[req_idx])
            end = int(query_start_loc_np[req_idx + 1])
            num_scheduled_this_step = end - start
            if num_scheduled_this_step <= 0:
                continue

            num_computed_before = int(self.input_batch.num_computed_tokens_cpu[req_idx])

            if num_computed_before == 0:
                if num_scheduled_this_step != record.num_kept:
                    raise NotImplementedError(
                        f"Pruned request {req_id!r} prefill was scheduled "
                        f"across multiple steps ({num_scheduled_this_step} "
                        f"of {record.num_kept} pruned tokens this step) -- "
                        f"chunked pruned prefill is out of scope for this "
                        f"pass, see EXPERIMENT_PLAN.md's Implementation "
                        f"status and the approved plan's 'Scope for this "
                        f"pass'."
                    )
                kept_positions_gpu = torch.tensor(
                    record.kept_positions, dtype=self.positions.dtype, device=self.device
                )
                self.positions[start:end] = kept_positions_gpu
            else:
                self.positions[start:end] += record.decode_offset

        pruning_registry.discard_finished(scheduler_output.finished_req_ids)
