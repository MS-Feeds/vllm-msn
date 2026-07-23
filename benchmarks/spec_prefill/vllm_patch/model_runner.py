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
    def _prepare_inputs(self, scheduler_output: "SchedulerOutput", *args, **kwargs):
        # *args/**kwargs (rather than a hardcoded signature) deliberately:
        # confirmed on real hardware that the base class's real signature is
        # `_prepare_inputs(self, scheduler_output, num_scheduled_tokens:
        # np.ndarray)` -- not `(self, scheduler_output)` alone as first
        # assumed (TypeError: takes 2 positional arguments but 3 were given).
        # Passing through opaquely means a future signature change upstream
        # (e.g. another parameter added) doesn't silently break this thin
        # wrapper the same way. Return value is `(logits_indices,
        # spec_decode_metadata)`, per the base class's own docstring --
        # returned unchanged here, only `self.positions` is touched.
        result = super()._prepare_inputs(scheduler_output, *args, **kwargs)
        self._apply_spec_prefill_position_overrides(scheduler_output)
        return result

    def _apply_spec_prefill_position_overrides(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Patch `self.positions` (in place, via the persistent instance
        buffer -- see module docstring) for any request in this step's batch
        that has a `PruneRecord` in `pruning_registry`.

        Delegates the actual case split to `PruneRecord.positions_for_step`
        (pruning_registry.py), which distinguishes a prefill chunk (first or
        a later continuation -- chunked prefill may split a long pruned
        prompt's kept tokens across multiple steps once they exceed the
        scheduler's per-step token budget) from a decode step:

        - **Prefill chunk** (`positions_for_step` returns a list): replace
          this request's slice of `self.positions` with the returned
          original, scattered positions for exactly the tokens scheduled
          this step -- correct whether this is the first chunk or a later
          continuation, since `positions_for_step` slices from
          `num_computed_before`.
        - **Decode** (`positions_for_step` returns `None`): the stock
          formula already gives correct *relative* spacing (each step
          advances by the right amount), it's just anchored to the pruned
          length `N` instead of the original length `M`. Add the constant
          `PruneRecord.decode_offset` (`M - N`) to continue the *original*
          prompt's numbering.

        Preemption was checked and needs no special-casing here either --
        see `positions_for_step`'s own docstring.
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

            kept_slice = record.positions_for_step(num_computed_before, num_scheduled_this_step)
            if kept_slice is not None:
                kept_positions_gpu = torch.tensor(
                    kept_slice, dtype=self.positions.dtype, device=self.device
                )
                self.positions[start:end] = kept_positions_gpu
            else:
                self.positions[start:end] += record.decode_offset

        pruning_registry.discard_finished(scheduler_output.finished_req_ids)
