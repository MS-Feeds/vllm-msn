"""SpecPrefillWorker — swaps in `SpecPrefillGPUModelRunner` without editing
vLLM's own `vllm/v1/worker/gpu_worker.py`.

Verified mechanism (see EXPERIMENT_PLAN.md's Implementation status and the
approved plan): `parallel_config.worker_cls` is a dotted-path string,
resolved via `resolve_obj_by_qualname` in
`vllm/v1/worker/worker_base.py:250-253`, and `Worker.init_device()`
(`vllm/v1/worker/gpu_worker.py:291-330`) hardcodes construction of the stock
runner as its last act, with nothing afterward depending on which runner
class was built. So this subclass just runs the stock `init_device()` (all
~90 lines of CUDA/distributed/memory-snapshot setup, including its own
throwaway stock `GPUModelRunner`) via `super()`, then replaces
`self.model_runner` with our subclass. One-time, process-startup-only
double allocation; zero edits to vLLM's files.

Usage: pass this class's dotted path as `worker_cls` when constructing the
engine, e.g.:

    from vllm import LLM
    llm = LLM(
        model=...,
        worker_cls="vllm_patch.worker.SpecPrefillWorker",
        ...
    )

`resolve_obj_by_qualname` imports the module by name, so the driver process
needs `benchmarks/spec_prefill` on `sys.path` (or run from within that
directory) for `vllm_patch.worker` to resolve -- same requirement as every
other `vllm_patch` import in this package.

**Confirmed on real hardware (2026-07-22)**: `EngineCore` (which owns this
Worker) runs in a genuinely separate OS process from the driver script that
calls `LLM(...)`/`add_request(...)`, even under the default single-GPU
`UniProcExecutor` -- visible via the `(EngineCore pid=...)` log prefix, and
confirmed the hard way: `LLMEngine.collective_rpc` serializes whatever it
sends across that boundary (`vllm/v1/serial_utils.py`), and a raw Python
function object isn't serializable by default (`TypeError: Object of type
<class 'function'> is not serializable`, a deliberate security restriction).
This means `pruning_registry`'s in-memory dict, if only ever written to from
the driver process (e.g. a plain `pruning_registry.register(...)` call
inside `pruner.py`), would be invisible here -- `model_runner.py` reads from
*this* process's copy of the module, a completely separate Python runtime
from the driver's. `register_prune_record`/`discard_prune_record` below are
the actual fix: RPC-callable *by name* (the `str` form of `collective_rpc`,
which only needs to serialize primitive args, not a function), so
`pruner.py` can push a `PruneRecord` into the process that actually needs
it.
"""

from vllm.v1.worker.gpu_worker import Worker

from . import pruning_registry
from .model_runner import SpecPrefillGPUModelRunner


class SpecPrefillWorker(Worker):
    def init_device(self) -> None:
        super().init_device()
        self.model_runner = SpecPrefillGPUModelRunner(self.vllm_config, self.device)

    def register_prune_record(
        self, request_id: str, kept_positions: list, orig_len: int
    ) -> None:
        """RPC-callable via `llm_engine.collective_rpc("register_prune_record",
        args=(request_id, kept_positions, orig_len))` -- see `pruner.py`'s
        `prune_and_add_request`. Must run inside this process (not called
        directly from the driver) for `model_runner.py` to see it -- see
        module docstring."""
        pruning_registry.register(request_id, kept_positions, orig_len)

    def discard_prune_record(self, request_id: str) -> None:
        """RPC-callable, same reasoning as `register_prune_record` above."""
        pruning_registry.discard(request_id)
