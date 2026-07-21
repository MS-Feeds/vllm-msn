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
"""

from vllm.v1.worker.gpu_worker import Worker

from .model_runner import SpecPrefillGPUModelRunner


class SpecPrefillWorker(Worker):
    def init_device(self) -> None:
        super().init_device()
        self.model_runner = SpecPrefillGPUModelRunner(self.vllm_config, self.device)
