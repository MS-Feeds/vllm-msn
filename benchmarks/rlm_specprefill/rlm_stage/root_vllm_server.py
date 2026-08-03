"""Lifecycle management for a self-hosted `vllm serve` process, used when
RLM's root model (rlm_stage/evidence_rlm.py) is self-hosted instead of a
hosted Claude API -- see that module's own docstring for why this became an
option (per IMPLEMENTATION_PLAN.md's original note that root hosting "may
change later if the team decides root also needs to be self-hosted").

Root-model calls happen interactively throughout RLM's REPL loop (arbitrary
recursive sub-calls, not one precomputed batch), so they need an actual
HTTP-servable endpoint -- unlike target_stage/vllm_offline_engine.py's
target-answering step, which uses vLLM's OFFLINE `LLM` class specifically
because it needs `worker_cls=SpecPrefillWorker` (unvalidated under `vllm
serve`'s server mode, per that module's own docstring / IMPLEMENTATION_PLAN.md
decision 2). So when root and target are the SAME checkpoint (e.g.
Qwen3-Coder-480B-A35B serving as both), the two phases run as separate,
non-overlapping serving instances: this module's `vllm serve` subprocess for
the RLM evidence stage, torn down before target_stage builds its own offline
engine over the same GPUs -- runner/run_arm.py enforces that ordering (both
because a `vllm serve` subprocess and an in-process `LLM(...)` would compete
for the same GPU memory, and per IMPLEMENTATION_PLAN.md decision 3's "at
most one vLLM engine ever live" invariant, now extended to cover this
separate serving process too).

**Not yet run on real hardware** -- same caveat as everything else this
project has built without GPU access. The health-check below only confirms
the HTTP listener answers `/v1/models`; it does not confirm the model has
finished warming up enough to serve a real completion without added latency
on the very first request. If the very first root-model call in a run comes
back oddly slow or errors, that's the first thing to suspect, not
necessarily a real bug.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


class RootServerStartupError(RuntimeError):
    """Raised when the vLLM server process exits, or never becomes healthy,
    before the configured startup timeout."""


def build_vllm_serve_command(
    model_path: str,
    *,
    port: int,
    tensor_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    served_model_name: str | None = None,
    gpu_memory_utilization: float = 0.9,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Builds the `vllm serve` argv. Mirrors the flags
    target_stage/vllm_offline_engine.py's build_plain_target_engine passes
    to the offline `LLM(...)` class where an equivalent `vllm serve` flag
    exists (--tensor-parallel-size, --enable-expert-parallel,
    --gpu-memory-utilization, --trust-remote-code, --enforce-eager) -- kept
    consistent so root-serving and target-answering behave predictably the
    same way for the same checkpoint, not because either was independently
    tuned for this serving mode.

    `served_model_name` is what the OpenAI-compatible API expects in a
    completion request's `model` field -- without `--served-model-name`,
    vLLM registers the model under its own resolved model_path/repo id,
    which is usually fine to also pass as `model_name` to
    evidence_rlm.py's RLM(backend="vllm", ...) call, but an explicit
    served name avoids any ambiguity when `model_path` is a long local
    snapshot directory rather than a clean HF repo id.
    """
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--trust-remote-code",
        "--enforce-eager",
    ]
    if enable_expert_parallel:
        cmd.append("--enable-expert-parallel")
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def start_root_vllm_server(
    model_path: str,
    *,
    port: int,
    tensor_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    served_model_name: str | None = None,
    gpu_memory_utilization: float = 0.9,
    extra_args: list[str] | None = None,
    log_path: Path | None = None,
) -> subprocess.Popen:
    """Launches `vllm serve` as a background subprocess and returns
    immediately -- does NOT wait for it to become healthy, see
    `wait_for_server_healthy` for that. Caller owns the returned process
    and must eventually call `stop_root_vllm_server` on it, including on
    the exception path (runner/run_arm.py wraps evidence collection in a
    try/finally for exactly this -- an orphaned server process holds GPU
    memory the target-answering stage needs next).

    `log_path`, if given, redirects the subprocess's combined stdout/stderr
    there (vLLM's own startup/request logs are voluminous and would
    otherwise interleave confusingly with this script's own prints);
    otherwise the subprocess inherits this process's stdout/stderr."""
    cmd = build_vllm_serve_command(
        model_path,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        served_model_name=served_model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        extra_args=extra_args,
    )
    print(f"[root_vllm_server] launching: {' '.join(cmd)}", flush=True)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        print(f"[root_vllm_server] logging server output to {log_path}", flush=True)
        return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def wait_for_server_healthy(
    base_url: str,
    *,
    process: subprocess.Popen | None = None,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 5.0,
) -> None:
    """Polls `{base_url}/models` (the OpenAI-compatible list-models
    endpoint, present as soon as vLLM's API server is up and has registered
    the model) until it returns 200, or raises `RootServerStartupError`.

    Loading a 480B checkpoint across several GPUs can genuinely take many
    minutes -- the default 1800s (30 min) timeout is a starting guess, not a
    profiled value; raise it if a real run needs longer. If `process` is
    given and exits before becoming healthy, this raises immediately
    (rather than waiting out the full timeout on a process that's already
    dead) with whatever exit code it saw.
    """
    import requests

    url = base_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RootServerStartupError(
                f"vLLM server process exited (code={process.returncode}) before "
                f"becoming healthy at {url} -- check its log output."
            )
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return
            last_error = RuntimeError(f"HTTP {resp.status_code} from {url}")
        except requests.exceptions.RequestException as e:
            last_error = e
        time.sleep(poll_interval_s)

    raise RootServerStartupError(
        f"vLLM server at {url} did not become healthy within {timeout_s}s "
        f"(last error: {last_error!r}). Loading a large checkpoint can "
        f"legitimately take longer than this default -- consider raising "
        f"--root-server-startup-timeout-s before assuming something's wrong."
    )


def stop_root_vllm_server(process: subprocess.Popen, *, term_timeout_s: float = 30.0) -> None:
    """SIGTERM, then SIGKILL if it hasn't exited within `term_timeout_s` --
    same escalation pattern as any well-behaved process-group teardown.
    Waits for the process to actually exit (not just for the signal to be
    sent) so the caller can rely on GPU memory being released before
    building the next engine -- vLLM's own shutdown may take a few seconds
    to free CUDA memory after the process exits; callers with a tight
    GPU-reuse window immediately after this call may still want a short
    explicit delay, not assumed safe here."""
    if process.poll() is not None:
        return  # already exited

    process.terminate()
    try:
        process.wait(timeout=term_timeout_s)
    except subprocess.TimeoutExpired:
        print(
            f"[root_vllm_server] process didn't exit within {term_timeout_s}s "
            f"of SIGTERM -- sending SIGKILL.",
            flush=True,
        )
        process.kill()
        process.wait()
