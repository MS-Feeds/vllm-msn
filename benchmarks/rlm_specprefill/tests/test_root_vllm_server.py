"""Tests for rlm_stage/root_vllm_server.py's pure command-building logic --
no real subprocess, no network, no GPU. Process lifecycle (start/health-
check/stop) needs a real vllm serve binary and is exercised on the GPU
node instead (see REPRODUCE.md), same split every GPU-dependent piece in
this project uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlm_stage.root_vllm_server import build_vllm_serve_command  # noqa: E402


def test_build_vllm_serve_command_basic():
    cmd = build_vllm_serve_command("/path/to/model", port=8000, tensor_parallel_size=4)
    assert cmd[:3] == ["vllm", "serve", "/path/to/model"]
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "8000"
    assert "--tensor-parallel-size" in cmd and cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
    assert "--trust-remote-code" in cmd
    assert "--enforce-eager" in cmd
    # Not requested, must not appear:
    assert "--enable-expert-parallel" not in cmd
    assert "--served-model-name" not in cmd


def test_build_vllm_serve_command_enable_expert_parallel():
    cmd = build_vllm_serve_command(
        "/path/to/model", port=8000, tensor_parallel_size=8, enable_expert_parallel=True
    )
    assert "--enable-expert-parallel" in cmd


def test_build_vllm_serve_command_served_model_name():
    cmd = build_vllm_serve_command(
        "/path/to/model", port=8000, tensor_parallel_size=1, served_model_name="my-model"
    )
    assert "--served-model-name" in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "my-model"


def test_build_vllm_serve_command_extra_args_passthrough():
    cmd = build_vllm_serve_command(
        "/path/to/model", port=8000, tensor_parallel_size=1, extra_args=["--max-model-len", "131072"]
    )
    assert cmd[-2:] == ["--max-model-len", "131072"]


def test_build_vllm_serve_command_gpu_memory_utilization_default():
    cmd = build_vllm_serve_command("/path/to/model", port=8000, tensor_parallel_size=1)
    assert "--gpu-memory-utilization" in cmd
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.9"
