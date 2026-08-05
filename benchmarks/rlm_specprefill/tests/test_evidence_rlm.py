"""Tests for rlm_stage/evidence_rlm.py's backend_kwargs construction --
specifically the vllm-backend fast-timeout/no-retry defaults added after a
confirmed real-hardware bug (every sample hitting the exact 900s hard
timeout against a self-hosted Qwen3-Coder-480B root, consistent with a
silent client-level retry-and-rewait chain rather than legitimate search).

Monkeypatches evidence_rlm.RLM with a stub that captures its constructor
kwargs -- no network, no real rlm package internals exercised, same
monkeypatch-a-module-level-name pattern used throughout tests/test_runner.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlm_stage import evidence_rlm as evidence_rlm_module  # noqa: E402


class _StubCompletion:
    response = '{"excerpts": [], "question": null}'


class _StubRLM:
    captured_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _StubRLM.captured_kwargs = kwargs

    def completion(self, prompt):
        return _StubCompletion()


def test_vllm_backend_defaults_to_fast_timeout_and_no_retries(monkeypatch):
    monkeypatch.setattr(evidence_rlm_module, "RLM", _StubRLM)

    evidence_rlm_module.run_evidence_extraction(
        "s1",
        "find it",
        "ctx",
        guardrails={},
        root_backend="vllm",
        root_base_url="http://127.0.0.1:8000/v1",
        log_dir=None,
    )

    backend_kwargs = _StubRLM.captured_kwargs["backend_kwargs"]
    assert backend_kwargs["timeout"] == evidence_rlm_module._DEFAULT_VLLM_CLIENT_TIMEOUT_S
    assert backend_kwargs["max_retries"] == evidence_rlm_module._DEFAULT_VLLM_CLIENT_MAX_RETRIES


def test_anthropic_backend_unaffected_by_vllm_defaults(monkeypatch):
    monkeypatch.setattr(evidence_rlm_module, "RLM", _StubRLM)

    evidence_rlm_module.run_evidence_extraction(
        "s1",
        "find it",
        "ctx",
        guardrails={},
        root_backend="anthropic",
        api_key="fake-key",
        log_dir=None,
    )

    backend_kwargs = _StubRLM.captured_kwargs["backend_kwargs"]
    assert "timeout" not in backend_kwargs
    assert "max_retries" not in backend_kwargs


def test_explicit_vllm_client_overrides_respected(monkeypatch):
    monkeypatch.setattr(evidence_rlm_module, "RLM", _StubRLM)

    evidence_rlm_module.run_evidence_extraction(
        "s1",
        "find it",
        "ctx",
        guardrails={},
        root_backend="vllm",
        root_base_url="http://127.0.0.1:8000/v1",
        root_client_timeout_s=120.0,
        root_client_max_retries=2,
        log_dir=None,
    )

    backend_kwargs = _StubRLM.captured_kwargs["backend_kwargs"]
    assert backend_kwargs["timeout"] == 120.0
    assert backend_kwargs["max_retries"] == 2


def test_explicit_anthropic_client_overrides_respected(monkeypatch):
    monkeypatch.setattr(evidence_rlm_module, "RLM", _StubRLM)

    evidence_rlm_module.run_evidence_extraction(
        "s1",
        "find it",
        "ctx",
        guardrails={},
        root_backend="anthropic",
        api_key="fake-key",
        root_client_timeout_s=45.0,
        root_client_max_retries=1,
        log_dir=None,
    )

    backend_kwargs = _StubRLM.captured_kwargs["backend_kwargs"]
    assert backend_kwargs["timeout"] == 45.0
    assert backend_kwargs["max_retries"] == 1


def test_root_max_tokens_per_turn_defaults_to_a_real_cap(monkeypatch):
    """Confirmed real-hardware bug (2026-08-05): with no max_tokens cap on
    the root's own completion calls, a non-terminating generation can run
    for as long as the server allows (nvidia-smi showed 100% GPU util for
    the full 900s hard-timeout duration on a real stuck sample) -- the
    default here must actually be a real, finite number, not None/uncapped."""
    monkeypatch.setattr(evidence_rlm_module, "RLM", _StubRLM)

    evidence_rlm_module.run_evidence_extraction(
        "s1", "find it", "ctx", guardrails={}, root_backend="anthropic", api_key="fake-key", log_dir=None
    )

    sampling_args = _StubRLM.captured_kwargs["sampling_args"]
    assert sampling_args == {"max_tokens": evidence_rlm_module._DEFAULT_ROOT_MAX_TOKENS_PER_TURN}
    assert evidence_rlm_module._DEFAULT_ROOT_MAX_TOKENS_PER_TURN > 0


def test_root_max_tokens_per_turn_explicit_override_respected(monkeypatch):
    monkeypatch.setattr(evidence_rlm_module, "RLM", _StubRLM)

    evidence_rlm_module.run_evidence_extraction(
        "s1",
        "find it",
        "ctx",
        guardrails={},
        root_backend="anthropic",
        api_key="fake-key",
        root_max_tokens_per_turn=1234,
        log_dir=None,
    )

    assert _StubRLM.captured_kwargs["sampling_args"] == {"max_tokens": 1234}


def test_root_max_tokens_per_turn_none_opts_out_uncapped(monkeypatch):
    monkeypatch.setattr(evidence_rlm_module, "RLM", _StubRLM)

    evidence_rlm_module.run_evidence_extraction(
        "s1",
        "find it",
        "ctx",
        guardrails={},
        root_backend="anthropic",
        api_key="fake-key",
        root_max_tokens_per_turn=None,
        log_dir=None,
    )

    assert _StubRLM.captured_kwargs["sampling_args"] is None
