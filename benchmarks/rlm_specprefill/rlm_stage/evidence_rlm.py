#!/usr/bin/env python3
"""Runs RLM as the retrieval front-end described in
../prompts/evidence_extraction.py, over one (context, question) sample.

Root model is selectable via `root_backend`: `"anthropic"` (hosted Claude,
the original default -- see IMPLEMENTATION_PLAN.md's decision record) or
`"vllm"` (self-hosted, e.g. Qwen3-Coder-480B-A35B serving as both root and
SpecPrefill target -- see ../rlm_stage/root_vllm_server.py for the serving
lifecycle this requires, and ../runner/run_arm.py for how the two are
sequenced around each other). The vendored `rlm` package already supports
this natively (`rlm/clients/__init__.py`'s `backend="vllm"` case, which
just points its OpenAI-compatible client at a custom `base_url`) -- no
changes needed there, only this module gained the selection.

This module deliberately does ONE query per call and returns the raw
`RLMChatCompletion` alongside the parsed evidence -- caching/replay across
arms A/B/C is evidence_cache.py's job (not yet built), not this module's.
Keeping this module single-purpose (construct RLM, run it once, parse the
result) makes it independently testable and matches
../../rlm/examples/quickstart_anthropic.py's own scope.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from rlm import RLM
from rlm.core.types import RLMChatCompletion
from rlm.logger import RLMLogger

sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts.evidence_extraction import (  # noqa: E402
    EVIDENCE_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_evidence_prompt,
    parse_evidence_response,
)

THIS_DIR = Path(__file__).resolve().parent.parent  # benchmarks/rlm_specprefill/
DEFAULT_GUARDRAILS_PATH = THIS_DIR / "configs" / "guardrails.yaml"
# Honors $RLM_SPECPREFILL_LOG_DIR (see runner/run_arm.py's
# DEFAULT_RESULTS_DIR for the matching fix and fuller writeup -- this
# module's own default was equally vestigial-until-fixed).
DEFAULT_LOG_DIR = Path(os.environ.get("RLM_SPECPREFILL_LOG_DIR", str(THIS_DIR / "logs")))

# Loaded from ../../rlm/.env (that project's own dotenv file, which already
# has a real ANTHROPIC_API_KEY -- see ../../rlm/examples/quickstart_anthropic.py
# and IMPLEMENTATION_PLAN.md's root-hosting decision). Not duplicated here;
# .env_exports.sh sources the same file for shell-level tooling.
RLM_DOTENV_PATH = THIS_DIR.parent / "rlm" / ".env"

DEFAULT_MODEL_NAME = "claude-haiku-4-5-20251001"  # matches quickstart_anthropic.py

# Confirmed real-hardware bug (2026-08-05): rlm's BaseLM.DEFAULT_TIMEOUT is
# 300s (../../rlm/rlm/clients/base_lm.py), and neither OpenAIClient nor
# AnthropicClient overrides the underlying SDK's own default max_retries
# (2, both openai-python and anthropic-python) -- worst case for a single
# stuck LM call: (1 + 2 retries) x 300s = 900s, which matched EXACTLY the
# hard-timeout ceiling observed against a self-hosted Qwen3-Coder-480B root
# over 150K-500K-token S-NIAH contexts (every sample was reaching precisely
# that ceiling, not a spread of times -- the signature of a silent
# client-level retry-and-rewait chain, not legitimate multi-turn search
# that would vary sample to sample). Hosted Claude essentially never
# approaches a 300s single-request timeout, so this retry path is
# effectively invisible for root_backend='anthropic' -- these defaults are
# only applied for 'vllm' below, where a queued/overloaded self-hosted
# server is a real, observed risk, not a hypothetical one.
#
# NOTE this does NOT make a stuck call internally retried/recovered by RLM
# itself -- confirmed by reading rlm/core/rlm.py's completion() loop
# directly: the only exception handler around the whole iteration loop is
# `except KeyboardInterrupt`, and guardrails.yaml's own `max_errors`
# threshold (`_check_iteration_limits`) only counts REPL/code-execution
# stderr errors, never LM-completion-call exceptions -- an exception from
# `lm_handler.completion(...)` propagates straight out of `RLM.completion()`
# uncaught. The real win here is turnaround speed and diagnosability: a
# fast, loud client timeout means run_arm.py's own exception handler SKIPs
# the sample with a clear error instead of silently consuming the sample's
# entire wall-clock budget on one hung HTTP call that no guardrail can
# preempt (RLM's max_timeout is only checked between root-loop iterations,
# never inside one -- see run_arm.py's _HARD_TIMEOUT_BUFFER_S comment).
#
# 60s -> 300s (2026-08-05): the 60s value was too aggressive in practice --
# even tiny 500-1000-token S-NIAH contexts were timing out at a ~50% rate
# against the self-hosted Qwen3-Coder-480B root, consistent with genuine
# request queuing/contention on that single vllm serve instance (e.g. from
# guardrails.yaml's max_concurrent_subcalls=4 firing several parallel
# sub-calls at one server) rather than a stuck exec()/infinite loop (which
# wouldn't produce a clean APITimeoutError). 300s matches BaseLM's own
# DEFAULT_TIMEOUT -- the fix that matters is keeping max_retries=0 (below),
# which is what actually removes the 900s (3x300s) worst case; the timeout
# value itself doesn't need to be shorter than the client's normal default,
# just not silently retried on top of it.
_DEFAULT_VLLM_CLIENT_TIMEOUT_S = 300.0
_DEFAULT_VLLM_CLIENT_MAX_RETRIES = 0


def load_guardrails(path: Path = DEFAULT_GUARDRAILS_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    # max_budget: null in YAML -> None in Python, which is RLM's own "no cap"
    # sentinel for that param, so no special-casing needed here.
    return config


@dataclass
class EvidenceResult:
    """What evidence_cache.py persists per query -- bundles the parsed
    evidence with the full RLMChatCompletion (needed by
    rlm_stage/timing_decomposition.py for per-query latency reporting even
    on a cache hit, per IMPLEMENTATION_PLAN.md decision 4)."""

    sample_id: str
    completion: RLMChatCompletion
    evidence: dict[str, Any]
    wall_time_s: float
    prompt_version: str = PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        # completion.to_dict()/from_dict() (rlm/core/types.py) handle the
        # nested UsageSummary/metadata structure -- a plain dataclasses.asdict
        # wouldn't round-trip those correctly, so this delegates rather than
        # reimplementing that serialization.
        return {
            "sample_id": self.sample_id,
            "completion": self.completion.to_dict(),
            "evidence": self.evidence,
            "wall_time_s": self.wall_time_s,
            "prompt_version": self.prompt_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceResult":
        return cls(
            sample_id=data["sample_id"],
            completion=RLMChatCompletion.from_dict(data["completion"]),
            evidence=data["evidence"],
            wall_time_s=data["wall_time_s"],
            prompt_version=data.get("prompt_version", PROMPT_VERSION),
        )


def run_evidence_extraction(
    sample_id: str,
    question: str,
    context: str,
    *,
    guardrails: dict[str, Any] | None = None,
    root_backend: str = "anthropic",
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
    root_base_url: str | None = None,
    root_client_timeout_s: float | None = None,
    root_client_max_retries: int | None = None,
    log_dir: Path | None = DEFAULT_LOG_DIR,
    verbose: bool = False,
) -> EvidenceResult:
    """Runs one RLM evidence-extraction query and returns the parsed result.

    `root_backend="anthropic"` (default): loads ../../rlm/.env for
    ANTHROPIC_API_KEY if `api_key` isn't passed explicitly (load_dotenv()
    with an explicit path -- NOT relying on python-dotenv's cwd-relative
    auto-discovery, since this module may be invoked from a different
    working directory than benchmarks/rlm/).

    `root_backend="vllm"`: routes to the same `rlm.clients.OpenAIClient`
    the package uses for hosted OpenAI, just pointed at `root_base_url`
    (a self-hosted `vllm serve` endpoint -- see root_vllm_server.py; the
    caller, typically runner/run_arm.py, is responsible for that server
    already being up and healthy before this is called). `model_name` here
    must match whatever the server was launched with (its
    `--served-model-name`, or the model path/repo id if that wasn't set).
    `api_key` is ignored for this backend -- vLLM's OpenAI-compatible
    server doesn't validate keys, but the underlying `openai` SDK still
    requires a non-empty string, so a placeholder is always sent.

    `root_client_timeout_s`/`root_client_max_retries` override the
    underlying client's per-request HTTP timeout/retry count. For
    `root_backend='vllm'`, these default to `_DEFAULT_VLLM_CLIENT_TIMEOUT_S`/
    `_DEFAULT_VLLM_CLIENT_MAX_RETRIES` (see that constant's own comment for
    why) unless explicitly overridden; for `root_backend='anthropic'`
    they're left at the client's own default (300s/2 retries) unless
    explicitly passed, since that path has never shown this failure mode.
    """
    if root_backend == "anthropic":
        if api_key is None:
            load_dotenv(dotenv_path=RLM_DOTENV_PATH)
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    f"ANTHROPIC_API_KEY not set and not found in {RLM_DOTENV_PATH}. "
                    "Set it in that file or pass api_key= explicitly."
                )
        backend_kwargs = {"model_name": model_name, "api_key": api_key}
    elif root_backend == "vllm":
        if not root_base_url:
            raise ValueError(
                "root_base_url is required when root_backend='vllm' (e.g. "
                "'http://127.0.0.1:8000/v1' -- use the loopback IP, not "
                "'localhost': confirmed real-hardware bug where this "
                "client (openai SDK/httpx) and the health check (plain "
                "requests) resolved 'localhost' to different addresses, "
                "so the health check passed while every real request "
                "failed with APIConnectionError -- match the port a real "
                "vllm serve process is answering on)."
            )
        backend_kwargs = {"model_name": model_name, "base_url": root_base_url, "api_key": "EMPTY"}
    else:
        raise ValueError(f"Unknown root_backend {root_backend!r}, expected 'anthropic' or 'vllm'.")

    # See _DEFAULT_VLLM_CLIENT_TIMEOUT_S's own comment for why 'vllm' gets a
    # fast-fail default and 'anthropic' doesn't -- both remain overridable
    # via the explicit kwargs regardless of backend.
    resolved_timeout_s = root_client_timeout_s
    resolved_max_retries = root_client_max_retries
    if root_backend == "vllm":
        if resolved_timeout_s is None:
            resolved_timeout_s = _DEFAULT_VLLM_CLIENT_TIMEOUT_S
        if resolved_max_retries is None:
            resolved_max_retries = _DEFAULT_VLLM_CLIENT_MAX_RETRIES
    if resolved_timeout_s is not None:
        backend_kwargs["timeout"] = resolved_timeout_s
    if resolved_max_retries is not None:
        backend_kwargs["max_retries"] = resolved_max_retries

    guardrails = dict(guardrails or load_guardrails())

    logger = RLMLogger(log_dir=str(log_dir)) if log_dir else None

    rlm_instance = RLM(
        backend=root_backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        custom_system_prompt=EVIDENCE_SYSTEM_PROMPT,
        logger=logger,
        verbose=verbose,
        **guardrails,
    )

    full_prompt = build_evidence_prompt(question, context)

    start = time.perf_counter()
    completion = rlm_instance.completion(full_prompt)
    wall_time_s = time.perf_counter() - start

    evidence = parse_evidence_response(completion.response)

    return EvidenceResult(
        sample_id=sample_id,
        completion=completion,
        evidence=evidence,
        wall_time_s=wall_time_s,
    )


def _smoke_test() -> None:
    """Small, cheap end-to-end check against the REAL Anthropic API --
    the highest-value validation achievable without a GPU (exercises the
    actual retrieval-framing prompt design, not just scaffolding). See
    REPRODUCE.md step 4.
    """
    import random
    import string

    rng = random.Random(1234)
    secret = str(rng.randint(100_000_000, 999_999_999))
    filler = [
        "".join(rng.choices(string.ascii_lowercase + " ", k=80)) for _ in range(300)
    ]
    insert_at = 150
    filler.insert(insert_at, f"NIAH_SECRET_1={secret}")
    context = "\n".join(filler)
    question = (
        "The context contains 1 hidden line matching the pattern "
        "NIAH_SECRET_1=<digits>. Find it."
    )

    print(f"[smoke_test] running evidence extraction, expecting secret={secret} ...")
    # verbose=False: RLM's rich-based verbose printer (rlm/logger/verbose.py)
    # writes a box-drawing character that crashes on Windows consoles using
    # a non-UTF8 codepage (UnicodeEncodeError under cp1252) -- an upstream
    # rlm issue, not this harness's. Not needed for the smoke test anyway;
    # this function prints its own summary below.
    result = run_evidence_extraction(
        sample_id="smoke_test",
        question=question,
        context=context,
        verbose=False,
    )

    print(f"[smoke_test] wall_time_s={result.wall_time_s:.2f}")
    print(f"[smoke_test] parse_error={result.evidence['parse_error']}")
    print(f"[smoke_test] excerpts={result.evidence['excerpts']}")

    found = any(secret in e.get("text", "") for e in result.evidence["excerpts"])
    print(f"[smoke_test] secret found in evidence: {found}")
    if not found:
        print(
            "[smoke_test] WARNING: secret not found in extracted evidence -- "
            "either the prompt design needs work, or this was just a bad "
            "sample (single smoke-test run, not a real eval)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RLM evidence-extraction stage")
    parser.add_argument("--smoke-test", action="store_true", help="Run a small live-API smoke test.")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
