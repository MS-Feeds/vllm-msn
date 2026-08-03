#!/usr/bin/env python3
"""Runs RLM (root = hosted Claude, per IMPLEMENTATION_PLAN.md decision on
root-model hosting) as the retrieval front-end described in
../prompts/evidence_extraction.py, over one (context, question) sample.

This module deliberately does ONE query per call and returns the raw
`RLMChatCompletion` alongside the parsed evidence -- caching/replay across
arms A/B/C is evidence_cache.py's job (not yet built), not this module's.
Keeping this module single-purpose (construct RLM, run it once, parse the
result) makes it independently testable and matches
../../rlm/examples/quickstart_anthropic.py's own scope.
"""

from __future__ import annotations

import argparse
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
DEFAULT_LOG_DIR = THIS_DIR / "logs"

# Loaded from ../../rlm/.env (that project's own dotenv file, which already
# has a real ANTHROPIC_API_KEY -- see ../../rlm/examples/quickstart_anthropic.py
# and IMPLEMENTATION_PLAN.md's root-hosting decision). Not duplicated here;
# .env_exports.sh sources the same file for shell-level tooling.
RLM_DOTENV_PATH = THIS_DIR.parent / "rlm" / ".env"

DEFAULT_MODEL_NAME = "claude-haiku-4-5-20251001"  # matches quickstart_anthropic.py


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
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
    log_dir: Path | None = DEFAULT_LOG_DIR,
    verbose: bool = False,
) -> EvidenceResult:
    """Runs one RLM evidence-extraction query and returns the parsed result.

    Loads ../../rlm/.env for ANTHROPIC_API_KEY if `api_key` isn't passed
    explicitly (load_dotenv() with an explicit path -- NOT relying on
    python-dotenv's cwd-relative auto-discovery, since this module may be
    invoked from a different working directory than benchmarks/rlm/).
    """
    if api_key is None:
        load_dotenv(dotenv_path=RLM_DOTENV_PATH)
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                f"ANTHROPIC_API_KEY not set and not found in {RLM_DOTENV_PATH}. "
                "Set it in that file or pass api_key= explicitly."
            )

    guardrails = dict(guardrails or load_guardrails())

    logger = RLMLogger(log_dir=str(log_dir)) if log_dir else None

    rlm_instance = RLM(
        backend="anthropic",
        backend_kwargs={"model_name": model_name, "api_key": api_key},
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
