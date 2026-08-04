"""Content-hash cache/replay layer for RLM evidence extraction.

Per IMPLEMENTATION_PLAN.md decision 4: this is the confound-control
mechanism for arms A/B/C, not an optional optimization. AnthropicClient
never forwards `sampling_args`/temperature (see IMPLEMENTATION_PLAN.md's
"load-bearing findings" -- ../../rlm/rlm/clients/anthropic.py), so there is
no temperature=0 fallback for making repeated RLM runs deterministic. Every
arm must see byte-identical evidence for a given query, which means RLM is
run at most once per query, ever -- arms B and C always read from this
cache and never re-invoke RLM.

`run_fn` is injected (not imported/hardcoded) specifically so this module's
caching/hashing/persistence logic can be unit-tested against a cheap stub,
independent of rlm_stage.evidence_rlm's real dependency on a live Anthropic
API call. `RLM`'s constructor takes a `backend` string routed through
`rlm.clients.get_client` (../../rlm/rlm/clients/__init__.py) with no clean
seam for substituting a mock client short of monkeypatching rlm internals --
injecting the run function here sidesteps that entirely. See
tests/test_evidence_cache.py for how the stub is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_data.schema import EvalSample  # noqa: E402
from rlm_stage.evidence_rlm import PROMPT_VERSION, EvidenceResult  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent.parent
# Honors $RLM_SPECPREFILL_RESULTS_DIR (see runner/run_arm.py's
# DEFAULT_RESULTS_DIR for the fuller writeup of why -- this module's own
# default was equally vestigial-until-fixed, even though run_arm.py itself
# never actually relies on it, always passing cache_dir= explicitly).
DEFAULT_CACHE_DIR = Path(os.environ.get("RLM_SPECPREFILL_RESULTS_DIR", str(THIS_DIR / "results"))) / "evidence_cache"


def compute_cache_key(
    sample: EvalSample,
    *,
    prompt_version: str = PROMPT_VERSION,
    guardrails: dict[str, Any] | None = None,
) -> str:
    """sha256 over (sample_id, context, question, prompt_version,
    guardrails) -- see module docstring for why every one of these must be
    part of the key: a change to any of them changes what RLM would
    actually produce, so a stale cache entry keyed on fewer fields could
    silently serve evidence gathered under a different prompt/config to a
    later arm.
    """
    hasher = hashlib.sha256()
    hasher.update(sample.id.encode("utf-8"))
    hasher.update(sample.context.encode("utf-8"))
    hasher.update(sample.question.encode("utf-8"))
    hasher.update(prompt_version.encode("utf-8"))
    # sort_keys: guardrail dict key order must not affect the hash.
    hasher.update(json.dumps(guardrails or {}, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def _cache_path(key: str, cache_dir: Path) -> Path:
    return cache_dir / f"{key}.json"


def load_cached(key: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> EvidenceResult | None:
    path = _cache_path(key, cache_dir)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return EvidenceResult.from_dict(data)


def save_cached(key: str, result: EvidenceResult, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key, cache_dir)
    # Write to a temp file then rename: avoids leaving a half-written cache
    # entry behind if the process is killed mid-write (e.g. hitting
    # max_timeout on a later query in the same batch run).
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def get_or_run(
    sample: EvalSample,
    run_fn: Callable[[EvalSample], EvidenceResult],
    *,
    prompt_version: str = PROMPT_VERSION,
    guardrails: dict[str, Any] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> tuple[EvidenceResult, bool]:
    """Returns (result, was_cache_hit).

    Arms B/C should always call this (never `run_fn` directly) so they are
    structurally incapable of re-invoking RLM for a query Arm A (or an
    earlier B/C run) already cached -- see module docstring for why that
    matters here specifically (no sampling-based determinism available).
    """
    key = compute_cache_key(sample, prompt_version=prompt_version, guardrails=guardrails)

    if not force_refresh:
        cached = load_cached(key, cache_dir)
        if cached is not None:
            return cached, True

    result = run_fn(sample)
    save_cached(key, result, cache_dir)
    return result, False
