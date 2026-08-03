"""Tests for rlm_stage/evidence_cache.py using a stub run_fn -- no real RLM
call, no live Anthropic API, no GPU. See that module's docstring for why a
stub is used instead of monkeypatching RLM's client internals.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_data.schema import EvalSample  # noqa: E402
from rlm.core.types import RLMChatCompletion, UsageSummary  # noqa: E402
from rlm_stage import evidence_cache  # noqa: E402
from rlm_stage.evidence_rlm import EvidenceResult  # noqa: E402


def _make_sample(sample_id: str = "s1", context: str = "ctx", question: str = "q?") -> EvalSample:
    return EvalSample(id=sample_id, source="test", context=context, question=question, answer="a")


def _stub_run_fn(calls: list[str]):
    def run_fn(sample: EvalSample) -> EvidenceResult:
        calls.append(sample.id)
        completion = RLMChatCompletion(
            root_model="mock",
            prompt=sample.context,
            response=f"mock evidence for {sample.id}",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=1.23,
        )
        return EvidenceResult(
            sample_id=sample.id,
            completion=completion,
            evidence={"excerpts": [], "question": sample.question, "parse_error": False},
            wall_time_s=1.23,
        )

    return run_fn


def test_cache_miss_then_hit(tmp_path):
    """First call runs RLM (miss); second call for the identical sample
    reuses the cache (hit) and never re-invokes run_fn -- the core
    confound-control guarantee (IMPLEMENTATION_PLAN.md decision 4)."""
    sample = _make_sample()
    calls: list[str] = []
    run_fn = _stub_run_fn(calls)

    result1, hit1 = evidence_cache.get_or_run(sample, run_fn, cache_dir=tmp_path)
    result2, hit2 = evidence_cache.get_or_run(sample, run_fn, cache_dir=tmp_path)

    assert hit1 is False
    assert hit2 is True
    assert calls == ["s1"]  # run_fn invoked exactly once
    assert result2.completion.response == result1.completion.response
    assert result2.evidence == result1.evidence


def test_force_refresh_bypasses_cache(tmp_path):
    sample = _make_sample()
    calls: list[str] = []
    run_fn = _stub_run_fn(calls)

    evidence_cache.get_or_run(sample, run_fn, cache_dir=tmp_path)
    _, hit = evidence_cache.get_or_run(sample, run_fn, cache_dir=tmp_path, force_refresh=True)

    assert hit is False
    assert calls == ["s1", "s1"]  # run_fn invoked twice


def test_different_samples_get_different_cache_entries(tmp_path):
    calls: list[str] = []
    run_fn = _stub_run_fn(calls)

    sample_a = _make_sample(sample_id="a", context="context A")
    sample_b = _make_sample(sample_id="b", context="context B")

    evidence_cache.get_or_run(sample_a, run_fn, cache_dir=tmp_path)
    evidence_cache.get_or_run(sample_b, run_fn, cache_dir=tmp_path)

    assert calls == ["a", "b"]  # both ran, no accidental collision


def test_cache_key_sensitive_to_context_question_and_guardrails():
    """Every field IMPLEMENTATION_PLAN.md decision 4 lists as part of the
    key must actually change the key -- otherwise a prompt/config change
    could silently serve stale evidence to a later arm."""
    base = _make_sample()
    key_base = evidence_cache.compute_cache_key(base, guardrails={"max_depth": 2})

    different_context = _make_sample(context="a completely different context")
    assert evidence_cache.compute_cache_key(different_context, guardrails={"max_depth": 2}) != key_base

    different_question = _make_sample(question="a completely different question?")
    assert evidence_cache.compute_cache_key(different_question, guardrails={"max_depth": 2}) != key_base

    different_guardrails = evidence_cache.compute_cache_key(base, guardrails={"max_depth": 3})
    assert different_guardrails != key_base

    different_prompt_version = evidence_cache.compute_cache_key(
        base, guardrails={"max_depth": 2}, prompt_version="v2"
    )
    assert different_prompt_version != key_base


def test_cache_key_stable_and_guardrail_key_order_independent():
    """Same sample + same guardrails (regardless of dict key order) must
    always produce the same key -- otherwise dict iteration order could
    cause spurious cache misses across process runs."""
    sample = _make_sample()

    key1 = evidence_cache.compute_cache_key(sample, guardrails={"max_depth": 2, "max_iterations": 30})
    key2 = evidence_cache.compute_cache_key(sample, guardrails={"max_iterations": 30, "max_depth": 2})
    key3 = evidence_cache.compute_cache_key(sample, guardrails={"max_depth": 2, "max_iterations": 30})

    assert key1 == key2 == key3


def test_load_cached_returns_none_for_missing_key(tmp_path):
    assert evidence_cache.load_cached("nonexistent_key_1234", cache_dir=tmp_path) is None


def test_evidence_result_round_trips_through_json(tmp_path):
    """save_cached()/load_cached() must preserve every field, including the
    nested RLMChatCompletion structure -- not just the top-level dataclass
    shape (dataclasses.asdict alone wouldn't round-trip UsageSummary)."""
    sample = _make_sample()
    calls: list[str] = []
    run_fn = _stub_run_fn(calls)

    original, _ = evidence_cache.get_or_run(sample, run_fn, cache_dir=tmp_path)
    key = evidence_cache.compute_cache_key(sample)
    loaded = evidence_cache.load_cached(key, cache_dir=tmp_path)

    assert loaded is not None
    assert loaded.sample_id == original.sample_id
    assert loaded.completion.response == original.completion.response
    assert loaded.completion.execution_time == original.completion.execution_time
    assert loaded.evidence == original.evidence
    assert loaded.wall_time_s == original.wall_time_s
    assert loaded.prompt_version == original.prompt_version
