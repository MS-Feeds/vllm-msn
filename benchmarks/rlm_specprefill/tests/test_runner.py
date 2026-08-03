"""Tests for runner/run_arm.py and runner/run_all_arms.py's pure logic
(validation, arm ordering, evidence-collection wiring) -- no real Anthropic
API calls, no vllm/GPU. `collect_evidence_for_dataset` is tested by
monkeypatching `run_evidence_extraction` in run_arm's own namespace with a
stub, same pattern as tests/test_evidence_cache.py's injected `run_fn` (see
that module's docstring for why: RLM's `backend=` string-routed client
construction has no clean seam for a mock otherwise).

The real end-to-end path (real Anthropic API, real caching, via the actual
CLI) is exercised separately by runner/smoke_test.py and by manually running
`run_arm.py --arm A --dry-run` against a tiny generated dataset -- both
confirmed working during implementation (see IMPLEMENTATION_PLAN.md's
Verification section).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from eval_data.schema import EvalSample, write_jsonl  # noqa: E402
from rlm.core.types import RLMChatCompletion, UsageSummary  # noqa: E402
from rlm_stage import evidence_cache  # noqa: E402
from rlm_stage.evidence_rlm import EvidenceResult  # noqa: E402
from runner import run_arm as run_arm_module  # noqa: E402
from runner.run_arm import (  # noqa: E402
    VALID_ARMS,
    _all_samples_cached,
    collect_evidence_for_dataset,
    run_arm,
)
from runner.run_all_arms import run_all_arms  # noqa: E402


def _fake_run_evidence_extraction(sample_id, question, context, **kwargs):
    completion = RLMChatCompletion(
        root_model="mock",
        prompt=context,
        response=json.dumps({"excerpts": [{"text": f"evidence for {sample_id}", "loc_hint": None}], "question": question}),
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=1.0,
    )
    return EvidenceResult(
        sample_id=sample_id,
        completion=completion,
        evidence={"excerpts": [{"text": f"evidence for {sample_id}", "loc_hint": None}], "question": question, "parse_error": False},
        wall_time_s=1.0,
    )


def test_run_arm_rejects_unknown_arm(tmp_path):
    dataset_path = tmp_path / "empty.jsonl"
    write_jsonl([], dataset_path)
    with pytest.raises(ValueError, match="Unknown arm"):
        run_arm("Z", dataset_path)


def test_run_arm_rejects_empty_dataset(tmp_path):
    dataset_path = tmp_path / "empty.jsonl"
    write_jsonl([], dataset_path)
    with pytest.raises(ValueError, match="No samples loaded"):
        run_arm("A", dataset_path)


def test_collect_evidence_for_dataset_uses_cache_across_calls(tmp_path, monkeypatch):
    calls: list[str] = []

    def counting_stub(sample_id, question, context, **kwargs):
        calls.append(sample_id)
        return _fake_run_evidence_extraction(sample_id, question, context, **kwargs)

    monkeypatch.setattr(run_arm_module, "run_evidence_extraction", counting_stub)

    sample = EvalSample(id="s1", source="test", context="ctx", question="q?", answer="a")
    cache_dir = tmp_path / "cache"

    results_1 = collect_evidence_for_dataset([sample], cache_dir=cache_dir, guardrails={})
    results_2 = collect_evidence_for_dataset([sample], cache_dir=cache_dir, guardrails={})

    assert calls == ["s1"]  # RLM only ever invoked once
    assert results_1[0].was_cache_hit is False
    assert results_2[0].was_cache_hit is True
    assert results_2[0].evidence_result.completion.response == results_1[0].evidence_result.completion.response


def test_run_arm_dry_run_writes_timing_jsonl_and_stops_before_target(tmp_path, monkeypatch):
    monkeypatch.setattr(run_arm_module, "run_evidence_extraction", _fake_run_evidence_extraction)

    samples = [
        EvalSample(id="a", source="test", context="ctx a", question="qa?", answer="a"),
        EvalSample(id="b", source="test", context="ctx b", question="qb?", answer="b"),
    ]
    dataset_path = tmp_path / "samples.jsonl"
    write_jsonl(samples, dataset_path)
    results_dir = tmp_path / "results"

    run_arm("A", dataset_path, results_dir=results_dir, dry_run=True)

    timing_path = results_dir / "A" / "timing.jsonl"
    assert timing_path.exists()
    rows = [json.loads(line) for line in timing_path.read_text(encoding="utf-8").splitlines()]
    assert {r["sample_id"] for r in rows} == {"a", "b"}
    for row in rows:
        assert row["wall_time_s"] == 1.0

    # dry-run must not have produced a predictions.jsonl (that only happens
    # after the target-model stage, which --dry-run explicitly skips).
    assert not (results_dir / "A" / "predictions.jsonl").exists()


def test_run_arm_dry_run_respects_max_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(run_arm_module, "run_evidence_extraction", _fake_run_evidence_extraction)

    samples = [
        EvalSample(id=f"s{i}", source="test", context=f"ctx {i}", question=f"q{i}?", answer="a")
        for i in range(5)
    ]
    dataset_path = tmp_path / "samples.jsonl"
    write_jsonl(samples, dataset_path)
    results_dir = tmp_path / "results"

    run_arm("A", dataset_path, results_dir=results_dir, dry_run=True, max_samples=2)

    timing_path = results_dir / "A" / "timing.jsonl"
    rows = [json.loads(line) for line in timing_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2


def test_run_all_arms_forces_a_first_regardless_of_requested_order(tmp_path, monkeypatch):
    call_order: list[str] = []

    def fake_run_arm(arm, dataset_path, **kwargs):
        call_order.append(arm)

    import runner.run_all_arms as run_all_arms_module

    monkeypatch.setattr(run_all_arms_module, "run_arm", fake_run_arm)

    run_all_arms(tmp_path / "dataset.jsonl", arms=["C", "B", "A"])

    assert call_order == ["A", "B", "C"]


def test_run_all_arms_dedupes_and_respects_subset(tmp_path, monkeypatch):
    call_order: list[str] = []

    def fake_run_arm(arm, dataset_path, **kwargs):
        call_order.append(arm)

    import runner.run_all_arms as run_all_arms_module

    monkeypatch.setattr(run_all_arms_module, "run_arm", fake_run_arm)

    run_all_arms(tmp_path / "dataset.jsonl", arms=["B", "B", "A"])

    assert call_order == ["A", "B"]  # deduped, and still A-first
    assert set(call_order).issubset(set(VALID_ARMS))


def test_all_samples_cached_false_then_true(tmp_path):
    samples = [
        EvalSample(id="a", source="test", context="ctx a", question="qa?", answer="a"),
        EvalSample(id="b", source="test", context="ctx b", question="qb?", answer="b"),
    ]
    cache_dir = tmp_path / "cache"

    assert _all_samples_cached(samples, guardrails={}, cache_dir=cache_dir) is False

    for sample in samples:
        key = evidence_cache.compute_cache_key(sample, guardrails={})
        evidence_cache.save_cached(key, _fake_run_evidence_extraction(sample.id, sample.question, sample.context), cache_dir)

    assert _all_samples_cached(samples, guardrails={}, cache_dir=cache_dir) is True


def test_all_samples_cached_false_if_only_some_cached(tmp_path):
    samples = [
        EvalSample(id="a", source="test", context="ctx a", question="qa?", answer="a"),
        EvalSample(id="b", source="test", context="ctx b", question="qb?", answer="b"),
    ]
    cache_dir = tmp_path / "cache"
    key_a = evidence_cache.compute_cache_key(samples[0], guardrails={})
    evidence_cache.save_cached(key_a, _fake_run_evidence_extraction("a", "qa?", "ctx a"), cache_dir)

    assert _all_samples_cached(samples, guardrails={}, cache_dir=cache_dir) is False


def test_run_arm_vllm_root_backend_skips_server_when_fully_cached(tmp_path, monkeypatch):
    """The whole point of _all_samples_cached's pre-check: a run_arm() call
    with root_backend='vllm' over an already-fully-cached dataset (e.g.
    Arm B/C after Arm A already populated the cache) must never even try to
    start a vllm serve subprocess -- confirmed here by making
    start_root_vllm_server raise if it's ever called at all."""
    monkeypatch.setattr(run_arm_module, "run_evidence_extraction", _fake_run_evidence_extraction)

    samples = [EvalSample(id="a", source="test", context="ctx a", question="qa?", answer="a")]
    dataset_path = tmp_path / "samples.jsonl"
    write_jsonl(samples, dataset_path)
    results_dir = tmp_path / "results"

    # Pre-warm the cache exactly like Arm A would have -- using the SAME
    # guardrails run_arm()'s own _all_samples_cached/collect_evidence_for_dataset
    # calls resolve to by default (load_guardrails()'s real config file, not
    # an empty dict), since compute_cache_key hashes guardrails in: a
    # mismatched guardrails dict here would produce a different cache key
    # and defeat the whole point of this test.
    from rlm_stage.evidence_rlm import load_guardrails

    real_guardrails = load_guardrails()
    key = evidence_cache.compute_cache_key(samples[0], guardrails=real_guardrails)
    evidence_cache.save_cached(
        key, _fake_run_evidence_extraction("a", "qa?", "ctx a"), results_dir / "evidence_cache"
    )

    import rlm_stage.root_vllm_server as root_vllm_server_module

    def _explode(*args, **kwargs):
        raise AssertionError("start_root_vllm_server must not be called when the cache is fully warm")

    monkeypatch.setattr(root_vllm_server_module, "start_root_vllm_server", _explode)

    # dry_run=True: stops before the target stage, which is all this test
    # needs -- it only cares whether the root server was (correctly) never
    # started, not about the target-model path.
    run_arm(
        "A",
        dataset_path,
        results_dir=results_dir,
        dry_run=True,
        root_backend="vllm",
        root_model_path="/fake/model/path",
    )
