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
from rlm_stage.evidence_rlm import EvidenceResult  # noqa: E402
from runner import run_arm as run_arm_module  # noqa: E402
from runner.run_arm import VALID_ARMS, collect_evidence_for_dataset, run_arm  # noqa: E402
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
