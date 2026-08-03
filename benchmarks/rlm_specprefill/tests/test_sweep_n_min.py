"""Tests for calibration/sweep_n_min.py's pure-Python pieces (pooling,
bucketing/truncation, crossover computation, config writing) -- no vllm/GPU
needed. The GPU-only sweep orchestration (run_sweep, which builds real
vLLM engines) is not exercised here; see REPRODUCE.md's GPU-node steps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from calibration.sweep_n_min import (  # noqa: E402
    BinTiming,
    bucket_candidates,
    compute_crossover,
    pool_candidate_texts,
    write_n_min_config,
)
from eval_data.schema import EvalSample, write_jsonl  # noqa: E402
from rlm.core.types import RLMChatCompletion, UsageSummary  # noqa: E402
from rlm_stage.evidence_cache import save_cached  # noqa: E402
from rlm_stage.evidence_rlm import EvidenceResult  # noqa: E402
from target_stage.gate import load_n_min  # noqa: E402


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids):
        return " ".join(token_ids)


# ---------------------------------------------------------------------------
# pool_candidate_texts
# ---------------------------------------------------------------------------


def _make_cached_evidence_result(sample_id: str, excerpt_text: str, cache_dir: Path) -> None:
    completion = RLMChatCompletion(
        root_model="mock", prompt="...", response="...",
        usage_summary=UsageSummary(model_usage_summaries={}), execution_time=1.0,
    )
    result = EvidenceResult(
        sample_id=sample_id, completion=completion,
        evidence={"excerpts": [{"text": excerpt_text, "loc_hint": None}], "question": None, "parse_error": False},
        wall_time_s=1.0,
    )
    save_cached(f"key-{sample_id}", result, cache_dir=cache_dir)


def test_pool_candidate_texts_from_evidence_cache(tmp_path):
    cache_dir = tmp_path / "evidence_cache"
    _make_cached_evidence_result("s1", "excerpt one text", cache_dir)
    _make_cached_evidence_result("s2", "excerpt two text", cache_dir)

    texts = pool_candidate_texts(evidence_cache_dir=cache_dir)

    assert len(texts) == 2
    assert any("excerpt one text" in t for t in texts)
    assert any("excerpt two text" in t for t in texts)


def test_pool_candidate_texts_from_synthetic_niah(tmp_path):
    samples = [
        EvalSample(id="n1", source="synthetic_niah", context="context one", question="q1", answer="a1"),
        EvalSample(id="n2", source="synthetic_niah", context="context two", question="q2", answer="a2"),
    ]
    path = tmp_path / "niah.jsonl"
    write_jsonl(samples, path)

    texts = pool_candidate_texts(synthetic_niah_path=path)

    assert texts == ["context one", "context two"]


def test_pool_candidate_texts_combines_both_sources(tmp_path):
    cache_dir = tmp_path / "evidence_cache"
    _make_cached_evidence_result("s1", "cached excerpt", cache_dir)

    niah_path = tmp_path / "niah.jsonl"
    write_jsonl([EvalSample(id="n1", source="synthetic_niah", context="niah context", question="q", answer="a")], niah_path)

    texts = pool_candidate_texts(evidence_cache_dir=cache_dir, synthetic_niah_path=niah_path)

    assert len(texts) == 2


def test_pool_candidate_texts_raises_when_empty(tmp_path):
    with pytest.raises(ValueError, match="No candidate texts found"):
        pool_candidate_texts(evidence_cache_dir=tmp_path / "nonexistent", synthetic_niah_path=tmp_path / "nonexistent.jsonl")


# ---------------------------------------------------------------------------
# bucket_candidates
# ---------------------------------------------------------------------------


def test_bucket_candidates_truncates_to_exact_bin_size():
    tok = _FakeTokenizer()
    texts = [" ".join(f"w{i}" for i in range(5000))]  # one long text

    bins = bucket_candidates(texts, tok, [500, 1000, 2000])

    assert set(bins) == {500, 1000, 2000}
    for n, token_ids in bins.items():
        assert len(token_ids) == n
        assert token_ids == [f"w{i}" for i in range(n)]  # exact prefix, not a random slice


def test_bucket_candidates_omits_bins_with_no_long_enough_candidate():
    tok = _FakeTokenizer()
    texts = [" ".join(f"w{i}" for i in range(100))]  # only 100 tokens available

    bins = bucket_candidates(texts, tok, [50, 500, 5000])

    assert set(bins) == {50}  # only the bin the pool can actually cover


def test_bucket_candidates_picks_first_long_enough_candidate():
    """When multiple pooled texts could cover a bin, the first one long
    enough in pool order wins -- deterministic, not e.g. shortest-fit."""
    tok = _FakeTokenizer()
    short_text = " ".join(f"a{i}" for i in range(200))
    long_text = " ".join(f"b{i}" for i in range(2000))
    texts = [short_text, long_text]

    bins = bucket_candidates(texts, tok, [100])

    assert bins[100][0] == "a0"  # came from short_text, the first candidate that qualifies


# ---------------------------------------------------------------------------
# compute_crossover
# ---------------------------------------------------------------------------


def test_compute_crossover_finds_the_point_where_specprefill_wins_permanently():
    timings = [
        BinTiming(n_tokens=500, t_plain_s=0.1, t_specprefill_s=0.3),   # specprefill loses (fixed overhead dominates)
        BinTiming(n_tokens=1000, t_plain_s=0.2, t_specprefill_s=0.25),  # still loses
        BinTiming(n_tokens=5000, t_plain_s=1.0, t_specprefill_s=0.6),   # wins
        BinTiming(n_tokens=10000, t_plain_s=2.0, t_specprefill_s=0.9),  # wins
    ]
    assert compute_crossover(timings) == 5000


def test_compute_crossover_requires_winning_at_every_larger_bin_not_just_once():
    """A single early win that doesn't hold at larger sizes must NOT be
    reported as the crossover -- that would be noise, not a real
    crossover."""
    timings = [
        BinTiming(n_tokens=500, t_plain_s=0.5, t_specprefill_s=0.4),   # wins once (noise)
        BinTiming(n_tokens=1000, t_plain_s=0.5, t_specprefill_s=0.6),  # loses again
        BinTiming(n_tokens=5000, t_plain_s=1.0, t_specprefill_s=0.5),  # wins and holds from here
    ]
    assert compute_crossover(timings) == 5000


def test_compute_crossover_returns_none_when_specprefill_never_wins():
    timings = [
        BinTiming(n_tokens=500, t_plain_s=0.1, t_specprefill_s=0.3),
        BinTiming(n_tokens=5000, t_plain_s=1.0, t_specprefill_s=1.5),
    ]
    assert compute_crossover(timings) is None


def test_compute_crossover_handles_unsorted_input():
    timings = [
        BinTiming(n_tokens=5000, t_plain_s=1.0, t_specprefill_s=0.5),
        BinTiming(n_tokens=500, t_plain_s=0.1, t_specprefill_s=0.3),
    ]
    assert compute_crossover(timings) == 5000


# ---------------------------------------------------------------------------
# write_n_min_config (round-trips through target_stage/gate.py's loader)
# ---------------------------------------------------------------------------


def test_write_n_min_config_round_trips_through_gate_load_n_min(tmp_path):
    timings = [
        BinTiming(n_tokens=500, t_plain_s=0.1, t_specprefill_s=0.3),
        BinTiming(n_tokens=5000, t_plain_s=1.0, t_specprefill_s=0.5),
    ]
    path = tmp_path / "n_min.json"

    write_n_min_config(5000, timings, path)

    assert load_n_min(path) == 5000
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["curve"]) == 2
    assert data["curve"][0]["n_tokens"] == 500  # sorted ascending
