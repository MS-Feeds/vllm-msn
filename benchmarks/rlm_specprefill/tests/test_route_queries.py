"""Tests for target_stage/route_queries.py. Pure Python (fake tokenizer, no
network/GPU) -- see test_vllm_offline_engine.py's _FakeTokenizer for why a
hand-written fake is used instead of a real HF download.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from target_stage.route_queries import count_query_tokens, route_queries  # noqa: E402
from target_stage.vllm_offline_engine import TargetQuery  # noqa: E402


class _FakeTokenizer:
    """encode() = one "token" per whitespace-separated word -- deterministic
    and easy to reason about in assertions."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


def _make_query(sample_id: str, n_words: int) -> TargetQuery:
    excerpt_text = " ".join(f"word{i}" for i in range(n_words))
    return TargetQuery(sample_id=sample_id, question="q?", excerpts=[{"text": excerpt_text, "loc_hint": None}])


# render_excerpts_text adds a "[Excerpt 1]" header per excerpt (see
# vllm_offline_engine.py), which under this word-splitting fake tokenizer
# contributes 2 extra "tokens" ("[Excerpt", "1]") beyond the excerpt's own
# word count -- accounted for explicitly here rather than assumed away, so
# these tests reflect what count_query_tokens actually measures (the
# rendered excerpts block, headers included, not just raw excerpt text).
_HEADER_TOKEN_OVERHEAD = 2


def test_count_query_tokens_counts_rendered_excerpts_including_header():
    tok = _FakeTokenizer()
    query = _make_query("s1", n_words=10)
    assert count_query_tokens(tok, query) == 10 + _HEADER_TOKEN_OVERHEAD


def test_route_queries_splits_by_n_min():
    tok = _FakeTokenizer()
    n_min = 100
    queries = [
        _make_query("small", n_words=5),
        _make_query("large", n_words=500),
        # Exactly at n_min after accounting for header overhead -- should
        # NOT compress (gate.py's rule is strictly N > N_min).
        _make_query("boundary", n_words=n_min - _HEADER_TOKEN_OVERHEAD),
        # One token over the boundary -- SHOULD compress.
        _make_query("just_over", n_words=n_min - _HEADER_TOKEN_OVERHEAD + 1),
    ]

    skip_bucket, compress_bucket = route_queries(queries, tok, n_min=n_min)

    skip_ids = {q.sample_id for q in skip_bucket}
    compress_ids = {q.sample_id for q in compress_bucket}

    assert skip_ids == {"small", "boundary"}
    assert compress_ids == {"large", "just_over"}


def test_route_queries_preserves_all_queries_exactly_once():
    tok = _FakeTokenizer()
    queries = [_make_query(f"s{i}", n_words=i * 10) for i in range(20)]

    skip_bucket, compress_bucket = route_queries(queries, tok, n_min=100)

    all_routed_ids = [q.sample_id for q in skip_bucket] + [q.sample_id for q in compress_bucket]
    assert sorted(all_routed_ids) == sorted(q.sample_id for q in queries)
    assert len(all_routed_ids) == len(queries)  # nothing duplicated, nothing dropped


def test_route_queries_empty_input():
    tok = _FakeTokenizer()
    skip_bucket, compress_bucket = route_queries([], tok, n_min=100)
    assert skip_bucket == []
    assert compress_bucket == []
