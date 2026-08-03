"""Tests for the pure-Python pieces of target_stage/vllm_offline_engine.py
(prompt formatting, budget sizing, data shapes) -- everything that doesn't
need vllm/torch installed. Engine construction and request submission
(build_plain_target_engine, build_specprefill_target_engine, answer_batch,
_submit_plain, _submit_pruned) are GPU-only and covered by REPRODUCE.md's
validation steps instead, not here.

Uses a hand-written fake tokenizer (not a real HF download) so these tests
are deterministic and don't depend on network access -- it mimics just
enough of a real chat-template tokenizer's shape (apply_chat_template
wrapping content in fixed before/after text, encode() as a length-bearing
function) to exercise chat_wrapper_pieces/build_prompt_pieces correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_data.schema import EvalSample  # noqa: E402
from rlm.core.types import RLMChatCompletion, UsageSummary  # noqa: E402
from rlm_stage.evidence_rlm import EvidenceResult  # noqa: E402
from target_stage.vllm_offline_engine import (  # noqa: E402
    TargetAnswer,
    TargetQuery,
    build_prompt_pieces,
    chat_wrapper_pieces,
    percentile,
    render_excerpts_text,
    resolve_max_num_batched_tokens,
)


class _FakeTokenizer:
    """Mimics a chat-template tokenizer's externally-visible shape: content
    gets sandwiched between fixed BEFORE/AFTER strings, and encode() is a
    simple deterministic word-count so token-length assertions are easy to
    reason about."""

    BEFORE = "<user>"
    AFTER = "<assistant>"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        content = messages[0]["content"]
        return f"{self.BEFORE}{content}{self.AFTER}"

    def encode(self, text, add_special_tokens=False):
        return text.split()  # one "token" per whitespace-separated word


def test_render_excerpts_text_formats_with_loc_hints():
    excerpts = [
        {"text": "first excerpt", "loc_hint": "para 3"},
        {"text": "second excerpt", "loc_hint": None},
    ]
    rendered = render_excerpts_text(excerpts)

    assert "[Excerpt 1 -- para 3]" in rendered
    assert "first excerpt" in rendered
    assert "[Excerpt 2]" in rendered  # no loc_hint -> no " -- " suffix
    assert "second excerpt" in rendered
    # excerpts must appear in order, separated
    assert rendered.index("first excerpt") < rendered.index("second excerpt")


def test_render_excerpts_text_empty_list():
    assert render_excerpts_text([]) == ""


def test_chat_wrapper_pieces_splits_before_after():
    tok = _FakeTokenizer()
    before, after = chat_wrapper_pieces(tok)
    assert before == _FakeTokenizer.BEFORE
    assert after == _FakeTokenizer.AFTER


def test_build_prompt_pieces_structure():
    tok = _FakeTokenizer()
    excerpts = [{"text": "the answer is 42", "loc_hint": "line 5"}]
    prefix, context, suffix = build_prompt_pieces(tok, "What is the answer?", excerpts)

    assert prefix.startswith(_FakeTokenizer.BEFORE)
    assert "EXCERPTS" in prefix
    assert "the answer is 42" in context
    assert "What is the answer?" in suffix
    assert suffix.endswith(_FakeTokenizer.AFTER)
    # question must be in the suffix, NOT the prunable context -- the
    # question must never compete with excerpts for SpecPrefill's scoring
    # budget (see module docstring's gibberish-output history).
    assert "What is the answer?" not in context


def test_percentile_basic():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 0.5) == 3.0
    assert percentile(vals, 0.0) == 1.0
    assert percentile(vals, 1.0) == 5.0


def test_percentile_empty_and_single():
    assert percentile([], 0.5) is None
    assert percentile([7.0], 0.9) == 7.0


def test_resolve_max_num_batched_tokens_explicit_wins():
    assert resolve_max_num_batched_tokens(explicit=1234, token_lengths=[100, 200, 99999]) == 1234


def test_resolve_max_num_batched_tokens_empty_falls_back_to_default():
    from target_stage.vllm_offline_engine import DEFAULT_MAX_NUM_BATCHED_TOKENS

    assert resolve_max_num_batched_tokens(explicit=None, token_lengths=[]) == DEFAULT_MAX_NUM_BATCHED_TOKENS


def test_resolve_max_num_batched_tokens_sizes_off_p90_not_max():
    """A single extreme outlier must not dictate the auto-sized budget --
    the confirmed real-hardware OOM predict_longbench_v2.py's own version
    of this function was written to avoid."""
    token_lengths = [100] * 9 + [999999]  # p90 is ~100, max is 999999
    result = resolve_max_num_batched_tokens(explicit=None, token_lengths=token_lengths)

    assert result < 999999  # must NOT scale off the outlier
    assert result >= 100  # must comfortably cover the p90 body of samples
    assert result % 1024 == 0  # rounded up to a clean multiple, per the function's own convention


def test_target_query_from_evidence_result_uses_sample_question_not_echoed_one():
    """TargetQuery must use the ORIGINAL sample's question, not
    EvidenceResult.evidence['question'] -- that field is RLM's own echo and
    is None whenever parse_evidence_response fell back to the unstructured
    path (see module docstring)."""
    sample = EvalSample(id="s1", source="test", context="ctx", question="the real question?", answer="a")
    completion = RLMChatCompletion(
        root_model="mock",
        prompt="ctx",
        response="...",
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=1.0,
    )
    evidence_result = EvidenceResult(
        sample_id="s1",
        completion=completion,
        evidence={"excerpts": [{"text": "e1", "loc_hint": None}], "question": None, "parse_error": True},
        wall_time_s=1.0,
    )

    query = TargetQuery.from_evidence_result(sample, evidence_result)

    assert query.question == "the real question?"  # from the sample, not evidence["question"]
    assert query.excerpts == [{"text": "e1", "loc_hint": None}]


def test_target_answer_keep_rate():
    plain_answer = TargetAnswer(
        sample_id="s1", answer_text="a", finish_reason="stop", n_output_tokens=5,
        n_prompt_tokens_full=100, n_prompt_tokens_kept=None, ttft_ms=50.0,
        generation_time_s=1.2,
    )
    assert plain_answer.keep_rate is None  # no pruning happened

    pruned_answer = TargetAnswer(
        sample_id="s2", answer_text="a", finish_reason="stop", n_output_tokens=5,
        n_prompt_tokens_full=100, n_prompt_tokens_kept=30, ttft_ms=20.0,
        generation_time_s=0.6,
    )
    assert pruned_answer.keep_rate == 0.3
