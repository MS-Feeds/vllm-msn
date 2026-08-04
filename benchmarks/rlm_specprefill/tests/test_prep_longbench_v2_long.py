"""Tests for eval_data/prep_longbench_v2_long.py's pure-logic
`build_long_samples` -- synthetic rows, no real HF download/network needed
(matches this project's convention of separating pure-logic pieces from
network/GPU-dependent ones for unit testing)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval_data"))

from prep_longbench_v2_long import VALID_LENGTHS, build_long_samples, default_output_for  # noqa: E402


def _make_row(row_id: str, length: str, *, context_words: int = 10) -> dict:
    return {
        "_id": row_id,
        "domain": "test",
        "sub_domain": "test",
        "difficulty": "easy",
        "length": length,
        "question": "What is the answer?",
        "choice_A": "one",
        "choice_B": "two",
        "choice_C": "three",
        "choice_D": "four",
        "answer": "b",  # deliberately lowercase -- confirms .upper() normalization
        "context": " ".join(["word"] * context_words),
    }


def test_default_length_is_long():
    rows = [_make_row("a", "long"), _make_row("b", "short"), _make_row("c", "medium")]
    samples = build_long_samples(rows)
    assert [s.id for s in samples] == ["a"]
    assert samples[0].source == "longbench_v2_long"


def test_length_short_filters_to_short_only():
    rows = [_make_row("a", "long"), _make_row("b", "short"), _make_row("c", "medium")]
    samples = build_long_samples(rows, length="short")
    assert [s.id for s in samples] == ["b"]
    assert samples[0].source == "longbench_v2_short"


def test_length_medium_filters_to_medium_only():
    rows = [_make_row("a", "long"), _make_row("b", "short"), _make_row("c", "medium")]
    samples = build_long_samples(rows, length="medium")
    assert [s.id for s in samples] == ["c"]
    assert samples[0].source == "longbench_v2_medium"


def test_invalid_length_raises():
    import pytest

    with pytest.raises(ValueError, match="length must be one of"):
        build_long_samples([], length="bogus")


def test_answer_letter_normalized_to_uppercase():
    rows = [_make_row("a", "short")]
    samples = build_long_samples(rows, length="short")
    assert samples[0].answer == "B"


def test_bad_row_skipped_regardless_of_length():
    row = _make_row("a", "short")
    row["question"] = ""  # missing required field
    samples = build_long_samples([row], length="short")
    assert samples == []


def test_word_count_mismatch_only_tracked_for_long(capsys):
    # A "long" row that's actually short by word count should still be
    # kept (length label is authoritative) but should trigger the WARNING
    # print; a "short" row with the same tiny word count should NOT
    # trigger an equivalent warning, since no threshold is defined for it.
    rows_long = [_make_row("a", "long", context_words=5)]
    build_long_samples(rows_long, length="long")
    assert "WARNING" in capsys.readouterr().out

    rows_short = [_make_row("b", "short", context_words=5)]
    build_long_samples(rows_short, length="short")
    assert "WARNING" not in capsys.readouterr().out


def test_max_keep_caps_and_sorts_by_id():
    rows = [_make_row(f"id{i}", "short") for i in range(5)]
    samples = build_long_samples(rows, length="short", max_keep=2, seed=1)
    assert len(samples) == 2
    assert samples == sorted(samples, key=lambda s: s.id)


def test_valid_lengths_constant():
    assert set(VALID_LENGTHS) == {"short", "medium", "long"}


def test_default_output_for_is_length_aware():
    assert default_output_for("short").name == "longbench_v2_short_samples.jsonl"
    assert default_output_for("long").name == "longbench_v2_long_samples.jsonl"
