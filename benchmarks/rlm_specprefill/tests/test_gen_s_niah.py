"""Tests for eval_data/gen_s_niah.py's pure-logic `generate_sample` -- no
network/GPU needed, same convention as tests/test_prep_longbench_v2_long.py:
seeded random.Random, direct import of the eval_data module."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval_data"))

import pytest  # noqa: E402

from gen_s_niah import (  # noqa: E402
    VALID_NEEDLE_TYPES,
    _ADJECTIVES,
    _NOUNS,
    _gen_needle_value,
    generate_sample,
)


def test_exactly_one_needle_per_sample():
    sample = generate_sample(random.Random(1), 1000, "number", 0)
    assert len(sample.extra["needles"]) == 1
    assert sample.extra["n_needles"] == 1


def test_source_is_s_niah():
    sample = generate_sample(random.Random(1), 1000, "number", 0)
    assert sample.source == "s_niah"


def test_needle_value_length_constant_across_context_sizes():
    # The RLM paper's key S-NIAH claim: the sought information scales as
    # O(1) with respect to input length -- the needle itself must not grow
    # just because the haystack does.
    small = generate_sample(random.Random(42), 500, "number", 0)
    large = generate_sample(random.Random(42), 500_000, "number", 0)
    assert len(small.extra["needles"][0]["value"]) == len(large.extra["needles"][0]["value"]) == 9


def test_needle_type_number_value_is_digits():
    sample = generate_sample(random.Random(2), 1000, "number", 0)
    assert sample.extra["needles"][0]["value"].isdigit()
    assert sample.extra["needle_type"] == "number"


def test_needle_type_phrase_value_is_two_words_from_wordlist():
    sample = generate_sample(random.Random(3), 1000, "phrase", 0)
    value = sample.extra["needles"][0]["value"]
    parts = value.split(" ")
    assert len(parts) == 2
    assert parts[0] in _ADJECTIVES
    assert parts[1] in _NOUNS
    assert sample.extra["needle_type"] == "phrase"


def test_mixed_needle_type_uses_both_variants_with_seeded_rng():
    rng = random.Random(7)
    types_seen = {generate_sample(rng, 1000, "mixed", i).extra["needle_type"] for i in range(20)}
    assert types_seen == {"number", "phrase"}


def test_question_and_answer_number():
    sample = generate_sample(random.Random(4), 1000, "number", 0)
    assert sample.answer == sample.extra["needles"][0]["value"]
    assert "number" in sample.question


def test_question_and_answer_phrase():
    sample = generate_sample(random.Random(5), 1000, "phrase", 0)
    assert sample.answer == sample.extra["needles"][0]["value"]
    assert "phrase" in sample.question


def test_answer_is_findable_in_context():
    sample = generate_sample(random.Random(6), 2000, "mixed", 0)
    assert sample.extra["needles"][0]["value"] in sample.context


def test_default_n_tasks_is_50(tmp_path, monkeypatch):
    # Exercises the REAL CLI parser end-to-end (not a re-declared copy of
    # its defaults) -- one context-size bucket, no --n-tasks flag passed,
    # so the output row count directly reveals the real default.
    import gen_s_niah

    output_path = tmp_path / "s_niah_default_check.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["gen_s_niah.py", "--context-tokens", "500", "--output", str(output_path)]
    )
    gen_s_niah.main()
    rows = output_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 50


def test_valid_needle_types_constant():
    assert set(VALID_NEEDLE_TYPES) == {"number", "phrase", "mixed"}


def test_invalid_needle_type_raises():
    with pytest.raises(ValueError, match="needle_type must be"):
        _gen_needle_value(random.Random(1), "bogus")
    with pytest.raises(ValueError, match="needle_type must be one of"):
        generate_sample(random.Random(1), 1000, "bogus", 0)
