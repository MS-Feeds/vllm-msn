"""Common sample record shared by every dataset source in this directory
(LongBench v2 "long" bucket, synthetic NIAH), so runner/run_arm.py and
friends can stay dataset-agnostic.

Deliberately narrower than spec_prefill_llama/datasets/prep_longbench_v2.py's
row shape (which also carries domain/sub_domain/difficulty/choices/answer as
a single letter, LongBench-v2-specific fields) -- those are preserved as
`extra` rather than promoted to top-level fields, since a synthetic NIAH
sample has no equivalent for most of them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalSample:
    """One eval-set row: a (context, question) pair with a known-correct
    answer, plus enough metadata to trace it back to its source dataset and
    to sanity-check the >131K-token eval-set constraint downstream.
    """

    id: str
    source: str  # e.g. "longbench_v2_long", "synthetic_niah"
    context: str
    question: str
    answer: str
    # Token count under the TARGET model's real tokenizer (Llama), not an
    # estimate -- populated by filter_by_token_length.py. None until then.
    context_tokens: int | None = None
    # Source-specific fields that don't fit the common shape above (e.g.
    # LongBench v2's domain/sub_domain/difficulty/choices, or NIAH's needle
    # positions/count) -- kept so nothing is lost, without forcing every
    # source to share every field.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSample":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def write_jsonl(samples: list[EvalSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[EvalSample]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(EvalSample.from_dict(json.loads(line)))
    return samples
