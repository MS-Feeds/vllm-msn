"""Arm C's query router: partitions queries into a plain-engine ("skip")
bucket and a SpecPrefill-engine ("compress") bucket BEFORE either vLLM
engine loads.

Per IMPLEMENTATION_PLAN.md decision 3, this ordering is not just an
optimization -- `worker_cls` is fixed at `LLM()` construction time (vLLM has
no hot-swap), so `run_all_arms.py` (not yet built) must know the full
skip/compress split up front to run the plain engine once over the skip
bucket, tear it down, then build the SpecPrefill engine once over the
compress bucket -- never two engines live at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from target_stage.gate import should_compress  # noqa: E402
from target_stage.vllm_offline_engine import TargetQuery, render_excerpts_text  # noqa: E402


def count_query_tokens(tokenizer, query: TargetQuery) -> int:
    """Tokenizes the query's excerpts text (the prunable portion) under the
    real target tokenizer -- the gate is about the compressed EVIDENCE
    size RLM produced, not the original raw document (already reduced away
    by the time this stage runs)."""
    text = render_excerpts_text(query.excerpts)
    return len(tokenizer.encode(text, add_special_tokens=False))


def route_queries(
    queries: list[TargetQuery], tokenizer, n_min: int
) -> tuple[list[TargetQuery], list[TargetQuery]]:
    """Returns (skip_bucket, compress_bucket). `skip_bucket` goes to
    `build_plain_target_engine`, `compress_bucket` to
    `build_specprefill_target_engine` (target_stage/vllm_offline_engine.py)."""
    skip_bucket: list[TargetQuery] = []
    compress_bucket: list[TargetQuery] = []

    for query in queries:
        n_tokens = count_query_tokens(tokenizer, query)
        if should_compress(n_tokens, n_min):
            compress_bucket.append(query)
        else:
            skip_bucket.append(query)

    return skip_bucket, compress_bucket
