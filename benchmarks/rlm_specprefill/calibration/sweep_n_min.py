#!/usr/bin/env python3
"""N_min crossover calibration: sweeps candidate-text token counts against
the target+speculator pair ALONE, decoupled from live RLM.

Per ../rlm_specprefill_ablation_plan.md's SPECPREFILL GATE section:
"Calibrate N_min empirically, decoupled from RLM's variance: use
synthetic/fixed candidate texts at controlled token counts (drawn from real
RLM outputs already collected, not from repeatedly re-running RLM), sweep
against the target+speculator pair alone, and take the empirical crossover.
This is a prerequisite step, not part of the main comparison." This module
never invokes RLM.

Candidate texts are pooled from two sources (either alone is sufficient):
- `results/evidence_cache/`'s cached `EvidenceResult`s, once
  `runner/run_arm.py` has populated it with real RLM evidence.
- `eval_data/`'s synthetic NIAH samples -- the day-1 bootstrap, usable
  before any real RLM evidence exists.

Pooled texts are bucketed into fixed token-count bins (truncated, not
padded -- see `bucket_candidates`'s docstring), each bin is fed through
both the plain and SpecPrefill target engines (`target_stage/vllm_offline_engine.py`),
and the empirical crossover is written to `configs/n_min.json` in the shape
`target_stage/gate.py::load_n_min` expects.

Usage (GPU node):
    python3 calibration/sweep_n_min.py \\
        --synthetic-niah eval_data/synthetic_niah_samples.jsonl \\
        --evidence-cache results/evidence_cache \\
        --target-model $LLAMA31_8B_MODEL_PATH --speculator-model $LLAMA32_1B_MODEL_PATH
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))

from eval_data.schema import read_jsonl  # noqa: E402
from rlm_stage.evidence_rlm import EvidenceResult  # noqa: E402
from target_stage.gate import DEFAULT_N_MIN_PATH  # noqa: E402
from target_stage.vllm_offline_engine import render_excerpts_text  # noqa: E402

DEFAULT_BIN_SIZES = [500, 1000, 2000, 5000, 10000, 20000, 50000]
DEFAULT_SPEC_CONFIG_PATH = THIS_DIR / "configs" / "spec_config_always_on.yaml"


# ---------------------------------------------------------------------------
# Candidate pooling + bucketing (pure Python -- no vllm needed).
# ---------------------------------------------------------------------------


def pool_candidate_texts(
    evidence_cache_dir: Path | None = None,
    synthetic_niah_path: Path | None = None,
) -> list[str]:
    """Gathers candidate excerpt-blob texts from cached RLM evidence and/or
    synthetic NIAH sample contexts. At least one source must yield
    something -- raises rather than silently sweeping over an empty pool
    (a silent empty sweep would write a meaningless N_min with no signal
    it was meaningless)."""
    texts: list[str] = []

    if evidence_cache_dir is not None and evidence_cache_dir.exists():
        for path in sorted(evidence_cache_dir.glob("*.json")):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            evidence_result = EvidenceResult.from_dict(data)
            excerpts = evidence_result.evidence.get("excerpts", [])
            if excerpts:
                texts.append(render_excerpts_text(excerpts))

    if synthetic_niah_path is not None and synthetic_niah_path.exists():
        for sample in read_jsonl(synthetic_niah_path):
            texts.append(sample.context)

    if not texts:
        raise ValueError(
            "No candidate texts found -- pass at least one of "
            "evidence_cache_dir (populated by runner/run_arm.py) or "
            "synthetic_niah_path (eval_data/gen_synthetic_niah.py's output)."
        )
    return texts


def bucket_candidates(texts: list[str], tokenizer, bin_sizes: list[int]) -> dict[int, list[int]]:
    """For each bin size N, finds the FIRST pooled text with at least N
    tokens and truncates its token ids to exactly N. Truncation (not
    padding) keeps every bin's content a genuine, if shortened, excerpt of
    something real rather than injecting artificial filler that would
    change what's actually being measured (SpecPrefill's chunk-based
    scoring is content-sensitive, so padding with repeated/synthetic
    tokens could bias the comparison). Bins with no long-enough candidate
    anywhere in the pool are silently omitted here -- the caller (`main`)
    reports which bins were actually covered, since a sparse pool covering
    only some bins is still useful for the ones it does cover.
    """
    tokenized = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    bins: dict[int, list[int]] = {}
    for n in sorted(bin_sizes):
        for token_ids in tokenized:
            if len(token_ids) >= n:
                bins[n] = token_ids[:n]
                break
    return bins


# ---------------------------------------------------------------------------
# Crossover computation (pure Python).
# ---------------------------------------------------------------------------


@dataclass
class BinTiming:
    n_tokens: int
    t_plain_s: float
    t_specprefill_s: float

    @property
    def specprefill_faster(self) -> bool:
        return self.t_specprefill_s < self.t_plain_s


def compute_crossover(bin_timings: list[BinTiming]) -> int | None:
    """N_min = the smallest measured N at or above which SpecPrefill is
    faster for that bin AND every larger bin measured. Requiring it to hold
    for every larger bin (not just the first N where it happens to win
    once) is a more conservative, defensible estimate -- a single crossing
    could be measurement noise, especially on a small sweep.

    Returns None if SpecPrefill never wins across the measured range --
    that's a real, actionable result (the gate should never fire at these
    sizes; Arm C would degenerate to always-skip), not an error, so it's
    surfaced to the caller rather than raised.
    """
    ordered = sorted(bin_timings, key=lambda b: b.n_tokens)
    for i, candidate in enumerate(ordered):
        if all(b.specprefill_faster for b in ordered[i:]):
            return candidate.n_tokens
    return None


def write_n_min_config(n_min: int, bin_timings: list[BinTiming], path: Path = DEFAULT_N_MIN_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    curve = [
        {"n_tokens": b.n_tokens, "t_plain_s": b.t_plain_s, "t_specprefill_s": b.t_specprefill_s}
        for b in sorted(bin_timings, key=lambda x: x.n_tokens)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_min": n_min, "curve": curve}, f, indent=2)
    print(f"[sweep_n_min] wrote N_min={n_min} (from {len(curve)} bin(s)) -> {path}")


# ---------------------------------------------------------------------------
# GPU-only sweep orchestration (vllm imports deferred to inside functions).
# ---------------------------------------------------------------------------


def run_sweep(
    bins: dict[int, list[int]],
    target_model_path: str,
    speculator_model_path: str,
    spec_config_path: Path,
    *,
    max_tokens: int = 32,
) -> list[BinTiming]:
    """Feeds each bin's fixed-size token sequence through both the plain
    and SpecPrefill target engines -- built and torn down sequentially,
    never two engines live at once (IMPLEMENTATION_PLAN.md decision 3,
    same constraint `runner/run_arm.py`'s Arm C observes). A short
    `max_tokens` generation cap is deliberate: this sweep is measuring
    prefill/TTFT-dominated cost (where SpecPrefill's savings actually
    apply), not full generation length, so there's no reason to pay for
    long completions here.

    One request per bin, run individually (not batched together) -- keeps
    each bin's own wall-clock time cleanly attributable to that bin's
    prompt size, rather than mixed into a shared batch's aggregate timing.
    """
    import time

    from target_stage.vllm_offline_engine import (
        TargetQuery,
        answer_batch,
        build_plain_target_engine,
        build_specprefill_target_engine,
        load_spec_config,
        teardown_engine,
    )

    def build_queries(tok) -> dict[int, TargetQuery]:
        # Re-decoding each bin's exact token ids under THIS engine's own
        # tokenizer -- both engines load the same target checkpoint, so
        # this should be the identical tokenizer both times, but going
        # through the engine's own instance (rather than assuming re-use)
        # avoids any hidden state/config mismatch.
        return {
            n: TargetQuery(
                sample_id=f"nmin-{n}",
                question="Summarize the key points.",
                excerpts=[{"text": tok.decode(token_ids), "loc_hint": None}],
            )
            for n, token_ids in bins.items()
        }

    print(f"[sweep_n_min] building plain engine for {len(bins)} bin(s) ...")
    plain_engine = build_plain_target_engine(target_model_path, max_tokens=max_tokens)
    plain_queries = build_queries(plain_engine.tokenizer)
    plain_times: dict[int, float] = {}
    for n, query in plain_queries.items():
        t0 = time.perf_counter()
        answer_batch(plain_engine, [query])
        plain_times[n] = time.perf_counter() - t0
        print(f"[sweep_n_min]   plain N={n}: {plain_times[n]:.3f}s")
    teardown_engine(plain_engine)

    print(f"[sweep_n_min] building SpecPrefill engine for {len(bins)} bin(s) ...")
    spec_config = load_spec_config(spec_config_path)
    spec_engine = build_specprefill_target_engine(
        target_model_path, speculator_model_path, spec_config, max_tokens=max_tokens
    )
    spec_queries = build_queries(spec_engine.tokenizer)
    spec_times: dict[int, float] = {}
    for n, query in spec_queries.items():
        t0 = time.perf_counter()
        answer_batch(spec_engine, [query])
        spec_times[n] = time.perf_counter() - t0
        print(f"[sweep_n_min]   specprefill N={n}: {spec_times[n]:.3f}s")
    teardown_engine(spec_engine)

    return [BinTiming(n_tokens=n, t_plain_s=plain_times[n], t_specprefill_s=spec_times[n]) for n in bins]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-cache", type=Path, default=THIS_DIR / "results" / "evidence_cache")
    parser.add_argument("--synthetic-niah", type=Path, default=THIS_DIR / "eval_data" / "synthetic_niah_samples.jsonl")
    parser.add_argument("--bin-sizes", default=",".join(str(b) for b in DEFAULT_BIN_SIZES))
    parser.add_argument("--target-model", default=os.environ.get("LLAMA31_8B_MODEL_PATH"))
    parser.add_argument("--speculator-model", default=os.environ.get("LLAMA32_1B_MODEL_PATH"))
    parser.add_argument("--spec-config", type=Path, default=DEFAULT_SPEC_CONFIG_PATH)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_N_MIN_PATH)
    args = parser.parse_args()

    if not args.target_model or not args.speculator_model:
        parser.error("--target-model and --speculator-model are required (or set $LLAMA31_8B_MODEL_PATH / $LLAMA32_1B_MODEL_PATH)")

    bin_sizes = [int(b) for b in args.bin_sizes.split(",") if b.strip()]

    texts = pool_candidate_texts(args.evidence_cache, args.synthetic_niah)
    print(f"[sweep_n_min] pooled {len(texts)} candidate text(s).")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    bins = bucket_candidates(texts, tok, bin_sizes)

    missing = [n for n in bin_sizes if n not in bins]
    if missing:
        print(
            f"[sweep_n_min] WARNING: no candidate long enough for bin(s) {missing} -- "
            f"pool's longest text is too short. Sweeping only {sorted(bins)}."
        )
    if not bins:
        raise ValueError("No bins covered by the candidate pool -- nothing to sweep.")

    bin_timings = run_sweep(bins, args.target_model, args.speculator_model, args.spec_config, max_tokens=args.max_tokens)

    n_min = compute_crossover(bin_timings)
    if n_min is None:
        print(
            "[sweep_n_min] WARNING: SpecPrefill was never faster than the plain "
            "engine across the measured range -- Arm C's gate should probably "
            "not fire at these sizes. Writing n_min = the largest measured bin "
            "+ 1 as a conservative 'never compress' placeholder; re-sweep with "
            "larger bin sizes before trusting this."
        )
        n_min = max(b.n_tokens for b in bin_timings) + 1

    write_n_min_config(n_min, bin_timings, args.output)


if __name__ == "__main__":
    main()
