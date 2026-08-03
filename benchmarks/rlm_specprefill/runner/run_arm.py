#!/usr/bin/env python3
"""CLI: run one arm (A/B/C) of the ablation over a dataset.

Arms A/B/C all consume identical cached RLM evidence per query (see
rlm_stage/evidence_cache.py) -- they differ only in what happens between
that evidence and the target model:
  A: plain target, no SpecPrefill
  B: SpecPrefill always-on
  C: SpecPrefill gated (N > N_min, target_stage/gate.py)

Evidence collection (RLM + cache) needs only the Anthropic API -- runnable
on any machine with network access, no GPU. The target-call stage needs a
self-hosted vLLM engine -- GPU-only, deferred imports (target_stage/
vllm_offline_engine.py's own convention) so this module stays importable
without vllm installed.

`--dry-run` collects/caches evidence for every sample and stops there,
never importing vllm at all -- a genuinely useful workflow split, not just
a testing convenience: evidence can be pre-warmed on a machine with
Anthropic API access but no GPU (this one), then the actual arm sweep run
later on a GPU node that may have no internet access at all.
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

from eval_data.schema import EvalSample, read_jsonl  # noqa: E402
from rlm_stage import evidence_cache  # noqa: E402
from rlm_stage.evidence_rlm import EvidenceResult, load_guardrails, run_evidence_extraction  # noqa: E402
from rlm_stage.timing_decomposition import decompose_trajectory  # noqa: E402

DEFAULT_RESULTS_DIR = THIS_DIR / "results"
DEFAULT_SPEC_CONFIG_PATH = THIS_DIR / "configs" / "spec_config_always_on.yaml"
VALID_ARMS = ("A", "B", "C")


@dataclass
class EvidenceCollectionResult:
    sample: EvalSample
    evidence_result: EvidenceResult
    was_cache_hit: bool


def collect_evidence_for_dataset(
    samples: list[EvalSample],
    *,
    guardrails: dict | None = None,
    cache_dir: Path = evidence_cache.DEFAULT_CACHE_DIR,
    model_name: str | None = None,
) -> list[EvidenceCollectionResult]:
    """The RLM-evidence half of run_arm -- runnable on any machine with
    Anthropic API access, no vLLM/GPU needed. Always goes through
    `evidence_cache.get_or_run` (never calls `run_evidence_extraction`
    directly), so calling this repeatedly across arms/runs never
    re-invokes RLM for a query that's already cached
    (IMPLEMENTATION_PLAN.md decision 4 -- the confound-control guarantee).
    """
    guardrails = guardrails if guardrails is not None else load_guardrails()

    def run_fn(sample: EvalSample) -> EvidenceResult:
        kwargs: dict = {"guardrails": guardrails}
        if model_name:
            kwargs["model_name"] = model_name
        return run_evidence_extraction(sample.id, sample.question, sample.context, **kwargs)

    results: list[EvidenceCollectionResult] = []
    for i, sample in enumerate(samples):
        print(f"[run_arm] evidence {i + 1}/{len(samples)}: sample id={sample.id!r} ...", flush=True)
        try:
            evidence_result, was_hit = evidence_cache.get_or_run(
                sample, run_fn, guardrails=guardrails, cache_dir=cache_dir
            )
        except Exception as e:
            # Confirmed in practice: RLM's own fallback path when
            # max_iterations is exhausted without answer["ready"]=True
            # (rlm/core/rlm.py's `_default_answer`) sends a synthetic
            # "give me a final answer" message with role="assistant" (an
            # Anthropic prefill-continuation, not an instruction -- almost
            # certainly meant to be role="user") and can get back an
            # empty-content response, which AnthropicClient.completion()
            # (rlm/clients/anthropic.py) doesn't guard against --
            # `response.content[0].text` raises a bare IndexError. This is
            # an upstream `rlm` package issue, not something to patch in a
            # vendored dependency here -- but a single non-converging
            # sample must not take down an entire arm's evidence
            # collection for every OTHER sample, so it's skipped and
            # reported instead (same "skip and report" convention
            # ../spec_prefill_llama/predict_longbench_v2.py uses for
            # over-budget samples).
            print(
                f"[run_arm]   SKIP sample id={sample.id!r}: evidence extraction "
                f"failed ({type(e).__name__}: {e}). Likely RLM not converging "
                f"within max_iterations -- see run_evidence_extraction's "
                f"docstring / IMPLEMENTATION_PLAN.md's load-bearing findings."
            )
            continue
        print(
            f"[run_arm]   {'cache hit' if was_hit else 'ran RLM'}, "
            f"{len(evidence_result.evidence['excerpts'])} excerpt(s), "
            f"parse_error={evidence_result.evidence['parse_error']}"
        )
        results.append(
            EvidenceCollectionResult(sample=sample, evidence_result=evidence_result, was_cache_hit=was_hit)
        )
    return results


def _write_timing_row(f, sample: EvalSample, evidence_result: EvidenceResult) -> None:
    """One JSONL row per sample: the ../rlm_specprefill_ablation_plan.md
    LATENCY MODEL's T_RLM_root/T_REPL_compute/T_RLM_subcalls terms, derived
    from the cached RLMChatCompletion's own trajectory metadata (works
    identically whether this sample was a cache hit or a fresh RLM run --
    the full trajectory is always persisted, see evidence_cache.py)."""
    trajectory = evidence_result.completion.metadata
    row: dict = {"sample_id": sample.id, "wall_time_s": evidence_result.wall_time_s}
    if trajectory:
        timing = decompose_trajectory(trajectory)
        row.update(
            {
                "t_rlm_root": timing.t_rlm_root,
                "t_repl_compute": timing.t_repl_compute,
                "t_rlm_subcalls": timing.t_rlm_subcalls,
                "n_iterations": timing.n_iterations,
                "n_direct_subcalls": timing.n_direct_subcalls,
                "max_realized_depth": timing.max_realized_depth,
                "total_calls": timing.total_calls,
            }
        )
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_arm(
    arm: str,
    dataset_path: Path,
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    max_samples: int | None = None,
    dry_run: bool = False,
    target_model_path: str | None = None,
    speculator_model_path: str | None = None,
    spec_config_path: Path | None = None,
    n_min: int | None = None,
) -> None:
    if arm not in VALID_ARMS:
        raise ValueError(f"Unknown arm {arm!r}, must be one of {VALID_ARMS}")

    samples = read_jsonl(dataset_path)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No samples loaded from {dataset_path}")

    arm_dir = results_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_arm] Arm {arm}: collecting evidence for {len(samples)} sample(s) ...")
    evidence_results = collect_evidence_for_dataset(samples, cache_dir=results_dir / "evidence_cache")

    n_cache_hits = sum(1 for r in evidence_results if r.was_cache_hit)
    print(f"[run_arm] evidence collection done: {n_cache_hits}/{len(evidence_results)} cache hits.")

    with open(arm_dir / "timing.jsonl", "w", encoding="utf-8") as f:
        for r in evidence_results:
            _write_timing_row(f, r.sample, r.evidence_result)

    if dry_run:
        print(
            f"[run_arm] --dry-run: stopping before the target-model stage "
            f"(no vllm imported). Evidence cached at {results_dir / 'evidence_cache'}."
        )
        return

    # --- Target-call stage: GPU-only from here on. ---
    from target_stage.gate import load_n_min
    from target_stage.route_queries import route_queries
    from target_stage.vllm_offline_engine import (
        TargetQuery,
        answer_batch,
        build_plain_target_engine,
        build_specprefill_target_engine,
        load_spec_config,
        teardown_engine,
    )

    queries = [TargetQuery.from_evidence_result(r.sample, r.evidence_result) for r in evidence_results]

    if not target_model_path:
        raise ValueError("target_model_path is required once --dry-run is not set (e.g. $LLAMA31_8B_MODEL_PATH).")

    answers = []
    if arm == "A":
        engine = build_plain_target_engine(target_model_path)
        answers = answer_batch(engine, queries)
        teardown_engine(engine)

    elif arm == "B":
        if not speculator_model_path:
            raise ValueError("speculator_model_path is required for Arm B (e.g. $LLAMA32_1B_MODEL_PATH).")
        spec_config = load_spec_config(spec_config_path or DEFAULT_SPEC_CONFIG_PATH)
        engine = build_specprefill_target_engine(target_model_path, speculator_model_path, spec_config)
        answers = answer_batch(engine, queries)
        teardown_engine(engine)

    elif arm == "C":
        if not speculator_model_path:
            raise ValueError("speculator_model_path is required for Arm C (e.g. $LLAMA32_1B_MODEL_PATH).")
        resolved_n_min = n_min if n_min is not None else load_n_min()

        # Per IMPLEMENTATION_PLAN.md decision 3: the gate decision is made
        # BEFORE either engine loads, using the plain engine's own
        # tokenizer (loading the tokenizer is cheap; it doesn't require
        # constructing the full LLM a second time later).
        plain_engine = build_plain_target_engine(target_model_path)
        skip_bucket, compress_bucket = route_queries(queries, plain_engine.tokenizer, resolved_n_min)
        print(
            f"[run_arm] Arm C gate (N_min={resolved_n_min}): "
            f"{len(skip_bucket)} skip (plain), {len(compress_bucket)} compress (SpecPrefill)."
        )

        skip_answers = answer_batch(plain_engine, skip_bucket) if skip_bucket else []
        teardown_engine(plain_engine)  # never two engines live at once (decision 3)

        compress_answers = []
        if compress_bucket:
            spec_config = load_spec_config(spec_config_path or DEFAULT_SPEC_CONFIG_PATH)
            spec_engine = build_specprefill_target_engine(target_model_path, speculator_model_path, spec_config)
            compress_answers = answer_batch(spec_engine, compress_bucket)
            teardown_engine(spec_engine)

        answers = skip_answers + compress_answers

    predictions_path = arm_dir / "predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for answer in answers:
            row = {
                "id": answer.sample_id,
                "pred": answer.answer_text,
                "finish_reason": answer.finish_reason,
                "n_output_tokens": answer.n_output_tokens,
                "n_prompt_tokens_full": answer.n_prompt_tokens_full,
                "n_prompt_tokens_kept": answer.n_prompt_tokens_kept,
                "keep_rate": answer.keep_rate,
                "ttft_ms": answer.ttft_ms,
                "generation_time_s": answer.generation_time_s,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[run_arm] Arm {arm}: wrote {len(answers)} prediction(s) -> {predictions_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one arm (A/B/C) of the RLM+SpecPrefill ablation.")
    parser.add_argument("--arm", required=True, choices=VALID_ARMS)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect/cache evidence only; skip the target-model stage (no vllm import, no GPU needed).",
    )
    parser.add_argument("--target-model", default=os.environ.get("LLAMA31_8B_MODEL_PATH"))
    parser.add_argument("--speculator-model", default=os.environ.get("LLAMA32_1B_MODEL_PATH"))
    parser.add_argument("--spec-config", type=Path, default=None)
    parser.add_argument("--n-min", type=int, default=None)
    args = parser.parse_args()

    run_arm(
        args.arm,
        args.dataset,
        results_dir=args.results_dir,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
        target_model_path=args.target_model,
        speculator_model_path=args.speculator_model,
        spec_config_path=args.spec_config,
        n_min=args.n_min,
    )


if __name__ == "__main__":
    main()
