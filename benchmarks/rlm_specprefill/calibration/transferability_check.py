#!/usr/bin/env python3
"""Score transferability check: verifies SpecPrefill's keep-rate/quality
relationship holds on RLM's own candidate format, not just the
prompt-formatted long-context benchmark format it was originally validated
against.

Per ../rlm_specprefill_ablation_plan.md's SPECPREFILL GATE section: "same-
family speculator resolves cross-model transfer, but SpecPrefill was
calibrated on prompt-formatted long-context benchmarks, not RLM's
REPL-restructured candidate text (sliced, reordered, concatenated across
noncontiguous regions). Verify the keep-rate/quality relationship holds on
RLM's actual candidate format before trusting N_min; recalibrate if it
doesn't." This is separate from `sweep_n_min.py`'s crossover calibration --
that's about WHEN to compress; this is about whether compression still
PRESERVES QUALITY the way it was validated to, once the input is RLM's
format instead of LongBench v2's.

Compares two curves:
- REFERENCE: `../spec_prefill_llama/`'s own P001-P006 keep-rate sweep,
  read from `grade_longbench_v2.py`'s per-experiment result.json files
  (LongBench-v2-format prompts, multiple-choice accuracy).
- OURS: a fresh sweep over RLM-format candidate text (this project's own
  synthetic NIAH samples, which have well-defined ground truth needle
  values -- LongBench v2's multiple-choice format doesn't transfer
  directly to our open-ended-QA target prompt, see module docstring in
  target_stage/vllm_offline_engine.py for why the target call asks for a
  free-text answer, not a lettered choice), scored by needle recall
  instead of exact-letter accuracy.

Usage (GPU node, after spec_prefill_llama's own P001-P006 sweep + grading
has produced `results/{exp_id}_result.json` files there):
    python3 calibration/transferability_check.py \\
        --reference-dir ../spec_prefill_llama/results \\
        --niah-samples eval_data/synthetic_niah_samples.jsonl \\
        --target-model $LLAMA31_8B_MODEL_PATH --speculator-model $LLAMA32_1B_MODEL_PATH
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))

from eval_data.schema import EvalSample, read_jsonl  # noqa: E402

# Duplicated from ../spec_prefill_llama/predict_longbench_v2.py's
# PRUNE_EXPERIMENTS dict (NOT imported -- importing that script has
# module-level side effects, e.g. `OUT_DIR.mkdir(exist_ok=True)` against
# whatever the importer's CWD happens to be; see
# target_stage/vllm_offline_engine.py's module docstring for the same
# "copy, don't import" reasoning applied there). Keep this in sync if that
# pipeline's experiment matrix changes.
KEEP_PERCENTAGE_BY_EXP_ID = {
    "P001": 1.0,  # baseline, no pruning
    "P002": 0.1,
    "P003": 0.3,
    "P004": 0.5,
    "P005": 0.7,
    "P006": 0.9,
}

DEFAULT_PERCENTAGES = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_DIVERGENCE_THRESHOLD_PCT = 15.0


# ---------------------------------------------------------------------------
# Reference curve loading (pure Python -- no vllm needed).
# ---------------------------------------------------------------------------


def load_reference_curve(result_json_dir: Path) -> list[tuple[float, float]]:
    """Reads `{exp_id}_result.json` files (as produced by
    `grade_longbench_v2.py --output <dir>/{exp_id}_result.json`, one
    invocation per P00X experiment) and returns
    `[(keep_percentage, overall_accuracy_pct), ...]` sorted by
    keep_percentage. Skips any P00X whose result file isn't present rather
    than requiring the full matrix -- a partial reference curve is still
    useful for whichever points it covers."""
    curve: list[tuple[float, float]] = []
    for exp_id, keep_pct in sorted(KEEP_PERCENTAGE_BY_EXP_ID.items(), key=lambda kv: kv[1]):
        path = result_json_dir / f"{exp_id}_result.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        curve.append((keep_pct, result["overall"]))

    if not curve:
        raise ValueError(
            f"No {{exp_id}}_result.json files found in {result_json_dir} -- "
            f"run spec_prefill_llama's predict_longbench_v2.py + "
            f"grade_longbench_v2.py first (see that project's REPRODUCE.md)."
        )
    return curve


# ---------------------------------------------------------------------------
# Our own curve: recall-based, computed over RLM-format (synthetic NIAH)
# candidate text.
# ---------------------------------------------------------------------------


def compute_recall(needle_values: list[str], text: str) -> float:
    """Fraction of ground-truth needle values found verbatim in `text`.
    Well-defined ground truth (exact substrings at known positions) is
    exactly why synthetic NIAH samples -- not LongBench v2's multiple-choice
    format -- are used for OUR curve; see module docstring."""
    if not needle_values:
        return 1.0
    found = sum(1 for value in needle_values if value in text)
    return found / len(needle_values)


def run_rlm_format_sweep(
    niah_samples: list[EvalSample],
    percentages: list[float],
    target_model_path: str,
    speculator_model_path: str,
    *,
    chunk_size: int = 32,  # matches spec_prefill_llama's own CHUNK_SIZE -- see this function's docstring
    look_ahead_cnt: int = 8,
    pool_kernel_size: int = 13,
    max_tokens: int = 64,
    target_tensor_parallel_size: int = 1,
    target_enable_expert_parallel: bool = False,
    spec_prefill_dir_env: str = "SPEC_PREFILL_LLAMA_DIR",
) -> list[tuple[float, float]]:
    """For each keep percentage, builds a fresh SpecPrefill engine (a
    `SpecConfig` is fixed per engine -- IMPLEMENTATION_PLAN.md decision 3 --
    so a multi-percentage sweep necessarily rebuilds the engine per point,
    same as spec_prefill_llama's own P002-P006 sweep does per experiment),
    runs every `niah_samples` through it, and computes mean needle-recall
    across samples. `chunk_size`/`look_ahead_cnt`/`pool_kernel_size` default
    to the SAME values `configs/spec_config_always_on.yaml` and
    `spec_prefill_llama`'s own sweep use, so this curve is comparable to the
    reference one -- only `percentage` varies. NOTE: these chunk_size/
    look_ahead_cnt/pool_kernel_size defaults are Llama-specific (see
    ../configs/spec_config_always_on.yaml's own note that chunk_size=32 was
    set deliberately for Llama, not universal) -- pass the Qwen3-Coder
    port's own values (chunk_size=64, matching
    ../configs/spec_config_always_on_qwen_coder.yaml) explicitly when
    `spec_prefill_dir_env` selects that port, rather than assuming these
    defaults transfer.

    Returns `[(percentage, mean_recall), ...]`.
    """
    from target_stage.vllm_offline_engine import (
        TargetQuery,
        answer_batch,
        build_specprefill_target_engine,
        ensure_spec_prefill_on_path,
        teardown_engine,
    )

    ensure_spec_prefill_on_path(spec_prefill_dir_env)
    from vllm_patch.config import SpecConfig

    queries = [
        TargetQuery(sample_id=s.id, question=s.question, excerpts=[{"text": s.context, "loc_hint": None}])
        for s in niah_samples
    ]
    needle_values_by_id = {
        s.id: [n["value"] for n in s.extra.get("needles", [])] for s in niah_samples
    }

    curve: list[tuple[float, float]] = []
    for pct in percentages:
        spec_config = SpecConfig(
            keep_strategy="percentage",
            keep_kwargs={"chunk": True, "chunk_size": chunk_size, "percentage": pct},
            look_ahead_cnt=look_ahead_cnt,
            pool_kernel_size=pool_kernel_size,
        )
        engine = build_specprefill_target_engine(
            target_model_path,
            speculator_model_path,
            spec_config,
            max_tokens=max_tokens,
            tensor_parallel_size=target_tensor_parallel_size,
            enable_expert_parallel=target_enable_expert_parallel,
            spec_prefill_dir_env=spec_prefill_dir_env,
        )
        answers = answer_batch(engine, queries)
        teardown_engine(engine)

        recalls = [compute_recall(needle_values_by_id.get(a.sample_id, []), a.answer_text) for a in answers]
        mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
        curve.append((pct, mean_recall))
        print(
            f"[transferability_check] percentage={pct}: mean_recall={mean_recall:.3f} "
            f"over {len(recalls)}/{len(queries)} sample(s) answered"
        )

    return curve


# ---------------------------------------------------------------------------
# Comparison (pure Python).
# ---------------------------------------------------------------------------


def compare_curves(
    reference: list[tuple[float, float]],
    ours: list[tuple[float, float]],
    *,
    threshold_pct: float = DEFAULT_DIVERGENCE_THRESHOLD_PCT,
) -> dict:
    """Aligns each of OUR points to the nearest REFERENCE keep-percentage
    and compares quality on a common 0-100 scale (reference reports
    multiple-choice accuracy as a percentage already; ours reports recall
    on [0,1], scaled by 100 here). Flags any point whose delta exceeds
    `threshold_pct` as a transferability concern -- per the ablation doc:
    "recalibrate if it doesn't [transfer]." A missing/empty `ours` or
    `reference` is a caller error, not silently handled here (both curves
    should be non-empty by the time this runs).
    """
    points = []
    for pct, our_recall in ours:
        ref_pct, ref_accuracy = min(reference, key=lambda r: abs(r[0] - pct))
        our_scaled = our_recall * 100
        delta = our_scaled - ref_accuracy
        points.append(
            {
                "percentage": pct,
                "our_recall_pct": our_scaled,
                "reference_accuracy_pct": ref_accuracy,
                "reference_percentage_matched": ref_pct,
                "delta": delta,
                "diverges": abs(delta) > threshold_pct,
            }
        )

    return {
        "threshold_pct": threshold_pct,
        "points": points,
        "any_divergence": any(p["diverges"] for p in points),
    }


def render_report(comparison: dict) -> str:
    lines = [f"Transferability check (divergence threshold: {comparison['threshold_pct']:.1f} pts)", ""]
    lines.append("| Keep % | Our recall % | Reference accuracy % (matched %) | Delta | Diverges? |")
    lines.append("|---:|---:|---:|---:|:---:|")
    for p in comparison["points"]:
        lines.append(
            f"| {p['percentage']:.0%} | {p['our_recall_pct']:.1f} | "
            f"{p['reference_accuracy_pct']:.1f} ({p['reference_percentage_matched']:.0%}) | "
            f"{p['delta']:+.1f} | {'YES' if p['diverges'] else 'no'} |"
        )
    lines.append("")
    if comparison["any_divergence"]:
        lines.append(
            "RESULT: at least one point diverges beyond threshold -- per the ablation "
            "doc, recalibrate N_min (and reconsider whether the keep-rate sweep in "
            "configs/spec_config_always_on.yaml is even appropriate) before trusting "
            "Arms B/C's results on RLM-format evidence."
        )
    else:
        lines.append("RESULT: no point diverges beyond threshold -- the reference sweep's keep-rate/quality relationship appears to transfer.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=THIS_DIR.parent / "spec_prefill_llama" / "results")
    parser.add_argument("--niah-samples", type=Path, default=THIS_DIR / "eval_data" / "synthetic_niah_samples.jsonl")
    parser.add_argument("--percentages", default=",".join(str(p) for p in DEFAULT_PERCENTAGES))
    parser.add_argument("--target-model", default=os.environ.get("LLAMA31_8B_MODEL_PATH"))
    parser.add_argument("--speculator-model", default=os.environ.get("LLAMA32_1B_MODEL_PATH"))
    parser.add_argument("--threshold-pct", type=float, default=DEFAULT_DIVERGENCE_THRESHOLD_PCT)
    # Matches runner/run_arm.py's DEFAULT_RESULTS_DIR fix -- honor
    # $RLM_SPECPREFILL_RESULTS_DIR the same way for output-location
    # consistency across pairings.
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("RLM_SPECPREFILL_RESULTS_DIR", str(THIS_DIR / "results")))
        / "transferability_check.json",
    )
    parser.add_argument(
        "--target-tensor-parallel-size",
        type=int,
        default=int(os.environ.get("TARGET_TENSOR_PARALLEL_SIZE", "1")),
        help="See runner/run_arm.py's flag of the same name (defaults from $TARGET_TENSOR_PARALLEL_SIZE, e.g. 4 for Qwen3-Coder).",
    )
    parser.add_argument(
        "--spec-prefill-dir-env",
        default="SPEC_PREFILL_LLAMA_DIR",
        help="See runner/run_arm.py's flag of the same name (e.g. 'SPEC_PREFILL_QWEN_CODER_DIR').",
    )
    parser.add_argument(
        "--target-enable-expert-parallel",
        action="store_true",
        help="See runner/run_arm.py's flag of the same name -- required by some quantized MoE checkpoints at TP>1.",
    )
    args = parser.parse_args()

    if not args.target_model or not args.speculator_model:
        parser.error("--target-model and --speculator-model are required (or set $LLAMA31_8B_MODEL_PATH / $LLAMA32_1B_MODEL_PATH)")

    reference = load_reference_curve(args.reference_dir)
    print(f"[transferability_check] loaded reference curve: {reference}")

    niah_samples = read_jsonl(args.niah_samples)
    if not niah_samples:
        raise ValueError(f"No samples in {args.niah_samples} -- run eval_data/gen_synthetic_niah.py first.")

    percentages = [float(p) for p in args.percentages.split(",") if p.strip()]
    ours = run_rlm_format_sweep(
        niah_samples,
        percentages,
        args.target_model,
        args.speculator_model,
        target_tensor_parallel_size=args.target_tensor_parallel_size,
        target_enable_expert_parallel=args.target_enable_expert_parallel,
        spec_prefill_dir_env=args.spec_prefill_dir_env,
    )

    comparison = compare_curves(reference, ours, threshold_pct=args.threshold_pct)
    print()
    print(render_report(comparison))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[transferability_check] wrote {args.output}")


if __name__ == "__main__":
    main()
