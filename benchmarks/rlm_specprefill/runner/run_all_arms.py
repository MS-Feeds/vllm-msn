#!/usr/bin/env python3
"""Sequences Arm A -> Arm B -> Arm C over the same dataset.

Each `run_arm()` call builds, uses, and tears down its own engine(s) before
returning (see target_stage/vllm_offline_engine.py's build_*_target_engine
/ teardown_engine, and run_arm.py's Arm C branch, which already tears down
the plain engine before building the SpecPrefill one). Calling them in
sequence within one process is what keeps IMPLEMENTATION_PLAN.md decision
3's guarantee -- at most one vLLM `LLM` instance ever live -- true across
the WHOLE sweep, not just within a single arm.

Arm A always runs first regardless of `--arms` ordering: it's the pass that
populates the evidence cache every other arm reads from (arms B/C's own
evidence-collection calls cache-hit against whatever A already ran, per
rlm_stage/evidence_cache.py's confound-control guarantee). Silently
reordering `--arms` avoids a user accidentally running B before any
evidence exists to replay -- B/C would still work in that case too (their
own collect_evidence_for_dataset call would just run RLM itself on a miss),
but forcing A first keeps the *reason* each run of RLM happened obvious
across a sweep, rather than depending on invocation order.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))

from runner.run_arm import DEFAULT_RESULTS_DIR, VALID_ARMS, run_arm  # noqa: E402


def run_all_arms(
    dataset_path: Path,
    *,
    arms: list[str] = list(VALID_ARMS),
    results_dir: Path = DEFAULT_RESULTS_DIR,
    max_samples: int | None = None,
    dry_run: bool = False,
    target_model_path: str | None = None,
    speculator_model_path: str | None = None,
    spec_config_path: Path | None = None,
    n_min: int | None = None,
    target_tensor_parallel_size: int = 1,
    target_enable_expert_parallel: bool = False,
    spec_prefill_dir_env: str = "SPEC_PREFILL_LLAMA_DIR",
) -> None:
    ordered_arms = sorted(set(arms), key=lambda a: VALID_ARMS.index(a))  # A first, see module docstring

    for arm in ordered_arms:
        print(f"\n{'=' * 70}\nrun_all_arms: starting Arm {arm}\n{'=' * 70}")
        run_arm(
            arm,
            dataset_path,
            results_dir=results_dir,
            max_samples=max_samples,
            dry_run=dry_run,
            target_model_path=target_model_path,
            speculator_model_path=speculator_model_path,
            spec_config_path=spec_config_path,
            n_min=n_min,
            target_tensor_parallel_size=target_tensor_parallel_size,
            target_enable_expert_parallel=target_enable_expert_parallel,
            spec_prefill_dir_env=spec_prefill_dir_env,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(VALID_ARMS), help="Comma-separated subset, e.g. 'A,C'.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-model", default=os.environ.get("LLAMA31_8B_MODEL_PATH"))
    parser.add_argument("--speculator-model", default=os.environ.get("LLAMA32_1B_MODEL_PATH"))
    parser.add_argument("--spec-config", type=Path, default=None)
    parser.add_argument("--n-min", type=int, default=None)
    parser.add_argument(
        "--target-tensor-parallel-size",
        type=int,
        default=int(os.environ.get("TARGET_TENSOR_PARALLEL_SIZE", "1")),
        help=(
            "New for the Qwen3-Coder-480B-A35B target (../spec_prefill_qwen_coder/), "
            "which doesn't fit on one GPU. Falls back to 1 (preserves the original "
            "Llama-8B single-GPU behavior unchanged) unless $TARGET_TENSOR_PARALLEL_SIZE "
            "is set -- .env_exports_qwen_coder.sh sets it to 4 (see that file's own "
            "comment / EXPERIMENT_PLAN.md's 'Resource requirements' for why 4, not 8, "
            "is the current starting point: TP=8 leaves no GPU for the speculator)."
        ),
    )
    parser.add_argument(
        "--spec-prefill-dir-env",
        default="SPEC_PREFILL_LLAMA_DIR",
        help=(
            "Env var (set by the matching .env_exports*.sh) pointing at the "
            "SpecPrefill port directory whose vllm_patch/ to import -- default "
            "selects ../spec_prefill_llama/. Pass 'SPEC_PREFILL_QWEN_CODER_DIR' "
            "for the Qwen3-Coder-480B/30B pairing (../spec_prefill_qwen_coder/)."
        ),
    )
    parser.add_argument(
        "--target-enable-expert-parallel",
        action="store_true",
        help=(
            "Required by at least some quantized MoE target checkpoints at "
            "--target-tensor-parallel-size > 1 -- e.g. QuantTrio's AWQ quant "
            "of Qwen3-Coder-480B-A35B-Instruct documents this as REQUIRED at "
            "tensor-parallel-size 8. Default off preserves prior behavior."
        ),
    )
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    invalid = [a for a in arms if a not in VALID_ARMS]
    if invalid:
        parser.error(f"Unknown arm(s): {invalid} (must be a subset of {VALID_ARMS})")

    run_all_arms(
        args.dataset,
        arms=arms,
        results_dir=args.results_dir,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
        target_model_path=args.target_model,
        speculator_model_path=args.speculator_model,
        spec_config_path=args.spec_config,
        n_min=args.n_min,
        target_tensor_parallel_size=args.target_tensor_parallel_size,
        target_enable_expert_parallel=args.target_enable_expert_parallel,
        spec_prefill_dir_env=args.spec_prefill_dir_env,
    )


if __name__ == "__main__":
    main()
