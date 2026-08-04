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

from rlm_stage.root_vllm_server import DEFAULT_ROOT_MAX_MODEL_LEN  # noqa: E402
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
    root_backend: str = "anthropic",
    root_model_path: str | None = None,
    root_model_name: str | None = None,
    root_base_url: str = "http://127.0.0.1:8000/v1",
    root_port: int = 8000,
    root_tensor_parallel_size: int | None = None,
    root_enable_expert_parallel: bool | None = None,
    root_max_model_len: int | None = DEFAULT_ROOT_MAX_MODEL_LEN,
    root_server_startup_timeout_s: float = 1800.0,
    evidence_extraction_timeout_s: float | None = None,
) -> None:
    """When `root_backend='vllm'`, each `run_arm()` call below independently
    starts/stops its own root vLLM server -- Arm A's call will actually use
    it (cache misses on a fresh dataset), but Arms B/C's calls should be
    all cache hits (per this module's own confound-control guarantee) and
    will skip server startup entirely via run_arm.py's `_all_samples_cached`
    pre-check, not pay for a second/third restart. That pre-check is what
    makes it safe to just pass the same root_* args through unconditionally
    here, rather than hoisting server lifecycle up into this function."""
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
            root_backend=root_backend,
            root_model_path=root_model_path,
            root_model_name=root_model_name,
            root_base_url=root_base_url,
            root_port=root_port,
            root_tensor_parallel_size=root_tensor_parallel_size,
            root_enable_expert_parallel=root_enable_expert_parallel,
            root_max_model_len=root_max_model_len,
            root_server_startup_timeout_s=root_server_startup_timeout_s,
            evidence_extraction_timeout_s=evidence_extraction_timeout_s,
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
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("TARGET_ENABLE_EXPERT_PARALLEL", "0") == "1",
        help=(
            "Required by at least some quantized MoE target checkpoints at "
            "--target-tensor-parallel-size > 1 -- e.g. QuantTrio's AWQ quant "
            "of Qwen3-Coder-480B-A35B-Instruct documents this as REQUIRED at "
            "tensor-parallel-size 8. Falls back to False unless "
            "$TARGET_ENABLE_EXPERT_PARALLEL=1 is set."
        ),
    )
    parser.add_argument(
        "--root-backend",
        choices=["anthropic", "vllm"],
        default=os.environ.get("ROOT_BACKEND", "anthropic"),
        help="See runner/run_arm.py's flag of the same name.",
    )
    parser.add_argument(
        "--root-model-path",
        default=os.environ.get("ROOT_MODEL_PATH"),
        help="See runner/run_arm.py's flag of the same name -- defaults to --target-model.",
    )
    parser.add_argument(
        "--root-model-name",
        default=os.environ.get("ROOT_MODEL_NAME"),
        help="See runner/run_arm.py's flag of the same name.",
    )
    parser.add_argument(
        "--root-base-url",
        default=os.environ.get("ROOT_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="See runner/run_arm.py's flag of the same name.",
    )
    parser.add_argument(
        "--root-port",
        type=int,
        default=int(os.environ.get("ROOT_PORT", "8000")),
        help="See runner/run_arm.py's flag of the same name.",
    )
    parser.add_argument(
        "--root-tensor-parallel-size",
        type=int,
        default=None,
        help="See runner/run_arm.py's flag of the same name -- defaults to --target-tensor-parallel-size.",
    )
    parser.add_argument(
        "--root-enable-expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="See runner/run_arm.py's flag of the same name -- defaults to --target-enable-expert-parallel.",
    )
    parser.add_argument(
        "--root-max-model-len",
        type=int,
        default=int(os.environ.get("ROOT_MAX_MODEL_LEN", str(DEFAULT_ROOT_MAX_MODEL_LEN))),
        help="See runner/run_arm.py's flag of the same name -- NOT the checkpoint's native max, see that flag's help for why.",
    )
    parser.add_argument(
        "--root-server-startup-timeout-s",
        type=float,
        default=1800.0,
        help="See runner/run_arm.py's flag of the same name.",
    )
    parser.add_argument(
        "--evidence-extraction-timeout-s",
        type=float,
        default=None,
        help="See runner/run_arm.py's flag of the same name.",
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
        root_backend=args.root_backend,
        root_model_path=args.root_model_path,
        root_model_name=args.root_model_name,
        root_base_url=args.root_base_url,
        root_port=args.root_port,
        root_tensor_parallel_size=args.root_tensor_parallel_size,
        root_enable_expert_parallel=args.root_enable_expert_parallel,
        root_max_model_len=args.root_max_model_len,
        root_server_startup_timeout_s=args.root_server_startup_timeout_s,
        evidence_extraction_timeout_s=args.evidence_extraction_timeout_s,
    )


if __name__ == "__main__":
    main()
