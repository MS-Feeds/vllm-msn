#!/usr/bin/env python3
"""CLI: run one arm (A/B/C) of the ablation over a dataset.

Arms A/B/C all consume identical cached RLM evidence per query (see
rlm_stage/evidence_cache.py) -- they differ only in what happens between
that evidence and the target model:
  A: plain target, no SpecPrefill
  B: SpecPrefill always-on
  C: SpecPrefill gated (N > N_min, target_stage/gate.py)

Evidence collection (RLM + cache) needs only the Anthropic API when
`root_backend="anthropic"` (the default) -- runnable on any machine with
network access, no GPU. With `root_backend="vllm"` (self-hosted root, see
rlm_stage/root_vllm_server.py), evidence collection instead needs a real
`vllm serve` process already up and healthy -- this function assumes the
caller (below) already arranged that, it does not start one itself. The
target-call stage needs a self-hosted vLLM engine of its own (a SEPARATE,
non-overlapping serving instance even when it's the same checkpoint as
root -- see root_vllm_server.py's module docstring for why) -- GPU-only,
deferred imports (target_stage/vllm_offline_engine.py's own convention) so
this module stays importable without vllm installed.

`--dry-run` collects/caches evidence for every sample and stops there,
never touching the target-call stage -- a genuinely useful workflow split,
not just a testing convenience: evidence can be pre-warmed on a machine
with Anthropic API access but no GPU (root_backend="anthropic" case), then
the actual arm sweep run later on a GPU node that may have no internet
access at all. (For root_backend="vllm", --dry-run still needs a GPU and a
running root server -- see run_arm()'s own docstring below.)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from dataclasses import dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))

from eval_data.schema import EvalSample, read_jsonl  # noqa: E402
from rlm_stage import evidence_cache  # noqa: E402
from rlm_stage.evidence_rlm import EvidenceResult, load_guardrails, run_evidence_extraction  # noqa: E402
from rlm_stage.root_vllm_server import DEFAULT_ROOT_MAX_MODEL_LEN as _DEFAULT_ROOT_MAX_MODEL_LEN  # noqa: E402
from rlm_stage.timing_decomposition import decompose_trajectory  # noqa: E402

# Confirmed real gap (2026-08-04): .env_exports.sh/.env_exports_qwen_coder.sh
# both export RLM_SPECPREFILL_RESULTS_DIR (the latter specifically
# namespaced under results/qwen_coder/ so a Qwen-Coder sweep can't clobber
# an existing Llama arm's results) but neither this module nor any other
# script ever actually read it -- purely vestigial/documentation-only
# before this fix, so results silently landed in plain results/ regardless
# of which pairing's env was sourced. Now honored as the default, still
# overridable via --results-dir.
DEFAULT_RESULTS_DIR = Path(os.environ.get("RLM_SPECPREFILL_RESULTS_DIR", str(THIS_DIR / "results")))
DEFAULT_SPEC_CONFIG_PATH = THIS_DIR / "configs" / "spec_config_always_on.yaml"
VALID_ARMS = ("A", "B", "C")

# Confirmed real-hardware bug (2026-08-05): RLM's own `max_timeout` guardrail
# (guardrails.yaml) only checks wall-clock elapsed *between* root-loop
# iterations (rlm/core/rlm.py's `_check_timeout`) -- it does NOT protect
# against a single iteration hanging forever, e.g. model-generated REPL code
# with an infinite loop (confirmed via py-spy: a real sample stuck for over
# an hour inside `exec(code, ...)` in rlm/environments/local_repl.py,
# `max_timeout` never even got a chance to fire since that iteration never
# returned). A per-sample hard wall-clock timeout is the backstop --
# `_HARD_TIMEOUT_BUFFER_S` added on top of whatever `max_timeout` a given
# run's guardrails.yaml specifies, so RLM's own (working, just insufficient
# alone) soft check gets first chance to produce a clean TimeoutExceededError
# before this cruder hard kill ever needs to fire.
_HARD_TIMEOUT_BUFFER_S = 300.0


@dataclass
class EvidenceCollectionResult:
    sample: EvalSample
    evidence_result: EvidenceResult
    was_cache_hit: bool


def _evidence_extraction_subprocess_target(
    result_queue,
    sample_id: str,
    question: str,
    context: str,
    kwargs: dict,
) -> None:
    """Module-level (NOT a nested closure) so multiprocessing's fork start
    method (Linux default, used explicitly below) can hand this off to the
    child cleanly. Deliberately calls the bare name `run_evidence_extraction`
    (resolved from this module's own namespace at call time, not a captured
    reference) so a test's `monkeypatch.setattr(run_arm_module,
    "run_evidence_extraction", stub)` -- done in the PARENT before forking --
    is still in effect in the forked child too (fork copies the parent's
    already-patched module state at the moment `Process.start()` runs).

    `kwargs` is forwarded to `run_evidence_extraction` as-is (built by the
    caller, e.g. `collect_evidence_for_dataset`'s `run_fn`, using the SAME
    conditional-inclusion pattern the pre-existing direct call used) rather
    than accepting fixed named params here -- several of
    `run_evidence_extraction`'s keyword args (e.g. `model_name`) default to
    a non-`None` sentinel (`DEFAULT_MODEL_NAME`), so explicitly passing
    `model_name=None` here when the caller didn't set one would incorrectly
    override that default instead of inheriting it."""
    try:
        result = run_evidence_extraction(sample_id, question, context, **kwargs)
        result_queue.put(("ok", result))
    except Exception as e:
        result_queue.put(("error", f"{type(e).__name__}: {e}"))


def run_evidence_extraction_with_hard_timeout(
    sample_id: str,
    question: str,
    context: str,
    *,
    timeout_s: float,
    **kwargs,
) -> EvidenceResult:
    """Runs `run_evidence_extraction` in a forked subprocess with a REAL
    wall-clock timeout that force-kills the subprocess if exceeded --
    confirmed necessary (see `_HARD_TIMEOUT_BUFFER_S`'s own comment) because
    RLM's own `max_timeout` guardrail can't interrupt a single stuck
    iteration (e.g. model-generated REPL code with an infinite loop). A
    thread-based timeout can't reliably solve this either -- Python can't
    forcibly kill a thread stuck in a CPU-bound `exec()`, only a whole
    process can be killed reliably (`Process.terminate()`/`.kill()`).

    Uses `multiprocessing.get_context("fork")` explicitly (Linux's own
    default, but pinned rather than left implicit) -- fork, not spawn,
    is what makes monkeypatch-based testing of this function work (see
    `_evidence_extraction_subprocess_target`'s docstring) and avoids the
    overhead of re-importing this whole module in a fresh interpreter per
    sample. This project's whole scope is GPU-cluster/Linux already, so a
    POSIX-only mechanism isn't a new platform constraint.

    `**kwargs` (e.g. `guardrails`, `root_backend`, optionally `model_name`/
    `root_base_url`) is forwarded verbatim to `run_evidence_extraction`
    inside the subprocess -- see `_evidence_extraction_subprocess_target`'s
    docstring for why this isn't a set of fixed named params.
    """
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_evidence_extraction_subprocess_target,
        args=(result_queue, sample_id, question, context, kwargs),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        print(
            f"[run_arm]   sample id={sample_id!r} exceeded the {timeout_s}s "
            f"hard timeout -- force-killing its evidence-extraction subprocess.",
            flush=True,
        )
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise TimeoutError(
            f"evidence extraction for sample {sample_id!r} exceeded the "
            f"{timeout_s}s hard wall-clock timeout (RLM's own max_timeout "
            f"guardrail didn't catch this -- see run_arm.py's "
            f"_HARD_TIMEOUT_BUFFER_S comment for why) and was force-killed."
        )

    if result_queue.empty():
        raise RuntimeError(
            f"evidence-extraction subprocess for sample {sample_id!r} exited "
            f"(code={proc.exitcode}) without producing a result -- likely crashed "
            f"(e.g. OOM) rather than raising a normal Python exception."
        )
    status, payload = result_queue.get()
    if status == "error":
        raise RuntimeError(payload)
    return payload


def collect_evidence_for_dataset(
    samples: list[EvalSample],
    *,
    guardrails: dict | None = None,
    cache_dir: Path = evidence_cache.DEFAULT_CACHE_DIR,
    root_backend: str = "anthropic",
    model_name: str | None = None,
    root_base_url: str | None = None,
    evidence_extraction_timeout_s: float | None = None,
) -> list[EvidenceCollectionResult]:
    """The RLM-evidence half of run_arm -- runnable on any machine with
    Anthropic API access, no vLLM/GPU needed, when `root_backend="anthropic"`
    (the default). With `root_backend="vllm"`, this instead needs a real
    `vllm serve` process already up and healthy at `root_base_url` -- see
    run_arm()'s own docstring for where that gets started/stopped; this
    function itself has no server-lifecycle awareness, it just calls
    run_evidence_extraction with whichever backend it's told to use.
    Always goes through `evidence_cache.get_or_run` (never calls
    `run_evidence_extraction` directly), so calling this repeatedly across
    arms/runs never re-invokes RLM for a query that's already cached
    (IMPLEMENTATION_PLAN.md decision 4 -- the confound-control guarantee).

    `evidence_extraction_timeout_s`: hard per-sample wall-clock timeout
    (see `run_evidence_extraction_with_hard_timeout`'s own docstring for why
    this backstop exists on top of RLM's own `max_timeout` guardrail).
    Defaults to `guardrails["max_timeout"] + _HARD_TIMEOUT_BUFFER_S` when not
    given explicitly, so a caller only needs to override this if they want a
    tighter/looser buffer than the default.
    """
    guardrails = guardrails if guardrails is not None else load_guardrails()
    resolved_timeout_s = (
        evidence_extraction_timeout_s
        if evidence_extraction_timeout_s is not None
        else guardrails.get("max_timeout", 900.0) + _HARD_TIMEOUT_BUFFER_S
    )

    def run_fn(sample: EvalSample) -> EvidenceResult:
        kwargs: dict = {"guardrails": guardrails, "root_backend": root_backend}
        if model_name:
            kwargs["model_name"] = model_name
        if root_base_url:
            kwargs["root_base_url"] = root_base_url
        return run_evidence_extraction_with_hard_timeout(
            sample.id, sample.question, sample.context, timeout_s=resolved_timeout_s, **kwargs
        )

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


def _all_samples_cached(
    samples: list[EvalSample],
    *,
    guardrails: dict | None,
    cache_dir: Path,
) -> bool:
    """Cheap, GPU/root-model-free pre-check: does every sample already have
    a cache entry? Used to skip starting the root vLLM server entirely
    when it would only ever answer cache hits -- e.g. run_all_arms.py's
    Arm B/C calls, run after Arm A already populated the cache for the
    same dataset, would otherwise spin up (and immediately tear down) a
    480B-scale server for nothing, potentially costing many real minutes
    per arm. Uses evidence_cache.load_cached directly (not
    collect_evidence_for_dataset/get_or_run) specifically so this never
    touches run_fn -- must be true GPU/root-free."""
    guardrails = guardrails if guardrails is not None else load_guardrails()
    return all(
        evidence_cache.load_cached(
            evidence_cache.compute_cache_key(sample, guardrails=guardrails), cache_dir
        )
        is not None
        for sample in samples
    )


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
    root_max_model_len: int | None = _DEFAULT_ROOT_MAX_MODEL_LEN,
    root_server_startup_timeout_s: float = 1800.0,
    evidence_extraction_timeout_s: float | None = None,
) -> None:
    """`root_backend='vllm'` (self-hosted root, e.g. Qwen3-Coder-480B-A35B
    serving as both root and SpecPrefill target -- see rlm_stage/
    evidence_rlm.py and rlm_stage/root_vllm_server.py) starts a `vllm serve`
    subprocess BEFORE evidence collection, waits for it to become healthy,
    then tears it down again before this function returns to its caller
    (or before the target-model stage below, on the non-dry-run path) --
    root-serving and target-answering are separate, non-overlapping vLLM
    processes even when they're the same checkpoint (see
    root_vllm_server.py's module docstring for why), so the server must be
    fully torn down before target_stage tries to claim the same GPUs.
    `root_model_path` defaults to `target_model_path` (the common case:
    the same checkpoint serves both roles) if not given separately.
    `root_enable_expert_parallel` defaults to `target_enable_expert_parallel`
    (same reasoning as `root_tensor_parallel_size` above: root and target
    are typically the same checkpoint, so whatever MoE expert-splitting
    requirement applies to one at a given TP degree applies to the other)
    -- pass it explicitly only to make root's setting diverge from the
    target's.
    `root_max_model_len` defaults to `root_vllm_server.DEFAULT_ROOT_MAX_MODEL_LEN`
    (131072, NOT the checkpoint's native max) -- confirmed real-hardware
    requirement: leaving it unset makes vLLM reserve KV cache for the full
    native context, which can exceed what's actually available at a given
    tensor_parallel_size (see that constant's own docstring for the exact
    failure and how to size a real override for your node).
    `evidence_extraction_timeout_s` is passed straight through to
    `collect_evidence_for_dataset` -- see its own docstring for the default
    (guardrails' `max_timeout` + `_HARD_TIMEOUT_BUFFER_S`) when left unset.
    """
    if arm not in VALID_ARMS:
        raise ValueError(f"Unknown arm {arm!r}, must be one of {VALID_ARMS}")

    samples = read_jsonl(dataset_path)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError(f"No samples loaded from {dataset_path}")

    arm_dir = results_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    root_server_proc = None
    resolved_root_model_name = root_model_name
    if root_backend == "vllm" and _all_samples_cached(
        samples, guardrails=None, cache_dir=results_dir / "evidence_cache"
    ):
        print(
            "[run_arm] all evidence already cached -- skipping root vLLM "
            "server startup entirely (it would only ever answer cache hits)."
        )
    elif root_backend == "vllm":
        from rlm_stage.root_vllm_server import (
            start_root_vllm_server,
            stop_root_vllm_server,
            wait_for_server_healthy,
        )

        resolved_root_model_path = root_model_path or target_model_path
        if not resolved_root_model_path:
            raise ValueError(
                "root_model_path (or target_model_path, since they're "
                "typically the same checkpoint) is required when "
                "root_backend='vllm'."
            )
        resolved_root_tp = (
            root_tensor_parallel_size
            if root_tensor_parallel_size is not None
            else target_tensor_parallel_size
        )
        resolved_root_ep = (
            root_enable_expert_parallel
            if root_enable_expert_parallel is not None
            else target_enable_expert_parallel
        )
        if not resolved_root_model_name:
            resolved_root_model_name = Path(resolved_root_model_path).name

        print(
            f"[run_arm] root_backend='vllm': starting vllm serve for "
            f"{resolved_root_model_path} on port {root_port} "
            f"(tensor_parallel_size={resolved_root_tp}, "
            f"enable_expert_parallel={resolved_root_ep}, "
            f"max_model_len={root_max_model_len}) ..."
        )
        root_server_proc = start_root_vllm_server(
            resolved_root_model_path,
            port=root_port,
            tensor_parallel_size=resolved_root_tp,
            enable_expert_parallel=resolved_root_ep,
            served_model_name=resolved_root_model_name,
            max_model_len=root_max_model_len,
            log_path=results_dir / "root_vllm_server.log",
        )
        try:
            wait_for_server_healthy(
                root_base_url, process=root_server_proc, timeout_s=root_server_startup_timeout_s
            )
            print("[run_arm] root vLLM server healthy.")
        except Exception:
            stop_root_vllm_server(root_server_proc)
            raise

    try:
        print(f"[run_arm] Arm {arm}: collecting evidence for {len(samples)} sample(s) ...")
        evidence_results = collect_evidence_for_dataset(
            samples,
            cache_dir=results_dir / "evidence_cache",
            root_backend=root_backend,
            model_name=resolved_root_model_name,
            root_base_url=root_base_url if root_backend == "vllm" else None,
            evidence_extraction_timeout_s=evidence_extraction_timeout_s,
        )
    finally:
        if root_server_proc is not None:
            print("[run_arm] stopping root vLLM server to free GPUs for the target stage ...")
            stop_root_vllm_server(root_server_proc)

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
        engine = build_plain_target_engine(
            target_model_path,
            tensor_parallel_size=target_tensor_parallel_size,
            enable_expert_parallel=target_enable_expert_parallel,
        )
        answers = answer_batch(engine, queries)
        teardown_engine(engine)

    elif arm == "B":
        if not speculator_model_path:
            raise ValueError("speculator_model_path is required for Arm B (e.g. $LLAMA32_1B_MODEL_PATH).")
        spec_config = load_spec_config(
            spec_config_path or DEFAULT_SPEC_CONFIG_PATH, spec_prefill_dir_env=spec_prefill_dir_env
        )
        engine = build_specprefill_target_engine(
            target_model_path,
            speculator_model_path,
            spec_config,
            tensor_parallel_size=target_tensor_parallel_size,
            enable_expert_parallel=target_enable_expert_parallel,
            spec_prefill_dir_env=spec_prefill_dir_env,
        )
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
        plain_engine = build_plain_target_engine(
            target_model_path,
            tensor_parallel_size=target_tensor_parallel_size,
            enable_expert_parallel=target_enable_expert_parallel,
        )
        skip_bucket, compress_bucket = route_queries(queries, plain_engine.tokenizer, resolved_n_min)
        print(
            f"[run_arm] Arm C gate (N_min={resolved_n_min}): "
            f"{len(skip_bucket)} skip (plain), {len(compress_bucket)} compress (SpecPrefill)."
        )

        skip_answers = answer_batch(plain_engine, skip_bucket) if skip_bucket else []
        teardown_engine(plain_engine)  # never two engines live at once (decision 3)

        compress_answers = []
        if compress_bucket:
            spec_config = load_spec_config(
                spec_config_path or DEFAULT_SPEC_CONFIG_PATH, spec_prefill_dir_env=spec_prefill_dir_env
            )
            spec_engine = build_specprefill_target_engine(
                target_model_path,
                speculator_model_path,
                spec_config,
                tensor_parallel_size=target_tensor_parallel_size,
                enable_expert_parallel=target_enable_expert_parallel,
                spec_prefill_dir_env=spec_prefill_dir_env,
            )
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
            "tensor-parallel-size 8, or expert tensors don't split evenly "
            "across ranks. Falls back to False unless $TARGET_ENABLE_EXPERT_PARALLEL=1 "
            "is set -- .env_exports.sh sets this alongside TARGET_TENSOR_PARALLEL_SIZE=8 "
            "(both flipped together, since the requirement is specifically "
            "documented at that TP degree). Pass --no-target-enable-expert-parallel "
            "to force it off even if the env var is set."
        ),
    )
    parser.add_argument(
        "--root-backend",
        choices=["anthropic", "vllm"],
        default=os.environ.get("ROOT_BACKEND", "anthropic"),
        help=(
            "RLM's root model. 'anthropic' (default, or whatever "
            "$ROOT_BACKEND is set to) is the original hosted-Claude "
            "design. 'vllm' self-hosts the root instead (e.g. the same "
            "Qwen3-Coder-480B-A35B checkpoint serving as both root and "
            "SpecPrefill target, per the paper's default same-model "
            "recursion) -- starts a real vllm serve subprocess for the "
            "evidence-collection stage, see rlm_stage/root_vllm_server.py."
        ),
    )
    parser.add_argument(
        "--root-model-path",
        default=os.environ.get("ROOT_MODEL_PATH"),
        help="Checkpoint to serve when --root-backend=vllm. Defaults to --target-model (the common case: same checkpoint serves both roles).",
    )
    parser.add_argument(
        "--root-model-name",
        default=os.environ.get("ROOT_MODEL_NAME"),
        help=(
            "Model name for RLM's completion calls. For --root-backend=vllm, "
            "must match what the server is launched with (its "
            "--served-model-name) -- defaults to --root-model-path's "
            "basename if not given. Also usable to override the Claude "
            "model name for --root-backend=anthropic."
        ),
    )
    parser.add_argument(
        "--root-base-url",
        default=os.environ.get("ROOT_BASE_URL", "http://127.0.0.1:8000/v1"),
        help=(
            "Only used for --root-backend=vllm -- the vllm serve endpoint "
            "RLM's root-model calls hit. 127.0.0.1, not 'localhost' -- "
            "confirmed real-hardware bug (2026-08-05): the health check "
            "(plain `requests`) and RLM's actual completion calls (openai "
            "SDK/httpx) can resolve 'localhost' to different addresses "
            "(IPv4 vs IPv6, or disagree on NO_PROXY exemption), causing "
            "the health check to pass while every real request fails with "
            "APIConnectionError."
        ),
    )
    parser.add_argument(
        "--root-port",
        type=int,
        default=int(os.environ.get("ROOT_PORT", "8000")),
        help="Only used for --root-backend=vllm -- the port to launch vllm serve on.",
    )
    parser.add_argument(
        "--root-tensor-parallel-size",
        type=int,
        default=None,
        help="Only used for --root-backend=vllm. Defaults to --target-tensor-parallel-size (the common case: same checkpoint, same TP degree).",
    )
    parser.add_argument(
        "--root-enable-expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Only used for --root-backend=vllm. Defaults to "
            "--target-enable-expert-parallel's resolved value (same "
            "reasoning as --root-tensor-parallel-size's own default: root "
            "and target are typically the same checkpoint) -- pass "
            "--root-enable-expert-parallel / --no-root-enable-expert-parallel "
            "explicitly only to make root's setting diverge from the target's."
        ),
    )
    parser.add_argument(
        "--root-max-model-len",
        type=int,
        default=int(os.environ.get("ROOT_MAX_MODEL_LEN", str(_DEFAULT_ROOT_MAX_MODEL_LEN))),
        help=(
            "Only used for --root-backend=vllm. Confirmed real-hardware "
            "requirement (2026-08-04): NOT the checkpoint's native max "
            "(262144) -- leaving this unset makes vLLM reserve KV cache "
            "for a full-native-context request, which can exceed what's "
            "actually available at your --root-tensor-parallel-size "
            "(a real ValueError, not hypothetical). Defaults to 131072 "
            f"(currently {_DEFAULT_ROOT_MAX_MODEL_LEN}) unless "
            "$ROOT_MAX_MODEL_LEN is set -- raise it only after confirming "
            "your node's real KV-cache budget at the chosen TP degree "
            "supports it (a first failed attempt's own error message "
            "reports the achievable ceiling)."
        ),
    )
    parser.add_argument(
        "--root-server-startup-timeout-s",
        type=float,
        default=1800.0,
        help="Only used for --root-backend=vllm -- how long to wait for vllm serve to become healthy before giving up. A large checkpoint can legitimately take a while to load.",
    )
    parser.add_argument(
        "--evidence-extraction-timeout-s",
        type=float,
        default=None,
        help=(
            "Hard per-sample wall-clock timeout that force-kills a stuck "
            "evidence-extraction subprocess. Confirmed real-hardware bug "
            "(2026-08-05, see _HARD_TIMEOUT_BUFFER_S's own comment): RLM's "
            "own guardrails.yaml max_timeout can't interrupt a single stuck "
            "iteration (e.g. model-generated REPL code with an infinite "
            "loop) -- py-spy confirmed a real sample stuck for over an hour "
            "inside exec(). Defaults to guardrails['max_timeout'] + "
            f"{_HARD_TIMEOUT_BUFFER_S}s when not given explicitly."
        ),
    )
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
        target_tensor_parallel_size=args.target_tensor_parallel_size,
        target_enable_expert_parallel=args.target_enable_expert_parallel,
        spec_prefill_dir_env=args.spec_prefill_dir_env,
        root_backend=args.root_backend,
        root_model_path=args.root_model_path,
        root_model_name=args.root_model_name,
        root_base_url=args.root_base_url,
        root_port=args.root_port,
        root_tensor_parallel_size=args.root_tensor_parallel_size,
        root_max_model_len=args.root_max_model_len,
        root_enable_expert_parallel=args.root_enable_expert_parallel,
        root_server_startup_timeout_s=args.root_server_startup_timeout_s,
        evidence_extraction_timeout_s=args.evidence_extraction_timeout_s,
    )


if __name__ == "__main__":
    main()
