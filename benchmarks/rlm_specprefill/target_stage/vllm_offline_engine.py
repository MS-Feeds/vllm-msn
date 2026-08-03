"""Drives vLLM's OFFLINE `LLM` class (not `vllm serve`'s HTTP API) to answer
a question from RLM-extracted evidence excerpts -- optionally pruned via
SpecPrefill. Adapted from (not importing) `../../spec_prefill_llama/predict_longbench_v2.py`,
which is SpecPrefill's only validated driver pattern (see
IMPLEMENTATION_PLAN.md decision 2 for why the offline engine was chosen over
building an HTTP shim or testing `worker_cls` under `vllm serve`, both
unvalidated). Every request-handling pattern below that's marked "confirmed
on real hardware" is carried over unchanged from that script; nothing in
this file has been independently re-validated (this machine has no GPU and
`vllm` isn't installed here -- see IMPLEMENTATION_PLAN.md's Verification
section).

Unlike `predict_longbench_v2.py`'s LongBench-v2-specific multiple-choice
prompt (context + question + 4 choices), this module answers an open-ended
question from a list of evidence excerpts (RLM's output, see
`rlm_stage/evidence_rlm.py` / `prompts/evidence_extraction.py`) -- the
PREFIX/CONTEXT/SUFFIX split below (force-keep prefix/suffix, prune only the
excerpts blob) is the same pattern for the same reason: SpecPrefill's
chunk-based scorer has no special protection for a short trailing
instruction/question competing against many prunable chunks (confirmed real
gibberish-output bug in `predict_longbench_v2.py`'s own history), so the
question must never be left to the pruner's own scoring decision.

All `vllm`/`vllm_patch`/`torch`/`transformers` imports are deferred to
inside functions (matching `predict_longbench_v2.py`'s own convention) so
that the dataclasses, prompt-formatting, and budget-sizing helpers below
stay importable and unit-testable on a machine with none of those installed
-- see `gate.py`/`route_queries.py`, which depend on that.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))
from eval_data.schema import EvalSample  # noqa: E402
from rlm_stage.evidence_rlm import EvidenceResult  # noqa: E402

# Starting point, not a profiled value -- our prompts are RLM-compressed
# evidence, which should be far smaller than the raw >131K-token contexts
# they were extracted from (that's the whole point of the compression
# stage), so this default is much smaller than predict_longbench_v2.py's
# own auto-sizing would produce for RAW LongBench v2 contexts. Override via
# resolve_max_num_batched_tokens() once real evidence-size data exists
# (e.g. from calibration/sweep_n_min.py's candidate pool).
DEFAULT_MAX_NUM_BATCHED_TOKENS = 32_768


# ---------------------------------------------------------------------------
# Data shapes (no vllm/torch/transformers dependency -- safe to import and
# construct anywhere).
# ---------------------------------------------------------------------------


@dataclass
class TargetQuery:
    """One target-model call's input: a question plus the evidence excerpts
    RLM gathered for it. Deliberately uses the ORIGINAL sample's question
    (from EvalSample), not `EvidenceResult.evidence["question"]` -- that
    field is RLM's own echo of the question and is `None` whenever
    `parse_evidence_response` fell back to the unstructured-blob path
    (../../prompts/evidence_extraction.py), so it isn't reliable as the
    sole source of truth."""

    sample_id: str
    question: str
    excerpts: list[dict[str, Any]]

    @classmethod
    def from_evidence_result(cls, sample: EvalSample, evidence_result: EvidenceResult) -> "TargetQuery":
        return cls(
            sample_id=sample.id,
            question=sample.question,
            excerpts=evidence_result.evidence.get("excerpts", []),
        )


@dataclass
class TargetAnswer:
    sample_id: str
    answer_text: str
    finish_reason: str
    n_output_tokens: int
    n_prompt_tokens_full: int  # unpruned prompt length (prefix+excerpts+suffix)
    n_prompt_tokens_kept: int | None  # None for the plain (unpruned) engine
    ttft_ms: float | None
    # Wall-clock time from batch submission to this request's own
    # completion (drive_engine_to_completion's finish_time_by_id minus the
    # batch's start time) -- this IS T_target for analysis/aggregate_metrics.py's
    # per-sample `f` computation, not an estimate. See
    # drive_engine_to_completion's docstring for the "batch-relative, not
    # live-server-accurate" caveat.
    generation_time_s: float

    @property
    def keep_rate(self) -> float | None:
        if self.n_prompt_tokens_kept is None or self.n_prompt_tokens_full == 0:
            return None
        return self.n_prompt_tokens_kept / self.n_prompt_tokens_full


@dataclass
class EngineHandle:
    """Wraps a constructed vLLM `LLM` (and, for the SpecPrefill case, the
    speculator proposer) -- opaque `Any` typing throughout since `vllm`
    types aren't importable in every environment this module gets imported
    in (see module docstring)."""

    llm: Any
    tokenizer: Any
    is_pruned: bool
    max_num_batched_tokens: int
    max_tokens: int
    proposer: Any = None
    spec_config: Any = None
    speculator_device: Any = None
    head_dim: int | None = None


@dataclass
class _SubmittedRequest:
    query: TargetQuery
    n_full: int
    n_kept: int | None = None


# ---------------------------------------------------------------------------
# Prompt construction (pure text -- no vllm/torch dependency).
# ---------------------------------------------------------------------------

_PREFIX_TEXT = (
    "You are answering a question using ONLY the excerpts below, gathered "
    "by a separate retrieval step from a much larger document. If the "
    "excerpts don't contain enough information to answer, say so "
    "explicitly rather than guessing.\n\nEXCERPTS:\n\n"
)

_SUFFIX_TEMPLATE = "\n\nQUESTION: {question}\n\nAnswer concisely, based only on the excerpts above."

_CHAT_WRAPPER_PLACEHOLDER = "@@TARGET_CONTENT_PLACEHOLDER@@"


def render_excerpts_text(excerpts: list[dict[str, Any]]) -> str:
    """The prunable CONTEXT piece: each excerpt rendered with its loc_hint
    as a heading. Used both to build the full prompt and (in route_queries.py)
    to measure N for the gate -- the gate is about the compressed EVIDENCE
    size, not the original raw document RLM already reduced away."""
    parts = []
    for i, excerpt in enumerate(excerpts, start=1):
        loc_hint = excerpt.get("loc_hint")
        header = f"[Excerpt {i}" + (f" -- {loc_hint}" if loc_hint else "") + "]"
        parts.append(f"{header}\n{excerpt.get('text', '')}")
    return "\n\n".join(parts)


def chat_wrapper_pieces(tok) -> tuple[str, str]:
    """Returns (before, after): the fixed chat-template text surrounding a
    single user turn's content (e.g. `<|start_header_id|>user<|end_header_id|>`
    / `<|eot_id|><|start_header_id|>assistant<|end_header_id|>` for Llama's
    template), independent of what that content is -- same technique as
    `predict_longbench_v2.py`'s identically-named helper, needed to tokenize
    the always-kept prefix/suffix separately from the prunable excerpts blob
    (see module docstring).

    No `enable_thinking` kwarg here -- that flag only means something to
    Qwen3's own Jinja chat template (the sibling Qwen3 pipeline's version of
    this function needs it to suppress an unwanted reasoning-mode preamble);
    Llama's chat template has no equivalent concept, so passing it would be
    meaningless (per `predict_longbench_v2.py`'s own `render_chat` docstring).
    Must still call `apply_chat_template` with the exact same kwargs as this
    module's `build_prompt_pieces` uses for the full render, whatever they
    are -- the "after" piece computed here would otherwise silently diverge
    from what a full render actually produces.
    """
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": _CHAT_WRAPPER_PLACEHOLDER}],
        add_generation_prompt=True,
        tokenize=False,
    )
    before, after = rendered.split(_CHAT_WRAPPER_PLACEHOLDER)
    return before, after


def build_prompt_pieces(tok, question: str, excerpts: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Returns (prefix_text, context_text, suffix_text) -- the three pieces
    `_submit_plain` concatenates as one prompt, and `_submit_pruned`
    tokenizes separately (only `context_text` -- the excerpts -- is ever
    subject to SpecPrefill's pruning decision)."""
    chat_before, chat_after = chat_wrapper_pieces(tok)
    prefix_text = chat_before + _PREFIX_TEXT
    context_text = render_excerpts_text(excerpts)
    suffix_text = _SUFFIX_TEMPLATE.format(question=question) + chat_after
    return prefix_text, context_text, suffix_text


# ---------------------------------------------------------------------------
# Budget sizing (pure Python -- ported from predict_longbench_v2.py's
# confirmed real-hardware fix; no vllm dependency).
# ---------------------------------------------------------------------------


def percentile(sorted_vals: list[float], q: float) -> float | None:
    """Same shape as predict_longbench_v2.py's own helper."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def resolve_max_num_batched_tokens(explicit: int | None, token_lengths: list[int]) -> int:
    """Auto-sizes off p90 (not the raw max) with a margin, same fix
    `predict_longbench_v2.py` adopted after a confirmed real-hardware CUDA
    OOM: one extreme-outlier sample auto-sizing the budget off the raw max
    made `LLM()`'s own startup memory-profiling OOM before any real request
    ran. An outlier here should get skipped-and-reported at submit time
    (see `_submit_plain`/`_submit_pruned`), not dictate the budget for
    every other sample."""
    if explicit is not None:
        return explicit
    if not token_lengths:
        return DEFAULT_MAX_NUM_BATCHED_TOKENS

    sorted_lengths = sorted(token_lengths)
    p90_len = int(percentile(sorted_lengths, 0.9))
    margin = max(256, p90_len // 5)
    return ((p90_len + margin + 1023) // 1024) * 1024


# ---------------------------------------------------------------------------
# Engine construction (vllm/torch/transformers imports deferred to here).
# ---------------------------------------------------------------------------


def ensure_spec_prefill_llama_on_path() -> None:
    """`vllm_patch` (SpecPrefillWorker/proposer/pruner/config) lives in
    ../spec_prefill_llama/, not an installed package -- resolved via
    $SPEC_PREFILL_LLAMA_DIR (set by .env_exports.sh) rather than a hardcoded
    relative path, matching that file's own rationale for target_stage/'s
    dependency on it."""
    spec_prefill_dir = os.environ.get("SPEC_PREFILL_LLAMA_DIR")
    if not spec_prefill_dir:
        raise RuntimeError(
            "SPEC_PREFILL_LLAMA_DIR is not set -- source .env_exports.sh first "
            "(it points this at ../spec_prefill_llama so vllm_patch/ is importable)."
        )
    if spec_prefill_dir not in sys.path:
        sys.path.insert(0, spec_prefill_dir)


def load_spec_config(path: Path):
    """Loads a SpecConfig from YAML (e.g. configs/spec_config_always_on.yaml)
    via spec_prefill_llama's own loader -- see that module's docstring for
    why this file is pure Python/YAML with no vLLM engine dependency (safe
    to call before the engine is constructed)."""
    ensure_spec_prefill_llama_on_path()
    from vllm_patch.config import SpecConfig

    return SpecConfig.from_path(str(path))


def build_plain_target_engine(
    target_model_path: str,
    *,
    gpu_memory_utilization: float = 0.9,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    max_tokens: int = 512,
    enforce_eager: bool = True,
) -> EngineHandle:
    """Arm A (and Arm C's "skip" bucket): no worker_cls, no proposer --
    SpecPrefillWorker is provably a no-op without a registered
    PruneRecord (`predict_longbench_v2.py`'s own rationale for skipping it
    on baselines), so there's no correctness reason to pay for it here."""
    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    resolved_max_num_batched_tokens = max_num_batched_tokens or DEFAULT_MAX_NUM_BATCHED_TOKENS
    resolved_max_model_len = max_model_len or (resolved_max_num_batched_tokens + max_tokens)

    llm = LLM(
        model=target_model_path,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        disable_log_stats=False,  # required for RequestOutput.metrics (TTFT) -- confirmed real-hardware requirement
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=resolved_max_num_batched_tokens,
        max_model_len=resolved_max_model_len,
    )
    return EngineHandle(
        llm=llm,
        tokenizer=tok,
        is_pruned=False,
        max_num_batched_tokens=resolved_max_num_batched_tokens,
        max_tokens=max_tokens,
    )


def build_specprefill_target_engine(
    target_model_path: str,
    speculator_model_path: str,
    spec_config,
    *,
    gpu_memory_utilization: float = 0.9,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    max_tokens: int = 512,
    enforce_eager: bool = True,
    speculator_device: str | None = None,
) -> EngineHandle:
    """Arm B (and Arm C's "compress" bucket): `worker_cls=vllm_patch.worker.SpecPrefillWorker`
    plus a standalone `SpecPrefillProposer` for the speculator -- mirrors
    `predict_longbench_v2.py::run_experiment`'s non-baseline branch exactly
    (speculator device selection, `torch.cuda.set_device` before model
    load -- confirmed real-hardware requirement that weight placement
    follows torch's *current* device, not an explicit device arg)."""
    ensure_spec_prefill_llama_on_path()
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.config import ModelConfig, VllmConfig
    from vllm_patch.proposer import SpecPrefillProposer

    tok = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    resolved_max_num_batched_tokens = max_num_batched_tokens or DEFAULT_MAX_NUM_BATCHED_TOKENS
    resolved_max_model_len = max_model_len or (resolved_max_num_batched_tokens + max_tokens)

    llm = LLM(
        model=target_model_path,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=resolved_max_num_batched_tokens,
        max_model_len=resolved_max_model_len,
        worker_cls="vllm_patch.worker.SpecPrefillWorker",
    )

    if speculator_device is not None:
        device = torch.device(speculator_device)
    elif torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cuda")
        print(
            f"WARNING: only {torch.cuda.device_count()} GPU(s) visible -- "
            f"speculator will share the target's GPU."
        )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    speculator_model_config = ModelConfig(model=speculator_model_path, trust_remote_code=True, dtype="bfloat16")
    base_vllm_config = VllmConfig(model_config=speculator_model_config)
    proposer = SpecPrefillProposer(
        base_vllm_config=base_vllm_config,
        speculator_model_config=speculator_model_config,
        device=device,
    )
    head_dim = proposer._speculator_layers[0].head_dim

    return EngineHandle(
        llm=llm,
        tokenizer=tok,
        is_pruned=True,
        max_num_batched_tokens=resolved_max_num_batched_tokens,
        max_tokens=max_tokens,
        proposer=proposer,
        spec_config=spec_config,
        speculator_device=device,
        head_dim=head_dim,
    )


def teardown_engine(engine: EngineHandle) -> None:
    """Frees the engine (and proposer, if any) before building a different
    engine in the same process -- required by IMPLEMENTATION_PLAN.md
    decision 3 (Arm C tears down the plain engine before building the
    SpecPrefill one; `worker_cls` can't be hot-swapped on a live `LLM`)."""
    import gc

    del engine.llm
    if engine.proposer is not None:
        del engine.proposer
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Request submission + driving (vllm imports deferred to here).
# ---------------------------------------------------------------------------


def drive_engine_to_completion(llm_engine) -> tuple[dict, dict]:
    """Runs llm_engine.step() until no unfinished requests remain,
    accumulating each request's latest RequestOutput by request_id. The
    core loop is copied verbatim from predict_longbench_v2.py's
    identically-named function -- see that function's docstring for the
    two confirmed real-hardware hazards this avoids (early-abort leaves
    output-delivery bookkeeping inconsistent; skipping
    has_unfinished_requests() can block forever).

    Extended beyond that original (predict_longbench_v2.py never needed
    per-request total latency, only TTFT + aggregate batch throughput):
    also records a wall-clock timestamp the first time each request's
    output reports `finished=True`. Combined with the batch's own start
    time (see `answer_batch`), this gives a real per-request "time from
    submission to finish" within the batch -- not identical to a live
    server's per-request latency (all requests here are submitted
    ~simultaneously, same caveat `predict_longbench_v2.py`'s own TTFT
    documentation already makes), but a genuine measured duration, not an
    estimate, and exactly the T_target this project's `f` metric
    (../rlm_specprefill_ablation_plan.md's LATENCY MODEL) needs per sample.

    Returns (outputs_by_id, finish_time_by_id).
    """
    import time

    outputs_by_id: dict = {}
    finish_time_by_id: dict = {}
    while llm_engine.has_unfinished_requests():
        for output in llm_engine.step():
            outputs_by_id[output.request_id] = output
            if output.finished and output.request_id not in finish_time_by_id:
                finish_time_by_id[output.request_id] = time.perf_counter()
    return outputs_by_id, finish_time_by_id


def _submit_plain(
    llm, tok, queries: list[TargetQuery], max_tokens: int, max_num_batched_tokens: int
) -> tuple[dict[str, _SubmittedRequest], int]:
    """Mirrors predict_longbench_v2.py::submit_baseline_requests. Keyed by
    the ORIGINAL (caller-supplied) request_id, NOT add_request()'s rewritten
    return value -- confirmed on real hardware in that script: step()'s
    RequestOutput objects are always keyed by the original id
    (vllm/v1/engine/output_processor.py's `_new_request_output`, literally
    commented "request_id is what was provided externally")."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    id_to_submitted: dict[str, _SubmittedRequest] = {}
    num_skipped = 0
    for i, query in enumerate(queries):
        prefix, context, suffix = build_prompt_pieces(tok, query.question, query.excerpts)
        prompt_token_ids = tok.encode(prefix + context + suffix, add_special_tokens=False)
        if len(prompt_token_ids) > max_num_batched_tokens:
            num_skipped += 1
            print(
                f"[vllm_offline_engine] SKIP sample id={query.sample_id!r}: "
                f"prompt length {len(prompt_token_ids)} exceeds "
                f"max_num_batched_tokens={max_num_batched_tokens}."
            )
            continue
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        request_id = f"target-{query.sample_id}-{i}"
        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        llm.llm_engine.add_request(request_id, prompt, sampling_params)
        id_to_submitted[request_id] = _SubmittedRequest(query=query, n_full=len(prompt_token_ids))
    return id_to_submitted, num_skipped


def _submit_pruned(
    llm,
    tok,
    queries: list[TargetQuery],
    proposer,
    spec_config,
    speculator_device,
    head_dim: int,
    max_tokens: int,
    max_num_batched_tokens: int,
) -> tuple[dict[str, _SubmittedRequest], int]:
    """Mirrors predict_longbench_v2.py::submit_pruned_requests, including
    its two confirmed real-hardware fixes:
    (1) prefix/suffix (here: the chat wrapper + instructions, and the
        question) are tokenized SEPARATELY from the excerpts and
        force-included regardless of SpecPrefill's own scoring -- an
        earlier version of that script fed everything as one blob and
        produced gibberish once the trailing instruction got pruned away
        at aggressive keep rates.
    (2) `real_id` (add_request()'s rewritten return value) is required for
        `collective_rpc`'s `register_prune_record` call -- model_runner.py's
        pruning_registry lookup runs in the Worker process against the
        rewritten id, a different namespace from what step()'s outputs use.
    """
    import gc

    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm_patch.pruner import compute_pruned_prompt

    id_to_submitted: dict[str, _SubmittedRequest] = {}
    num_skipped = 0

    for i, query in enumerate(queries):
        request_id = f"target-{query.sample_id}-{i}"
        prefix_text, context_text, suffix_text = build_prompt_pieces(tok, query.question, query.excerpts)

        prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
        context_ids = tok.encode(context_text, add_special_tokens=False)
        suffix_ids = tok.encode(suffix_text, add_special_tokens=False)
        full_prompt_token_ids = prefix_ids + context_ids + suffix_ids

        if len(full_prompt_token_ids) > max_num_batched_tokens:
            num_skipped += 1
            print(
                f"[vllm_offline_engine] SKIP sample id={query.sample_id!r}: "
                f"full (unpruned) prompt length {len(full_prompt_token_ids)} "
                f"exceeds max_num_batched_tokens={max_num_batched_tokens} -- "
                f"the speculator's bootstrap prefill must process the whole "
                f"prompt in one shot to score it."
            )
            continue

        raw_pruned_ids, raw_kept_positions = compute_pruned_prompt(
            proposer=proposer,
            spec_config=spec_config,
            prompt_token_ids=full_prompt_token_ids,
            device=speculator_device,
            head_dim=head_dim,
            eos_token_id=tok.eos_token_id,
        )
        # Proactively release cached-but-idle CUDA blocks after every
        # sample -- confirmed real-hardware fragmentation/accumulation
        # risk in predict_longbench_v2.py's own history (a later OOM
        # reported far more allocated on the speculator's GPU than the
        # checkpoint itself needs).
        gc.collect()
        torch.cuda.empty_cache()

        context_start = len(prefix_ids)
        context_end = len(prefix_ids) + len(context_ids)
        kept_context_pairs = [
            (tok_id, pos)
            for tok_id, pos in zip(raw_pruned_ids, raw_kept_positions)
            if context_start <= pos < context_end
        ]
        kept_context_ids = [t for t, _ in kept_context_pairs]
        kept_context_positions = [p for _, p in kept_context_pairs]

        pruned_token_ids = prefix_ids + kept_context_ids + suffix_ids
        kept_positions = (
            list(range(context_start))
            + kept_context_positions
            + list(range(context_end, context_end + len(suffix_ids)))
        )

        if len(pruned_token_ids) > max_num_batched_tokens:
            num_skipped += 1
            print(
                f"[vllm_offline_engine] SKIP sample id={query.sample_id!r}: "
                f"pruned length {len(pruned_token_ids)} exceeds "
                f"max_num_batched_tokens={max_num_batched_tokens}."
            )
            continue

        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        prompt = TokensPrompt(prompt_token_ids=pruned_token_ids, cache_salt=request_id)
        real_id = llm.llm_engine.add_request(request_id, prompt, sampling_params)
        llm.llm_engine.collective_rpc(
            "register_prune_record", args=(real_id, kept_positions, len(full_prompt_token_ids))
        )

        id_to_submitted[request_id] = _SubmittedRequest(
            query=query, n_full=len(full_prompt_token_ids), n_kept=len(pruned_token_ids)
        )

    return id_to_submitted, num_skipped


def answer_batch(engine: EngineHandle, queries: list[TargetQuery]) -> list[TargetAnswer]:
    """Submits every query to `engine` (pruned or plain, per
    `engine.is_pruned`), drives the engine to completion, and returns one
    `TargetAnswer` per query that got a captured output (queries skipped
    for exceeding the token budget are reported via stdout, not silently
    dropped -- matching `predict_longbench_v2.py`'s own convention)."""
    if engine.is_pruned:
        id_to_submitted, num_skipped = _submit_pruned(
            engine.llm,
            engine.tokenizer,
            queries,
            engine.proposer,
            engine.spec_config,
            engine.speculator_device,
            engine.head_dim,
            engine.max_tokens,
            engine.max_num_batched_tokens,
        )
    else:
        id_to_submitted, num_skipped = _submit_plain(
            engine.llm, engine.tokenizer, queries, engine.max_tokens, engine.max_num_batched_tokens
        )

    import time

    batch_start = time.perf_counter()
    outputs_by_id, finish_time_by_id = drive_engine_to_completion(engine.llm.llm_engine)

    answers: list[TargetAnswer] = []
    for req_id, submitted in id_to_submitted.items():
        output = outputs_by_id.get(req_id)
        if output is None:
            print(
                f"WARNING: no output captured for sample id={submitted.query.sample_id!r} "
                f"(request_id={req_id!r}) -- skipping from results."
            )
            continue
        completion = output.outputs[0]

        # NOT (first_token_ts - arrival_time) -- confirmed on real hardware
        # in predict_longbench_v2.py that this produces a huge negative
        # number (two different clocks). first_token_latency is vLLM's own
        # already-computed value on a single consistent basis.
        ttft_ms = None
        if output.metrics is not None and output.metrics.first_token_latency:
            ttft_ms = output.metrics.first_token_latency * 1000

        finish_time = finish_time_by_id.get(req_id)
        generation_time_s = (finish_time - batch_start) if finish_time is not None else 0.0

        answers.append(
            TargetAnswer(
                sample_id=submitted.query.sample_id,
                answer_text=completion.text,
                finish_reason=completion.finish_reason,
                n_output_tokens=len(completion.token_ids),
                n_prompt_tokens_full=submitted.n_full,
                n_prompt_tokens_kept=submitted.n_kept,
                ttft_ms=ttft_ms,
                generation_time_s=generation_time_s,
            )
        )

    if num_skipped:
        print(f"[vllm_offline_engine] {num_skipped} sample(s) skipped (over budget) out of {len(queries)}.")

    return answers
