#!/usr/bin/env python3
"""Prediction-generation driver for SpecPrefill's LongBench v2 keep-rate sweep
(EXPERIMENT_PLAN.md's experiment matrix P001-P006). Runs the target model
(Qwen3-8B), optionally pruned via SpecPrefill (vllm_patch/), against
datasets/prep_longbench_v2.py's samples and writes a predictions JSONL that
grade_longbench_v2.py scores directly (`{"id", "pred"}` per line).

This is a Qwen3 port of `spec_prefill/predict_longbench_v2.py`. The
engine-driving loop, TTFT/output-length metrics, and CSV/JSONL output are
carried over unchanged (architecture-agnostic); only the env-var-backed
default model paths and the config-deviation rationale below were updated.

Config deviations from EXPERIMENT_PLAN.md, and why (see that file's
"SpecPrefill settings" section, updated alongside this script):

1. chunk_size=64, not the prose section's stale "32" -- the experiment matrix
   itself (P002-P006) and validate_runner_integration.py both already use 64.
2. pool_kernel_size=13, not validate_runner_integration.py's `None` -- 13 is
   what the matrix's own cited basis configs (config_p{1,3,5,7,9}_full_lah8.yaml)
   use; `None` there was a validation-script simplification, not a protocol
   choice.
3. `enable_chunked_prefill` is left at the model's own supported default (on)
   for now, matching the sibling Gemma4 pipeline's approach -- ported here as
   a starting point, NOT independently re-confirmed for Qwen3-8B. That
   pipeline's own reason (Gemma-4-26B-A4B-it doesn't officially support
   disabling chunked prefill, and disabling it would require
   max_num_batched_tokens >= its native max_model_len of 262144 on every
   step) is Gemma4-specific and may not apply to Qwen3-8B at all -- see
   validate_runner_integration.py's module docstring for the fuller
   discussion. Re-check this deviation with the user once Qwen3-8B has been
   validated, rather than assuming it still holds.

Known residual risk this creates, guarded by the pre-flight check below: a
confirmed-real upstream bug means a request whose prompt exceeds
max_num_batched_tokens never correctly resumes prefill past its first chunk
(silently generates against an incomplete KV cache). LongBench v2 "short"
prompts run up to ~32k words -- for P001 (no pruning) the FULL prompt must fit
in one chunk; for P002-P006 only the PRUNED (kept) token count must.

**Multi-request engine-driving loop.** validate_runner_integration.py proves
the underlying pattern (has_unfinished_requests()/step(), never early-abort)
for 1-2 requests driven purely for a position-capture diagnostic (it never
needed to correlate step()'s output back to a specific request by content,
since it read positions via a separate collective_rpc channel instead). This
generalizes the pattern to N concurrently-submitted requests whose actual
generated text is collected, by accumulating each step()'s returned
RequestOutputs into a request_id-keyed dict.

**Confirmed on real hardware (2026-07-23): id_to_sample must be keyed by the
ORIGINAL (caller-supplied) request_id, NOT the rewritten id add_request()
returns.** An earlier version of this script keyed it by the rewritten id
(following pruner.py's "always use the returned real_request_id" guidance,
which is correct for THAT function's purpose -- matching the WORKER-side
req_id for collective_rpc registration) and got zero captured outputs for
every single request, even though has_unfinished_requests()/step() drove them
to real, correct completion (confirmed via wall-clock time spent). Root cause:
vllm/v1/engine/output_processor.py's `_new_request_output` always builds
`RequestOutput(request_id=external_req_id, ...)` -- literally commented
"request_id is what was provided externally" in that file -- so step()'s
output objects are keyed by the ORIGINAL id in every case, while
add_request()'s return value (and the WORKER-side req_id pruner.py/
model_runner.py depend on) is the rewritten one. These are two genuinely
different id namespaces serving two different purposes; see
submit_baseline_requests/submit_pruned_requests below for exactly where each
one is used now.

TTFT is a *batched-generate offline approximation*: all requests in one
experiment are submitted ~simultaneously (not a live server's Poisson-arrival
stream), so this is not directly comparable to benchmark_serving.py's
concurrent-streaming-client TTFT -- both are real, vLLM-internal-clock-accurate
numbers, but they measure different things.

**Confirmed on real hardware (2026-07-23): use `metrics.first_token_latency`,
NOT `metrics.first_token_ts - metrics.arrival_time`.** An earlier version of
this script computed TTFT that way and got a huge negative number in
practice -- `arrival_time` is a wall-clock timestamp but `first_token_ts` is
an "engine core timestamp (monotonic)" (vllm/v1/metrics/stats.py:207-214), a
different clock with an arbitrary epoch; subtracting across the two is
meaningless. `first_token_latency` is vLLM's own already-computed value on a
single consistent basis (`IterationStats._time_since(arrival_time)`,
stats.py:349-351/369-371) -- use that directly instead of re-deriving it.

Requires disable_log_stats=False
(see run_pipeline.py's identical use for the same reason).

Usage:
    python3 predict_longbench_v2.py --list
    python3 predict_longbench_v2.py --exp P001 --max-keep 2   # smoke test first
    python3 predict_longbench_v2.py --exp P001,P002,P003,P004,P005,P006
    python3 predict_longbench_v2.py --exp P000  # speculator standalone baseline
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))  # for `vllm_patch` imports

OUT_DIR = Path(os.environ.get("BENCH_RESULTS_DIR", "results"))
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "all_runs.csv"
DEFAULT_SAMPLES = Path(__file__).parent / "datasets" / "longbench_v2_samples.jsonl"

CSV_FIELDS = [
    "ts", "exp_id", "label",
    "keep_percentage", "chunk_size", "look_ahead_cnt", "pool_kernel_size",
    "target_gpu_memory_utilization", "target_max_num_batched_tokens",
    "rep", "seed", "max_tokens",
    "num_samples", "num_skipped_too_large",
    "elapsed_time", "requests_per_second",
    "actual_keep_rate_mean",
    "ttft_mean_ms", "ttft_p50_ms", "ttft_p90_ms",
    "out_len_mean", "out_len_stdev", "out_len_p50", "out_len_p90", "out_len_max",
    "finish_stop", "finish_length", "finish_other",
]

# See module docstring items 1-2 -- deviate from EXPERIMENT_PLAN.md's stale
# "32" prose value and validate_runner_integration.py's `None` simplification.
CHUNK_SIZE = 64
LOOK_AHEAD_CNT = 8
POOL_KERNEL_SIZE = 13

PRUNE_EXPERIMENTS = {
    # Standalone speculator (Qwen3-1.7B) baseline -- no pruning, no target
    # model involved at all. Answers "how well does the small speculator do
    # on its own?", a useful reference point distinct from P001 (the 8B
    # target's own unpruned baseline) -- e.g. to sanity-check that
    # SpecPrefill's pruned target-model accuracy (P002-P006) isn't secretly
    # just tracking the much weaker speculator's own ceiling. Numbered P000
    # (not part of the official P001-P006 experiment matrix) so it sorts
    # before P001 in --list output and slots into the same --exp comma-
    # separated syntax as everything else.
    "P000": {"label": "Speculator standalone baseline (no pruning)", "keep_percentage": None, "model": "speculator"},
    "P001": {"label": "Baseline (no SpecPrefill)", "keep_percentage": None, "model": "target"},
    "P002": {"label": "Keep 10%", "keep_percentage": 0.1, "model": "target"},
    "P003": {"label": "Keep 30%", "keep_percentage": 0.3, "model": "target"},
    "P004": {"label": "Keep 50%", "keep_percentage": 0.5, "model": "target"},
    "P005": {"label": "Keep 70%", "keep_percentage": 0.7, "model": "target"},
    "P006": {"label": "Keep 90%", "keep_percentage": 0.9, "model": "target"},
}


def ensure_csv_header() -> None:
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(row: dict) -> None:
    with CSV_PATH.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def percentile(sorted_vals: list, q: float):
    """Same shape as gemma4_moe_benchmarks/bench_experiment.py's helper."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def load_samples(path: Path, max_keep: int) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if max_keep >= 0 and len(samples) >= max_keep:
                break
    if not samples:
        raise FileNotFoundError(
            f"Samples file empty or not found: {path}\n"
            "Run datasets/prep_longbench_v2.py first."
        )
    return samples


def render_chat(tok, prompt: str) -> str:
    """Required for this instruction-tuned checkpoint -- see
    validate_runner_integration.py's confirmed finding: a bare
    completion-style prompt produces degenerate repetition loops under greedy
    decoding, even with stock vLLM.

    `enable_thinking=False`: EXPERIMENT_PLAN.md's header table specifies
    "thinking mode off, greedy decoding" for this protocol -- confirmed on
    real hardware (2026-07-28) that Qwen3's chat template defaults to
    reasoning mode ON otherwise (observed a leading `<think>...` block in an
    unrelated smoke-test generation), which Gemma4 has no equivalent of, so
    this wasn't something the sibling pipeline's render_chat needed to
    handle. `tok.apply_chat_template` forwards unrecognized kwargs straight
    into the Jinja template context, which is how Qwen3's own template reads
    this flag."""
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


_CHAT_WRAPPER_PLACEHOLDER = "@@SPEC_PREFILL_CONTENT_PLACEHOLDER@@"


def chat_wrapper_pieces(tok) -> tuple[str, str]:
    """Returns (before, after): the fixed chat-template text surrounding a
    single user turn's content (e.g. `<start_of_turn>user\\n` /
    `<end_of_turn>\\n<start_of_turn>model\\n`), independent of what that
    content actually is. Used by submit_pruned_requests to tokenize the
    always-kept prefix/suffix pieces separately from the prunable context --
    see that function's docstring for why.

    `enable_thinking=False` must match render_chat's own setting exactly --
    Qwen3's template changes what it appends after the generation prompt
    depending on this flag (e.g. an empty `<think>\\n\\n</think>\\n\\n`
    block when explicitly disabled), so the "after" piece computed here
    would silently diverge from what render_chat actually produces
    elsewhere if this call disagreed with it."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": _CHAT_WRAPPER_PLACEHOLDER}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    before, after = rendered.split(_CHAT_WRAPPER_PLACEHOLDER)
    return before, after


def token_length_summary(tok, samples: list[dict]) -> list[int]:
    """Tokenizes every sample's rendered prompt and returns the sorted token
    counts (full, unpruned length) -- used both for the printed percentile
    summary and for P001's pre-flight hard budget check (see module
    docstring's "known residual risk")."""
    lengths = []
    for sample in samples:
        rendered = render_chat(tok, sample["prompt"])
        lengths.append(len(tok.encode(rendered, add_special_tokens=False)))
    lengths.sort()
    return lengths


def resolve_max_num_batched_tokens(
    explicit: Optional[int], token_lengths: list[int], is_baseline: bool
) -> int:
    """If the user didn't pass --target-max-num-batched-tokens explicitly,
    auto-size it off **p90**, not the raw max.

    Confirmed on real hardware (2026-07-24): auto-sizing off the raw max
    directly caused a CUDA OOM -- one dataset sample (813963 tokens, ~14x
    p90's 58180) made the auto-sized budget balloon to 855040, which is far
    more than an 80GB GPU can hold even before any real request runs (vLLM's
    own internal profile_run()/`_dummy_run` simulates a full-budget-sized
    batch to measure available memory, and OOM'd trying to allocate for it).
    Whether that one sample is a genuine (if extreme) "short" LongBench v2
    entry or a data/filtering anomaly in prep_longbench_v2.py is a separate
    question worth checking directly against the samples file -- either way,
    a single extreme sample should not be allowed to dictate a budget that
    makes every OTHER sample's run impossible.

    Both P001 (baseline, full prompt) and P002-P006 (pruned, kept-token
    count) now use the SAME skip-and-report treatment for any individual
    sample that doesn't fit the resolved budget (see submit_baseline_requests/
    submit_pruned_requests) -- there is no longer a hard requirement that the
    budget cover literally every sample, baseline included."""
    p90_len = int(percentile(token_lengths, 0.9)) if token_lengths else 0
    max_len = token_lengths[-1] if token_lengths else 0
    if explicit is not None:
        return explicit

    # Auto-size off p90 with a safety margin, capped so a single extreme
    # outlier can't balloon the budget -- oversized samples get skipped
    # per-request instead (see docstring above). Round up to a clean
    # multiple of 1024 for legibility, not for any correctness reason.
    margin = max(256, p90_len // 5)
    auto = ((p90_len + margin + 1023) // 1024) * 1024
    print(
        f"[predict_longbench_v2] --target-max-num-batched-tokens not given -- "
        f"auto-sized to {auto} from p90 ({p90_len} tokens + {margin}-token "
        f"margin), NOT the raw max ({max_len} tokens) -- a single extreme "
        f"outlier should not dictate a budget that risks OOM for everyone "
        f"else. Samples exceeding {auto} tokens will be skipped and reported "
        f"rather than included. Pass --target-max-num-batched-tokens "
        f"explicitly to override."
    )
    return auto


def drive_engine_to_completion(llm_engine) -> dict:
    """Runs llm_engine.step() until no unfinished requests remain,
    accumulating each request's latest RequestOutput by request_id (later
    steps' entries overwrite earlier ones for the same id; the final,
    `.finished=True` entry is what's kept once the loop exits). See module
    docstring's "Multi-request engine-driving loop" section.

    Never breaks out early and never calls abort_request() to "clean up" --
    both are confirmed real hazards on real hardware (validate_runner_
    integration.py: an unconditional step() without checking
    has_unfinished_requests() first can block forever once all requests
    finish; an early abort left engine output-delivery bookkeeping
    inconsistent and crashed a later llm.generate() call)."""
    outputs_by_id: dict = {}
    while llm_engine.has_unfinished_requests():
        for output in llm_engine.step():
            outputs_by_id[output.request_id] = output
    return outputs_by_id


def submit_baseline_requests(
    llm, samples: list[dict], tok, max_tokens: int, max_num_batched_tokens: int
) -> tuple[dict, int]:
    """P001: plain add_request, no worker_cls/proposer/pruning at all --
    SpecPrefillWorker is provably a no-op without any registered PruneRecord
    (see vllm_patch/model_runner.py's docstring), so there's no correctness
    reason to pay for it on the baseline.

    Skips (and reports, doesn't silently drop) any individual sample whose
    full prompt exceeds max_num_batched_tokens -- same treatment as
    submit_pruned_requests, no longer a hard requirement that the resolved
    budget cover literally every sample (see resolve_max_num_batched_tokens's
    docstring for the real-hardware OOM this replaces).

    Returns: (id_to_sample, num_skipped_too_large).
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    id_to_sample: dict = {}
    num_skipped = 0
    for i, sample in enumerate(samples):
        rendered = render_chat(tok, sample["prompt"])
        prompt_token_ids = tok.encode(rendered, add_special_tokens=False)
        if len(prompt_token_ids) > max_num_batched_tokens:
            num_skipped += 1
            print(
                f"[predict_longbench_v2] SKIP sample id={sample['id']!r}: "
                f"full prompt length {len(prompt_token_ids)} exceeds "
                f"--target-max-num-batched-tokens={max_num_batched_tokens}."
            )
            continue
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        request_id = f"lbv2-{sample['id']}-{i}"
        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        # add_request()'s return value is the REWRITTEN id (InputProcessor.
        # assign_request_id appends "-{8 random chars}") -- needed for
        # anything that must match the WORKER-side req_id (e.g. pruner.py's
        # collective_rpc registration, in submit_pruned_requests below), but
        # NOT for correlating step()'s returned RequestOutputs back to a
        # sample: output_processor.py's _new_request_output always builds
        # RequestOutput(request_id=external_req_id, ...) -- the ORIGINAL
        # caller-supplied id, never the rewritten one (confirmed on real
        # hardware, 2026-07-23: keying this dict by the rewritten id instead
        # left every single request's output uncaptured, "no output
        # captured" for 100% of requests, even though has_unfinished_
        # requests()/step() drove them to real completion). Deliberately
        # NOT capturing add_request()'s return value here at all -- nothing
        # in this function needs it.
        llm.llm_engine.add_request(request_id, prompt, sampling_params)
        id_to_sample[request_id] = sample
    return id_to_sample, num_skipped


def submit_pruned_requests(
    llm,
    samples: list[dict],
    tok,
    proposer,
    spec_config,
    speculator_device,
    head_dim: int,
    max_tokens: int,
    max_num_batched_tokens: int,
) -> tuple[dict, dict, int]:
    """P002-P006: prunes each sample via SpecPrefill, then adds it -- with a
    per-request budget check BEFORE add_request (skip + report, don't
    silently corrupt into the upstream chunked-prefill-resume bug -- see
    module docstring). This calls vllm_patch.pruner.compute_pruned_prompt
    directly rather than prune_and_add_request, specifically so the kept-
    token count can be checked before committing to add_request (kept length
    isn't known until pruning runs, unlike the baseline's full-prompt case
    which is checked entirely up front in resolve_max_num_batched_tokens).
    The add_request + collective_rpc sequencing below otherwise mirrors
    prune_and_add_request exactly -- see that function's docstring for why
    the ordering (add_request first, THEN collective_rpc with the returned
    real id) is load-bearing.

    **Confirmed on real hardware (2026-07-24): the question/choices/answer-
    format instruction must be force-kept, never left to SpecPrefill's own
    chunk-based scoring.** An earlier version fed the ENTIRE combined
    "prompt" (context + question + choices + instruction) into
    compute_pruned_prompt as one blob. P001 (no pruning) produced perfectly
    coherent answers; P002-P006 (pruned) produced pure gibberish -- ruling
    out the model/checkpoint/environment (same for both) and pointing at
    pruning specifically. Root cause: at aggressive keep rates, the trailing
    "Answer with a single letter..." instruction is a tiny fraction of the
    prompt and has no special protection -- it competes for a spot against
    thousands of document chunks and can be pruned away entirely, so the
    model never learns it's supposed to answer a multiple-choice question at
    all. It just free-continues whatever document fragments survived, which
    is exactly the observed gibberish (not a positions bug -- Step B2 in
    validate_runner_integration.py passed).

    Fix: the speculator still SCORES the full context+question+choices+
    instruction sequence (so its importance scoring still sees the actual
    question, unaffected), but the prefix ("Please read the following long
    text...") and suffix (question/choices/instruction + chat closing
    wrapper) are tokenized SEPARATELY from the context and force-included in
    the final pruned request regardless of what compute_pruned_prompt
    decided for those positions -- only the CONTEXT region's own scoring
    decision is honored. Tokenizing prefix/context/suffix as separate pieces
    (rather than slicing one jointly-tokenized string at a computed
    boundary) avoids subword-tokenization boundary ambiguity entirely.

    **`id_to_sample`/`keep_stats` are keyed by the ORIGINAL (caller-supplied)
    request_id, NOT the rewritten one add_request() returns** -- confirmed on
    real hardware (2026-07-23, see submit_baseline_requests's docstring for
    the full finding): step()'s RequestOutput objects always carry
    `external_req_id` (the original id), never the rewritten one. The
    rewritten id (`real_id` below) is used ONLY for the collective_rpc call,
    which must match the WORKER-side req_id -- a completely different
    namespace from what correlates back to a sample here.

    Returns: (id_to_sample, keep_stats, num_skipped_too_large) -- both dicts
    keyed by the original request_id; keep_stats values are
    (num_kept, orig_len) per request.
    """
    import gc

    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm_patch.pruner import compute_pruned_prompt

    # Mirrors datasets/prep_longbench_v2.py's _PROMPT_TEMPLATE exactly (must
    # stay in sync with it) -- split into the piece that comes before
    # {context} and the piece that comes after, so each can be tokenized
    # independently of the (potentially huge) context.
    _PREFIX_TEXT = "Please read the following long text and answer the question below.\n\n"
    _SUFFIX_TEMPLATE = (
        "\n\nQuestion: {question}\n(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}"
        "\n(D) {choice_D}\n\nAnswer with ONLY a single letter."
    )

    id_to_sample: dict = {}
    keep_stats: dict = {}
    num_skipped = 0

    chat_before, chat_after = chat_wrapper_pieces(tok)

    for i, sample in enumerate(samples):
        request_id = f"lbv2-{sample['id']}-{i}"

        prefix_text = chat_before + _PREFIX_TEXT
        context_text = sample["context"]
        suffix_text = _SUFFIX_TEMPLATE.format(
            question=sample["question"],
            choice_A=sample["choices"][0],
            choice_B=sample["choices"][1],
            choice_C=sample["choices"][2],
            choice_D=sample["choices"][3],
        ) + chat_after

        prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
        context_ids = tok.encode(context_text, add_special_tokens=False)
        suffix_ids = tok.encode(suffix_text, add_special_tokens=False)
        full_prompt_token_ids = prefix_ids + context_ids + suffix_ids

        # Confirmed on real hardware (2026-07-24): the speculator's bootstrap
        # prefill (inside compute_pruned_prompt/run_lookahead_steps) must
        # process the FULL, UNPRUNED prompt in one shot to score it -- there
        # was no check here against the full prompt length before this,
        # only against the PRUNED length afterward. In the sibling Gemma4
        # pipeline this OOM'd even on an otherwise-idle GPU holding only the
        # ~9.5 GiB E2B speculator -- long full prompts risk OOM on the
        # speculator's own GPU regardless of the specific mechanism, since
        # the bootstrap prefill's activations/attention scores scale with
        # prompt length either way. Reusing max_num_batched_tokens as the
        # ceiling here too -- not a
        # precisely profiled speculator-specific budget, just a reasonable,
        # already-established bound rather than inventing a new untested
        # threshold. Skip and report (loud, not silent) rather than crash.
        if len(full_prompt_token_ids) > max_num_batched_tokens:
            num_skipped += 1
            print(
                f"[predict_longbench_v2] SKIP sample id={sample['id']!r}: "
                f"full (unpruned) prompt length {len(full_prompt_token_ids)} "
                f"exceeds --target-max-num-batched-tokens="
                f"{max_num_batched_tokens} -- the speculator's bootstrap "
                f"prefill must process the whole prompt in one shot to "
                f"score it, and this risks OOM on the speculator's GPU "
                f"before pruning even happens."
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
        # Confirmed on real hardware (2026-07-24): build_lookahead_metadata
        # (called inside compute_pruned_prompt, once per sample) allocates a
        # FRESH dummy KV cache every call, sized for THIS sample's own
        # prompt_len -- 178 different-length samples in this loop means 178
        # differently-sized CUDA allocations in sequence, with nothing
        # explicitly releasing one sample's tensors before the next sample's
        # (potentially much larger) allocation. This is a real fragmentation/
        # accumulation risk (confirmed in the sibling Gemma4 pipeline: a
        # later OOM reported 58+ GiB already allocated on the speculator's
        # GPU, far more than the small speculator checkpoint itself needs)
        # -- proactively releasing PyTorch's cached-
        # but-idle blocks after every sample keeps each iteration's peak
        # memory closer to what THAT sample alone needs, rather than
        # accumulating across the whole 178-sample loop.
        gc.collect()
        torch.cuda.empty_cache()

        # Force-keep prefix/suffix regardless of what SpecPrefill's own
        # scoring decided for those positions -- only honor its decision for
        # the CONTEXT region (see this function's docstring for why). The
        # scorer still SAW the full context+question+choices+instruction
        # sequence when computing importance, so context scoring quality is
        # unaffected -- only the FINAL kept set is overridden here.
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
                f"[predict_longbench_v2] SKIP sample id={sample['id']!r}: "
                f"pruned length {len(pruned_token_ids)} (after force-keeping "
                f"the question/choices/instruction) exceeds "
                f"--target-max-num-batched-tokens={max_num_batched_tokens} -- "
                f"would hit the confirmed upstream chunked-prefill-resume "
                f"bug (see module docstring) rather than fitting in one "
                f"scheduling chunk."
            )
            continue

        orig_len = len(full_prompt_token_ids)
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        prompt = TokensPrompt(prompt_token_ids=pruned_token_ids, cache_salt=request_id)
        # real_id (rewritten) is required here -- model_runner.py's
        # pruning_registry.get(req_id) lookup runs in the Worker process
        # against the WORKER's own self.input_batch.req_ids, which reflects
        # this rewritten id, not the original request_id. See pruner.py's
        # docstring for the full history of this specific requirement.
        real_id = llm.llm_engine.add_request(request_id, prompt, sampling_params)
        llm.llm_engine.collective_rpc(
            "register_prune_record", args=(real_id, kept_positions, orig_len)
        )

        # Keyed by the ORIGINAL request_id, not real_id -- see this
        # function's docstring; this is what step()'s RequestOutputs use.
        id_to_sample[request_id] = sample
        keep_stats[request_id] = (len(kept_positions), orig_len)

    return id_to_sample, keep_stats, num_skipped


def run_experiment(exp_id: str, exp_cfg: dict, args) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM
    from vllm_patch.config import SpecConfig

    label = exp_cfg["label"]
    keep_percentage = exp_cfg["keep_percentage"]
    is_baseline = keep_percentage is None
    # Which checkpoint this experiment actually generates from -- "target"
    # (Qwen3-8B, P001-P006) or "speculator" (Qwen3-1.7B standalone, "P000"):
    # no pruning machinery involved either way, same as any other
    # is_baseline=True row, just a different model path.
    model_path = args.speculator_model if exp_cfg.get("model") == "speculator" else args.target_model

    print(f"\n{'=' * 70}\n{exp_id}: {label}\n{'=' * 70}")

    samples = load_samples(args.samples, args.max_keep)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    token_lengths = token_length_summary(tok, samples)
    print(
        f"[predict_longbench_v2] {len(samples)} sample(s), rendered prompt "
        f"token lengths: min={token_lengths[0]} p50={percentile(token_lengths, 0.5):.0f} "
        f"p90={percentile(token_lengths, 0.9):.0f} max={token_lengths[-1]}"
    )
    max_num_batched_tokens = resolve_max_num_batched_tokens(
        args.target_max_num_batched_tokens, token_lengths, is_baseline
    )

    # Confirmed on real hardware (2026-07-28): unlike the sibling Gemma4
    # pipeline (native max_model_len 262144, always far larger than any
    # resolved budget below), Qwen3-8B's native max_position_embeddings is
    # only 40960 -- and LongBench v2's longer "short" (<32k-word) documents
    # can tokenize past that (this run's own p90/max: 36865/38274 tokens).
    # vLLM's ModelConfig validation rejects max_model_len exceeding the
    # checkpoint's own derived native value outright (VLLM_ALLOW_LONG_
    # MAX_MODEL_LEN=1 would bypass it, but is NOT safe for a RoPE model like
    # Qwen3 -- positions beyond the trained range produce NaN, per vLLM's
    # own error message -- so not used here). Clamp BOTH max_num_batched_
    # tokens and max_model_len to the native ceiling -- a single request can
    # never need a batched-token budget larger than the model's own max
    # context anyway, so clamping only max_model_len while leaving
    # max_num_batched_tokens larger would just be internally inconsistent,
    # not a real fix.
    native_max_model_len = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True
    ).max_position_embeddings
    if native_max_model_len is not None and max_num_batched_tokens > native_max_model_len:
        print(
            f"[predict_longbench_v2] max_num_batched_tokens "
            f"({max_num_batched_tokens}) exceeds {model_path}'s native "
            f"max_position_embeddings ({native_max_model_len}) -- clamping "
            f"down to it. Samples whose full prompt (P001) or kept-token "
            f"count (P002-P006) exceeds this will be skipped and reported, "
            f"same as any other over-budget sample (see "
            f"submit_baseline_requests/submit_pruned_requests)."
        )
        max_num_batched_tokens = native_max_model_len

    # Explicitly cap max_model_len at what this run actually needs (resolved
    # prompt budget + generation length) rather than leaving it at the
    # model's native default -- confirmed on real hardware for the sibling
    # Gemma4 pipeline (2026-07-24): vLLM's own startup sanity check
    # (_check_enough_kv_cache_memory) requires enough KV cache to serve at
    # least ONE request at max_model_len, regardless of whether this run
    # will ever actually submit one that long -- leaving it at a much larger
    # native default demands far more KV cache than necessary. Also clamped
    # to native_max_model_len above (not just floored by it) -- residual
    # risk, not fully resolved here: a sample whose prompt lands exactly at
    # the clamped max_num_batched_tokens ceiling leaves zero room within
    # max_model_len for args.max_tokens of actual generation, which would
    # surface as a separate, narrower per-request scheduling failure rather
    # than this startup-time one; not expected to be common given the
    # auto-sizing margin in resolve_max_num_batched_tokens, but not
    # independently verified against every sample in the full dataset.
    max_model_len = max_num_batched_tokens + args.max_tokens
    if native_max_model_len is not None:
        max_model_len = min(max_model_len, native_max_model_len)

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=False,  # required for RequestOutput.metrics (TTFT)
        gpu_memory_utilization=args.target_gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        # enable_chunked_prefill deliberately left unset (model's own
        # supported default, on) -- see module docstring items 3 for why
        # this deviates from EXPERIMENT_PLAN.md's literal enable_chunked_
        # prefill=False mandate.
    )
    if not is_baseline:
        llm_kwargs["worker_cls"] = "vllm_patch.worker.SpecPrefillWorker"

    t_engine = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[predict_longbench_v2] engine constructed in {time.time() - t_engine:.1f}s")

    proposer = None
    speculator_device = None
    head_dim = None
    spec_config = None
    if not is_baseline:
        import torch
        from vllm.config import ModelConfig, VllmConfig
        from vllm_patch.proposer import SpecPrefillProposer

        if args.speculator_device is not None:
            speculator_device = torch.device(args.speculator_device)
        elif torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            speculator_device = torch.device("cuda:1")
        else:
            speculator_device = torch.device(args.device)
            print(
                f"WARNING: only {torch.cuda.device_count()} GPU(s) visible -- "
                f"speculator will share the target's GPU."
            )
        if speculator_device.type == "cuda":
            # get_model()'s weight placement follows torch's *current* CUDA
            # device, not an explicit device arg -- see
            # validate_runner_integration.py's confirmed finding.
            torch.cuda.set_device(speculator_device)

        speculator_model_config = ModelConfig(
            model=args.speculator_model, trust_remote_code=True, dtype="bfloat16"
        )
        base_vllm_config = VllmConfig(model_config=speculator_model_config)
        proposer = SpecPrefillProposer(
            base_vllm_config=base_vllm_config,
            speculator_model_config=speculator_model_config,
            device=speculator_device,
        )
        head_dim = proposer._speculator_layers[0].head_dim

        spec_config = SpecConfig(
            keep_strategy="percentage",
            keep_kwargs={
                "chunk": True,
                "chunk_size": CHUNK_SIZE,
                "percentage": keep_percentage,
            },
            look_ahead_cnt=LOOK_AHEAD_CNT,
            pool_kernel_size=POOL_KERNEL_SIZE,
        )

    try:
        for rep in range(1, args.reps + 1):
            t0 = time.time()
            if is_baseline:
                id_to_sample, num_skipped = submit_baseline_requests(
                    llm, samples, tok, args.max_tokens, max_num_batched_tokens
                )
                keep_stats: dict = {}
            else:
                id_to_sample, keep_stats, num_skipped = submit_pruned_requests(
                    llm, samples, tok, proposer, spec_config, speculator_device,
                    head_dim, args.max_tokens, max_num_batched_tokens,
                )

            outputs_by_id = drive_engine_to_completion(llm.llm_engine)
            elapsed = time.time() - t0

            out_lens, ttfts = [], []
            finish_counts = {"stop": 0, "length": 0, "other": 0}
            predictions = []
            for req_id, sample in id_to_sample.items():
                output = outputs_by_id.get(req_id)
                if output is None:
                    print(f"WARNING: no output captured for sample id={sample['id']!r} "
                          f"(request_id={req_id!r}) -- skipping from predictions.")
                    continue
                completion = output.outputs[0]
                out_lens.append(len(completion.token_ids))
                finish_counts[completion.finish_reason if completion.finish_reason in
                              finish_counts else "other"] += 1
                # NOT (first_token_ts - arrival_time): confirmed on real
                # hardware (2026-07-23) that this produces garbage (a huge
                # negative number) -- arrival_time is a wall-clock timestamp
                # but first_token_ts is an "engine core timestamp (monotonic)"
                # (vllm/v1/metrics/stats.py:207-214), a different clock with
                # an arbitrary epoch. Use first_token_latency instead --
                # vLLM's own internal metric, already computed correctly on
                # a single consistent (wall-clock) basis via
                # IterationStats._time_since(arrival_time)
                # (vllm/v1/metrics/stats.py:349-351, 369-371).
                if output.metrics is not None and output.metrics.first_token_latency:
                    ttfts.append(output.metrics.first_token_latency * 1000)
                predictions.append({"id": sample["id"], "pred": completion.text})

            if rep == args.reps:
                pred_path = OUT_DIR / f"{exp_id}_predictions.jsonl"
                with open(pred_path, "w", encoding="utf-8") as f:
                    for row in predictions:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[predict_longbench_v2] wrote {len(predictions)} predictions -> {pred_path}")

            out_lens_sorted = sorted(out_lens)
            ttfts_sorted = sorted(ttfts)
            actual_keep_rates = [n / o for n, o in keep_stats.values() if o > 0]

            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "exp_id": exp_id, "label": label,
                "keep_percentage": keep_percentage,
                "chunk_size": CHUNK_SIZE if not is_baseline else None,
                "look_ahead_cnt": LOOK_AHEAD_CNT if not is_baseline else None,
                "pool_kernel_size": POOL_KERNEL_SIZE if not is_baseline else None,
                "target_gpu_memory_utilization": args.target_gpu_memory_utilization,
                "target_max_num_batched_tokens": max_num_batched_tokens,
                "rep": rep, "seed": 0, "max_tokens": args.max_tokens,
                "num_samples": len(id_to_sample), "num_skipped_too_large": num_skipped,
                "elapsed_time": elapsed,
                "requests_per_second": len(id_to_sample) / elapsed if elapsed > 0 else None,
                "actual_keep_rate_mean": statistics.mean(actual_keep_rates) if actual_keep_rates else None,
                "ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
                "ttft_p50_ms": percentile(ttfts_sorted, 0.5),
                "ttft_p90_ms": percentile(ttfts_sorted, 0.9),
                "out_len_mean": statistics.mean(out_lens) if out_lens else None,
                "out_len_stdev": statistics.stdev(out_lens) if len(out_lens) > 1 else 0.0,
                "out_len_p50": percentile(out_lens_sorted, 0.5),
                "out_len_p90": percentile(out_lens_sorted, 0.9),
                "out_len_max": out_lens_sorted[-1] if out_lens_sorted else None,
                "finish_stop": finish_counts["stop"],
                "finish_length": finish_counts["length"],
                "finish_other": finish_counts["other"],
            }
            append_csv_row(row)
            tag = f"{exp_id}_rep{rep}"
            with open(OUT_DIR / f"{tag}.json", "w", encoding="utf-8") as f:
                json.dump({**row, "out_lens": out_lens}, f, indent=2, default=str)

            print(
                f"[predict_longbench_v2] rep {rep}/{args.reps}: "
                f"{row['num_samples']} samples in {elapsed:.1f}s "
                f"({row['requests_per_second']:.2f} req/s), "
                f"skipped={num_skipped}, "
                f"ttft_mean={row['ttft_mean_ms']}, "
                f"actual_keep_rate_mean={row['actual_keep_rate_mean']}"
            )
    finally:
        del llm
        if proposer is not None:
            del proposer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", help="Comma-separated experiment IDs (see --list)")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--target-model", default=os.environ.get("QWEN3_MODEL_PATH"))
    parser.add_argument("--speculator-model", default=os.environ.get("QWEN3_1_7B_MODEL_PATH"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speculator-device", default=None)
    parser.add_argument("--target-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--target-max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=16,
                         help="Generation cap -- LongBench v2 only needs a letter answer.")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--max-keep", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for exp_id, cfg in PRUNE_EXPERIMENTS.items():
            print(f"{exp_id}: {cfg['label']} (keep_percentage={cfg['keep_percentage']}, "
                  f"model={cfg.get('model', 'target')})")
        return

    if args.output_dir is not None:
        global OUT_DIR, CSV_PATH
        OUT_DIR = args.output_dir
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CSV_PATH = OUT_DIR / "all_runs.csv"

    if not args.exp:
        parser.error("--exp is required (or use --list)")

    ensure_csv_header()
    exp_ids = [x.strip() for x in args.exp.split(",")]
    failed_exp_ids = []
    for exp_id in exp_ids:
        if exp_id not in PRUNE_EXPERIMENTS:
            print(f"Unknown experiment ID: {exp_id!r} (see --list)")
            sys.exit(1)
        exp_cfg = PRUNE_EXPERIMENTS[exp_id]
        # --target-model is only required for experiments that actually
        # generate from the target checkpoint -- "P000" runs entirely off
        # --speculator-model and needs no target model at all.
        if exp_cfg.get("model", "target") == "target" and not args.target_model:
            parser.error(f"{exp_id} requires --target-model or $QWEN3_MODEL_PATH")
        # --speculator-model is required either for pruning (P002-P006) or
        # for a standalone speculator experiment like "P000".
        if (exp_cfg["keep_percentage"] is not None or exp_cfg.get("model") == "speculator") \
                and not args.speculator_model:
            parser.error(f"{exp_id} requires --speculator-model or $QWEN3_1_7B_MODEL_PATH")
        try:
            run_experiment(exp_id, PRUNE_EXPERIMENTS[exp_id], args)
        except Exception as e:
            import traceback
            print(f"!!! experiment {exp_id} FAILED: {e}")
            traceback.print_exc()
            failed_exp_ids.append(exp_id)

    if failed_exp_ids:
        print(f"\n{len(failed_exp_ids)} experiment(s) failed: {failed_exp_ids}")
        sys.exit(1)


if __name__ == "__main__":
    main()
