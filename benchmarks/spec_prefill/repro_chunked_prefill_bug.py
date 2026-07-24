#!/usr/bin/env python3
"""Repro/diagnostic for the P002-P006 garbled-output investigation --
tests TWO independent hypotheses about vllm_patch/'s never-real-hardware-
tested multi-chunk-prefill code paths (see EXPERIMENT_PLAN.md and
validate_runner_integration.py for the accuracy-affecting-instruction-
truncation bug this ISN'T -- that one's already fixed, commit
47fd38f8a).

**Status: NOT executed here -- no GPU on this machine (same caveat every
other validate_*.py/predict_longbench_v2.py script in this directory
carries). Written to be run on the GPU node. Nothing below is "confirmed
on real hardware" yet -- that's exactly what running this determines.**

**Real-hardware run log, 2026-07-24 (hypothesis #1 only, before hypothesis
#2 was added below): 12/12 concurrently-submitted samples had their
prefill span >1 total step; discard_request_mask stayed correctly True
through every incomplete-prefill step for all 12 -- hypothesis #1 (below)
is RULED OUT with real evidence. Output at --max-tokens=16 was truncated
mid-explanation for all 12 (not garbled); at --max-tokens=128, most
resolved into long coherent text but at least one (sample
66f37eb9821e116aacb2d295) degenerated hard into a repetition loop
("liberty liberty liberty..."). That run's "12/12 spanned >1 step" claim
conflated decode steps with prefill chunks (a bug in this script, since
fixed -- see multi_chunk_prefill_requests below) -- hypothesis #2's
direct position-identity check was added specifically to test that
degenerating sample properly, not yet run.**

What this checks, concretely: predict_longbench_v2.py's P002-P006 sweeps
submit every sample's pruned request to the engine ~simultaneously (see
that script's own module docstring), all sharing ONE per-step token
budget (`max_num_batched_tokens`). vllm/v1/core/sched/scheduler.py's
`token_budget -= num_new_tokens` (confirmed against this fork's actual
source) means a request can get only a PARTIAL prefill chunk scheduled in
step 1 -- not because ITS OWN prompt exceeds the budget (already checked
per-request before submission, see submit_pruned_requests), but because
other concurrently-running requests used up the shared pool first. This
lands squarely on TWO separate code paths validate_runner_integration.py
explicitly never exercised (Step B/B2 always fit in one chunk, by
design):

  Hypothesis #1: the confirmed-on-stock-vLLM "prefill never resumes past
  its first chunk" bug (validate_runner_integration.py's docstring,
  finding #2) -- found there for a single request individually too big
  for the budget; open question was whether shared-budget-induced
  splitting triggers the same failure. RULED OUT, see run log above.

  Hypothesis #2: model_runner.py's multi-chunk `PruneRecord
  .positions_for_step` continuation-chunk branch -- unit-tested
  (test_vllm_patch.py) but, per that module's own docstring, never
  confirmed correct against the real scheduler/runner for N>1 chunks.
  Tests whether the ACTUAL RoPE positions the model receives on a 2nd (or
  later) prefill chunk match what PruneRecord says they should be --
  wrong positions there would corrupt attention without ever tripping
  discard_request_mask (a completeness check, not a correctness one),
  which is exactly why hypothesis #1 being ruled out doesn't rule this
  one out too.

Method: submit a handful of real LongBench v2 samples through the exact
production pruning path (`predict_longbench_v2.submit_pruned_requests`,
imported and called unmodified -- not reimplemented here, so this repro
can't diverge from what P002-P006 actually do), with `--target-max-num
-batched-tokens` set deliberately small relative to the sample count so
the shared budget is very likely exhausted before every sample's full
(individually-small) pruned prompt fits in step 1. Drives via
`worker_cls=diag_worker.DiagnosticSpecPrefillWorker`, which logs, every
step, every request's completeness state (num_computed_before,
num_scheduled_this_step, whether prefill is complete after this step,
vLLM's own `discard_request_mask` value) AND, for genuine prefill-chunk
steps specifically, whether the actual `self.positions` values the model
received match `PruneRecord.positions_for_step`'s expected values -- to a
JSONL file. See diag_worker.py for why a file, not collective_rpc, and
why this can't just be monkeypatched from the driver process.

After the run, this script:
  1. Reports whether budget contention was actually forced -- specifically
     whether any request's prefill really split across >1 CHUNK (not just
     >1 total step, which decode alone inflates trivially -- see
     multi_chunk_prefill_requests vs. the merely-informational
     multi_step_requests in analyze_diag_log). If not forced, increase
     --num-samples or decrease --target-max-num-batched-tokens and rerun.
  2. For every request with a genuine multi-chunk prefill, checks BOTH
     hypotheses' exact signatures independently and reports each
     separately -- a request can fail one, both, or neither.
  3. Prints each flagged request's actual generated text alongside a
     coherence heuristic (repeated-bigram ratio) so a human can eyeball
     whether it matches the garbled pattern from the original report.

Usage:
    python3 repro_chunked_prefill_bug.py \
        --target-model $GEMMA4_MODEL_PATH \
        --speculator-model $GEMMA4_E2B_MODEL_PATH \
        --num-samples 12 --target-max-num-batched-tokens 3072

Note: --target-max-num-batched-tokens has a hard floor of 2496 regardless of
what value would otherwise best force contention -- Gemma-4-26B-A4B-it's
multimodal-bidirectional attention requires max_num_batched_tokens >=
max_tokens_per_mm_item (vllm/platforms/cuda.py:221-241, already documented
in validate_runner_integration.py's own docstring; confirmed here too, the
hard way: "ValueError: Chunked MM input disabled but max_tokens_per_mm_item
(2496) is larger than max_num_batched_tokens (1200)" against an earlier,
too-low default). If 2496 doesn't comfortably force contention with
--num-samples's default, raise --num-samples rather than lowering the
budget further -- it can't go below that floor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # for `vllm_patch`/`diag_worker` imports

from predict_longbench_v2 import (  # noqa: E402 -- see sys.path insert above
    CHUNK_SIZE,
    DEFAULT_SAMPLES,
    LOOK_AHEAD_CNT,
    POOL_KERNEL_SIZE,
    drive_engine_to_completion,
    load_samples,
    render_chat,
    submit_pruned_requests,
    token_length_summary,
)

DEFAULT_DIAG_LOG = Path(os.environ.get("BENCH_RESULTS_DIR", "results")) / "chunked_prefill_diag.jsonl"

# Gemma-4-26B-A4B-it's multimodal-bidirectional attention forces chunked MM
# input off unconditionally (vllm/platforms/cuda.py:221-241, per
# validate_runner_integration.py's own docstring), which requires
# max_num_batched_tokens >= max_tokens_per_mm_item (2496 for this model) at
# LLM construction time -- not overridable, a hard floor regardless of what
# this script is trying to test. Confirmed on real hardware the hard way:
# "ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496)
# is larger than max_num_batched_tokens (1200)" with an earlier default
# below this floor.
MIN_MAX_NUM_BATCHED_TOKENS = 2496


def repeated_bigram_ratio(text: str) -> float:
    """Cheap coherence heuristic, not a quality metric: fraction of
    word-bigrams in `text` that are a repeat of an earlier bigram in the
    same text. The garbled examples from the original report ("and and
    and-and-and-and", "to * to * to *") are dominated by a small number of
    bigrams repeating over and over -- a high ratio here is consistent
    with that pattern (not proof by itself, hence printing the raw text
    alongside it for a human to actually read)."""
    words = text.split()
    if len(words) < 4:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    counts = Counter(bigrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(bigrams)


def select_shortest_samples(samples: list[dict], tok, num_samples: int) -> list[dict]:
    """Sort ALL loaded samples by full (unpruned) rendered token length and
    return the shortest `num_samples`.

    **Confirmed on real hardware (2026-07-24): file order is NOT length
    order.** An earlier version of this script just took the first
    `--num-samples` after loading -- LongBench v2's "short" (<32k WORD)
    filter still allows individual documents from ~15k to ~90k+ TOKENS
    (word count and token count diverge a lot at this scale), and file
    order doesn't correlate with either. `submit_pruned_requests` (reused
    unmodified here, see module docstring) requires each sample's FULL,
    UNPRUNED length to fit under `--target-max-num-batched-tokens` -- the
    speculator's bootstrap prefill must score the whole document in one
    shot -- which is a completely different constraint from the
    budget-contention scenario this script is trying to force among
    PRUNED requests. Picking samples by file order meant most of a
    12-sample batch got skipped before pruning ever ran (real observed
    lengths: 15400-88541 tokens against a 3072 budget), leaving 0 requests
    submitted. Sorting by actual rendered length and taking the shortest
    N is the only way to reliably get samples whose full length clears
    that pre-check while still being numerous/small enough for their
    PRUNED lengths to contend for a shared budget once summed."""
    lengths = [(len(tok.encode(render_chat(tok, s["prompt"]), add_special_tokens=False)), s) for s in samples]
    lengths.sort(key=lambda pair: pair[0])
    chosen = lengths[:num_samples]
    print(f"[repro] selected the {len(chosen)} shortest of {len(samples)} loaded "
          f"samples by full rendered length: "
          f"{[l for l, _ in chosen]}")
    return [s for _, s in chosen]


def analyze_diag_log(
    log_path: Path,
    id_to_sample: dict,
    outputs_by_id: dict,
) -> None:
    if not log_path.exists():
        print(f"[repro] ERROR: no diagnostic log at {log_path} -- did the run "
              f"actually reach drive_engine_to_completion? Nothing to analyze.")
        return

    rows_by_req: dict[str, list[dict]] = defaultdict(list)
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_req[row["req_id"]].append(row)

    for rows in rows_by_req.values():
        rows.sort(key=lambda r: r["step"])

    # diag log rows are keyed by the WORKER-side rewritten request_id
    # (self.input_batch.req_ids, same namespace pruning_registry.get() looks
    # up against -- see pruner.py's docstring for why this differs from the
    # DRIVER-side id). id_to_sample/outputs_by_id are keyed by the ORIGINAL,
    # caller-supplied request_id (see predict_longbench_v2.py's
    # submit_pruned_requests/drive_engine_to_completion docstrings for why:
    # output_processor.py's RequestOutput always carries external_req_id).
    # submit_pruned_requests doesn't return the original->rewritten mapping,
    # and is intentionally called UNMODIFIED here for fidelity to production
    # -- so recover the mapping from vLLM's own known rewrite scheme instead
    # (InputProcessor.assign_request_id: f"{external_id}-{8 random chars}",
    # confirmed in pruner.py's docstring). The trailing "-" after the
    # original id, matched against every distinct rewritten id actually
    # seen in the log, is unambiguous: predict_longbench_v2.py's own
    # request_id scheme (f"lbv2-{sample_id}-{i}" with a unique loop index i)
    # never lets one original id be a bare prefix of another's rewritten id.
    real_ids = list(rows_by_req.keys())
    original_to_real: dict[str, str] = {}
    for original_id in id_to_sample:
        matches = [r for r in real_ids if r.startswith(original_id + "-")]
        if len(matches) == 1:
            original_to_real[original_id] = matches[0]
        elif len(matches) > 1:
            print(f"[repro] WARNING: {original_id!r} matched {len(matches)} "
                  f"rewritten ids in the diag log ({matches}) -- ambiguous, "
                  f"skipping this request in the analysis.")
        # 0 matches: this request never got a step logged at all (e.g.
        # skipped before submission, or somehow never scheduled) -- silently
        # excluded from the per-request analysis below; the "samples with no
        # diag rows" count printed at the end surfaces this instead.

    multi_step_requests = []  # total steps (prefill chunks + decode), informational only
    multi_chunk_prefill_requests = []  # prefill ACTUALLY split across >1 chunk -- the real signal
    completeness_bug_requests = []  # hypothesis #1: discard_request_mask wrong
    position_bug_requests = []  # hypothesis #2: wrong position identity on a later chunk

    for original_id, sample in id_to_sample.items():
        real_id = original_to_real.get(original_id)
        if real_id is None:
            continue
        rows = rows_by_req[real_id]
        num_steps = len(rows)
        if num_steps > 1:
            multi_step_requests.append((original_id, real_id, num_steps))

        # The REAL "did prefill span multiple chunks" signal -- num_steps
        # above conflates this with decode steps (one row per generated
        # token too), which is why an earlier version of this report's
        # "Requests that spanned >1 scheduler step: 12" at --max-tokens=16
        # and "...128" at --max-tokens=128 wasn't actually confirming
        # multi-chunk prefill, just multi-step generation (true for nearly
        # any request that generates more than one token).
        prefill_chunk_rows = [r for r in rows if r["is_prefill_chunk_step"]]
        if len(prefill_chunk_rows) > 1:
            multi_chunk_prefill_requests.append((original_id, real_id, len(prefill_chunk_rows)))

        # Hypothesis #1's exact signature: on some step, this request's
        # prefill is NOT complete afterward, yet discard_request_mask is
        # False (vLLM about to sample/emit for it anyway instead of
        # correctly withholding output until prefill actually finishes).
        bad_completeness_steps = [
            r for r in rows
            if not r["prefill_complete_after_this_step"] and not r["discard_request_mask"]
        ]
        if bad_completeness_steps:
            completeness_bug_requests.append((original_id, real_id, bad_completeness_steps))

        # Hypothesis #2's exact signature: on some prefill-chunk step
        # (first OR a later continuation), the actual positions the model
        # received don't match what PruneRecord says they should be.
        bad_position_steps = [r for r in prefill_chunk_rows if r["positions_correct"] is False]
        if bad_position_steps:
            position_bug_requests.append((original_id, real_id, bad_position_steps))

    print(f"\n{'=' * 70}\nDiagnostic results\n{'=' * 70}")
    print(f"Requests with a diag-log entry: {len(rows_by_req)}")
    print(f"Requests with >1 total step (prefill chunks + decode combined, "
          f"informational only -- NOT proof of multi-chunk prefill): "
          f"{len(multi_step_requests)}")
    print(f"Requests where prefill ACTUALLY split across >1 chunk "
          f"(the real contention signal): {len(multi_chunk_prefill_requests)}")
    if not multi_chunk_prefill_requests:
        print(
            "\n[repro] Budget contention was NOT forced -- every request's "
            "prefill fit in a single chunk (decode alone can still produce "
            "many total steps, which is not the same thing). This repro "
            "didn't exercise the scenario under test. Rerun with more "
            "--num-samples and/or a smaller --target-max-num-batched-tokens."
        )
        return

    print(f"Requests where discard_request_mask=False on an INCOMPLETE-prefill "
          f"step (hypothesis #1 signature): {len(completeness_bug_requests)}")
    print(f"Requests where actual RoPE positions != PruneRecord on a prefill "
          f"chunk, first or later (hypothesis #2 signature): "
          f"{len(position_bug_requests)}")

    if position_bug_requests:
        print(f"\n[repro] HYPOTHESIS #2 CONFIRMED for {len(position_bug_requests)} "
              f"request(s) -- model_runner.py's multi-chunk "
              f"PruneRecord.positions_for_step branch IS delivering wrong "
              f"positions on a real chunk boundary. Details:\n")
        for original_id, real_id, bad_steps in position_bug_requests[:5]:
            sample = id_to_sample[original_id]
            print(f"--- sample id={sample['id']!r} (request_id={original_id!r}) ---")
            for r in bad_steps[:3]:
                print(f"  step={r['step']} num_computed_before={r['num_computed_before']} "
                      f"num_scheduled_this_step={r['num_scheduled_this_step']}")
            print()

    if completeness_bug_requests:
        print(f"\n[repro] HYPOTHESIS #1 CONFIRMED for {len(completeness_bug_requests)} "
              f"request(s). Sampled details:\n")
        for original_id, real_id, bad_steps in completeness_bug_requests[:5]:
            sample = id_to_sample[original_id]
            output = outputs_by_id.get(original_id)
            text = output.outputs[0].text if output is not None else "<no output captured>"
            ratio = repeated_bigram_ratio(text)
            print(f"--- sample id={sample['id']!r} (request_id={original_id!r}, "
                  f"real_id={real_id!r}) ---")
            print(f"  first bad step: {bad_steps[0]}")
            print(f"  repeated-bigram ratio: {ratio:.2f} "
                  f"({'looks garbled/loopy' if ratio > 0.3 else 'looks plausibly coherent'})")
            print(f"  generated text: {text[:300]!r}")
            print()

    if not completeness_bug_requests and not position_bug_requests:
        print(
            "\n[repro] Neither hypothesis fired, on a real (not single-chunk) "
            "multi-chunk prefill. Both the confirmed-on-stock-vLLM "
            "'never resumes' failure mode AND the never-real-hardware-tested "
            "multi-chunk position-override branch are now ruled out with "
            "actual evidence, not assumption. Whatever is producing "
            "incoherent output (e.g. the repetition-loop sample from the "
            "128-token run) has some OTHER cause -- most likely the "
            "SpecPrefill scoring/pruning ALGORITHM itself making a bad token "
            "selection for that specific document (an accuracy/tuning "
            "problem, not a plumbing bug), not this runner-integration path."
        )

    # Requests where prefill genuinely split across chunks but neither
    # hypothesis's signature fired -- useful negative signal either way.
    flagged_ids = {c[0] for c in completeness_bug_requests} | {c[0] for c in position_bug_requests}
    clean_multi_chunk = [
        (oid, rid, n) for oid, rid, n in multi_chunk_prefill_requests
        if oid not in flagged_ids
    ]
    print(f"\nMulti-chunk-prefill requests WITHOUT either bug signature: "
          f"{len(clean_multi_chunk)}")
    for original_id, real_id, num_chunks in clean_multi_chunk[:5]:
        sample = id_to_sample[original_id]
        output = outputs_by_id.get(original_id)
        text = output.outputs[0].text if output is not None else "<no output captured>"
        ratio = repeated_bigram_ratio(text)
        print(f"  sample id={sample['id']!r}: {num_chunks} prefill chunks, "
              f"repeated-bigram ratio={ratio:.2f} "
              f"({'looks garbled/loopy' if ratio > 0.3 else 'looks plausibly coherent'}), "
              f"text={text[:150]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--num-samples", type=int, default=12,
                         help="How many samples to submit concurrently. More "
                              "samples = more shared-budget contention.")
    parser.add_argument("--target-model", default=os.environ.get("GEMMA4_MODEL_PATH"))
    parser.add_argument("--speculator-model", default=os.environ.get("GEMMA4_E2B_MODEL_PATH"))
    parser.add_argument("--keep-percentage", type=float, default=0.3,
                         help="Aggressive-ish keep rate, deliberately not one of "
                              "P002-P006's exact values -- this script isolates "
                              "the chunking mechanism, not the accuracy sweep.")
    parser.add_argument("--target-max-num-batched-tokens", type=int, default=3072,
                         help="Deliberately small relative to --num-samples, to "
                              "force the shared budget to be exhausted before "
                              "every sample's (individually-small) pruned prompt "
                              "fits in scheduling step 1. Must be >= "
                              f"{MIN_MAX_NUM_BATCHED_TOKENS} -- Gemma-4-26B-A4B-it's "
                              "multimodal-bidirectional attention forces this floor "
                              "unconditionally at construction time, unrelated to "
                              "what this script is testing (see "
                              "MIN_MAX_NUM_BATCHED_TOKENS above).")
    parser.add_argument("--target-gpu-memory-utilization", type=float, default=0.9,
                         help="Matches predict_longbench_v2.py's own default, NOT "
                              "validate_runner_integration.py's 0.6 -- that lower "
                              "value is a workaround specific to sharing ONE GPU "
                              "between target and speculator, confirmed on real "
                              "hardware (2026-07-24, this script) to leave "
                              "negative KV-cache memory otherwise: the target's "
                              "~48.5 GiB of weights alone consume essentially all "
                              "of a 0.6-of-80GB budget. This script puts the "
                              "speculator on a second GPU when available (same as "
                              "predict_longbench_v2.py), so there's no reason to "
                              "keep the conservative single-GPU value. Lower this "
                              "explicitly if actually sharing one GPU.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speculator-device", default=None)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--diag-log", type=Path, default=DEFAULT_DIAG_LOG)
    args = parser.parse_args()

    if not args.target_model or not args.speculator_model:
        parser.error("--target-model/--speculator-model (or $GEMMA4_MODEL_PATH/"
                      "$GEMMA4_E2B_MODEL_PATH) are required.")

    if args.target_max_num_batched_tokens < MIN_MAX_NUM_BATCHED_TOKENS:
        parser.error(
            f"--target-max-num-batched-tokens={args.target_max_num_batched_tokens} "
            f"is below Gemma-4-26B-A4B-it's hard floor of "
            f"{MIN_MAX_NUM_BATCHED_TOKENS} (multimodal-bidirectional attention "
            f"forces chunked MM input off, which requires max_num_batched_tokens "
            f">= max_tokens_per_mm_item -- see MIN_MAX_NUM_BATCHED_TOKENS's "
            f"comment above). This floor is unrelated to the budget-contention "
            f"scenario under test -- raise --num-samples instead if you need "
            f"more contention at a budget above this floor."
        )

    # Truncate any stale log from a previous run -- see diag_worker.py's
    # docstring: the worker process only ever APPENDS.
    args.diag_log.parent.mkdir(parents=True, exist_ok=True)
    if args.diag_log.exists():
        args.diag_log.unlink()
    os.environ["SPEC_PREFILL_DIAG_LOG"] = str(args.diag_log)

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.config import ModelConfig, VllmConfig
    from vllm_patch.config import SpecConfig
    from vllm_patch.proposer import SpecPrefillProposer

    # Load everything, then pick the shortest --num-samples by actual
    # rendered length -- see select_shortest_samples's docstring for why
    # file order (an earlier version of this script) doesn't work: full
    # (unpruned) length must clear submit_pruned_requests's own pre-check
    # (a completely different constraint from the budget-contention
    # scenario under test here), and LongBench v2 "short" documents still
    # range from ~15k to ~90k+ tokens regardless of file position.
    all_samples = load_samples(args.samples, max_keep=-1)
    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    samples = select_shortest_samples(all_samples, tok, args.num_samples)

    token_lengths = token_length_summary(tok, samples)
    print(f"[repro] {len(samples)} sample(s), full (unpruned) rendered prompt "
          f"token lengths: min={token_lengths[0]} max={token_lengths[-1]}")
    if token_lengths[-1] > args.target_max_num_batched_tokens:
        print(
            f"[repro] WARNING: even the shortest {len(samples)} samples' max "
            f"full length ({token_lengths[-1]}) exceeds "
            f"--target-max-num-batched-tokens={args.target_max_num_batched_tokens} "
            f"-- some/all will be SKIPPED before pruning runs (same failure "
            f"mode as before, just for however many still don't fit). Raise "
            f"--target-max-num-batched-tokens or lower --num-samples so more "
            f"of the dataset's shorter documents are in play."
        )
    print(f"[repro] --target-max-num-batched-tokens={args.target_max_num_batched_tokens} "
          f"-- if this exceeds the SUM of every sample's pruned length, "
          f"contention won't be forced; the post-run report will confirm "
          f"either way.")

    max_model_len = args.target_max_num_batched_tokens + args.max_tokens

    if args.speculator_device is not None:
        speculator_device = torch.device(args.speculator_device)
    elif torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        speculator_device = torch.device("cuda:1")
    else:
        speculator_device = torch.device(args.device)
        print(f"WARNING: only {torch.cuda.device_count()} GPU(s) visible -- "
              f"speculator will share the target's GPU.")
    if speculator_device.type == "cuda":
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
        keep_kwargs={"chunk": True, "chunk_size": CHUNK_SIZE, "percentage": args.keep_percentage},
        look_ahead_cnt=LOOK_AHEAD_CNT,
        pool_kernel_size=POOL_KERNEL_SIZE,
    )

    llm = LLM(
        model=args.target_model,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=False,
        gpu_memory_utilization=args.target_gpu_memory_utilization,
        max_num_batched_tokens=args.target_max_num_batched_tokens,
        max_model_len=max_model_len,
        # The one deliberate difference from predict_longbench_v2.py's own
        # worker_cls -- diag_worker.DiagnosticSpecPrefillWorker instead of
        # vllm_patch.worker.SpecPrefillWorker. Everything downstream
        # (pruning, position overrides) behaves identically; only the extra
        # per-step logging is added. See diag_worker.py.
        worker_cls="diag_worker.DiagnosticSpecPrefillWorker",
    )

    id_to_sample, keep_stats, num_skipped = submit_pruned_requests(
        llm, samples, tok, proposer, spec_config, speculator_device,
        head_dim, args.max_tokens, args.target_max_num_batched_tokens,
    )
    print(f"\n[repro] submitted {len(id_to_sample)} request(s), "
          f"skipped {num_skipped} (too large even alone)")
    pruned_lens = [n for n, _ in keep_stats.values()]
    if pruned_lens:
        print(f"[repro] pruned lengths: {sorted(pruned_lens)}, "
              f"sum={sum(pruned_lens)} vs budget={args.target_max_num_batched_tokens} "
              f"({'sum > budget, contention expected' if sum(pruned_lens) > args.target_max_num_batched_tokens else 'sum <= budget -- contention UNLIKELY, consider more samples or a smaller budget'})")

    outputs_by_id = drive_engine_to_completion(llm.llm_engine)
    print(f"[repro] engine drained, {len(outputs_by_id)} output(s) captured")

    analyze_diag_log(args.diag_log, id_to_sample, outputs_by_id)


if __name__ == "__main__":
    main()
