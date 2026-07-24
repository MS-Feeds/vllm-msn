#!/usr/bin/env python3
"""Repro/diagnostic for the "shared-budget-induced multi-chunk prefill"
hypothesis discussed for the P002-P006 garbled-output investigation.

**Status: NOT executed here -- no GPU on this machine (same caveat every
other validate_*.py/predict_longbench_v2.py script in this directory
carries). Written to be run on the GPU node. Nothing below is "confirmed
on real hardware" yet -- that's exactly what running this determines.**

What this checks, concretely: predict_longbench_v2.py's P002-P006 sweeps
submit every sample's pruned request to the engine ~simultaneously (see
that script's own module docstring), all sharing ONE per-step token
budget (`max_num_batched_tokens`). vllm/v1/core/sched/scheduler.py's
`token_budget -= num_new_tokens` (confirmed against this fork's actual
source) means a request can get only a PARTIAL prefill chunk scheduled in
step 1 -- not because ITS OWN prompt exceeds the budget (already checked
per-request before submission, see submit_pruned_requests), but because
other concurrently-running requests used up the shared pool first.

validate_runner_integration.py's module docstring already documents a
CONFIRMED real-hardware bug, reproduced with PLAIN STOCK vLLM (no
spec_prefill code involved): a request whose prefill spans >1 scheduler
step never actually resumes past its first chunk -- the scheduler treats
it as finished and starts sampling against an incomplete KV cache. That
finding was for a single request individually too long for the budget.
The open question this script answers: does the SAME failure fire when a
request is split across steps purely from losing the shared-budget race
against OTHER concurrent requests (predict_longbench_v2.py's actual
scenario), which was never specifically tested?

Method: submit a handful of real LongBench v2 samples through the exact
production pruning path (`predict_longbench_v2.submit_pruned_requests`,
imported and called unmodified -- not reimplemented here, so this repro
can't diverge from what P002-P006 actually do), with `--target-max-num
-batched-tokens` set deliberately small relative to the sample count so
the shared budget is very likely exhausted before every sample's full
(individually-small) pruned prompt fits in step 1. Drives via
`worker_cls=diag_worker.DiagnosticSpecPrefillWorker`, which logs, every
step, every request's (num_computed_before, num_scheduled_this_step,
whether its prefill is complete after this step, and vLLM's own
`discard_request_mask` value for it) to a JSONL file -- see diag_worker.py
for why a file, not collective_rpc, and why this can't just be
monkeypatched from the driver process.

After the run, this script:
  1. Reports whether budget contention was actually forced (i.e. did any
     request really take >1 step) -- if not, the repro didn't create the
     target scenario at all; increase --num-samples or decrease
     --target-max-num-batched-tokens and rerun.
  2. For every request that DID span >1 step, checks whether
     `discard_request_mask` was ever False on a step where its prefill was
     still incomplete afterward -- that's the bug's exact signature (vLLM
     about to sample/emit output for a request it should still be
     silently discarding). Flags CONFIRMED for those.
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

    multi_step_requests = []
    confirmed_bug_requests = []

    for original_id, sample in id_to_sample.items():
        real_id = original_to_real.get(original_id)
        if real_id is None:
            continue
        rows = rows_by_req[real_id]
        num_steps = len(rows)
        if num_steps <= 1:
            continue
        multi_step_requests.append((original_id, real_id, num_steps))

        # The bug's exact signature: on some step, this request's prefill is
        # NOT complete afterward, yet discard_request_mask is False (vLLM
        # about to sample/emit for it anyway instead of correctly
        # withholding output until prefill actually finishes).
        bad_steps = [
            r for r in rows
            if not r["prefill_complete_after_this_step"] and not r["discard_request_mask"]
        ]
        if bad_steps:
            confirmed_bug_requests.append((original_id, real_id, bad_steps))

    print(f"\n{'=' * 70}\nDiagnostic results\n{'=' * 70}")
    print(f"Requests with a diag-log entry: {len(rows_by_req)}")
    print(f"Requests that spanned >1 scheduler step: {len(multi_step_requests)}")
    if not multi_step_requests:
        print(
            "\n[repro] Budget contention was NOT forced -- every request's "
            "prefill fit in a single step. This repro didn't exercise the "
            "scenario under test. Rerun with more --num-samples and/or a "
            "smaller --target-max-num-batched-tokens."
        )
        return

    print(f"Requests where discard_request_mask=False on an INCOMPLETE-prefill "
          f"step (bug signature): {len(confirmed_bug_requests)}")

    if not confirmed_bug_requests:
        print(
            "\n[repro] Contention was forced (some requests took >1 step), but "
            "discard_request_mask correctly stayed True until prefill actually "
            "finished for all of them -- no evidence of the "
            "never-resumes-past-first-chunk bug firing here. The garbled "
            "output has some OTHER cause; the untested "
            "PruneRecord.positions_for_step multi-chunk branch (never "
            "confirmed correct on real hardware either) is the next thing to "
            "check directly, e.g. by comparing kept token IDENTITY (not just "
            "count) against what was registered."
        )
    else:
        print(f"\n[repro] BUG SIGNATURE CONFIRMED for {len(confirmed_bug_requests)} "
              f"request(s). Sampled details:\n")
        for original_id, real_id, bad_steps in confirmed_bug_requests[:5]:
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

    # Requests that took >1 step but were NOT flagged -- useful negative
    # signal either way (proves contention was forced without the bug
    # firing for them specifically, or shows partial firing).
    clean_multi_step = [
        (oid, rid, n) for oid, rid, n in multi_step_requests
        if oid not in {c[0] for c in confirmed_bug_requests}
    ]
    print(f"\nMulti-step requests WITHOUT the bug signature: {len(clean_multi_step)}")
    for original_id, real_id, num_steps in clean_multi_step[:5]:
        sample = id_to_sample[original_id]
        output = outputs_by_id.get(original_id)
        text = output.outputs[0].text if output is not None else "<no output captured>"
        print(f"  sample id={sample['id']!r}: {num_steps} steps, "
              f"repeated-bigram ratio={repeated_bigram_ratio(text):.2f}, "
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
    parser.add_argument("--target-gpu-memory-utilization", type=float, default=0.6)
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

    # No special selection logic -- just the first --num-samples after
    # loading. Pruned length isn't known until after the (expensive)
    # speculator scoring pass runs, so this can't pre-filter for
    # "individually small but collectively over budget" without running
    # pruning twice. Rely instead on --target-max-num-batched-tokens being
    # set small relative to --num-samples (the post-submission report below
    # confirms whether contention was actually achieved either way).
    samples = load_samples(args.samples, max_keep=args.num_samples)
    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    token_lengths = token_length_summary(tok, samples)
    print(f"[repro] {len(samples)} sample(s), full (unpruned) rendered prompt "
          f"token lengths: min={token_lengths[0]} max={token_lengths[-1]}")
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
