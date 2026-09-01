#!/usr/bin/env python3
"""GPU-node validation for `vllm_patch/sparse_target_runner.py`'s
`_build_attention_metadata` override -- the block-table-gather mechanism
that restricts decode-step attention to a speculator-chosen subset of an
otherwise-fully-resident KV cache. Run this AFTER
`validate_resumable_session.py` passes (confirms the persistence foundation
this builds on) and BEFORE trusting any real sparse-attention experiment
sweep.

**Deliberately does NOT use the resumable-session mechanism** --
`validate_resumable_session.py` already validates that separately; the
sparse-attention restriction itself applies to ANY request's own decode
phase, resumable or not, so this script uses a single, ordinary one-shot
request (`add_request` + a normal `TokensPrompt`) to keep this validation
focused on just the ONE new mechanism it exists to check, per this
pipeline's established "validate one new thing at a time" discipline (see
`validate_proposer.py`, `validate_runner_integration.py` for the same
pattern applied to the other pipeline's own new pieces).

**Why a behavioral "needle in a haystack" test, not a direct tensor
inspection**: this pipeline doesn't have infrastructure for reading back
raw attention weights (unlike `kv_cache_utils.py`'s Q/K read-back, which is
built for the SPECULATOR's scoring purposes, not for post-hoc inspecting
what the TARGET's own decode step actually attended to). Instead, this
verifies the mechanism the same way `EXPERIMENT_PLAN.md`'s own verification
section proposes: construct a context where a specific fact is only
findable in one specific block, then confirm the model's actual generated
answer changes depending on whether that block is included in the
registered selection. Concretely:

  A. Baseline sanity: with EVERY block selected (degenerate to full
     attention, `compute_sparse_gather_view`'s own documented no-op case),
     the model should correctly answer a question whose answer is only in
     one specific "needle" block -- confirms the plumbing (registry,
     worker_cls wiring, per-layer metadata patching) doesn't ITSELF break
     normal generation, before testing whether restriction changes
     anything.
  B. Exclusion: with a selection that omits the needle's block, the model
     should NOT reliably produce the correct answer -- direct evidence
     decode-step attention was genuinely restricted away from that block's
     content, not just that the selection was recorded somewhere inert.
  C. Re-inclusion: with a selection that (differently from A, to rule out
     "A only worked because it's literally the same as no restriction at
     all") explicitly includes just the needle's block plus a couple of
     others, the model should correctly answer again -- confirms inclusion
     genuinely restores access, not just that exclusion breaks things in
     some unrelated way (e.g. a crash or garbage output regardless of which
     block is chosen).

Usage:
    python3 validate_sparse_attention.py --model $LLAMA31_8B_MODEL_PATH
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Same reasoning as validate_resumable_session.py's identical default --
# this script doesn't import proposer.py/pruner.py, so nothing else sets
# this for it.
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

NEEDLE_FACT = "The secret code is XYZQR7."
NEEDLE_QUESTION = "\n\nQuestion: What is the secret code? Answer with just the code.\nAnswer:"
FILLER_SENTENCE = (
    "The weather today is mild with a gentle breeze from the west. "
)


def build_context_with_needle(tok, block_size: int, num_filler_blocks_each_side: int):
    """Builds a context with NEEDLE_FACT embedded after
    `num_filler_blocks_each_side` blocks worth of filler, followed by more
    filler after it -- returns (full_token_ids, needle_block_index,
    all_block_indices) so the caller can construct "everything except the
    needle's block" and "just the needle's block plus a couple of others"
    selections precisely, rather than guessing at tokenization boundaries.

    Uses real block_size (queried from the engine, not assumed) to size the
    filler so the needle reliably lands inside its OWN block, not spanning
    a boundary ambiguously -- pads with extra filler tokens before/after to
    center it within a block."""
    filler_ids = tok.encode(FILLER_SENTENCE, add_special_tokens=False)
    needle_ids = tok.encode(f" {NEEDLE_FACT} ", add_special_tokens=False)

    prefix_blocks_ids = []
    while len(prefix_blocks_ids) < num_filler_blocks_each_side * block_size:
        prefix_blocks_ids.extend(filler_ids)
    prefix_blocks_ids = prefix_blocks_ids[: num_filler_blocks_each_side * block_size]

    # Pad prefix so the needle starts near a block boundary -- not exact
    # (tokenization isn't guaranteed to align perfectly), but close enough
    # that the needle's own span is very likely contained in one or two
    # blocks near a known index, which the caller resolves precisely below
    # using the ACTUAL token position, not an assumed one.
    needle_start = len(prefix_blocks_ids)

    suffix_blocks_ids = []
    while len(suffix_blocks_ids) < num_filler_blocks_each_side * block_size:
        suffix_blocks_ids.extend(filler_ids)
    suffix_blocks_ids = suffix_blocks_ids[: num_filler_blocks_each_side * block_size]

    full_ids = prefix_blocks_ids + needle_ids + suffix_blocks_ids
    needle_end = needle_start + len(needle_ids)

    needle_block_indices = sorted(
        {needle_start // block_size, (needle_end - 1) // block_size}
    )
    total_blocks = -(-len(full_ids) // block_size)
    all_block_indices = list(range(total_blocks))

    return full_ids, needle_block_indices, all_block_indices


def run_one_probe(llm_engine, tok, request_id, context_ids, question_text, sampling_params,
                   selected_block_indices, block_size):
    """Registers a sparse selection covering exactly `selected_block_indices`
    (one representative position per block -- compute_sparse_gather_view
    only needs `position // block_size` to land in the right block, doesn't
    need every position in it) BEFORE add_request, per
    sparse_selection_registry.py's documented ordering requirement, then
    drives the request to completion and returns the generated text."""
    from vllm.inputs import TokensPrompt

    representative_positions = [idx * block_size for idx in selected_block_indices]
    llm_engine.collective_rpc(
        "register_sparse_selection", args=(request_id, representative_positions)
    )

    question_ids = tok.encode(question_text, add_special_tokens=False)
    full_prompt_ids = context_ids + question_ids
    prompt = TokensPrompt(prompt_token_ids=full_prompt_ids)
    real_id = llm_engine.add_request(request_id, prompt, sampling_params)
    assert real_id == request_id, (
        f"expected request_id={request_id!r} verbatim, got {real_id!r} -- "
        f"VLLM_DISABLE_REQUEST_ID_RANDOMIZATION must be set (this module "
        f"sets it at import time)."
    )

    last_output = None
    while llm_engine.has_unfinished_requests():
        for output in llm_engine.step():
            if output.request_id == request_id:
                last_output = output

    llm_engine.collective_rpc("discard_sparse_selection", args=(request_id,))

    if last_output is None:
        return None
    return last_output.outputs[0].text


def _str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Defaults to $LLAMA31_8B_MODEL_PATH.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--num-filler-blocks-each-side", type=int, default=4)
    parser.add_argument("--enforce-eager", type=_str2bool, default=True,
                         help="true (default, this script's original behavior) or false -- runs the needle test "
                              "with CUDA graphs on, to behaviorally confirm the sparse restriction is still "
                              "correct under graph replay (not just that block_table/seq_lens LOOK shrunk -- see "
                              "sparse_decode_microbench.py's --enforce-eager help for the specific, unverified "
                              "capture-time-dummy-run risk this is meant to catch).")
    parser.add_argument("--max-model-len", type=int, default=4096,
                         help="Explicit, not left at vLLM's default -- every OTHER script in this benchmark set "
                              "(sparse_decode_microbench.py, gpu_vs_host_timing.py, ncu_kv_bytes_probe.py, "
                              "diagnose_h1_metadata.py) sets this explicitly and none of them have hit the "
                              "get_supported_tasks() engine-construction hang this script hit on real hardware "
                              "(LLM.__init__ -> get_supported_tasks -> call_utility -> future.result() never "
                              "returning) -- this script previously left it at vLLM's default (which for "
                              "Llama-3.1-8B means ~131072), the one identified difference worth testing directly "
                              "rather than guessing further. The needle test's own context is small (a handful "
                              "of filler blocks + one short question), so 4096 is generous headroom, not a tight fit.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--kv-granularity", type=int, default=16,
                         help="Explicit block_size, matching every other script in this set (was previously left "
                              "at vLLM's default here).")
    args = parser.parse_args()

    model_path = args.model or os.environ.get("LLAMA31_8B_MODEL_PATH")
    if not model_path:
        parser.error("--model or $LLAMA31_8B_MODEL_PATH is required")

    from vllm import LLM, SamplingParams

    print(f"[validate_sparse_attention] constructing target engine (worker_cls="
          f"SparseTargetWorker) from {model_path}, enforce_eager={args.enforce_eager}")
    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
        enable_prefix_caching=True,
        worker_cls="vllm_patch.sparse_target_runner.SparseTargetWorker",
        gpu_memory_utilization=args.gpu_memory_utilization,
        async_scheduling=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.kv_granularity,
    )
    # Same reasoning as predict_scbench.py / proposer.py: this check is
    # text-only, but a natively multimodal checkpoint still reserves an
    # encoder cache and PROFILES it -- 36s and several GiB of a budget this
    # script deliberately keeps small. Zeroing the modality limits leaves
    # `active_modalities` empty, so the profiling never runs.
    from vllm_patch.model_structure import has_multimodal_tower

    if has_multimodal_tower(model_path):
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}
        print("[validate_sparse_attention] multimodal checkpoint; zeroing "
              "modality limits (text-only check)")
    llm = LLM(**llm_kwargs)
    llm_engine = llm.llm_engine
    tok = llm.get_tokenizer()

    cc = llm.llm_engine.vllm_config.compilation_config
    print(f"[validate_sparse_attention] resolved: enforce_eager={llm.llm_engine.vllm_config.model_config.enforce_eager} "
          f"cudagraph_mode={cc.cudagraph_mode} "
          f"cudagraph_capture_sizes={list(cc.cudagraph_capture_sizes) if cc.cudagraph_capture_sizes else []}")

    block_size = llm.llm_engine.vllm_config.cache_config.block_size
    print(f"[validate_sparse_attention] engine block_size={block_size}")

    context_ids, needle_block_indices, all_block_indices = build_context_with_needle(
        tok, block_size, args.num_filler_blocks_each_side
    )
    print(
        f"[validate_sparse_attention] built context: {len(context_ids)} tokens, "
        f"{len(all_block_indices)} blocks total, needle in block(s) "
        f"{needle_block_indices}"
    )

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    # Step A: baseline sanity, every block selected (degenerate to full
    # attention per compute_sparse_gather_view's own documented behavior).
    text_a = run_one_probe(
        llm_engine, tok, "validate-sparse-A-full",
        context_ids, NEEDLE_QUESTION, sampling_params,
        selected_block_indices=all_block_indices, block_size=block_size,
    )
    found_a = "XYZQR7" in (text_a or "")
    print(f"[validate_sparse_attention] Step A (full attention): generated={text_a!r} "
          f"-- contains code: {found_a} (expected True)")
    if not found_a:
        print(
            "[validate_sparse_attention] Step A: FAIL -- the model couldn't answer "
            "correctly even with every block selected (i.e. no real restriction at "
            "all). This means either the plumbing itself is broken (registry/"
            "worker_cls wiring/metadata patching), or the needle wasn't placed "
            "correctly in the context -- fix this before Steps B/C can mean "
            "anything, since they depend on A being a working, correct baseline."
        )
        sys.exit(1)

    # Step B: exclude the needle's block(s) -- select everything EXCEPT it.
    excluding_needle = [b for b in all_block_indices if b not in needle_block_indices]
    text_b = run_one_probe(
        llm_engine, tok, "validate-sparse-B-exclude",
        context_ids, NEEDLE_QUESTION, sampling_params,
        selected_block_indices=excluding_needle, block_size=block_size,
    )
    found_b = "XYZQR7" in (text_b or "")
    print(f"[validate_sparse_attention] Step B (needle block EXCLUDED): "
          f"generated={text_b!r} -- contains code: {found_b} (expected False)")

    # Step C: re-include the needle's block(s) plus a couple of neighbors
    # (deliberately NOT identical to Step A's "everything" selection, to
    # rule out "A only worked because it was literally unrestricted").
    including_needle = sorted(set(needle_block_indices) | set(all_block_indices[:2]))
    text_c = run_one_probe(
        llm_engine, tok, "validate-sparse-C-include",
        context_ids, NEEDLE_QUESTION, sampling_params,
        selected_block_indices=including_needle, block_size=block_size,
    )
    found_c = "XYZQR7" in (text_c or "")
    print(f"[validate_sparse_attention] Step C (needle block RE-INCLUDED, "
          f"partial selection {including_needle}): generated={text_c!r} -- "
          f"contains code: {found_c} (expected True)")

    print()
    if found_b:
        print(
            "[validate_sparse_attention] OVERALL: WARNING -- Step B still found the "
            "code despite the needle's block being excluded from the selection. "
            "Either the gather isn't actually restricting attention (metadata patch "
            "not taking effect), or the model is recalling the fact from its own "
            "prefill-time processing rather than genuinely needing decode-time "
            "attention to it (plausible for a short, salient fact -- consider a "
            "longer/less memorable needle if this persists) -- do not trust a real "
            "sweep until this is understood."
        )
    elif not found_c:
        print(
            "[validate_sparse_attention] OVERALL: WARNING -- Step C did NOT recover "
            "the code even with the needle's block explicitly re-included. This "
            "suggests the gather mechanism is broken in a way that doesn't just "
            "restrict attention, but corrupts it (wrong blocks selected, wrong "
            "seq_len, garbled generation) -- inspect Step C's generated text above "
            "for signs of corruption (garbage tokens, repetition) vs. a clean but "
            "wrong answer."
        )
    else:
        print(
            "[validate_sparse_attention] OVERALL: PASS -- the model answers "
            "correctly with the needle's block included (A, C) and fails to when "
            "it's excluded (B), consistent with decode-step attention genuinely "
            "being restricted to the registered block selection."
        )


if __name__ == "__main__":
    main()
