"""Real-hardware validation for the SpecPrefill runner integration
(vllm_patch/worker.py, model_runner.py, pruner.py, pruning_registry.py).

Run on the actual GPU node (needs both Gemma-4-26B-A4B-it and
Gemma-4-E2B-it checkpoints, a GPU, and vLLM's full runtime -- none available
on the machine this was written on; carefully reasoned through against this
fork's verified V1 APIs but NOT executed. Expect to iterate against real
errors.)

Usage:
    export HF_TOKEN=<your token>
    source .env_exports.sh   # this directory's local copy (not the shared
                              # one in ../gemma4_moe_benchmarks/)
    python3 validate_runner_integration.py \
        --target-model $GEMMA4_MODEL_PATH \
        --speculator-model $GEMMA4_E2B_MODEL_PATH

**Confirmed gotcha (2026-07-22, real hardware, single 80GB A100)**: the
target `LLM(...)` defaults to `gpu_memory_utilization=0.9`, leaving no room
to also load the speculator standalone in the same process -- CUDA OOM
loading Gemma-4-E2B-it's very first layer. Worked around via
`--target-gpu-memory-utilization` (default here: 0.6) -- lower it further if
the speculator still OOMs. This is a validation-script workaround only; the
real KV-cache budgeting between two concurrently-loaded Gemma-4 models
remains deferred, unsolved work (see EXPERIMENT_PLAN.md).

What this validates, in order:

Step A -- basic wiring: constructs a real `LLM(..., worker_cls=
"vllm_patch.worker.SpecPrefillWorker")` and confirms it loads without
crashing, then runs one ordinary (non-pruned) generation to confirm
`SpecPrefillGPUModelRunner` doesn't break normal requests -- every request
without a `PruneRecord` must behave identically to stock vLLM, since
`gemma4_moe_benchmarks`/`evaluation_pipeline` never opt into `worker_cls`
and are unaffected regardless, but this is still the right sanity check
before trusting anything else here.

Step B -- the actual open question (approved plan's risk #1): does the
`self.positions` view-aliasing that `model_runner.py`'s override depends on
actually hold? Rather than inferring this indirectly from generation
quality, this installs a *diagnostic* hook on the target model's
`Gemma4Attention` layers (same instance-level `types.MethodType` technique
`proposer.py` already uses on the speculator, applied here purely to
observe, not modify, behavior -- see `_install_position_capture_hook`) that
records the exact `positions` tensor each layer receives during its real
forward pass. A pruned request is registered via `pruner.
prune_and_add_request` with known `kept_positions`; if the captured
positions exactly match what was registered, the view-aliasing assumption
holds end-to-end. If they instead match the *stock* contiguous numbering,
the override isn't reaching the model and `model_runner.py`'s override
point needs to move earlier (see its own docstring for what that implies).

Step B2 -- same check as Step B, but for a pruned prompt long enough that
vLLM's chunked-prefill scheduling splits its kept tokens across multiple
engine steps (Step B's own prompt always fits in one step, so it can't catch
a chunk-boundary bug in `model_runner.py`'s `PruneRecord.positions_for_step`
-- see that method's docstring). To actually force multi-step scheduling
from a manageable prompt, `--target-max-num-batched-tokens` (default 3072)
is passed into `LLM(...)` along with `enable_chunked_prefill=True`
explicitly.

**Confirmed gotcha (2026-07-23, real hardware)**: this can't be pushed down
to a small value like 64 the way a non-multimodal model would allow --
Gemma-4-26B-A4B-it has "multimodal-bidirectional attention", and
`vllm/platforms/cuda.py:221-241` unconditionally force-disables chunked MM
input for such models, which then requires `max_num_batched_tokens >=
max_tokens_per_mm_item` (2496 for this model) at `LLM(...)` construction
time -- not overridable via any kwarg, since it's forced by the platform
regardless of what's passed in. 3072 clears that floor with margin.
Consequently Step B2's synthetic prompt (`_build_long_test_prompt_text`)
needs to produce well over 3072 *kept* tokens (not just over
`--target-max-num-batched-tokens`) to still force multiple chunks -- its
default `target_word_count` accounts for this.

**This deliberately diverges from EXPERIMENT_PLAN.md's real-benchmark
config** (`enable_chunked_prefill=False`) -- confirmed against this fork's
scheduler config validation (`vllm/config/scheduler.py`'s
`verify_max_model_len`) that with chunked prefill off, `max_num_batched_tokens`
is required to be `>= max_model_len`, making single-request prefill
all-or-nothing by construction and this multi-step path unreachable to test
under that config at all. Whoever eventually writes the LongBench v2 driver
script should revisit this reachability question then -- vLLM's own default
is chunked-prefill-on, so it likely matters regardless of what this
validation script assumes. Note also: Step B2's longer prompt exercises the
speculator's lookahead path at a scale beyond what `validate_proposer.py`
has verified -- if B2 fails, triage against `validate_proposer.py`-style
speculator diagnostics before assuming the `model_runner.py`/`proposer.py`
fixes under test are wrong.

Step C -- coherence smoke test: generates a few tokens for a heavily-pruned
prompt and just prints the output for a human eyeball check -- not a
substitute for Step B/B2's direct check, but a useful secondary signal
(if Step B/B2 pass but output is obvious garbage, something *else* is wrong,
e.g. the KV-cache slot-mapping assumption in the module docstring).
"""

import argparse
import os
import sys
import types
from functools import partial
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

# collective_rpc serializes whatever's sent to the (separate-process, even
# under UniProcExecutor -- confirmed on real hardware via the
# "(EngineCore pid=...)" log prefix) Worker; raw function objects aren't
# serializable by default (security restriction against arbitrary code
# execution over that channel). This script's own diagnostic hooks
# (_install_position_capture_hook etc.) are plain functions, not named
# Worker methods -- acceptable to relax for a throwaway, locally-run
# validation script; NOT something to set for production use. Must be set
# before vLLM reads it, so this is as early as possible.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
from vllm import LLM, SamplingParams
from vllm.config import ModelConfig, VllmConfig

from vllm_patch.config import SpecConfig
from vllm_patch.proposer import SpecPrefillProposer
from vllm_patch.pruner import prune_and_add_request

_CAPTURE_ATTR = "_spec_prefill_debug_positions"

_SHORT_TEST_PROMPT_TEXT = (
    "This is a moderately long test prompt used only to validate that "
    "SpecPrefill's position-restoration mechanism actually reaches the "
    "model during a real forward pass, not to test generation quality."
)

# A small pool of distinct filler sentences, cycled (not one sentence
# repeated verbatim) -- avoids feeding SpecPrefill's importance scorer an
# unnaturally-degenerate, fully-uniform attention pattern, which is an
# unnecessary risk for a prompt whose only job is to force multi-step
# scheduling (Step B2), not to test scoring quality.
_LONG_TEST_PROMPT_FILLER_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Distant mountains fade into a pale haze as the afternoon wears on.",
    "A small boat drifts slowly across the calm surface of the lake.",
    "Researchers continue to debate the exact origins of the old manuscript.",
    "The market square was busy with vendors selling fruit and bread.",
]


def _build_long_test_prompt_text(target_word_count: int = 16000) -> str:
    """Cycles through a small pool of distinct filler sentences until the
    text reaches roughly `target_word_count` words. Content is irrelevant --
    only length matters, to force multi-step chunked-prefill scheduling in
    Step B2 (see module docstring).

    16000 words (not the original 500) because the *kept* token count (after
    spec_config's ~30% pruning) needs to clear `--target-max-num-batched-
    tokens` (3072), not just be long -- and 3072 itself is a floor forced by
    Gemma-4-26B-A4B-it's multimodal encoder budget, not a free choice (see
    module docstring's 'Step B2' section). ~16000 words -> comfortably
    several thousand kept tokens, several chunks at that budget."""
    sentences: List[str] = []
    word_count = 0
    i = 0
    while word_count < target_word_count:
        sentence = _LONG_TEST_PROMPT_FILLER_SENTENCES[i % len(_LONG_TEST_PROMPT_FILLER_SENTENCES)]
        sentences.append(sentence)
        word_count += len(sentence.split())
        i += 1
    return "Ignore sentence content; only length matters here. " + " ".join(sentences)


def _capturing_forward(self, positions, hidden_states, _orig_forward, _captured, **kwargs):
    """Records `positions` (moved to CPU immediately -- this runs on the
    driver's own process/thread for UniProcExecutor, but avoid holding GPU
    refs longer than needed) then delegates to the real, unmodified
    forward -- this must not change model behavior, only observe it."""
    _captured.append(positions.detach().to("cpu").clone())
    return _orig_forward(positions, hidden_states, **kwargs)


def _install_position_capture_hook(worker) -> int:
    """collective_rpc target: installs the diagnostic hook on every
    Gemma4Attention layer of the TARGET model (worker.model_runner.model),
    storing captured positions on the worker itself so a later collective_rpc
    call can read them back. Returns the number of layers hooked.

    Same `Gemma4ForConditionalGeneration` unwrap as `proposer.py`'s
    `_find_gemma4_attention_layers` -- confirmed on real hardware that the
    target model is this wrapper class too (`AttributeError:
    'Gemma4ForConditionalGeneration' object has no attribute 'layers'`
    without it), not plain `Gemma4ForCausalLM`. Mirrors this fork's own
    `vllm/v1/spec_decode/llm_base_proposer.py:1213-1247` pattern."""
    from vllm.model_executor.models import supports_multimodal

    model = worker.model_runner.model
    text_model = model.get_language_model() if supports_multimodal(model) else model
    inner = text_model.model if hasattr(text_model, "model") else text_model

    captured: List[torch.Tensor] = []
    setattr(worker, _CAPTURE_ATTR, captured)

    num_hooked = 0
    for layer in inner.layers:
        self_attn = layer.self_attn
        original_forward = self_attn.forward  # already-bound original method
        self_attn.forward = types.MethodType(
            partial(_capturing_forward, _orig_forward=original_forward, _captured=captured),
            self_attn,
        )
        num_hooked += 1
    return num_hooked


def _read_captured_positions(worker) -> List[List[int]]:
    captured = getattr(worker, _CAPTURE_ATTR, [])
    return [p.tolist() for p in captured]


def _clear_captured_positions(worker) -> None:
    """Must mutate the existing list in place, NOT reassign a new one --
    every hooked layer's `partial(_capturing_forward, _captured=...)`
    (see `_install_position_capture_hook`) closed over the *original* list
    object at install time. `setattr(worker, _CAPTURE_ATTR, [])` here would
    only repoint the attribute at a new list, leaving the hooks writing into
    an orphaned one that `_read_captured_positions` can never see again --
    confirmed on real hardware as the actual cause of Step B always
    reporting 'no positions were captured' even though the hook fires."""
    captured = getattr(worker, _CAPTURE_ATTR, None)
    if captured is not None:
        captured.clear()


def _render_chat(tokenizer, prompt: str) -> str:
    """Renders a single user turn through the checkpoint's own chat
    template, same as gemma4_moe_benchmarks' render_chat (bench_offline.py).
    Required for an instruction-tuned checkpoint like Gemma-4-*-it: passing
    a bare completion-style string straight to generate() skips the
    <start_of_turn>/generation-prompt markers the model was RLHF'd against,
    which reliably produces degenerate repetition loops under greedy
    decoding -- confirmed on real hardware (2026-07-22): even STOCK vLLM
    with no vllm_patch code loaded at all produced the same garbage
    (' is is is is') for a raw prompt, fully exonerating SpecPrefillWorker
    for what looked like a runner-swap bug but was actually just a missing
    chat template on the test script's own prompts."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def step_a_basic_wiring(llm: LLM, tokenizer) -> None:
    print("=== Step A: basic wiring (worker_cls loaded, normal generation) ===")
    outputs = llm.generate(
        [_render_chat(tokenizer, "The capital of France is")],
        SamplingParams(max_tokens=4, temperature=0.0),
    )
    text = outputs[0].outputs[0].text
    print(f"Normal (non-pruned) generation succeeded: {text!r}")
    if not text.strip():
        print("WARNING: empty output -- something may be wrong even for the non-pruned path.")


def _run_position_check(
    llm: LLM,
    proposer: SpecPrefillProposer,
    spec_config: SpecConfig,
    speculator_device: torch.device,
    head_dim: int,
    tokenizer,
    *,
    request_id: str,
    prompt_text: str,
    label: str,
    min_expected_chunks: int = 1,
    max_steps: int = 20,
) -> bool:
    """Shared driver for Step B / Step B2 (see module docstring): registers
    a pruned request, drives the engine step-by-step -- reading AND clearing
    captured positions after every single step, not just once up front, so
    each step's capture is isolated -- then checks that concatenating those
    per-step captures IN STEP ORDER reproduces the registered
    `kept_positions` EXACTLY (not just as a set: order is exactly what a
    multi-chunk check needs to catch a chunk-boundary/ordering bug in
    `PruneRecord.positions_for_step`). Works identically whether the
    request's prefill lands in one step (Step B) or several (Step B2).

    Assumes `_install_position_capture_hook` has already been called once
    for this `llm` (installing it again per-call would nest wrappers around
    the already-wrapped forward and orphan the previous call's capture
    list -- see `_install_position_capture_hook`'s own docstring)."""
    print(f"\n=== {label} ===")
    llm.collective_rpc(_clear_captured_positions)

    # add_special_tokens=False: the chat template string already embeds
    # whatever special tokens (e.g. <bos>) the model expects -- re-adding
    # them here would double them up.
    prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # NOTE: do NOT read back via pruning_registry.get(request_id) here --
    # that would read the *driver* process's copy, which prune_and_add_request
    # never writes to by design (the record only exists in the Worker's
    # process, pushed via collective_rpc -- see pruner.py/worker.py
    # docstrings). Confirmed on real hardware this is a real trap: use the
    # return value instead.
    _, kept_positions, orig_len = prune_and_add_request(
        llm_engine=llm.llm_engine,
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        proposer=proposer,
        spec_config=spec_config,
        device=speculator_device,
        head_dim=head_dim,
    )
    # kept_positions is already ascending -- chunk_select_from_smoothed_
    # attention's own sortedness invariant (see test_aggregate_and_select_
    # pipeline in test_vllm_patch.py) -- NOT re-sorted here, since order
    # (not just set membership) is exactly what this check verifies.
    expected_positions = list(kept_positions)
    print(f"Registered PruneRecord: kept {len(kept_positions)} of {orig_len} tokens.")

    # Drive the engine through however many steps this request's prefill
    # actually takes. **Root cause of a real-hardware hang this script used
    # to have (2026-07-22, confirmed against this fork's own source, not
    # inferred)**: an earlier version of this loop called
    # `llm.llm_engine.step()` unconditionally without checking
    # `has_unfinished_requests()` first -- once a request finished,
    # `engine_core.get_output()` (vllm/v1/engine/llm_engine.py:295) blocked
    # forever on `outputs_queue.get()` since EngineCore had nothing left to
    # report. `llm.generate()` (Steps A/C) avoids this via vLLM's own
    # required idiom -- `while self.llm_engine.has_unfinished_requests():
    # step()` -- see vllm/entrypoints/llm.py:1285-1286 -- which this loop
    # now follows too.
    observed_chunks: List[List[int]] = []
    step_idx = 0
    while llm.llm_engine.has_unfinished_requests() and step_idx < max_steps:
        llm.llm_engine.step()
        step_idx += 1
        captured_per_worker = llm.collective_rpc(_read_captured_positions)
        captured_this_step = captured_per_worker[0] if captured_per_worker else []
        llm.collective_rpc(_clear_captured_positions)  # read-then-clear: keeps
            # each observed_chunks entry scoped to exactly this step.
        if captured_this_step:
            # Every hooked layer sees an identical positions tensor within
            # one forward pass -- any one representative entry suffices.
            observed_chunks.append(captured_this_step[0])

    if llm.llm_engine.has_unfinished_requests():
        print(f"FAIL: request did not finish within {max_steps} step() call(s).")
        return False
    if not observed_chunks:
        print("FAIL: no positions were captured at all -- the hook may not "
              "have fired, or the request never reached the model.")
        return False
    if len(observed_chunks) < min_expected_chunks:
        print(f"FAIL: only {len(observed_chunks)} step-chunk(s) observed, "
              f"expected >= {min_expected_chunks} -- this run did not "
              f"actually exercise the scenario {label!r} is meant to test "
              f"(check prompt length vs --target-max-num-batched-tokens and "
              f"spec_config's keep percentage).")
        return False

    actual_positions = [p for chunk in observed_chunks for p in chunk]
    print(f"Captured {len(observed_chunks)} step-chunk(s), "
          f"{len(actual_positions)} position(s) total.")

    if actual_positions == expected_positions:
        print("PASS: concatenated per-step positions the model actually "
              "received EXACTLY MATCH the registered kept_positions (T), "
              "in order, across all step(s). The self.positions "
              "view-aliasing assumption holds.")
        return True

    stock_positions = list(range(len(kept_positions)))
    if actual_positions == stock_positions:
        print("FAIL: the model received the STOCK contiguous positions "
              f"({stock_positions[:5]}...), not our overridden ones "
              f"({expected_positions[:5]}...). The override in "
              "model_runner.py is not reaching the model -- see its "
              "docstring's 'Residual risk' section for what to check next "
              "(something after line 2280 in _prepare_inputs likely "
              "re-touches self.positions, or the attention backend clones "
              "the positions field before use).")
    else:
        print(f"FAIL: captured positions {actual_positions[:10]}... match "
              f"neither expected {expected_positions[:10]}... nor stock "
              f"{stock_positions[:10]}... -- unexpected, needs investigation.")
    return False


def step_c_coherence_smoke_test(llm: LLM, tokenizer) -> None:
    print("\n=== Step C: coherence smoke test (secondary signal only) ===")
    outputs = llm.generate(
        [_render_chat(tokenizer, "Explain in one sentence why the sky is blue.")],
        SamplingParams(max_tokens=20, temperature=0.0),
    )
    print(f"Output for a normal request run through the SpecPrefill worker: "
          f"{outputs[0].outputs[0].text!r}")
    print("(Eyeball check only -- rigorous accuracy validation is LongBench "
          "v2's job, separately, per EXPERIMENT_PLAN.md.)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--speculator-model", required=True)
    parser.add_argument("--device", default="cuda", help="Target model's GPU.")
    parser.add_argument(
        "--speculator-device",
        default=None,
        help=(
            "Speculator's GPU, e.g. 'cuda:1'. Defaults to the second visible "
            "GPU if one exists (the protocol's own resource requirement is "
            "2x A100 -- see EXPERIMENT_PLAN.md), so the target and speculator "
            "don't compete for the same GPU's memory. Falls back to sharing "
            "--device (with a warning) if only one GPU is visible; in that "
            "case, lower --target-gpu-memory-utilization to make room."
        ),
    )
    parser.add_argument(
        "--target-gpu-memory-utilization",
        type=float,
        default=0.9,
        help=(
            "Only matters if the speculator ends up sharing the target's GPU "
            "(single-GPU fallback above) -- vLLM's own default (0.9) then "
            "leaves no room to also load the speculator standalone in the "
            "same process (confirmed on real hardware: CUDA OOM loading "
            "Gemma-4-E2B-it's first layer). With a real second GPU for the "
            "speculator (the normal case), this can stay near vLLM's default."
        ),
    )
    parser.add_argument(
        "--target-max-num-batched-tokens",
        type=int,
        default=3072,
        help=(
            "Passed through to LLM(...)'s max_num_batched_tokens, along with "
            "enable_chunked_prefill=True, to force genuine multi-step "
            "chunked-prefill scheduling for Step B2's longer prompt (while "
            "Step B's existing short prompt still fits in a single step, "
            "preserving single-step coverage too). Can't be pushed much "
            "lower than this for Gemma-4-26B-A4B-it specifically: its "
            "multimodal-bidirectional attention forces "
            "max_num_batched_tokens >= max_tokens_per_mm_item (2496 for "
            "this model, vllm/platforms/cuda.py's CUDA platform forcing, "
            "not overridable) at LLM(...) construction time -- see the "
            "module docstring's 'Step B2' section for the real-hardware "
            "confirmation of this gotcha. NOTE: this whole setup "
            "deliberately diverges from EXPERIMENT_PLAN.md's real-benchmark "
            "config (enable_chunked_prefill=False) -- see the same section."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.speculator_device is not None:
        speculator_device = torch.device(args.speculator_device)
    elif torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        speculator_device = torch.device("cuda:1")
    else:
        speculator_device = device
        print(
            f"WARNING: only {torch.cuda.device_count()} GPU(s) visible -- "
            f"speculator will share the target's GPU ({device}) instead of "
            f"getting its own. This is why the target's "
            f"gpu_memory_utilization ({args.target_gpu_memory_utilization}) "
            f"matters here; lower it further if the speculator still OOMs."
        )

    target_gpu_memory_utilization = (
        args.target_gpu_memory_utilization if speculator_device == device else 0.9
    )
    print(f"Loading target model {args.target_model} via SpecPrefillWorker "
          f"(gpu_memory_utilization={target_gpu_memory_utilization}) on {device}...")
    llm = LLM(
        model=args.target_model,
        worker_cls="vllm_patch.worker.SpecPrefillWorker",
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=target_gpu_memory_utilization,
        max_num_batched_tokens=args.target_max_num_batched_tokens,
        enable_chunked_prefill=True,  # explicit, not relying on the (already
            # default-True) upstream default -- see
            # --target-max-num-batched-tokens' help text for why this script
            # intentionally diverges from EXPERIMENT_PLAN.md's real-
            # benchmark config here.
    )
    print("Target model loaded.")

    tokenizer = llm.get_tokenizer()
    step_a_basic_wiring(llm, tokenizer)

    print(f"\nLoading speculator {args.speculator_model} standalone (via "
          f"SpecPrefillProposer, same as validate_proposer.py) on "
          f"{speculator_device}...")
    if speculator_device.type == "cuda":
        # get_model()'s internal weight placement follows torch's *current*
        # CUDA device, not an explicit device argument threaded through
        # VllmConfig (DeviceConfig.device is deprecated/generic-only, not a
        # GPU index -- confirmed by reading vllm/config/device.py). Without
        # this, weights would land on whatever device is "current" (GPU 0,
        # already holding the target model), regardless of the device=...
        # passed to SpecPrefillProposer below.
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
    print(f"Speculator loaded, head_dim={head_dim}.")

    spec_config = SpecConfig(
        keep_strategy="percentage",
        keep_kwargs={"chunk": True, "chunk_size": 64, "percentage": 0.3},
        # look_ahead_cnt=8 matches EXPERIMENT_PLAN.md's documented default --
        # was pinned to 1 here since proposer.py's multi-step lookahead had
        # a real bug (see run_lookahead_steps' docstring, fixed 2026-07-23);
        # now that it's fixed, this deliberately exercises the multi-step
        # path since this script already proved (2026-07-22, Step B/B2 PASS)
        # it can run the full pruner.py pipeline against the real
        # heterogeneous-head-dim speculator end-to-end.
        look_ahead_cnt=8,
        pool_kernel_size=None,
    )

    # Installed once, before both position checks below -- calling this
    # per-check would nest wrappers around the already-wrapped forward and
    # orphan the previous call's capture list (see its own docstring).
    num_hooked = llm.collective_rpc(_install_position_capture_hook)
    print(f"Diagnostic hook installed on {num_hooked} worker(s)' target-model "
          f"attention layers.")

    passed_b = _run_position_check(
        llm, proposer, spec_config, speculator_device, head_dim, tokenizer,
        request_id="spec_prefill_validation_request_b_singlestep",
        prompt_text=_render_chat(tokenizer, _SHORT_TEST_PROMPT_TEXT),
        label="Step B: self.positions view-aliasing check, single-step prefill (risk #1)",
        min_expected_chunks=1,
    )
    passed_b2 = _run_position_check(
        llm, proposer, spec_config, speculator_device, head_dim, tokenizer,
        request_id="spec_prefill_validation_request_b2_multistep",
        prompt_text=_render_chat(tokenizer, _build_long_test_prompt_text()),
        label="Step B2: multi-step chunked-prefill check (position "
              "restoration across 2+ scheduler steps for one pruned request)",
        min_expected_chunks=2,
    )
    passed = passed_b and passed_b2

    step_c_coherence_smoke_test(llm, tokenizer)

    if not passed:
        print("\nOVERALL: Step B/B2 FAILED -- see above for what to check next.")
        sys.exit(1)
    print("\nOVERALL: all checks passed.")


if __name__ == "__main__":
    main()
