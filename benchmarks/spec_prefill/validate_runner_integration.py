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

Step C -- coherence smoke test: generates a few tokens for a heavily-pruned
prompt and just prints the output for a human eyeball check -- not a
substitute for Step B's direct check, but a useful secondary signal
(if Step B passes but output is obvious garbage, something *else* is wrong,
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
from vllm.inputs import TokensPrompt

from vllm_patch import pruning_registry
from vllm_patch.config import SpecConfig
from vllm_patch.proposer import SpecPrefillProposer
from vllm_patch.pruner import prune_and_add_request

_CAPTURE_ATTR = "_spec_prefill_debug_positions"


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
    call can read them back. Returns the number of layers hooked."""
    model = worker.model_runner.model
    inner = model.model if hasattr(model, "model") else model

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
    setattr(worker, _CAPTURE_ATTR, [])


def step_a_basic_wiring(llm: LLM) -> None:
    print("=== Step A: basic wiring (worker_cls loaded, normal generation) ===")
    outputs = llm.generate(
        ["The capital of France is"], SamplingParams(max_tokens=4, temperature=0.0)
    )
    text = outputs[0].outputs[0].text
    print(f"Normal (non-pruned) generation succeeded: {text!r}")
    if not text.strip():
        print("WARNING: empty output -- something may be wrong even for the non-pruned path.")


def step_b_position_aliasing_check(
    llm: LLM,
    proposer: SpecPrefillProposer,
    spec_config: SpecConfig,
    speculator_device: torch.device,
    head_dim: int,
    tokenizer,
) -> bool:
    print("\n=== Step B: self.positions view-aliasing check (risk #1) ===")
    num_hooked = llm.collective_rpc(_install_position_capture_hook)
    print(f"Diagnostic hook installed on {num_hooked} worker(s)' target-model attention layers.")
    llm.collective_rpc(_clear_captured_positions)

    prompt_text = (
        "This is a moderately long test prompt used only to validate that "
        "SpecPrefill's position-restoration mechanism actually reaches the "
        "model during a real forward pass, not to test generation quality."
    )
    prompt_token_ids = tokenizer.encode(prompt_text)

    request_id = "spec_prefill_validation_request"
    pruning_registry.discard(request_id)  # clean slate if re-run
    prune_and_add_request(
        llm_engine=llm.llm_engine,
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        proposer=proposer,
        spec_config=spec_config,
        device=speculator_device,
        head_dim=head_dim,
    )
    record = pruning_registry.get(request_id)
    if record is None:
        raise RuntimeError(
            "PruneRecord was not registered -- prune_and_add_request may have "
            "failed silently or discard_finished already cleaned it up."
        )
    expected_positions = sorted(record.kept_positions)
    print(f"Registered PruneRecord: kept {record.num_kept} of {record.orig_len} tokens.")

    # Drive the engine through the prefill step.
    llm.llm_engine.step()

    captured_per_worker = llm.collective_rpc(_read_captured_positions)
    captured = captured_per_worker[0] if captured_per_worker else []
    if not captured:
        print("FAIL: no positions were captured at all -- the hook may not "
              "have fired, or the request never reached the model this step.")
        return False

    # First captured entry should be the prefill step's positions for our
    # request (there may be other requests' data too if the batch wasn't
    # isolated -- take the entry whose length matches our pruned prompt).
    matching = [p for p in captured if len(p) == record.num_kept]
    if not matching:
        print(f"FAIL: no captured position tensor has length {record.num_kept} "
              f"(our pruned prompt length). Captured lengths: "
              f"{[len(p) for p in captured]}")
        return False

    actual_positions = sorted(matching[0])
    if actual_positions == expected_positions:
        print("PASS: positions the model actually received during its real "
              "forward pass EXACTLY MATCH the registered kept_positions (T). "
              "The self.positions view-aliasing assumption holds.")
        return True

    stock_positions = sorted(range(record.num_kept))
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


def step_c_coherence_smoke_test(llm: LLM) -> None:
    print("\n=== Step C: coherence smoke test (secondary signal only) ===")
    outputs = llm.generate(
        ["Explain in one sentence why the sky is blue."],
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
    )
    print("Target model loaded.")

    step_a_basic_wiring(llm)

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
        look_ahead_cnt=1,  # see proposer.py's known limitation -- >1 not yet reliable
        pool_kernel_size=None,
    )

    tokenizer = llm.get_tokenizer()
    passed = step_b_position_aliasing_check(
        llm, proposer, spec_config, speculator_device, head_dim, tokenizer
    )

    step_c_coherence_smoke_test(llm)

    if not passed:
        print("\nOVERALL: Step B FAILED -- see above for what to check next.")
        sys.exit(1)
    print("\nOVERALL: all checks passed.")


if __name__ == "__main__":
    main()
