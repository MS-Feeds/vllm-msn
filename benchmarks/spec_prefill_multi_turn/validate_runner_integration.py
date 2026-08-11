#!/usr/bin/env python3
"""GPU-node validation for the TARGET-side integration: `vllm_patch/worker.py`
+ `vllm_patch/model_runner.py` (unchanged from the single-turn pipeline) +
`vllm_patch/pruner.py`'s NEW conversation-aware `compute_pruned_turn`/
`prune_and_add_turn`. Run this AFTER `validate_proposer.py` passes -- see
`REPRODUCE.md`'s validation order.

Steps, in order:

  A. `worker_cls=vllm_patch.worker.SpecPrefillWorker` resolves and
     constructs; a normal (non-pruned) generation call still works --
     confirms `SpecPrefillWorker.init_device()`'s runner-swap doesn't break
     ordinary generation (same check as the single-turn pipeline's Step A).
  B. **The real multi-turn-specific check**: run a 2-turn synthetic
     conversation through `compute_pruned_turn` + `prune_and_add_turn` +
     `ConversationState`, at an aggressive-enough keep rate that pruning is
     guaranteed to actually drop something (not the single-turn pipeline's
     documented 100%-retention blind spot -- see that pipeline's
     model_runner.py docstring for why a no-op-equivalent "pass" there was
     a false positive). Installs a diagnostic query-capture hook on the
     TARGET's own attention layers (same technique
     `speculator_worker.py` uses on the speculator, applied here for
     read-only position verification, not real query capture) to directly
     confirm the positions `SpecPrefillGPUModelRunner` computes for a
     PRUNED, MULTI-TURN request (absolute conversation-ledger positions,
     not turn-local ones -- see `pruning_registry.py`'s docstring) actually
     reach the model's forward pass unmodified.

Usage:
    python3 validate_runner_integration.py \\
        --target-model $LLAMA31_8B_MODEL_PATH \\
        --speculator-model $LLAMA32_1B_MODEL_PATH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _install_position_capture_hook(model):
    """Read-only diagnostic hook -- captures (positions, layer_idx) for the
    FIRST attention layer's forward call only (sufficient to confirm the
    positions vLLM's real forward pass receives, since
    SpecPrefillGPUModelRunner patches ONE shared `self.positions` buffer
    read by every layer identically -- no need to hook all of them).
    Mirrors the single-turn pipeline's own diagnostic hook in spirit
    (installed on the TARGET, not the speculator), but written fresh here
    rather than imported, since it's read-only and deliberately minimal."""
    import types

    captured = {"positions": None}
    inner = model.model if hasattr(model, "model") else model
    first_layer = inner.layers[0].self_attn
    original_forward = first_layer.forward

    def _capturing_forward(self, positions, hidden_states, **kwargs):
        captured["positions"] = positions.detach().cpu().clone()
        return original_forward(positions, hidden_states, **kwargs)

    first_layer.forward = types.MethodType(_capturing_forward, first_layer)
    return captured


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-model",
        default=os.environ.get("LLAMA31_8B_MODEL_PATH"),
        help="Defaults to $LLAMA31_8B_MODEL_PATH (see .env_exports.sh).",
    )
    parser.add_argument(
        "--speculator-model",
        default=os.environ.get("LLAMA32_1B_MODEL_PATH"),
        help="Defaults to $LLAMA32_1B_MODEL_PATH (see .env_exports.sh).",
    )
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--speculator-device", default="cuda:1")
    parser.add_argument("--target-gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--speculator-gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--look-ahead-cnt", type=int, default=8)
    args = parser.parse_args()
    if not args.target_model:
        parser.error("--target-model or $LLAMA31_8B_MODEL_PATH is required (source .env_exports.sh first)")
    if not args.speculator_model:
        parser.error("--speculator-model or $LLAMA32_1B_MODEL_PATH is required (source .env_exports.sh first)")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from vllm_patch.config import SpecConfig
    from vllm_patch.conversation_state import ConversationState
    from vllm_patch.proposer import SpecPrefillProposer
    from vllm_patch.pruner import compute_pruned_turn, prune_and_add_turn

    print(f"[validate_runner_integration] Step A: constructing target engine ({args.target_model})")
    # `LLM()`/`EngineArgs` has no `device=` constructor kwarg in this fork
    # (confirmed on real hardware: TypeError -- see proposer.py's
    # SpecPrefillProposer.__init__ docstring for the full "why" and the
    # CUDA_VISIBLE_DEVICES-scoping fix this mirrors). Same fix here, applied
    # inline since this is the target's own one-off LLM() construction, not
    # something SpecPrefillProposer wraps.
    import os

    target_device_obj = torch.device(args.target_device)
    target_device_index = target_device_obj.index if target_device_obj.type == "cuda" else None
    if target_device_index is None:
        llm = LLM(
            model=args.target_model,
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=False,
            worker_cls="vllm_patch.worker.SpecPrefillWorker",
            gpu_memory_utilization=args.target_gpu_memory_utilization,
        )
    else:
        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(target_device_index)
        try:
            llm = LLM(
                model=args.target_model,
                trust_remote_code=True,
                enforce_eager=True,
                disable_log_stats=False,
                worker_cls="vllm_patch.worker.SpecPrefillWorker",
                gpu_memory_utilization=args.target_gpu_memory_utilization,
            )
        finally:
            if prev_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    smoke_prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Say the word 'hello' and nothing else."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    smoke_ids = tok.encode(smoke_prompt, add_special_tokens=False)
    llm.llm_engine.add_request(
        "validate-smoke", {"prompt_token_ids": smoke_ids}, SamplingParams(max_tokens=8, temperature=0.0)
    )
    output = None
    while llm.llm_engine.has_unfinished_requests():
        for o in llm.llm_engine.step():
            if o.request_id == "validate-smoke":
                output = o
    assert output is not None and output.outputs[0].text, (
        "expected non-empty generation from a normal (non-pruned) request "
        "through SpecPrefillWorker -- worker_cls wiring or the runner swap "
        "itself may be broken."
    )
    print(f"[validate_runner_integration] Step A: PASS -- generated: {output.outputs[0].text!r}")

    print(f"[validate_runner_integration] Step B: constructing speculator engine ({args.speculator_model})")
    proposer = SpecPrefillProposer(
        speculator_model_path=args.speculator_model,
        device=torch.device(args.speculator_device),
        gpu_memory_utilization=args.speculator_gpu_memory_utilization,
    )

    # A synthetic but real-vocabulary "context" long enough to guarantee
    # pruning at a 20% keep rate actually drops tokens -- avoids the
    # single-turn pipeline's documented 100%-retention false-positive blind
    # spot (see this script's module docstring).
    context_text = "The quick brown fox jumps over the lazy dog. " * 40
    context_ids = tok.encode(context_text, add_special_tokens=False)

    spec_config = SpecConfig(
        keep_strategy="percentage",
        keep_kwargs={"chunk": True, "chunk_size": 16, "percentage": 0.2},
        look_ahead_cnt=args.look_ahead_cnt,
        pool_kernel_size=13,
        keep_mode="keep",
    )
    state = ConversationState("validate-conv", context_ids, keep_mode="keep")

    captured = _install_position_capture_hook(llm.llm_engine.model_executor.driver_worker.worker.model_runner.model) \
        if hasattr(llm.llm_engine, "model_executor") else None
    if captured is None:
        print(
            "[validate_runner_integration] WARNING: could not locate the "
            "target model instance via llm.llm_engine.model_executor... to "
            "install the diagnostic position-capture hook -- this fork's "
            "exact attribute path may differ from what was assumed here. "
            "Locate the real path (e.g. via a debugger or by grepping "
            "gpu_worker.py for how the model is stored) and update this "
            "script before trusting Step B's PASS/FAIL below on faith."
        )

    for turn_idx, question in enumerate(["What animal is mentioned?", "What did it jump over?"]):
        query_ids = tok.encode(f"\n\nQuestion: {question}\nAnswer:", add_special_tokens=False)
        result = compute_pruned_turn(proposer, spec_config, state, query_ids, eos_token_id=tok.eos_token_id)
        print(
            f"[validate_runner_integration] Step B: turn {turn_idx} kept "
            f"{len(result.kept_positions)}/{result.orig_len} tokens "
            f"({100 * len(result.kept_positions) / result.orig_len:.1f}%)"
        )
        assert len(result.kept_positions) < result.orig_len, (
            f"turn {turn_idx}: expected pruning to actually drop some tokens "
            f"at a 20% keep rate over a {result.orig_len}-token candidate "
            f"pool -- got 100% retention, which would make this check a "
            f"false positive (see module docstring's 100%-retention blind "
            f"spot warning)."
        )

        request_id = f"validate-conv::turn{turn_idx}"
        sampling_params = SamplingParams(max_tokens=8, temperature=0.0)
        prune_and_add_turn(llm.llm_engine, request_id, result, sampling_params)

        turn_output = None
        while llm.llm_engine.has_unfinished_requests():
            for o in llm.llm_engine.step():
                if o.request_id == request_id:
                    turn_output = o
        assert turn_output is not None, f"turn {turn_idx}: no output captured"

        if captured is not None and captured["positions"] is not None:
            observed = captured["positions"].tolist()[: len(result.kept_positions)]
            expected = result.kept_positions
            assert observed == expected, (
                f"turn {turn_idx}: MISMATCH -- SpecPrefillGPUModelRunner's "
                f"position override did NOT reach the model's real forward "
                f"pass as written. Expected (absolute, multi-turn-ledger) "
                f"positions {expected[:10]}{'...' if len(expected) > 10 else ''}, "
                f"observed {observed[:10]}{'...' if len(observed) > 10 else ''} "
                f"-- this is the riskiest assumption in the whole pruning "
                f"mechanism (see model_runner.py's docstring); do not trust "
                f"any accuracy numbers from predict_scbench.py until this "
                f"passes."
            )
            print(f"[validate_runner_integration] Step B: turn {turn_idx} positions VERIFIED correct")

        golden_answer_ids = tok.encode(" a fox jumped over a dog", add_special_tokens=False)
        state.complete_turn(result.kept_history_pairs, golden_answer_ids)

    proposer.discard_conversation("validate-conv")
    print("\n[validate_runner_integration] ALL STEPS PASSED")


if __name__ == "__main__":
    main()
