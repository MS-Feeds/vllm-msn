"""Baseline check: does stock vLLM (no worker_cls) generate correctly for the
exact same prompt/config Step A of validate_runner_integration.py uses?

Step A (which goes through SpecPrefillWorker) produced garbage (' is is is
is' instead of continuing "The capital of France is" with "Paris") even
though _apply_spec_prefill_position_overrides is a no-op for non-pruned
requests -- so the bug, if it's ours, has to be in SpecPrefillWorker's
runner-swap itself, not the pruning logic. This script isolates that by
removing worker_cls entirely, everything else held constant.

Usage:
    python3 check_stock_baseline.py --target-model $GEMMA4_MODEL_PATH
"""

import argparse

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    llm = LLM(
        model=args.target_model,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    outputs = llm.generate(
        ["The capital of France is"], SamplingParams(max_tokens=4, temperature=0.0)
    )
    text = outputs[0].outputs[0].text
    print(f"Stock (no worker_cls) generation: {text!r}")
    if "paris" in text.lower():
        print("PASS: coherent -- SpecPrefillWorker is implicated.")
    else:
        print("FAIL: garbage even on stock vLLM -- unrelated to spec_prefill "
              "(checkpoint/dtype/trust_remote_code/quantization issue).")


if __name__ == "__main__":
    main()
