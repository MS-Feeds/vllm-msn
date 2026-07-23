"""Baseline check: does STOCK vLLM (no worker_cls, no spec_prefill code at
all) correctly handle a long (>max_num_batched_tokens) prompt with
SamplingParams(max_tokens=1) under chunked prefill, or does it also finish
prematurely (finish_reason='length') after just the first chunk?

validate_runner_integration.py's Step B2 observed: a 5418-token pruned
prompt, max_num_batched_tokens=3072, got exactly 1 chunk of 3072 tokens
scheduled, then RequestOutput.finished=True/finish_reason='length' --
despite only 3072 of 5418 tokens having been prefilled. gpu_model_runner.py's
own discard_request_mask logic (confirmed by reading the source directly)
looks structurally correct and untouched by spec_prefill's patch -- this
script isolates whether the bug is generic to this fork's chunked-prefill
implementation or specific to spec_prefill's model_runner.py override.

Usage:
    python3 check_chunked_prefill_baseline.py --target-model $GEMMA4_MODEL_PATH
"""

import argparse

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, default=3072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    llm = LLM(
        model=args.target_model,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
    )
    tokenizer = llm.get_tokenizer()

    # Long enough to need >1 chunk at max_num_batched_tokens, short enough to
    # be fast. Content doesn't matter, only length.
    filler = ("The quick brown fox jumps over the lazy dog near the riverbank. "
              "Distant mountains fade into a pale haze as the afternoon wears on. ") * 400
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Ignore content, only length matters. " + filler}],
        add_generation_prompt=True, tokenize=False,
    )
    prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"Prompt length: {len(prompt_token_ids)} tokens "
          f"(max_num_batched_tokens={args.max_num_batched_tokens}, "
          f"so this needs >1 chunk if chunking works correctly).")

    request_id = "chunked_prefill_baseline_check"
    from vllm.inputs import TokensPrompt
    llm.llm_engine.add_request(
        request_id, TokensPrompt(prompt_token_ids=prompt_token_ids),
        SamplingParams(max_tokens=1, temperature=0.0),
    )

    step_idx = 0
    while llm.llm_engine.has_unfinished_requests() and step_idx < 20:
        step_outputs = llm.llm_engine.step()
        step_idx += 1
        for out in step_outputs:
            if out.request_id == request_id:
                completion = out.outputs[0] if out.outputs else None
                print(f"step {step_idx}: finished={out.finished}, "
                      f"num_cached_tokens={out.num_cached_tokens}, "
                      f"finish_reason={getattr(completion, 'finish_reason', None)!r}, "
                      f"token_ids={list(getattr(completion, 'token_ids', []))!r}")

    if step_idx == 1:
        print("\nFAIL (matches spec_prefill's Step B2 symptom): finished in "
              "1 step despite prompt exceeding max_num_batched_tokens -- this "
              "is a STOCK vLLM behavior on this fork, NOT specific to "
              "spec_prefill's model_runner.py override.")
    else:
        print(f"\nPASS: took {step_idx} steps as expected for a "
              f"{len(prompt_token_ids)}-token prompt -- stock chunked "
              f"prefill + max_tokens=1 works fine here, so the bug IS "
              f"specific to spec_prefill's patch.")


if __name__ == "__main__":
    main()
