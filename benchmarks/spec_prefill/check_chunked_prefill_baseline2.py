"""Deeper baseline probe: does stock vLLM (no worker_cls, no spec_prefill
code) actually SCHEDULE the remaining prefill after a request's first
max_num_batched_tokens-sized chunk, or does it transition into decode mode
prematurely (i.e. treat the request as prefill-complete after just the first
chunk, generating bogus "continuation" tokens instead of resuming prefill)?

validate_runner_integration.py's Step B2 (max_tokens=4) took 7 steps to
finish but only ever captured ONE prefill chunk (3072 of 5440 tokens) --
this script watches every single step's RequestOutput.token_ids to see
exactly what happens after step 1.

Usage:
    python3 check_chunked_prefill_baseline2.py --target-model $GEMMA4_MODEL_PATH
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

    filler = ("The quick brown fox jumps over the lazy dog near the riverbank. "
              "Distant mountains fade into a pale haze as the afternoon wears on. ") * 400
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Ignore content, only length matters. " + filler}],
        add_generation_prompt=True, tokenize=False,
    )
    prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"Prompt length: {len(prompt_token_ids)} tokens "
          f"(max_num_batched_tokens={args.max_num_batched_tokens}).")

    request_id = "chunked_prefill_baseline2"
    from vllm.inputs import TokensPrompt
    llm.llm_engine.add_request(
        request_id, TokensPrompt(prompt_token_ids=prompt_token_ids),
        SamplingParams(max_tokens=8, temperature=0.0),
    )

    step_idx = 0
    while llm.llm_engine.has_unfinished_requests() and step_idx < 20:
        step_outputs = llm.llm_engine.step()
        step_idx += 1
        matched = [o for o in step_outputs if o.request_id == request_id]
        if not matched:
            print(f"step {step_idx}: (no output for our request this step)")
            continue
        out = matched[0]
        completion = out.outputs[0] if out.outputs else None
        print(f"step {step_idx}: finished={out.finished}, "
              f"num_cached_tokens={out.num_cached_tokens}, "
              f"finish_reason={getattr(completion, 'finish_reason', None)!r}, "
              f"this_step_token_ids={list(getattr(completion, 'token_ids', []))!r}, "
              f"cumulative_len_so_far={len(completion.token_ids) if completion else 0}")

    print(f"\nTotal steps: {step_idx}")


if __name__ == "__main__":
    main()
