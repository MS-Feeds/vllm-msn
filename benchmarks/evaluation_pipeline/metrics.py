#!/usr/bin/env python3
"""Speculative-decoding metric collection for the evaluation pipeline.

Will provide snapshot_spec_decode_counters(llm) and
diff_spec_decode_counters(before, after), built on llm.get_metrics()
(vllm/entrypoints/llm.py) reading the vllm:spec_decode_num_draft_tokens_total
and vllm:spec_decode_num_accepted_tokens_total counters documented in
vllm/v1/spec_decode/metrics.py.

Computes:
    acceptance_rate = accepted / draft
    mean_accept_length = 1 + accepted / num_drafts

Must return None/n/a cleanly when speculative decoding is off (no
spec_model configured on the LLM) rather than erroring.

Not yet implemented.
"""
