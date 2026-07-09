#!/usr/bin/env python3
"""Main driver for the evaluation pipeline: initialize -> evaluate -> present.

Will define an EXPERIMENTS dict sweeping
{dataset: aime/livecodebench/gpqa_diamond} x {spec_decode: off/on} x
{mtp_k: 1/3/5/8 (only when spec_decode=on)}, modeled on the EXPERIMENTS
dict in ../gemma4_moe_benchmarks/bench_experiment.py.

Three stages:
    initialize_engine(exp_cfg) -- builds vllm.LLM(...), reusing the
        llm_kwargs / spec_model / spec_tokens pattern from
        bench_experiment.py:498-517.
    evaluate(llm, dataset, exp_cfg) -- loads prompts + ground truth,
        calls llm.generate(), times it, scores via the matching
        scorers/*.py, and calls metrics.py before/after for acceptance
        rate / draft length.
    present(all_results) -- writes one JSON row per
        (experiment x dataset x rep) to results/.

Not yet implemented.
"""
