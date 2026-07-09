#!/usr/bin/env python3
"""Prepares a LiveCodeBench sample subset for the evaluation pipeline.

Will download/build a few-hundred-sample LiveCodeBench subset and write it
to datasets/livecodebench_samples.jsonl as
{"prompt": ..., "test_cases": [...], "starter_code": ...} rows.

Open question: shell out to the official `livecodebench` pip package
(same reuse strategy tests/evals/gpt_oss uses for gpt_oss.evals) instead
of hand-rolling problem loading — see plan doc.

Not yet implemented.
"""
