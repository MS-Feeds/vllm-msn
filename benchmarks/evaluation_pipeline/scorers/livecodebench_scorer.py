#!/usr/bin/env python3
"""Scores LiveCodeBench model outputs against ground truth.

Will run generated code against each problem's test cases in a sandboxed
subprocess (hard timeout, memory cap, no network access) and record
pass/fail per problem.

SECURITY NOTE: this executes model-generated code. Needs a real
sandboxing design pass (subprocess + resource limits, at minimum) before
implementation — do not run untrusted output without those guards.

Not yet implemented.
"""
