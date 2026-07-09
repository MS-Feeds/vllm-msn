#!/usr/bin/env python3
"""Aggregates evaluation pipeline results into a summary report.

Will read all results/*.json, produce results/summary.md with
accuracy/QPS/acceptance-rate/mean-accept-length per experiment, plus a
spec-decode-on-vs-off comparison table per dataset per k value. Modeled
on ../gemma4_moe_benchmarks/analyze_results.py.

Not yet implemented.
"""
