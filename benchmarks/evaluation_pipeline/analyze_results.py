#!/usr/bin/env python3
"""Aggregates evaluation pipeline results into a summary report.

No accuracy scoring in this pipeline -- will read all results/*.json and
produce results/summary.md with QPS/end-to-end-time/acceptance-rate/
mean-accept-length per experiment, plus a spec-decode-on-vs-off
comparison table per dataset per k value. Modeled on
../gemma4_moe_benchmarks/analyze_results.py.

Not yet implemented.
"""
