#!/bin/bash
# Env setup + entrypoint wrapper for the evaluation pipeline.
#
# Will source ../gemma4_moe_benchmarks/.env_exports.sh for
# GEMMA4_MODEL_PATH / GEMMA4_ASSISTANT_MODEL_PATH / HF_TOKEN / conda env
# activation, then invoke run_pipeline.py --all or a specific experiment
# ID. Modeled on ../gemma4_moe_benchmarks/run_experiments.sh.
#
# Not yet implemented.
set -xe
echo "Not yet implemented" >&2
exit 1
