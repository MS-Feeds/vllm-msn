#!/usr/bin/env python3
"""No-GPU end-to-end smoke test: exercises eval_data's sample generation,
rlm_stage/evidence_rlm.py (a real Anthropic API call), and
rlm_stage/evidence_cache.py together -- the full evidence-collection half
of runner/run_arm.py (via collect_evidence_for_dataset), without ever
touching vllm/target_stage. Complements rlm_stage/evidence_rlm.py's own
--smoke-test (which exercises RLM alone) by also proving the caching layer
sitting on top of it. See REPRODUCE.md step 4.

Usage:
    python3 runner/smoke_test.py
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))

from eval_data.gen_synthetic_niah import generate_sample  # noqa: E402
from runner.run_arm import collect_evidence_for_dataset  # noqa: E402


MAX_ATTEMPTS = 3


def main() -> None:
    rng = random.Random(42)
    # n_needles=1: keeps the task itself trivial. Even so, confirmed during
    # implementation that RLM sometimes doesn't converge (set
    # answer["ready"]=True) within max_iterations (configs/guardrails.yaml,
    # default 30) on this prompt design -- ordinary LLM-call variance, not
    # a pipeline bug -- which trips a real upstream bug in the vendored
    # `rlm` package's max-iterations fallback path (see run_arm.py's
    # collect_evidence_for_dataset docstring for the full finding).
    # collect_evidence_for_dataset skips-and-reports such samples rather
    # than crashing; this smoke test retries a few times rather than
    # treating one unlucky non-convergent run as a pipeline failure --
    # each retry is a genuinely fresh RLM attempt (a skipped sample is
    # never cached, so evidence_cache.get_or_run() will re-invoke RLM, not
    # silently reuse a nonexistent cache entry).

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        results_1 = None
        sample = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            sample = generate_sample(rng, context_tokens_target=150, n_needles=1, sample_idx=attempt)
            print(f"[smoke_test] First pass (attempt {attempt}/{MAX_ATTEMPTS}): expect a real RLM call (cache miss) ...")
            results_1 = collect_evidence_for_dataset([sample], cache_dir=cache_dir)
            if results_1:
                break
            print(f"[smoke_test] attempt {attempt} didn't converge -- retrying with a fresh sample.")

        assert results_1, (
            f"RLM failed to converge in {MAX_ATTEMPTS} attempts -- if this "
            f"persists, it's worth investigating the prompt design "
            f"(prompts/evidence_extraction.py) or guardrails.yaml's "
            f"max_iterations, not just re-running."
        )
        needle_values = [n["value"] for n in sample.extra["needles"]]
        assert results_1[0].was_cache_hit is False, "expected a cache MISS on first run"

        print("[smoke_test] Second pass: expect a cache hit, no RLM call ...")
        results_2 = collect_evidence_for_dataset([sample], cache_dir=cache_dir)
        assert results_2[0].was_cache_hit is True, "expected a cache HIT on second run"
        assert (
            results_2[0].evidence_result.completion.response
            == results_1[0].evidence_result.completion.response
        ), "cached response must be byte-identical to the original run (confound control)"

        excerpts_text = " ".join(
            e.get("text", "") for e in results_1[0].evidence_result.evidence["excerpts"]
        )
        found = [v for v in needle_values if v in excerpts_text]
        print(f"[smoke_test] {len(found)}/{len(needle_values)} needle(s) found in extracted evidence.")
        if len(found) < len(needle_values):
            print(
                "[smoke_test] WARNING: not all needles recovered -- worth a closer look "
                "at the prompt design, though a single smoke-test run isn't a real eval."
            )

    print("[smoke_test] PASSED: evidence collection + caching pipeline works end-to-end.")


if __name__ == "__main__":
    main()
