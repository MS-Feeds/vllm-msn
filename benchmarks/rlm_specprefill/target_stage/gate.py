"""SpecPrefill's per-query gate: decide whether a query's evidence is large
enough for SpecPrefill compression to be worth paying for.

Per ../rlm_specprefill_ablation_plan.md's SPECPREFILL GATE section: gate on
`N` (candidate token count) alone, `N > N_min` -- SpecPrefill's keep rate
`r` isn't known until it actually runs, so it can't be the gate signal; the
dominant cost at small `N` is fixed overhead (speculator invocation, any
inter-service round trip), independent of `r`. `N_min` itself is calibrated
separately (calibration/sweep_n_min.py, not yet built) and written to
configs/n_min.json -- this module only implements the decision rule and the
config loader, not the calibration itself.
"""

from __future__ import annotations

import json
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_N_MIN_PATH = THIS_DIR / "configs" / "n_min.json"


def should_compress(n_tokens: int, n_min: int) -> bool:
    """Arm C's gate rule. `n_tokens` is the token count of the PRUNABLE
    excerpts blob under the target's real tokenizer (see
    target_stage/vllm_offline_engine.py's render_excerpts_text /
    route_queries.py's count_query_tokens) -- not the original raw context,
    which RLM already reduced away before this stage ever runs."""
    return n_tokens > n_min


def load_n_min(path: Path = DEFAULT_N_MIN_PATH) -> int:
    """Reads the calibrated N_min value written by calibration/sweep_n_min.py.

    Raises a clear error rather than falling back to a guessed default --
    per ../rlm_specprefill_ablation_plan.md's ORDER OF OPERATIONS, N_min
    calibration is a prerequisite step, not optional configuration. A
    silent default here would make Arm C's gate decision meaningless
    without anyone noticing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run calibration/sweep_n_min.py first. "
            "Arm C's gate has no sane default N_min; see "
            "../rlm_specprefill_ablation_plan.md's ORDER OF OPERATIONS "
            "(N_min calibration is a prerequisite step, not optional config)."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["n_min"]
