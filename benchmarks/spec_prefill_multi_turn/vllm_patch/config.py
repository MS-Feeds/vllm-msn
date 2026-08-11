"""SpecPrefill config surface.

Ported from `../../spec_prefill_llama/vllm_patch/config.py` (itself ported
as-is from the original reference implementation) -- pure Python/YAML with no
vLLM engine dependency, so it needs no V1 porting work.

One field added for this multi-turn pipeline: `keep_mode`, selecting between
the two conversation-history-retention settings the protocol calls for (see
EXPERIMENT_PLAN.md's "KEEP vs. DISCARD candidate pools"):

- `"keep"` (the protocol's first setting -- "FIRST: KEEP"): the candidate
  pool for top-k selection at every turn is the FULL original conversation
  history, rescored fresh each turn. A token pruned away at turn 2 is still
  eligible to be picked back up at turn 5.
- `"discard"`: the candidate pool at turn N is only turn N-1's own kept
  subset (plus this turn's new query) -- once a token is pruned away, it
  never comes back.

This field is consumed by `conversation_state.py` (candidate-pool
construction), not by `scoring.py` itself -- `chunk_select_from_smoothed_
attention` doesn't need to know which mode produced the token sequence it
was handed, it just scores/selects from whatever candidate pool it's given.
`keep_kwargs.chunk_size` already covers the protocol's "KV entry size" grid
axis (token/16/32/64: token = `chunk=False`, others = the corresponding
`chunk_size` value) -- no new field needed for that.
"""

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import yaml


@dataclass
class SpecConfig:
    keep_strategy: Optional[str]
    keep_kwargs: Optional[Dict[str, Any]] = None
    look_ahead_cnt: int = 8
    pool_kernel_size: Optional[int] = None
    ignore_eos: bool = False  # for benchmarking only ideally
    # "keep" (rescore the full original conversation history every turn) or
    # "discard" (candidate pool shrinks monotonically to last turn's
    # survivors) -- see module docstring. Defaults to "keep" per the
    # protocol's own stated priority ("Keep old history or discard (FIRST:
    # KEEP)").
    keep_mode: str = "keep"

    @classmethod
    def from_path(cls, config_path: Optional[str] = None):
        if config_path is None:
            return cls()

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        field_names = {f.name for f in fields(cls)}

        unused_fields = set(data.keys()) - field_names
        if unused_fields:
            raise ValueError(f"Unused fields in YAML: {unused_fields}.")

        used_data = {k: v for k, v in data.items() if k in field_names}

        return cls(**used_data)

    def __post_init__(self):
        if self.keep_kwargs is None:
            self.keep_kwargs = {}

        if self.keep_strategy is None:
            self.keep_strategy = "percentage"
            self.keep_kwargs["percentage"] = 0.5

        assert self.keep_strategy in ["percentage"]
        assert self.keep_mode in ["keep", "discard"], (
            f"keep_mode must be 'keep' or 'discard', got {self.keep_mode!r}"
        )


_SPEC_CONFIG: Optional[SpecConfig] = None


def init_spec_config():
    global _SPEC_CONFIG
    if _SPEC_CONFIG is None:
        _SPEC_CONFIG = SpecConfig.from_path(
            os.environ.get("SPEC_CONFIG_PATH")
        )
        print("\033[92m{}\033[00m".format(
            f"Using spec config:\n{json.dumps(asdict(_SPEC_CONFIG), indent=4)}"))


def get_spec_config() -> SpecConfig:
    global _SPEC_CONFIG
    assert _SPEC_CONFIG is not None
    return _SPEC_CONFIG
