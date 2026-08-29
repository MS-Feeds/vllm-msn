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


# Enumerated rather than free-form (no "layers 3,7,11-19" mini-language, no
# arbitrary callables): every value here has to survive a `collective_rpc`
# hop into the speculator's own process and end up in a CSV column, and a
# closed set keeps both of those honest. Extend the set when a variant earns
# a row in the experiment matrix.
SCORE_AGGREGATIONS = frozenset({"max", "mean", "zmean"})
SCORE_LAYER_SELECTIONS = frozenset(
    {None, "skip_first2", "second_half", "last_quarter", "global_only"}
)


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
    # How per-(layer, head) attention is collapsed into ONE importance score
    # per prompt token, and which layers get a vote. Both exist because
    # ORACLE-k20 measured the 1B speculator's estimation error at 17.0 of the
    # 25.0-point `scbench_kv` degradation -- 68%, against the sparse-decode
    # mechanism's 8.0 -- while the scoring pass costs ~0.007% of a turn's
    # FLOPs. See ACCURACY_IMPROVEMENTS.md's Step 0 and §1.
    #
    # `score_aggregation`:
    #   "max"   -- max over (layer, head). The reference implementation's
    #              behavior and the DEFAULT, so an unconfigured run is
    #              byte-identical to every already-published row.
    #   "mean"  -- mean over (layer, head): total attention mass rather than
    #              the single most peaked head's opinion.
    #   "zmean" -- z-score each (layer, head)'s distribution across the
    #              context first, then mean. Removes per-head scale
    #              differences, so a head with a naturally peaked
    #              distribution cannot outvote the rest by magnitude alone.
    #
    # `score_layers` (None == all layers, the default):
    #   "skip_first2" -- drop layers 0-1, which are near-universally
    #                    positional/sink-dominated.
    #   "second_half" -- layers >= L//2.
    #   "last_quarter"-- layers >= 3L//4.
    #   "global_only" -- only the FULL-attention layers of an interleaved
    #                    sliding-window model (Gemma 3/3n/4, Llama 4, ...).
    #                    Unlike the three above it is not a fixed slice: it
    #                    needs the model's own `layer_types`, supplied at
    #                    call time via `scoring.LayerGeometry`, and raises
    #                    if that isn't available. See the plan's blocker A3
    #                    -- a sliding layer's score for a position outside
    #                    its window is arbitrary, and `max` aggregation lets
    #                    one such layer decide a token's importance by
    #                    itself. On a uniform-attention model (every layer
    #                    "full_attention") this selects every layer, i.e. it
    #                    degenerates to the default.
    score_aggregation: str = "max"
    score_layers: Optional[str] = None
    # Drop cross-layer-KV-sharing layers (Gemma 3n/4's last
    # `num_kv_shared_layers`) from the vote.
    #
    # DEFAULT FALSE, and the reasoning is worth recording because the first
    # version of this defaulted to True on a mistaken premise. A KV-shared
    # layer reads another layer's K -- but it computes its OWN Q, and
    # `gpu_model_runner.initialize_kv_cache_tensors` aliases its
    # `attn.kv_cache` to the target's tensor (`kv_caches[layer_name] =
    # kv_caches[target_layer_name]`), so the K read back for it is the
    # target's REAL K, not garbage. `softmax(Q_i K_target^T)` is therefore a
    # legitimate, distinct attention distribution -- exactly as distinct as
    # any two layers are from each other. Dropping it discards signal rather
    # than removing a duplicate.
    #
    # On Gemma-4-E2B that mistake was expensive: 20 of 35 layers are
    # KV-shared, so dropping them cut `score_layers="global_only"` from 7
    # voting layers to 3.
    #
    # Set True only for a caller whose K read-back does NOT reproduce that
    # aliasing -- e.g. a hand-built dummy KV cache that allocates every layer
    # its own tensor, where a shared layer's cache is never written and
    # scoring it means scoring uninitialized memory.
    drop_kv_shared_layers: bool = False
    # Retrieval-head filtering (ACCURACY_IMPROVEMENTS.md §1.3): explicit
    # (layer, head) indices into the FLATTENED layer*head axis -- head
    # `l * num_heads + h` -- that alone get a vote. `None` == every head,
    # the reference behavior.
    #
    # Measured, not assumed: `diagnose_retrieval_heads.py` found that a
    # FIXED 2-head mask, ranked on one set of conversations and scored
    # out-of-sample on another, lifts gold-answer survival from 54.0% to
    # 82.0% (+28.0), capturing 93% of what per-input clairvoyant head
    # selection could achieve. The heads are stable (top-2 Jaccard 0.71
    # against the global ranking; 20 distinct heads across 100 turns), which
    # is what makes a static list viable at all.
    #
    # Indices are into the full, unrestricted layer*head space, so this is
    # mutually exclusive with `score_layers` -- see __post_init__.
    score_head_set: Optional[list] = None

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
        assert self.score_aggregation in SCORE_AGGREGATIONS, (
            f"score_aggregation must be one of {sorted(SCORE_AGGREGATIONS)}, "
            f"got {self.score_aggregation!r}"
        )
        assert self.score_layers in SCORE_LAYER_SELECTIONS, (
            f"score_layers must be one of {sorted(x for x in SCORE_LAYER_SELECTIONS if x)} "
            f"or None, got {self.score_layers!r}"
        )
        if self.score_head_set is not None:
            self.score_head_set = [int(h) for h in self.score_head_set]
            assert self.score_head_set, (
                "score_head_set is an empty list -- that would score with no "
                "heads at all and produce a NaN importance vector. Pass None "
                "for 'every head'."
            )
            assert len(set(self.score_head_set)) == len(self.score_head_set), (
                f"score_head_set contains duplicates: {self.score_head_set}. A "
                f"duplicated head would vote twice under `mean`, silently "
                f"reweighting the aggregation."
            )
            assert min(self.score_head_set) >= 0, (
                f"score_head_set has a negative index: {self.score_head_set}"
            )
            # Both restrict which heads vote, but `score_head_set`'s indices
            # are into the FULL layer*head axis while `score_layers` removes
            # layers from it -- combining them would silently reinterpret
            # every index as pointing at a different head.
            assert self.score_layers is None, (
                f"score_head_set and score_layers cannot be combined "
                f"(got score_layers={self.score_layers!r}): head indices are "
                f"into the full layer*head axis, which score_layers changes."
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
