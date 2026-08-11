"""SpecPrefill Algorithm 1 implementation, extended for multi-turn
conversations -- see `../EXPERIMENT_PLAN.md` for the protocol this
implements and its architectural decisions.

Status: code-complete but NOT run on real hardware (no GPU on the machine
this was written on) -- same caveat as `../spec_prefill_llama/` (the
single-turn pipeline this was built on top of), compounded here by a
genuinely new integration (the speculator as a persistent engine, see
`speculator_worker.py`'s module docstring) that has no single-turn
precedent to lean on. See `../REPRODUCE.md`'s validation steps for what to
run first on a GPU node, and each module's own "Known risk areas" /
"residual risk" notes for what's most likely to need a fix on first real
run.

Module map (relative to the single-turn `../spec_prefill_llama/vllm_patch/`
this was built from -- see EXPERIMENT_PLAN.md's "Key architectural
decisions" for the full reasoning behind each column):

    Copied verbatim (architecture- and conversation-agnostic):
        scoring.py            -- (12, 14, 16) attention scoring/selection math.
        kv_cache_utils.py     -- (11, K half) generic paged-KV-cache read-back.
        prefill_split.py      -- (1) split_prefill_decode_requests.
        pruning_registry.py   -- process-local PruneRecord store (docstring
                                  updated to clarify multi-turn absolute-
                                  position semantics; zero logic changes).
        model_runner.py       -- (18-19) SpecPrefillGPUModelRunner, the
                                  TARGET's RoPE-position-override runner.
        worker.py              -- wires model_runner.py in via worker_cls,
                                  for the TARGET model only.

    New (no single-turn analog):
        conversation_state.py -- per-conversation absolute-position ledger
                                  + KEEP/DISCARD candidate-pool construction.
        speculator_worker.py  -- runs the SPECULATOR as a persistent
                                  worker_cls-based engine (query capture +
                                  KV read-back via collective_rpc, since a
                                  real engine's Worker process is separate
                                  from the driver -- see its own docstring).

    Rewritten (same responsibility, new implementation):
        proposer.py            -- (3-11) SpecPrefillProposer: now a thin
                                  driver-facing orchestrator over a
                                  persistent speculator LLM(), not a
                                  standalone model + hand-rolled lookahead
                                  loop.
        pruner.py               -- driver-facing entry point: conversation-
                                  aware (threads conversation_state.py
                                  through), plus a separate oracle-upper-
                                  bound scoring path sharing the same
                                  selection core.
        config.py               -- SpecConfig, +1 field (keep_mode).

Public re-exports below are for convenience only -- each module also works
standalone. Guarded the same way the single-turn pipeline's `__init__.py`
guards them: `config.py`/`prefill_split.py`/`scoring.py`/
`pruning_registry.py`/`conversation_state.py` have no vLLM runtime
dependency and are importable/testable anywhere; the rest pull in vLLM's
full runtime.
"""

from .config import SpecConfig, get_spec_config, init_spec_config
from .conversation_state import ConversationState
from .prefill_split import split_prefill_decode_requests
from .pruning_registry import PruneRecord
from .pruning_registry import discard as discard_prune_record
from .pruning_registry import discard_finished as discard_finished_prune_records
from .pruning_registry import get as get_prune_record
from .pruning_registry import register as register_prune_record
from .scoring import (
    aggregate_attention_score,
    chunk_select_from_smoothed_attention,
    compute_attention_score,
)

__all__ = [
    "SpecConfig",
    "get_spec_config",
    "init_spec_config",
    "ConversationState",
    "split_prefill_decode_requests",
    "compute_attention_score",
    "aggregate_attention_score",
    "chunk_select_from_smoothed_attention",
    "PruneRecord",
    "register_prune_record",
    "get_prune_record",
    "discard_prune_record",
    "discard_finished_prune_records",
]

try:
    from .kv_cache_utils import (
        gather_keys_for_slots,
        read_layer_keys,
        retrieve_keys_per_sample,
    )
    from .model_runner import SpecPrefillGPUModelRunner
    from .proposer import SpecPrefillProposer
    from .pruner import (
        PrunedTurnResult,
        compute_oracle_kept_pairs,
        compute_pruned_turn,
        prune_and_add_turn,
    )
    from .speculator_worker import SpeculatorGPUModelRunner, SpeculatorWorker
    from .worker import SpecPrefillWorker

    __all__ += [
        "SpecPrefillProposer",
        "read_layer_keys",
        "gather_keys_for_slots",
        "retrieve_keys_per_sample",
        "PrunedTurnResult",
        "compute_pruned_turn",
        "prune_and_add_turn",
        "compute_oracle_kept_pairs",
        "SpecPrefillGPUModelRunner",
        "SpecPrefillWorker",
        "SpeculatorGPUModelRunner",
        "SpeculatorWorker",
    ]
except ImportError:
    # vLLM's full runtime isn't importable in this environment -- the
    # engine-agnostic pieces above still work. Import vllm_patch.proposer /
    # vllm_patch.worker etc. directly (where vLLM *is* installed) to get the
    # real ImportError instead of this silent skip.
    pass
