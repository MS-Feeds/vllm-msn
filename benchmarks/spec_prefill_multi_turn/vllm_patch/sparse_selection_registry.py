"""Process-local registry of per-conversation sparse-attention block
selections -- the persistent-cache + speculator-guided sparse attention
counterpart to `pruning_registry.py`'s `PruneRecord` store (used by the
OTHER, physical-pruning pipeline).

## What this stores, and why it's simpler than PruneRecord

`pruning_registry.py`'s `PruneRecord` has to carry both `kept_positions`
(absolute conversation positions) AND reasoning about how those map into a
PHYSICALLY SHORTENED prompt with patched RoPE positions, because that
pipeline's target only ever sees a subset of tokens at all.

This pipeline's target session always sees and retains EVERY token of the
conversation -- nothing is ever physically dropped (see
`sparse_target_runner.py`'s module docstring for the full architecture).
Because of that, a token's ABSOLUTE conversation-ledger position (the same
numbering `vllm_patch.conversation_state.ConversationState` already
produces, reused unchanged for this pipeline -- see `predict_scbench.py`'s
sparse-experiment driving function) is IDENTICAL to its LOCAL position
within the target's own persistent session request, since that session's
own token stream IS the full, gapless conversation ledger, position for
position. No position-translation layer is needed here the way `pruner.py`
needs one for the physically-pruned pipeline -- this registry just stores
"the plain list of positions to attend to during decode," in the exact
numbering the target's own runner already uses locally.

## Lifecycle

Registered once per turn (by the driver, via `collective_rpc`, BEFORE that
turn's query update is submitted -- same "register before add_request"
ordering `pruner.py`'s `prune_and_add_turn` already establishes as
necessary, for the same reason: `EngineCore` runs its own autonomous
stepping loop, so registering reactively after submission risks the first
few decode steps running before the registration RPC lands). Stays active
for the WHOLE turn's decode phase (potentially many decode steps, one
per generated token) -- the selection is decided ONCE per turn, not
re-derived per decode step (see the approved plan's "recomputed every
turn" framing: the SELECTION is what's recomputed each turn, not
mid-turn). Overwritten (not accumulated) by the next turn's registration.

Populated inside the target Worker's own process (see
`sparse_target_runner.py::SparseTargetWorker.register_sparse_selection`)
-- never call `register()` directly from driver-side code; `EngineCore`
always runs out-of-process from the driver, confirmed on real hardware for
the physical-pruning pipeline's own identical cross-process requirement
(see `worker.py`'s docstring).
"""

import threading
from typing import Dict, List, Optional

_registry: Dict[str, List[int]] = {}
_lock = threading.Lock()


def register(request_id: str, selected_positions: List[int]) -> None:
    """`selected_positions` -- absolute conversation-ledger positions
    (== local positions in the target's own persistent session, per module
    docstring), sorted or not (callers should sort for determinism, but
    this module doesn't require it -- `sparse_target_runner.py`'s gather
    doesn't care about input order, it just goes into a block-index set)."""
    with _lock:
        _registry[request_id] = list(selected_positions)


def get(request_id: str) -> Optional[List[int]]:
    """Returns None for any request with no active selection -- e.g. the
    force-full-attention prefill portion of a turn, or a conversation not
    using this pipeline at all."""
    with _lock:
        return _registry.get(request_id)


def discard(request_id: str) -> None:
    with _lock:
        _registry.pop(request_id, None)


def clear() -> None:
    """For tests only -- resets the whole registry."""
    with _lock:
        _registry.clear()
