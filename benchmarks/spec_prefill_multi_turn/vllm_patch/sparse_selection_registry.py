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

## Generation counter -- a real bug this fixes, not defensive plumbing

`sparse_target_runner.py`'s per-turn caches used to key their invalidation
on `id(selected_positions)` -- the theory being that `register()` always
stores a NEW list object, so a fresh `id()` each turn reliably signals
"this is a different selection, recompute." That's true only while the
OLD list is still alive. Once a turn ends (`discard()` drops this
registry's own reference) and nothing else holds the object, CPython frees
it immediately and its address becomes eligible for reuse by the very next
allocation -- which can easily be the NEXT turn's own `list(selected_
positions)` call. If that new list happens to land at the same freed
address, `id()` collides with the OLD (already-freed) list's id, and the
runner's cache -- which is never proactively cleared, only compared by
key -- silently believes "same selection as last turn, reuse the cached
block indices," gathering blocks that belong to an entirely different,
stale selection.

Confirmed as the real cause of a production symptom, not a theoretical
gap: `predict_scbench.py` SPARSE runs showed genuinely non-deterministic
output (different generations from IDENTICAL re-runs at temperature=0)
that degenerated into repetition loops -- exactly what attending to a
stale, mismatched selection would produce, and exactly the kind of bug
that would depend on incidental memory-allocator timing rather than on
the model's own (deterministic) computation, explaining the run-to-run
variance.

`_next_generation` is a simple monotonic counter, incremented on every
`register()` call regardless of `request_id` -- it has no relationship to
any object's lifetime or memory address, so it cannot collide the way
`id()` did. Exposed via `get_with_generation()` (not a separate call from
`get()`) specifically so a caller reads the positions and the generation
that produced them as one atomic snapshot under a single lock acquisition
-- reading them via two separate calls would reopen a narrower version of
the same class of bug if `register()` ran in between.
"""

import threading
from typing import Dict, List, Optional, Tuple

_registry: Dict[str, List[int]] = {}
_generations: Dict[str, int] = {}
_next_generation = 0
_lock = threading.Lock()


def register(request_id: str, selected_positions: List[int]) -> None:
    """`selected_positions` -- absolute conversation-ledger positions
    (== local positions in the target's own persistent session, per module
    docstring), sorted or not (callers should sort for determinism, but
    this module doesn't require it -- `sparse_target_runner.py`'s gather
    doesn't care about input order, it just goes into a block-index set)."""
    global _next_generation
    with _lock:
        _registry[request_id] = list(selected_positions)
        _generations[request_id] = _next_generation
        _next_generation += 1


def get(request_id: str) -> Optional[List[int]]:
    """Returns None for any request with no active selection -- e.g. the
    force-full-attention prefill portion of a turn, or a conversation not
    using this pipeline at all."""
    with _lock:
        return _registry.get(request_id)


def get_with_generation(request_id: str) -> Optional[Tuple[int, List[int]]]:
    """Returns (generation, selected_positions) as one atomic snapshot, or
    None if there's no active selection -- see module docstring's
    "Generation counter" section for why this must be one call, not
    `get()` plus a separate generation lookup. Callers that only need the
    positions (e.g. tests, or anything not doing per-turn caching) should
    keep using plain `get()`."""
    with _lock:
        positions = _registry.get(request_id)
        if positions is None:
            return None
        return _generations[request_id], positions


def discard(request_id: str) -> None:
    with _lock:
        _registry.pop(request_id, None)
        _generations.pop(request_id, None)


def clear() -> None:
    """For tests only -- resets the whole registry."""
    with _lock:
        _registry.clear()
        _generations.clear()
