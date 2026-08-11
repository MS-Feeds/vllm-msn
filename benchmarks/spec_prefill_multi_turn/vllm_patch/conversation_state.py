"""Per-conversation absolute-position ledger and KEEP/DISCARD candidate-pool
construction -- the one genuinely new piece of bookkeeping the multi-turn
port needs that has no single-turn analog (see the approved plan's "Files to
create / change": `pruning_registry.py`/`model_runner.py`/`worker.py` need
zero changes; this module is where the multi-turn-specific logic actually
lives).

Pure Python, no vLLM/torch dependency -- same spirit as `pruning_registry.py`
-- fully unit-testable without a GPU or model. Keeps `pruner.py` itself
mode-agnostic: it asks a `ConversationState` for "the candidate pool for
this turn" and hands back "here's what got kept", without needing to know
whether KEEP or DISCARD produced that pool.

## The absolute-position ledger

`PruneRecord.kept_positions` (`pruning_registry.py`) must be each kept
token's position in the ENTIRE conversation so far, not an index into the
current turn's own already-pruned prompt -- see that module's docstring.
This class maintains that ledger as an append-only list: `context` tokens
occupy positions `0..len(context)-1`, then each turn's query tokens are
appended, then (once golden-context history is available for that turn)
its golden reference answer tokens are appended -- every token ever
presented, whether or not any particular turn's scoring decided to keep it.
Positions are never reused or compacted; `total_len` only grows.

## KEEP vs. DISCARD candidate pools

- **KEEP** (the protocol's first setting): the candidate pool for turn N's
  scoring is the ENTIRE ledger built so far (context + every prior turn's
  query + golden answer), rescored fresh every turn. A token pruned away at
  turn 2 can be picked back up at turn 5, because the pool never shrinks.
- **DISCARD**: the candidate pool for turn N is exactly turn N-1's own kept
  history subset, plus turn N-1's own query tokens (which were force-kept
  in turn N-1's final prompt, and become ordinary — no longer
  automatically protected — scoreable content from turn N onward). Turn
  N-1's golden answer is deliberately NOT added to this pool (a documented
  simplification, not an oversight -- see "Modeling choice" below); it's
  still appended to the ledger (needed for absolute-position bookkeeping
  further downstream in the conversation), just not offered up for
  re-selection. This makes turn N's final prompt a genuine extension of
  turn N-1's (see `scoring.py`'s docstring for why sorting kept indices by
  absolute position, not by score, is what makes this automatic rather than
  requiring an explicit "don't reorder" rule).

**Modeling choice, not an accident**: DISCARD's candidate pool is defined
here as "last turn's kept tokens + this turn's query" specifically, per
EXPERIMENT_PLAN.md's own wording, not "last turn's kept tokens + last
turn's golden answer + this turn's query." Once history is discarded from
the *scoring* pool it does not come back in any form, including via the
answer that was conditioned on it -- if this later proves too aggressive
(e.g. DISCARD accuracy degrades faster than expected because it can't even
reference its own immediately-preceding answer), revisit by having
`complete_turn` fold the golden answer into `_discard_pool` too; that's a
small, local change confined to this file.

## Force-keep

Each turn's own new query is never subject to pruning -- `begin_turn`
returns it separately (`force_keep_query`) from the prunable
`candidate_pool`, mirroring the single-turn pipeline's force-kept
question/instruction suffix (`predict_longbench_v2.py`'s
`submit_pruned_requests`). `pruner.py` is responsible for actually enforcing
this (scoring the full candidate+query sequence for signal, but only
honoring the scorer's decision within the candidate region) -- this class
only hands back the boundary, it doesn't itself call into any scoring code.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PendingTurn:
    """Bookkeeping for a turn between `begin_turn` and `complete_turn`."""

    query_start: int
    query_end: int


class ConversationState:
    def __init__(
        self,
        conversation_id: str,
        context_token_ids: List[int],
        keep_mode: str,
    ) -> None:
        if keep_mode not in ("keep", "discard"):
            raise ValueError(f"keep_mode must be 'keep' or 'discard', got {keep_mode!r}")
        self.conversation_id = conversation_id
        self.keep_mode = keep_mode

        # Append-only: every token ever presented, in absolute-position order.
        self._ledger: List[int] = list(context_token_ids)

        # DISCARD mode's carried-forward candidate pool -- before turn 1,
        # nothing has been pruned yet, so it starts as the full context.
        # Unused (never read) in KEEP mode, but always maintained so a
        # caller could in principle switch modes on a running conversation
        # (not currently exercised, but no reason to make it invalid).
        self._discard_pool: List[Tuple[int, int]] = [
            (tok_id, pos) for pos, tok_id in enumerate(context_token_ids)
        ]

        self._pending: PendingTurn | None = None
        self.turn_idx = 0  # 0-indexed count of turns completed so far

    @property
    def total_len(self) -> int:
        """Current absolute conversation length -- grows by len(context) at
        construction, then by each turn's query + golden-answer length."""
        return len(self._ledger)

    def begin_turn(
        self, query_token_ids: List[int]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Start a new turn: appends `query_token_ids` to the ledger (so
        their absolute positions are assigned) and returns
        (candidate_pool, force_keep_query), both as ordered
        [(token_id, absolute_position), ...] lists, ascending by position.

        `candidate_pool` is what this turn's scoring/selection should run
        over (see module docstring for KEEP vs. DISCARD); `force_keep_query`
        must be included in this turn's final prompt unconditionally,
        regardless of what scoring decides for `candidate_pool`.

        Must be paired with exactly one `complete_turn` call before the next
        `begin_turn` -- raises if called again while a turn is already
        pending, since `total_len`/the ledger would otherwise be ambiguous
        about whether the previous turn's golden answer was ever appended.
        """
        if self._pending is not None:
            raise RuntimeError(
                f"begin_turn called again for conversation "
                f"{self.conversation_id!r} while turn {self.turn_idx} is "
                f"still pending -- call complete_turn first."
            )

        query_start = len(self._ledger)
        self._ledger.extend(query_token_ids)
        query_end = len(self._ledger)
        self._pending = PendingTurn(query_start, query_end)

        force_keep_query = [
            (tok_id, query_start + i) for i, tok_id in enumerate(query_token_ids)
        ]

        if self.keep_mode == "keep":
            candidate_pool = [
                (tok_id, pos) for pos, tok_id in enumerate(self._ledger[:query_start])
            ]
        else:
            candidate_pool = list(self._discard_pool)

        return candidate_pool, force_keep_query

    def complete_turn(
        self,
        kept_history_pairs: List[Tuple[int, int]],
        golden_answer_token_ids: List[int],
    ) -> None:
        """Finish the pending turn: records `kept_history_pairs` (this
        turn's scoring decision over `candidate_pool`, NOT including the
        force-kept query -- ascending by absolute position, same convention
        as `begin_turn`'s return values) as DISCARD mode's next candidate
        pool, then appends `golden_answer_token_ids` to the ledger so future
        turns' KEEP-mode pool includes them (see module docstring's
        "Modeling choice" for why DISCARD's pool does not include them).

        `golden_answer_token_ids` may be `[]` for a turn whose golden answer
        is unused/unknown (e.g. a driver running in a hypothetical
        self-generated-history mode) -- an empty extend is a no-op on the
        ledger, not an error.
        """
        if self._pending is None:
            raise RuntimeError(
                f"complete_turn called for conversation "
                f"{self.conversation_id!r} with no pending begin_turn."
            )
        pending = self._pending

        if self.keep_mode == "discard":
            query_pairs = [
                (self._ledger[pos], pos)
                for pos in range(pending.query_start, pending.query_end)
            ]
            self._discard_pool = list(kept_history_pairs) + query_pairs

        self._ledger.extend(golden_answer_token_ids)
        self._pending = None
        self.turn_idx += 1

    @property
    def query_span(self) -> Tuple[int, int]:
        """(query_start, query_end) for the currently-pending turn -- for
        callers (pruner.py) that need to know orig_len (== query_end) before
        calling complete_turn."""
        if self._pending is None:
            raise RuntimeError("no turn is currently pending")
        return self._pending.query_start, self._pending.query_end
