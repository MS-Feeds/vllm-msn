"""SpecPrefill conversation-turn pruning driver -- the multi-turn
counterpart of the single-turn pipeline's `pruner.py`, threading
`conversation_state.py`'s absolute-position ledger and KEEP/DISCARD
candidate-pool construction through to `pruning_registry.py`'s
`PruneRecord` (consumed by the unchanged `model_runner.py`/`worker.py` on
the TARGET side).

Called by `predict_scbench.py`, once per conversation turn, in strict
sequence (a conversation's turn N+1 cannot be pruned until turn N's
`ConversationState.complete_turn` has run -- see `conversation_state.py`).

## Force-keep, mirroring the single-turn pipeline's question/instruction fix

The single-turn pipeline's `predict_longbench_v2.py` found (real hardware,
documented there) that letting SpecPrefill's own chunk-based scoring decide
whether to keep the trailing question/instruction produced gibberish at
aggressive keep rates -- the instruction is a tiny fraction of a huge
prompt and has no special protection. The same risk applies here to each
turn's own new query: `conversation_state.py`'s `force_keep_query` is never
subject to scoring, only `candidate_pool` is -- see `_score_and_select`
below for exactly how that's enforced (score the whole
candidate+query sequence for signal, honor the decision only within the
candidate region).

## Speculator-path vs. oracle-path share `_score_and_select`, not
## `compute_pruned_turn` itself

`compute_pruned_turn` (speculator path) drives `SpecPrefillProposer.run_turn`
to get Q from the speculator's own lookahead decode. A second, teacher-forced
path gets Q from a forward pass over the TARGET's own next-turn query,
orchestrated separately by `predict_scbench.py` (it needs the target's own
attention layers hooked, a driver-side concern this module doesn't own) --
but once IT has a `(query_buffer, actual_look_ahead_cnt, key_buffer)` triple
in hand, from whatever source, the scoring/selection math is identical either
way, so `compute_oracle_kept_pairs` below reuses the same `_score_and_select`
helper rather than duplicating the force-keep/chunk-selection logic.

**`compute_oracle_kept_pairs` is not what the ORACLE-k* rows actually run.**
That teacher-forced target-side capture hook was never built; when the oracle
ceiling was wired up, it was wired up a different way -- `predict_scbench.py`
runs the ordinary speculator path with `SpecPrefillProposer` pointed at the
TARGET checkpoint instead of the 1B speculator, which needs no new capture
hook and no forward pass through the target's own (persistent, resumable,
mustn't-be-corrupted) conversation session. `compute_pruned_turn` above
therefore serves both SPARSE-k*-g* and ORACLE-k*. This function is kept as
the ready-made entry point for the teacher-forced variant if it's ever built
(scoring from the golden ANSWER's queries, a strictly stronger oracle than
scoring from a lookahead the target generated itself) -- see
EXPERIMENT_PLAN.md's "Oracle upper bound".
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .config import SpecConfig
from .conversation_state import ConversationState
from .proposer import SpecPrefillProposer
from .scoring import score_and_select_indices

logger = logging.getLogger(__name__)


@dataclass
class PrunedTurnResult:
    pruned_token_ids: List[int]
    kept_positions: List[int]  # absolute conversation-ledger positions
    orig_len: int  # absolute conversation length as of this turn's query end
    kept_history_pairs: List[Tuple[int, int]]  # for ConversationState.complete_turn
    actual_look_ahead_cnt: int
    num_cached_tokens: int  # 0 for the oracle path (no speculator cache involved)


def _positions_from_kept_indices(
    candidate_pool: List[Tuple[int, int]],
    force_keep_query: List[Tuple[int, int]],
    kept_local_indices: Optional[List[int]],
) -> Tuple[List[int], List[int], int, List[Tuple[int, int]]]:
    """Converts a scorer's chosen LOCAL indices (0..len(candidate_pool)+
    len(force_keep_query)-1) into the four `PrunedTurnResult` fields --
    shared by both the speculator path (`compute_pruned_turn`, indices
    already computed IN-PROCESS by `speculator_worker.py`'s
    `end_capture_and_score`, see that method's docstring for why) and the
    oracle path (`_score_and_select` below, indices computed here
    driver-side from the target's own Q/K).

    `kept_local_indices=None` means "no scoring signal at all" (real
    callers only pass None when `actual_look_ahead_cnt == 0` -- EOS on the
    very first candidate token, or `look_ahead_cnt=0`) -- keeps the full,
    unpruned candidate pool + query rather than scoring with nothing to
    score on, logging a warning since this is a real, if rare, degraded
    case worth knowing about.

    Returns (pruned_token_ids, kept_positions, orig_len, kept_history_pairs)
    -- see `PrunedTurnResult`'s fields (`orig_len` computed here from
    `force_keep_query`'s own positions, not passed in, since both callers
    already have it via that list).
    """
    full_sequence = candidate_pool + force_keep_query
    candidate_len = len(candidate_pool)
    # query_start == candidate_len by construction (force_keep_query always
    # comes right after candidate_pool in the submitted sequence -- see
    # proposer.py's docstring on local-position numbering).
    orig_len = force_keep_query[-1][1] + 1 if force_keep_query else candidate_pool[-1][1] + 1

    if kept_local_indices is None:
        logger.warning(
            "SpecPrefill: 0 lookahead steps completed for this turn (EOS on "
            "the very first candidate token, or look_ahead_cnt=0) -- keeping "
            "the full, unpruned candidate pool + query rather than scoring "
            "with no signal."
        )
        pruned_token_ids = [tid for tid, _ in full_sequence]
        kept_positions = [pos for _, pos in full_sequence]
        return pruned_token_ids, kept_positions, orig_len, list(candidate_pool)

    # Force-keep: only honor the scorer's decision within [0, candidate_len)
    # -- see module docstring's "Force-keep" section. Indices >= candidate_len
    # (the query region) are discarded here, not because they're unimportant,
    # but because force_keep_query is unconditionally appended below anyway.
    kept_history_local = [i for i in kept_local_indices if i < candidate_len]
    kept_history_pairs = [candidate_pool[i] for i in kept_history_local]

    pruned_token_ids = [tid for tid, _ in kept_history_pairs] + [
        tid for tid, _ in force_keep_query
    ]
    kept_positions = [pos for _, pos in kept_history_pairs] + [
        pos for _, pos in force_keep_query
    ]
    return pruned_token_ids, kept_positions, orig_len, kept_history_pairs


def _score_and_select(
    candidate_pool: List[Tuple[int, int]],
    force_keep_query: List[Tuple[int, int]],
    query_buffer: List[torch.Tensor],
    key_buffer_per_layer: List[torch.Tensor],
    actual_look_ahead_cnt: int,
    spec_config: SpecConfig,
) -> Tuple[List[int], List[int], int, List[Tuple[int, int]]]:
    """Oracle-path scoring core: computes indices driver-side (the oracle
    path has no in-process speculator worker to run scoring inside -- its
    Q/K come from a teacher-forced forward pass on the TARGET's own
    attention layers, hooked by `predict_scbench.py`, not from
    `proposer.py` at all), then delegates the shared index -> position
    conversion to `_positions_from_kept_indices`.

    `key_buffer_per_layer[layer_idx]` is a single [full_len, num_kv_heads,
    head_size] tensor (one sample) -- see `score_and_select_indices`'s own
    docstring for the wrapping this passes through.
    """
    kept_local_indices = (
        None
        if actual_look_ahead_cnt == 0
        else score_and_select_indices(
            query_buffer, key_buffer_per_layer, actual_look_ahead_cnt, spec_config
        )
    )
    return _positions_from_kept_indices(candidate_pool, force_keep_query, kept_local_indices)


def compute_pruned_turn(
    proposer: SpecPrefillProposer,
    spec_config: SpecConfig,
    conversation_state: ConversationState,
    query_token_ids: List[int],
) -> PrunedTurnResult:
    """Runs one turn of the speculator-based pruning path: asks
    `conversation_state` for this turn's candidate pool (KEEP/DISCARD, see
    that module), submits it to the speculator via
    `SpecPrefillProposer.run_turn_and_score`, which does K retrieval AND
    scoring IN-PROCESS inside the speculator's own worker (see that
    method's docstring, and `speculator_worker.py::end_capture_and_score`'s,
    for why: a first real run measured the OLD design -- ship Q back here,
    separately ask for K, score locally -- moving up to 720M K elements
    across `collective_rpc`'s process boundary for a single turn, taking
    89s on its own. Only the tiny resulting index list crosses the
    boundary now).

    No `eos_token_id` parameter (a previous version had one -- it was
    silently unused, since real early-stopping is controlled entirely by
    `SamplingParams.ignore_eos`, not any id passed around out-of-band; see
    `proposer.py::run_turn`'s docstring for the real-hardware finding this
    fixes). Early stopping is instead controlled by `spec_config.ignore_eos`
    directly, threaded straight into `proposer.run_turn_and_score`.

    Does NOT call `conversation_state.complete_turn` -- the caller
    (`predict_scbench.py`) must do that itself once it also knows this
    turn's golden-answer tokens to append to the ledger (this function has
    no dataset access), passing back this result's `kept_history_pairs`.
    Does NOT register the `PruneRecord` or call `add_request` either -- see
    `prune_and_add_turn` below for the combined, race-safe version (mirrors
    the single-turn pipeline's `compute_pruned_prompt` vs.
    `prune_and_add_request` split, for the same reason: some callers need to
    check the kept-token count against a budget before committing to
    `add_request`, same as `predict_longbench_v2.py`'s `submit_pruned_requests`).
    """
    candidate_pool, force_keep_query = conversation_state.begin_turn(query_token_ids)
    full_sequence = candidate_pool + force_keep_query
    full_token_ids = [tid for tid, _ in full_sequence]

    kept_local_indices, actual_look_ahead_cnt, num_cached_tokens = proposer.run_turn_and_score(
        conversation_salt=conversation_state.conversation_id,
        turn_idx=conversation_state.turn_idx,
        full_sequence_token_ids=full_token_ids,
        look_ahead_cnt=spec_config.look_ahead_cnt,
        pool_kernel_size=spec_config.pool_kernel_size,
        keep_kwargs=spec_config.keep_kwargs,
        ignore_eos=spec_config.ignore_eos,
        # Scoring variants (ACCURACY_IMPROVEMENTS.md §1) -- threaded from the
        # driver's SpecConfig because the speculator's worker process has no
        # access to it, same reason `keep_kwargs`/`pool_kernel_size` are
        # passed rather than read.
        score_aggregation=spec_config.score_aggregation,
        score_layers=spec_config.score_layers,
        score_head_set=spec_config.score_head_set,
    )

    pruned_token_ids, kept_positions, orig_len, kept_history_pairs = _positions_from_kept_indices(
        candidate_pool, force_keep_query, kept_local_indices
    )

    return PrunedTurnResult(
        pruned_token_ids=pruned_token_ids,
        kept_positions=kept_positions,
        orig_len=orig_len,
        kept_history_pairs=kept_history_pairs,
        actual_look_ahead_cnt=actual_look_ahead_cnt,
        num_cached_tokens=num_cached_tokens,
    )


def prune_and_add_turn(
    llm_engine,
    request_id: str,
    result: PrunedTurnResult,
    sampling_params,
) -> str:
    """Registers `result`'s `PruneRecord` into the TARGET Worker's process
    via `collective_rpc` BEFORE calling `add_request` -- same ordering, and
    for the same reason (a confirmed real race on the single-turn pipeline,
    documented in the single-turn `pruner.py`'s module docstring: a
    fire-and-forget `add_request()` can be scheduled and stepped before a
    LATER `collective_rpc` registration call lands), as the single-turn
    pipeline's `prune_and_add_request`.

    Caller is responsible for having already resolved `request_id` to
    something unique within this engine's lifetime (e.g.
    `f"{conversation_id}::turn{turn_idx}"`, mirroring the speculator's own
    request-id convention in `proposer.py`, though the target's own
    `pruning_registry`/`model_runner.py` don't parse it -- any unique string
    works on the target side).
    """
    from vllm.inputs import TokensPrompt

    llm_engine.collective_rpc(
        "register_prune_record",
        args=(request_id, result.kept_positions, result.orig_len),
    )

    prompt = TokensPrompt(prompt_token_ids=result.pruned_token_ids, cache_salt=request_id)
    real_request_id = llm_engine.add_request(request_id, prompt, sampling_params)
    assert real_request_id == request_id, (
        f"expected add_request() to use request_id={request_id!r} verbatim "
        f"(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 must be set -- see "
        f"proposer.py's module-import-time default) but got "
        f"{real_request_id!r} back instead -- the PruneRecord above was "
        f"registered under the wrong id."
    )
    return real_request_id


def compute_oracle_kept_pairs(
    spec_config: SpecConfig,
    conversation_state: ConversationState,
    query_token_ids: List[int],
    query_buffer: List[torch.Tensor],
    key_buffer_per_layer: List[torch.Tensor],
    actual_look_ahead_cnt: int,
) -> PrunedTurnResult:
    """Teacher-forced oracle path -- **currently unused**, see this module's
    docstring: the wired-up ORACLE-k* rows get their target-quality scores
    by running `compute_pruned_turn` against a proposer that wraps the
    TARGET checkpoint, not through here. Kept for the stronger variant it
    enables (Q captured from the golden ANSWER's own queries rather than
    from a lookahead the scorer generated itself).

    Identical scoring/selection to `compute_pruned_turn`, but the caller
    (`predict_scbench.py`) would supply `query_buffer`/`key_buffer_per_layer`
    from the TARGET model's own attention (a teacher-forced forward pass
    over the NEXT turn's already-known golden query, hooked the same way
    `speculator_worker.py` hooks the speculator -- that orchestration does
    not exist yet in `predict_scbench.py`) instead of the speculator's.

    Still calls `conversation_state.begin_turn` (so the ledger/candidate
    pool bookkeeping stays correct for whichever KEEP/DISCARD mode this
    oracle run is evaluating against -- EXPERIMENT_PLAN.md's MVP scope
    pairs the oracle with KEEP mode only, see that file), but never touches
    `proposer.py`/the speculator engine at all -- `num_cached_tokens` is
    always 0 here (no speculator cache involved; not a meaningful metric
    for this path).
    """
    candidate_pool, force_keep_query = conversation_state.begin_turn(query_token_ids)
    pruned_token_ids, kept_positions, orig_len, kept_history_pairs = _score_and_select(
        candidate_pool,
        force_keep_query,
        query_buffer,
        key_buffer_per_layer,
        actual_look_ahead_cnt,
        spec_config,
    )
    return PrunedTurnResult(
        pruned_token_ids=pruned_token_ids,
        kept_positions=kept_positions,
        orig_len=orig_len,
        kept_history_pairs=kept_history_pairs,
        actual_look_ahead_cnt=actual_look_ahead_cnt,
        num_cached_tokens=0,
    )
