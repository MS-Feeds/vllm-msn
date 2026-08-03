"""SpecPrefill pruning driver entry point — Algorithm 1 lines 1-16 orchestration
for one request, plus registering the position-restoration data lines 18-19
need (consumed on the runner side by `model_runner.py`).

Called by a benchmark driver script **before** `LLMEngine.add_request()` — not
by any code running inside the vLLM engine process's per-step loop. See the
approved plan's "Where pruning is triggered: before the Request even exists".

**Correction (2026-07-22, confirmed on real hardware)**: the approved plan
originally assumed the driver and `EngineCore`/`Worker` share one OS process
under the default single-GPU `UniProcExecutor`, making a plain
`pruning_registry.register(...)` call here sufficient. That assumption was
wrong -- `EngineCore` always runs in its own separate process (visible via
the `(EngineCore pid=...)` log prefix), regardless of executor topology. A
direct `pruning_registry.register(...)` call in this module would only
populate the *driver* process's copy of that module, invisible to
`model_runner.py` (which runs inside the Worker's process). `prune_and_add_request`
below instead pushes the record via `llm_engine.collective_rpc(
"register_prune_record", ...)`, which calls a real method on
`SpecPrefillWorker` (`worker.py`) running inside the correct process -- see
that method's docstring for the full reasoning.

**Correction #2 (2026-07-28, confirmed on real hardware, intermittent --
about 1 in 4 runs)**: registering the `PruneRecord` *after* calling
`add_request()` (using its returned, vLLM-rewritten id) has its own race,
independent of the process-separation issue above. `add_request()` sends a
fire-and-forget message to `EngineCore`, which runs its own continuous
execution loop and schedules+steps newly-added requests autonomously --
entirely decoupled from whether some *separate* `collective_rpc` call the
driver happens to issue afterward has completed. There is no engine-level
ordering guarantee between "this request is admitted and may be stepped"
and "this application-level side-channel RPC has landed in the Worker."
Confirmed via timestamped logging: a step already executed a newly-added
request's entire (single-chunk) prefill using stock positions at
`t=723.527`, while the registration RPC's own worker-side handler didn't
run until `t=723.922` -- 400ms later, in the same process. By the time
registration became visible, the request was already past its prefill
boundary and the override never got a chance to apply.

Fixed by reordering: register the record *before* calling `add_request()`
at all, using a request id the caller fully controls rather than vLLM's own
post-hoc rewritten one -- see `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` below
and `prune_and_add_request`'s own docstring. This isn't just "call things
in the order that happened to work" -- `collective_rpc()` is a genuinely
blocking round-trip (confirmed via `vllm/v1/engine/core_client.py`'s
`SyncMPClient.call_utility`, which blocks on `future.result()`), so by
construction nothing can schedule this request before the registration
call this function makes returns, because the request doesn't exist from
EngineCore's perspective until the *later* `add_request()` call sends it.
"""

import os
from typing import List, Optional, Tuple

import torch
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine

from .config import SpecConfig
from .proposer import SpecPrefillProposer
from .scoring import (
    aggregate_attention_score,
    chunk_select_from_smoothed_attention,
    compute_attention_score,
)

logger = init_logger(__name__)

# Must be set before LLMEngine.add_request() -> InputProcessor.assign_request_id
# first reads it (that read happens in the DRIVER process, the same process
# this module runs in -- no cross-process propagation needed, unlike the
# VLLM_ALLOW_INSECURE_SERIALIZATION/VLLM_USE_V2_MODEL_RUNNER flags the
# validation scripts set for the EngineCore subprocess). Without this,
# add_request() unconditionally appends a random 8-character suffix to
# whatever request_id the caller supplies, making it impossible to know the
# real id in advance -- which is exactly what prune_and_add_request below
# needs to register a PruneRecord *before* calling add_request() at all (see
# module docstring's "Correction #2" for why that ordering, not the reverse,
# is what actually closes the registration-vs-scheduling race). Callers must
# themselves guarantee request_id uniqueness now (predict_longbench_v2.py's
# `f"lbv2-{sample['id']}-{i}"` and validate_runner_integration.py's fixed
# per-step literal strings both already are, within one engine's lifetime).
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")


def _last_token_only(sampled_token_ids: torch.Tensor) -> torch.Tensor:
    """Only the prediction at the LAST position is a real autoregressive
    continuation -- earlier positions were teacher-forced during prefill
    (the model saw the true prompt token there, not its own prediction) and
    must be discarded, not fed back in."""
    return sampled_token_ids[-1:].int()


def _make_next_positions_fn(prompt_len: int):
    def _next_positions_fn(positions: torch.Tensor, step: int) -> torch.Tensor:
        # First decode token continues right after the prompt; each
        # subsequent step advances by one. See run_lookahead_steps's loop:
        # this is called once per completed step to produce the position for
        # the *next* step's single-token input.
        return torch.tensor(
            [prompt_len + step], dtype=torch.int64, device=positions.device
        )

    return _next_positions_fn


def compute_pruned_prompt(
    proposer: SpecPrefillProposer,
    spec_config: SpecConfig,
    prompt_token_ids: List[int],
    device: torch.device,
    head_dim: int,
    eos_token_id: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Runs algorithm lines 3-16 for one prompt: speculator lookahead loop,
    Q/K retrieval, importance scoring, chunk selection.

    Returns:
        (pruned_token_ids, kept_positions) -- kept_positions[i] is the
        original index in `prompt_token_ids` of `pruned_token_ids[i]`
        (this is T, algorithm line 16's output).
    """
    prompt_len = len(prompt_token_ids)
    look_ahead_cnt = spec_config.look_ahead_cnt

    lookahead_meta = proposer.build_lookahead_metadata(prompt_len, look_ahead_cnt, head_dim)

    input_ids = torch.tensor(prompt_token_ids, dtype=torch.int32, device=device)
    positions = torch.arange(prompt_len, dtype=torch.int64, device=device)

    query_buffer = proposer.run_lookahead_steps(
        initial_input_ids=input_ids,
        initial_positions=positions,
        look_ahead_cnt=look_ahead_cnt,
        prefill_attn_metadata=lookahead_meta.prefill_attn_metadata,
        prefill_slot_mapping=lookahead_meta.prefill_slot_mapping,
        per_step_attn_metadata=lookahead_meta.per_step_attn_metadata,
        per_step_slot_mapping=lookahead_meta.per_step_slot_mapping,
        next_input_fn=_last_token_only,
        next_positions_fn=_make_next_positions_fn(prompt_len),
        eos_token_id=eos_token_id,
    )

    # Actual steps completed (may be < look_ahead_cnt on early EOS) -- one
    # sample (this single prompt) in this batch.
    actual_look_ahead_cnts = [query_buffer[0].shape[1]]

    # Zero real lookahead steps ran (EOS fired on the bootstrap prefill's own
    # sampled token, or look_ahead_cnt=0 was configured) -- confirmed by
    # direct execution that feeding this into aggregate_attention_score's
    # `attn.mean(0)` over an empty step axis produces an all-NaN tensor with
    # NO exception, and chunk_select_from_smoothed_attention's torch.topk on
    # that NaN tensor returns an arbitrary, implementation-defined "kept"
    # set with no error signal -- a silent-wrong-answer footgun in the exact
    # path that decides what the target model gets to see. Short-circuit to
    # keeping the whole prompt instead of feeding this into scoring at all.
    if actual_look_ahead_cnts[0] == 0:
        logger.warning(
            "SpecPrefill: 0 lookahead steps completed for this request (EOS "
            "on the speculator's very first candidate token, or "
            "look_ahead_cnt=0) -- keeping the full, unpruned prompt rather "
            "than scoring with no signal."
        )
        return list(prompt_token_ids), list(range(prompt_len))

    gathered_qk = proposer.tp_gather_qk(query_buffer)
    per_sample_slot_mapping = [lookahead_meta.slot_mapping]
    _, key_buffer = proposer.retrieve_qk(
        gathered_qk,
        per_sample_slot_mapping,
        lookahead_meta.block_size,
        lookahead_meta.num_kv_heads,
        head_dim,
    )

    attn_scores = compute_attention_score(gathered_qk, key_buffer, actual_look_ahead_cnts)
    token_importance = aggregate_attention_score(attn_scores, spec_config)
    kept_indices = chunk_select_from_smoothed_attention(token_importance, spec_config)[0]

    kept_positions = kept_indices.tolist()
    pruned_token_ids = [prompt_token_ids[i] for i in kept_positions]
    return pruned_token_ids, kept_positions


def prune_and_add_request(
    llm_engine: LLMEngine,
    request_id: str,
    prompt_token_ids: List[int],
    sampling_params: SamplingParams,
    proposer: SpecPrefillProposer,
    spec_config: SpecConfig,
    device: torch.device,
    head_dim: int,
    eos_token_id: Optional[int] = None,
) -> Tuple[str, List[int], int]:
    """Full driver-facing entry point for one request: prune (lines 3-16),
    push the PruneRecord the runner subclass needs for RoPE-position
    restoration (lines 18-19) into the Worker's own process (see module
    docstring -- NOT the same process as this function runs in) *before*
    the request exists from the engine's perspective at all, then call
    `add_request` with the pruned prompt and `cache_salt` set to isolate
    this request's prefix-cache block hashes (see the approved plan's
    "Prefix caching" section -- pruned requests must never coincidentally
    share a block hash with another request, since the hash doesn't account
    for position).

    **History (two real, confirmed-on-hardware bugs, both about
    `request_id` identity -- see module docstring's "Correction #2" for the
    second one's full timing evidence):**

    1. (2026-07-23) `LLMEngine.add_request()` (via `InputProcessor.
       assign_request_id`) unconditionally rewrites the caller-supplied
       `request_id`, appending a random suffix to guarantee uniqueness.
       Registering under the caller-supplied id (not the rewritten one)
       meant the lookup never matched for *any* pruned request.
    2. (2026-07-28) The fix for #1 -- call `add_request()` first, register
       under its *returned* (rewritten) id afterward -- introduced a
       different race: `add_request()`'s message to `EngineCore` is
       fire-and-forget, and `EngineCore` schedules+steps newly-added
       requests autonomously, independent of whether a later, separate
       `collective_rpc` call has completed. About 1 in 4 runs, a step
       already executed the request's entire prefill with stock positions
       before the registration RPC landed.

    Fixed by eliminating the ordering dependency entirely: this module sets
    `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` at import time, so
    `add_request()` uses the caller-supplied `request_id` **verbatim** --
    letting this function register the `PruneRecord` under that *known*
    id *before* the request is sent to `EngineCore` at all. Since
    `collective_rpc()` is a genuinely blocking round-trip, by construction
    nothing can schedule this request before this function's registration
    call returns, because `add_request()` (which is what makes the request
    exist from `EngineCore`'s perspective) hasn't been called yet.

    Call this instead of `llm_engine.add_request(...)` directly for any
    request that should go through SpecPrefill pruning. Caller-supplied
    `request_id` must be unique within this engine's lifetime (vLLM's own
    randomization, which used to paper over duplicates, is disabled).

    Returns:
        (request_id, kept_positions, orig_len) -- `request_id` is returned
        unchanged (no longer rewritten by vLLM), included for interface
        compatibility with callers that use the return value to reference
        this request later (e.g. `abort_request`).
    """
    orig_len = len(prompt_token_ids)
    pruned_token_ids, kept_positions = compute_pruned_prompt(
        proposer, spec_config, prompt_token_ids, device, head_dim, eos_token_id
    )

    # Must reach the Worker's own process -- see module docstring and
    # SpecPrefillWorker.register_prune_record's docstring (worker.py). A
    # direct pruning_registry.register(...) call here would silently do
    # nothing, since this function runs in the driver process. Registered
    # BEFORE add_request() below -- see this function's own docstring and
    # the module docstring's "Correction #2" for why this ordering (not the
    # reverse) is what actually closes the registration-vs-scheduling race.
    llm_engine.collective_rpc(
        "register_prune_record", args=(request_id, kept_positions, orig_len)
    )

    prompt = TokensPrompt(prompt_token_ids=pruned_token_ids, cache_salt=request_id)
    real_request_id = llm_engine.add_request(request_id, prompt, sampling_params)
    assert real_request_id == request_id, (
        f"expected add_request() to use request_id={request_id!r} verbatim "
        f"(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 is set at this module's "
        f"import time) but got {real_request_id!r} back instead -- the "
        f"PruneRecord above was registered under the wrong id."
    )

    return real_request_id, kept_positions, orig_len
