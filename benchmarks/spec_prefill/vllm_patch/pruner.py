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

"""

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
    docstring -- NOT the same process as this function runs in), then call
    `add_request` with the pruned prompt and `cache_salt` set to isolate
    this request's prefix-cache block hashes (see the approved plan's
    "Prefix caching" section -- pruned requests must never coincidentally
    share a block hash with another request, since the hash doesn't account
    for position).

    Call this instead of `llm_engine.add_request(...)` directly for any
    request that should go through SpecPrefill pruning.

    Returns:
        (request_id, kept_positions, orig_len) -- the caller may want these
        for logging/verification. Deliberately NOT read back via
        `pruning_registry.get(request_id)` afterward -- that would read the
        *driver* process's copy of the module, which this function never
        writes to (confirmed on real hardware this is a real trap, not
        hypothetical: see `validate_runner_integration.py`'s history for the
        exact failure -- `pruning_registry.get()` in the driver always
        returns None here, by design, since the record only exists in the
        Worker's process).
    """
    orig_len = len(prompt_token_ids)
    pruned_token_ids, kept_positions = compute_pruned_prompt(
        proposer, spec_config, prompt_token_ids, device, head_dim, eos_token_id
    )

    # Must reach the Worker's own process -- see module docstring and
    # SpecPrefillWorker.register_prune_record's docstring (worker.py). A
    # direct pruning_registry.register(...) call here would silently do
    # nothing, since this function runs in the driver process.
    llm_engine.collective_rpc(
        "register_prune_record", args=(request_id, kept_positions, orig_len)
    )

    prompt = TokensPrompt(prompt_token_ids=pruned_token_ids, cache_salt=request_id)
    llm_engine.add_request(request_id, prompt, sampling_params)
    return request_id, kept_positions, orig_len
