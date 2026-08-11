"""SpecPrefillProposer -- driver-facing orchestration for the multi-turn
speculator, Algorithm 1 lines 3-11's conversation-aware counterpart.

**Structurally different from the single-turn pipeline's `proposer.py`**,
which loaded the speculator standalone (`get_model()`, bypassing the engine
entirely) and manually drove a fixed-shape lookahead loop against a
throwaway per-call dummy KV cache. This version runs the speculator as a
genuine persistent `vllm.LLM(enable_prefix_caching=True)` (see
`speculator_worker.py`'s module docstring for the full "why a real engine"
reasoning) -- this class's job is reduced to: submit one request per turn,
detect when that request's bootstrap prefill has finished (so query capture
can be switched on for exactly the lookahead decode steps, not the prefill
itself -- mirroring the single-turn pipeline's "discard the bootstrap's own
capture" rule, just achieved by *timing* the RPC call instead of discarding
a buffer), and retrieve the resulting Q/K via `collective_rpc` into
`speculator_worker.py`'s `SpeculatorWorker`.

## Why the speculator never needs a RoPE-position override (unlike the target)

Every request this class ever submits to the speculator is a real,
contiguous, never-pruned prompt -- `pruner.py`'s `conversation_state.py`
candidate pool (whatever KEEP/DISCARD produces) concatenated with the new
turn's query, submitted verbatim. The speculator only ever *scores*
candidates for the TARGET to prune; it never prunes its own input. So it
gets ordinary, contiguous 0..N-1 positions from vLLM automatically, no
`worker_cls` position-patching needed -- `speculator_worker.py`'s
`SpeculatorWorker` exists only for query-capture and KV read-back, not for
anything RoPE-related.

## Local positions vs. absolute conversation positions -- two different
## numberings, both real, easy to conflate

`conversation_state.py`'s absolute ledger positions (used for the TARGET's
`PruneRecord`) are NOT the same numbers as the speculator's own LOCAL
prompt positions used here. A turn's candidate pool is `[(token_id,
absolute_ledger_position), ...]`; when concatenated with the query and
submitted to the speculator as `prompt_token_ids`, vLLM assigns that
sequence ordinary LOCAL positions `0..len(sequence)-1` -- unrelated to the
`absolute_ledger_position` values riding along in the same list. This class
only ever deals in LOCAL positions (needed to ask `speculator_worker.py`
"give me K for local positions [3, 7, 19, ...]"); `pruner.py` is the only
place that needs to translate a scoring result (expressed in local indices)
back into `absolute_ledger_position`s for `PruneRecord`, via the
`(token_id, absolute_position)` pairs `conversation_state.py` already
carries alongside each local index. **Why this works even though DISCARD
mode's candidate pool is a non-contiguous subsequence of the true history**
(see `conversation_state.py`'s docstring): the speculator is never told
"here are the true absolute positions, with gaps" -- it's simply handed
a compacted, contiguous LOCAL sequence (gaps already removed) and scores
it as an ordinary prompt. Local position `i` in that submission is
whatever token conversation_state.py's candidate pool put at index `i`,
full stop; the speculator has no notion of "gap" at all.

## One request per turn, one in-flight speculator request at a time (MVP scope)

`run_turn` drives its own request to completion before returning -- this
pipeline never submits a second conversation's speculator request while one
is still in flight. This sacrifices some throughput (no cross-conversation
batching on the speculator's own engine) in exchange for a much simpler,
easier-to-reason-about driving loop; `speculator_worker.py`'s
request-id-keyed (not global) query buffer and RPC surface would support
concurrent in-flight requests if this were relaxed later, but that's not
attempted here -- flagged as a real, deliberate scope boundary for this
pass, not an oversight (mirrors the single-turn pipeline's own
`tensor_model_parallel_size=1`-only scoping in spirit, just a batching axis
instead of a parallelism one).
"""

import os
from typing import List, Optional, Tuple

import torch

from .kv_cache_utils import gather_keys_for_slots  # noqa: F401  (re-exported for callers that want the raw utility)

# Same reasoning/placement as the single-turn pipeline's pruner.py: must be
# set before any add_request() call reads it. Both the speculator's and the
# target's engines share this pipeline's need for caller-controlled,
# non-randomized request ids (the speculator so `request_id` reliably
# encodes `conversation_salt` via the "{salt}::turn{n}" convention
# `speculator_worker.py` parses; the target for the pre-add_request
# PruneRecord-registration race documented in pruner.py).
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")


class SpecPrefillProposer:
    def __init__(
        self,
        speculator_model_path: str,
        device: torch.device,
        gpu_memory_utilization: float = 0.2,
        **extra_llm_kwargs,
    ) -> None:
        """Builds the persistent speculator engine. `gpu_memory_utilization`
        defaults low (0.2) -- the speculator (Llama-3.2-1B) is small and,
        per EXPERIMENT_PLAN.md's KEEP-mode risk note, needs enough headroom
        that a long-running conversation's own growing KV cache isn't the
        first thing evicted under pressure from whatever else shares this
        GPU; raise it if `num_cached_tokens` logging (see `pruner.py`)
        shows real eviction happening in practice.

        `enforce_eager=True` for the same reason as the single-turn
        pipeline's speculator: `@support_torch_compile`-wrapped forward
        can't trace through the `functools.partial`-wrapped query-capture
        hook installed in `speculator_worker.py`.
        """
        from vllm import LLM

        self.device = device
        llm_kwargs = dict(
            model=speculator_model_path,
            trust_remote_code=True,
            worker_cls="vllm_patch.speculator_worker.SpeculatorWorker",
            enable_prefix_caching=True,
            enforce_eager=True,
            disable_log_stats=False,  # needed for RequestOutput.num_cached_tokens
            gpu_memory_utilization=gpu_memory_utilization,
            device=str(device),
        )
        llm_kwargs.update(extra_llm_kwargs)
        self.llm = LLM(**llm_kwargs)
        self.llm_engine = self.llm.llm_engine

    def run_turn(
        self,
        conversation_salt: str,
        turn_idx: int,
        full_sequence_token_ids: List[int],
        look_ahead_cnt: int,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], int, int]:
        """Submit one turn's full (candidate_pool + query) sequence as a
        fresh request under `conversation_salt`'s stable cache lineage,
        drive it through its bootstrap prefill (a genuine prefix-cache-hit
        short-prefill step for KEEP mode's cache-hitting portion -- see this
        module's docstring on why DISCARD mode instead recomputes from
        scratch each turn, which is expected, not a bug) plus
        `look_ahead_cnt` lookahead decode steps, capturing queries only for
        the decode portion.

        Returns:
            (query_buffer, actual_look_ahead_cnt, num_cached_tokens) --
            query_buffer is per-layer [1, actual_look_ahead_cnt, H*D]
            tensors on `self.device` (empty steps if actual_look_ahead_cnt
            is 0 -- caller must short-circuit, same rule as the single-turn
            pipeline: aggregating over zero lookahead steps produces silent
            NaN, not an error). `num_cached_tokens` is vLLM's own
            already-computed count of how many of this request's prompt
            tokens were served from the prefix cache -- log this per turn
            (see `pruner.py`) as the real, measured "only prefill new
            tokens" signal EXPERIMENT_PLAN.md's KEEP-mode risk note calls
            for, rather than assuming it.
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        request_id = f"{conversation_salt}::turn{turn_idx}"
        prompt = TokensPrompt(
            prompt_token_ids=full_sequence_token_ids, cache_salt=conversation_salt
        )
        sampling_params = SamplingParams(max_tokens=1 + look_ahead_cnt, temperature=0.0)
        real_request_id = self.llm_engine.add_request(request_id, prompt, sampling_params)
        assert real_request_id == request_id, (
            f"expected add_request() to use request_id={request_id!r} verbatim "
            f"(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 is set at this module's "
            f"import time) but got {real_request_id!r} back instead."
        )

        capturing = False
        num_cached_tokens = 0
        # Same "never break early / never abort" discipline as
        # predict_longbench_v2.py's drive_engine_to_completion -- an
        # unconditional step() without checking has_unfinished_requests()
        # first can block forever once this (the only in-flight) request
        # finishes.
        while self.llm_engine.has_unfinished_requests():
            for output in self.llm_engine.step():
                if output.request_id != request_id:
                    # Shouldn't happen under this pass's one-in-flight-
                    # request scope (see module docstring), but don't
                    # silently mis-handle another request's output if it
                    # does -- just ignore it, it'll be driven to completion
                    # by whatever submitted it.
                    continue
                if output.num_cached_tokens:
                    num_cached_tokens = output.num_cached_tokens
                if not capturing and len(output.outputs[0].token_ids) >= 1:
                    # The forward pass that just produced this first token
                    # (the bootstrap prefill's own next-token prediction)
                    # has already happened WITHOUT capture enabled -- this
                    # is what makes capture start "after the bootstrap",
                    # the same rule the single-turn pipeline enforces by
                    # discarding a buffer instead. Every step from here
                    # onward is a genuine one-new-token decode step.
                    self.llm_engine.collective_rpc("begin_capture", args=(request_id,))
                    capturing = True

        # collective_rpc returns one result per worker (TP ranks) -- this
        # pipeline is TP=1 only (see module docstring), so unwrap index 0.
        query_buffer_cpu = self.llm_engine.collective_rpc("end_capture", args=(request_id,))[0]
        query_buffer = [q.to(self.device) for q in query_buffer_cpu]
        actual_look_ahead_cnt = query_buffer[0].shape[1] if query_buffer else 0
        return query_buffer, actual_look_ahead_cnt, num_cached_tokens

    def retrieve_keys(
        self, conversation_salt: str, local_positions: List[int]
    ) -> List[torch.Tensor]:
        """Per-layer [len(local_positions), num_kv_heads, head_size] K
        tensors on `self.device`, for LOCAL positions within
        `conversation_salt`'s own speculator prompt (see module docstring's
        "Local positions vs. absolute conversation positions")."""
        keys_cpu = self.llm_engine.collective_rpc(
            "retrieve_keys", args=(conversation_salt, local_positions)
        )[0]
        return [k.to(self.device) for k in keys_cpu]

    def discard_conversation(self, conversation_salt: str) -> None:
        """Call once a whole conversation (not just one turn) is finished,
        to bound `speculator_worker.py`'s per-conversation slot-history
        accumulation across a long benchmark run."""
        self.llm_engine.collective_rpc("discard_conversation", args=(conversation_salt,))
