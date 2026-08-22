"""SpecPrefillProposer -- driver-facing orchestration for the multi-turn
speculator, Algorithm 1 lines 3-11's conversation-aware counterpart.

**Structurally different from the single-turn pipeline's `proposer.py`**,
which loaded the speculator standalone (`get_model()`, bypassing the engine
entirely) and manually drove a fixed-shape lookahead loop against a
throwaway per-call dummy KV cache. This version runs the speculator as a
genuine persistent `vllm.LLM(enable_prefix_caching=True)` (see
`speculator_worker.py`'s module docstring for the full "why a real engine"
reasoning) -- this class's job is reduced to: submit one request per turn
(registering capture BEFORE `add_request`, not reactively -- see
`speculator_worker.py::begin_capture`'s docstring for a real race this
fixes), drive it to completion, and retrieve the resulting Q/K via
`collective_rpc` into `speculator_worker.py`'s `SpeculatorWorker`, which is
responsible for excluding the bootstrap prefill's own capture from the
returned decode-only stack (by shape, not by capture timing -- see
`speculator_worker.py::end_capture`'s docstring) -- mirroring the
single-turn pipeline's "discard the bootstrap's own capture" rule, just
implemented as a filter on the way out instead of a timing gate on the way
in.

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
import time
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

# Lets a raw torch.Tensor cross collective_rpc's process boundary (driver
# <-> EngineCore) as an actual zero-copy tensor buffer, instead of the
# `tensor_to_wire`/`tensor_from_wire` Python-list round trip this module
# used to require -- see `end_capture`/`retrieve_keys` below for what this
# replaces, and their docstrings for the exact vllm/v1/serial_utils.py code
# path this depends on (traced, not assumed).
#
# **Why this is safe to set here, even though the flag's own name is
# "insecure"**: `MsgpackEncoder.enc_hook` always encodes a raw `torch.Tensor`
# via its OWN dedicated, lossless `_encode_tensor` path regardless of this
# flag -- Tensors are never pickled. What the flag actually gates is (1)
# `UtilityResult` (the wrapper `collective_rpc`'s return value travels in)
# being allowed to carry TYPE INFO for its nested contents at all -- without
# it, a `List[torch.Tensor]` result decodes back as a bare nested list with
# no way to know it was ever a Tensor (confirmed on real hardware: this is
# the literal `AttributeError: 'list' object has no attribute 'to'` the
# previous wire-encoding workaround existed to avoid), and (2), for OTHER,
# non-Tensor/ndarray/slice/MultiModalKwargs object types this pipeline never
# actually sends across collective_rpc (every other RPC method here returns
# None or plain ints/lists, which msgpack already handles natively), a
# `pickle.loads` fallback that WOULD be a real remote-code-execution risk if
# the serialized data ever crossed a trust boundary. It does not here: this
# process spawns its own EngineCore subprocess locally (never over a
# network, never accepting input from another user or host), so there is no
# attacker position from which to inject a malicious payload into this
# specific IPC channel -- the same reasoning that makes `enforce_eager=True`
# and other same-host-only settings elsewhere in this pipeline uncontroversial.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


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

        **GPU placement.** `vllm.LLM()`/`EngineArgs` in this fork has no
        `device=` constructor kwarg at all -- confirmed the hard way on real
        hardware (`TypeError: EngineArgs.__init__() got an unexpected
        keyword argument 'device'`), not assumed; an earlier version of this
        method passed one, which was simply wrong. GPU placement in this
        fork is controlled entirely via `CUDA_VISIBLE_DEVICES`, applied per
        Worker process (confirmed against `vllm/platforms/cuda.py`'s own
        `device_control_env_var = "CUDA_VISIBLE_DEVICES"`) -- the same
        mechanism the single-turn pipeline's `predict_longbench_v2.py` uses
        for its OWN speculator placement, just via `torch.cuda.set_device(...)`
        beforehand there (valid for its `get_model()`-based standalone
        loading, which follows torch's *current* device with no engine/
        subprocess involved at all). That trick doesn't apply here: this
        `LLM(...)` call spawns its own out-of-process `EngineCore` (see
        `speculator_worker.py`'s module docstring), which only reads
        `CUDA_VISIBLE_DEVICES` from the environment it's spawned INTO, not
        from the parent driver process's `torch.cuda`-current-device state
        at any later point. So: temporarily override `CUDA_VISIBLE_DEVICES`
        in THIS process's environment for the duration of this one
        (synchronous, blocks until the engine + model are fully loaded)
        `LLM(...)` call, then restore it -- scoped narrowly so it doesn't
        affect the TARGET model's own, separately-constructed `LLM()` in the
        same driver process (which must see its own GPU correctly,
        regardless of construction order relative to this one).
        """
        import os

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
        )
        llm_kwargs.update(extra_llm_kwargs)

        device_index = device.index if device.type == "cuda" else None
        if device_index is None:
            self.llm = LLM(**llm_kwargs)
        else:
            prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
            try:
                self.llm = LLM(**llm_kwargs)
            finally:
                if prev_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd

        self.llm_engine = self.llm.llm_engine

    def run_turn(
        self,
        conversation_salt: str,
        turn_idx: int,
        full_sequence_token_ids: List[int],
        look_ahead_cnt: int,
        ignore_eos: bool = False,
    ) -> Tuple[List[torch.Tensor], int, int]:
        """Submit one turn's full (candidate_pool + query) sequence as a
        fresh request under `conversation_salt`'s stable cache lineage,
        drive it through its bootstrap prefill (a genuine prefix-cache-hit
        short-prefill step for KEEP mode's cache-hitting portion -- see this
        module's docstring on why DISCARD mode instead recomputes from
        scratch each turn, which is expected, not a bug) plus up to
        `look_ahead_cnt` lookahead decode steps, capturing queries only for
        the decode portion.

        `ignore_eos` is forwarded to vLLM's own `SamplingParams` (see
        `SpecConfig.ignore_eos` -- threaded through by `pruner.py`'s
        `compute_pruned_turn`, not read from spec_config here, to keep this
        class spec_config-agnostic). **This is the ONLY thing that controls
        early stopping** -- a previous version of this method accepted an
        `eos_token_id` parameter that looked like it did something similar
        but was never actually referenced in the body at all (confirmed on
        real hardware: a synthetic-token smoke test that expected exactly
        `look_ahead_cnt` steps got fewer, because the real model legitimately
        hit its own real EOS token partway through decoding from
        out-of-distribution input -- `ignore_eos` not `eos_token_id` was the
        missing piece; SamplingParams's own default already respects the
        model's real EOS regardless of any id this method might otherwise
        be told about).

        Returns:
            (query_buffer, actual_look_ahead_cnt, num_cached_tokens) --
            query_buffer is per-layer [1, actual_look_ahead_cnt, H*D]
            tensors on `self.device` (empty steps if actual_look_ahead_cnt
            is 0 -- caller must short-circuit, same rule as the single-turn
            pipeline: aggregating over zero lookahead steps produces silent
            NaN, not an error). actual_look_ahead_cnt may be less than
            look_ahead_cnt on real EOS (expected, not an error, unless
            ignore_eos=True was requested -- see above). `num_cached_tokens`
            is vLLM's own already-computed count of how many of this
            request's prompt tokens were served from the prefix cache --
            log this per turn (see `pruner.py`) as the real, measured "only
            prefill new tokens" signal EXPERIMENT_PLAN.md's KEEP-mode risk
            note calls for, rather than assuming it.
        """
        request_id, num_cached_tokens, t_start = self._submit_and_drive_turn(
            conversation_salt, turn_idx, full_sequence_token_ids, look_ahead_cnt, ignore_eos
        )

        # collective_rpc returns one result per worker (TP ranks) -- this
        # pipeline is TP=1 only (see module docstring), so unwrap index 0.
        # SpeculatorWorker.end_capture now returns raw torch.Tensor objects
        # directly -- this module's own VLLM_ALLOW_INSECURE_SERIALIZATION=1
        # setting (see top of file) makes collective_rpc's UtilityResult
        # wrapper carry proper type info for nested Tensors, so msgspec's
        # own dedicated zero-copy tensor codec (`_encode_tensor`/
        # `_decode_tensor` in vllm/v1/serial_utils.py) reconstructs them
        # correctly -- no more `tensor_to_wire`/`tensor_from_wire` Python-
        # list round trip needed (that workaround predates this fix; see
        # kv_cache_utils.py for why it's kept, not deleted, as a documented
        # fallback). Still need `.to(self.device)`: the decoded tensor
        # lands on CPU (an IPC buffer reconstruction, not a CUDA-IPC share),
        # same as the old path.
        t_before_end_capture = time.time()
        query_buffer_cpu = self.llm_engine.collective_rpc("end_capture", args=(request_id,))[0]
        query_buffer = [q.to(self.device) for q in query_buffer_cpu]
        actual_look_ahead_cnt = query_buffer[0].shape[1] if query_buffer else 0
        print(
            f"[proposer.run_turn] {request_id!r}: query capture retrieved "
            f"in {time.time() - t_before_end_capture:.2f}s "
            f"(actual_look_ahead_cnt={actual_look_ahead_cnt}, total turn "
            f"{time.time() - t_start:.2f}s)"
        )
        return query_buffer, actual_look_ahead_cnt, num_cached_tokens

    def run_turn_and_score(
        self,
        conversation_salt: str,
        turn_idx: int,
        full_sequence_token_ids: List[int],
        look_ahead_cnt: int,
        pool_kernel_size: Optional[int],
        keep_kwargs: dict,
        ignore_eos: bool = False,
        score_aggregation: str = "max",
        score_layers: Optional[str] = None,
    ) -> Tuple[Optional[List[int]], int, int]:
        """Same driving/submission as `run_turn` (via the shared
        `_submit_and_drive_turn` helper), but retrieves K and runs the
        scoring/selection pipeline IN-PROCESS inside the speculator's own
        worker (`speculator_worker.py`'s `end_capture_and_score`), instead
        of shipping Q back here and separately asking for K -- see that
        method's docstring for the real, measured cost this avoids (a
        single `retrieve_keys` call moving 720M K elements took 89s on a
        real SCBench turn). This is what `pruner.py::compute_pruned_turn`
        calls; `run_turn` itself is unchanged and still used directly by
        `validate_proposer.py`, which wants the raw Q buffer to validate
        the capture mechanism in isolation from scoring.

        `score_aggregation`/`score_layers` select the scoring variant (see
        `SpecConfig`'s fields and ACCURACY_IMPROVEMENTS.md §1); they default
        to the reference behavior, so a caller that doesn't pass them scores
        exactly as every already-published row did. They have to travel as
        arguments rather than being read from a config on the far side: the
        speculator's worker runs in its own process and never sees the
        driver's `SpecConfig`, the same reason `keep_kwargs`/
        `pool_kernel_size` are passed.

        Returns `(kept_local_indices, actual_look_ahead_cnt,
        num_cached_tokens)` -- `kept_local_indices` is `None` if
        `actual_look_ahead_cnt == 0` (see `speculator_worker.py`'s
        docstring for when/why), otherwise a `List[int]` of LOCAL indices
        into the full submitted sequence (candidate history + this turn's
        query) -- `pruner.py` still owns filtering that down to just the
        candidate-history portion and converting to absolute positions.
        """
        request_id, num_cached_tokens, t_start = self._submit_and_drive_turn(
            conversation_salt, turn_idx, full_sequence_token_ids, look_ahead_cnt, ignore_eos
        )

        t_before_score = time.time()
        kept_local_indices, actual_look_ahead_cnt = self.llm_engine.collective_rpc(
            "end_capture_and_score",
            args=(
                request_id,
                conversation_salt,
                len(full_sequence_token_ids),
                pool_kernel_size,
                keep_kwargs,
                score_aggregation,
                score_layers,
            ),
        )[0]
        num_kept = len(kept_local_indices) if kept_local_indices is not None else None
        print(
            f"[proposer.run_turn_and_score] {request_id!r}: in-process K "
            f"retrieval + scoring done in {time.time() - t_before_score:.2f}s "
            f"(actual_look_ahead_cnt={actual_look_ahead_cnt}, kept={num_kept}, "
            f"total turn {time.time() - t_start:.2f}s)"
        )
        return kept_local_indices, actual_look_ahead_cnt, num_cached_tokens

    def _submit_and_drive_turn(
        self,
        conversation_salt: str,
        turn_idx: int,
        full_sequence_token_ids: List[int],
        look_ahead_cnt: int,
        ignore_eos: bool,
    ) -> Tuple[str, int, float]:
        """Shared submission/driving core for `run_turn` and `run_turn_and_
        score` -- begin_capture, add_request, drive to completion, and the
        bootstrap-prefill/lookahead-speculation timing logs and DIAGNOSTIC
        check, all identical regardless of how the caller retrieves the
        result afterward. Returns (request_id, num_cached_tokens, t_start)
        -- `t_start` handed back so callers' own post-driving timing logs
        report a "total turn" figure that includes submission/driving, not
        just their own retrieval step."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        request_id = f"{conversation_salt}::turn{turn_idx}"
        t_start = time.time()

        # Register capture BEFORE add_request -- NOT reactively, in response
        # to observed output progress, as an earlier version of this method
        # did. Confirmed on real hardware: EngineCore runs its own
        # autonomous stepping loop, decoupled from the driver's own step()
        # calls, so a reactive begin_capture can miss steps that already ran
        # in the background before the RPC round-trip lands (a second real
        # occurrence of the exact race pruner.py's "Correction #2" already
        # documents and fixes the same way -- see
        # speculator_worker.py::begin_capture's docstring for the full
        # history and speculator_worker.py::end_capture's docstring for how
        # the bootstrap prefill's own (now-always-captured) entry gets
        # excluded on the way out instead of by capture timing).
        self.llm_engine.collective_rpc("begin_capture", args=(request_id,))

        prompt = TokensPrompt(
            prompt_token_ids=full_sequence_token_ids, cache_salt=conversation_salt
        )
        sampling_params = SamplingParams(
            max_tokens=1 + look_ahead_cnt, temperature=0.0, ignore_eos=ignore_eos
        )
        real_request_id = self.llm_engine.add_request(request_id, prompt, sampling_params)
        assert real_request_id == request_id, (
            f"expected add_request() to use request_id={request_id!r} verbatim "
            f"(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 is set at this module's "
            f"import time) but got {real_request_id!r} back instead."
        )

        num_cached_tokens = 0
        final_output = None
        t_prefill_done = None
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
                if final_output is None:
                    # The FIRST output seen for a fresh (non-resumable,
                    # non-streamed) request is exactly the point the
                    # bootstrap prefill finished and the first token was
                    # sampled -- there is no earlier signal available from
                    # outside the engine.
                    t_prefill_done = time.time()
                    print(
                        f"[proposer.run_turn] {request_id!r}: bootstrap "
                        f"prefill done in {t_prefill_done - t_start:.2f}s "
                        f"({len(full_sequence_token_ids)} tokens, "
                        f"num_cached_tokens={output.num_cached_tokens})"
                    )
                final_output = output
                if output.num_cached_tokens:
                    num_cached_tokens = output.num_cached_tokens

        t_decode_done = time.time()
        if t_prefill_done is not None and final_output is not None:
            print(
                f"[proposer.run_turn] {request_id!r}: lookahead speculation "
                f"done in {t_decode_done - t_prefill_done:.2f}s "
                f"({len(final_output.outputs[0].token_ids) - 1} lookahead "
                f"steps, total turn so far {t_decode_done - t_start:.2f}s)"
            )

        # Diagnostic, not a hard assertion -- still worth surfacing if
        # generation itself stopped short of the requested length (distinct
        # from end_capture's own decode-step count, which is now expected to
        # match generation length exactly since capture timing is no longer
        # the limiting factor -- see speculator_worker.py::end_capture).
        if final_output is not None:
            total_generated = len(final_output.outputs[0].token_ids)
            if total_generated != 1 + look_ahead_cnt:
                print(
                    f"[proposer.run_turn] DIAGNOSTIC: request {request_id!r} "
                    f"generated {total_generated} tokens total (expected "
                    f"{1 + look_ahead_cnt} = 1 bootstrap + {look_ahead_cnt} "
                    f"lookahead), finish_reason="
                    f"{final_output.outputs[0].finish_reason!r}, "
                    f"ignore_eos={ignore_eos} -- if finish_reason is 'stop' "
                    f"despite ignore_eos=True, ignore_eos isn't reaching "
                    f"SamplingParams as intended; if 'length', something is "
                    f"capping max_tokens/max_model_len below what was "
                    f"requested."
                )

        return request_id, num_cached_tokens, t_start

    def retrieve_keys(
        self, conversation_salt: str, local_positions: List[int]
    ) -> List[torch.Tensor]:
        """Per-layer [len(local_positions), num_kv_heads, head_size] K
        tensors on `self.device`, for LOCAL positions within
        `conversation_salt`'s own speculator prompt (see module docstring's
        "Local positions vs. absolute conversation positions").

        Timed and logged on completion -- for a large KEEP-mode candidate
        pool (`local_positions` can be tens of thousands of entries for a
        big SCBench context) this RPC moves a lot of data every turn, so
        it's worth watching even after the fix below.

        **Real measurement, not a guess**: a first real run showed this one
        call taking 89s for 87,972 positions x 16 layers (720M elements) --
        traced to the OLD `tensor_to_wire` implementation's `.tolist()`
        (converts the whole tensor to a Python list, one boxed float at a
        time), which then had to cross msgpack serialization AGAIN as a
        plain list, then get rebuilt via `torch.tensor(list)` on this end --
        three full non-vectorized passes over ~720M scalars. Fixed by this
        module's `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (see top of file):
        `SpeculatorWorker.retrieve_keys` now returns raw `torch.Tensor`s,
        and msgspec's own dedicated tensor codec moves the underlying
        buffer directly (no Python-level element-by-element conversion at
        all) -- expected to cut this dramatically, not yet confirmed on
        real hardware past this change (re-run `validate_proposer.py`,
        which already exercises this exact method in its Step C)."""
        t_start = time.time()
        keys_cpu = self.llm_engine.collective_rpc(
            "retrieve_keys", args=(conversation_salt, local_positions)
        )[0]
        keys = [k.to(self.device) for k in keys_cpu]
        total_elements = sum(k.numel() for k in keys)
        print(
            f"[proposer.retrieve_keys] {conversation_salt!r}: retrieved K "
            f"for {len(local_positions)} positions across {len(keys)} "
            f"layers ({total_elements} total elements) in "
            f"{time.time() - t_start:.2f}s"
        )
        return keys

    def discard_conversation(self, conversation_salt: str) -> None:
        """Call once a whole conversation (not just one turn) is finished,
        to bound `speculator_worker.py`'s per-conversation slot-history
        accumulation across a long benchmark run."""
        self.llm_engine.collective_rpc("discard_conversation", args=(conversation_salt,))
