"""SpeculatorWorker / SpeculatorGPUModelRunner -- runs the speculator
(Llama-3.2-1B) as a genuine persistent `vllm.LLM()` engine, not the
single-turn pipeline's standalone `get_model()`-loaded, manually-driven
model. This is the piece of this port with the least real-hardware
confidence -- see "Known risk areas" at the end of this docstring before
relying on it; `validate_proposer.py`'s Step C is where these should be
checked first.

## Why a real engine at all (recap of the approved plan's reasoning)

The single-turn pipeline's `proposer.py` bypassed `LLM()`/`Worker` entirely
and built a throwaway per-call dummy KV cache (`build_lookahead_metadata`).
Making that persist across turns would mean reimplementing a miniature
paged-KV allocator (growth, eviction) by hand. Using a real, persistent
`vllm.LLM(enable_prefix_caching=True)` instead gets "only prefill the new
tokens each turn" for free from vLLM's own prefix-cache matching -- exactly
the protocol's "each turn only prefills the new query tokens against its own
cache" requirement -- at the cost of the speculator's model/attention layers
now living in a genuinely separate OS process from the driver (`EngineCore`
always runs out-of-process, confirmed on real hardware for the *target*
model's own `SpecPrefillWorker`, see that module's docstring; the same fact
applies here). Everything below exists to cross that process boundary via
the SAME `worker_cls` + `collective_rpc` extension points `worker.py`/
`pruner.py` already use for the target model -- not a new integration
pattern, just the second application of an already-proven one.

## What's structurally different from the target's SpecPrefillWorker

The target's `SpecPrefillWorker`/`SpecPrefillGPUModelRunner` exist to
override RoPE *positions* for a request whose PROMPT has already been
pruned (shrunk) before it ever reached the engine -- the runner never needs
to know anything about *other* requests' data.

The speculator never prunes its own prompt -- every request it's given is a
real, contiguous, ever-growing prefix (see `proposer.py`'s docstring for why
this is true under both KEEP and DISCARD history modes), so it needs NO
RoPE-position override at all. What it needs instead is two things the
target never does:

1. **Capture post-RoPE query vectors** during a request's forward passes
   (same idea as the single-turn pipeline's hook, but now must run inside
   a real scheduler step that may BATCH multiple requests together --
   e.g. this conversation's own ephemeral lookahead branch running
   alongside its persistent extension request, and/or another conversation
   entirely, submitted concurrently for throughput. The single-turn
   pipeline never had to disambiguate this: its forward calls were always
   manually issued one-request-at-a-time. Here, capture must be sliced by
   request via `query_start_loc`, mirroring exactly how `model_runner.py`'s
   `_apply_spec_prefill_position_overrides` already slices `self.positions`
   per request from a batched step -- same technique, applied to `q`
   instead of `positions`.
2. **Recover K vectors for tokens computed in a PAST turn's forward call**,
   not just the current step's. A prefix-cache hit means a token that's
   already cached is NEVER rescheduled/recomputed for a later turn's
   request -- so a later turn's own `_prepare_inputs` call never mentions
   it, and there is nothing "live" to read a fresh slot_mapping from for
   it. This class instead ACCUMULATES a per-conversation `{local_position:
   physical_slot}` map incrementally, every time `_prepare_inputs` runs for
   ANY request under that conversation's salt -- a token's slot, once
   observed, need never be observed again, and stays valid for as long as
   its physical block isn't evicted+reallocated (see risk #2 below).
   `kv_cache_utils.py`'s `read_layer_keys`/`gather_keys_for_slots` are
   reused completely unchanged for the actual cache-tensor read once a
   slot_mapping is in hand -- this class's job is only to RECOVER that
   mapping, which the single-turn pipeline never needed to do because it
   built (and therefore already knew) its own dummy cache's slot mapping
   directly.

## Conversation-salt keying, not request-id keying

Both the query buffer and the slot-history map are keyed by
`conversation_salt` (stable across a whole conversation's turns), not
`request_id` (a fresh string every turn -- see `proposer.py`). `request_id`
is expected to encode its `conversation_salt` as a `"{salt}::turn{n}"`
prefix (`_conversation_salt_from_request_id` below) so this runner can
recover it without the driver having to separately RPC "here is the salt
for this request_id" ahead of every single forward step.

## Known risk areas (checked against this fork's real source where noted,
but NOT executed -- see `validate_proposer.py`'s Step C)

1. **Single KV-cache-group assumption.** `self.input_batch.block_table` is
   indexed by KV-cache group (plural, for hybrid-attention models with
   multiple cache layouts in one model); this file always reads group 0.
   Confirmed correct in spirit for Llama-3.2-1B (dense, uniform attention,
   one cache layout for every layer, same reasoning `kv_cache_utils.py`'s
   own docstring gives for why it degenerates to a no-op simplification
   there) -- not independently re-confirmed against this fork's actual
   `input_batch.block_table` construction for a real load.
2. **Eviction invalidates the accumulated slot map silently.** If a
   conversation's early blocks are evicted (LRU, `ref_cnt==0`) under memory
   pressure and later re-requested, vLLM recomputes them into DIFFERENT
   physical blocks -- this class's stale `{position: old_slot}` entry would
   then point at the wrong (possibly since-reallocated-to-another-
   conversation) physical cache contents, with no error raised. This is the
   same risk the approved plan flags for `proposer.py`'s
   `RequestOutput.num_cached_tokens` check -- mitigated the same way
   (generous `gpu_memory_utilization`, monitor `num_cached_tokens` per
   turn), not independently solved here.
3. **`_prepare_inputs`'s exact signature/timing for slot_mapping
   availability.** Confirmed against this fork's real
   `gpu_model_runner.py` source (not assumed): `self.input_batch.
   block_table.compute_slot_mapping(...)` is called INSIDE the stock
   `_prepare_inputs` (before it returns), and the computed result is
   readable afterward via `self.input_batch.block_table[0].slot_mapping.gpu`
   -- the same "call `super()._prepare_inputs()`, then read the still-valid
   persistent instance buffers it populated" pattern `model_runner.py`
   already relies on and documents as real-hardware-confirmed for the
   *positions* buffer specifically. The *slot_mapping* buffer's read-after-
   return safety follows the same reasoning but has not itself been
   independently confirmed on real hardware.
"""

import types
from functools import partial
from typing import Dict, List, Optional

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from .kv_cache_utils import (
    gather_keys_for_slots,
    is_decode_query_slice,
    read_layer_keys,
    stack_decode_only_steps,
)


def _conversation_salt_from_request_id(request_id: str) -> Optional[str]:
    """request_id convention: "{conversation_salt}::turn{n}" or
    "{conversation_salt}::lookahead-turn{n}" -- see `proposer.py`. Returns
    None for any request_id not following this convention (shouldn't occur
    in practice, since this engine only ever serves speculator requests
    submitted by `proposer.py`, but fails loudly rather than mis-keying
    silently if it ever does)."""
    if "::" not in request_id:
        return None
    return request_id.split("::", 1)[0]


def _find_llama_attention_layers(model) -> List:
    """Locate the speculator's LlamaAttention layers -- identical logic to
    the single-turn pipeline's `proposer.py::_find_llama_attention_layers`,
    duplicated here (not imported) because this must run inside the Worker
    process, importing only from this module and `kv_cache_utils.py`
    (already vLLM-runtime-free) rather than pulling in `proposer.py`'s own
    driver-side imports (`vllm.model_executor.model_loader.get_model`, etc.)
    which are meaningless/unused inside the Worker process."""
    model_name = type(model).__name__
    if "Llama" not in model_name:
        raise NotImplementedError(
            f"SpeculatorWorker currently only supports Llama models as the "
            f"speculator (got {model_name}). The query-capture hook is a "
            f"faithful copy of LlamaAttention.forward's body and is not "
            f"architecture-generic."
        )
    inner = model.model if hasattr(model, "model") else model
    if not hasattr(inner, "layers"):
        raise NotImplementedError(
            f"Could not locate decoder layers on {model_name} -- neither "
            f".model.layers nor .layers resolved."
        )
    return [layer.self_attn for layer in inner.layers]


def _query_capturing_llama_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    runner: "SpeculatorGPUModelRunner",
    layer_idx: int,
    **kwargs,
) -> torch.Tensor:
    """Instance-level replacement for `LlamaAttention.forward`, identical
    body to the single-turn pipeline's own version (see that module for the
    "why a faithful copy, not a generic wrapper" reasoning) with one
    difference: capture is delegated to `runner._capture_queries_by_request`
    instead of appending directly to a flat list, because a real engine step
    may batch multiple requests' tokens into one `hidden_states` tensor --
    see module docstring's "What's structurally different" section.
    """
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)

    runner._capture_queries_by_request(layer_idx, q.detach())

    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


class SpeculatorGPUModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hooked_layers: List = []
        # request_id -> per-layer list of per-forward-call query slices,
        # only populated for request_ids in _capturing_request_ids (toggled
        # by begin_capture/end_capture -- see those methods) so an
        # in-flight persistent-extension request never gets scored, mirroring
        # the single-turn pipeline's "bootstrap prefill is not a scored
        # step" rule.
        self._query_buffer: Dict[str, List[List[torch.Tensor]]] = {}
        self._capturing_request_ids: set = set()
        # conversation_salt -> {local_position: physical_slot} -- see module
        # docstring's "Recover K vectors" section.
        self._slot_history: Dict[str, Dict[int, int]] = {}

    def install_query_capture_hooks(self) -> None:
        """Called once, from `SpeculatorWorker.load_model()` (after weights
        are loaded, unlike the target's hooks which don't exist at all --
        this timing mirrors the single-turn pipeline's own
        `install_query_capture_hooks`, just relocated to run inside the
        Worker process instead of the driver process)."""
        self._hooked_layers = _find_llama_attention_layers(self.model)
        for layer_idx, self_attn in enumerate(self._hooked_layers):
            self_attn.forward = types.MethodType(
                partial(
                    _query_capturing_llama_attention_forward,
                    runner=self,
                    layer_idx=layer_idx,
                ),
                self_attn,
            )

    def begin_capture(self, request_id: str) -> None:
        """Must be called BEFORE `add_request` for this `request_id`, not
        reactively in response to observing the request's own progress.

        **Confirmed on real hardware, a second real occurrence of a race
        already documented once in this codebase (`pruner.py`'s "Correction
        #2"): `EngineCore` runs its own autonomous stepping loop, decoupled
        from the driver's own `step()` calls.** An earlier version of this
        method's caller (`proposer.py::run_turn`) called `begin_capture`
        REACTIVELY -- inside the driver's stepping loop, in response to
        first observing the bootstrap prefill's own output token -- on the
        theory that this correctly excludes the bootstrap from capture (the
        same goal the single-turn pipeline achieves by discarding a buffer
        instead). That reasoning silently assumed EngineCore only advances
        when the driver calls `step()`. It doesn't: by the time the
        driver's blocking `collective_rpc("begin_capture", ...)` round-trip
        actually lands and takes effect inside THIS process, EngineCore may
        have already run several more decode steps in the background,
        which were never captured -- confirmed exactly this shape of bug on
        real hardware (`look_ahead_cnt=8` requested, generation genuinely
        ran the full length, only 6 steps captured -- 2 decode steps
        silently missed between "driver observed token 1" and "begin_capture
        actually took effect").

        Fixed the same way `pruner.py`'s analogous race was fixed: register
        BEFORE the request can be scheduled at all (before `add_request`),
        so there is no window for EngineCore to run ahead uncaptured.
        Capturing now starts from the request's very first forward call
        (the bootstrap prefill itself) -- `end_capture` is responsible for
        excluding that entry from the returned decode-only stack (see its
        own docstring), not this method.
        """
        self._capturing_request_ids.add(request_id)
        self._query_buffer[request_id] = [[] for _ in self._hooked_layers]

    def end_capture(self, request_id: str) -> List[torch.Tensor]:
        """Stops capturing for `request_id` and returns its per-layer
        captured DECODE-ONLY queries, each stacked to [1,
        num_captured_decode_steps, H*D] -- num_captured_decode_steps may be
        less than look_ahead_cnt if the request ended early (EOS or
        max_tokens).

        Since `begin_capture` now runs before the request even exists (see
        that method's docstring), every one of this request's forward calls
        REACHES the capture hook, INCLUDING the bootstrap prefill's own --
        which must be excluded by shape, not by timing. As of the
        capture-time skip in `_capture_queries_by_request` (added to stop
        prefill queries pinning tens of GiB -- see that method's docstring)
        prefill entries no longer arrive here at all; the filter below is
        kept as the second line of defense, and as the statement of the
        rule both places implement. The bootstrap's own capture
        is NOT reliably "just the first entry": if the new-suffix being
        prefilled this turn is long enough to need more than one scheduler
        step (chunked prefill), there could be several leading multi-token
        entries before real single-token decode begins. Filtering by SHAPE
        (keep only entries whose captured-token count is exactly 1) is
        robust to this regardless of how many prefill chunks preceded
        decode -- every genuine decode step captures exactly 1 new token's
        query; every prefill/prefill-continuation step captures more than 1
        (or, in the degenerate case where the entire prompt was already
        prefix-cache-hit and the very first forward call IS a 1-token
        decode step, there is no multi-token entry to filter out at all,
        which this same rule handles correctly with no special-casing).

        **Stays on-device here** (a change from this method's original
        design, which moved to CPU itself) -- `end_capture` is called from
        TWO contexts now: `SpeculatorWorker.end_capture` below (an RPC-
        exposed method, whose result DOES cross `collective_rpc`'s process
        boundary and so DOES need a CPU tensor -- that method does the
        `.to("cpu")` itself, right at the boundary) and
        `end_capture_and_score` below (an IN-PROCESS caller that
        immediately feeds this straight into GPU-accelerated scoring math
        -- forcing CPU here would mean an unnecessary GPU->CPU copy AND run
        `compute_attention_score`'s matmuls on CPU instead of GPU, a real,
        measured cost: a first real run of `end_capture_and_score` took 34s
        for one turn with Q/K both on CPU, most of that plausibly this
        exact inefficiency). Keeping this method itself device-agnostic and
        pushing the CPU decision to whichever caller actually needs it is
        the correct place for that decision to live."""
        self._capturing_request_ids.discard(request_id)
        per_layer_steps = self._query_buffer.pop(request_id, None)
        if per_layer_steps is None:
            return [torch.empty(1, 0, 0, device=self.device) for _ in self._hooked_layers]

        result = []
        for layer_idx, steps in enumerate(per_layer_steps):
            hidden_dim_fallback = (
                self._hooked_layers[layer_idx].num_heads
                * self._hooked_layers[layer_idx].head_dim
            )
            result.append(stack_decode_only_steps(steps, hidden_dim_fallback))
        return result

    def _capture_queries_by_request(self, layer_idx: int, q: torch.Tensor) -> None:
        """Appends this forward call's per-request query slice to the
        capture buffer -- DECODE steps only.

        ## Why the `end - start != 1` skip is load-bearing, in GiB

        `end_capture` has always thrown prefill entries away by shape (keep
        only entries covering exactly 1 token -- see its docstring for why
        that rule is the robust one). Doing it only THERE meant every
        prefill token's query stayed resident until the turn's scoring
        pass: `q[start:end]` is a VIEW, so retaining it pins the whole
        forward's query tensor, per layer, for every prefill (or prefill-
        chunk) call.

        That is `num_prefill_tokens x hidden_size x 2 bytes x num_layers`
        of memory doing nothing but waiting to be discarded. A real OOM
        this fixes, not a theoretical one -- on `scbench_kv-0`'s 124,009-
        token turn 0:

          Llama-3.2-1B speculator: 16 x 0.47GiB =  7.6GiB retained
          Llama-3.1-8B scorer:     32 x 0.95GiB = 30.3GiB retained

        The 1B's 7.6GiB was invisible for the whole SPARSE sweep, absorbed
        by the ~63GiB the card had free at `--speculator-gpu-memory-
        utilization 0.2`. The 8B ORACLE scorer's 30.3GiB was not: 15GiB
        weights + ~30GiB KV pool + 30.3GiB of prefill queries = the 78.00
        GiB `torch.cuda.memory_allocated()` reported at the OOM, on a
        79.25GiB card. Note this is INVARIANT to prefill chunk size --
        chunking the prefill splits the same total across more, smaller
        views -- which is why capping `max_num_batched_tokens` alone did
        not fix it (it did shrink the per-step activation transient, from
        a 3.31GiB failed allocation to 896MiB, just not this).

        Skipping at capture time is exactly equivalent to `end_capture`'s
        existing shape filter -- same rule, same predicate, applied before
        the memory is pinned rather than after. That filter stays where it
        is as defense in depth; this one is what makes it free.

        The decode slice is `.clone()`d rather than kept as a view, so what
        stays resident is bounded by the captured token itself (1 x
        hidden_size, a few KiB) and not by whatever else shared that
        step's batch -- cheap insurance against a future scheduler that
        batches this request's decode alongside another request's prefill
        chunk, which would otherwise pin that chunk's whole tensor through
        a 1-token view."""
        if not self._capturing_request_ids:
            return
        num_reqs = self.input_batch.num_reqs
        query_start_loc_np = self.query_start_loc.np[: num_reqs + 1]
        for req_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[req_idx]
            if req_id not in self._capturing_request_ids:
                continue
            start = int(query_start_loc_np[req_idx])
            end = int(query_start_loc_np[req_idx + 1])
            if not is_decode_query_slice(start, end):
                # Prefill or prefill-continuation chunk (or an empty slice):
                # never scored, so never retained. See this method's
                # docstring for the memory this saves, and
                # `kv_cache_utils.is_decode_query_slice` for why the
                # predicate lives over there.
                continue
            self._query_buffer[req_id][layer_idx].append(q[start:end].clone())

    def _prepare_inputs(self, scheduler_output, *args, **kwargs):
        # *args/**kwargs passthrough -- see model_runner.py's identical
        # reasoning (base class's real signature takes an extra
        # num_scheduled_tokens positional argument this shouldn't hardcode).
        result = super()._prepare_inputs(scheduler_output, *args, **kwargs)
        self._record_slot_mapping()
        return result

    def _record_slot_mapping(self) -> None:
        """Accumulate this step's newly-computed slot_mapping into the
        per-conversation running history -- see module docstring's "Recover
        K vectors" and risk #3."""
        num_reqs = self.input_batch.num_reqs
        query_start_loc_np = self.query_start_loc.np[: num_reqs + 1]
        # Single KV-cache-group assumption -- see module docstring's risk #1.
        slot_mapping_gpu = self.input_batch.block_table[0].slot_mapping.gpu

        for req_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[req_idx]
            conversation_salt = _conversation_salt_from_request_id(req_id)
            if conversation_salt is None:
                continue

            start = int(query_start_loc_np[req_idx])
            end = int(query_start_loc_np[req_idx + 1])
            if end <= start:
                continue
            num_computed_before = int(self.input_batch.num_computed_tokens_cpu[req_idx])

            slots_this_step = slot_mapping_gpu[start:end].tolist()
            history = self._slot_history.setdefault(conversation_salt, {})
            for i, slot in enumerate(slots_this_step):
                history[num_computed_before + i] = slot

    def get_slots_for_positions(
        self, conversation_salt: str, positions: List[int]
    ) -> List[int]:
        history = self._slot_history.get(conversation_salt, {})
        missing = [p for p in positions if p not in history]
        if missing:
            preview = missing[:5]
            raise KeyError(
                f"conversation {conversation_salt!r}: no recorded physical "
                f"slot for local position(s) {preview}"
                f"{'...' if len(missing) > 5 else ''} -- either these "
                f"positions were never actually prefilled under this salt, "
                f"or their blocks were evicted and silently recomputed "
                f"under a new slot this accumulator never observed (see "
                f"module docstring's risk #2)."
            )
        return [history[p] for p in positions]

    def retrieve_keys(
        self, conversation_salt: str, positions: List[int]
    ) -> List[torch.Tensor]:
        """Per-layer [len(positions), num_kv_heads, head_size] K tensors,
        ON-DEVICE (see `end_capture`'s docstring for why this method no
        longer moves to CPU itself -- same reasoning applies here: this
        method is called from both `SpeculatorWorker.retrieve_keys` (RPC-
        exposed, needs CPU, does the `.to("cpu")` itself at that boundary)
        and `end_capture_and_score` (in-process, wants to keep K on-device
        for GPU-accelerated scoring -- moving ~720M elements to CPU only to
        immediately run CPU-bound matmuls over them was the single largest
        remaining cost after the collective_rpc transfer itself was fixed)
        -- for the given absolute LOCAL positions within this
        conversation's own speculator prompt (i.e. indices into what
        `proposer.py` submitted as this conversation's growing prompt --
        NOT the multi-turn ledger's absolute positions used for the
        TARGET's RoPE restoration, a separate numbering -- see
        `proposer.py`'s docstring for how the two relate)."""
        slots = self.get_slots_for_positions(conversation_salt, positions)
        slot_tensor = torch.tensor(slots, dtype=torch.int64, device=self.device)
        block_size = self.vllm_config.cache_config.block_size
        keys = []
        for self_attn in self._hooked_layers:
            num_kv_heads = self_attn.num_kv_heads
            head_size = self_attn.head_dim
            flat_keys = read_layer_keys(self_attn.attn, block_size, num_kv_heads, head_size)
            keys.append(gather_keys_for_slots(flat_keys, slot_tensor))
        return keys

    def end_capture_and_score(
        self,
        request_id: str,
        conversation_salt: str,
        full_sequence_len: int,
        pool_kernel_size: Optional[int],
        keep_kwargs: dict,
        score_aggregation: str = "max",
        score_layers: Optional[str] = None,
    ):
        """Combines `end_capture` + `retrieve_keys` + the scoring/selection
        pipeline (`scoring.score_and_select_indices`) into ONE in-process
        call, so only the tiny resulting index list -- not Q or K -- ever
        needs to cross `collective_rpc`'s process boundary.

        **Why this exists, concretely measured, not theoretical**: Q and K
        both already live in THIS process (Q captured during this turn's
        own forward passes, K resident in this engine's own KV cache) --
        there was never an architectural need to ship either across the
        process boundary, only the FINAL SELECTION DECISION does. The OLD
        design (ship Q back via `end_capture`, separately ask for K via
        `retrieve_keys`, score driver-side) measured a single `retrieve_
        keys` call taking 89s for 87,972 positions x 16 layers (720M
        elements) on a real SCBench turn -- this method exists specifically
        to make that number irrelevant by never moving that data at all.

        Returns `(kept_local_indices, actual_look_ahead_cnt)` --
        `kept_local_indices` is `None` if `actual_look_ahead_cnt == 0` (no
        lookahead steps completed: real EOS on the very first candidate
        token, or `look_ahead_cnt=0`), meaning the caller should keep
        everything rather than score with no signal (same fallback
        `pruner.py`'s `_positions_from_kept_indices` already implements for
        the oracle path -- this method's caller, `proposer.py::run_turn`,
        passes this straight through rather than duplicating that logic).
        Otherwise a plain `List[int]` of LOCAL indices into the full
        `full_sequence_len`-token sequence (candidate history + this
        turn's own query, in that order -- see `proposer.py`'s module
        docstring on local-position numbering), covering the WHOLE
        sequence, not pre-filtered to just the candidate-history portion
        -- `pruner.py` still owns that filter, unchanged, since it's cheap
        once the result is this small.

        **On-device, not CPU**: `self.end_capture`/`self.retrieve_keys`
        called here are the `SpeculatorGPUModelRunner`-level methods (not
        `SpeculatorWorker`'s RPC-exposed wrappers of the same name, which
        DO move to CPU for the collective_rpc boundary those cross) -- see
        both methods' own docstrings. Q and K therefore stay on `self.
        device` all the way through `score_and_select_indices`'s matmul/
        softmax/topk pipeline, letting it run GPU-accelerated instead of
        CPU-bound -- a first real run before this distinction existed took
        34s total with everything forced onto CPU, most of that plausibly
        this exact inefficiency (a large matmul + softmax over an
        88k-token context on CPU instead of GPU)."""
        from .config import SpecConfig
        from .scoring import score_and_select_indices

        query_buffer = self.end_capture(request_id)
        actual_look_ahead_cnt = query_buffer[0].shape[1] if query_buffer else 0
        if actual_look_ahead_cnt == 0:
            return None, 0

        key_buffer_per_layer = self.retrieve_keys(
            conversation_salt, list(range(full_sequence_len))
        )
        # `score_aggregation`/`score_layers` have to be passed in rather than
        # read from a config here: scoring runs inside the speculator's OWN
        # worker process, which never sees the driver's `SpecConfig` (see
        # this class's module docstring on the process split). They are
        # defaulted to the reference behavior so an older caller that doesn't
        # pass them scores exactly as before.
        spec_config = SpecConfig(
            keep_strategy="percentage",
            keep_kwargs=keep_kwargs,
            pool_kernel_size=pool_kernel_size,
            score_aggregation=score_aggregation,
            score_layers=score_layers,
        )
        kept_local_indices = score_and_select_indices(
            query_buffer, key_buffer_per_layer, actual_look_ahead_cnt, spec_config
        )
        return kept_local_indices, actual_look_ahead_cnt

    def end_capture_and_head_diagnostics(
        self,
        request_id: str,
        conversation_salt: str,
        full_sequence_len: int,
        pool_kernel_size,
        keep_kwargs: dict,
        gold_positions: list,
        top_n_list: list,
        fixed_head_sets: list = None,
    ):
        """The §1.3 gate (ACCURACY_IMPROVEMENTS.md): how much would
        retrieval-head filtering buy, AT MOST?

        Selects heads using the gold answer's own position -- i.e. cheating --
        so whatever survival this reports is an UPPER BOUND on any honest
        head-identification method. The point is to kill or justify a
        multi-day build with a half-day measurement, the same way ORACLE-k20
        did for the scoring sweep.

        Runs entirely in this process for the same reason
        `end_capture_and_score` does, only more so: the per-head importance
        tensor it needs is `[num_layers*num_heads, look_ahead, context_len]`
        -- ~1GB for the 1B speculator at a 124k context -- and only a handful
        of scalars per turn ever need to come back.

        Returns a dict (or None if no lookahead step was captured):
          `gold_mass`        per-(layer, head) share of that head's own
                             attention mass landing on the gold span. The
                             raw material for "are the useful heads the SAME
                             heads across turns", which the driver answers;
                             a fixed head set cannot work if they are not.
          `baseline_survived` gold kept under the reference all-head `max`.
          `oracle_survived`  gold kept under `max` restricted to the top-N
                             heads BY GOLD MASS, per N in `top_n_list`.
          `fixed_survived`   gold kept under each `(label, head_indices)` in
                             `fixed_head_sets` -- a head list chosen WITHOUT
                             seeing this turn's gold, i.e. the honest §1.3
                             number rather than the ceiling. This is what
                             settles a split verdict where the ceiling is
                             high but the per-turn head sets look unstable:
                             low top-16 overlap is also what you would see
                             if a couple of heads are stable and the ranks
                             below them are noise, and only a fixed set's
                             actual survival can tell those apart.

        Survival is "every gold token position kept", matching
        `diagnose_gold_survival.py`'s text-containment check in strictness:
        a partially-kept answer is not retrievable either.
        """
        import torch

        from .config import SpecConfig
        from .scoring import (
            chunk_select_from_smoothed_attention,
            compute_attention_score,
        )

        query_buffer = self.end_capture(request_id)
        actual_look_ahead_cnt = query_buffer[0].shape[1] if query_buffer else 0
        if actual_look_ahead_cnt == 0 or not gold_positions:
            return None

        key_buffer_per_layer = self.retrieve_keys(
            conversation_salt, list(range(full_sequence_len))
        )
        attn = compute_attention_score(
            query_buffer, [[k] for k in key_buffer_per_layer], [actual_look_ahead_cnt]
        )[0]

        # Same pipeline as `aggregate_attention_score`, stopped one step
        # early: the (layer, head) axis is kept instead of collapsed, which
        # is the whole point.
        original_dtype = attn.dtype
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(
            original_dtype
        )
        attn = attn.flatten(0, 1)
        if pool_kernel_size:
            attn = torch.nn.functional.avg_pool1d(
                attn, kernel_size=pool_kernel_size,
                padding=pool_kernel_size // 2, stride=1,
            )

        gold_idx = torch.tensor(
            sorted(set(gold_positions)), device=attn.device, dtype=torch.long
        )
        per_head = attn.mean(1)  # [LH, ctx] -- mean over lookahead, for ranking
        gold_mass = (
            per_head.index_select(-1, gold_idx).sum(-1)
            / per_head.sum(-1).clamp_min(1e-9)
        )

        spec_config = SpecConfig(
            keep_strategy="percentage",
            keep_kwargs=keep_kwargs,
            pool_kernel_size=pool_kernel_size,
        )
        gold_set = set(int(p) for p in gold_idx.tolist())

        def _survived(head_rows) -> bool:
            # `head_rows is None` == every head, and is NOT expressed as an
            # arange index: indexing would copy the whole ~1GB tensor to
            # reproduce it.
            collapsed = (attn if head_rows is None else attn[head_rows]).max(0)[0].mean(0)
            kept = chunk_select_from_smoothed_attention([collapsed], spec_config)[0]
            return gold_set.issubset(set(kept.tolist()))

        order = torch.argsort(gold_mass, descending=True)
        oracle_survived = {}
        for n in top_n_list:
            n = int(n)
            if n <= 0 or n > attn.shape[0]:
                continue
            oracle_survived[n] = bool(_survived(order[:n]))

        fixed_survived = {}
        for label, head_indices in (fixed_head_sets or []):
            rows = torch.tensor(
                [int(h) for h in head_indices if 0 <= int(h) < attn.shape[0]],
                device=attn.device, dtype=torch.long,
            )
            if rows.numel():
                fixed_survived[str(label)] = bool(_survived(rows))

        return {
            "gold_mass": gold_mass.float().cpu().tolist(),
            "baseline_survived": bool(_survived(None)),
            "oracle_survived": oracle_survived,
            "fixed_survived": fixed_survived,
            "num_heads": int(attn.shape[0]),
        }

    def discard_conversation(self, conversation_salt: str) -> None:
        """Drop this conversation's accumulated slot history -- call once
        the whole conversation (not just one turn) is done, to avoid
        unbounded growth across a long benchmark run."""
        self._slot_history.pop(conversation_salt, None)


class SpeculatorWorker(Worker):
    """Wires `SpeculatorGPUModelRunner` in via the same `parallel_config.
    worker_cls` extension point the target's `SpecPrefillWorker` uses -- see
    that module's docstring for the underlying mechanism (`resolve_obj_by_
    qualname`, `Worker.init_device()` hardcoding stock-runner construction
    as its last act with nothing depending on which class was built).

    Usage: pass this class's dotted path as `worker_cls` when constructing
    the SPECULATOR's own `LLM(...)` (never the target's -- see
    `proposer.py`):

        speculator_llm = LLM(
            model=speculator_model_path,
            worker_cls="vllm_patch.speculator_worker.SpeculatorWorker",
            enable_prefix_caching=True,
            enforce_eager=True,
            ...
        )
    """

    def init_device(self) -> None:
        super().init_device()
        self.model_runner = SpeculatorGPUModelRunner(self.vllm_config, self.device)

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        self.model_runner.install_query_capture_hooks()

    def begin_capture(self, request_id: str) -> None:
        self.model_runner.begin_capture(request_id)

    def end_capture(self, request_id: str) -> List[torch.Tensor]:
        """Returns raw `torch.Tensor`s directly -- safe to do because
        `proposer.py` sets `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (see that
        module's docstring for the full reasoning and the exact
        `vllm/v1/serial_utils.py` code path this depends on) before this
        engine is ever constructed, which makes `collective_rpc`'s
        `UtilityResult` wrapper carry proper type info for nested Tensors
        -- msgspec's own dedicated `_encode_tensor` codec then moves the
        buffer directly, no Python-list round trip needed.

        The `.to("cpu")` here (not inside `SpeculatorGPUModelRunner.
        end_capture`, which now stays on-device -- see that method's
        docstring) is deliberately done at THIS boundary: this is an
        RPC-exposed method whose result crosses `collective_rpc`'s process
        boundary, where a CUDA tensor's storage isn't valid to reconstruct
        in the driver process's own (different) CUDA context. Caller
        (`proposer.py::run_turn`) then does `.to(self.device)` to bring it
        back onto the SPECULATOR's device from the DRIVER process's own
        perspective (a separate, necessary hop -- collective_rpc's decode
        happens in the driver process, which has no CUDA context of the
        worker's device unless it moves the data itself)."""
        return [t.to("cpu") for t in self.model_runner.end_capture(request_id)]

    def retrieve_keys(
        self, conversation_salt: str, positions: List[int]
    ) -> List[torch.Tensor]:
        """Returns raw `torch.Tensor`s directly -- see `end_capture`'s
        docstring above for why the `.to("cpu")` lives here now, not in
        `SpeculatorGPUModelRunner.retrieve_keys`. Kept as a standalone RPC
        method (not removed now that `end_capture_and_score` below does K
        retrieval internally, on-device) because `validate_proposer.py`'s
        Step C calls it directly to validate K read-back in isolation, and
        `proposer.py` still exposes it as a public method for the same
        reason."""
        return [t.to("cpu") for t in self.model_runner.retrieve_keys(conversation_salt, positions)]

    def end_capture_and_score(
        self,
        request_id: str,
        conversation_salt: str,
        full_sequence_len: int,
        pool_kernel_size: Optional[int],
        keep_kwargs: dict,
        score_aggregation: str = "max",
        score_layers: Optional[str] = None,
    ):
        """RPC-callable wrapper -- see `SpeculatorGPUModelRunner.
        end_capture_and_score`'s docstring for the full reasoning (in-
        process K retrieval + scoring, only the small resulting index list
        crosses `collective_rpc`).

        `score_aggregation`/`score_layers` default to the reference
        behavior, so this stays callable with the old 5-argument signature."""
        return self.model_runner.end_capture_and_score(
            request_id, conversation_salt, full_sequence_len, pool_kernel_size,
            keep_kwargs, score_aggregation, score_layers,
        )

    def end_capture_and_head_diagnostics(
        self,
        request_id: str,
        conversation_salt: str,
        full_sequence_len: int,
        pool_kernel_size,
        keep_kwargs: dict,
        gold_positions: list,
        top_n_list: list,
        fixed_head_sets: list = None,
    ):
        """RPC-callable wrapper -- see `SpeculatorGPUModelRunner.
        end_capture_and_head_diagnostics`'s docstring."""
        return self.model_runner.end_capture_and_head_diagnostics(
            request_id, conversation_salt, full_sequence_len, pool_kernel_size,
            keep_kwargs, gold_positions, top_n_list, fixed_head_sets,
        )

    def discard_conversation(self, conversation_salt: str) -> None:
        self.model_runner.discard_conversation(conversation_salt)
