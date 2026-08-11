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
    read_layer_keys,
    stack_decode_only_steps,
    tensor_to_wire,
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
        that method's docstring), every one of this request's forward
        calls gets captured, INCLUDING the bootstrap prefill's own -- which
        must be excluded here, not by timing. The bootstrap's own capture
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

        Moved to CPU before returning -- collective_rpc results cross a
        process boundary; a CUDA tensor's storage isn't valid to
        reconstruct in the driver process's own (different) CUDA context,
        and the tensors here are small (bounded by look_ahead_cnt *
        num_layers * hidden_size) so the CPU copy is cheap relative to the
        forward passes that produced them."""
        self._capturing_request_ids.discard(request_id)
        per_layer_steps = self._query_buffer.pop(request_id, None)
        if per_layer_steps is None:
            return [torch.empty(1, 0, 0) for _ in self._hooked_layers]

        result = []
        for layer_idx, steps in enumerate(per_layer_steps):
            hidden_dim_fallback = (
                self._hooked_layers[layer_idx].num_heads
                * self._hooked_layers[layer_idx].head_dim
            )
            result.append(stack_decode_only_steps(steps, hidden_dim_fallback).to("cpu"))
        return result

    def _capture_queries_by_request(self, layer_idx: int, q: torch.Tensor) -> None:
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
            if end <= start:
                continue
            self._query_buffer[req_id][layer_idx].append(q[start:end])

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
        """Per-layer [len(positions), num_kv_heads, head_size] K tensors
        (CPU, see `end_capture`'s docstring for why) for the given absolute
        LOCAL positions within this conversation's own speculator prompt
        (i.e. indices into what `proposer.py` submitted as this
        conversation's growing prompt -- NOT the multi-turn ledger's
        absolute positions used for the TARGET's RoPE restoration, a
        separate numbering -- see `proposer.py`'s docstring for how the two
        relate)."""
        slots = self.get_slots_for_positions(conversation_salt, positions)
        slot_tensor = torch.tensor(slots, dtype=torch.int64, device=self.device)
        block_size = self.vllm_config.cache_config.block_size
        keys = []
        for self_attn in self._hooked_layers:
            num_kv_heads = self_attn.num_kv_heads
            head_size = self_attn.head_dim
            flat_keys = read_layer_keys(self_attn.attn, block_size, num_kv_heads, head_size)
            keys.append(gather_keys_for_slots(flat_keys, slot_tensor).to("cpu"))
        return keys

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

    def end_capture(self, request_id: str) -> List[dict]:
        """Wire-encoded (see `kv_cache_utils.tensor_to_wire`'s docstring for
        why) -- a raw `torch.Tensor` does NOT survive `collective_rpc`'s
        process boundary intact in this fork (confirmed on real hardware:
        comes back as a bare nested list, silently losing dtype/shape).
        Caller (`proposer.py`) must decode with `tensor_from_wire`."""
        return [tensor_to_wire(t) for t in self.model_runner.end_capture(request_id)]

    def retrieve_keys(
        self, conversation_salt: str, positions: List[int]
    ) -> List[dict]:
        """Wire-encoded -- see `end_capture`'s docstring above."""
        return [
            tensor_to_wire(t)
            for t in self.model_runner.retrieve_keys(conversation_salt, positions)
        ]

    def discard_conversation(self, conversation_salt: str) -> None:
        self.model_runner.discard_conversation(conversation_salt)
