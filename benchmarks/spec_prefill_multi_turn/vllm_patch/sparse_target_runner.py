"""SparseTargetWorker / SparseTargetGPUModelRunner -- the target-side half
of the persistent-KV-cache + speculator-guided sparse attention
architecture (see the approved plan, `Target-Side Persistent KV Cache +
Speculator-Guided Sparse Attention`, and `validate_resumable_session.py`'s
module docstring for the resumable-session foundation this builds on,
confirmed working on real hardware: turn 2's TTFT was dramatically lower
than turn 1's, consistent with genuine cross-turn KV persistence).

## What this does, in one sentence

Every token's KV entry is computed once and never discarded (the target
session's own persistent cache, via the resumable-request mechanism); this
runner restricts which of those already-resident entries actually
participate in attention, per a selection the speculator (unchanged,
`proposer.py`/`speculator_worker.py`) recomputes fresh every turn -- see
`sparse_selection_registry.py`'s module docstring for why no
position-translation layer is needed between the speculator's selection and
this runner's own consumption of it (both track the same absolute, gapless
conversation-ledger numbering, unlike the OTHER pipeline's
physically-pruned, gap-containing prompts).

## Two scopes: decode-only (default) and prefill+decode (opt-in)

Which STEPS get restricted is decided per registration, by whether the
driver passes a `prefill_turn_start` (see `sparse_selection_registry.
register`):

- **Decode-only** (`prefill_turn_start=None`, the default): only decode
  steps are gathered. Each turn's own new query tokens prefill against the
  full, unrestricted resident cache. This is the originally-shipped
  behavior and what every published `SPARSE-k*`/`ORACLE-k*` row was
  measured under, so it is what a driver that says nothing still gets.
- **Prefill + decode** (`prefill_turn_start=<this turn's start position>`):
  the turn's own prompt tokens additionally attend only to the selected
  blocks of the pre-existing history, plus a contiguous force-kept tail
  covering the turn's own tokens so far. See
  `kv_cache_utils.compute_prefill_gather_view` for the mechanism and, in
  particular, for the causal-mask invariant that makes a multi-token query
  chunk legal against a compacted K/V view at all -- the reason this was
  scoped out of the first pass rather than merely deferred.

Turn 0 is dense under BOTH scopes, and not by special-casing: its
`turn_start` is 0, so the force-kept tail spans everything and the gather
degenerates to a no-op. That is the required answer, not a convenience --
turn 0's prefill is where the context's KV is computed for the first time,
and computing it under a restricted view would poison the persistent cache
that every later turn's selection reads from.

What does NOT change under either scope: `slot_mapping` is never patched,
so every token's KV is still WRITTEN in full, at its true physical slot.
The selection only ever masks reads. That is what keeps a token dropped at
turn 2 genuinely re-selectable at turn 5, which is the whole premise of
KEEP mode and the one thing the physically-pruned sibling pipeline cannot
do.

## Why `_build_attention_metadata`, not `_prepare_inputs`

The OTHER pipeline's `model_runner.py` overrides `_prepare_inputs()` to
patch RoPE positions -- traced (for THIS architecture) that `_prepare_
inputs()` runs too early: it never touches `AttentionMetadata` at all, only
`self.positions`/`self.query_start_loc`/slot_mapping bookkeeping. The real
call chain in `gpu_model_runner.py`: `execute_model()` -> `_prepare_
inputs()` -> `_build_attention_metadata()` (builds real per-layer
`AttentionMetadata` via `builder.build()`/`update_block_table()`) ->
`set_forward_context()` -> model forward -> attention kernel. Overriding
`_build_attention_metadata()` runs strictly after the stock per-layer
metadata exists and strictly before the kernel ever sees it -- the correct
injection point for this feature, confirmed by reading the real method
chain, not assumed by analogy to the other pipeline's different hook.

## No RoPE position patching needed here at all

Unlike the physically-pruned pipeline, this one needs NO position-restoration
machinery (`pruning_registry.py`/the other `model_runner.py` stay completely
untouched, used only by that other, still-running pipeline). Two independent
reasons, both confirmed against real source, not assumed:
1. RoPE is baked into K at WRITE time, not read time -- a gathered K vector
   already carries its correct rotation regardless of where it lands in a
   shrunk/gathered view.
2. The query token's own RoPE position comes from `self.positions` (a
   SEPARATE `CommonAttentionMetadata` field, populated by `_prepare_inputs`,
   which this runner never touches) -- reflecting its true, ever-incrementing
   absolute position in the continuously-open session, correct by
   construction with no override needed.

## Why `.query_start_loc` doesn't need patching, but `.max_seq_len` does

`query_start_loc` describes where each request's QUERY tokens start/end
within the batched query tensor -- for a single decode step (always exactly
1 new query token per request under this pipeline's own established
one-in-flight-request scope, same convention `proposer.py`/`predict_
scbench.py` already use elsewhere), this is unaffected by how much of the
KEY/VALUE side (`block_table`/`seq_lens`) gets gathered. It stays unpatched
under the prefill scope too, and for the same reason: it is a QUERY-side
description, and gathering changes only the K/V side. What the multi-token
query chunk DOES require of the gather is that the chunk's own tokens land
at the tail of the compacted view, so bottom-right causal alignment still
picks out the right keys -- enforced by `compute_prefill_gather_view`'s
contiguous force-kept tail, documented there.

`max_seq_len`, by contrast, DOES need patching, and originally wasn't --
a real bug caught via `diagnose_h1_metadata.py` against real hardware,
not a theoretical gap. Every backend's per-layer `AttentionMetadata` also
carries a batch-wide `max_seq_len` scalar (e.g.
`FlashAttentionMetadata.max_seq_len`) computed ONCE by the stock metadata
builder from the ORIGINAL, unrestricted `seq_lens`, separate from the
per-request `seq_lens` tensor this module already patches. The flash-attn
backend feeds it straight to the kernel as `max_seqlen_k` alongside the
(correctly shrunk) `block_table`/`seqused_k`
(`vllm/v1/attention/backends/flash_attn.py`'s `forward()`), where it sizes
the kernel's own K/V-block tiling/grid for the call -- left stale, the
kernel kept reading/iterating the FULL pre-gather span regardless of how
aggressively `block_table`/`seq_lens` were shrunk, which is exactly why an
early microbenchmark measured ~flat decode latency between keep=1.0 and
keep=0.2 despite `block_table`'s distinct block count and `seq_lens`
themselves being genuinely, correctly restricted. See
`_apply_sparse_attention_overrides`'s post-loop `max_seq_len` fix-up below.

## Force-keep: this turn's own in-progress decode tokens

The driver registers a selection (via `sparse_selection_registry.py`)
BEFORE a turn's query update is submitted -- which necessarily happens
before any of THIS turn's own response tokens exist yet, so they can never
be part of that registered selection. This runner therefore always
force-includes, on top of whatever's registered, every block spanning
`[num_prompt_tokens, num_computed_tokens)` for the active request --
i.e. whatever this turn has generated so far, unconditionally -- so the
model always has coherent access to its own just-generated tokens within
the same turn, mirroring the physically-pruned pipeline's own
force-keep-the-query-span rule, generalized here to "force-keep this
turn's own tokens so far."

## Per-turn caching of the block-index computation (real perf fix, not
## speculative)

A first real smoke test against SCBench (large contexts, KEEP mode, so
`selected_positions` can be tens of thousands of absolute ledger positions)
measured a catastrophic per-conversation slowdown vs. the physically-pruned
sibling pipeline. Traced to `compute_sparse_gather_view` originally
recomputing `{p // block_size for p in selected_positions}` -- an
O(len(selected_positions)) Python set comprehension -- on every single call
to `_apply_sparse_attention_overrides`, which runs once per DECODE STEP
(`_build_attention_metadata` is called every step, not once per turn), i.e.
up to `max_tokens` times (e.g. 64) per turn, even though the registered
selection is constant for the whole turn. `_get_base_block_indices` above
computes and caches this set ONCE per turn, invalidated by
`sparse_selection_registry`'s monotonic generation counter changing (which
happens exactly once per `register_sparse_selection` call -- see that
module's docstring). This cache was originally keyed on `id()` of the
registry's `selected_positions` object instead -- a real, confirmed bug
(non-deterministic output from stale cache hits after the old list was
freed and its address reused by a later turn's own allocation), fixed by
switching to the generation counter; see `sparse_selection_registry.py`'s
"Generation counter" section for the full trace.

That fix alone turned out to be incomplete: `compute_sparse_gather_view`
was still redoing a full `sorted()` + tensor-gather-from-`full_block_table_
row` + seq_len Python loop over the ENTIRE cached selection on every one of
those up-to-`max_tokens` decode-step calls, even though only the small
force-keep tail (and possibly one boundary block's occupancy) actually
changes step to step -- a synthetic benchmark reproducing a 64-step decode
turn at SCBench-scale selection sizes measured this remaining per-step cost
at 5-32x the alternative below. `_get_base_gather_view` now additionally
caches `kv_cache_utils.compute_base_gather_view`'s result (the gathered
rows + summed seq_len for every block that's PROVABLY stable for the rest
of the turn -- see that function's own docstring for the historical-
position invariant this relies on) the same way, ONCE per turn; each decode
step then calls `kv_cache_utils.compute_sparse_gather_view_incremental`,
which reuses that cached base view as-is and only computes the bounded,
per-step delta (the force-keep tail plus the one block whose occupancy can
still be growing).

## Known risk areas -- NOT independently confirmed on real hardware yet
(unlike the resumable-session foundation this builds on, which IS confirmed
-- see `validate_resumable_session.py`). `validate_sparse_attention.py`
is where these should be checked first, before trusting any real sweep:

1. **Per-layer `AttentionMetadata` field names.** Assumed `.block_table`
   and `.seq_lens` based on the research pass's citations of
   `TritonAttentionMetadata`/`FlashAttentionMetadata`'s real field names
   -- not independently re-verified here. If a backend uses a different
   field name, this module's `_gather_fields_for_layer` will raise a clear
   `AttributeError`-derived message rather than silently doing nothing.
2. **Single KV-cache-group, batch-size-1 assumption.** Mirrors
   `speculator_worker.py`'s own documented "single KV-cache-group"
   assumption (reasonable for Llama-3.1-8B, dense/uniform attention) and
   this pipeline's own established "one in-flight request at a time"
   scope -- this module does not attempt the general multi-request virtual-
   batch reconstruction `make_local_attention_virtual_batches` implements,
   only a single-row, single-group in-place patch.
3. **Block occupancy at the gather boundary.** The gathered view's
   `seq_lens` value must account for the LAST included block possibly
   being only PARTIALLY filled (the request's own most-recently-written
   block) -- computed explicitly in `_compute_gathered_view`, not assumed
   to always be a clean multiple of `block_size`. Getting this wrong would
   let the kernel read uninitialized/garbage slots past the true occupied
   region of that block.
4. **Whether `attn_metadata[layer_name]` objects are safely shared or
   need per-layer independent gathers.** Assumed uniform across all layers
   (same block_size, same resident cache shape) for Llama-3.1-8B's dense,
   non-heterogeneous attention -- the SAME gathered block_table/seq_lens is
   written into every layer's metadata object identically, not
   independently recomputed per layer.
5. **`max_seq_len` was originally NOT in this list and should have been --
   confirmed as a real bug, not a theoretical gap, via
   `diagnose_h1_metadata.py` against real hardware** (see "Why
   `.max_seq_len` does [need patching]" above): a first pass at this
   mechanism patched only `.block_table`/`.seq_lens`, leaving `max_seq_len`
   at its stale, pre-gather value -- which measurably kept the flash-attn
   backend reading the FULL pre-gather KV span regardless of how much
   `block_table`/`seq_lens` were shrunk. Fixed by recomputing it as
   `seq_lens.max()` after the per-request loop, gated on `any_patched` --
   and computed from a SINGLE representative layer, not once per layer
   (a first version of this exact fix called `.max().item()` separately for
   each of Llama-3.1-8B's 32 layers, adding 32 GPU-sync points to every
   sparse-path decode step and -- confirmed on real hardware -- making the
   sparse path measurably SLOWER than dense, not just a smaller
   improvement than expected).
   The follow-on question this raised -- whether any OTHER backend-specific
   scheduling field similarly needed recomputing after the gather -- is now
   answered: see item 7 below. `scheduler_metadata` staleness turned out to
   be exactly that, and a correctness bug, not just the performance one
   `max_seq_len` was.
6. **The per-turn caches were keyed on `id(selected_positions)` -- a real,
   confirmed correctness bug, not a style nit.** `id()` is only guaranteed
   unique while the object is alive; the cache entry it invalidates against
   OUTLIVES the object (nothing clears `_sparse_base_block_indices_cache`/
   `_sparse_base_gather_view_cache` when a turn ends), so once the old
   `selected_positions` list is freed and CPython reuses its address for
   the very next turn's own `list(...)` allocation, the cache can silently
   serve a STALE, mismatched selection for the new turn. Confirmed as the
   real cause of a production symptom: `predict_scbench.py` SPARSE runs
   produced genuinely non-deterministic output -- different generations
   from IDENTICAL temperature=0 re-runs of the same conversation/turn --
   that degenerated into repetition loops, consistent with attending to a
   selection computed for a different (older, shorter) context. Fixed by
   replacing `id()` with a monotonic generation counter in
   `sparse_selection_registry.py`, which has no relationship to any
   object's memory address and so cannot collide this way. See that
   module's "Generation counter" section for the full trace.
7. **`scheduler_metadata` was stale too -- a second, independent
   correctness bug, confirmed even AFTER items 5 and 6 above were both
   fixed.** With the `max_seq_len` and `id()` bugs both fixed, real SCBench
   runs became deterministic but still produced garbled, repetition-loop
   output from the very first decode token -- even at `keep=0.8` (losing
   only 20% of context), with the SAME model/dataset's `M000` baseline
   clean throughout. `diagnose_target_gather_metadata.py` proved
   `block_table`/`seq_lens`/`max_seq_len` were all being patched exactly
   as expected (an independently-computed block count matched the real
   gather's reported count at every decode step) -- yet the output was
   still broken, meaning the bug had to be in something the gather never
   touched at all. `scheduler_metadata` (`vllm/v1/attention/backends/
   flash_attn.py`'s FA3 ahead-of-time work-scheduling tensor, see
   `_SCHEDULER_METADATA_FIELD`'s own comment above for the mechanism) is
   built ONCE by the stock builder from the PRE-GATHER `seq_lens`/
   `max_seq_len`, encoding how the kernel should partition/process K/V
   tiles for the ORIGINAL, unrestricted context length -- never patched by
   anything in this file until now, regardless of how correctly the
   logical fields were shrunk. Fixed by nulling it out (not recomputing
   it) for any patched request, falling back to FA3's own non-AOT
   scheduling from the corrected fields directly -- `schedule()`'s own
   `else: return None` branch in the stock builder already proves `None`
   is a legitimate input the kernel must handle.
8. **Off-by-one: the gather was built for `num_computed_tokens`, not the
   step's real sequence length -- a THIRD independent correctness bug,
   confirmed after 5, 6 and 7 were all fixed and still producing
   repetition loops at every keep rate below the dense baseline.** This
   method fed `self.input_batch.num_computed_tokens_cpu_tensor[req_idx]`
   straight into `kv_cache_utils`'s gather as if it were the sequence
   length. It isn't: the stock runner computes
   `self.seq_lens[:num_reqs] = self.num_computed_tokens[:num_reqs] +
   num_scheduled_tokens_gpu` (`gpu_model_runner.py`), i.e.
   `num_computed_tokens_cpu` is the count BEFORE this step's own scheduled
   token, so a decode step's true length is one MORE. Every length the
   gather derived (`num_full_blocks_present`, `last_real_block_index`,
   `last_real_block_occupancy`, and the `range(num_prompt, ...)`
   force-keep tail) was therefore one token short, with two consequences
   on EVERY decode step: the token currently being decoded could not
   attend to its own key/value, and whenever it happened to open a fresh
   block (`seq_len % block_size == 1`) the block physically holding it
   wasn't in the gathered block table at all. Fixed by reading
   `_step_seq_len()` (from `optimistic_seq_lens_cpu`, the same CPU tensor
   the stock builder derives `seq_lens`/`max_seq_len` from) and renaming
   the `kv_cache_utils` parameter from `num_computed` to `seq_len` so the
   contract can't be misread the same way again.

   Why the existing checks all missed it: the gather's own unit tests were
   self-consistent with the wrong caller (they passed the same
   "num_computed" convention the caller did), `compute_sparse_gather_view`
   returns `None` for a full selection so keep=1.0-shaped cases never
   exercised the arithmetic, and `diagnose_target_gather_metadata.py`
   compares only the BLOCK COUNT of the gather against an independent
   recomputation -- never the `seq_lens` value, which is where the whole
   defect lived.
"""

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from . import sparse_selection_registry

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

# Fields assumed present on every backend's per-layer AttentionMetadata
# dataclass that this module patches -- see module docstring's "Known risk
# areas" #1 for exactly how this assumption was arrived at (research-pass
# citations, not independently re-verified on real hardware here).
_BLOCK_TABLE_FIELD = "block_table"
_SEQ_LENS_FIELD = "seq_lens"
# Optional -- not every backend necessarily has this (unlike block_table/
# seq_lens, which this whole mechanism requires), so it's patched
# best-effort via hasattr() below rather than through the same strict
# _get_field() that raises for the other two. See module docstring's
# "Why .max_seq_len does" section for why this needs patching at all.
_MAX_SEQ_LEN_FIELD = "max_seq_len"
# Also optional/best-effort, same reasoning as _MAX_SEQ_LEN_FIELD -- FA3's
# ahead-of-time work-scheduling tensor, built ONCE by the stock builder
# from the PRE-GATHER seq_lens/max_seq_len (vllm/v1/attention/backends/
# flash_attn.py's `schedule()` -> `get_scheduler_metadata(cache_seqlens=
# seqlens, max_seqlen_k=max_seq_len, page_size=block_size, ...)`) and never
# recomputed by anything in this module. Confirmed as a REAL, correctness
# (not just performance) bug via `diagnose_target_gather_metadata.py`
# against real hardware: block_table/seq_lens/max_seq_len all patch
# correctly, but the target still produced garbled, repetition-loop output
# from the very first decode token -- because the kernel was still handed
# a work schedule built for the ORIGINAL, unrestricted context length,
# telling it to partition/process far more K/V tiles than the shrunk data
# actually has. Fixed by nulling it out for any patched request rather
# than trying to recompute it -- `schedule()`'s own `else: return None`
# branch (when `aot_schedule` is off) proves `None` is already a
# legitimate input `flash_attn_varlen_func` must handle, so this falls
# back to FA3's own non-AOT scheduling from the now-correct seq_lens/
# block_table directly, instead of risking a subtly wrong hand-rolled call
# to `get_scheduler_metadata` (which needs several backend-internal values
# -- num_heads, headdim, qkv_dtype, num_splits -- not readily available in
# this override's own scope).
_SCHEDULER_METADATA_FIELD = "scheduler_metadata"


class SparseTargetGPUModelRunner(GPUModelRunner):
    def _base_block_indices_cache(self) -> Dict[str, tuple]:
        # Lazily-initialized rather than added to __init__ -- same
        # *args/**kwargs-passthrough philosophy as this class's own
        # _build_attention_metadata override: GPUModelRunner's real
        # constructor signature is large and version-sensitive, so this
        # avoids needing to know or forward it just to add one cache dict.
        # Keyed by req_id -> (selection_generation at cache time, the
        # resulting block-index set). NOT id(selected_positions) -- a real,
        # confirmed bug: id() is only unique while the object is alive, and
        # the cache entry OUTLIVES it (nothing clears this dict when a turn
        # ends), so once the old list is freed and its address gets reused
        # by the NEXT turn's own `list(selected_positions)` allocation, a
        # stale cache entry could silently match a completely different
        # turn's selection. Confirmed as the real cause of non-deterministic
        # SPARSE output (different results from identical temperature=0
        # re-runs) that degenerated into repetition loops -- see
        # `sparse_selection_registry.py`'s "Generation counter" section for
        # the full trace. `selection_generation` is a monotonic counter with
        # no relationship to any object's memory address, so it can't
        # collide this way.
        if not hasattr(self, "_sparse_base_block_indices_cache"):
            self._sparse_base_block_indices_cache: Dict[str, tuple] = {}
        return self._sparse_base_block_indices_cache

    def _get_base_block_indices(
        self, req_id: str, selected_positions: List[int], block_size: int, selection_generation: int
    ):
        # Caches `block_indices_from_positions` per request, invalidated
        # whenever `selection_generation` changes (a new turn's
        # registration -- see module docstring): without this, this
        # O(len(selected_positions)) computation (which can be tens of
        # thousands of entries for a large SCBench context) was being redone
        # on EVERY decode step of a turn (up to `max_tokens` times), a real,
        # measured per-conversation slowdown.
        from .kv_cache_utils import block_indices_from_positions

        cache = self._base_block_indices_cache()
        cached = cache.get(req_id)
        if cached is not None and cached[0] == selection_generation:
            return cached[1]
        base_block_indices = block_indices_from_positions(selected_positions, block_size)
        cache[req_id] = (selection_generation, base_block_indices)
        return base_block_indices

    def _base_gather_view_cache(self) -> Dict[str, tuple]:
        # Same lazy-init / generation-keyed-invalidation reasoning as
        # _base_block_indices_cache above -- see this class's module
        # docstring's "Per-turn caching" section for why a SECOND cache
        # layer (on top of the block-index set) is needed: the block-index
        # set alone doesn't avoid re-sorting/re-gathering the whole
        # selection from full_block_table_row on every decode step.
        if not hasattr(self, "_sparse_base_gather_view_cache"):
            self._sparse_base_gather_view_cache: Dict[str, tuple] = {}
        return self._sparse_base_gather_view_cache

    def _get_base_gather_view(
        self,
        req_id: str,
        selected_positions: List[int],
        block_size: int,
        full_block_table_row,
        num_prompt: int,
        selection_generation: int,
    ):
        """Caches `kv_cache_utils.compute_base_gather_view`'s result per
        request, invalidated by the same `selection_generation` signal
        `_get_base_block_indices` already uses (see
        `sparse_selection_registry.py`'s "Generation counter" section for
        why this replaced an earlier, buggy `id(selected_positions)`
        scheme). Safe to cache `full_block_table_row`-derived data keyed
        only on the selection's generation (not also on `num_prompt`/the
        row's own contents) because both are themselves constant for the
        whole turn: `num_prompt` is fixed once a turn's query is submitted,
        and the STABLE blocks this caches are -- by `compute_base_gather_
        view`'s own historical-position invariant -- never touched again by
        this or any later decode step of the same turn."""
        from .kv_cache_utils import compute_base_gather_view

        cache = self._base_gather_view_cache()
        cached = cache.get(req_id)
        if cached is not None and cached[0] == selection_generation:
            return cached[1]
        base_block_indices = self._get_base_block_indices(
            req_id, selected_positions, block_size, selection_generation
        )
        base_view = compute_base_gather_view(
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            base_block_indices=base_block_indices,
            num_prompt=num_prompt,
        )
        cache[req_id] = (selection_generation, base_view)
        return base_view

    def _build_attention_metadata(self, *args, **kwargs):
        # *args/**kwargs passthrough -- see module docstring: this method's
        # real signature (num_tokens, num_reqs, max_query_len, ... many more
        # scalar/tensor params) is large and version-sensitive; passing
        # through opaquely (same reasoning as model_runner.py's identical
        # *args/**kwargs choice for _prepare_inputs) means this override
        # doesn't need to track every parameter the stock method takes, only
        # what it needs to read back afterward.
        attn_metadata, spec_decode_meta = super()._build_attention_metadata(*args, **kwargs)
        self._apply_sparse_attention_overrides(attn_metadata)
        return attn_metadata, spec_decode_meta

    def _gatherable_layer_names(self, attn_metadata) -> set:
        """Which layers the block-table gather may legally be applied to.

        **Sliding-window layers must be excluded, and this is a correctness
        requirement, not an optimisation.** The gather COMPACTS the KV view:
        selected blocks are packed to the front of the block table and
        `seq_lens` shrinks to match. A sliding-window kernel decides window
        membership from a key's index within `seqused_k` -- so after
        compaction, key j is treated as if it sat at position j, which it no
        longer does. Every distance the kernel computes is wrong: keys that
        are genuinely far away get admitted, near ones get masked out. This
        is the same position-shift hazard `kv_cache_utils.
        compute_prefill_gather_view` handles for causal masking with a
        contiguous force-kept tail, but a tail cannot fix it here -- a
        window must be contiguous in TRUE positions, and a top-k block
        selection is precisely what is not.

        Excluding them costs nothing that was ever really available. A
        sliding layer already reads at most its own window (512-1024 tokens),
        so there was no long-context KV traffic there to save. On
        Gemma-4-E2B that is 28 of 35 layers, which is the same fact the
        economics reach from the other side: this mechanism's headroom on an
        interleaved model lives on the ~1/6 of layers that attend globally.

        Computed once and cached -- layer types cannot change after load.
        On a uniform-attention model every layer is gatherable and this is a
        provable no-op, so the published Llama rows take an identical path.
        """
        cached = getattr(self, "_gatherable_layer_names_cache", None)
        if cached is not None:
            return cached

        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention import Attention

        from .model_structure import gatherable_layer_names

        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        gatherable = gatherable_layer_names(attn_metadata, layers)
        skipped = len(attn_metadata) - len(gatherable)
        if skipped:
            print(
                f"[SparseTarget] gather restricted to {len(gatherable)} of "
                f"{len(attn_metadata)} layers; {skipped} sliding-window "
                f"layer(s) left dense (a compacted view would misplace their "
                f"window -- see _gatherable_layer_names)",
                flush=True,
            )
        if not gatherable:
            raise NotImplementedError(
                "every attention layer in this model is sliding-window, so "
                "there is nothing the block gather can legally restrict. "
                "This mechanism has no effect on such a model."
            )
        self._gatherable_layer_names_cache = gatherable
        return gatherable

    def _privatise_gathered_seq_lens(self, attn_metadata, gatherable) -> None:
        """Give the gathered layers their own `seq_lens` tensor before it is
        written to.

        **Confirmed on real hardware, and the cause of total generation
        collapse on Gemma 4.** vLLM builds attention metadata per KV cache
        group, and on an interleaved model the sliding and full-attention
        layers land in different groups with genuinely separate
        `block_table` tensors -- but they SHARE one `seq_lens` tensor. So
        `seq_lens[req_idx] = gathered_seq_len`, applied to "only the
        gathered layers", also rewrites the sequence length every
        sliding-window layer reads.

        Their block table stays dense and correct, so they still address the
        right physical blocks; they are simply told the sequence is shorter
        than it is. A sliding-window kernel derives its window from the
        sequence length, so the window lands in the wrong place -- every one
        of them attends to a region that has nothing to do with the query.

        The observed signature matched exactly: correct output at keep=100%
        (the gather returns None, nothing is written), degenerate output at
        ANY lower keep rate regardless of how much was retained, and only
        mild damage in `validate_sparse_attention.py`, whose context is
        smaller than one window so a misplaced window still covers most of
        it.

        Invisible on Llama, and not by luck: a uniform-attention model has
        ONE KV cache group, so every layer is gathered and there is no
        unpatched layer left to corrupt. This bug can only exist where the
        gather is partial.

        Re-established every step. **Two metadata regimes exist and this
        must handle both**, which is not obvious and cost a real run:

        - EAGER: the stock builder rebuilds the metadata each step, so the
          gathered layer arrives aliasing a sliding layer's tensor again.
        - CUDAGRAPHS (`FULL_DECODE_ONLY`): vLLM keeps PERSISTENT metadata
          objects across steps -- that is precisely what capture requires,
          so the privatising `setattr` survives and the gathered layer no
          longer holds the shared tensor at all.

        In the second regime, copying from the gathered layer's own tensor
        would silently re-serve last step's values. So the fresh values are
        always sourced from a SLIDING layer, which is never patched and
        therefore always holds the true, in-place-updated tensor. Observed
        on Gemma-4-31B at `layers.17.self_attn.attn` the first time
        `--target-cudagraph-mode FULL_DECODE_ONLY` was run.

        Cheap either way: `seq_lens` holds one int per request, unlike
        `block_table`, which is already per-group and must NOT be copied on
        the hot path.

        **Written into a PERSISTENT buffer rather than a fresh `.clone()`,
        because a fresh clone is a hard CUDA-graph blocker.** Graph replay
        bakes in the ADDRESS of every tensor a captured kernel read, so
        handing the metadata a newly-allocated tensor each step means replay
        keeps reading whichever one existed at capture time -- silently
        stale, not an error. Allocating once and `copy_()`-ing into it keeps
        the address fixed while the contents track the step, which is the
        invariant full cudagraphs need.

        That is not a hypothetical requirement: on Gemma-4-31B, TP=2, a
        stock unmodified runner measures 85.9 ms/tok eager against 24.6
        ms/tok with `FULL_DECODE_ONLY` -- a 3.49x difference. The sparse
        path cannot claim any of it while it swaps tensors per step, and
        enabling graphs only where they already work would speed up the
        dense baseline alone and widen the very gap this pipeline is trying
        to close.
        """
        sliding_sources: Dict[int, object] = {}
        for name, md in attn_metadata.items():
            if name in gatherable:
                continue
            shared = self._get_field(md, _SEQ_LENS_FIELD)
            sliding_sources.setdefault(shared.data_ptr(), shared)
        if not sliding_sources:
            return  # uniform-attention model: nothing to protect

        buffers = self._private_seq_lens_buffers()
        assigned: Dict[int, object] = {}
        # Sorted so a buffer's identity is tied to a STABLE key across steps;
        # `attn_metadata` iteration order is not something to rely on when
        # the whole point is address stability.
        for name in sorted(gatherable):
            layer_metadata = attn_metadata[name]
            seq_lens = self._get_field(layer_metadata, _SEQ_LENS_FIELD)
            ptr = seq_lens.data_ptr()

            if ptr in sliding_sources:
                # Un-privatised: the builder rebuilt (or this is step one),
                # and this layer still aliases a sliding layer's tensor.
                source_ptr = ptr
            else:
                owned = buffers.index_by_ptr.get(ptr)
                if owned is None:
                    continue  # private already, and not shared with sliding
                # Already holding one of ours -- the metadata was NOT rebuilt.
                # That is the normal state under cudagraphs, which keep
                # persistent metadata objects precisely so captured kernels
                # keep reading the same addresses. Refresh from the sliding
                # layer that still holds the un-privatised tensor; reading
                # our own buffer would just re-copy last step's values.
                source_ptr = buffers.source_ptr_by_index.get(owned)
                if source_ptr not in sliding_sources:
                    raise RuntimeError(
                        f"layer {name!r} holds privatised seq_lens buffer "
                        f"{owned} whose source tensor is no longer present "
                        f"among the sliding layers, so its contents cannot "
                        f"be refreshed. Serving them would be a step stale, "
                        f"and a wrong seq_lens is the silent corruption this "
                        f"method exists to prevent."
                    )

            if source_ptr not in assigned:
                source = sliding_sources[source_ptr]
                buf = buffers.get(len(assigned), source, source_ptr)
                buf.copy_(source)
                assigned[source_ptr] = buf
            setattr(layer_metadata, _SEQ_LENS_FIELD, assigned[source_ptr])

    class _SeqLensBuffers:
        """Persistent private `seq_lens` tensors, keyed by a stable index.

        Also remembers, per buffer, WHICH shared tensor it mirrors. That is
        what makes the buffers refreshable in both metadata regimes -- see
        `_privatise_gathered_seq_lens`, which has to cope with the stock
        builder rebuilding metadata every step (eager) AND with it handing
        back the same objects every step (cudagraphs).
        """

        def __init__(self) -> None:
            self.by_index: Dict[int, object] = {}
            self.index_by_ptr: Dict[int, int] = {}
            self.source_ptr_by_index: Dict[int, int] = {}

        def get(self, index: int, like, source_ptr: int) -> object:
            buf = self.by_index.get(index)
            if (buf is None or buf.shape != like.shape
                    or buf.dtype != like.dtype or buf.device != like.device):
                # Reallocating changes the address and would invalidate any
                # captured graph -- but it only happens on a shape/dtype
                # change, which invalidates the graph anyway.
                if buf is not None:
                    self.index_by_ptr.pop(buf.data_ptr(), None)
                buf = torch.empty_like(like)
                self.by_index[index] = buf
                self.index_by_ptr[buf.data_ptr()] = index
            self.source_ptr_by_index[index] = source_ptr
            return buf

    def _private_seq_lens_buffers(self) -> "_SeqLensBuffers":
        buffers = getattr(self, "_seq_lens_buffer_cache", None)
        if buffers is None:
            buffers = self._seq_lens_buffer_cache = self._SeqLensBuffers()
        return buffers

    def _assert_sliding_layers_unaffected(self, attn_metadata, req_idx: int) -> None:
        """Verify, ONCE, that patching the gathered layers does not also
        mutate what the sliding-window layers read.

        The gather compacts the KV view, which is only legal for
        full-attention layers -- a sliding layer's kernel derives window
        membership from a key's index within `seqused_k`, so a compacted
        view silently misplaces its window (see
        `_gatherable_layer_names`). This runner therefore patches only the
        gathered layers' metadata.

        That is sound only if the two sets do not SHARE the underlying
        tensors. vLLM builds attention metadata per KV cache group, and
        several layers routinely reference one metadata object; if a
        `block_table` or `seq_lens` tensor is shared across the gathered and
        sliding sets, then "patching only the gathered layers" mutates both,
        and every sliding layer reads a compacted view it cannot interpret.

        The symptom would be exactly what was observed: correct output when
        the gather is a no-op (keep=100%), degenerate output at ANY lower
        keep rate regardless of how much is retained, and only mild damage in
        a small-context validator where the whole sequence fits inside one
        window anyway.

        Checked once per process rather than per step: tensor identity is
        fixed by how the metadata was built, so one comparison settles it,
        and a silent-corruption bug is worth one comparison.
        """
        if getattr(self, "_alias_checked", False):
            return
        self._alias_checked = True

        gatherable = self._gatherable_layer_names(attn_metadata)
        witnesses = [n for n in attn_metadata if n not in gatherable]
        if not witnesses or not gatherable:
            return

        witness = attn_metadata[witnesses[0]]
        before_blocks = self._get_field(witness, _BLOCK_TABLE_FIELD)[req_idx].clone()
        before_seq_len = int(self._get_field(witness, _SEQ_LENS_FIELD)[req_idx].item())

        sample = attn_metadata[next(iter(gatherable))]
        shares_block_table = (
            self._get_field(sample, _BLOCK_TABLE_FIELD).data_ptr()
            == self._get_field(witness, _BLOCK_TABLE_FIELD).data_ptr()
        )
        shares_seq_lens = (
            self._get_field(sample, _SEQ_LENS_FIELD).data_ptr()
            == self._get_field(witness, _SEQ_LENS_FIELD).data_ptr()
        )
        print(
            f"[SparseTarget] alias check: gathered vs sliding layers share "
            f"block_table={shares_block_table}, seq_lens={shares_seq_lens} "
            f"(witness={witnesses[0]!r}, sample={next(iter(gatherable))!r}); "
            f"witness seq_len={before_seq_len}",
            flush=True,
        )
        if shares_block_table or shares_seq_lens:
            raise RuntimeError(
                "the gathered (full-attention) layers and the sliding-window "
                "layers SHARE their attention-metadata tensors, so patching "
                "the former also compacts the view the latter read -- which "
                "misplaces every sliding layer's window and corrupts "
                "generation. The gather cannot be applied per-layer against "
                "this metadata layout; it needs per-group metadata objects, "
                "or a copy-on-write of the tensors it patches."
            )
        self._alias_witness = (witnesses[0], before_blocks, before_seq_len, req_idx)

    def _gatherable_group_block_size(self, attn_metadata) -> int:
        """Block size of the KV cache group the gather operates on, and the
        check that such a single group exists.

        **Why this is not `cache_config.block_size`.** That is the CONFIGURED
        value. For a model whose layers need different page sizes,
        `unify_kv_cache_spec_page_size` equalises them by MULTIPLYING the
        smaller-page layers' block_size, so a group's real block size can
        differ from the configured one. The gather is block-granular, so
        using the wrong number silently changes which tokens are selected.

        **Why one group is still required, but only across the GATHERED
        layers.** This runner writes one gathered block table into every
        layer it patches; that is meaningful only if those layers share a
        block table. Sliding-window layers are excluded from the gather
        anyway (see `_gatherable_layer_names` -- a compacted view misplaces
        their window), so they are free to live in their own group with their
        own, much smaller, windowed KV allocation. That is the whole point of
        letting the hybrid KV cache manager stay enabled: those layers stop
        being budgeted for the full context they can never read.

        Concretely, on Gemma-4-31B that was the difference between needing
        55.03 GiB of KV and having 34.05 GiB.

        Raises rather than guessing if the gathered layers straddle groups:
        picking one group's block ids and writing them into another's
        metadata reads unrelated physical memory, with nothing to notice it.
        """
        cached = getattr(self, "_gatherable_group_block_size_cache", None)
        if cached is not None:
            return cached

        gatherable = self._gatherable_layer_names(attn_metadata)
        group_of = {}
        for group in self.kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                group_of[layer_name] = group

        groups = {id(group_of[name]): group_of[name]
                  for name in gatherable if name in group_of}
        if len(groups) != 1:
            raise NotImplementedError(
                f"the {len(gatherable)} gathered (full-attention) layers span "
                f"{len(groups)} KV cache groups; the gather writes one block "
                f"table into all of them, which is only valid within a single "
                f"group. Construct the target engine with "
                f"`disable_hybrid_kv_cache_manager=True` to collapse them "
                f"(at the cost of budgeting every sliding layer for the full "
                f"context), or extend this runner to gather per group."
            )

        group = next(iter(groups.values()))
        block_size = group.kv_cache_spec.block_size
        configured = self.vllm_config.cache_config.block_size
        note = "" if block_size == configured else (
            f" -- NOTE: differs from the configured block_size={configured}, "
            f"so the effective gather granularity is {block_size} tokens, not "
            f"{configured}"
        )
        print(
            f"[SparseTarget] gathering {len(gatherable)} layer(s) in one KV "
            f"cache group at block_size={block_size}{note}",
            flush=True,
        )
        self._gatherable_group_block_size_cache = block_size
        return block_size

    def _apply_sparse_attention_overrides(self, attn_metadata: Dict[str, object]) -> None:
        if not attn_metadata:
            return

        num_reqs = self.input_batch.num_reqs
        block_size = self._gatherable_group_block_size(attn_metadata)
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor
        num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor

        any_patched = False
        # req_idx -> the gathered length written for it this step. Lets the
        # post-loop `max_seq_len` fix-up be computed entirely host-side; see
        # that block for the GPU sync this removes.
        patched_seq_lens: Dict[int, int] = {}
        for req_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[req_idx]
            # get_with_generation(), not get() -- one atomic snapshot of
            # (generation, positions, prefill_turn_start) under a single
            # lock, not three separate calls. See sparse_selection_
            # registry.py's "Generation counter" section: reading these
            # separately could race against a concurrent register() the
            # same way the id()-based cache this replaced already did, just
            # narrower.
            registered = sparse_selection_registry.get_with_generation(req_id)
            if registered is None:
                continue  # no active selection for this request -- leave untouched
            selection_generation, selected_positions, prefill_turn_start = registered

            num_computed = int(num_computed_tokens_cpu[req_idx])
            num_prompt = int(num_prompt_tokens_cpu[req_idx])
            # `num_computed < num_prompt` is this file's canonical
            # prefill/decode boundary (the same one every other probe in
            # this directory uses). This is the ONE place `num_computed` is
            # the right quantity to compare against -- everything below
            # needs `step_seq_len` instead, see the comment further down.
            is_prefill = num_computed < num_prompt
            if is_prefill and prefill_turn_start is None:
                # Decode-only scope: the default, and what every published
                # SPARSE-k*/ORACLE-k* row was measured under. This turn's
                # own new query tokens prefill at full, unrestricted
                # attention over the whole resident cache. A driver opts
                # into sparse prefill by passing `prefill_turn_start` to
                # `register_sparse_selection`; see that method and
                # `sparse_selection_registry.register`'s docstring for why
                # the scope rides on the registration rather than on a
                # process-wide mode flag.
                continue

            # The step's TOTAL sequence length -- NOT `num_computed`. See
            # module docstring's "Off-by-one" section: `num_computed_tokens
            # _cpu` is the count BEFORE this step's own scheduled token, so
            # for a decode step the real length is one more. This is read
            # from `optimistic_seq_lens_cpu` -- the exact CPU-side tensor
            # the stock builder itself derives `AttentionMetadata.seq_lens`
            # /`max_seq_len` from, already populated by `_prepare_inputs`
            # before this override runs -- so it needs no GPU sync (unlike
            # reading the built `seq_lens` GPU tensor back) and cannot
            # drift from what the kernel would otherwise have been told.
            step_seq_len = self._step_seq_len(req_idx)

            if is_prefill and not self._prefill_gather_applies(
                prefill_turn_start, num_computed, step_seq_len
            ):
                continue

            t_override_start = time.time()
            # Read once per request, not once inside _compute_gathered_view
            # AND again below for block_table_width -- same "an arbitrary
            # layer's metadata, assumed uniform across layers" reasoning as
            # before, just consolidated to one lookup.
            # Read from a layer the gather actually applies to, so the
            # "uniform across layers" assumption is scoped to the layers this
            # method touches rather than asserted over the whole model. Under
            # the single KV cache group this runner requires, every layer's
            # block table is the same anyway -- but scoping it keeps that a
            # consequence rather than a dependency.
            gatherable = self._gatherable_layer_names(attn_metadata)
            any_layer_metadata = attn_metadata[next(iter(gatherable))]
            full_block_table_row = self._get_field(any_layer_metadata, _BLOCK_TABLE_FIELD)[req_idx]

            if is_prefill:
                # Prefill and decode need genuinely different views, not the
                # same one with a different length plugged in -- see
                # `kv_cache_utils.PrefillGatherView`'s docstring for why
                # `BaseGatherView`'s `num_prompt`-centred stable/boundary
                # split is not merely suboptimal but WRONG here (it would
                # classify not-yet-written blocks as stable). The two share
                # `_get_base_block_indices`'s cache underneath, since the
                # selection's block-index set is the same set either way.
                prefill_view = self._get_prefill_base_view(
                    req_id, selected_positions, block_size, full_block_table_row,
                    prefill_turn_start, selection_generation,
                )
                gathered = self._compute_prefill_gathered_view(
                    base_view=prefill_view,
                    full_block_table_row=full_block_table_row,
                    block_size=block_size,
                    seq_len=step_seq_len,
                )
                # Same "record before the early-out" reasoning as the decode
                # accumulator below: a `None` gather is a DENSE chunk, not a
                # free one, and must be charged at its full length or the
                # measured prefill FLOPs understate what the GPU did.
                self._accumulate_prefill_step(
                    req_id,
                    step_seq_len - num_computed,
                    step_seq_len if gathered is None else gathered[1],
                )
                if gathered is None:
                    continue
                gathered_block_table_row, gathered_seq_len = gathered
                any_patched = True
                patched_seq_lens[req_idx] = gathered_seq_len
                self._apply_gathered_view(
                    attn_metadata, any_layer_metadata, req_idx,
                    gathered_block_table_row, gathered_seq_len,
                )
                self._accumulate_override_timing(req_id, time.time() - t_override_start)
                continue

            base_view = self._get_base_gather_view(
                req_id, selected_positions, block_size, full_block_table_row, num_prompt,
                selection_generation,
            )
            gathered = self._compute_gathered_view(
                base_view=base_view,
                full_block_table_row=full_block_table_row,
                block_size=block_size,
                num_prompt=num_prompt,
                seq_len=step_seq_len,
            )
            # Recorded BEFORE the `gathered is None` early-out below, not
            # after -- that branch is the degenerate "selection already
            # covers every resident block" case (keep_rate=1.0, or a
            # selection grown large enough to span everything), which is a
            # DENSE decode step, not a zero-work one. Skipping it would
            # make the k=1.0 row silently under-count its own attention
            # cost, i.e. look cheaper than the baseline it should exactly
            # match. `InstrumentedSparseTargetGPUModelRunner._compute_
            # gathered_view` (sparse_decode_microbench.py) already treats
            # this same branch as a full `ceil(seq_len / block_size)`
            # dense read for its block accounting -- same reasoning, same
            # fallback quantity, expressed in tokens instead of blocks.
            self._accumulate_attended_len(
                req_id, step_seq_len if gathered is None else gathered[1]
            )
            if gathered is None:
                continue
            gathered_block_table_row, gathered_seq_len = gathered
            any_patched = True
            patched_seq_lens[req_idx] = gathered_seq_len
            self._apply_gathered_view(
                attn_metadata, any_layer_metadata, req_idx,
                gathered_block_table_row, gathered_seq_len,
            )
            self._accumulate_override_timing(req_id, time.time() - t_override_start)

        if any_patched:
            # See module docstring's "Why .max_seq_len does [need patching]"
            # -- recomputed as the true post-gather max across the WHOLE
            # batch (not just the last-patched request's own
            # gathered_seq_len) so a batch mixing sparse-restricted and
            # untouched/dense requests still ends up with the correct
            # larger value. Gated on any_patched so this adds no extra
            # GPU-sync cost on steps/pipelines where nothing was actually
            # gathered (e.g. keep_rate=1.0's degenerate no-op path, or any
            # other worker_cls entirely).
            #
            # Computed via `.max().item()` ONCE here, from a single
            # representative layer's seq_lens -- NOT once per layer inside
            # the loop below. An earlier version called `.max().item()`
            # separately for each of Llama-3.1-8B's 32 layers, i.e. 32
            # GPU-sync points added to EVERY decode step on the sparse path
            # (and only the sparse path -- keep_rate=1.0 never hits this
            # block at all). Under enforce_eager=True with no CUDA-graph
            # batching to overlap/amortize a sync, that measurably made the
            # sparse path SLOWER than dense on real hardware, not just less
            # of an improvement -- confirmed by a real timing regression,
            # not theorized. Safe to read only one layer's seq_lens for
            # this, same "assumed uniform across all layers" reasoning this
            # function already relies on for `block_table_width` above.
            # Read the representative layer from the GATHERED set, and write
            # back only to that set. A sliding layer keeps its dense
            # block_table and seq_lens, so handing it a shrunken max_seq_len
            # would size its kernel's iteration to a span it no longer
            # matches -- the mirror image of the stale-max_seq_len bug that
            # made this fix-up necessary in the first place.
            gatherable = self._gatherable_layer_names(attn_metadata)
            any_layer_metadata = attn_metadata[next(iter(gatherable))]
            if hasattr(any_layer_metadata, _MAX_SEQ_LEN_FIELD):
                # Computed HOST-SIDE, with no GPU sync at all. Every input
                # is already known on the CPU: a patched request's length is
                # the `gathered_seq_len` this method just wrote, and an
                # unpatched one's is `_step_seq_len`, which reads
                # `optimistic_seq_lens_cpu` (the same CPU tensor the stock
                # builder derives `max_seq_len` from).
                #
                # The previous version called `.max().item()` on the GPU
                # tensor -- one blocking sync per decode step, already cut
                # down from 32. That was tolerable when eager dispatch cost
                # ~83 ms/tok; under FULL_DECODE_ONLY a decode step is ~42
                # ms/tok on this path, so a sync is a far larger share of
                # it. Measured: the override costs +17.7% of decode versus
                # the dense baseline, which is what this and the sibling
                # changes below are reclaiming.
                #
                # Equivalent to the old `.max()`, and arguably tighter: it
                # ranges over `num_reqs` rather than the whole tensor, so
                # padding entries beyond the batch cannot contribute.
                new_max_seq_len = max(
                    patched_seq_lens[i] if i in patched_seq_lens
                    else self._step_seq_len(i)
                    for i in range(num_reqs)
                )
                for layer_name in gatherable:
                    setattr(attn_metadata[layer_name], _MAX_SEQ_LEN_FIELD,
                            new_max_seq_len)

            # See _SCHEDULER_METADATA_FIELD's own comment above -- a real,
            # confirmed correctness bug (not a performance one, unlike
            # everything else patched in this method): FA3's ahead-of-time
            # work-scheduling tensor is built from the PRE-GATHER seq_lens/
            # max_seq_len and never recomputed, so the kernel was being
            # handed a schedule sized for the full, unrestricted context
            # regardless of how correctly block_table/seq_lens/max_seq_len
            # themselves were shrunk. No GPU sync needed to null this out
            # (plain attribute assignment), so it isn't gated on avoiding
            # sync cost the way the max_seq_len recompute above is --
            # always done whenever anything was actually patched.
            if hasattr(any_layer_metadata, _SCHEDULER_METADATA_FIELD):
                # Only the gathered layers' schedules are stale; a sliding
                # layer's was built for the dense view it still has.
                for layer_name in gatherable:
                    setattr(attn_metadata[layer_name],
                            _SCHEDULER_METADATA_FIELD, None)

    def _apply_gathered_view(
        self, attn_metadata, any_layer_metadata, req_idx: int,
        gathered_block_table_row, gathered_seq_len: int,
    ) -> None:
        """Write one request's gathered `(block_table_row, seq_len)` into
        every layer's metadata. Shared verbatim by the prefill and decode
        branches of `_apply_sparse_attention_overrides` -- the two differ
        entirely in HOW the view is computed and not at all in how it is
        installed, and factoring this out is what keeps that true (a second
        copy would be the obvious place for the two paths to drift on, say,
        the padding convention).

        Pads to the full block-table row width ONCE here, not inside the
        per-layer loop -- real hardware timing showed
        `_patch_layer_metadata`'s per-layer cost (previously 2 writes:
        gathered-row + zero-pad) adding up across every layer (32 for
        Llama-3.1-8B) on EVERY decode step, under `enforce_eager=True` (no
        CUDA-graph batching to amortize the per-op dispatch cost) -- a
        real, measured 0.5-1s/turn overhead vs. baseline, not theoretical.
        `block_table`'s column width is already assumed uniform across
        layers elsewhere, so building the padded row once and writing it
        whole into each layer's own tensor is safe under the same
        assumption, and cuts the hot per-layer loop from 2 GPU ops to 1.
        """
        block_table_width = self._get_field(any_layer_metadata, _BLOCK_TABLE_FIELD).shape[1]
        num_gathered = gathered_block_table_row.shape[0]
        if block_table_width > num_gathered:
            # Persistent scratch row, zeroed once on allocation and only
            # ever re-zeroed over the region a PREVIOUS step dirtied. The
            # old code allocated a fresh `torch.zeros(block_table_width)`
            # every decode step -- at the driver's default max_model_len
            # that is ~8192 int32s (32 KiB) allocated and fully zeroed per
            # step, to hold a few hundred meaningful entries.
            padded_block_table_row = self._padded_block_table_row(
                block_table_width, gathered_block_table_row, num_gathered)
        else:
            padded_block_table_row = gathered_block_table_row

        gatherable = self._gatherable_layer_names(attn_metadata)
        # MUST precede the write: the gathered layers share `seq_lens` with
        # the sliding ones, so writing without this corrupts every sliding
        # layer's window. See `_privatise_gathered_seq_lens`.
        self._privatise_gathered_seq_lens(attn_metadata, gatherable)
        self._assert_sliding_layers_unaffected(attn_metadata, req_idx)
        for layer_name, layer_metadata in attn_metadata.items():
            if layer_name not in gatherable:
                continue  # sliding-window layer: see _gatherable_layer_names
            self._patch_layer_metadata(
                layer_metadata, req_idx, padded_block_table_row, gathered_seq_len
            )

    def _padded_block_table_row(self, width: int, gathered_row, num_gathered: int):
        """A reusable full-width block-table row: gathered blocks in the
        leading columns, `NULL_BLOCK_ID` (0) after.

        Two costs removed versus rebuilding it each step. The allocation
        itself, and -- more importantly -- the zeroing: only the tail a
        PREVIOUS step actually dirtied needs clearing, not the whole row.
        The selection's block count is near-constant across the decode
        steps of one turn, so in the steady state that tail is empty and
        the zeroing collapses to nothing.

        Correctness rests on the same convention `_patch_layer_metadata`
        documents: entries past the gathered prefix must read as
        NULL_BLOCK_ID, matching this fork's own cudagraph padding for
        out-of-range block-table rows.
        """
        cached = getattr(self, "_padded_row_cache", None)
        if (cached is None or cached.shape[0] != width
                or cached.dtype != gathered_row.dtype
                or cached.device != gathered_row.device):
            cached = torch.zeros(width, dtype=gathered_row.dtype,
                                 device=gathered_row.device)
            self._padded_row_cache = cached
            self._padded_row_dirty = 0

        cached[:num_gathered] = gathered_row
        dirty = getattr(self, "_padded_row_dirty", width)
        if dirty > num_gathered:
            # Only the region the last step left non-zero.
            cached[num_gathered:dirty] = 0
        self._padded_row_dirty = num_gathered
        return cached

    @staticmethod
    def _prefill_gather_applies(
        prefill_turn_start: int, num_computed: int, seq_len: int
    ) -> bool:
        """Whether a prefill chunk may legally be gathered at all.

        Two bail-outs, both leaving the chunk fully dense (which is always
        a correct answer -- just a more expensive one):

        1. `num_computed < prefill_turn_start`: the engine is (re)computing
           tokens from BEFORE this turn began. That happens if a session
           resumption misses the prefix cache and earlier history has to be
           recomputed. Those tokens' KV is being written for the first
           time, and writing it under a restricted view would poison the
           persistent cache every later turn's selection reads from -- the
           same reason turn 0 must stay dense (see
           `kv_cache_utils.compute_prefill_gather_view`'s docstring). The
           tail-contiguity invariant would not hold for them either.
        2. `seq_len <= prefill_turn_start`: nothing of this turn is in the
           chunk yet, so there is no tail to force-keep and the causal
           alignment argument has nothing to stand on.

        Neither is expected in the steady state; both are cheap to check
        and silently wrong to skip.
        """
        if num_computed < prefill_turn_start:
            return False
        if seq_len <= prefill_turn_start:
            return False
        return True

    def _prefill_base_view_cache(self) -> Dict[str, tuple]:
        # Same lazy-init / generation-keyed-invalidation reasoning as
        # _base_gather_view_cache above, for the prefill counterpart. A
        # turn has far fewer prefill chunks than decode steps, so this
        # matters less than the decode caches do -- but the cached quantity
        # is O(len(selection)) to build (a filter + sort + tensor gather
        # over tens of thousands of positions at SCBench scale) and the
        # chunk count is not bounded by anything this module controls, so
        # rebuilding it per chunk would be the same mistake the decode path
        # already had to have fixed.
        if not hasattr(self, "_sparse_prefill_base_view_cache"):
            self._sparse_prefill_base_view_cache: Dict[str, tuple] = {}
        return self._sparse_prefill_base_view_cache

    def _get_prefill_base_view(
        self,
        req_id: str,
        selected_positions: List[int],
        block_size: int,
        full_block_table_row,
        prefill_turn_start: int,
        selection_generation: int,
    ):
        """Caches `kv_cache_utils.compute_prefill_base_view`'s result per
        request, invalidated by the same `selection_generation` signal the
        decode caches use (see `sparse_selection_registry.py`'s "Generation
        counter" section for why not `id()`).

        Safe to key on the generation alone, without also keying on
        `prefill_turn_start` or the block-table row's contents: the
        turn-start is registered in the SAME `register()` call the
        generation comes from, so it cannot change without the generation
        changing, and every block this caches is strictly below
        `first_tail_block` -- fully written before this turn started, and
        never rewritten by it.
        """
        from .kv_cache_utils import compute_prefill_base_view

        cache = self._prefill_base_view_cache()
        cached = cache.get(req_id)
        if cached is not None and cached[0] == selection_generation:
            return cached[1]
        base_block_indices = self._get_base_block_indices(
            req_id, selected_positions, block_size, selection_generation
        )
        prefill_view = compute_prefill_base_view(
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            base_block_indices=base_block_indices,
            turn_start=prefill_turn_start,
        )
        cache[req_id] = (selection_generation, prefill_view)
        return prefill_view

    def _compute_prefill_gathered_view(
        self,
        base_view,
        full_block_table_row,
        block_size: int,
        seq_len: int,
    ) -> Optional[tuple]:
        """Thin wrapper delegating to `kv_cache_utils.compute_prefill_
        gather_view` -- factored out for the same reason
        `_compute_gathered_view` is: so the block-selection arithmetic (and
        in particular the causal-mask invariant documented on that
        function) is unit-testable without a live `GPUModelRunner`."""
        from .kv_cache_utils import compute_prefill_gather_view

        return compute_prefill_gather_view(
            base_view=base_view,
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            seq_len=seq_len,
        )

    def _prefill_steps_accum(self) -> Dict[str, List[Tuple[int, int]]]:
        # Same lazily-initialized, req_id-keyed, pop-and-reset shape as
        # _attended_lens_accum above. Kept SEPARATE from it rather than
        # folded in: `pop_attended_lens` feeds `flops_model.target_decode_
        # flops`, which charges one linear+lm_head row per entry, and a
        # prefill chunk is neither one token nor one sampled row. Mixing
        # them would silently inflate the decode column by whatever the
        # prefill chunks happened to be.
        if not hasattr(self, "_sparse_prefill_steps_accum"):
            self._sparse_prefill_steps_accum: Dict[str, List[Tuple[int, int]]] = {}
        return self._sparse_prefill_steps_accum

    def _accumulate_prefill_step(
        self, req_id: str, num_query_tokens: int, attended_len: int
    ) -> None:
        self._prefill_steps_accum().setdefault(req_id, []).append(
            (int(num_query_tokens), int(attended_len))
        )

    def pop_prefill_steps(self, request_id: str) -> List[Tuple[int, int]]:
        """Returns this turn's `(num_query_tokens, attended_len)` per
        PREFILL chunk for `request_id` (one entry per chunk, in order),
        then resets it. The prefill counterpart of `pop_attended_lens`, and
        the FLOP driver for `flops_model.py::target_sparse_prefill_flops`.

        Empty unless the driver opted into sparse prefill by registering a
        `prefill_turn_start` -- under the default decode-only scope this
        method never sees a chunk, and the driver's own analytic
        `target_prefill_flops` remains the right model (nothing is
        restricted, so nothing needs measuring).

        `attended_len` is block-PADDED, exactly as `pop_attended_lens`
        documents for decode: it is what the kernel was told to read, not a
        token-exact ideal. `num_query_tokens` is the chunk's own scheduled
        token count (`seq_len - num_computed`), which is what makes the
        pair enough to reconstruct a chunk's causal key-visit count --
        `attn_prefill_flops(n_q, attended_len - n_q)`, valid because the
        gather's tail invariant puts this chunk's own tokens at the end of
        the gathered view (see `kv_cache_utils.compute_prefill_gather_
        view`'s "Why the tail must be contiguous").
        """
        return self._prefill_steps_accum().pop(request_id, [])

    def _override_timing_accum(self) -> Dict[str, tuple]:
        # Lazily-initialized, same reasoning as _base_block_indices_cache
        # above. Keyed by req_id -> (total_elapsed_seconds, num_steps) --
        # accumulated across every decode step of a turn, popped (and
        # reset) once by the driver via pop_override_timing below.
        if not hasattr(self, "_sparse_override_timing_accum"):
            self._sparse_override_timing_accum: Dict[str, tuple] = {}
        return self._sparse_override_timing_accum

    def _accumulate_override_timing(self, req_id: str, elapsed: float) -> None:
        accum = self._override_timing_accum()
        total, count = accum.get(req_id, (0.0, 0))
        accum[req_id] = (total + elapsed, count + 1)

    def pop_override_timing(self, request_id: str) -> Tuple[float, int]:
        """Returns `(total_seconds, num_decode_steps_patched)` accumulated
        for `request_id` since the last call, then resets it -- lets the
        driver directly confirm/measure the per-layer metadata-patch
        overhead (see this module's own "Per-layer caching" and
        `_apply_sparse_attention_overrides`'s comment above for why this
        was suspected as a real cost, not just theorized) instead of only
        inferring it from the overall turn-timing gap vs. baseline.

        **Measures CPU-side dispatch time, not confirmed GPU execution
        time** -- no `torch.cuda.synchronize()` is inserted around the
        timed region in `_apply_sparse_attention_overrides` (that would
        itself perturb the very cost being measured, forcing
        synchronization this pipeline wouldn't otherwise need at that
        point). Under `enforce_eager=True`, CPU-side dispatch overhead
        (the Python loop plus each op's kernel-launch cost) is exactly
        the thing hypothesized to dominate this mechanism's cost, so this
        is the right thing to measure for that question -- but treat the
        returned number as a lower bound on true wall-clock GPU cost, not
        an exact figure, since queued-but-not-yet-executed GPU work isn't
        captured here."""
        accum = self._override_timing_accum()
        return accum.pop(request_id, (0.0, 0))

    def _attended_lens_accum(self) -> Dict[str, List[int]]:
        # Same lazily-initialized, req_id-keyed, pop-and-reset shape as
        # _override_timing_accum above.
        if not hasattr(self, "_sparse_attended_lens_accum"):
            self._sparse_attended_lens_accum: Dict[str, List[int]] = {}
        return self._sparse_attended_lens_accum

    def _accumulate_attended_len(self, req_id: str, attended_len: int) -> None:
        self._attended_lens_accum().setdefault(req_id, []).append(int(attended_len))

    def pop_attended_lens(self, request_id: str) -> List[int]:
        """Returns this turn's per-decode-step attended KV length for
        `request_id` (one entry per step, in order), then resets it.

        The FLOP driver for `flops_model.py::target_decode_flops` -- the
        only quantity in that whole model that can't be derived from token
        counts the driver already holds, because it's decided per step
        inside `_compute_gathered_view` by how the selection happens to map
        onto block boundaries.

        **These are block-PADDED lengths, not token-exact ones** -- the
        gather works in whole KV blocks, so a selection of 3 tokens spread
        across 3 blocks costs `3 * block_size` attended positions, and this
        reports that larger number. That's deliberate: it's what the
        attention kernel was actually told to read, so the FLOPs derived
        from it are hardware-executed work rather than an idealized lower
        bound. It's also exactly the same quantity
        `sparse_decode_microbench.py`'s `pop_block_counts` reports (in
        blocks rather than tokens), which is what makes the two
        cross-checkable: `block_counts[i] == ceil(attended_lens[i] /
        block_size)` must hold step for step.

        Prefill steps are absent by construction -- `_apply_sparse_
        attention_overrides` branches on `num_computed < num_prompt` before
        reaching this accumulator, so this only ever sees genuine decode
        steps (the same prefill/decode boundary every other probe in this
        directory uses). Under the opt-in prefill scope those chunks are
        recorded by `pop_prefill_steps` instead, deliberately kept separate
        so `target_decode_flops` is never handed a multi-token chunk to
        charge as a single decode row.

        Costs one list append per decode step and no GPU sync -- negligible
        next to the per-layer metadata patching already in that loop, so
        it's always on rather than gated behind a flag.
        """
        return self._attended_lens_accum().pop(request_id, [])

    def _step_seq_len(self, req_idx: int) -> int:
        """This step's TOTAL sequence length for `req_idx` -- the same
        `num_computed_tokens + num_scheduled_tokens` the stock builder puts
        in `AttentionMetadata.seq_lens[req_idx]`, read from the CPU-side
        tensor it derives that from (`self.optimistic_seq_lens_cpu`,
        populated in `_prepare_inputs` well before this override runs, and
        used by the stock `_build_attention_metadata` itself for
        `max_seq_len`). Read from CPU deliberately: pulling the built
        `seq_lens` GPU tensor back would force a sync on every decode step
        of the sparse path, exactly the class of cost the `max_seq_len`
        fix-up already had to be restructured to avoid (see module
        docstring's Known risk area #5).

        Factored into its own method (rather than inlined) so a test can
        drive `_apply_sparse_attention_overrides` against a stub runner
        without having to fabricate the whole `optimistic_seq_lens_cpu`
        buffer, and so the "which length does the gather need?" question --
        the entire content of the off-by-one bug this fixes -- has exactly
        one answer in exactly one place.
        """
        try:
            return int(self.optimistic_seq_lens_cpu[req_idx])
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__} has no 'optimistic_seq_lens_cpu' -- "
                f"this module reads the step's true sequence length "
                f"(num_computed_tokens + num_scheduled_tokens) from it, the "
                f"same CPU tensor the stock _build_attention_metadata uses "
                f"for max_seq_len. If this vLLM version renamed it, update "
                f"_step_seq_len; do NOT fall back to "
                f"num_computed_tokens_cpu_tensor, which is one token short "
                f"for a decode step and was a real, confirmed correctness "
                f"bug (see this module's 'Off-by-one' docstring section)."
            ) from exc

    def _compute_gathered_view(
        self,
        base_view,
        full_block_table_row,
        block_size: int,
        num_prompt: int,
        seq_len: int,
    ) -> Optional[tuple]:
        """Thin wrapper delegating the actual block-selection/seq_len
        arithmetic to `kv_cache_utils.compute_sparse_gather_view_incremental`
        -- factored out specifically so that logic is unit-testable without
        a live `GPUModelRunner` or GPU (see that function's own docstring).
        `base_view` is the CACHED per-turn `BaseGatherView` from
        `_get_base_gather_view` above (already built from the block table
        and this turn's selection once) -- see that method's docstring for
        why this, not the raw selection or block-index set, is what should
        be cached and reused across a turn's decode steps.
        `full_block_table_row` is this request's real, already-built block
        table row (from an arbitrary layer's metadata -- assumed uniform
        across layers, see module docstring's Known risk area #4; read, not
        recomputed from `self.input_batch`, since `builder.build()`/
        `update_block_table()` may have applied backend-specific
        transformations this shouldn't bypass), read once by the caller and
        passed in here rather than re-read from `attn_metadata` again."""
        from .kv_cache_utils import compute_sparse_gather_view_incremental

        return compute_sparse_gather_view_incremental(
            base_view=base_view,
            full_block_table_row=full_block_table_row,
            block_size=block_size,
            num_prompt=num_prompt,
            seq_len=seq_len,
        )

    @staticmethod
    def _get_field(layer_metadata, field_name: str):
        if not hasattr(layer_metadata, field_name):
            raise AttributeError(
                f"{type(layer_metadata).__name__} has no {field_name!r} field -- "
                f"this module's field-name assumption (see its own module "
                f"docstring's Known risk area #1) does not hold for this "
                f"backend/vLLM version. Inspect {type(layer_metadata).__name__}'s "
                f"real fields (e.g. via dataclasses.fields(layer_metadata)) and "
                f"update _BLOCK_TABLE_FIELD/_SEQ_LENS_FIELD in this module."
            )
        return getattr(layer_metadata, field_name)

    def _patch_layer_metadata(
        self, layer_metadata, req_idx: int, padded_block_table_row, gathered_seq_len: int
    ) -> None:
        """`padded_block_table_row` is already the FULL row width (gathered
        blocks in the leading columns, zero-padded/NULL_BLOCK_ID after --
        matching this fork's own cudagraph-padding convention for unused
        block-table entries, see the real `_get_block_table`'s identical
        `fill_(NULL_BLOCK_ID)` for out-of-range rows) -- built ONCE by the
        caller (`_apply_sparse_attention_overrides`), not per layer, so
        this is a single write instead of two (see that call site's own
        comment for why this mattered on real hardware: 32 layers x 2 ops
        x every decode step, under `enforce_eager=True`, added up)."""
        block_table = self._get_field(layer_metadata, _BLOCK_TABLE_FIELD)
        seq_lens = self._get_field(layer_metadata, _SEQ_LENS_FIELD)

        block_table[req_idx, :] = padded_block_table_row
        seq_lens[req_idx] = gathered_seq_len


class SparseTargetWorker(Worker):
    """Wires `SparseTargetGPUModelRunner` in via the same `parallel_config.
    worker_cls` extension point the physically-pruned pipeline's
    `SpecPrefillWorker` uses -- see that module's docstring for the
    underlying mechanism. Also exposes the `collective_rpc`-callable methods
    the driver uses to register/discard a turn's sparse selection inside
    THIS process (not the driver's -- `EngineCore` always runs
    out-of-process, confirmed on real hardware for the physically-pruned
    pipeline's identical requirement, see `worker.py`'s docstring; the same
    fact applies here).

    Usage: pass this class's dotted path as `worker_cls` when constructing
    the TARGET's own `LLM(...)` for the sparse-attention experiments (never
    for the physically-pruned pipeline's own target, which keeps using
    `vllm_patch.worker.SpecPrefillWorker` unchanged):

        llm = LLM(
            model=target_model_path,
            worker_cls="vllm_patch.sparse_target_runner.SparseTargetWorker",
            enable_prefix_caching=True,
            enforce_eager=True,
            ...
        )
    """

    def init_device(self) -> None:
        super().init_device()
        self.model_runner = SparseTargetGPUModelRunner(self.vllm_config, self.device)

    def register_sparse_selection(
        self,
        request_id: str,
        selected_positions: List[int],
        prefill_turn_start: Optional[int] = None,
    ) -> None:
        """RPC-callable via `llm_engine.collective_rpc("register_sparse_selection",
        args=(request_id, selected_positions))` -- called by the driver
        BEFORE that turn's query update is submitted (see
        `sparse_selection_registry.py`'s module docstring for why this
        ordering, mirroring `pruner.py`'s identical "register before
        add_request" fix, is load-bearing here too).

        `prefill_turn_start` is optional and defaults to the original
        decode-only scope; pass this turn's own start position to
        additionally restrict its PREFILL. See
        `sparse_selection_registry.register`'s docstring for the full
        contract, and `kv_cache_utils.compute_prefill_gather_view`'s for
        what the restriction actually does."""
        sparse_selection_registry.register(
            request_id, selected_positions, prefill_turn_start
        )

    def discard_sparse_selection(self, request_id: str) -> None:
        sparse_selection_registry.discard(request_id)

    def pop_override_timing(self, request_id: str) -> Tuple[float, int]:
        """RPC-callable wrapper -- see `SparseTargetGPUModelRunner.
        pop_override_timing`'s docstring for what this measures and why."""
        return self.model_runner.pop_override_timing(request_id)

    def pop_attended_lens(self, request_id: str) -> List[int]:
        """RPC-callable wrapper -- see `SparseTargetGPUModelRunner.
        pop_attended_lens`'s docstring for what this measures and why."""
        return self.model_runner.pop_attended_lens(request_id)

    def pop_prefill_steps(self, request_id: str) -> List[Tuple[int, int]]:
        """RPC-callable wrapper -- see `SparseTargetGPUModelRunner.
        pop_prefill_steps`'s docstring for what this measures and why."""
        return self.model_runner.pop_prefill_steps(request_id)
