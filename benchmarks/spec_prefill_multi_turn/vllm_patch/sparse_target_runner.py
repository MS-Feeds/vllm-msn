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
participate in attention during DECODE steps only, per a selection the
speculator (unchanged, `proposer.py`/`speculator_worker.py`) recomputes
fresh every turn -- see `sparse_selection_registry.py`'s module docstring
for why no position-translation layer is needed between the speculator's
selection and this runner's own consumption of it (both track the same
absolute, gapless conversation-ledger numbering, unlike the OTHER
pipeline's physically-pruned, gap-containing prompts).

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
KEY/VALUE side (`block_table`/`seq_lens`) gets gathered.

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

    def _apply_sparse_attention_overrides(self, attn_metadata: Dict[str, object]) -> None:
        if not attn_metadata:
            return

        num_reqs = self.input_batch.num_reqs
        block_size = self.vllm_config.cache_config.block_size
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor
        num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor

        any_patched = False
        for req_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[req_idx]
            # get_with_generation(), not get() -- one atomic snapshot of
            # (generation, positions) under a single lock, not two separate
            # calls. See sparse_selection_registry.py's "Generation counter"
            # section: reading these separately could race against a
            # concurrent register() the same way the id()-based cache this
            # replaced already did, just narrower.
            registered = sparse_selection_registry.get_with_generation(req_id)
            if registered is None:
                continue  # no active selection for this request -- leave untouched
            selection_generation, selected_positions = registered

            num_computed = int(num_computed_tokens_cpu[req_idx])
            num_prompt = int(num_prompt_tokens_cpu[req_idx])
            if num_computed < num_prompt:
                # Still prefilling this turn's own new query tokens --
                # deliberately left at full, unrestricted attention (see
                # module docstring's "Force-keep" section and the approved
                # plan's decode-only scope decision). Only decode steps
                # (num_computed >= num_prompt) get gathered. This guard is
                # the ONE place `num_computed` is the right quantity to
                # compare against -- everything below needs `step_seq_len`
                # instead, see the next comment.
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

            t_override_start = time.time()
            # Read once per request, not once inside _compute_gathered_view
            # AND again below for block_table_width -- same "an arbitrary
            # layer's metadata, assumed uniform across layers" reasoning as
            # before, just consolidated to one lookup.
            any_layer_metadata = next(iter(attn_metadata.values()))
            full_block_table_row = self._get_field(any_layer_metadata, _BLOCK_TABLE_FIELD)[req_idx]

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

            # Pad to the full block-table row width ONCE here, not inside
            # the per-layer loop below -- real hardware timing showed
            # `_patch_layer_metadata`'s per-layer cost (previously 2 writes:
            # gathered-row + zero-pad) adding up across every layer (32 for
            # Llama-3.1-8B) on EVERY decode step, under enforce_eager=True
            # (no CUDA-graph batching to amortize the per-op dispatch cost)
            # -- a real, measured 0.5-1s/turn overhead vs. baseline, not
            # theoretical. block_table's column width is already assumed
            # uniform across layers elsewhere, so building the padded row
            # once here and writing it whole into each layer's own tensor
            # is safe under the same assumption, and cuts the hot per-layer
            # loop from 2 GPU ops to 1.
            block_table_width = self._get_field(any_layer_metadata, _BLOCK_TABLE_FIELD).shape[1]
            num_gathered = gathered_block_table_row.shape[0]
            if block_table_width > num_gathered:
                padded_block_table_row = torch.zeros(
                    block_table_width,
                    dtype=gathered_block_table_row.dtype,
                    device=gathered_block_table_row.device,
                )
                padded_block_table_row[:num_gathered] = gathered_block_table_row
            else:
                padded_block_table_row = gathered_block_table_row

            for layer_name, layer_metadata in attn_metadata.items():
                self._patch_layer_metadata(
                    layer_metadata, req_idx, padded_block_table_row, gathered_seq_len
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
            any_layer_metadata = next(iter(attn_metadata.values()))
            if hasattr(any_layer_metadata, _MAX_SEQ_LEN_FIELD):
                seq_lens = self._get_field(any_layer_metadata, _SEQ_LENS_FIELD)
                new_max_seq_len = int(seq_lens.max().item())
                for layer_metadata in attn_metadata.values():
                    setattr(layer_metadata, _MAX_SEQ_LEN_FIELD, new_max_seq_len)

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
                for layer_metadata in attn_metadata.values():
                    setattr(layer_metadata, _SCHEDULER_METADATA_FIELD, None)

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
        attention_overrides` `continue`s on `num_computed < num_prompt`
        before ever reaching the gather, so this only ever sees genuine
        decode steps (the same prefill/decode boundary every other probe in
        this directory uses).

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

    def register_sparse_selection(self, request_id: str, selected_positions: List[int]) -> None:
        """RPC-callable via `llm_engine.collective_rpc("register_sparse_selection",
        args=(request_id, selected_positions))` -- called by the driver
        BEFORE that turn's query update is submitted (see
        `sparse_selection_registry.py`'s module docstring for why this
        ordering, mirroring `pruner.py`'s identical "register before
        add_request" fix, is load-bearing here too)."""
        sparse_selection_registry.register(request_id, selected_positions)

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
