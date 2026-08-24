# Top-K KV Cache Selection for Multi-Turn Conversation — Experiment Plan

Status: **new pipeline, built on top of `../spec_prefill_llama/`'s
single-turn SpecPrefill port — code-complete but NOT run on real hardware**
(no GPU on the machine this was written on, same caveat as every
`spec_prefill*` pipeline in this repo). The multi-turn extension
(`vllm_patch/speculator_worker.py`'s persistent speculator engine,
`vllm_patch/conversation_state.py`'s absolute-position ledger) has no
single-turn precedent to lean on, so treat it as LESS validated than the
already-unvalidated single-turn baseline it's built on — see "Implementation
status" below and `REPRODUCE.md`'s validation steps.

| | |
|---|---|
| **Target model** | Llama-3.1-8B-Instruct (greedy decoding) |
| **Speculator** | Llama-3.2-1B-Instruct |
| **Precision** | BF16 |
| **Infra** | vLLM (this fork, V1 engine) |
| **Benchmark** | SCBench — `scbench_qa_eng` / `scbench_kv` / `scbench_summary` configs |
| **Hardware** | 2x A100 80GB (per the protocol; not yet empirically confirmed necessary) |
| **ETA** | TBD |

Reference the paper in this directory (`spec_prefill_paper.pdf`, same
SpecPrefill paper as `../spec_prefill_llama/`'s — this experiment extends
that algorithm, it isn't a different one) and the original research
protocol document this plan was built from (see chat history / plan file
`protocol-top-k-kv-dreamy-panda.md` if available).

---

## Motivation

SpecPrefill accelerates single-shot long-context prefill by using a small
speculator model to identify important tokens and prefilling only those into
the target model. This experiment asks whether the same idea generalizes to
**long, growing multi-turn conversations**, where context/prompt size is the
serving bottleneck across many sequential turns, not just once. Concretely:

1. Can a speculator that "keeps its own KV cache mirroring the whole
   [currently relevant] context" score a growing conversation cheaply --
   each turn costing roughly a short decode/prefill step, not a full
   re-scoring of the entire history?
2. Does re-pruning the SAME conversation's growing context, turn after
   turn, preserve accuracy for both the current turn's question AND
   unrelated future questions -- i.e., does the compression generalize, or
   does it overfit to whatever the most recent question happened to be?
3. Two settings for what survives across turns -- **KEEP** (rescored fresh
   from the full original history every turn) vs. **DISCARD** (the kept set
   only ever shrinks) -- which trades accuracy for speed better, and by how
   much? Per the protocol, KEEP is evaluated first.

---

## Algorithm reference

Same core algorithm as `../spec_prefill_llama/EXPERIMENT_PLAN.md`'s
"Algorithm reference" section (draft-scoring via lookahead decode →
softmax/pool/max/mean importance aggregation → chunked top-k selection →
sparse prefill with RoPE position restoration) -- not re-derived here. What's
NEW for multi-turn is entirely about WHEN and WITH WHAT CANDIDATE POOL that
algorithm runs, not the scoring math itself (`vllm_patch/scoring.py` is
reused verbatim, unmodified).

### Key architectural decisions

**1. Golden-context mode, plain-text turn rendering.** Each turn's target
generation is what gets graded, but the token sequence carried forward into
FUTURE turns' history is SCBench's own reference answer, not the model's own
output (matches SCBench's own official reference harness's default
`disable_golden_context=False` behavior). This makes every turn's token
sequence for every conversation knowable statically up front, enabling a
tractable, batchable driver instead of a fully serialized "wait for
generation, then build the next prompt" loop.

Turns are rendered as one flattened plain-text block (`Question 1: ...
Answer 1: ... Question 2: ...`, all inside a single chat-template user/
assistant exchange, no role tags between individual turns) rather than real
alternating per-turn chat messages -- confirmed (not assumed) to match
SCBench's own official reference harness's DEFAULT evaluation mode
(`use_chat_template=False` in `microsoft/MInference/scbench/eval_utils.py`'s
`create_multiturn_prompt`; their `follow_up_template` differs in exact
wording from ours but is the same structural shape), so this should be
comparable to published SCBench numbers, not an ad hoc simplification.
Real-run evidence (`predict_scbench.py` M000, 2026-08-11): turns 1-4 of a
5-turn `scbench_qa_eng` conversation produced direct, on-topic answers, but
turn 5 showed the model apparently misreading the accumulated Q&A history as
one compound question needing re-enumeration rather than a sequence of
already-closed exchanges -- plausibly connected to the lack of `<|eot_id|>`
turn-closing signals in this rendering (not yet confirmed systematic across
more conversations/turn positions).

**Two settings deferred to a later pass, not built in this one:**

- **Chat-template rendering** (SCBench's OWN alternative,
  `use_chat_template=True`: each turn as a real `{"role": "user"/
  "assistant", ...}` message) -- the more benchmark-sanctioned response to
  the turn-5 confusion above, if it proves systematic, rather than a
  from-scratch design. Would require real per-turn token-boundary tracking
  in `conversation_state.py` (chat-template role/`<|eot_id|>` wrapper tokens
  between turns would need to be tracked and excluded from
  scoring/pruning, not just the single constant wrapper this pass tracks) --
  a real, scoped implementation task, not a flag flip.
- **Self-generated history** (conversation history built from the model's
  own prior-turn completions instead of SCBench's golden reference answers,
  `disable_golden_context=True` in SCBench's own harness) -- would remove
  the "every turn's tokens known up front" property this pass's tractable,
  batchable driver design depends on (see above), forcing a fully
  serialized per-conversation execution model instead. Also a real
  redesign, not a flag flip.

  **Since partially realized**: `M000` (`run_baseline`) now uses
  self-generated history -- its own actual per-turn completions
  (`completion.token_ids`, used verbatim, INCLUDING whatever special/EOS
  token the model generated -- `skip_special_tokens` only ever affects
  `.text` decoding, never `.token_ids`) instead of golden answers. Safe to
  do without the "fully serialized" cost the bullet above warns about,
  because `run_baseline` was ALREADY fully serialized per-conversation (one
  one-shot `add_request` per turn, no cross-turn batching attempted, see
  decision #1's docstring in `predict_scbench.py`) -- there was no
  "tokens known up front" property to lose in the first place, unlike
  `M-k*-g*`/oracle (`run_specprefill`), which genuinely still need it
  (DISCARD mode's candidate-pool bookkeeping and the pre-flight
  speculator-budget check both depend on knowing turn content ahead of
  driving it), so those keep golden-context unchanged. Motivation: this
  matches `SPARSE-k*-g*` (`run_sparse_attention`, which was FORCED into
  self-generated history by its own resumable-session mechanism -- see
  that pipeline's own section below) on this axis, removing a real
  confound when comparing the two -- previously `M000`'s golden answers
  (typically short reference text) were much shorter than `SPARSE`'s own
  `max_tokens`-bounded completions, making `SPARSE`'s conversations grow
  faster turn over turn for a reason unrelated to either architecture.
  `M-k*-g*` was NOT changed, so it still has this asymmetry against
  `SPARSE` -- worth keeping in mind if comparing `M-k*-g*` vs. `SPARSE-k*-
  g*` context-length-sensitive metrics (candidate-pool size, scoring time)
  turn over turn.

**2. Speculator persistence: a real, persistent `vllm.LLM()`, not a
hand-extended scratch cache.** `vllm_patch/speculator_worker.py` runs the
speculator as a genuine `vllm.LLM(enable_prefix_caching=True)`, wired via the
same `worker_cls` + `collective_rpc` extension points
`../spec_prefill_llama/vllm_patch/worker.py` already proved out for the
TARGET model, applied here to the SPECULATOR for the first time. Each turn's
request is submitted under a **stable per-conversation `cache_salt`**, so
vLLM's own prefix-cache matching gives "only prefill the new tokens" for
free -- `RequestOutput.num_cached_tokens` is logged per turn
(`predict_scbench.py`'s CSV output) as the real, measured version of this
claim, not an assumption. The `look_ahead_cnt` lookahead-decode steps used
for scoring are captured via query-capture hooks installed once inside the
Worker process (`speculator_worker.py`'s `SpeculatorGPUModelRunner`) --
capture is switched on only AFTER the bootstrap prefill's own first token is
observed, so (mirroring the single-turn pipeline's "discard the bootstrap's
own capture" rule) the prefill itself is never scored.

**Real, measured tradeoff, not assumed uniform across modes**: KEEP mode's
per-turn speculator prompt is a literal, monotonic extension of the previous
turn's (new content = previous turn's golden answer + this turn's query),
so prefix-cache hits are expected to be large. **DISCARD mode's candidate
pool is a COMPACTED (gap-removed) subsequence of history**, which is
generally NOT a literal continuation of anything submitted before -- so
DISCARD is expected to get little or no prefix-cache benefit on the
speculator side, recomputing its (smaller) candidate pool from scratch each
turn. This is an inherent consequence of what DISCARD mode IS (compaction
removes literal-content continuity), not an implementation gap -- DISCARD's
efficiency case rests on having fewer tokens to consider each turn, not on
cache reuse. `num_cached_tokens` logged per turn is where this becomes a
measured fact rather than a guess.

**3. KEEP vs. DISCARD candidate pools** (`vllm_patch/conversation_state.py`):

- **KEEP**: candidate pool at turn N = the FULL original conversation
  history (context + every prior turn's query + golden answer), rescored
  fresh every turn. A token dropped at turn 2 can be picked back up at turn
  5.
- **DISCARD**: candidate pool at turn N = turn N-1's own kept subset + turn
  N-1's own query (now ordinary, no longer force-kept) -- monotonically
  shrinking, never regrows. Turn N-1's golden answer is deliberately NOT
  retained in this pool (a documented modeling choice, not an oversight --
  see `conversation_state.py`'s docstring for the reasoning and how to
  revisit it if DISCARD's accuracy degrades faster than expected).

  **Not yet realized in code**: because DISCARD's kept-token sequence is a
  genuine monotonic extension turn-to-turn (unlike KEEP's, which can
  legitimately reselect a different subset each turn), its TARGET-side
  prompt is, in principle, eligible for the same kind of cross-turn
  prefix-cache reuse the speculator already gets (decision #2 above). The
  current implementation does not realize this: `vllm_patch/pruner.py`'s
  `prune_and_add_turn` sets `cache_salt=request_id`, unconditionally unique
  PER TURN regardless of `keep_mode` -- correct and necessary for KEEP
  (a stale partial cache match across turns could silently serve WRONG
  content, since the kept subset can genuinely differ turn to turn), but
  leaves DISCARD's own theoretical reuse benefit unrealized too, since the
  same per-turn-unique salt applies to both modes today. Not a blocker for
  the current KEEP-only MVP sweep, but a real, scoped fix needed (make
  `cache_salt` mode-aware: `conversation_salt`-based for `discard`,
  per-turn for `keep`) before DISCARD mode is actually implemented and run
  -- flagged here rather than silently assumed to already work.

**RoPE positions are absolute-conversation-relative, not turn-local**:
`vllm_patch/pruning_registry.py`'s `PruneRecord.kept_positions` (consumed,
unmodified, by `model_runner.py`/`worker.py`) must be each kept token's
position in the ENTIRE conversation so far, not an index into the current
turn's own already-pruned prompt -- this is the one place a turn-2-onward
multi-turn request differs from a single-turn one, and it's handled entirely
upstream, in `conversation_state.py`'s append-only absolute-position ledger
and `pruner.py`'s use of it -- `pruning_registry.py`/`model_runner.py`/
`worker.py` themselves are copied verbatim, unchanged.

**4. Force-keep.** Each turn's own new query is never subject to pruning --
scored for signal (alongside the candidate pool) but always force-included
in the final prompt regardless of what the scorer decided, mirroring the
single-turn pipeline's confirmed-on-hardware fix for its own
question/instruction suffix (`../spec_prefill_llama/predict_longbench_v2.py`'s
documented gibberish-output bug at aggressive keep rates).

**5. Oracle upper bound — WIRED UP, by a different route than planned.**
Still the same idea: score using the TARGET model's own attention instead of
the speculator's estimate of it, so the `ORACLE-k{N}` vs. `SPARSE-k{N}-g32`
gap measures estimator quality and the `ORACLE-k{N}` vs. `M000` gap measures
what the sparse-attention mechanism costs with the estimator held perfect.
Those two gaps are the entire point of the row: the graded SPARSE sweep shows
a large `scbench_kv` degradation, and nothing in it says whether the 1B
speculator is picking the wrong blocks or whether block-granular sparse
decode is lossy no matter who picks.

What changed is HOW. The original plan was a teacher-forced forward pass over
the next turn's golden query, driven through the target's own engine with a
query-capture hook on its attention layers (`vllm_patch/pruner.py`'s
`compute_oracle_kept_pairs`, still present, still unused). Two problems with
that, both structural rather than incidental:

1. The target's engine now holds a **persistent, resumable conversation
   session** (see the sparse-attention architecture section below) whose KV
   cache IS the conversation. A teacher-forced scoring pass through it would
   write its own tokens into that session, corrupting the conversation it is
   supposed to be scoring, and there is no unwind mechanism.
2. It needs a new capture hook on the target runner -- new, unvalidated
   plumbing, on the one code path this pipeline's correctness rests on.

The wired-up version avoids both by keeping the scorer in a **separate
engine**: `predict_scbench.py` runs the ordinary sparse driving loop with
`SpecPrefillProposer` constructed over the TARGET checkpoint
(`--oracle-scorer-model`, defaulting to `--target-model`) instead of the 1B
speculator, on the same GPU slot the speculator would have used
(`--oracle-scorer-device`, defaulting to `--speculator-device`). No new
mechanism at all: the scoring math, the capture hook, the driving loop, and
the block-gather are all the already-validated speculator-path code, with one
constructor argument different. `run_sparse_attention` serves both rows and
branches on nothing.

The cost of the deviation, stated plainly: the oracle's lookahead queries come
from tokens the scorer generated itself (the target's own dense-context
continuation), not from the dataset's golden answer. So this is the ceiling
for *"what if the draft model were as good as the target?"* -- SpecPrefill's
own self-speculation limit -- not for *"what if we knew the right answer?"*.
The latter is strictly stronger and would still need `compute_oracle_kept_
pairs` plus teacher-forced Q from the golden answer's positions; it remains
available as a follow-up, and `compute_oracle_kept_pairs` is kept for it.

Two further consequences worth knowing before reading the numbers:

- **ORACLE-k\* moves into the SPARSE rendering group.** It drives
  `run_sparse_attention`, so it uses genuine `<|eot_id|>`-delimited chat turns
  like `M000`/`SPARSE-k*-g*`, not `run_specprefill`'s flattened
  single-exchange rendering. This is required, not incidental: a ceiling
  measured under a different prompt rendering than the rows it bounds is not a
  ceiling.
- **Scoring cost rises ~8x per turn** (8B scorer instead of 1B). Prefix caching
  keeps that to turn 0 of each conversation for KEEP mode, but an oracle row
  is still meaningfully slower than its SPARSE partner. This is why the row
  pairs with one representative granularity (32) rather than the full cross.

**6. KV entry granularity.** Maps directly onto the existing `chunk_size`
parameter (`token` = flat/non-chunked selection, `16`/`32`/`64` = the
corresponding `chunk_size`) -- no new mechanism needed.

**7. Baseline methods (H2O, StreamingLLM, Quest, KVzip, HeadKV) — explicitly
out of scope for this pass**, confirmed with the user. The natural seam for
adding them later is `pruner.py`'s `(pruned_token_ids, kept_positions)`
return contract -- a future baseline selector is a sibling function with the
same return shape, not a new abstraction layer (no `BaseSelector` interface
built now, on the theory that a real second implementation should inform its
shape, not guesswork against zero real implementations).

---

## Implementation status

Built directly on `../spec_prefill_llama/vllm_patch/` (Llama-3.1-8B target /
Llama-3.2-1B speculator, the exact pair this protocol calls for):

1. **Carried over unchanged**: `scoring.py`, `kv_cache_utils.py`,
   `prefill_split.py`, `pruning_registry.py` (docstrings updated to describe
   multi-turn absolute-position semantics; zero logic changes),
   `model_runner.py`, `worker.py` (docstrings updated; zero logic changes --
   the RoPE-position-override mechanism is correct as designed once
   `PruneRecord`'s fields carry absolute conversation positions, a fix that
   lives entirely upstream in `pruner.py`/`conversation_state.py`).
2. **New**: `conversation_state.py` (absolute-position ledger + KEEP/DISCARD
   candidate-pool construction -- pure Python, no vLLM dependency, fully
   unit-tested, see `test_vllm_patch.py`), `speculator_worker.py` (persistent
   speculator engine via `worker_cls`, query capture + KV read-back via
   `collective_rpc` -- the least-precedented, least-validated new piece of
   this whole port, see its own "Known risk areas" docstring section).
3. **Rewritten**: `proposer.py` (driver-facing orchestration over the
   persistent speculator engine, replacing the single-turn pipeline's
   standalone-model + hand-rolled-lookahead-loop design), `pruner.py`
   (conversation-aware: threads `conversation_state.py` through, plus a
   shared scoring core for the oracle path), `config.py` (+1 field,
   `keep_mode`).
4. **Oracle path: wired up** (`--exp oracle`), via a target-checkpoint
   scorer engine rather than the originally-planned teacher-forced capture
   hook -- see decision 5 above for the full reasoning and for what the
   resulting ceiling does and does not bound.
   `vllm_patch/pruner.py::compute_oracle_kept_pairs` is still present and
   still unused, kept as the entry point for the stronger golden-answer
   teacher-forced variant.
5. **Multi-conversation batching -- not attempted.** Both the speculator
   (`proposer.py`) and the target driver (`predict_scbench.py`) run ONE
   conversation's ONE in-flight request at a time, by deliberate MVP-scope
   choice (see each module's own docstring) -- sacrifices throughput
   (no cross-conversation batching) for a much simpler, easier-to-validate
   sequential driving loop. A natural follow-up once correctness is
   confirmed on real hardware.
6. **A second, architecturally distinct pipeline** (persistent target-side
   KV cache + speculator-guided sparse attention, `SPARSE-k*-g*`) is built
   and wired up alongside the physical-pruning pipeline above -- see its
   own "Persistent KV cache + sparse attention (SPARSE-k*-g*)" section
   below for the full design, validation status, and how it differs.

---

## Persistent KV cache + sparse attention (SPARSE-k*-g*)

**A second, architecturally distinct pipeline, run alongside M-k*-g* as a
comparison baseline, not a replacement for it.** Everything above this
section ("Algorithm reference" through "Implementation status") describes
**physical pruning**: each turn, the target receives a genuinely SHORTER
prompt (`pruned_token_ids`), faithful to the original SpecPrefill paper's
own mechanism. That compresses the target's KV cache SIZE.

This section's architecture instead compresses ATTENTION COMPUTE: **the
target retains the full conversation's KV cache persistently across every
turn — nothing is ever discarded — and the speculator's job is to tell the
target which subset of that already-resident cache to actually attend to
when decoding each turn's response.** Confirmed as the user's real intent
partway through the M-k*-g* build (the physical-pruning pipeline was built
first, matching the paper literally, before this distinction surfaced); the
full research trail (vLLM V1 scheduler/attention internals, DeepSeek sparse
MLA precedent ruled out as non-portable to GQA, block-table gathering
adopted instead) lives in the approved plan this section summarizes.

### Mechanism

1. **Target-side persistence**: this fork's native "resumable session"
   request lifecycle (`Request.resumable`, `RequestStatus.
   WAITING_FOR_STREAMING_REQ`, `Scheduler._update_request_as_session`) --
   one target request per CONVERSATION (not per turn), with each turn
   delivered as a session update against the same `request_id`. KV blocks
   stay `ref_cnt`-pinned for the whole conversation, never torn down
   between turns. Verified on real hardware via TTFT comparison
   (`validate_resumable_session.py` -- turn 2's TTFT was dramatically lower
   than turn 1's, confirming genuine reuse, not recomputation;
   `num_cached_tokens` was checked and found NOT meaningful for session
   continuations, a scheduler-internal gating detail, not a bug).
2. **Sparse attention via block-table gathering**: the target runner
   (`vllm_patch/sparse_target_runner.py`) overrides
   `_build_attention_metadata()` to substitute a gathered
   `(block_table, seq_lens)` view -- reusing the STOCK, unmodified
   attention kernel underneath, the same technique
   `make_local_attention_virtual_batches()` already uses for chunked-local
   attention. RoPE needs no adjustment either way (it is baked into K at
   write time, so gathered vectors carry correct rotation regardless of
   where they land in the shrunk view). Verified on real hardware via a
   needle-in-a-haystack behavioral test (`validate_sparse_attention.py`,
   all 3 steps passed: full-attention baseline finds the needle, excluding
   its block loses it, re-including a different partial selection recovers
   it).

   **Scope was originally decode-only, and is now a per-run choice.** The
   first pass restricted decode steps only -- each turn's own new query
   tokens prefilled with full, unrestricted attention over the whole
   resident cache -- because a single-token decode step needs nothing said
   about the ORDER of the compacted view, while a multi-token prefill
   chunk does: FlashAttention aligns its causal mask bottom-right, so the
   chunk's own tokens must be the final entries of the gathered view or
   every query in it silently reads the wrong keys. `--sparse-prefill`
   opts into the prefill scope, satisfying that invariant by force-keeping
   a CONTIGUOUS tail from the turn's start position onward (see
   `vllm_patch/kv_cache_utils.py::compute_prefill_gather_view`). Turn 0
   stays dense under both scopes -- restricting the prefill that first
   computes the context's KV would poison the persistent cache every later
   turn's selection reads from. Decode-only remains the DEFAULT: every
   published `SPARSE-k*`/`ORACLE-k*` row was measured under it, and the
   two scopes measure structurally different things (under
   `--sparse-prefill` the prefill stage joins the shrinking side, so
   whether the decode saving alone can pay for the speculator is no longer
   the question the row answers).
3. **Block-level granularity only** (`16`/`32`/`64`) -- `token`-level
   sparse selection would need a genuine custom kernel (DeepSeek's sparse
   MLA path is architecturally similar but MQA/MLA-specific, not portable
   to Llama's GQA); out of scope for this pass, `SPARSE-k*-gtoken` rows do
   not exist.
4. **No RoPE position-override machinery needed** (unlike the
   physical-pruning pipeline) -- gathered K already carries correct
   rotation, and the query's own position is just its true, ever-
   incrementing position in a continuously-open session.
   `pruning_registry.py`/`model_runner.py`/`worker.py` are untouched by
   this pipeline; it uses a separate, simpler registry
   (`vllm_patch/sparse_selection_registry.py`) instead.

### Self-generated history (a real decision, not an oversight)

The physical-pruning pipeline above runs in **golden-context mode**: future
turns' history is the dataset's reference answer, not the model's own
output, which is what makes every turn's tokens knowable statically up
front (see "Structurally different..." point 2 in `predict_scbench.py`'s
own module docstring). This pipeline cannot do that: the resumable-session
mechanism's own scheduler logic auto-appends the target's ACTUAL generated
output tokens to its KV cache as part of normal decoding -- there is no
hook to substitute a different, golden continuation into an already-
persistent cache without contradicting the "persistent, never discarded"
premise this whole architecture exists to test. Fighting that native
behavior (e.g. forcibly overwriting cache contents to match golden text)
would be substantially more complex than embracing it, so
`predict_scbench.py::run_sparse_attention` feeds the target's own actual
generated output into `conversation_state.py`'s ledger
(`state.complete_turn(result.kept_history_pairs, actual_output_ids)`, not
the golden answer) -- both for the speculator's own candidate-pool
construction (via the ledger) and to keep `LedgerToTargetPositionMap`'s
position translation in lockstep with what the target session actually
contains. Golden answers are still used for grading (`grade_scbench.py`
reads the dataset directly, independent of what was fed forward during
generation); the metric comparison against M-k*-g*'s golden-context numbers
should account for this difference (self-generated history is a harder,
more realistic setting -- a lower score here isn't necessarily attention-
restriction-induced degradation).

A `LedgerToTargetPositionMap` (per conversation) translates
`conversation_state.py`'s pure-content absolute positions into the target
session's real, wrapper-token-inclusive stream positions before every
`register_sparse_selection` call -- needed because this pipeline (unlike
M-k*-g*'s single flattened text block) uses genuine per-turn chat-template
boundaries (`chat_turn_boundary_pieces`), which insert real wrapper tokens
between turns that the content-only ledger doesn't know about.

### KEEP mode only, and for a different reason than M-k*-g*'s

Nothing is ever physically discarded from the target's cache under this
architecture, so "re-selecting" a token dropped from attention at turn 2
costs nothing extra at turn 5 (it was never evicted) -- DISCARD mode's
whole reason for existing (reconstructing genuinely-lost data) doesn't
apply here. `SPARSE-k*-g*` rows always use `keep_mode="keep"`.

### Files

New: `vllm_patch/sparse_target_runner.py` (`SparseTargetGPUModelRunner`/
`SparseTargetWorker`), `vllm_patch/sparse_selection_registry.py`,
`vllm_patch/kv_cache_utils.py::compute_sparse_gather_view` (pure-function,
CPU-unit-tested), `predict_scbench.py::run_sparse_attention`/
`LedgerToTargetPositionMap`/`chat_turn_boundary_pieces`/
`build_sparse_session_request`/`drive_one_turn_of_session`,
`validate_resumable_session.py`, `validate_sparse_attention.py`. Reused
unchanged: `proposer.py`/`speculator_worker.py` (the speculator's own job
doesn't change at all under this design), `conversation_state.py`,
`pruner.py::compute_pruned_turn` (its `kept_positions` field is exactly the
sparse selection needed; `pruned_token_ids` goes unused by this path, since
the target always receives the FULL, unpruned content).

### Status

Both validation scripts (`validate_resumable_session.py`,
`validate_sparse_attention.py`) pass on real hardware; the driving loop
(`run_sparse_attention`, `run_experiment`'s `mode == "sparse"` dispatch) is
wired up and CPU-tested (`test_vllm_patch.py`).

**A real first-smoke-test finding, since fixed**: `RequestOutput.
outputs[0].token_ids`/`.text` are CUMULATIVE for a request's WHOLE
lifetime under this engine's default (non-`DELTA`) `output_kind` --
confirmed by reading `output_processor.py` (`RequestState.
_new_completion_output`: `token_ids = self.detokenizer.output_token_ids`
when not delta) and `detokenizer.py` (`self.output_text += ...`) -- and
NEITHER is reset by `apply_streaming_update`/`_update_request_as_session`
on a session resumption. `run_sparse_attention`'s first version fed this
raw cumulative output straight into `state.complete_turn`, re-appending
every prior turn's output to the ledger each turn -- a compounding drift
that crashed `sparse_target_runner.py`'s block-index bounds check a few
turns into a real SCBench conversation (a registered position translated
to a target position far beyond anything actually computed) and would
independently have corrupted every recorded prediction from turn 2 onward
(each `pred` field would have contained all previous turns' text
prepended). Fixed by tracking each turn's cumulative-output-length-so-far
and slicing out just the new tokens/text per turn. A second, smaller
related fix: `_update_request_as_session` itself discards the final
sampled token of each turn when a session resumes ("its own KV was never
computed, since it was never fed back into the model before the request
stopped") -- `state.complete_turn`'s ledger-feeding slice now drops that
same last token too (predictions/grading still use the FULL per-turn
slice, only the ledger is truncated), or `LedgerToTargetPositionMap` would
drift by one token per turn against what the target's session actually
retains.

**Second real finding, from the fixed pipeline's first successful smoke
test (4 processed conversations)**: SPARSE ran at 695s/conversation vs.
M000 baseline's 28s/conversation -- a 25x gap far larger than "pays for an
extra speculator forward pass" alone should explain. Root-caused to
`compute_sparse_gather_view` (`kv_cache_utils.py`) recomputing
`{p // block_size for p in selected_positions}` -- an
O(len(selected_positions)) Python set comprehension, and
`selected_positions` can be tens of thousands of absolute ledger positions
for a large SCBench context under KEEP mode -- on EVERY decode step
(`_apply_sparse_attention_overrides` runs once per `_build_attention_
metadata` call, i.e. once per decode step, not once per turn), even though
the registered selection is constant for a whole turn's ~`max_tokens`
(e.g. 64) decode steps. Fixed by caching the block-index set once per turn
(`sparse_target_runner.py`'s new `_get_base_block_indices`, invalidated by
`id()` of the registry's `selected_positions` object changing) -- see that
module's own "Per-turn caching" docstring section. **Note this doesn't
necessarily close the whole 25x gap**: M000 pays zero speculator cost at
all (no proposer, no K read-back, no scoring), while SPARSE and M-k*-g*
both pay for a full speculator forward pass plus `proposer.py`'s
cross-process K-tensor transfer (`tensor_to_wire`'s `.tolist()`, expensive
for a large candidate pool, shared unchanged by both pipelines) -- so
comparing SPARSE against M-k*-g*'s own `seconds_per_conversation` (not
M000's) is the fair way to isolate whether further sparse-specific
overhead remains after this fix.

**Third real finding, from stage-level timing logs added to `proposer.py`/
`predict_scbench.py`**: `proposer.retrieve_keys` alone took 89s for one
call (87,972 positions x 16 layers, 720M elements) -- confirming the
`tensor_to_wire`/`tensor_from_wire` suspicion above as real, not
theoretical, and quantifying it as the likely dominant cost (this call
happens once per turn, growing with the KEEP-mode candidate pool). Root
cause: `tensor_to_wire`'s `.tolist()`, msgpack's own list serialization,
and `tensor_from_wire`'s `torch.tensor(list)` reconstruction are three
separate non-vectorized, element-by-element Python passes over the full
result -- unavoidable under the DEFAULT `collective_rpc` serialization
mode, which (traced through `vllm/v1/serial_utils.py`) doesn't carry
enough type info to reconstruct a `torch.Tensor` from a `UtilityResult`
without `VLLM_ALLOW_INSECURE_SERIALIZATION=1`. Fixed by setting that flag
in `proposer.py` (before either engine is constructed) and having
`speculator_worker.py`'s `end_capture`/`retrieve_keys` return raw
`torch.Tensor`s directly -- msgspec's own dedicated `_encode_tensor`/
`_decode_tensor` codec then moves the underlying buffer directly, with no
Python-list conversion at any point. **Judged safe for this pipeline**
specifically because the flag's real risk (a `pickle.loads` fallback for
otherwise-unserializable types, a genuine RCE vector if triggered by
attacker-controlled data) requires a trust boundary this pipeline doesn't
have: `collective_rpc` here only ever crosses between a locally-spawned
`EngineCore` subprocess and its own parent driver process, never a
network or multi-tenant boundary -- see `proposer.py`'s own docstring for
the full reasoning. `tensor_to_wire`/`tensor_from_wire` are kept in
`kv_cache_utils.py` (unused now, but still CPU-tested) as a documented
fallback, since **this specific change has not been validated on real
hardware** -- re-run `validate_proposer.py` (its Step C already exercises
`retrieve_keys` directly) before trusting it.

**Fourth change, superseding the third's zero-copy fix with a better one**:
rather than moving the K tensor across `collective_rpc`'s process boundary
more efficiently, avoid moving it at all. Q and K both already live inside
the speculator's own worker process (Q captured during that turn's forward
passes, K resident in its own KV cache) -- only the FINAL SELECTION
DECISION (a list of at most a few tens of thousands of small ints, not
720M floats) ever needs to leave it. Implemented as:
- `scoring.py`: new `score_and_select_indices(query_buffer, key_buffer_
  per_layer, actual_look_ahead_cnt, spec_config) -> List[int]`, the shared
  3-call scoring pipeline (`compute_attention_score` ->
  `aggregate_attention_score` -> `chunk_select_from_smoothed_attention`)
  factored out so both the oracle path (still driver-side) and the new
  in-process speculator path can call it without duplicating logic.
- `speculator_worker.py`: new `SpeculatorGPUModelRunner.end_capture_and_
  score` / `SpeculatorWorker.end_capture_and_score` -- combines
  `end_capture` + `retrieve_keys` + scoring into ONE in-process call,
  returning just `(kept_local_indices_or_None, actual_look_ahead_cnt)`.
  The old standalone `end_capture`/`retrieve_keys` RPC methods are
  UNCHANGED and still used directly by `validate_proposer.py` (which wants
  the raw Q buffer / K tensors to validate capture and read-back in
  isolation from scoring) -- nothing about that validation script needed
  to change.
- `proposer.py`: driving logic (submit, drive to completion, timing logs,
  the DIAGNOSTIC check) factored into a shared `_submit_and_drive_turn`
  helper. `run_turn` (unchanged signature/behavior, still returns the raw
  Q buffer) now calls it, as does a NEW `run_turn_and_score` method (takes
  `pool_kernel_size`/`keep_kwargs` instead of returning Q, calls the new
  combined RPC) -- `pruner.py::compute_pruned_turn` now calls
  `run_turn_and_score` instead of `run_turn`+`retrieve_keys`+local scoring.
- `pruner.py`: `_score_and_select`'s position-conversion logic (turning
  local indices into `PrunedTurnResult`'s four fields) split out into
  `_positions_from_kept_indices`, shared by `_score_and_select` (oracle
  path, still scores driver-side against the target's own Q/K) and
  `compute_pruned_turn` (speculator path, now receives indices already
  computed in-process).

All CPU-testable pieces covered (`score_and_select_indices`,
`_positions_from_kept_indices` -- confirmed `pruner.py` itself imports
cleanly without a live vLLM engine, so these are now exercised in
`test_vllm_patch.py` alongside the rest). **This is the largest,
least-validated change made in this troubleshooting pass** -- it touches
the core scoring data flow shared by BOTH `M-k*-g*` and `SPARSE-k*-g*`.
Validate in order: `validate_proposer.py` first (confirms `run_turn`'s
unchanged behavior still works, and `retrieve_keys` still works standalone
-- neither exercises the NEW `run_turn_and_score`/`end_capture_and_score`
path, so this alone is NOT sufficient), then a small `M-k80-g32` or
`SPARSE-k80-g32` smoke test (either exercises `compute_pruned_turn`'s new
call path), checking the new `[proposer.run_turn_and_score]` log line's
"in-process K retrieval + scoring done" duration against the old ~89s
figure, and checking predictions/keep-rates look sane (not empty, not
100% kept when a keep rate < 100% was requested) before trusting a full
sweep.

**Fifth finding, from the fourth change's own first real measurement**:
`[proposer.run_turn_and_score]` logged 34.26s for in-process K retrieval +
scoring (kept=70,317 of an 87,972-position candidate pool,
actual_look_ahead_cnt=8 -- full lookahead completed, so this isn't an EOS-
short-circuit case). A real improvement over the old design (which spent
89s on `retrieve_keys` ALONE, before any scoring even started), but still
large for what should be a GPU-bound matmul/softmax/topk pipeline over a
context this size. Root cause: `SpeculatorGPUModelRunner.end_capture`/
`retrieve_keys` both moved their results to CPU (`.to("cpu")`) as their
OWN last step, a leftover from when they were ONLY ever called by
`SpeculatorWorker`'s RPC-exposed wrapper methods (which genuinely need
CPU tensors to cross `collective_rpc`'s process boundary). Once
`end_capture_and_score` started calling them IN-PROCESS too, that CPU
move happened before scoring had a chance to use the GPU -- forcing
`compute_attention_score`'s matmul and `aggregate_attention_score`'s
softmax/pooling (over an 88k-token context, 16 layers) onto CPU instead
of GPU, plausibly the majority of the 34s.

Fixed by moving the `.to("cpu")` responsibility DOWN to where it actually
belongs: `SpeculatorGPUModelRunner.end_capture`/`retrieve_keys` now stay
on-device unconditionally; `SpeculatorWorker.end_capture`/`retrieve_keys`
(the RPC-exposed wrappers `validate_proposer.py` and `proposer.py`'s
`run_turn`/`retrieve_keys` still call directly) do the `.to("cpu")`
themselves, right at the actual process boundary. `end_capture_and_score`
needed no changes at all -- it already just calls `self.end_capture`/
`self.retrieve_keys` (the model-runner-level, now on-device versions) and
lets whatever device Q/K land on flow through to `score_and_select_
indices` naturally. **Also unvalidated on real hardware** -- same
validation order as the fourth change above; watch the SAME
`[proposer.run_turn_and_score]` log line's duration for the actual
improvement.

**Sixth finding, on the TARGET side this time (not the speculator)**: real
runs showed `SPARSE-k*-g*`'s own target-side generation taking ~0.5-1s
longer per turn than `M000` baseline's, a gap `M-k*-g*` doesn't show to
the same degree. Root-caused to `SparseTargetGPUModelRunner.
_apply_sparse_attention_overrides`'s per-layer loop
(`sparse_target_runner.py`) -- unlike the physically-pruned pipeline's
RoPE-position patch (`model_runner.py`, O(1) per decode step: one dict
lookup + one in-place tensor add), this one runs `_patch_layer_metadata`
once per LAYER (32 for Llama-3.1-8B) on EVERY decode step, each call
previously doing 2-3 separate small GPU tensor writes. Under
`enforce_eager=True` (required elsewhere in this pipeline for the hook
mechanisms, so no CUDA-graph batching to amortize per-op dispatch
overhead), that's up to ~2,000 individual small GPU ops per turn
(`max_tokens` x layers x ops/layer) purely for this bookkeeping, with
nothing analogous on the baseline side at all. Fixed by building the
zero-padded block-table row ONCE per decode step (not once per layer) and
writing it as a single op per layer instead of two -- cuts the per-layer
GPU-op count from 3 to 2 (build-once + one write per layer, vs. the old
per-layer gather-write + zero-pad + seq_len-write). Purely mechanical --
`compute_sparse_gather_view` (the correctness-critical, CPU-tested
arithmetic) is completely untouched, only HOW the already-computed result
gets written into each layer's metadata changed. **Unvalidated on real
hardware** -- re-run `validate_sparse_attention.py` (its Steps A/B/C
exercise this exact write path) before trusting it, then compare the
target-generation timing log against the pre-fix ~0.5-1s gap.

**Seventh finding -- SUPERSEDES the sixth's framing, a real CORRECTNESS
bug, not a performance one.** Added prefill/decode-split logging
(`drive_single_request_to_completion`/`drive_one_turn_of_session`) plus
direct per-turn timing of the metadata-patch overhead itself
(`pop_override_timing`) specifically to check the sixth finding's theory
against real numbers. The metadata-patch overhead measured tiny (~0.12-
0.17s/turn) -- real, but nowhere near enough to explain the gap. What
actually explained it: `SPARSE-k*-g*` was hitting `max_tokens` (~63
decode steps) on **every single turn**, while `M000` stopped naturally
after 13-48 tokens -- a 3-5x difference in decode STEP COUNT, not
per-step cost. Inspecting the actual predictions confirmed why: the model
would produce a correct, concise answer, then continue generating (having
no way to stop) into the literal words "assistant"/"Assistant" repeated
until the token cap, or a different degenerate repetition loop --
classic "never found a recognized stop condition" behavior, not a model-
quality/attention-restriction issue.

Root cause, confirmed by reading vLLM source directly: `LLMEngine.
add_request()` accepts a pre-built `EngineCoreRequest` VERBATIM
(`llm_engine.py`'s `isinstance(prompt, EngineCoreRequest)` branch just
does `request = prompt`), which means `InputProcessor.process_inputs()`
-- the NORMAL request-construction path `add_request()` otherwise uses --
never runs. That method is the ONLY place `SamplingParams.
update_from_generation_config(...)` gets called
(`input_processor.py:315-318`), which populates a request's real
stop-token set from the model's own `generation_config.json` -- for a
Llama-3.x-Instruct model, this is where the CHAT-TEMPLATE's own
turn-ending token (e.g. `<|eot_id|>`) gets added, IN ADDITION to the base
tokenizer's `eos_token_id` (confirmed by reading `update_from_generation_
config`'s real body: without this call, `_eos_token_id` is never set at
all, and `stop_token_ids` never gets the chat-specific ids). Both
`predict_scbench.py::build_sparse_session_request` and
`validate_resumable_session.py::build_resumable_request` construct
`EngineCoreRequest` directly (the only confirmed way to set
`resumable=True`, since `add_request()` has no such kwarg) with a bare
`SamplingParams(max_tokens=..., temperature=...)` -- neither had ever
been through this population step, for ANY resumable-session turn, ever.

**Why `validate_sparse_attention.py`'s "all passed" and
`validate_resumable_session.py`'s own TTFT-based "PASS" didn't catch
this**: neither validation script's success criteria depends on the
model actually stopping cleanly -- the needle-in-a-haystack test only
checks whether a short, fixed-`max_tokens` generation contains an
expected substring, and the TTFT test only checks turn-2 latency, not
answer quality/termination. This bug could hide behind both without
either failing.

Fixed in both files: `build_sparse_session_request`/`build_resumable_
request` now take the already-constructed `llm_engine` and call `self.
input_processor.generation_config_fields`/`renderer.get_eos_token_id()`/
`tokenizer` -- the exact same objects the normal path would have used --
to populate `sampling_params` via `update_from_generation_config`/
`update_from_tokenizer` before building the `EngineCoreRequest`, reusing
vLLM's own logic rather than reimplementing it.

**Eighth finding -- a SECOND, previously-latent bug the seventh's fix
exposed (not introduced), a real engine crash.** Re-running after the
seventh fix hit `RuntimeError: Invalid request status: RUNNING` inside
`Scheduler.schedule()`, on a session's very first resumption. Traced
through vLLM source to a genuine race between two mechanisms that had
never been exercised together before:

1. **Async scheduling** (`SchedulerConfig.async_scheduling`, on by
   default in this fork) makes `UniProcExecutor.max_concurrent_batches`
   return 2, routing `EngineCore` through `step_with_batch_queue` -- a
   PIPELINED mode where `AsyncScheduler._update_after_schedule`
   optimistically schedules a request's NEXT step before the CURRENT
   step's real output (and thus whether it actually stopped) is known.
   This is fine for an ordinary request. For a **resumable session**,
   `Scheduler._handle_stopped_request` immediately re-parks/re-enqueues
   the SAME request the instant a real stop is observed -- while a
   "phantom" already-pipelined step for that same request can still be
   in flight. When the scheduler later reaches that phantom step, it
   finds the request already back in `RUNNING` (from the resumption)
   where it expected `WAITING`/`PREEMPTED`, and crashes.
2. This combination was never exercised before the seventh fix: every
   turn previously ran to `max_tokens` (a scheduler-predictable stopping
   point), never stopping mid-pipeline via a genuine EOS match, which is
   exactly the condition needed to trigger this race.

Fixed by passing `async_scheduling=False` when constructing any engine
that uses the resumable-session mechanism -- `predict_scbench.py`'s
sparse-mode target (`run_experiment`) and `validate_resumable_session.py`'s
own engine, both updated. Forces the plain, non-pipelined `step()` path
(`batch_queue` stays `None`), sidestepping the race entirely. Real
tradeoff: no overlap between scheduling and execution, some throughput
cost -- but a correctness requirement for this pipeline's own use of
resumable sessions until/unless fixed further upstream in vLLM itself.

**Both the seventh and eighth fixes are unvalidated on real hardware as
a PAIR** -- re-run `validate_resumable_session.py` first (cheap, fast,
and its own turns can hit real EOS now too), then a `SPARSE-k*-g*` smoke
test, and actually read a few turns of the resulting predictions file
(not just the timing logs) to confirm generation stops cleanly instead
of degenerating AND the engine doesn't crash on resumption.

---

## SpecPrefill settings

- BF16 precision
- Chunk-based attention scoring, look-ahead count **8**, `pool_kernel_size`
  **13** (same values as `../spec_prefill_llama/`'s matrix -- these are
  algorithm hyperparameters, not re-derived for the multi-turn extension).
- `enforce_eager=True` on both the target and speculator engines (same
  `functools.partial`-hook-vs-torch.compile reasoning as the single-turn
  pipeline, now applying to BOTH engines since the speculator is hooked
  in-process too, not just the target).
- `keep_mode="keep"` for the MVP sweep (protocol's "FIRST: KEEP") --
  `discard` is implemented (`conversation_state.py`, unit-tested) but not
  yet run as part of the default experiment matrix below.

---

## Experiment matrix

Confirmed MVP scope (with the user): 3 SCBench configs
(`scbench_qa_eng`/`scbench_kv`/`scbench_summary`), no baseline methods
implemented this pass.

| ID | Label | Keep rate | KV granularity | Keep mode |
|---|---|---:|---:|---|
| M000 | Baseline (no pruning) | 100% | — | — |
| M-k80-g{token,16,32,64} | Keep 80% | 80% | token/16/32/64 | keep |
| M-k60-g{token,16,32,64} | Keep 60% | 60% | token/16/32/64 | keep |
| M-k40-g{token,16,32,64} | Keep 40% | 40% | token/16/32/64 | keep |
| M-k20-g{token,16,32,64} | Keep 20% | 20% | token/16/32/64 | keep |
| ORACLE-k{80,60,40,20} | Oracle upper bound (SPARSE architecture, target checkpoint as scorer) | 80/60/40/20% | 32 (representative, pairs with SPARSE-k\*-g32) | keep |
| SPARSE-k{80,60,40,20}-g{16,32,64} | Persistent cache + sparse attention (see that section above) | 80/60/40/20% | 16/32/64 (no `token` -- block-gather is block-granular only) | keep (only) |

`predict_scbench.py --list` prints this matrix (generated programmatically,
not hand-enumerated -- see that script's `_build_experiments`). Use
`--exp specprefill` for all M-k*-g* rows, `--exp sparse` for all
SPARSE-k*-g* rows, `--exp oracle` for the 4 ORACLE-k* ceiling rows, or
`--exp all` for everything.

Metrics captured per turn: per-config metric (`grade_scbench.py` --
`in_match` for `scbench_kv`, `qa_f1_score` for `scbench_qa_eng`, ROUGE-L for
`scbench_summary`), TTFT, `num_cached_tokens` (speculator), actual keep
rate. Broken down by `(config, turn_idx)` in grading -- the `turn_idx` axis
is the multi-turn-specific signal (does accuracy degrade as the conversation
lengthens?) that a single-turn benchmark never needed.

`all_runs.csv` also records per-turn wall-clock time: `seconds_per_turn_mean`
(every completed turn) and `seconds_per_turn_excl_turn0_mean` (excludes
`turn_idx == 0` for every conversation, not just the first one processed).
Turn 0 pays each conversation's own cold-start cost -- the full context's
first prefill, into the target directly for baseline/specprefill, or into
both the target session AND the speculator's own growing cache for sparse
-- a fundamentally different, much larger cost than any later turn's
incremental one; averaging it in with steady-state turns would make "time
per turn" mostly reflect how big turn 0's context was, not the recurring
per-turn cost the metric is meant to isolate. Both are computed from a new
per-turn timer (`t_turn_start`) in each `run_*` function, recorded only for
turns that actually completed (not skipped).

`out_tokens_per_second` records output-token throughput: total generated
tokens (`sum(stats["out_lens"])`) divided by the experiment's total
`elapsed_time` -- the same "over the whole run's wall time" convention
`turns_per_second` already uses, not a decode-only figure. That means it
folds in prefill, speculator scoring (specprefill/sparse), and everything
else the pipeline actually spends time on for that mode -- it answers
"how many output tokens does this configuration produce per second of
real wall clock," not "how fast is the model's decode loop in isolation."

`DISCARD` mode and the `ORACLE` rows' full granularity cross are natural
next steps once `M000`/`M-k*-g*` are validated and run — not part of this
pass's default sweep.

---

## Benchmark

**SCBench** (arXiv:2412.10319, `microsoft/SCBench` on Hugging Face) --
`scbench_qa_eng` (semantic retrieval / free-form QA), `scbench_kv` (string /
exact retrieval), `scbench_summary` (global-information / summarization).
Dataset prep (`datasets/prep_scbench.py`), prediction generation
(`predict_scbench.py`), and grading (`grade_scbench.py`) are all new for
this pipeline (SCBench's genuinely multi-turn structure -- confirmed
empirically as exactly 5 turns sharing one long context per row, across all
3 MVP configs, not the HF dataset card's stated "2-4" -- has no analog in
the single-turn LongBench-v2 pipelines this was built from).

---

## Success criteria

- Score drop ≤5% (per-config metric, see `grade_scbench.py`) at each
  keep-rate row compared to M000, broken down by `turn_idx` -- report
  whether degradation is uniform across turns or concentrated in later
  turns (the multi-turn-specific failure mode this whole experiment exists
  to check for).
- Report TTFT/throughput improvement over M000 for each keep-rate row (no
  fixed pass/fail threshold -- the sweep itself is the signal).
- Report the oracle rows as an accuracy ceiling reference (wired up --
  see decision 5 and "Implementation status" #4). Read them as a pair of
  gaps, not one number: `ORACLE-k{N}` vs. `SPARSE-k{N}-g32` is how much
  the 1B speculator's estimate costs, `ORACLE-k{N}` vs. `M000` is how much
  block-granular sparse decode costs when the estimate is as good as this
  method can make it. Which of the two dominates decides where accuracy
  work should go next, and the graded SPARSE sweep alone cannot tell them
  apart.

---

## Resource requirements

**2x A100 80GB**, per the protocol document this plan was built from. Not
yet empirically re-derived for this specific pipeline (Llama-3.1-8B target +
Llama-3.2-1B speculator, same combined ~9B-parameter footprint as
`../spec_prefill_llama/`'s single-turn version, which itself estimates "likely
fits on a single GPU" but hasn't confirmed it) -- follow the protocol's
stated requirement rather than the single-turn sibling's smaller estimate
until this pipeline's own validation scripts (`REPRODUCE.md` step 5) confirm
otherwise, since the multi-turn speculator's growing, long-lived KV cache
(vs. the single-turn pipeline's per-call throwaway one) is a real, new
memory-footprint variable that estimate didn't have to account for.

**ETA**: TBD

---

## References

- [SpecPrefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation](https://arxiv.org/abs/2502.02789) (ICML 2025)
- [SCBench: A KV Cache-Centric Analysis of Long-Context Methods](https://arxiv.org/pdf/2412.10319)
- `microsoft/SCBench` (Hugging Face) / `microsoft/MInference` (GitHub, `scbench/` -- reference harness `eval_utils.py`'s metric functions ported into `grade_scbench.py`)
- `../spec_prefill_llama/EXPERIMENT_PLAN.md` -- the single-turn pipeline this was built on top of
- `../spec_prefill_llama/REPRODUCE.md` -- environment setup this pipeline's own `REPRODUCE.md` follows the same conventions as

---

## Files in this directory

| File | Purpose |
|---|---|
| `EXPERIMENT_PLAN.md` | This file |
| `README.md` | Overview / index |
| `REPRODUCE.md` | Environment setup + reproduction steps |
| `.env_exports.sh` | Local env config (model paths, HF token) |
| `vllm_patch/` | The multi-turn Algorithm 1 implementation -- see its own `__init__.py` module map for the copied/new/rewritten breakdown |
| `test_vllm_patch.py` | Unit tests -- engine-agnostic pieces + `conversation_state.py` (no GPU needed) |
| `validate_proposer.py` | GPU-node validation: persistent speculator engine, cross-turn KV read-back |
| `validate_runner_integration.py` | GPU-node validation: `worker_cls` wiring + multi-turn RoPE position-override correctness |
| `validate_resumable_session.py` | GPU-node validation: target-side session persistence (TTFT evidence) -- see "Persistent KV cache + sparse attention" section |
| `validate_sparse_attention.py` | GPU-node validation: decode-step block-gather sparse attention (needle-in-haystack) -- see "Persistent KV cache + sparse attention" section |
| `datasets/prep_scbench.py` | Downloads `microsoft/SCBench`'s 3 MVP configs, writes `datasets/scbench_samples.jsonl` |
| `predict_scbench.py` | Runs the M000/M-k*-g*/ORACLE-k*/SPARSE-k*-g* matrix, writes a per-turn predictions JSONL per experiment |
| `grade_scbench.py` | Scores a predictions file against `prep_scbench.py`'s samples, per-config metrics |
| `datasets/` | SCBench prep output (gitignored) |
| `results/` | Output directory (gitignored) |
