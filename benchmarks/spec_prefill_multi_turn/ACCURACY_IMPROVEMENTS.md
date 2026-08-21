# Ways to improve accuracy

Candidate improvements for the SPARSE pipeline's accuracy, drawn from the two
reference papers (SpecPrefill, arXiv:2502.02789; SCBench, arXiv:2412.10319)
and from what this pipeline's own code and graded sweep actually do. Every
item names the specific place it would change.

**Read this document gated on the ORACLE-k\* result.** Most of what follows is
speculative until the ceiling rows say which half of the pipeline is losing
the points, and the two halves have disjoint fixes. See "Step 0" below.

---

## Step 0: the gate (run this before anything else)

`ORACLE-k{N}` is `SPARSE-k{N}-g32` with the target checkpoint scoring instead
of the 1B speculator — see README's "ORACLE-k\*: what it is and what it
bounds". It splits the unexplained `scbench_kv` degradation in two:

| Gap | Reading | Where the work goes |
|---|---|---|
| `ORACLE-k{N}` − `SPARSE-k{N}-g32` | the 1B speculator's estimation error | §1 Scoring signal |
| `M000` − `ORACLE-k{N}` | what block-granular sparse decode costs with a perfect-as-this-method-allows estimator | §2 Allocation, §3 Mechanism |

Decision rule:

- **Oracle recovers most of M000** → the mechanism is fine, the estimator is
  the problem. Go to §1, and §2's per-region/nucleus items.
- **Oracle is also far below M000** → better scoring cannot save this. Go to
  §3 and the structural items in §2 (per-layer selection, dilation).
- **Both gaps are large** → both, but §3 first: a better estimator feeding a
  lossy mechanism still loses.

Compare on an identical conversation subset (`compare_ceiling.py`), not
against the published full-sweep numbers — a 20-conversation oracle against a
100-conversation SPARSE row is not a comparison.

---

## §1 Scoring signal (speculator side)

The FLOP model puts the speculator's scoring pass at ~0.007% of a turn
(~0.04 TFLOP vs ~538 for its own prefill). **Scoring quality is essentially
free to improve** — that is the single most useful fact in the project, and
almost nothing here costs measurable time.

Current pipeline ([scoring.py:103](vllm_patch/scoring.py:103)): softmax →
`avg_pool1d(kernel=13)` → **max over (layer, head)** → mean over 8 lookahead
steps.

1. **Drop `max` over (layer, head).** Max is winner-take-all: one peaked
   early-layer positional head sets the entire importance vector. Try
   mean-over-heads of softmax mass, a top-q quantile, or per-head z-scoring
   before combining so no single head dominates the scale.
2. **Restrict which layers vote.** Layers 0–1 are near-universally
   positional/sink-dominated and carry little retrieval signal. Score from
   mid/late layers only. One-line change; plausibly the largest cheap win.
3. **Retrieval-head filtering.** Only a small head subset does copy/retrieval.
   Identify them offline on the speculator with a needle probe and score using
   only those. The strongest version of items 1–2, and aimed squarely at
   `scbench_kv`'s failure mode.
4. **Mask sinks before pooling.** Position 0 and delimiters absorb enormous
   softmax mass and skew both the pooled scores and the chunk means around
   them. Zero them in the score and force-keep them separately — wrapper spans
   are already unioned into the selection, so removing them from *scoring*
   costs nothing.
5. **Better lookahead queries.** Currently 8 tokens the 1B generated itself;
   if it drifts, the queries are off-distribution. Ablations: (a) use the last
   N tokens of the user query directly, no generation; (b) union query-token
   queries with lookahead queries; (c) raise `LOOK_AHEAD_CNT` 8 → 16/32 and
   look for saturation.
6. **Speculator/target geometry mismatch.** Llama-3.2-1B's head structure is
   not Llama-3.1-8B's. ORACLE quantifies exactly this; if the gap is large,
   consider scoring from a few of the *target's* own layers.

---

## §2 Budget allocation (where the k% goes)

**The sweep already contains the strongest hint in the project.** On
`scbench_kv` at fixed keep rate, score rises monotonically with granularity —
k20: g16 46.6 → g32 56.6 → g64 62.2. Coarser blocks mean *worse* selection
precision but *better* contiguity, and they win by 16 points. **Precision is
not the binding constraint; recall and locality are.**

1. **Block-neighborhood dilation.** At g16, also keep the ±1 neighbouring
   blocks of every selected block. Prediction from your own data: g16 +
   dilation should approach or beat g64 at equal token budget — g64's locality
   without g64's quantization of the budget. ~10 lines in
   [`_chunked_topk_indices`](vllm_patch/scoring.py:143), and the cleanest test
   of the contiguity hypothesis.
2. **Nucleus budget instead of fixed top-k.** Keep chunks until cumulative
   importance mass reaches p (e.g. 0.9) rather than a fixed k%. `scbench_kv`
   needs high recall; `summary` provably doesn't. A mass-based rule spends
   budget where the distribution is actually concentrated and should dominate
   fixed-k on the accuracy/compute Pareto.
3. **Recency window.** Force-keep the last W tokens of the ledger (previous
   turn's answer tail + boundary). Currently only *this* turn's own generated
   tokens, query, and wrapper spans are protected. Classic Λ-pattern; cheap;
   multi-turn coherence leans on it.
4. **Sticky / cross-turn selection.** Selection is recomputed from scratch and
   overwritten each turn ([sparse_selection_registry.py](vllm_patch/sparse_selection_registry.py)).
   SCBench's central multi-turn finding is cross-turn attention drift — a block
   that mattered at turn 1 often matters again at turn 4. Try half the budget
   as a union with turn *t−1*'s selection, or an EMA-decayed cumulative score
   held in `ConversationState`.
5. **Per-region floor.** A global top-k can zero out an entire span. Guarantee
   a minimum fraction per prior turn / per context segment so a needle in a
   globally-unattended region stays reachable.
6. **Per-layer selection.** All 32 layers currently share one identical block
   set — the same padded row is written into every layer's metadata
   ([sparse_target_runner.py:523](vllm_patch/sparse_target_runner.py:523),
   assumption #4 at [:158](vllm_patch/sparse_target_runner.py:158)). This is
   the crudest possible choice and out of step with Quest/MInference. The
   patch loop is *already per-layer*, so this is mostly bookkeeping: per-layer
   index sets, per-layer `seq_lens`/`max_seq_len`, and stop max-ing scores
   across layers. Higher effort; the structural ceiling-raiser.

---

## §3 Mechanism (target side)

1. **Mid-turn re-selection.** The selection is frozen for all 64 decode steps
   of a turn. For `scbench_kv` the model needs different context as it emits
   different parts of the answer. Re-score every ~16 steps from the target's
   own recent query vectors (true Quest-style dynamic sparsity). If ORACLE
   shows a *small* gap but SPARSE is still bad, this is the prime suspect.
2. **Dense warm-up.** Run the first N decode steps dense, then go sparse. The
   first token conditions all 63 after it. Costs almost nothing.
3. **Self-generated history compounding.** History is self-generated, so a bad
   k20 turn-1 answer poisons turn 2's context — degradation currently charged
   to sparsity. A golden-history variant would separate "sparsity hurt this
   turn" from "sparsity hurt an earlier turn". The resumable session has no
   hook to substitute golden text; feeding golden answer tokens as part of the
   next turn's delta instead of the generated ones is the likely route.

---

## §4 Measurement (do regardless)

1. **Per-`turn_idx` breakdowns.** `grade_scbench.py` already computes
   `by_config_and_turn_idx` — the aggregate tables hide whether degradation
   compounds across turns, which is the entire multi-turn thesis.
2. **Confidence intervals.** `scbench_qa_eng` (~24 convs) and `summary` (~70)
   are noise-dominated; ±1.64 points is not a finding. `compare_ceiling.py`
   reports a cluster bootstrap over conversations — use it before believing
   any ordering.
3. **A non-LCS metric for summary**, e.g. entailment or an LLM judge, to test
   whether `rouge_l_f1` is hiding real degradation (the sweep's flat summary
   row has two live explanations and they are currently indistinguishable).
4. **Null controls** (deferred by request, but cheap and clarifying): random
   block selection and recent-window-only at matched keep rate. If summary is
   flat for random too, that table measures metric insensitivity rather than
   selection quality.

---

## Shortlist

Ranked by expected value per unit of effort, assuming the oracle gate has been
read first.

| # | Change | Effort | Risk | Section |
|---|---|---|---|---|
| 1 | De-max the (layer, head) aggregation; restrict voting layers | ~1 hour | low | §1.1–1.2 |
| 2 | Block-neighborhood dilation at g16 | ~half day | low | §2.1 |
| 3 | Nucleus (mass-based) keep budget | ~half day | low | §2.2 |
| 4 | Recency window + sticky cross-turn union | ~1 day | low | §2.3–2.4 |
| 5 | Retrieval-head filtering | ~2 days | medium | §1.3 |
| 6 | Mid-turn re-selection | ~3 days | medium | §3.1 |
| 7 | Per-layer selection | ~1 week | high | §2.6 |

Items 1–3 are all testable against gold-answer block recall
([diagnose_gold_survival.py](diagnose_gold_survival.py)) before paying for a
full graded sweep — full grading is far too slow to iterate scoring variants
against, and recall of the gold span correlates directly with `in_match`.
