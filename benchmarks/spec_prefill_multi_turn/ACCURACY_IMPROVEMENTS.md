# Ways to improve accuracy

Candidate improvements for the SPARSE pipeline's accuracy, drawn from the two
reference papers (SpecPrefill, arXiv:2502.02789; SCBench, arXiv:2412.10319)
and from what this pipeline's own code and graded sweep actually do. Every
item names the specific place it would change.

**The gate has been run.** `ORACLE-k20` over all 100 `scbench_kv`
conversations says the 1B speculator's estimation error is **68% of the loss**
and the sparse-decode mechanism itself is the other **32%**. §1 is therefore
the priority and §2/§3 are real but secondary. Full numbers and what they rule
out in "Step 0" below.

---

## Step 0: the gate — RESOLVED

`ORACLE-k{N}` is `SPARSE-k{N}-g32` with the target checkpoint scoring instead
of the 1B speculator — see README's "ORACLE-k\*: what it is and what it
bounds". Run over all 100 `scbench_kv` conversations (500 turns, the same set
behind every published number in the README):

| Row | Score | vs. M000 |
|---|---:|---:|
| M000 | 81.6 | — |
| **ORACLE-k20** | **73.6** | **−8.0** |
| SPARSE-k20-g32 | 56.6 | −25.0 |

The 25.0-point degradation splits:

| Term | Points | Share | Section |
|---|---:|---:|---|
| **Estimator** (`ORACLE` − `SPARSE`) — the 1B's estimation error | **17.0** | **68%** | **§1 Scoring signal** |
| **Mechanism** (`M000` − `ORACLE`) — what block-granular sparse decode costs with the estimator held as good as this method allows | 8.0 | 32% | §2 Allocation, §3 Mechanism |

**Verdict: the scorer is the dominant term, by roughly 2:1.** Given §1's items
cost essentially nothing at runtime (the scoring pass is ~0.007% of a turn),
that is the cheapest 17 points in the project. The mechanism's 8 points are
real and worth §2/§3 afterwards — they are not zero, and they bound whatever
§1 can achieve.

### What the per-turn breakdown rules out

| Deficit vs. M000 | turn 0 | turn 1 | turn 2 | turn 3 | turn 4 |
|---|---:|---:|---:|---:|---:|
| M000 (absolute) | 70.0 | 81.0 | 84.0 | 87.0 | 86.0 |
| mechanism (M000 − ORACLE) | 9 | 8 | 8 | 12 | 3 |
| estimator (ORACLE − SPARSE) | 23 | 14 | 18 | 9 | 21 |

- **Degradation does not compound across turns.** Both gaps are flat-to-
  shrinking in `turn_idx`; nothing accumulates. This is a real negative result
  for §3.3 (self-generated-history compounding) — **deprioritized on
  evidence**, not deferred for cost. It also means the multi-turn setting is
  not adding a failure mode on top of the single-turn one here; it is the
  same per-turn selection problem, five times.
- **Turn 0 is the worst turn for every row**, M000 included (70.0 vs 86.0 at
  turn 4 — the documented turn-0 dip), and it is where sparsification hurts
  most (−32). 23 of those 32 points are the estimator. So §1's improvements
  should show up first and largest at turn 0, which makes turn 0 the cheapest
  place to measure whether a scoring change worked.

Read gaps by `compare_ceiling.py`'s **paired** CI column, not by whether the
marginal CIs overlap — every row answers the same conversations, so the
conversation-difficulty variance that dominates each marginal interval
cancels out of the difference. Here the marginal intervals for M000
[76.4, 86.4] and ORACLE-k20 [68.2, 78.6] overlap while the paired 8-point
difference is solid.

Compare on an identical conversation subset (`compare_ceiling.py`), not
against the published full-sweep numbers — a 20-conversation oracle against a
100-conversation SPARSE row is not a comparison.

**Read the agreement table before the means.** Two rows can have nearly equal
means while disagreeing about almost every individual conversation, if one
wins some and loses others and the errors cancel. On a near-binary metric like
`in_match` that is the *expected* shape of a selection method that scrambles
which conversations succeed rather than uniformly losing points — and the
means alone report it as "no difference", which is the opposite of what it
means. First observed on the 20-conversation ORACLE-k20 run: means 54.0
(oracle) vs 52.0 (sparse), a 2-point gap readable as "the estimator buys
nothing", while the oracle agreed with M000 on 14/20 conversations (mean
|diff| 11.0) and sparse on 5/20 (mean |diff| 33.0). The estimator mattered
enormously; its errors simply cancelled at that keep rate.

---

## §1 Scoring signal (speculator side)

The FLOP model puts the speculator's scoring pass at ~0.007% of a turn
(~0.04 TFLOP vs ~538 for its own prefill). **Scoring quality is essentially
free to improve** — that is the single most useful fact in the project, and
almost nothing here costs measurable time.

Current pipeline ([scoring.py:103](vllm_patch/scoring.py:103)): softmax →
`avg_pool1d(kernel=13)` → **max over (layer, head)** → mean over 8 lookahead
steps.

1. **Drop `max` over (layer, head).** ❌ **FALSIFIED — and backwards.**
2. **Restrict which layers vote.** ❌ **Null.**

### The §1.1/§1.2 sweep: what it actually found

Six variants, all 100 `scbench_kv` conversations, against the
`SPARSE-k20-g32` baseline (56.6). `*` = paired 95% CI excludes 0.

| Variant | Score | Δ | paired 95% CI | |
|---|---:|---:|---|---|
| `lyr2h` (layers ≥ L/2) | 58.2 | **+1.6** | [−2.2, +5.2] | ns |
| `lyrskip2` (drop layers 0–1) | 56.2 | −0.4 | [−3.0, +2.4] | ns |
| `aggmean-lyr2h` | 52.6 | −4.0 | [−8.2, −0.4] | * |
| `lyr4q` (layers ≥ 3L/4) | 48.4 | −8.2 | [−12.6, −3.8] | * |
| `aggmean` | 44.2 | −12.4 | [−17.0, −8.4] | * |
| `aggzmean` | 39.4 | −17.2 | [−22.0, −12.6] | * |

**`max` was right, and the hypothesis behind replacing it was wrong.** Every
step away from winner-take-all made things worse, monotonically with how far
it went: `mean` −12.4, `zmean` (which additionally equalizes every head's
scale) −17.2. This is not a null result, it is a large effect in the opposite
direction, well outside anything six-way multiple testing could manufacture.

The retrospective reading, which is a *better* model of the problem than the
one that motivated the sweep: for needle retrieval the signal **is** a single
peaked head. Llama-3.2-1B has 16 × 32 = 512 (layer, head) distributions, of
which a handful know where the needle is; averaging lets the ~500 that don't
outvote them, and z-scoring guarantees it by removing exactly the magnitude
advantage that made the informed heads distinguishable. `max` is a crude,
free, implicit retrieval-head selector — which is why §1.3 (explicit
retrieval-head filtering) is now the *better-motivated* item in this section,
not a redundant one: it is the principled version of what `max` does by
accident, and the right form is "restrict to known retrieval heads, then
aggregate within that set".

Layer restriction is null in both directions for the same reason. Early
layers rarely win the `max`, so dropping them barely perturbs anything
(`lyrskip2` agrees with the baseline on 464/500 turns, mean |diff| 7.2 —
the smallest change of all six). Dropping *late* layers removes real signal:
`lyr4q` keeps only 4 of 16 and loses 8.2 points. `lyr2h`'s +1.6 is the best
point estimate in the sweep and is not significant; even if real it is 9% of
the 17-point estimator gap, so it is not the answer either way.

**What this does not change:** the 17-point estimator gap is still there and
`ORACLE-k20` still proves it is achievable. What the sweep rules out is that
it is reachable by changing how the 1B's attention is *collapsed*. The
remaining candidates are about what is collapsed, not how: §1.3 (which heads),
§1.5 (which queries), §1.6 (which model) — see "After the sweep" below.

### After the sweep: the live hypotheses

The oracle differs from SPARSE in two ways at once — whose attention, and
whose lookahead tokens. Post-hoc collapse choices are now eliminated, so the
next discriminator should separate those two:

- **Scorer capacity (§1.6).** ❌ **Answered — works, and is the wrong
  currency.** See "The capacity probe" below.
- **Retrieval-head filtering (§1.3)** — now the best-motivated scoring change,
  per the reading above, and one of only two left that cost no capacity.
  **Gated before building**: `diagnose_retrieval_heads.py` measures the
  ceiling by picking heads with the gold answer's own position (cheating), so
  no honest identification method can beat what it reports, and it separately
  reports whether the winning heads are the *same* heads across turns — a
  retrieval head set is static by definition, so unstable heads kill §1.3
  regardless of the ceiling.

  ```bash
  python diagnose_retrieval_heads.py --predictions-file results/SPARSE-k20-g32_predictions.jsonl --keep-percentage 0.2 --max-conversations 20 --head-mass-out results/head_mass.json
  ```

  Read it as: ceiling near the all-head baseline (currently ~70% survival) →
  the 1B does not localize the needle, §1.3 is dead for one run's cost.
  Ceiling far above it *and* stable heads → it is a readout problem, build it,
  and `--head-mass-out` is already the head list to build from.
- **Lookahead source (§1.5)** — the other axis the oracle changed, and the
  other capacity-free one.

### The capacity probe: smooth, and economically backwards

Scoring the same `SPARSE-k20-g32` configuration with progressively larger
checkpoints, all 100 `scbench_kv` conversations:

| Scorer | Params | Score | vs. 1B | paired 95% CI |
|---|---:|---:|---:|---|
| Llama-3.2-1B (the speculator) | 1.24B | 56.6 | — | — |
| Llama-3.2-3B | 3.21B | 65.0 | +8.4 | [+2.4, +14.2] * |
| Llama-3.1-8B (= the target, ORACLE-k20) | 8.03B | 73.6 | +17.0 | [+10.2, +23.4] * |

**Log-linear in scorer size, with no plateau anywhere:**

    score ≈ 56.6 + 9.1 · ln(params / 1.24B)      (+6.3 points per doubling)

The fit through the two endpoints predicts **65.3** for 3B against an actual
**65.0** — a 0.3-point residual on an interpolated point. 1B→3B buys +8.4 and
3B→8B buys +8.6: the returns do not diminish, so there is no cheap
"just-big-enough" draft model hiding between the sizes. The estimator gap
closes smoothly and only fully when the scorer *is* the target.

**Why that settles it against capacity as the fix.** SPARSE already costs
**+21% FLOPs and +36% latency** against the dense M000 baseline it is supposed
to beat (its case rests on memory bandwidth, not arithmetic — see README's
FLOP sections). The speculator's incremental cost is ~180 TFLOP/turn of
SPARSE's ~1038; a 3B scorer is ~2.6x that, pushing the total to roughly 154%
of M000's FLOPs to buy +8.4 points while still sitting **16.6 below** the
baseline that is also ~3x faster. Scaling to the 8B scorer means running an
8B model twice per turn, which is not an acceleration method. Read the actual
cost off `all_runs.csv` (`seconds_per_turn_excl_turn0_mean`,
`total_tflops_per_turn_mean`) rather than this projection — the 3B run
recorded its own.

So capacity is a real lever on accuracy and a dominated one on the
accuracy-per-FLOP frontier this project exists to move. **What is left in §1
is the capacity-free items only: §1.3 (which heads) and §1.5 (which
queries).** If neither moves the number, the honest conclusion is that a
~1B draft cannot estimate an 8B model's retrieval attention well enough for
aggressive keep rates, and the project's result is that bound rather than a
way around it.

> **Running 1 and 2.** Six rows at the `k20-g32` probe point — the grid's
> worst corner, where the 17-point estimator gap has the most room to show
> above noise:
>
> ```bash
> python predict_scbench.py --exp score --scbench-config scbench_kv
> ```
>
> `SPARSE-k20-g32-{aggmean, aggzmean, lyrskip2, lyr2h, lyr4q, aggmean-lyr2h}`,
> compared against the existing `SPARSE-k20-g32` (56.6) and the `ORACLE-k20`
> ceiling (73.6) with `compare_ceiling.py`. Defaults are unchanged and
> bit-identical to the reference implementation (locked by a test), so every
> published row keeps its meaning; the variant is opt-in per row.
>
> **Iterate on gold-answer survival first**, not on graded sweeps — it needs
> only the speculator (no target engine, no generation, no grading), so a
> variant is judged in minutes:
>
> ```bash
> python diagnose_gold_survival.py --predictions-file results/SPARSE-k20-g32_predictions.jsonl --keep-percentage 0.2 --score-aggregation mean --score-layers second_half
> ```
>
> Turn 0 is the cheapest place to look: it is where sparsification costs most
> (−32) and where the estimator owns the largest share of it (23 of 32).
3. **Retrieval-head filtering.** ✅ **Gate passed, built, awaiting a graded
   run.** `SpecConfig.score_head_set`; rows `SPARSE-k20-g32-heads{1,2,4}`.

   `diagnose_retrieval_heads.py` measured it before it was built. Ranked on
   conversations 1–20, scored **fully out-of-sample** on 21–40:

   | Head set | Gold survived | vs. all-head `max` |
   |---|---:|---:|
   | all 512 (`max`, the reference) | 54.0% | — |
   | **fixed global top-2** | **82.0%** | **+28.0** |
   | fixed global top-4 | 78.0% | +24.0 |
   | fixed global top-16 | 67.0% | +13.0 |
   | *clairvoyant top-2 (ceiling)* | *84.0%* | *+30.0* |

   A **fixed 2-head mask captures 93% of what per-input clairvoyance could
   achieve**, and the heads are genuinely stable — top-2 Jaccard 0.71 against
   the global ranking, 0.63 between consecutive turns, and only **20 distinct
   heads across 100 turns** out of 512. Stability peaks at exactly the budget
   where the ceiling peaks.

   **A methodological note worth keeping.** The first run of this gate
   reported top-16 stability alone (Jaccard 0.26) and read as "the useful
   heads are chosen per input, §1.3 is dead". That was an artifact of
   measuring at the wrong budget: 2 stable heads plus 14 slots of noise
   produce exactly that number. Reporting stability at the budget where the
   ceiling peaks reversed the verdict. A stability metric has to be evaluated
   at the set size the method would actually use.

   **Graded result** (all 100 `scbench_kv` conversations, paired CIs):

   | Row | Score | Δ vs. baseline | Share of the 17.0-point estimator gap |
   |---|---:|---:|---:|
   | `SPARSE-k20-g32` (all heads) | 56.6 | — | — |
   | `-heads1` | 61.8 | +5.2 (ns) | 31% |
   | `-heads2` | 64.4 | +7.8 * | 46% |
   | **`-heads4`** | **65.2** | **+8.6 \*** | **51%** |
   | *ORACLE-k20, 3B scorer* | *65.0* | *+8.4* | *49%* |
   | *ORACLE-k20, 8B scorer (the ceiling)* | *73.6* | *+17.0* | *100%* |

   **A free 4-head mask on the 1B matches a 3B scorer** (65.2 vs 65.0) — the
   same accuracy the capacity probe bought for ~2.6x the speculator's FLOPs,
   for a mask that makes scoring marginally *cheaper*. This is the first
   change in §1 that both works and is economically sensible, and it closes
   **51% of the estimator gap**.

   After it, the two remaining gaps are equal: **8.4 points** of estimator
   error still unrecovered, **8.0 points** of mechanism (M000 − the 8B
   oracle). The next marginal point is equally available from §1 or §2.

   ### Calibration: gold survival overstates score gains ~2.5x on the margin

   Predicted before the run, from the gate's +28.0 survival at an ~80%
   average survival-to-score conversion: **+20**. Actual: **+8.6** — the
   prediction was 2.3x too optimistic, and the error is systematic rather
   than noise.

   The marginal conversion rate is ~31%, not the ~80% average, and in
   hindsight that is forced: the turns a better head set newly rescues are
   *by construction* the ones the old selection already failed on, so they
   are the harder turns, where the gold surviving is less often sufficient.
   Any survival-based projection should be discounted accordingly. The fast
   loop remains valid for **ranking** variants — it ranked these three
   correctly — but its magnitudes must not be read as score deltas.
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

Ranked by expected value per unit of effort, **given the resolved gate**: the
estimator owns 17.0 of the 25.0 points, so §1 leads. Items 1 and 5 attack that
68% directly; items 2-4 attack the mechanism's 8 points and are what remains
once §1 is exhausted.

| # | Change | Effort | Risk | Section |
|---|---|---|---|---|
| ~~1~~ | ~~De-max the (layer, head) aggregation; restrict voting layers~~ — **falsified, see §1** | — | — | §1.1–1.2 |
| ~~1~~ | ~~Scorer-capacity probe~~ — **answered: log-linear, economically dominated** | — | — | §1.6 |
| 1 | Retrieval-head filtering (capacity-free) | ~2 days | medium | §1.3 |
| 2 | Block-neighborhood dilation at g16 | ~half day | low | §2.1 |
| 3 | Nucleus (mass-based) keep budget | ~half day | low | §2.2 |
| 4 | Recency window + sticky cross-turn union | ~1 day | low | §2.3–2.4 |
| 5 | Lookahead-query source (capacity-free) | ~1 day | low | §1.5 |
| 6 | Mid-turn re-selection | ~3 days | medium | §3.1 |
| 7 | Per-layer selection | ~1 week | high | §2.6 |

Items 1–3 are all testable against gold-answer block recall
([diagnose_gold_survival.py](diagnose_gold_survival.py)) before paying for a
full graded sweep — full grading is far too slow to iterate scoring variants
against, and recall of the gold span correlates directly with `in_match`.
