# When speculation pays for itself

The derivation behind the deck's Analysis slide. Everything here comes from
`flops_model.py`'s coefficients, and the one closed-form prediction it makes
is checked against a measured run at the end.

## Notation

| symbol | meaning |
|---|---|
| `L` | resident context length entering the turn (tokens) |
| `d` | this turn's new query tokens (the delta submitted to the session) |
| `o` | this turn's output tokens |
| `k` | keep rate (nominal) |
| `A` | target attention cost per query-key pair, summed over all layers |
| `B` | the same for the speculator |
| `r` | `B / A` — the scorer's per-token cost ratio to the target |

For Llama-3.1-8B target and Llama-3.2-1B speculator:

```
A  = 32 layers x 4 x 32 heads x 128 head_dim = 524,288
B  = 16 layers x 4 x 32 heads x  64 head_dim = 131,072      ->  r = 1/4
SC = 16 layers x 2 x 32 heads x  64 head_dim x 8 look_ahead = 524,288
```

`SC` is the scoring pass (`QK^T` only, no `AV`, hence `2x` rather than `4x`).
Note `SC / A = 1` exactly, and more usefully `SC = 4rA` — scoring is worth
precisely four target-token-equivalents of context traffic.

## The steady-state win condition

Per steady-state turn (turn 0 excluded), against the dense M000 baseline:

**Savings.** Every query token the target processes attends `L` keys densely
and `kL` under the gather. That applies to the turn's `d` prefill tokens and
its `o` decode steps alike:

```
saving = A · L · (d + o) · (1 - k)
```

**Cost.** The speculator prefills the same delta plus the previous turn's
output into its own cache (its context is a ~99.5% prefix-cache hit, confirmed
by `num_cached_tokens_speculator_mean`), runs 8 look-ahead decode steps, and
scores:

```
cost = B · L · (d + o) + 8 · B · L + SC · L
     = A · L · [ r(d + o) + 8r + 4r ]
```

Dividing through by `A · L` gives the whole thing:

> **`(d + o)(1 - r - k) > 12r`**

At `r = 1/4` that is **`(d + o)(0.75 - k) > 3`**.

Both constants are the same number wearing two hats:

- **`1 - r` is a hard ceiling on the useful keep rate.** The scorer costs a
  quarter of the target per token, so at `k ≥ 0.75` there is no turn shape,
  context length or turn count that makes speculation cheaper. This is why
  every `k80` row in the measured sweep costs *more* than dense.
- **`12r` is a fixed per-turn overhead** — 8 look-ahead steps plus a scoring
  pass — paid whatever the turn contains.

## What follows

**Question and answer length help equally and linearly.** `d` and `o` enter
identically; only their sum matters. The fixed `12r` overhead amortises over
that sum, so short Q&A is the worst possible shape.

**Context length cancels out.** `L` scales the savings and the scorer's cost
in the same proportion, so it does not appear in the condition at all. It
enters only the break-even turn count below. This is counterintuitive for a
long-context method: a *shorter* shared context with the same per-turn traffic
pays back sooner.

**Turn count does not enter the condition either** — it decides whether the
per-turn margin has repaid turn 0, not whether a margin exists.

## Break-even turn count

Turn 0 is dense for the target under both scopes, but SPARSE additionally
pays the speculator's own full prefill of the context — `~(A/8)L²` against
the target's `~(A/2)L²`, i.e. a ~21% penalty (measured: kv 7,019 vs 5,770
TFLOP, summary 5,159 vs 4,258, qa_eng 4,711 vs 3,890).

Amortising that against the per-turn margin:

> **`N ≈ L / ( 8 · [ (d + o)(1 - r - k) - 12r ] )`**

### By turn shape (L = 100k, k = 20%)

| scenario | `d` | `o` | `d + o` | break-even turns |
|---|---:|---:|---:|---:|
| SCBench-shaped | 35 | 26 | 61 | **409** |
| code assistant | 500 | 500 | 1,000 | 23 |
| document pasted each turn | 2,000 | 100 | 2,100 | **11** |
| long generation (agent) | 50 | 2,000 | 2,050 | **11** |
| both long | 2,000 | 2,000 | 4,000 | 6 |

### By keep rate (L = 100k, `d + o` = 61)

| keep rate | min `d + o` to win at all | break-even turns |
|---|---:|---:|
| 80% | never | never |
| 60% | 20 | 2,033 |
| 40% | 9 | 681 |
| 20% | 5 | 409 |

### By context length (`d + o` = 61, k = 20%)

| `L` | break-even turns |
|---|---:|
| 100k | 409 |
| 32k | 131 |
| 8k | 33 |
| 2k | 8 |

## Validation

The model's only free-standing prediction is the break-even turn count, and
one is measured. At `L = 102k`, `d = 35`, `o = 26`, `k = 0.2` the formula
gives **417 turns** against a measured **418** — within 2%.

Two caveats. The model uses the *nominal* keep rate, while block granularity
makes the effective keep higher (84.8% measured at a nominal 80%), so the
figures run slightly optimistic. And it assumes the speculator keeps its
~99.5% prefix-cache hit rate, which holds in every run recorded so far.

## The bind, and what moves it

Read the keep-rate table against the accuracy results: on `scbench_kv`, only
**k80 clears a 5% accuracy drop** (−1.0%), and k80 is precisely the rate that
can never pay off. **k60 just misses** at −5.1% and would need ~2,000 turns.
k20 breaks even soonest and costs 30 points.

The keep rates cheap enough to pay for themselves are too aggressive to be
accurate; the ones accurate enough are too gentle to pay. That is structural,
not a tuning problem — no setting of `k` escapes it.

Three things would:

1. **A structurally cheaper scorer.** The ceiling *is* `1 - r`, and the
   overhead *is* `12r`, so halving `r` moves both:

   | scorer : target | max useful keep | fixed overhead | min `d + o` at k=60% |
   |---|---:|---:|---:|
   | 1 : 4 (today) | 75% | 3.0 | 20 |
   | 1 : 8 | 87.5% | 1.5 | 6 |
   | 1 : 16 | 93.8% | 0.75 | 2 |

   At 1:8, keep 60% becomes viable at realistic turn sizes.

2. **Scoring less often than every turn.** Every term in the overhead is
   per-turn. Reusing one selection across `m` turns divides `12r` by `m` and
   drops the `r(d + o)` term too. This is the "adaptively choose when to
   apply selection" item on Next Steps, and the model says it is the
   highest-leverage one there.

3. **Scoring with the target's own early layers** rather than a separate
   model. This is the cheapest lever of the three to reason about, because
   `r` needs no measurement at all — the first `n` layers of the target have
   `B = n · 4 · 32 · 128`, so

   > **`r = n/32` exactly.**

   | n | r | max useful keep | fixed overhead `12r` | min `d + o` at k=60% |
   |---:|---:|---:|---:|---:|
   | 1 | 1/32 | 96.9% | 0.375 | 2 |
   | 2 | 1/16 | 93.8% | 0.75 | 2 |
   | 4 | 1/8 | 87.5% | 1.5 | 6 |
   | 8 | 1/4 | 75.0% | 3.0 | 20 |

   `n = 8` reproduces the 1B speculator's `r` exactly, which is why the
   sweep stops there: past it the idea is strictly worse than the status quo
   on the axis it exists to improve. At `n = 2`, keep 60% and even keep 80%
   clear the ceiling — the two rates that carry acceptable accuracy and can
   never pay for themselves today.

   In a fully *fused* implementation the scorer's prefill disappears into
   the target's own, collapsing turn 0's +21% penalty as well. The
   `EARLY-k*-g32-L<n>` grid measures the **separate-engine** version, which
   still pays that prefill and is therefore the pessimistic bound on it.

   Two caveats the grid has to be read against. The truncated scorer decodes
   its lookahead tokens through the target's final norm and `lm_head`, which
   were trained for layer-32 outputs — an untuned early-exit head, so those
   tokens degrade as `n` shrinks. And the table above is a cost argument
   only: whether early-layer attention *localizes* the answer is a separate,
   empirical question, gated cheaply by
   `diagnose_retrieval_heads.py --layer-prefix-budgets` before any of the
   grid is run.

Note that retrieval-head filtering (`ACCURACY_IMPROVEMENTS.md` §1.3) is the
complement rather than a substitute: `-heads4` matched a 3B scorer's accuracy
at *lower* scoring cost, which buys accuracy headroom without touching `r`.
