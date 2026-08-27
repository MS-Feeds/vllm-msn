# Related sparse-KV / long-context methods

Speaker notes for the deck's Literature Review slide. One or two lines each,
plus how the method differs from what this project does — the slide frames
these as context for what this pipeline deliberately does *not* implement.

## The five

### H2O — Heavy-Hitter Oracle
Observes that a small, stable set of "heavy hitter" tokens carries most of
the attention mass, and **evicts** KV entries during generation using
accumulated attention scores, keeping heavy hitters plus a recent window.

*Differs here:* H2O shrinks the cache by throwing entries away. This project
never evicts — every token's KV stays resident and the gather only masks
which entries are *read*, so a block skipped at turn 2 can be re-selected at
turn 5.

### StreamingLLM
Identifies **attention sinks**: the first few tokens absorb disproportionate
attention regardless of content, and dropping them collapses generation.
Keeping those few initial tokens plus a sliding recent window lets a model
stream indefinitely without fine-tuning.

*Differs here:* it enables unbounded streaming rather than extending usable
context — old middle tokens are gone. But the sink result is load-bearing
for this pipeline: the wrapper-span union in `predict_scbench.py` exists
because dropping the `<|begin_of_text|>` block reproduced exactly the
fluent-start-then-repetition-loop failure StreamingLLM describes.

### Quest
Query-aware page selection at decode time: keeps the whole KV cache but
scores each page by an upper bound on its attention (from per-page min/max
key vectors) and loads only the top-K pages for that query.

*Differs here:* closest relative of the three — same "keep everything,
restrict reads" shape. Quest derives its estimate from the KV cache itself
per decode step; this project derives it from a **separate small model's**
attention, once per turn.

### KVzip
Query-agnostic compression: scores KV pairs by how well they let the model
reconstruct the original context, then evicts the rest. Because the scoring
does not depend on any particular question, the compressed cache is meant to
be reused across many later queries.

*Differs here:* directly addresses the multi-turn reuse problem, but by
eviction — once compressed, dropped content cannot come back. This pipeline's
KEEP mode rescores the full ledger every turn precisely to avoid that.

### HeadKV
Allocates the cache budget **per attention head** rather than uniformly,
giving more capacity to heads that matter for retrieval and reasoning and
less to the rest.

*Differs here:* orthogonal, and the closest to something this project could
adopt. `ACCURACY_IMPROVEMENTS.md` §1.3 found that restricting the 1B
speculator's scoring to 4 retrieval heads closed 51% of the estimator gap at
*lower* cost — the same head-importance idea applied to the scorer rather
than to the budget.

## What this project does instead

| axis | these methods | this pipeline |
|---|---|---|
| cache | evict (H2O, KVzip, StreamingLLM) or keep (Quest) | always keep — nothing is ever evicted |
| what shrinks | KV memory, or KV bytes read | attention compute and KV bandwidth, prefill and decode |
| selection signal | the target's own attention or KV statistics | a separate 1B model's attention, recomputed per turn |
| setting | single long prompt | multi-turn, one persistent cache growing across turns |

The multi-turn framing is the gap: these methods are evaluated on one long
prompt, and the question here is whether token preselection stays cheap turn
after turn against a context that only grows.

## Before citing

The technical summaries above are ones I'm confident in. **Verify the arXiv
IDs and publication venues before they go on a slide** — particularly for
KVzip and HeadKV, which are the two most recent and the two I'd least trust
myself to have exactly right. H2O and StreamingLLM are the well-established
pair; Quest sits between.
