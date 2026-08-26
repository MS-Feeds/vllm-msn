# FLOP Calculation Notesheet — Speculative Prefill Proof of Concept

Backing detail for the "Proof of Concept: A Basic FLOP Calculation" slide.
Methodology matches `flops_model.py` exactly, so these numbers stay
consistent with the rest of the deck.

---

## Models

| | Target | Speculator |
|---|---|---|
| Checkpoint | Llama-3.1-8B-Instruct | Llama-3.2-1B-Instruct |
| Layers | 32 | 16 |
| Hidden size | 4096 | 2048 |
| Heads / KV heads | 32 / 8 | 32 / 8 |
| head_dim | 128 | 64 |
| Intermediate size | 14336 | 8192 |

## Conventions (from `flops_model.py`)

- 1 MAC = 2 FLOPs.
- GQA shrinks the **KV cache**, not the attention arithmetic — every one of
  `num_heads` query heads still does a full `head_dim` dot product against
  its group's K/V. Attention FLOPs scale with `num_heads`; only the QKV
  *projection* scales with `num_kv_heads`.
- `lm_head` is charged once per turn (one logits row), never per token.
- RMSNorm / RoPE / softmax / residual excluded — all <1% of the total at
  these shapes.

---

## The two cost terms

**Linear** (QKV + O-proj + MLP) — constant per token, independent of
context length:

```
linear/token = 2·hidden·head_dim·(num_heads + 2·num_kv_heads)   [QKV]
             + 2·num_heads·head_dim·hidden                       [O-proj]
             + 6·hidden·intermediate                              [MLP, SwiGLU]
```
...summed over all layers.

**Attention** — causal self-attention over `L` tokens visits
`L(L+1)/2 ≈ L²/2` query-key pairs:

```
attn(L) = num_layers · 4·num_heads·head_dim · L(L+1)/2
```

## Computed coefficients

| | Target (8B) | Speculator (1B) | Ratio |
|---|---:|---:|---:|
| Linear, GFLOP/token | 13.9586 | 1.9462 | 7.17x |
| Attention coefficient (`num_layers·4·num_heads·head_dim`) | 524,288 | 131,072 | **4.0x exact** |

The 4.0x is architectural, not approximate: target has 2x the layers and
2x the head_dim of the speculator, 2×2=4.

---

## Worked example — L = 128,000 (Llama-3.1's context ceiling)

`L(L+1)/2 = 128,000 × 128,001 / 2 = 8,192,064,000`

| | Attention | Linear | `lm_head` | **Total** |
|---|---:|---:|---:|---:|
| Target (dense, = M000) | 4,295.0 TFLOP | 1,786.7 TFLOP | ~0.001 (negl.) | **≈6,081.7 TFLOP (~6.08 PFLOP)** |
| Speculator (dense prefill) | 1,073.8 TFLOP | 249.1 TFLOP | ~0.0005 (negl.) | **≈1,322.9 TFLOP** |

**Cross-check (independent validation):** at L=77,000 — this project's
*other* reference context, used elsewhere in the repo — the same formula
gives the speculator's dense prefill as ≈538.5 TFLOP, matching the figure
already written into `flops_model.py`'s own docstring ("~538 TFLOP")
almost exactly. That's why the formula is trusted before being evaluated
at 128,000, where no independent published figure exists to check against.

---

## Pruning to keep rate k

Only `k·L` tokens reach the target's prefill, so:

- Linear term scales with **k** — `linear(kL) = k · linear(L)`
- Attention term scales with **k²** — `attn(kL) ≈ k² · attn(L)`

```
pruned_target(k) = k · linear_target(L)  +  k² · attn_target(L)
```

This is the whole reason pruning has outsized leverage at long context:
cutting to 20% of tokens cuts attention cost to 4% (0.2²), not 20%.

## Total system cost

The speculator always scores the *full* context (it has to see everything
to decide what to keep), so its cost is fixed regardless of k:

```
system(k) = speculator_dense_prefill (≈1,322.9 TFLOP, constant)
          + pruned_target(k)
```

| k | pruned target | + speculator | system total | ÷ M000 (6,081.7) |
|---:|---:|---:|---:|---:|
| 80% | 4,178.2 | 1,322.9 | 5,501.0 | **90.5%** — barely a win |
| 60% | 2,618.2 | 1,322.9 | 3,941.1 | **64.8%** |
| 40% | 1,401.9 | 1,322.9 | 2,724.7 | **44.8%** |
| 20% | 529.1 | 1,322.9 | 1,852.0 | **30.5%** — ~3.3x reduction |

---

## The long-context asymptote

As L→∞, `system(k,L)/M000(L)` converges to a value that no longer depends
on L at all:

```
R(k, L→∞) = (speculator attn coeff / target attn coeff) + k²
          = 1/4 + k²
```

The 1/4 is exact (`131,072/524,288`) — purely a function of the two
models' architecture, not a fitted or approximate number.

| k | Asymptote (¼+k²) | Measured at L=128,000 |
|---:|---:|---:|
| 80% | 89.0% | 90.5% |
| 60% | 61.0% | 64.8% |
| 40% | 41.0% | 44.8% |
| 20% | 29.0% | 30.5% |

L=128,000 already sits close to the asymptote because it's well past the
target's own linear/attention crossover point
(`L* = 2·13.9586e9/524,288 ≈ 53,000` tokens) — beyond that, attention's
quadratic term dominates the target's own cost, and the whole
system-vs-baseline ratio stops caring how much longer the context gets.

---

## Key takeaway

- The compute win is **real but keep-rate dependent**. At k=80% it's
  barely a win (90.5%) because the speculator's fixed ~1,323 TFLOP tax
  eats most of what pruning the target saves.
- At k=20%, total system compute drops to 30.5% of dense — a ~3.3x
  reduction, even after paying for a whole second model.
- This is the **theoretical ceiling**, single-turn, proof-of-concept only.
  It is not the measured result — the multi-turn SPARSE mechanism later in
  this project measures ~120% of M000's FLOPs, a worse outcome, because of
  additional architectural overhead this simple model doesn't capture.
  Don't present this table as what was achieved; present it as what the
  core idea is capable of in principle.
