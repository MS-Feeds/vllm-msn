# Algorithm Notesheet — Speculative Prefill (Algorithm 1)

Companion to the "Algorithm" slide. Explains Algorithm 1 from Liu, Chen &
Zhang (2025), *Speculative Prefill: Turbocharging TTFT with Lightweight and
Training-Free Token Importance Estimation* (arXiv:2502.02789, ICML 2025),
line by line.

---

## Inputs

```
Require: Base model M, speculator S, look-ahead steps N,
         batch of mixed requests B,
         base model QKV cache C_b, speculator KV cache C_s
```

- **M** — the expensive target model (e.g. Llama-3.1-405B-Instruct-FP8).
- **S** — the cheap speculator model that decides which tokens survive.
- **N** — how many extra decode steps the speculator takes before scoring
  (this project uses N=8, matching the paper's evaluated default).
- **B** — a batch that mixes prefill requests (new prompts) and decode
  requests (continuing generations) — vLLM schedules both together.
- **C_b, C_s** — each model's own KV cache. The base model's cache is
  untouched by anything the speculator does.

---

## Step by step

### 1. Split the batch (line 1)

```
B_p, B_d ← split_prefill_decode_requests(B)
```

Only prefill requests (`B_p`) go through speculation — decode requests
(`B_d`) are set aside untouched and rejoined at the very end (line 19).
Speculation is a **prefill-only** mechanism.

### 2. Look-ahead loop (lines 3–7) — see §3.2.1

```
for i = 1 to N do
    B_p' ← model_forward(S, B_p, C_s, store_q=True)
    B_p  ← update_requests(B_p, B_p')
    B_p  ← check_for_eos(B_p)
end for
```

The speculator decodes **N** extra tokens per request, one forward pass at
a time. `store_q=True` is the important bit: each pass's **query vectors**
are saved (not thrown away like a normal decode step). This is what lets
step 4 use the *new* tokens' attention instead of only the prompt's last
token — which is what fixes the sink/proximity bias (see the "Look-Ahead"
slide). `check_for_eos` drops any batch entry that hit end-of-sequence
early, so a short completion doesn't force N steps on everyone.

### 3. Tensor-parallel query/key gather (lines 8–10)

```
if is_tensor_paralleled() then
    tp_gather_qk(C_s)
end if
```

Under tensor parallelism, each GPU only holds a shard of the speculator's
Q/K. This step gathers the full tensors before scoring — a distributed-
systems detail, not part of the core algorithm, but necessary for the
405B/70B multi-GPU runs the paper reports.

### 4. Compute the raw attention score (lines 11–12) — see §3.2

```
Q, K ← retrieve_qk(B_p, C_s)
A    ← compute_attention_score(Q, K)
```

This is the paper's core formula:

```
a_ij = Softmax(Q_{M+j} K^T)_i,   0 ≤ i < M,   0 ≤ j < N
```

`M` = context length, `N` = look-ahead steps. `a_ij` is how much the
*j*-th look-ahead token attends to prompt token *i*. Doing this for every
layer produces a tensor of shape **[N, L, S, H]** — look-ahead steps ×
layers × sequence length × heads.

### 5. Aggregate into one importance score per token (line 14) — see §3.2.2

```
A ← aggregate_attention_score(A)
```

Collapses `[N, L, S, H]` down to one scalar per prompt token:

- **max over (L, H)** — the maximum across layers and heads. Lets a single
  strongly-informed head/layer decide a token matters, instead of being
  diluted by averaging in every uninformative head.
- **mean over N** — averaged across the look-ahead steps, so no single
  decoded token dominates the importance signal.

### 6. Chunk, pool, and select (line 16) — see §3.2.3

```
T ← chunk_select_from_smoothed_attention(A)
```

Two things happen here, in order:

1. **1D average pooling** across the token-importance sequence, to smooth
   out artifacts at chunk boundaries.
2. **Chunking**: the (now-smoothed) context is split into contiguous
   chunks, each chunk's score is the average of its tokens, and the
   top-K chunks by score are kept. Chunking exploits *locality* — tokens
   near each other tend to matter for similar reasons.

### 7. Restore original position ids (line 18) — see §3.2.4

```
P ← restore_pos_ids(T, B_p)
```

The kept tokens `T` are a non-contiguous subset of the original prompt.
The base model must see their **original** positions, not a renumbered
0..k sequence — otherwise RoPE (or any position-dependent mechanism)
scrambles the model's sense of where each token actually sat. See the
"Restoring Position IDs" slide for the paper's worked example.

### 8. Merge and run the base model (lines 19–20)

```
B ← merge_requests(T, P, B_p, B_d)
Return model_forward(M, B, C_b)
```

The pruned, position-corrected prefill requests are recombined with the
untouched decode requests (`B_d` from step 1), and the whole batch goes
through **one** forward pass of the expensive base model. Everything
before this line only ever touched the cheap speculator.

---

## What this buys you

- The base model's prefill is now over `k·L` tokens instead of `L` — both
  its attention (quadratic in length) and its MLP (linear in length)
  shrink. This is the mechanism behind the "Basic FLOP Calculation" slide
  and notesheet (`FLOP_CALC_NOTES.md`) — same project, same style of
  accounting, applied to the multi-turn setting instead of single-turn.
- Nothing about Algorithm 1 requires fine-tuning the base model — the
  speculator does all the work, and the base model just receives a
  shorter, position-corrected prompt it already knows how to handle.
- Algorithm 1 is single-turn and stateless between calls. This project's
  own contribution starts exactly where this algorithm ends: what happens
  when the same base model has to run this every turn, against a KV cache
  that has to persist and grow instead of being thrown away — see the
  "Multi-Turn" slides.
