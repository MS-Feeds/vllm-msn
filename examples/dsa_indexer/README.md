# DSA lightning-indexer logit-distribution experiment

**Goal:** capture the raw logit distribution vLLM's **lightning indexer**
(the sparse-attention KV-token selector) produces, per layer per phase
(prefill/decode) — Stream B only. This data is eventually meant to inform a
radix-sort-vs-bitonic-sort top-k comparison, but that comparison (Streams
A/C/D) is future work — out of scope for now, don't enable those streams.

**Model:** DeepSeek-V3.2. The indexer (`Indexer` in
[`deepseek_v2.py`](../../vllm/model_executor/models/deepseek_v2.py), gated
by `hasattr(config, "index_topk")`) delegates to `sparse_attn_indexer()` in
[`sparse_attn_indexer.py`](../../vllm/model_executor/layers/sparse_attn_indexer.py).
GLM-5.1 shares the same code path (see `run02`-`run04` baseline data below).

Checkpoint: `DEEPSEEK_V32_MODEL_PATH` (default
`/import/ml-sc-scratch6/chenw/DeepSeek-V3.2` — 61 layers, `index_topk=2048`,
FP8). Override if running against a different copy.

## Running it

```bash
cd /import/snvm-sc-scratch1/chenw/vllm-msn
source .venv/bin/activate               # must be active - script shells out to bare `python3`

cd examples/dsa_indexer
bash run_v32_experiment.sh --dry-run     # sanity-check the plan, no side effects
bash run_v32_experiment.sh --server-only # launch + health-check only, no eval sweep
bash run_v32_experiment.sh               # full run: launch, eval sweep, shutdown, verify
```

Override `DEEPSEEK_V32_MODEL_PATH` / `VLLM_INSTALL_PATH` / `TP_SIZE` /
`EVAL_DATASETS` as env vars if the defaults don't fit. `--eval-only` skips
straight to the eval sweep against an already-running server.

### Settings that matter (don't change without reason)

| Setting | Value | Why |
|---|---|---|
| `INDEXER_STREAMS` | `B` | Only stream needed right now; A/C/D add a `torch.topk()` call + radix sort per invocation for a future comparison this stage doesn't need. |
| `INDEXER_LOGIT_DUMP_EVERY` | `5000` | Dumps to disk periodically, not just at exit. vLLM's worker teardown doesn't reliably give a clean interpreter exit — a full run once lost 100% of its data (zero `.npz` files) with this at `0`. |
| `--enforce-eager` | always on | `_indexer_logger.record()` does host-syncing ops (`.item()`, `.cpu()`) that CUDA graph capture forbids and would silently stop recording after the first capture anyway. |
| `INDEXER_IS_EXPERT_ROUTING` | `0` | Keeps the raw `[-50,50]` logit range. `1` switches to the `[0,1]` post-softmax range used by an unrelated MoE-routing experiment (see "Not in scope" below) — wrong range for this data. |

## Output & analysis

Each run writes to `indexer_logits/<run_id>/*.npz` (one per TP rank),
`logs/eval_<run_id>.jsonl`, and `logs/server_<run_id>.log`.

`deepseek_v32_20260706_212741` is the reference example of a clean run:
30/30 AIME prompts, 149.3M prefill + 30.1B decode raw-logit samples across
all 61 layers, zero values falling outside the `[-50,50]` capture range.

```bash
cd examples/dsa_indexer
python3 tools/plot_indexer_logits.py --inputs 'indexer_logits/<run_id>'/*.npz \
  --out figures/<run_id> --aggregate
python3 tools/threshold_analysis.py indexer_logits/<run_id>/indexer_logits_rank0.npz \
  > figures/<run_id>/threshold_recommendations.txt
cat figures/<run_id>/summary.txt
```

Pure `numpy`/`matplotlib` on saved `.npz` — no GPU needed, runs anywhere
with the repo checked out. `matplotlib` isn't a vLLM dependency, so install
it explicitly: `uv pip install --python .venv/bin/python3 matplotlib` (bare
`pip` has no binary in a `uv`-created venv and will silently install into
whatever other Python is on `PATH` instead).

With `INDEXER_STREAMS=B`, only the `stream_b_*.png` plots and
`summary.txt`'s Stream B section have real content — the A/C/D plots and
sections legitimately no-op on zero-count data. That's expected, not a
capture failure.

## Not in scope: MoE expert-routing

Earlier iterations of this experiment instrumented MoE expert-routing
top-k (`fused_topk_bias()`) instead of the lightning indexer — a different,
unrelated sparsity mechanism. That instrumentation never produced a
committed dataset and has been removed; the committed `run02`-`run04` data
is confirmed lightning-indexer/GLM-5.1, not MoE. Recover from git history
if ever needed.

## Files

`vllm_patches/` mirrors an installed `vllm/` site-packages tree (source of
truth for the patches this experiment applies). Named `vllm_patches/`, not
`vllm/`, on purpose — a same-named subdirectory here would shadow the real
installed package when running scripts from this folder.

| Path (→ `vllm/` at destination) | Purpose |
|---|---|
| `_indexer_logger.py` | Env-configured recorder: histograms + moments per layer per phase, periodic + atexit/SIGTERM dump. |
| `model_executor/layers/sparse_attn_indexer.py` | Production file + 1 import + 2 `_indexer_logger.record(...)` calls. Nothing else. If regenerating, always start from the current production file — don't resurrect the old diff in `INDEXER_LOGIT_EXPERIMENT.md` §12.3, which predates this fork's SM80 fallback and will reintroduce a hard DeepGEMM requirement. |

```bash
VLLM_INSTALL_PATH=/path/to/site-packages/vllm bash sync_to_env.sh    # push patches to installed env
VLLM_INSTALL_PATH=/path/to/site-packages/vllm bash sync_from_env.sh  # pull installed copies back here
pip install --force-reinstall --no-deps vllm==0.19.0                 # revert to stock vllm
```
