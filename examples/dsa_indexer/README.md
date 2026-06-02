# vllm patches for the indexer-logit experiment

Source of truth for every vllm file we've modified, mirroring the directory layout under
`/mnt/remote/guangtaow/conda_env/vllm_glm5_py312/lib/python3.12/site-packages/vllm/`.

## Files

| Path under `vllm/` | Status | Purpose |
|---|---|---|
| `_indexer_logger.py` | new | env-parsed recorder; histograms + moments per layer per phase; atexit/SIGTERM dump |
| `model_executor/layers/sparse_attn_indexer.py` | modified | +1 import, +2 `_indexer_logger.record(...)` calls after each `fp8_*_mqa_logits` call |

The patch to `sparse_attn_indexer.py` is documented as a diff in `INDEXER_LOGIT_EXPERIMENT.md` §12.3.

## Workflow

Edit in this folder first, then push to the installed env:

```bash
bash /mnt/remote/guangtaow/vllm_patches/sync_to_env.sh
```

To pull the installed-env copies back here (e.g. after a `pip install --force-reinstall`):

```bash
bash /mnt/remote/guangtaow/vllm_patches/sync_from_env.sh
```

To revert to stock vllm:

```bash
pip install --force-reinstall --no-deps vllm==0.19.0
```
