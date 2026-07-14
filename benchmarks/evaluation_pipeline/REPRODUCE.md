# Reproduction Steps

Compiled from an actual from-scratch environment rebuild on `node-0`
after a GPU wipe (2026-07-14) — every step and gotcha below was hit for
real that session, not written speculatively.

## 1. Conda environment

If this is a fresh node, conda's shell hook may need loading before
`conda activate` works (`CondaError: Run 'conda init' before 'conda
activate'`):

```bash
source /opt/conda/etc/profile.d/conda.sh
conda create -n vllm-ablation python=3.10 -y   # skip if the env already exists
conda activate vllm-ablation
```

If `/opt/conda/etc/profile.d/conda.sh` doesn't exist, conda itself isn't
installed on this node yet — install Miniconda first
(`curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash miniconda.sh -b -p /opt/conda`,
or `-p ~/miniconda3` without root).

## 2. PyTorch

Check `nvidia-smi` for the driver's max supported CUDA version first and
match the `--index-url` tag accordingly (cu126 below is what this repo's
docs were validated against — adjust if your driver reports a different
CUDA version):

```bash
nvidia-smi
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## 3. This repo's vLLM fork (editable install)

**Do not `pip install vllm` from PyPI.** A stock PyPI vLLM build pulls
in a `torchcodec` dependency (for video-multimodal support) that this
fork's own `vllm/multimodal/video.py` doesn't need — its `cv2`/`av`
imports are optional and gracefully handled. Installing the wrong
package caused an unrelated `libavutil.so.*: cannot open shared object
file` failure at `from vllm import LLM` time the first time this was
attempted, tracing all the way through `torchcodec`'s FFmpeg-shared-lib
loading, for a code path (video decoding) this pipeline never uses.

```bash
cd ~/vllm-msn
VLLM_USE_PRECOMPILED=1 pip install -e .
```

## 4. OS-level multimodal deps (cheap insurance)

Matches the known-working `gemma4_moe_benchmarks/Dockerfile`, which
installs these explicitly:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libgl1
```

## 5. Remaining Python deps

```bash
pip install transformers datasets
```

## 6. Model checkpoints

Set `HF_TOKEN` and check whether the previous cache actually survived
(a wipe may take `/scratch` too):

```bash
export HF_TOKEN=<your token>
ls -la /scratch/hf_cache/hub/ 2>/dev/null
df -h /scratch
```

If the checkpoint directories are missing/empty, download fresh:

```bash
hf download google/gemma-4-26B-A4B-it --cache-dir /scratch/hf_cache
hf download google/gemma-4-26B-A4B-it-assistant --cache-dir /scratch/hf_cache
```

**Gotcha**: `hf download --cache-dir X` places snapshots directly under
`X/models--.../snapshots/<hash>/`, **not** `X/hub/models--.../snapshots/<hash>/`
— that extra `hub/` nesting is specific to `HF_HOME`-based lazy fetching
via `transformers`/`huggingface_hub`'s default cache resolution, not to
this CLI command. Find the real path with:

```bash
ls -la /scratch/hf_cache/models--google--gemma-4-26B-A4B-it/snapshots/*/
du -sh /scratch/hf_cache/models--google--gemma-4-26B-A4B-it/
```
Confirm `*.safetensors` files and `model.safetensors.index.json` are
present and the total size is tens of GB (49G was the observed size for
the base model on 2026-07-14) — `AutoTokenizer.from_pretrained()` can
succeed even when only the small config/tokenizer files exist and the
weight shards are missing, since it never touches them; that failure
only surfaces later, inside vLLM's own model loader
(`RuntimeError: Cannot find any model weights with ...`).

## 7. Environment variables

`../gemma4_moe_benchmarks/.env_exports.sh` is the single source of truth
for `GEMMA4_MODEL_PATH`/`GEMMA4_ASSISTANT_MODEL_PATH`/`HF_TOKEN`/conda
activation — this pipeline does not duplicate it. (There is also a
`.env_exports.sh` directly in this directory,
`evaluation_pipeline/.env_exports.sh` — that one is an unfilled stub,
not wired into `run_experiments.sh` or `run_pipeline.py` at all; don't
confuse the two.)

```bash
source ../gemma4_moe_benchmarks/.env_exports.sh
```

Update the two model-path lines in that file whenever the checkpoints
are re-downloaded, since the snapshot hash changes each time (unless
HF's CDN happens to return the same revision).

## 8. Preflight

```bash
python3 -c "import vllm, transformers; print('OK')"
```
This is also run automatically inside `run_experiments.sh` before it
launches anything, with a clearer error message if it fails.

## 9. Running an experiment

Two independent suites share `run_pipeline.py`, selected via `--suite`
(default `spec`) — see `EXPERIMENT_PLAN.md` (spec-decode/k sweep) and
`EXPERIMENT_PLAN_MAX_NUM_SEQS.md` (`max_num_seqs` batch-size sweep).

```bash
cd benchmarks/evaluation_pipeline

# cheapest possible smoke test before committing to a full sweep:
./run_experiments.sh --suite batch B003 --reps 1 --datasets gpqa_diamond

# full sweeps:
./run_experiments.sh --all                       # spec suite (S0xx)
./run_experiments.sh --suite batch --all          # batch suite (B0xx)
```

Results land in `results/` — one JSON per (experiment × dataset × rep)
plus `results/all_runs.csv`. Then:
```bash
python3 analyze_results.py   # -> results/summary.md
```

## Troubleshooting quick-reference

| Symptom | Cause | Fix |
|---|---|---|
| `CondaError: Run 'conda init' before 'conda activate'` | Shell hook not loaded this session | `source /opt/conda/etc/profile.d/conda.sh` first |
| `OSError: libavutil.so.*: cannot open shared object file` at `from vllm import LLM` | Stock PyPI `vllm` installed instead of this fork (pulls in unneeded `torchcodec`) | Reinstall via step 3 (`pip uninstall vllm` first if needed) |
| `Repo id must be in the form 'repo_name' or 'namespace/repo_name'` | `GEMMA4_MODEL_PATH` points at a local directory that doesn't exist, so `huggingface_hub` falls through to treating it as a Hub repo id | Re-download (step 6) and fix the path |
| `RuntimeError: Cannot find any model weights with ...` | Snapshot directory exists but only has config/tokenizer files, not `*.safetensors` | Re-download with `hf download` (step 6); verify with `du -sh` |
| `hf download`'s reported path doesn't match what `run_pipeline.py` looked for | The `hub/` nesting mismatch — see step 6's gotcha | Use the path `hf download` actually printed, not the old `.../hub/...` convention |

## Expected runtime / hardware

TBD — not yet benchmarked end-to-end post-rebuild. Fill in once a full
suite run completes (GPU model, wall-clock time per suite, any
memory-pressure notes).
