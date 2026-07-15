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

Three independent suites share `run_pipeline.py`, selected via `--suite`
(default `spec`) — see `EXPERIMENT_PLAN.md` (spec-decode/k sweep),
`EXPERIMENT_PLAN_MAX_NUM_SEQS.md` (`max_num_seqs` batch-size sweep), and
`EXPERIMENT_PLAN_MNS_SPEC_CROSS.md` (`max_num_seqs` x MTP `k` cross-sweep).

```bash
cd benchmarks/evaluation_pipeline

# cheapest possible smoke test before committing to a full sweep:
./run_experiments.sh --suite batch B003 --reps 1 --datasets gpqa_diamond

# full sweeps:
./run_experiments.sh --all                       # spec suite (S0xx)
./run_experiments.sh --suite batch --all          # batch suite (B0xx)
./run_experiments.sh --suite cross --all          # cross suite (X0xx)
```

Results land in `results/` — one JSON per (experiment × dataset × rep)
plus `results/all_runs.csv`. Then:
```bash
python3 analyze_results.py   # -> results/summary.md
```

## 10. (Optional) Nsight profiling for one experiment

`run_pipeline.py --nsight-exp EXP_ID` (or `run_experiments.sh
--nsight-exp EXP_ID`, forwarded through) brackets *only that one
experiment's* `generate()` calls with `cudaProfilerStart()`/`Stop()` and
an NVTX range — it does not launch Nsight itself. This exists to narrow
a capture to one experiment when running a whole suite (or an `--exp`
list) in a single process invocation, since `nsys`/`ncu` otherwise
capture everything the wrapped process does.

### Additional setup this requires (not needed for normal runs)

1. **Install Nsight Systems/Compute.** Not guaranteed to already be
   present — this pipeline's vLLM install deliberately used
   `VLLM_USE_PRECOMPILED=1` to avoid needing a full CUDA Toolkit/compiler
   (see step 3), and `nsys`/`ncu` are separate packages from that:
   ```bash
   sudo apt-get install -y nsight-systems-cli nsight-compute
   # or standalone installers from developer.nvidia.com if apt access
   # isn't available on this node
   ```
   Verify: `nsys --version`, `ncu --version`.

2. **Check GPU performance-counter permissions.** NVIDIA's driver
   restricts hardware performance counter access to admin/root by
   default on many systems (`NVreg_RestrictProfilingToAdminUsers`),
   especially shared/cloud GPU nodes — plausible to hit the same kind of
   permissions wall as the `apt-get`/ffmpeg step earlier. Check with:
   ```bash
   cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
   ```
   If this is `1` and you don't have root, `ncu` (and some `nsys`
   counter collection) will fail with `ERR_NVGPUCTRPERM` — this may not
   be fixable without node-admin access on a managed compute node like
   `node-0`.

3. **Account for vLLM's multi-process engine.** The V1 engine runs
   actual model execution in a separate `EngineCore` subprocess (spawned
   via Python multiprocessing's `spawn` method — visible in vLLM's own
   startup log as `(EngineCore pid=...)`). `nsys` must be told to follow
   child processes, or it will trace an empty parent process instead of
   where the CUDA kernels actually run:
   ```bash
   nsys profile --trace-fork-before-exec=true ...
   ```

4. **Full invocation** combining all of the above with `--nsight-exp`
   (using `nsys`'s CUDA-Profiler-API capture-range mode so the trace
   window matches exactly the bracketed experiment, not the whole
   multi-experiment run):
   ```bash
   nsys profile \
       --capture-range=cudaProfilerApi \
       --capture-range-end=stop \
       --trace-fork-before-exec=true \
       -o report_B003 -- \
     python3 run_pipeline.py --suite batch --exp B001,B002,B003 --reps 1 --nsight-exp B003
   ```

5. **`ncu` (kernel-level counters) is much heavier** — kernel-replay
   profiling can be 10-100x slower, so only run it against the smallest
   practical workload (`--reps 1 --datasets gpqa_diamond`, the shortest
   dataset), not a full sweep. It doesn't use `cudaProfilerApi`
   capture-range the same way `nsys` does; instead, filter to the NVTX
   range `--nsight-exp` already pushes:
   ```bash
   ncu --set full --nvtx --nvtx-include "nsight_B003/" -o ncu_report -- \
     python3 run_pipeline.py --suite batch --exp B003 --reps 1 --datasets gpqa_diamond --nsight-exp B003
   ```

See the "analyze pros and cons of using Nsight for metrics" discussion
in chat history (2026-07-14) for why this is meant as an occasional
ground-truth validation of `hardware_metrics.py`'s analytical MFU/MBU
formulas, not a per-sweep-row metrics mechanism — overhead and lack of
CSV integration make it impractical to run on every experiment.

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
