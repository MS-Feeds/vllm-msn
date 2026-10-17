#!/usr/bin/env bash
# One-shot fixup for the .venv created earlier from a GPU-less sandbox.
#
# That install ran --torch-backend=auto on a machine with no CUDA device
# visible, so it silently resolved to CPU-only torch. Run this script from
# the actual GPU container so --torch-backend=auto detects the real CUDA
# environment and reinstalls torch/torchvision/torchaudio accordingly.
#
# Also redirects RUSTUP_HOME/CARGO_HOME/UV_CACHE_DIR off $HOME, which has a
# full 10GB NFS quota (confirmed: `quota -s` showed 10240M/10240M) -- without
# this, building the llguidance Rust extension fails with "Disk quota
# exceeded" partway through rustup's toolchain sync.
#
# Usage:
#   bash setup_gpu_env.sh

set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
    echo "ERROR: .venv not found at $(pwd)/.venv"
    echo "Run 'uv venv --python 3.12' first."
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "=========================================="
    echo "Step 0: Installing uv"
    echo "=========================================="
    # Install into this checkout, not ~/.local/bin -- $HOME's NFS quota is
    # already full (10240M/10240M), so writing there risks another "disk
    # quota exceeded" failure even for a small binary. UV_INSTALL_DIR must
    # be exported (not just prefixed on the curl line) so the piped `sh`
    # process -- not just curl -- inherits it.
    export UV_INSTALL_DIR="$(pwd)/.local/bin"
    # The binary install succeeds regardless, but the installer also tries
    # to append a PATH update to ~/.bashrc, which fails on this full quota.
    # We export PATH ourselves right after, so tolerate that failure here.
    curl -LsSf https://astral.sh/uv/install.sh | sh || true
    export PATH="$(pwd)/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "ERROR: uv binary not found at $(pwd)/.local/bin after install."
        exit 1
    fi
    echo ""
fi

echo "=========================================="
echo "Step 1: Redirecting build caches off \$HOME"
echo "=========================================="
export RUSTUP_HOME="$(pwd)/.rustup"
export CARGO_HOME="$(pwd)/.cargo"
export UV_CACHE_DIR="$(pwd)/.cache/uv"
mkdir -p "$RUSTUP_HOME" "$CARGO_HOME" "$UV_CACHE_DIR"
echo "RUSTUP_HOME=$RUSTUP_HOME"
echo "CARGO_HOME=$CARGO_HOME"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo ""

echo "=========================================="
echo "Step 2: Reinstalling torch for this machine's CUDA setup"
echo "=========================================="
# shellcheck disable=SC1091
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto \
    --reinstall-package torch \
    --reinstall-package torchvision \
    --reinstall-package torchaudio

echo ""
echo "=========================================="
echo "Step 3: Verifying CUDA is picked up"
echo "=========================================="
python -c "
import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('ERROR: torch.cuda.is_available() is False -- check that this container actually has GPU access (nvidia-smi, /dev/nvidia*) and that --torch-backend=auto detected it.')
print('CUDA device count:', torch.cuda.device_count())
"

echo ""
echo "✓ Environment ready. Next: run the DeepSeek-V3.2 experiment, e.g.:"
echo ""
echo "  cd examples/dsa_indexer"
echo "  DEEPSEEK_V32_MODEL_PATH=/import/ml-sc-scratch6/chenw/DeepSeek-V3.2 \\"
echo "    VLLM_INSTALL_PATH=$(pwd)/vllm \\"
echo "    bash run_v32_experiment.sh --server-only"
