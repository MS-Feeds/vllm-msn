#!/usr/bin/env bash
# Push patches from this folder into the installed vllm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/vllm_patches"
DST="${VLLM_INSTALL_PATH:-/mnt/remote/guangtaow/conda_env/vllm_glm5_py312/lib/python3.12/site-packages/vllm}"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: Source directory not found: $SRC"
    exit 1
fi

if [[ ! -d "$DST" ]]; then
    echo "ERROR: Destination vllm installation not found: $DST"
    echo "Set VLLM_INSTALL_PATH environment variable to override."
    exit 1
fi

echo "Syncing patches from $SRC to $DST..."

cd "$SRC"
mapfile -t FILES < <(find . -type f -name '*.py' -printf '%P\n')

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "WARNING: No Python files found in $SRC"
    exit 0
fi

for f in "${FILES[@]}"; do
    install -m 0644 -D "$SRC/$f" "$DST/$f"
    echo "  -> $DST/$f"
done

echo ""
echo "✓ Synced ${#FILES[@]} files successfully."
echo ""
echo "Patched files:"
echo "  - _indexer_logger.py (per-layer, per-phase logit recorder)"
echo "  - model_executor/layers/sparse_attn_indexer.py (records lightning-indexer"
echo "    attention logits; used by GLM-5.1 and DeepSeek-V3.2's Indexer class)"
echo ""
echo "To verify, run:"
echo "  python -c 'from vllm.model_executor.layers.sparse_attn_indexer import sparse_attn_indexer; print(\"OK\")'"
