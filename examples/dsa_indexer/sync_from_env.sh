#!/usr/bin/env bash
# Pull patched files back from the installed vllm into this folder.
set -euo pipefail
SRC="/mnt/remote/guangtaow/conda_env/vllm_glm5_py312/lib/python3.12/site-packages/vllm"
DST="$(cd "$(dirname "$0")" && pwd)/vllm"
[[ -d "$SRC" ]] || { echo "missing $SRC"; exit 1; }
cd "$DST"
mapfile -t FILES < <(find . -type f -name '*.py' -printf '%P\n')
for f in "${FILES[@]}"; do
  install -m 0644 -D "$SRC/$f" "$DST/$f"
  echo "  <-  $SRC/$f"
done
