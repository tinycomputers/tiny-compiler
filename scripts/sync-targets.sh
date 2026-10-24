#!/usr/bin/env bash
# Fetch the pinned target submodules and check that the artifacts we depend on exist.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
git submodule update --init
for f in targets/tiny-gpu/src/tinygpu/gpu.py \
         targets/tiny-tpu/web/public/tiny_tpu.mjs \
         targets/tiny-tpu/web/public/tiny_tpu.wasm; do
    [ -f "$f" ] || { echo "missing target artifact: $f" >&2; exit 1; }
done
echo "targets OK:"
git submodule status
