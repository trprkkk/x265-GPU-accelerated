#!/bin/bash
# Configure an MSVC build tree (build-msvc/) for the x265 fork.
# Run from anywhere: all paths are resolved relative to this script.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Bundled NASM: place nasm under toolchain/ and it takes precedence over PATH.
if [ -d "$ROOT/toolchain/nasm-2.16.03" ]; then
    export PATH="$ROOT/toolchain/nasm-2.16.03:$PATH"
fi

mkdir -p "$ROOT/build-msvc"
cd "$ROOT/build-msvc"

export CUDA_PATH="${CUDA_PATH:-C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3}"
# NOTE: ENABLE_ASSEMBLY=OFF is the fallback for machines without NASM.
# Drop nasm (e.g. 2.16.03) into toolchain/nasm-2.16.03/ and remove this
# flag for the full-speed build; the stock x265 comparison builds must
# use the same setting as the fork build.
cmake ../x265/source \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DENABLE_ASSEMBLY=OFF \
    -DENABLE_GPU_ME=ON \
    "$@"
echo "configured: $ROOT/build-msvc/x265.exe"
