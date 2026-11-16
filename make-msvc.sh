#!/bin/bash
# Build the x265 fork in build-msvc/ (run configure-msvc.sh first).
# Usage: make-msvc.sh [clean] [extra ninja args...]
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/build-msvc"
if [ "$1" = "clean" ]; then
    ninja clean
    shift
fi
ninja "$@"
