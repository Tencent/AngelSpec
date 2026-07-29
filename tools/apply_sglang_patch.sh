#!/bin/bash

# Apply sglang patch for AngelSpec.
#
# Usage:
#   ./tools/apply_sglang_patch.sh <path-to-sglang-repo>           # base patch (prefill only)
#   ./tools/apply_sglang_patch.sh --decode <path-to-sglang-repo>  # full patch (prefill + decode)
#
# Please note that this will overwrite all local changes and delete untracked files.

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PATCH_NAME="sglang.patch"
if [ "${1:-}" = "--decode" ]; then
    PATCH_NAME="sglang_decode.patch"
    shift
fi

SGLANG_VERSION="${SGLANG_VERSION:-v0.5.10.post1}"
SGLANG_DIR="$PROJECT_ROOT/docker/sglang/$SGLANG_VERSION"

if [ ! -d "$SGLANG_DIR" ]; then
    echo "Error: sglang version directory not found: $SGLANG_DIR"
    exit 1
fi

SGLANG_COMMIT=$(grep "^ARG SGLANG_COMMIT=" "$SGLANG_DIR/Dockerfile" | cut -d= -f2)

if [ -z "$SGLANG_COMMIT" ]; then
    echo "Error: Could not find SGLANG_COMMIT in $SGLANG_DIR/Dockerfile"
    exit 1
fi

SGLANG_PATH="${1:?Usage: $0 [--decode] <path-to-sglang-repo>}"

PATCH_FILE="$PROJECT_ROOT/tools/patches/sglang/$SGLANG_VERSION/$PATCH_NAME"

if [ ! -f "$PATCH_FILE" ]; then
    echo "Error: Patch file not found: $PATCH_FILE"
    exit 1
fi

echo "SGLANG_VERSION: $SGLANG_VERSION"
echo "SGLANG_COMMIT: $SGLANG_COMMIT"
echo "SGLANG_PATH:   $SGLANG_PATH"
echo "PATCH_FILE:    $PATCH_FILE"
echo ""

if [ ! -d "$SGLANG_PATH" ]; then
    echo "Error: $SGLANG_PATH directory not found"
    exit 1
fi

cd "$SGLANG_PATH"

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: $SGLANG_PATH is not a git repository"
    exit 1
fi

if ! git rev-parse "$SGLANG_COMMIT" > /dev/null 2>&1; then
    echo "Error: Commit $SGLANG_COMMIT not found in $SGLANG_PATH repository"
    exit 1
fi

echo "Resetting to base commit $SGLANG_COMMIT..."
git reset --hard "$SGLANG_COMMIT"
git clean -fd

echo ""
echo "Applying patch..."
git apply "$PATCH_FILE"

echo ""
echo "✓ Patch applied successfully."
echo ""
echo "Files modified:"
git status --short
