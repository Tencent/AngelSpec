#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

SGLANG_VERSION="${SGLANG_VERSION:-v0.5.10.post1}"
SGLANG_DIR="$PROJECT_ROOT/docker/sglang/$SGLANG_VERSION"

if [ ! -d "$SGLANG_DIR" ]; then
    echo "Error: sglang version directory not found: $SGLANG_DIR"
    exit 1
fi

SGLANG_COMMIT=$(grep "^ARG SGLANG_COMMIT=" "$SGLANG_DIR/Dockerfile" | cut -d= -f2)
SGLANG_FOLDER_NAME="${SGLANG_FOLDER_NAME:-$(grep "^SGLANG_FOLDER_NAME=" "$SCRIPT_DIR/build_conda.sh" | cut -d= -f2 | tr -d '"')}"

if [ -z "$SGLANG_COMMIT" ]; then
    echo "Error: Could not find SGLANG_COMMIT in $SGLANG_DIR/Dockerfile"
    exit 1
fi

if [ -z "$SGLANG_FOLDER_NAME" ]; then
    echo "Warning: SGLANG_FOLDER_NAME not found in build_conda.sh, using default '_sglang'"
    SGLANG_FOLDER_NAME="_sglang"
fi

if [[ "$SGLANG_FOLDER_NAME" = /* ]]; then
    SGLANG_PATH="$SGLANG_FOLDER_NAME"
else
    SGLANG_PATH="$PROJECT_ROOT/$SGLANG_FOLDER_NAME"
fi

echo "SGLANG_VERSION: $SGLANG_VERSION"
echo "SGLANG_COMMIT: $SGLANG_COMMIT"
echo "Using folder: $SGLANG_PATH"

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

# Check for uncommitted or untracked changes
has_uncommitted=false
if ! git diff --quiet HEAD 2>/dev/null; then
    has_uncommitted=true
fi
if [ -n "$(git ls-files --others --exclude-standard)" ]; then
    has_uncommitted=true
fi

if [ "$(git rev-parse HEAD)" = "$(git rev-parse $SGLANG_COMMIT)" ]; then
    echo "Error: No commits after $SGLANG_COMMIT."
    if [ "$has_uncommitted" = true ]; then
        echo ""
        echo "You have uncommitted changes:"
        git status --short
        echo ""
        echo "Please commit them first:"
        echo "  cd $SGLANG_PATH && git add -A && git commit -m 'your message'"
    else
        echo "Please make and commit your changes in $SGLANG_PATH first."
    fi
    exit 1
fi

if [ "$has_uncommitted" = true ]; then
    echo "Error: You have uncommitted changes that will NOT be included in the patch:"
    git status --short
    echo ""
    echo "Please commit them first:"
    echo "  cd $SGLANG_PATH && git add -A && git commit --amend --no-edit"
    exit 1
fi

PATCH_DIR="$PROJECT_ROOT/tools/patches/sglang/$SGLANG_VERSION"
mkdir -p "$PATCH_DIR"
PATCH_FILE="$PATCH_DIR/sglang.patch"

echo "Generating patch from $SGLANG_COMMIT to HEAD..."
# Write diffstat header as a comment, then the actual diff.
# git apply ignores lines before the first "diff --git" line,
# so the diffstat is purely informational for human readers.
{
    echo "angelspec sglang patch (base: ${SGLANG_COMMIT:0:10})"
    echo "---"
    git diff --stat "$SGLANG_COMMIT" HEAD
    echo ""
    git diff "$SGLANG_COMMIT" HEAD
} > "$PATCH_FILE"

if [ ! -s "$PATCH_FILE" ]; then
    echo "Error: Failed to generate patch or patch is empty"
    exit 1
fi

PATCH_SIZE=$(wc -l < "$PATCH_FILE")
echo "✓ Patch updated successfully: tools/patches/sglang/$SGLANG_VERSION/sglang.patch ($PATCH_SIZE lines)"

echo ""
echo "Files modified:"
git diff --name-status "$SGLANG_COMMIT" HEAD
