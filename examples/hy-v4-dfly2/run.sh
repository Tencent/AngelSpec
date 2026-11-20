#!/bin/bash
# Train the Hy4 DFly2 drafter (multi-node).
#
# Node allocation (6 nodes, 8 GPUs each):
#   4 nodes: vLLM DP4 x TP8 (target hidden states)
#   2 nodes: 16 training GPUs (USP SP8 x DP2)
#
# Prerequisites:
#   - Ray cluster running across the 6 nodes
#   - pip install "angelspec[tilelang]" (DSA kernels) and a vLLM build with Hy4
#   - HY_V4_TARGET_MODEL, HY_V4_TRAIN_DATA and ANGELSPEC_RUN_DIR set
#
# Usage:
#   ./examples/hy-v4-dfly2/run.sh [CONFIG] [EXTRA_ARGS...]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export ANGELSPEC_LOG_LEVEL=INFO

CONFIG_FILE="${1:-$ROOT_DIR/configs/vllm_hy_v4_dfly2_64k.yaml}"
if [[ -f "$CONFIG_FILE" ]]; then
    shift 1 || true
else
    CONFIG_FILE="$ROOT_DIR/configs/vllm_hy_v4_dfly2_64k.yaml"
fi

: "${HY_V4_TARGET_MODEL:?set HY_V4_TARGET_MODEL}"
: "${HY_V4_TRAIN_DATA:?set HY_V4_TRAIN_DATA}"
: "${ANGELSPEC_RUN_DIR:?set ANGELSPEC_RUN_DIR}"

echo "=============================================="
echo "DFly2 Training (multi-node) — Hy4"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Extra args: $*"
echo "=============================================="

cd "$ROOT_DIR"
python3 -m angelspec.train_entry --config "$CONFIG_FILE" "$@"
