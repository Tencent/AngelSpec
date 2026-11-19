#!/bin/bash
# Train the Hy4 flash DFly2 drafter (multi-node).
#
# Node allocation (4 nodes, 8 GPUs each):
#   Nodes 0-1: 2 vLLM replicas x TP8 (target hidden states)
#   Nodes 2-3: 16 training GPUs (FSDP2, DP16)
#
# Prerequisites:
#   - Ray cluster running across the 4 nodes
#   - pip install "angelspec[tilelang]" (DSA kernels) and a vLLM build with Hy4 flash
#   - HY_V4_FLASH_TARGET_MODEL, HY_V4_FLASH_TRAIN_DATA and ANGELSPEC_RUN_DIR set
#
# Usage:
#   ./examples/hy-v4-flash-dfly2/run.sh [CONFIG] [EXTRA_ARGS...]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export ANGELSPEC_LOG_LEVEL=INFO

CONFIG_FILE="${1:-$ROOT_DIR/configs/vllm_hy_v4_flash_dfly2_256k.yaml}"
if [[ -f "$CONFIG_FILE" ]]; then
    shift 1 || true
else
    CONFIG_FILE="$ROOT_DIR/configs/vllm_hy_v4_flash_dfly2_256k.yaml"
fi

: "${HY_V4_FLASH_TARGET_MODEL:?set HY_V4_FLASH_TARGET_MODEL}"
: "${HY_V4_FLASH_TRAIN_DATA:?set HY_V4_FLASH_TRAIN_DATA}"
: "${ANGELSPEC_RUN_DIR:?set ANGELSPEC_RUN_DIR}"

echo "=============================================="
echo "DFly2 Training (multi-node) — Hy4 flash"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Extra args: $*"
echo "=============================================="

cd "$ROOT_DIR"
python3 -m angelspec.train_entry --config "$CONFIG_FILE" "$@"
