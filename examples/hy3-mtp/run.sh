#!/bin/bash
# Train an MTP draft model for Hy3 (multi-node).
# Reproduces: AngelSlim/Hy3-MTP-TTT3
#
# Node allocation (default: 2 nodes, 8 GPUs each):
#   Node 0: 4 inference + 4 training
#   Node 1: 4 inference + 4 training
#
# Prerequisites:
#   - Ray cluster running across nodes
#   - Mooncake with RDMA configured
#   - Hy3 target model accessible on shared storage
#
# Usage:
#   ./examples/hy3-mtp/run.sh [EXTRA_ARGS...]

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export ANGELSPEC_LOG_LEVEL=INFO

CONFIG_FILE="${1:-$ROOT_DIR/configs/vllm_hy3_mtp.yaml}"
if [[ -f "$CONFIG_FILE" ]]; then
    shift 1 || true
else
    CONFIG_FILE="$ROOT_DIR/configs/vllm_hy3_mtp.yaml"
fi

NUM_NODES=${NUM_NODES:-2}
GPUS_PER_NODE=8
TRAIN_GPUS=$((NUM_NODES * 4))
INFERENCE_GPUS=$((NUM_NODES * 4))

echo "=============================================="
echo "MTP Training (multi-node) — Hy3"
echo "Reproduces: AngelSlim/Hy3-MTP-TTT3"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Nodes: $NUM_NODES × $GPUS_PER_NODE GPUs"
echo "  - Training GPUs: $TRAIN_GPUS (FSDP2, USP)"
echo "  - Inference GPUs: $INFERENCE_GPUS (vLLM)"
echo "Extra args: $*"
echo "=============================================="

python3 -m angelspec.train_entry \
    --config "$CONFIG_FILE" \
    training.training_num_gpus_per_node=4 \
    training.training_num_nodes="$NUM_NODES" \
    training.attention_backend=usp \
    training.sp_ulysses_size=4 \
    inference.inference_num_gpus="$INFERENCE_GPUS" \
    inference.inference_num_gpus_per_engine=4 \
    inference.inference_num_gpus_per_node="$GPUS_PER_NODE" \
    "$@"
