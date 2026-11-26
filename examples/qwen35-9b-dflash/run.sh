#!/bin/bash
# Train a DFlash draft model for Qwen3.5-9B (VL) from scratch.
#
# GPU allocation (default: 8 GPUs):
#   - 4 GPUs for inference (vLLM, tp=4)
#   - 4 GPUs for training (FSDP2)
#
# Usage:
#   ./examples/qwen35-9b-dflash/run.sh [EXTRA_ARGS...]

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export ANGELSPEC_LOG_LEVEL=INFO

CONFIG_FILE="$ROOT_DIR/configs/vllm_qwen35_9b_dflash.yaml"

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}

TRAIN_GPUS=4
INFERENCE_GPUS=4

python3 -m angelspec.train_entry \
    --config "$CONFIG_FILE" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    inference.inference_num_gpus="$INFERENCE_GPUS" \
    inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
    "$@"
