#!/bin/bash
# Continue training a DFly draft model from a released checkpoint.
# Base model: AngelSlim/Qwen3-8B-DFly-B8
#
# GPU allocation (default: 8 GPUs):
#   - 4 GPUs for inference (vLLM, tp=2, 2 engines)
#   - 4 GPUs for training (FSDP2)
#
# Usage:
#   ./examples/qwen3-8b-dfly-cpt/run.sh [EXTRA_ARGS...]

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export ANGELSPEC_LOG_LEVEL=INFO

CONFIG_FILE="$ROOT_DIR/configs/vllm_qwen3_8b_dfly.yaml"

# Download the released checkpoint if not already present
DRAFT_CKPT="${DRAFT_CKPT:-AngelSlim/Qwen3-8B-DFly-B8}"

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}

TRAIN_GPUS=4
INFERENCE_GPUS=4

echo "=============================================="
echo "DFly CPT (continual training) — Qwen3-8B"
echo "Base checkpoint: $DRAFT_CKPT"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Total GPUs: $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - Training GPUs: $TRAIN_GPUS (FSDP2)"
echo "  - Inference GPUs: $INFERENCE_GPUS (vLLM)"
echo "Extra args: $*"
echo "=============================================="

python3 -m angelspec.train_entry \
    --config "$CONFIG_FILE" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    training.load_path="$DRAFT_CKPT" \
    training.continual_training=true \
    inference.inference_num_gpus="$INFERENCE_GPUS" \
    inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
    "$@"
