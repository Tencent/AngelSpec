#!/bin/bash
# Train a DFly draft model for Qwen3-VL-30B-A3B-Instruct (text-only rollout).
#
# The target is Qwen3VLMoeForConditionalGeneration (qwen3_vl_moe), served
# natively by vLLM. The draft is a small dense text model;
# the target's MoE only affects vLLM serving.
#
# GPU allocation (single node, 8 GPUs):
#   - 4 GPUs for inference (vLLM, tp=4)
#   - 4 GPUs for training (FSDP2 FULL_SHARD)
#
# Usage:
#   ./examples/qwen3vl-30b-dfly/run.sh [EXTRA_ARGS...]

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export ANGELSPEC_LOG_LEVEL=INFO

CONFIG_FILE="${1:-$ROOT_DIR/configs/vllm_qwen3vl_30b_dfly.yaml}"
if [[ -f "$CONFIG_FILE" ]]; then
    shift 1 || true
else
    CONFIG_FILE="$ROOT_DIR/configs/vllm_qwen3vl_30b_dfly.yaml"
fi

echo "=============================================="
echo "DFly Training (text-only) — Qwen3-VL-30B-A3B-Instruct"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "  - Inference: vLLM tp=4 (4 GPUs)"
echo "  - Training: FSDP2 FULL_SHARD (4 GPUs)"
echo "Extra args: $*"
echo "=============================================="

python3 -m angelspec.train_entry \
    --config "$CONFIG_FILE" \
    "$@"
