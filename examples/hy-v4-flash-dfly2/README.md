# Hy4 flash DFly2 drafter

DFly2 is the block-parallel drafter Hy4 flash ships with: five gated-MLA layers that read
target layers 25/29/33/37/41 (their DSA candidates and MLA latents, and their attention
weights as the starting point), MTP fusion on the first draft slot, hidden correction and
a confidence head. See [DFly2](../../docs/concepts/dfly2.md).

The recipe, `configs/vllm_hy_v4_flash_dfly2_256k.yaml`, trains on 256k packed rows with two
inference and two training nodes.

## Setup

```bash
pip install "angelspec[tilelang]"
export HY_V4_FLASH_TARGET_MODEL=/path/to/hy-v4-flash
export HY_V4_FLASH_TRAIN_DATA=/path/to/train.jsonl
export ANGELSPEC_RUN_DIR=/path/to/run
```

Training data is conversation JSONL (`conversations` key) rendered with the `hy4-think` chat
template; conversations may contain images and videos.

## Training

On an existing Ray cluster:

```bash
RAY_ADDRESS=<head>:6379 bash examples/hy-v4-flash-dfly2/run.sh
```

With [cluster operations](../../docs/advanced_features/cluster_operations.md), set
`ANGELSPEC_NUM_NODES=4`, `ANGELSPEC_TRAIN_NODES=2` and `ANGELSPEC_INFER_NODES=2` in
`scripts/ops/cluster.env`, then:

```bash
bash scripts/ops/cluster.sh restart configs/vllm_hy_v4_flash_dfly2_256k.yaml
```

Training ends by writing the drafter in Hugging Face layout to
`$ANGELSPEC_RUN_DIR/dfly2/hf_final`. To convert an intermediate checkpoint, pass the
draft config, and the target for the embedding the drafter shares with it:

```bash
python tools/convert_to_hf.py --input-dir "$ANGELSPEC_RUN_DIR/dfly2/checkpoints/iter_0001000" \
  --config angelspec/config/dfly2_hy_v4_flash_draft_config.json \
  --target-model-path "$HY_V4_FLASH_TARGET_MODEL"
```

## Serving

Hy4 flash serves its drafter from the target checkpoint. Pack the trained drafter into a
checkpoint that links the target's shards, then serve it:

```bash
python tools/export_dfly_to_target.py --target "$HY_V4_FLASH_TARGET_MODEL" \
  --draft "$ANGELSPEC_RUN_DIR/dfly2/hf_final" --out /path/to/served-checkpoint

vllm serve /path/to/served-checkpoint --tensor-parallel-size 8 \
  --speculative-config '{"method": "dfly", "num_speculative_tokens": 7}'
```
