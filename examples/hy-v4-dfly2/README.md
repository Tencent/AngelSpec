# Hy4 DFly2 drafter

DFly2 is the block-parallel drafter Hy4 ships with: five gated-MLA layers that read target
layers 61/65/69/73/77 (their DSA candidates and MLA latents, and their attention weights as
the starting point), MTP fusion on the first draft slot, hidden correction and a confidence
head. See [DFly2](../../docs/concepts/dfly2.md).

The recipe, `configs/vllm_hy_v4_dfly2_64k.yaml`, trains on 64k packed rows with four
inference and two training nodes.

## Setup

```bash
pip install "angelspec[tilelang]"
export HY_V4_TARGET_MODEL=/path/to/hy-v4
export HY_V4_TRAIN_DATA=/path/to/train.jsonl
export ANGELSPEC_RUN_DIR=/path/to/run
```

Training data is conversation JSONL (`conversations` key) rendered with the `hy4-think` chat
template.

## Training

On an existing Ray cluster:

```bash
RAY_ADDRESS=<head>:6379 bash examples/hy-v4-dfly2/run.sh
```

With [cluster operations](../../docs/advanced_features/cluster_operations.md), set
`ANGELSPEC_NUM_NODES=6`, `ANGELSPEC_TRAIN_NODES=2` and `ANGELSPEC_INFER_NODES=4` in
`scripts/ops/cluster.env`, then:

```bash
bash scripts/ops/cluster.sh restart configs/vllm_hy_v4_dfly2_64k.yaml
```

Training ends by writing the drafter in Hugging Face layout to
`$ANGELSPEC_RUN_DIR/dfly2_hy_v4/hf_final`. To convert an intermediate checkpoint, pass the
draft config, and the target for the embedding the drafter shares with it:

```bash
python tools/convert_to_hf.py --input-dir "$ANGELSPEC_RUN_DIR/dfly2_hy_v4/checkpoints/iter_0001000" \
  --config angelspec/config/dfly2_hy_v4_draft_config.json \
  --target-model-path "$HY_V4_TARGET_MODEL"
```

## Serving

Hy4 serves its drafter from the target checkpoint. Pack the trained drafter into a
checkpoint that links the target's shards, with the training block size, then serve it as
the [Hy4 model card](https://huggingface.co/tencent/Hy4) shows:

```bash
python tools/export_dfly_to_target.py --target "$HY_V4_TARGET_MODEL" \
  --draft "$ANGELSPEC_RUN_DIR/dfly2_hy_v4/hf_final" --out /path/to/served-checkpoint \
  --block-size 7
```
