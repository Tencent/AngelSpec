# Qwen3-8B DSpark — Train from Scratch

Train a DSpark draft model for Qwen3-8B from random initialization.
DSpark combines the DFlash block-parallel backbone with an EAGLE-style
autoregressive head (Markov bias + confidence predictor).

## Requirements

- 8 GPUs (4 inference + 4 training)
- Target model: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

## Run

```bash
./examples/qwen3-8b-dspark/run.sh
```

## Config

Uses `configs/sglang_qwen3_8b_dspark.yaml`. Key settings:

```yaml
model:
  draft_model_config: angelspec/config/dspark_qwen3_8b_draft_config.json

training:
  num_epochs: 3
  dflash_block_size: 7
  dspark_num_anchors: 512
  learning_rate: 6e-4
```
