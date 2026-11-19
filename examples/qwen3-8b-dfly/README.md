# Qwen3-8B DFly — Train from Scratch

Train a DFly draft model for Qwen3-8B from scratch.

DFly (DFlareV2) combines DFlash shared-KV context with per-layer target fusion
and optional hidden-state correction.

## Requirements

- 8 GPUs (4 inference + 4 training)
- Target model: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

## Run

```bash
./examples/qwen3-8b-dfly/run.sh
```

## Config

Uses `configs/vllm_qwen3_8b_dfly.yaml`. Key settings:

```yaml
model:
  draft_model_config: angelspec/config/dfly_qwen3_8b_draft_config.json

training:
  dflash_block_size: 16
  num_train_steps: null
  learning_rate: 5e-4
```

The bundled sample dataset is for smoke testing, not for reproducing released model quality.
