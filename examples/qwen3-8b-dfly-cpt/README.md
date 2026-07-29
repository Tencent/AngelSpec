# Qwen3-8B DFly — Continual Training (CPT)

Continue training a DFly draft model from the released
[AngelSlim/Qwen3-8B-DFly-B8](https://huggingface.co/AngelSlim/Qwen3-8B-DFly-B8)
checkpoint on your own data.

This is the recommended path when you want to adapt the released drafter to a
specific domain (code, math, etc.) without training from scratch.

## Requirements

- 8 GPUs (4 inference + 4 training)
- Target model: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- Draft checkpoint: [AngelSlim/Qwen3-8B-DFly-B8](https://huggingface.co/AngelSlim/Qwen3-8B-DFly-B8) (downloaded automatically)

## Run

```bash
./examples/qwen3-8b-dfly-cpt/run.sh

# Or specify a local checkpoint path:
DRAFT_CKPT=/path/to/local/checkpoint ./examples/qwen3-8b-dfly-cpt/run.sh
```

## Key Difference from Scratch Training

The `continual_training: true` flag loads model weights only (no optimizer state,
no LR schedule). This lets you use a fresh learning rate schedule suited to your
domain data volume.

```yaml
training:
  load_path: AngelSlim/Qwen3-8B-DFly-B8
  continual_training: true
  learning_rate: 2e-5   # typically lower than from-scratch
```
