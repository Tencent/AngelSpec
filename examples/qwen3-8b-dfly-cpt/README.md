# Qwen3-8B DFly — Continual Training (CPT)

Continue training a DFly draft model from a local AngelSpec DCP checkpoint on your own data.

This is the recommended path when you want to adapt the released drafter to a
specific domain (code, math, etc.) without training from scratch.

## Requirements

- 8 GPUs (4 inference + 4 training)
- Target model: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- A local AngelSpec checkpoint root or `iter_*` directory

## Run

```bash
DRAFT_CKPT=/path/to/checkpoints ./examples/qwen3-8b-dfly-cpt/run.sh
```

Hugging Face serving checkpoints are not DCP training checkpoints and cannot be passed to `training.load_path`.

## Key Difference from Scratch Training

The `continual_training: true` flag loads model weights only (no optimizer state,
no LR schedule). This lets you use a fresh learning rate schedule suited to your
domain data volume.

```yaml
training:
  load_path: /path/to/checkpoints
  continual_training: true
  learning_rate: 2e-5   # typically lower than from-scratch
```
