# Quickstart

This walks through training a **DFly** draft model for **Qwen3-8B** on a single node — the
recommended entry point for first-time users.

## Prerequisites

- 8 GPUs (4 for inference, 4 for training)
- Access to `Qwen/Qwen3-8B` on HuggingFace
- AngelSpec installed ([Installation](installation.md))

## Run it

```bash
./examples/qwen3-8b-dfly/run.sh
```

This launches AngelSpec with `configs/vllm_qwen3_8b_dfly.yaml`:

- **Inference:** 4 GPUs serving the target model via vLLM (tp=2, 2 engines)
- **Training:** 4 GPUs training the DFly draft model under FSDP2
- Hidden states flow inference → Mooncake → trainer

## Common overrides

Config values can be overridden directly on the command line:

```bash
# Shorter run
./examples/qwen3-8b-dfly/run.sh training.num_train_steps=50

# Different learning rate
./examples/qwen3-8b-dfly/run.sh training.learning_rate=2e-5

# Use fewer GPUs (4 total: 2 inference + 2 training)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./examples/qwen3-8b-dfly/run.sh \
    training.training_num_gpus_per_node=2 \
    inference.inference_num_gpus=2
```

## What happens under the hood

The inference GPUs prefill the target model and extract hidden states from selected layers.
Those tensors are written to the Mooncake store; the training GPUs pull them and run the DFly
draft model's forward/backward. See [Disaggregated Architecture](../concepts/disaggregated_architecture.md)
for the full picture.

## Other architectures

To train a different draft architecture on the same target, swap the example:

```bash
# MTP (with sequence packing + USP)
./examples/qwen3-8b-mtp/run.sh

# DSpark (from scratch)
./examples/qwen3-8b-dspark/run.sh
```

See [The Draft-Model Family](../concepts/draft_model_family.md) for architecture details.

## Next steps

- Continue training from a released checkpoint: [`examples/qwen3-8b-dfly-cpt`](../../examples/qwen3-8b-dfly-cpt/)
- Scale to multi-node: [Multi-Node Training](../advanced_features/multi_node.md)
- Convert a checkpoint for serving: [Checkpoint Conversion](../basic_usage/checkpoint_conversion.md)
