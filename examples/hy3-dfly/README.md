# Hy3 DFly — Multi-Node Training

Train a DFly draft model for Hy3 across multiple nodes, reproducing
[AngelSlim/Hy3-DFly-B8](https://huggingface.co/AngelSlim/Hy3-DFly-B8).

This is a reference configuration for large MoE target models that require
multi-node inference (TP across nodes) and multi-node training.

## Requirements

- 2+ nodes, 8 GPUs each (H100/H200 recommended)
- Ray cluster running across all nodes
- Mooncake RDMA configured (see [Multi-Node docs](../../docs/advanced_features/multi_node.md))
- Hy3 target model on shared storage

## Run

```bash
# On the head node (Ray head already started):
NUM_NODES=2 ./examples/hy3-dfly/run.sh
```

## Config

Provide your own config pointing to the Hy3 target model:

```yaml
model:
  target_model_path: /shared/models/hy3
  draft_model_config: angelspec/config/dfly_qwen3_8b_draft_config.json

inference:
  engine_type: vllm
  inference_num_gpus_per_engine: 4  # TP=4 for large MoE

training:
  num_nodes: 2
  training_num_gpus_per_node: 4
```

## Notes

- The target model requires TP ≥ 4 due to MoE expert count; adjust
  `inference_num_gpus_per_engine` accordingly.
- Ensure Mooncake RDMA is configured for cross-node hidden-state transfer.
  TCP fallback works but is significantly slower.
