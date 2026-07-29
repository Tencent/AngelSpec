# Hy3 MTP — Multi-Node Training

Train an MTP (Multi-Token Prediction) draft model for Hy3 across multiple
nodes, reproducing [AngelSlim/Hy3-MTP-TTT3](https://huggingface.co/AngelSlim/Hy3-MTP-TTT3).

MTP uses a full MoE decoder layer as the draft head with test-time training (TTT).
This multi-node setup combines USP (Ulysses Sequence Parallelism) for long-context
training with multi-node inference for the large MoE target.

## Requirements

- 2+ nodes, 8 GPUs each (H100/H200 recommended)
- Ray cluster running across all nodes
- Mooncake RDMA configured (see [Multi-Node docs](../../docs/advanced_features/multi_node.md))
- Hy3 target model on shared storage

## Run

```bash
NUM_NODES=2 ./examples/hy3-mtp/run.sh
```

## Config

Provide your own config pointing to the Hy3 target model:

```yaml
model:
  target_model_path: /shared/models/hy3
  draft_model_config: angelspec/config/mtp_qwen3_8b_dense_draft_config.json

training:
  attention_backend: usp
  sp_ulysses_size: 4
  dflash_packing: true
  num_nodes: 2

inference:
  engine_type: vllm
  inference_num_gpus_per_engine: 4
```

## Notes

- USP shards long sequences across training GPUs for O(seq/sp) memory per rank.
- MTP + USP requires `micro_batch_size: 1` and `sp_ring_size: 1`.
- Sequence packing is enabled by default to maximize training throughput.
