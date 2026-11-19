# Qwen3-8B MTP — Train from Scratch

Train an MTP (Multi-Token Prediction) draft model for Qwen3-8B from scratch.

MTP uses a full MoE decoder layer as the draft head, trained with test-time training
(TTT) — on-policy multi-depth rollout with per-depth KV caching.

## Requirements

- 8 GPUs (4 inference + 4 training)
- Target model: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

## Run

```bash
./examples/qwen3-8b-mtp/run.sh
```

## Config

Uses `configs/vllm_qwen3_8b_mtp_pack_usp_40k.yaml`. Key settings:

```yaml
model:
  draft_model_config: angelspec/config/mtp_qwen3_8b_dense_draft_config.json

training:
  attention_backend: usp
  sp_ulysses_size: 4
  mtp_packing: true
  max_seq_length: 40960
```

This config enables both sequence packing and Ulysses sequence parallelism (USP)
for efficient long-context MTP training.

The default is a 20-step smoke recipe; increase `training.num_train_steps` and use production data for a full run.
