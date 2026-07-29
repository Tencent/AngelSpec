# Qwen3-8B MTP — Train from Scratch

Train an MTP (Multi-Token Prediction) draft model for Qwen3-8B, reproducing
[AngelSlim/Qwen3-8B-MTP-TTT3](https://huggingface.co/AngelSlim/Qwen3-8B-MTP-TTT3).

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
  dflash_packing: true
  max_seq_length: 40960
```

This config enables both sequence packing and Ulysses sequence parallelism (USP)
for efficient long-context MTP training.

## Expected Results

After 10k steps, expect MAL ~3.0–3.5 on typical benchmarks.
