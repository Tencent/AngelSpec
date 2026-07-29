# Qwen3-8B DFly — Train from Scratch

Train a DFly draft model for Qwen3-8B, reproducing
[AngelSlim/Qwen3-8B-DFly-B8](https://huggingface.co/AngelSlim/Qwen3-8B-DFly-B8).

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
  num_train_steps: 10000
  learning_rate: 5e-5
```

## Expected Results

After 10k steps on general-domain data, expect MAL (mean accepted length) ~3.5–4.0
on typical benchmarks when served with vLLM speculative decoding.
