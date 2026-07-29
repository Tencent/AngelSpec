# Draft Model Configs

Architecture configurations for Eagle3 draft models. These define the model structure (hidden size, number of layers, attention heads, vocabulary size, etc.) used to initialize the draft model before training.

## Files

| Config | Target model | Notes |
|--------|-------------|-------|
| `qwen3_8b_eagle3.json` | Qwen3-8B | 1-layer LlamaForCausalLMEagle3, hidden_size=4096 |

## Usage

Reference a draft model config via the `model.draft_model_config` field in your training config or as a CLI override:

```bash
python -m angelspec.train_entry --config configs/sglang_qwen3_8b.yaml \
    model.draft_model_config=configs/draft_models/qwen3_8b_eagle3.json
```
