# Data Preparation

AngelSpec trains on conversational data. Each training sample is a conversation; during training
the target model prefills it, and the draft model learns from the resulting hidden states over
the assistant tokens.

## Data format

Datasets are JSONL files (or HuggingFace Hub dataset IDs). Each row holds a conversation under a
configurable key (default `conversations`), in OpenAI-style role/content form:

```json
{
  "id": "example_001",
  "conversations": [
    {"role": "user", "content": "What is speculative decoding?"},
    {"role": "assistant", "content": "It speeds up inference by ..."}
  ]
}
```

Multi-turn conversations, system messages, and tool calls are supported. Some target models also
accept multimodal content (text plus images).

Point the trainer at your data with:

```yaml
dataset:
  train_data_path: /path/to/train.jsonl   # or a HuggingFace dataset ID
  eval_data_path: /path/to/eval.jsonl      # optional
  prompt_key: conversations
```

## Chat templates

The `chat_template` selects how conversations are rendered into token sequences and which tokens
are supervised. Templates are keyed by name (for example `llama3`, `qwen3`, and model-specific
variants). Set it to match your target model:

```yaml
dataset:
  chat_template: qwen3
```

To add a new template, register it in the chat-template registry — see
[Customization](../advanced_features/customization.md).

## Loss masks

Only assistant tokens contribute to the loss. The loss mask is computed during preprocessing
from the chat template (which spans are assistant turns). Related options:

- `last_turn_loss_only` — supervise only the final assistant turn (`"auto"`, `true`, or `false`).
- `min_loss_tokens` — skip sequences with fewer than *N* supervised tokens. For DFlash, set this
  to at least `2 * dflash_block_size` so every sequence has enough supervised tokens for a block.
- `drop_overlength` — drop (rather than truncate) samples longer than `max_seq_length`.

## Tokenization and caching

By default, data is tokenized up front and cached under `cache_dir`, keyed by a hash of the
dataset so later runs reuse it. Setting `defer_tokenization: true` moves tokenization into the
training stream (not supported for DFlash).

## Sequence length

`training.max_seq_length` bounds the sequence length. For very long contexts (128k+), see
[Long-Sequence Training](../advanced_features/long_sequence_usp.md); to reduce padding waste when
mixing short and long samples, see [Sequence Packing](../advanced_features/sequence_packing.md).
