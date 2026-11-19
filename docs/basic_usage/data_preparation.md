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
accept multimodal content, given as a list of parts:

```json
{"role": "user", "content": [
  {"type": "image", "image": "/data/images/0001.png"},
  {"type": "video", "video": "/data/videos/0001.mp4"},
  {"type": "text", "text": "What happens after the cut?"}
]}
```

The target's vLLM processor sizes the media, so train with `inference.vllm.enable_multimodal: true`
and the `inference.vllm.mm_processor_kwargs` (pixel bounds) you serve with. Hy4 flash images and
videos are expanded into the model's token layout during preprocessing: images must be local files
or data URLs, videos local files, whose frames are sampled the way vLLM's Hy4 processor samples
them (2 frames per second unless `mm_processor_kwargs` sets `fps` or `num_frames`).

Point the trainer at your data with:

```yaml
dataset:
  train_data_path: /path/to/train.jsonl   # or a HuggingFace dataset ID
  eval_data_path: /path/to/eval.jsonl      # optional
  # false = do not pin eval hidden states in CPU RAM; re-extract at each eval
  cache_eval_hidden_states: true
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

`training.max_seq_length` bounds the sequence length: longer conversations are truncated (or
dropped with `drop_overlength`; Megatron `.bin` conversations are always dropped), and
`dataset.min_seq_length` drops shorter training samples. For very long contexts (128k+), see
[Long-Sequence Training](../advanced_features/long_sequence_usp.md); to reduce padding waste when
mixing short and long samples, see [Sequence Packing](../advanced_features/sequence_packing.md).
