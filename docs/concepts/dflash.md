# DFlash

DFlash is a **block-parallel** draft model. Instead of predicting one token at a time, it
predicts a whole block of tokens in a single forward pass, so drafting a block costs one forward
pass rather than one per token.

## How it works

- **Block-parallel prediction.** DFlash predicts a fixed-size block of tokens at once (default
  `block_size=16`), using a **block-causal** attention mask (built with
  [FlexAttention](https://pytorch.org/blog/flexattention/)) so positions within a block can be
  produced together while still respecting causal order across blocks.
- **Dual-source KV context.** The target model's multi-layer hidden states are projected into
  the attention key/value space (`W_proj`) and attended to alongside the draft tokens' own K/V.
  This is how the target's features are injected, in contrast to Eagle3's input fusion.
- **Anchor sampling.** During training, block start positions ("anchors") are sampled across the
  sequence, and a mask token plus the anchor token embedding seed each block.
- **Loss.** Cross-entropy against the ground-truth tokens, with an exponential positional decay
  that weights earlier in-block positions more heavily (later positions are harder to predict).
  Optional distillation terms against the target model's logits can be mixed in: L1, top-K KL,
  or LK. An independent end-to-end multi-step TV term can also be added on top.

DFlash is a standalone architecture — it does not inherit the Eagle3 interface, because
dual-source KV and block-parallel prediction differ fundamentally from input fusion and
autoregressive TTT.

## Configuration

The draft model is described by a JSON config. Key fields:

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 5,
  "num_target_layers": 5,
  "target_hidden_size": 4096,
  "target_num_hidden_layers": 36,
  "target_layer_ids": [1, 9, 17, 25, 33],
  "mask_token_id": 151669,
  "vocab_size": 151936
}
```

- `num_target_layers` / `target_layer_ids` — how many, and which, target layers to extract and
  feed as context. Layer ids are typically sampled uniformly across the target's depth.
- `mask_token_id` — the token used to seed block positions; it is model-specific.
- When adapting to a new target model, update `vocab_size`, `mask_token_id`,
  `target_hidden_size`, `target_num_hidden_layers`, `target_layer_ids`, and `rope_theta`.

The draft model's embedding and `lm_head` are loaded from the target model and frozen.

## Related

- [DFlare](dflare.md) — DFlash with a learnable per-layer mix of the target layers
- [DSpark](dspark.md) — DFlash backbone with additional prediction heads
