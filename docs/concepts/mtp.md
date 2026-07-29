# MTP

MTP (multi-token prediction) uses a **full MoE decoder layer** as the draft model, trained with
test-time training (TTT). It is designed for models whose architecture includes a native MTP
head — the draft is a complete decoder layer (grouped-query attention with QK-norm, a
sigmoid-router mixture-of-experts, and a shared expert) plus the fusion modules that combine the
token embedding with the target's hidden state.

Unlike the DFlash family, MTP is not block-parallel; it predicts one token at a time and is
independent of the Eagle3 code path.

## How it works

- **Input fusion.** Each step normalizes the token embedding and the target's last hidden state
  (`enorm` / `hnorm`), concatenates them, and projects back to the hidden size (`eh_proj`) before
  the decoder layer.
- **TTT loop with a KV cache.** The output hidden of each step is fed back as the next step's
  input, and the inputs are shifted so that step *k* predicts the token *k* positions ahead. A
  per-step key/value cache lets each step reuse the previous steps' attention state.
- **Single last hidden state.** MTP consumes only the target's final hidden state — there is no
  multi-layer auxiliary fusion.
- **Tied `lm_head`.** The draft shares the target model's frozen `lm_head` for both the CE target
  and the KL teacher, so it adds no `lm_head` parameters.
- **Loss.** Cross-entropy against the target's argmax plus KL divergence against the target's
  soft distribution, with per-step weights that decay over the prediction horizon.

## Attention backends

MTP's cached attention has three interchangeable backends, selected by
`attention_backend`:

| Backend | Use case |
|---------|----------|
| `sdpa` | Default. Eager cached attention; simplest, but memory grows with sequence length. |
| `fa` | Long sequences. The first block runs through compiled FlexAttention (linear memory in sequence length); later cached steps merge in with log-sum-exp. |
| `usp` | Very long sequences. `fa` plus [Ulysses sequence parallelism](../advanced_features/long_sequence_usp.md) to shard a 128k+ sequence across GPUs. |

## Initialization

MTP can be initialized two ways:

- **Warm-start from a target's MTP layer.** If the target checkpoint contains a trained MTP
  layer, its weights are loaded as the draft's starting point (continual training).
- **From scratch.** For targets without a native MTP layer, define a dimension-matched draft
  (same hidden size and expert count as the target's decoder layers) and train from random
  initialization.

In both cases the draft must be dimension-matched to the target model that will serve the hidden
states, so acceptance-rate metrics are meaningful end to end.

## Serving

Training checkpoints are converted to HuggingFace format for serving. The converter is the
inverse of the initialization step: it unfuses the expert weights back to the per-expert layout
and writes the keys where the serving engine expects the MTP layer. The embedding and `lm_head`
are borrowed from the main model at load time, matching the tied/frozen design. See
[Checkpoint Conversion](../basic_usage/checkpoint_conversion.md).
