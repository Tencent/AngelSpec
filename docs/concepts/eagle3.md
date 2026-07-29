# Eagle3

Eagle3 is the baseline feature-space draft architecture and the method AngelSpec inherits from
[TorchSpec](https://github.com/lightseekorg/TorchSpec). It predicts tokens autoregressively and
is trained with **test-time training (TTT)**.

## How it works

Eagle3 fuses the target model's hidden states into the draft input (an `fc` layer concatenates
the token embedding with the projected hidden states) and then predicts the next token with a
small transformer. During training it runs a TTT loop: it predicts a token, feeds its own
prediction back as the next input, and repeats for a fixed number of steps (`ttt_length`). This
teaches the draft model to stay accurate over the multi-token horizon it will speculate at
serving time.

The loss is **forward KL divergence** against the target model's distribution, so the draft
model learns to match the target's next-token probabilities rather than only the argmax token.

## Model variants

Eagle3 draft models follow the architecture of the target model family:

- **Llama-family** targets use the Llama-based Eagle3 draft (`LlamaConfig`).
- **DeepSeek-family** targets, which use multi-head latent attention (MLA), use the
  MLA-based Eagle3 draft (`DeepseekV3Config`).

The draft model shares the target's embedding and `lm_head` (frozen), and supports vocabulary
pruning to shrink the `lm_head` to a smaller token set. See
[Checkpoint Conversion](../basic_usage/checkpoint_conversion.md) for pruning.

## When to use it

Eagle3 is a strong, well-understood default. If you want fewer forward passes per drafted block,
consider the block-parallel members of the [family](draft_model_family.md) (DFlash, DFlare,
DSpark).
