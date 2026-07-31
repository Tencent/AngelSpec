# DSpark

DSpark builds on the [DFlash](dflash.md) backbone and adds EAGLE-style prediction heads. It
reuses DFlash's block-parallel forward pass, block-causal mask, anchor sampling, and loss, and
attaches its extra components through DFlash's extension hooks rather than reimplementing the
forward.

## Components

- **Markov logit bias.** A low-rank bigram term that biases the draft logits based on the
  previous token. In its default form it adds only two small matrices and no other parameters,
  so it is inexpensive and keeps checkpoints compatible with the plain DFlash head.
- **Accept-rate confidence head.** A small head that predicts how likely each drafted token is
  to be accepted by the target model. It is trained with a binary cross-entropy target derived
  from the agreement between the draft and target (`1 - 0.5 * L1(draft, teacher)`). This signal
  can be used to prune unlikely branches at serving time.

## Loss

The total loss is the shared DFlash objective (CE with positional decay, plus optional L1/KL/LK
distillation) plus a confidence-head term when the accept-rate head is enabled:

```
loss = <DFlash objective> + confidence_head_alpha * confidence_loss
```

## Configuration

DSpark is selected by `DSparkConfig` (a superset of `DFlashConfig`). Relevant fields include the
Markov head (`markov_rank`, `markov_head_type`) and the confidence head
(`enable_confidence_head`, `confidence_head_with_markov`). Setting `model_arch` to
`"dflare"` builds DSpark on the DFlare backbone (layer-wise target fusion) instead of DFlash.

The loss objective and distillation weights come from the shared `dflash_*` training knobs; see
[Training](../basic_usage/training.md).
