# The Draft-Model Family

AngelSpec supports several draft architectures behind a common training pipeline. They differ in
how they predict tokens (autoregressive vs. block-parallel), how they consume the target model's
hidden states, and what loss they optimize. All are selected by the draft model's config, so
switching architectures is a config change, not a code change.

## Comparison

| Architecture | Prediction | How it uses target hidden states |
|--------------|------------|----------------------------------|
| **Eagle3** | Autoregressive (test-time training) | Input fusion — hidden states concatenated into the draft input |
| **DFlash** | Block-parallel | Dual-source KV — multi-layer hidden projected into attention K/V |
| **DFlare** | Block-parallel | DFlash, with a learnable per-layer mix of the captured target layers |
| **DFly** | Block-parallel | DFlash FC context plus DFlare's per-layer fusion as a residual (DFlareV2), with optional hidden-state correction |
| **DSpark** | Block-parallel + autoregressive head | DFlash backbone, plus a Markov bias and an accept-rate head |
| **MTP** | Single-head, test-time training | A full MoE decoder layer consuming the target's last hidden state |

- **Autoregressive** drafters (Eagle3, MTP) predict one token at a time, feeding each step's
  output back in — accurate per token, but *N* sequential forward passes to draft *N* tokens.
- **Block-parallel** drafters (DFlash, DFlare, DFly, DSpark) predict a whole block of tokens in a
  single forward pass using a block-causal attention mask, trading some per-token accuracy for
  far fewer forward passes.

## Shared training pipeline

All architectures reuse the same infrastructure — the [disaggregated pipeline](disaggregated_architecture.md),
FSDP2 training, the data loader, and evaluation. The block-parallel models (DFlash, DFlare, DFly,
DSpark) additionally share one training wrapper: DFlare, DFly, and DSpark are built on the DFlash
backbone and attach their differences through a small set of extension hooks, rather than
reimplementing the forward pass.

Draft and target also share weights where it helps: the draft model's token embedding and
`lm_head` are loaded from the target model and frozen, so they add no trainable parameters.

## Selecting an architecture

The draft config's `architectures` field (and, for the DFlash family, a `model_arch` field)
picks the model. The trainer dispatches on the config type:

```
DFlashConfig   → DFlash        (model_arch="dflare" → DFlare)
DSparkConfig   → DSpark        (model_arch="dflare" → DFlare-backbone variant, model_arch="dfly" → DFly)
MTPConfig      → MTP
LlamaConfig    → Eagle3 (Llama family)
DeepseekV3Config → Eagle3 (DeepSeek MLA family)
```

## Architecture pages

```{toctree}
:maxdepth: 1

eagle3
dflash
dflare
dfly
dspark
mtp
```
