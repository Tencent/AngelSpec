# DFlare

DFlare is [DFlash](dflash.md) with **layer-wise target fusion**. It keeps DFlash's block-parallel
prediction, block-causal mask, anchor sampling, and CE loss unchanged, and changes only how the
captured target layers are combined into context.

## What changes over DFlash

1. **Separate context and draft K/V projections.** DFlash shares one key/value projection for
   both the target hidden states and the draft tokens. DFlare gives the target hidden states
   their own `k_proj_target` / `v_proj_target`, so context and draft tokens are projected
   independently before attention.

2. **Learnable per-layer fusion.** DFlash collapses the *T* captured target layers into a single
   shared context with one linear projection. DFlare instead keeps the layers separate and
   learns a `layer_fusion_weights` matrix of shape `[num_draft_layers, T]`: each draft layer
   attends to its own softmax-weighted mixture of the target layers. This lets different draft
   layers draw on different depths of the target model rather than a single fixed blend.

## Configuration

DFlare reuses `DFlashConfig` and the DFlash architecture name; the choice between DFlash and
DFlare is the config's `model_arch` field:

```json
{
  "architectures": ["DFlashDraftModel"],
  "model_arch": "dflare"
}
```

The trainer and checkpoint tooling read `model_arch` to build (and reload) the correct model, so
a DFlare checkpoint's per-layer fusion weights are preserved rather than silently dropped.

All other fields match [DFlash](dflash.md).
