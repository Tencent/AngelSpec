# DFly

DFly (DFlareV2) combines the best of [DFlash](dflash.md) and [DFlare](dflare.md): DFlash's
shared-KV context efficiency with DFlare's learnable per-layer target fusion, plus an optional
hidden-state correction head.

## What changes over DFlare

1. **Shared FC context + fusion residual.** DFlash's shared fully-connected context projection
   is kept as the base signal. DFlare's per-layer fusion weights are applied as a *residual* on
   top, rather than replacing the shared context entirely. This gives the model a strong
   initialization (shared context ≈ DFlash) while retaining the expressiveness of per-layer
   adaptation.

2. **Hidden-state correction (optional).** An autoregressive head that applies a per-layer
   residual correction to the draft hidden states before the final `lm_head` projection. This
   addresses the mismatch between the draft model's shallow representations and the target
   model's deeper ones. Enabled via `enable_hidden_correction: true` in the draft config.

3. **Markov head disabled by default.** Unlike [DSpark](dspark.md), DFly does not use the
   low-rank bigram Markov bias (`markov_rank: 0`), relying solely on the attention-based
   prediction path.

## Configuration

DFly uses `DSparkConfig` with `model_arch: "dfly"`:

```json
{
  "architectures": ["Qwen3DSparkModel"],
  "model_type": "qwen3_dspark",
  "model_arch": "dfly",
  "markov_rank": 0,
  "enable_confidence_head": false,
  "enable_hidden_correction": true
}
```

## Dispatch

The trainer dispatches on `DSparkConfig` + `model_arch == "dfly"`:

- **Model:** `DFlyDraftModel` (in `angelspec/models/draft/dfly.py`)
- **Trainer:** `DSparkTrainer` (shared with DSpark via hooks)
- **Loss:** Inherits the DFlash composable loss (CE + decay/D-PACE + optional KL/LK)

## Relation to other architectures

| Feature | DFlash | DFlare | DFly | DSpark |
|---------|--------|--------|------|--------|
| Shared KV projection | ✓ | ✗ (separate) | ✓ (base) | ✓ |
| Per-layer fusion | ✗ | ✓ | ✓ (residual) | ✗ |
| Hidden-state correction | ✗ | ✗ | ✓ (optional) | ✓ (TreeFlash) |
| Markov head | ✗ | ✗ | ✗ | ✓ |
| Confidence head | ✗ | ✗ | ✗ | ✓ |

## Released models

- [AngelSlim/Qwen3-8B-DFly-B8](https://huggingface.co/AngelSlim/Qwen3-8B-DFly-B8)
- [AngelSlim/Hy3-DFly-B8](https://huggingface.co/AngelSlim/Hy3-DFly-B8)
- [AngelSlim/Hy3-DFly-B8-Think-High](https://huggingface.co/AngelSlim/Hy3-DFly-B8-Think-High)
- [AngelSlim/Hy3-DFly-B8-High](https://huggingface.co/AngelSlim/Hy3-DFly-B8-High)
