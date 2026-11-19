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

2. **Hidden-state correction (on by default).** A residual correction applied to the draft hidden
   states before the final `lm_head` projection, conditioning each block slot on the previous
   token. It is a DFlash-family module (see [DFlash](dflash.md#hidden-states-correction-optional),
   where it is off by default); DFly is the path that ships with
   `enable_hidden_correction: true`.

3. **Markov head disabled by default.** Unlike [DSpark](dspark.md), DFly does not use the
   low-rank bigram Markov bias (`markov_rank: 0`), relying solely on the attention-based
   prediction path.

## Configuration

DFly uses its own `DFlyConfig` (an extension of `DFlashConfig`) and the
`"Qwen3DFlyModel"` architecture. It is a DFlash-family drafter and does not
depend on DSpark. The hidden-states correction knobs
(`enable_hidden_correction`, `hidden_correction_intermediate_size`) are inherited
from `DFlashConfig`; DFly only changes the default to `true`:

```json
{
  "architectures": ["Qwen3DFlyModel"],
  "model_type": "qwen3",
  "enable_hidden_correction": true
}
```

The `Qwen3DFlyModel` / `qwen3` identifiers are kept for serving compatibility; DFly works with
any target family.

### Attention block

`attention_type` picks the attention block:

- `"gqa"` (default) — grouped-query attention with shared K/V projections.
- `"mla"` — Hy4's gated MLA, with an output gate and a per-head learnable sink.

With MLA, `index_topk` adds a DSA indexer that picks the context positions each query attends
to; the draft block itself is always attended. The indexer is loaded from the target rather
than trained. `indexer_types` lets layers share indexers as Hy4 does: `"full"` opens one,
`"shared"` reuses the most recent `"full"` layer's.

### DSA backends

`dsa_backend` picks how DSA attention is computed:

| Backend | Use |
|---|---|
| `"torch"` (default) | Dense reference, for short contexts |
| `"tilelang"` | Computes only the selected keys, for long contexts; needs `kv_lora_rank=512`, `qk_rope_head_dim=64`, an indexer and the `tilelang` package |
| `"auto"` | `"tilelang"` where it can run, `"torch"` elsewhere |

## Dispatch

The trainer dispatches on the `DFlyConfig` type:

- **Model:** `DFlyDraftModel` (in `angelspec/models/draft/dfly.py`)
- **Trainer:** `DFlyTrainer` (in `angelspec/training/dfly_trainer.py`) — a thin
  subclass of `DFlashTrainer` that swaps in the `DFlyModel` wrapper (in
  `angelspec/models/dfly.py`) via the DFlash model-build hooks. Reads the
  `dflash_*` hyperparameter namespace.
- **Loss:** Inherits the DFlash composable loss (CE + decay/D-PACE + optional KL/LK,
  plus an optional independent end-to-end multi-step TV term)

## Relation to other architectures

| Feature | DFlash | DFlare | DFly | DSpark |
|---------|--------|--------|------|--------|
| Shared KV projection | ✓ | ✗ (separate) | ✓ (base) | ✓ |
| Per-layer fusion | ✗ | ✓ | ✓ (residual) | ✗ |
| Hidden-state correction (TreeFlash) | opt-in | opt-in | ✓ (default) | opt-in |
| Hy4 gated MLA + DSA (`attention_type: "mla"`) | opt-in | ✗ | opt-in | opt-in |
| Markov head | ✗ | ✗ | ✗ | ✓ |
| Confidence head | ✗ | ✗ | ✗ | ✓ |

## Released models

- [AngelSlim/Qwen3-8B-DFly-B8](https://huggingface.co/AngelSlim/Qwen3-8B-DFly-B8)
- [AngelSlim/Hy3-DFly-B8](https://huggingface.co/AngelSlim/Hy3-DFly-B8)
- [AngelSlim/Hy3-DFly-B8-Think-High](https://huggingface.co/AngelSlim/Hy3-DFly-B8-Think-High)
- [AngelSlim/Hy3-DFly-B8-High](https://huggingface.co/AngelSlim/Hy3-DFly-B8-High)
