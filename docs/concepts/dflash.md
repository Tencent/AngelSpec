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
  feed as context. Layer ids are typically sampled uniformly across the target's depth. An id
  names that layer's output, so the target's last layer is `target_num_hidden_layers - 1`; when
  used, it must be listed last.
- `mask_token_id` — the token used to seed block positions; it is model-specific.
- When adapting to a new target model, update `vocab_size`, `mask_token_id`,
  `target_hidden_size`, `target_num_hidden_layers`, `target_layer_ids`, and `rope_theta`.

The draft model's embedding and `lm_head` are loaded from the target model and frozen.

## Hidden-states correction (optional)

`enable_hidden_correction` adds a correction head on the drafter's output, right before the
LM head:

```
h'_{t+i} = h_{t+i} + SwiGLU( norm(h_{t+i}) :: norm(e_{t+i-1}) )
```

`e_{t+i-1}` is the previous token's embedding, so each block slot is conditioned on the token
before it. The output projection starts at zero, so the head can be switched on when continuing
from a DFlash checkpoint. `hidden_correction_intermediate_size` sets its width (default
`hidden_size`). [DFly](dfly.md) enables it by default.

## MTP fusion (optional)

`mtp_fusion` adds the opening of the target's MTP layer to the block input:

```
f = eh_proj( enorm(E(x_a)) :: hnorm(norm(h_{a-1})) )
```

`h_{a-1}` is the target's final hidden state at the last accepted token and `x_a` the anchor.
`"slot0"` adds `f` to the anchor slot, `"slot1"` to the first predicted slot and `"broadcast"`
to every slot, each through a gate that starts at zero. `mtp_fusion_init: "target"` copies
`eh_proj` / `enorm` / `hnorm` from the target's MTP layer; `"random"` leaves them at init. The
fusion needs the vLLM engine with `inference.store_last_hidden_states=true` and does not
support USP.

## Decoder-layer body (`layer_arch`)

`layer_arch: "dflash"` is a pre-norm residual layer whose attention is set by `attention_type`
and whose FFN is set by `draft_mlp_type`: `"dense"` for a SwiGLU FFN, or `"moe"` for a routed
MoE with a shared expert, as in the MTP drafter. For MoE targets the FFN can start from the
target's layers:

| Option | Effect |
|---|---|
| `target_mlp_init_layers` | Copies each target layer's router, experts and post-attention norm into the matching draft layer |
| `target_mlp_init_experts` | Keeps only some experts: `"bias"`, or explicit ids per layer |
| `target_mlp_dense_init_layers` | Builds a dense FFN from the shared expert plus k routed experts; `intermediate_size` must be (shared + k) × the expert width |
| `moe_aux_free_bias_update` | Balances expert load through the router bias, without an auxiliary loss (FULL_SHARD only) |

With Muon (`training.optimizer_type=muon`), `training.muon_per_expert` (default on)
orthogonalizes each expert separately. `training.moe_lr_scale` and
`training.moe_router_lr_scale` scale the learning rate of the MoE parameters and of the router.

### Attention block (`attention_type`)

`"gqa"` (default) is grouped-query attention; `"mla"` is gated MLA, optionally with a DSA
indexer (set `index_topk`). The MLA and DSA options are described on the
[DFly page](dfly.md#attention-block). DFlare cannot use MLA.

### Warm-starting the drafter's weights

`training.draft_init_path` loads a drafter's own weights (an HF-style directory) before
training; parameters the directory lacks keep their init. `training.load_path` instead resumes
a training checkpoint, with its optimizer and LR state.

## Related

- [DFlare](dflare.md) — DFlash with a learnable per-layer mix of the target layers
- [DSpark](dspark.md) — DFlash backbone with additional prediction heads
