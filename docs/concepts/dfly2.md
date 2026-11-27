# DFly2

DFly2 (`model_type: "dfly2"`) is the block-parallel drafter Hy4 and Hy4 flash ship with.
AngelSpec also trains it for DeepSeek-V4.1-Flash and GLM-5.3-Flash. The
[technical report](https://arxiv.org/abs/2611.34776) covers its design and evaluation.

## What sets it apart

Where DFlash projects its context from captured target hidden states, DFly2 reads the
target's own attention state:

- **Target index picks** (`target_index_layer`, `target_index_topk`): each draft layer
  attends to the context positions that a target layer's sparse-attention indexer selects
  for the last accepted token, next to the draft block itself: the top 2048 on Hy4,
  Hy4 flash and GLM-5.3, the top 512 plus a 128-token window on DeepSeek-V4.1.
- **Target KV reuse** (`target_kv_layers`): each draft layer reads that target layer's
  MLA latents instead of projecting a context of its own, so the served drafter shares
  the target's KV cache.
- **Target-initialized attention** (`target_attn_init_layers`): draft attention starts
  as the target layer's (Hy4, Hy4 flash and DeepSeek-V4.1).
- **MTP fusion** (`mtp_fusion: slot0`): the target's final hidden state at the last
  accepted token, fused with the anchor embedding the way the target's MTP layer opens,
  is added to the anchor slot's input.
- Hidden correction and a confidence head.

The draft attention follows the target's: Hy4's gated MLA with a learnable sink (`mla`),
DeepSeek-V4.1's latent MQA with mHC residuals (`dsv41_mla`, `hc_mult: 4`), and GLM-5.3's
NoPE MLA (`glm53_mla`), which trains with `dflash_causal_attn` since its scores carry no
positions.

## Configuration and training

| Target | Draft config | Recipe | Walkthrough |
|---|---|---|---|
| Hy4 | `angelspec/config/dfly2_hy_v4_draft_config.json` | `configs/vllm_hy_v4_dfly2_64k.yaml` | [hy-v4-dfly2](https://github.com/Tencent/AngelSpec/tree/main/examples/hy-v4-dfly2) |
| Hy4 flash | `angelspec/config/dfly2_hy_v4_flash_draft_config.json` | `configs/vllm_hy_v4_flash_dfly2_256k.yaml` | [hy-v4-flash-dfly2](https://github.com/Tencent/AngelSpec/tree/main/examples/hy-v4-flash-dfly2) |
| DeepSeek-V4.1-Flash | `angelspec/config/dfly2_dsv41_flash_draft_config.json` | `configs/vllm_dsv41_flash_dfly2.yaml` | — |
| GLM-5.3-Flash | `angelspec/config/dfly2_glm53_flash_draft_config.json` | `configs/vllm_glm53_flash_dfly2.yaml` | — |

## Serving

`tools/export_dfly_to_target.py` packs a trained DFly2 drafter into the target
checkpoint the way vLLM serves the drafter Hy4 and Hy4 flash ship with
(`model.dfly_layers.*` and `dfly_*` config fields). vLLM support for serving the
DeepSeek-V4.1-Flash and GLM-5.3-Flash drafters is coming soon.
