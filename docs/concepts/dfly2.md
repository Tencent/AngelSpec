# DFly2

DFly2 (`model_type: "dfly2"`) is the block-parallel drafter Hy4 flash ships with.

## What sets it apart

Where DFlash projects its context from captured target hidden states, DFly2 reads the
target's own attention state:

- **Target index picks** (`target_index_layer`, `target_index_topk`): each draft layer
  attends to the top-2048 context positions that a target layer's DSA indexer selects
  for the last accepted token, next to the draft block itself.
- **Target KV reuse** (`target_kv_layers`): each draft layer reads that target layer's
  MLA latents instead of projecting a context of its own, so the served drafter shares
  the target's KV cache.
- **Target-initialized attention** (`target_attn_init_layers`): draft attention starts
  as the target layer's.
- **MTP fusion** (`mtp_fusion: slot0`): the target's final hidden state at the last
  accepted token, fused with the anchor embedding the way the target's MTP layer opens,
  is added to the anchor slot's input.
- Hidden correction and a confidence head, on gated MLA with a learnable sink.

## Configuration and training

| | |
|---|---|
| Draft config | `angelspec/config/dfly2_hy_v4_flash_draft_config.json` |
| Recipe | `configs/vllm_hy_v4_flash_dfly2_256k.yaml` |
| Walkthrough | [hy-v4-flash-dfly2](../../examples/hy-v4-flash-dfly2/) |

## Serving

`tools/export_dfly_to_target.py` packs a trained DFly2 drafter into the target
checkpoint the way vLLM serves Hy4 flash's drafter (`model.dfly_layers.*` and `dfly_*`
config fields).

On-policy distillation for DFly2 is coming soon.
