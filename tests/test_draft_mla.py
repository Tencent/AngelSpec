"""The DFlash-family draft attention registry, and DeepSeek-V4.1's and GLM-5.3's MLA."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from angelspec.models.dflash import _create_dflash_mask_mod
from angelspec.models.draft.attention_registry import (
    build_draft_attention,
    draft_attention_types,
    register_draft_attention,
)
from angelspec.models.draft.dflash import DFlashAttention, DFlashConfig
from angelspec.models.draft.dsv41_mla import DSV41DraftAttention, rope_inv_freq
from angelspec.models.draft.glm53_mla import Glm53DraftAttention
from angelspec.models.draft.hy4_mla import Hy4DraftMLAAttention
from angelspec.models.ops.latent_attention import latent_attention

H, CTX, BLOCK = 64, 10, 3

DSV41 = dict(
    attention_type="dsv41_mla",
    num_attention_heads=4,
    head_dim=32,
    qk_rope_head_dim=8,
    q_lora_rank=16,
    o_lora_rank=8,
    o_groups=2,
)
GLM53 = dict(
    attention_type="glm53_mla",
    num_attention_heads=4,
    q_lora_rank=16,
    kv_lora_rank=32,
    qk_nope_head_dim=8,
    qk_rope_head_dim=0,
    v_head_dim=6,
)


# DeepSeek-V4.1-Flash's rope_scaling, which its compressed-KV layers apply at theta 160000.
V41_YARN = dict(original_seq_len=65536, factor=16.0, beta_fast=32, beta_slow=1)


def _config(family, **kw):
    base = dict(hidden_size=H, rms_norm_eps=1e-6, rope_theta=10000.0)
    return DFlashConfig(**base, **family, **kw)


def _block_mask(anchors, window=None, causal=False, device="cpu"):
    keep = torch.ones(1, anchors.numel(), dtype=torch.bool, device=device)
    mask_mod = _create_dflash_mask_mod(
        anchors[None].to(device), keep, CTX, BLOCK, causal=causal, context_local_window=window
    )
    q_len = anchors.numel() * BLOCK
    return create_block_mask(mask_mod, 1, None, q_len, CTX + q_len, device=device)


def _inputs(anchors):
    g = torch.Generator().manual_seed(1)
    return dict(
        draft_hidden=torch.randn(1, anchors.numel() * BLOCK, H, generator=g),
        context_hidden=torch.randn(1, CTX, H, generator=g),
        draft_position_ids=(anchors[:, None] + torch.arange(BLOCK)).view(1, -1),
        context_position_ids=torch.arange(CTX)[None],
    )


def test_registry_builds_each_family():
    gqa = DFlashConfig(hidden_size=H, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    assert isinstance(build_draft_attention(gqa, 0), DFlashAttention)
    mla = DFlashConfig(
        hidden_size=H,
        num_attention_heads=4,
        attention_type="mla",
        q_lora_rank=16,
        kv_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
    )
    assert isinstance(build_draft_attention(mla, 0), Hy4DraftMLAAttention)
    assert isinstance(build_draft_attention(_config(DSV41), 0), DSV41DraftAttention)
    assert isinstance(build_draft_attention(_config(GLM53), 0), Glm53DraftAttention)
    assert set(draft_attention_types()) == {"gqa", "mla", "dsv41_mla", "glm53_mla"}
    with pytest.raises(ValueError, match="already registered"):
        register_draft_attention("gqa", "elsewhere:build")
    with pytest.raises(ValueError, match="attention_type must be one of"):
        DFlashConfig(attention_type="unknown")


def test_dsv41_yarn_matches_vllm_for_the_compressed_layers():
    rope = pytest.importorskip("vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope")
    layer = SimpleNamespace(
        base=160000.0, rotary_dim=64, beta_fast=32, beta_slow=1, max_position_embeddings=65536
    )
    want = rope.DeepseekScalingRotaryEmbedding._compute_inv_freq(layer, 16.0)
    torch.testing.assert_close(rope_inv_freq(64, 160000.0, **V41_YARN), want)


def test_glm53_absorbed_attention_matches_per_head_keys_and_values():
    torch.manual_seed(0)
    attn = Glm53DraftAttention(_config(GLM53))
    anchors = torch.tensor([2, 6])
    inputs = _inputs(anchors)
    mask = _block_mask(anchors, causal=True)
    heads, nope, v_dim = 4, 8, 6
    with torch.no_grad():
        got = attn(**inputs, block_mask=mask)
        x = inputs["draft_hidden"]
        latent = torch.cat([attn._latent(inputs["context_hidden"]), attn._latent(x)], dim=1)
        kv = attn.kv_b_proj(latent).view(1, latent.shape[1], heads, nope + v_dim)
        q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(x))).view(1, x.shape[1], heads, nope)
        logits = torch.einsum("bqhd,bkhd->bhqk", q, kv[..., :nope]) * nope**-0.5
        visible = create_mask(mask.mask_mod, 1, 1, x.shape[1], latent.shape[1], device="cpu")
        probs = logits.masked_fill(~visible, float("-inf")).softmax(-1)
        out = torch.einsum("bhqk,bkhd->bqhd", probs, kv[..., nope:]).reshape(1, x.shape[1], -1)
        want = attn.o_proj(out)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention needs CUDA")
def test_flex_latent_attention_matches_the_dense_path():
    torch.manual_seed(0)
    anchors = torch.arange(1, 9).repeat_interleave(16)[:128] % CTX
    q = torch.randn(1, 4, anchors.numel() * BLOCK, 64)
    latent = torch.randn(1, CTX + anchors.numel() * BLOCK, 64)
    sink = torch.randn(4)
    dense = latent_attention(q, latent, _block_mask(anchors), 0.125, sink)
    flex = latent_attention(
        q.cuda(), latent.cuda(), _block_mask(anchors, device="cuda"), 0.125, sink.cuda()
    )
    torch.testing.assert_close(flex.cpu(), dense, rtol=2e-3, atol=2e-3)
