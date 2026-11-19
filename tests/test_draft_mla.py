"""The DFlash-family draft attention registry."""

import pytest

from angelspec.models.draft.attention_registry import (
    build_draft_attention,
    draft_attention_types,
    register_draft_attention,
)
from angelspec.models.draft.dflash import DFlashAttention, DFlashConfig
from angelspec.models.draft.hy4_mla import Hy4DraftMLAAttention

H = 64


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
    assert set(draft_attention_types()) == {"gqa", "mla"}
    with pytest.raises(ValueError, match="already registered"):
        register_draft_attention("gqa", "elsewhere:build")
    with pytest.raises(ValueError, match="attention_type must be one of"):
        DFlashConfig(attention_type="unknown")
