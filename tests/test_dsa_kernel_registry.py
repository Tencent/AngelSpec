"""DSA kernels registered by name: backend checks, bound options, dispatch, recompute guard."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from angelspec.models.draft.dflash import DFlashConfig
from angelspec.models.draft.dsv41_mla import sparse_latent_attention
from angelspec.models.mtp import MTPModel
from angelspec.models.ops import dsa_kernels
from angelspec.models.ops.sparse_mla import torch_lightning_indexer, torch_sparse_mla
from angelspec.training.activation_checkpointing import enable_dflash_activation_checkpointing


@pytest.fixture
def fake_kernel_calls():
    calls = []

    def sparse_mla(q, kv, indices, *, scaling, sink=None, deterministic=False):
        calls.append(deterministic)
        return torch_sparse_mla(q, kv, indices, scaling=scaling, d_v=q.shape[-1], sink=sink)

    def lightning_indexer(*args, deterministic=False, **kwargs):
        return torch_lightning_indexer(*args, **kwargs)

    dsa_kernels.register_dsa_kernel(
        dsa_kernels.DSAKernel(
            "fake",
            available=lambda: True,
            sparse_mla=sparse_mla,
            lightning_indexer=lightning_indexer,
            dtype=torch.float32,
            options=lambda config: {"deterministic": bool(getattr(config, "fake_det", False))},
            recompute_safe=lambda options: options["deterministic"],
        )
    )
    yield calls
    dsa_kernels._KERNELS.pop("fake")


def test_a_registered_name_is_held_to_the_kernel_constraints(fake_kernel_calls):
    assert "fake" in dsa_kernels.dsa_kernel_names()
    gqa = dict(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    with pytest.raises(ValueError, match="dsa_backend='fake' requires attention_type"):
        DFlashConfig(**gqa, dsa_backend="fake")
    with pytest.raises(ValueError, match="registered DSA kernel"):
        DFlashConfig(**gqa, dsa_backend="missing")


def test_attention_runs_on_the_bound_kernel(fake_kernel_calls):
    kernel = dsa_kernels.bind_dsa_kernel("fake", SimpleNamespace(fake_det=True))
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 5, 2, 16, generator=g)
    latent = torch.randn(1, 7, 16, generator=g)
    indices = torch.tensor([[[0, 2, 4, -1]] * 5])
    want = sparse_latent_attention(q, latent, indices, scaling=0.25, sink=None)
    got = sparse_latent_attention(q, latent, indices, scaling=0.25, sink=None, kernel=kernel)
    torch.testing.assert_close(got, want)
    assert fake_kernel_calls == [True]


def test_activation_checkpointing_refuses_a_kernel_unsafe_to_recompute(fake_kernel_calls):
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.dsa_kernel = dsa_kernels.bind_dsa_kernel("fake", SimpleNamespace())
    model = nn.Module()
    model.layers = nn.ModuleList([layer])
    with pytest.raises(ValueError, match="cannot run under activation checkpointing"):
        enable_dflash_activation_checkpointing(model)


def test_mtp_gradient_checkpointing_refuses_a_kernel_unsafe_to_recompute(fake_kernel_calls):
    def draft(**options):
        kernel = dsa_kernels.bind_dsa_kernel("fake", SimpleNamespace(**options))
        attn = SimpleNamespace(dsa_kernel=kernel)
        return SimpleNamespace(config=SimpleNamespace(), midlayer=SimpleNamespace(self_attn=attn))

    with pytest.raises(ValueError, match="cannot run under gradient checkpointing"):
        MTPModel(draft(), gradient_checkpointing=True)
    MTPModel(draft(fake_det=True), gradient_checkpointing=True)
    MTPModel(draft(), gradient_checkpointing=False)
