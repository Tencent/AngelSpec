"""Native HSDP parity for USP-style sequence-shard gradients.

Run: torchrun --nproc_per_node=4 tests/test_hsdp_grad_parity.py
"""

import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.nn as nn

from angelspec.training.fsdp import apply_fsdp2
from angelspec.utils.distributed import (
    _build_usp_hsdp_mesh_ranks,
    _resolve_usp_hsdp_shard_size,
    get_usp_hsdp_device_mesh,
    init_usp_groups,
    init_usp_hsdp_mesh,
    resolve_hsdp_shard_size,
)

SP_SIZE = 2
NUM_MICROBATCHES = 2


def test_hsdp_mesh_ranks_use_sp_replication_and_dp_sharding():
    assert _build_usp_hsdp_mesh_ranks(world_size=16, sp_size=4, hsdp_shard_size=2) == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [8, 12],
        [9, 13],
        [10, 14],
        [11, 15],
    ]


def test_hsdp_mesh_ranks_shard_can_span_sp():
    # shard_size > dp_size: params shard across the full DP dim plus a contiguous
    # SP block (long-context memory relief). sp16 x dp2, shard=32 → one shard group
    # over all ranks (replicate=1).
    assert _build_usp_hsdp_mesh_ranks(world_size=32, sp_size=16, hsdp_shard_size=32) == [
        list(range(32))
    ]
    # shard=16 → sp_shard=8, sp_replicate=2: each group spans both DP ranks and an
    # 8-wide SP block.
    assert _build_usp_hsdp_mesh_ranks(world_size=32, sp_size=16, hsdp_shard_size=16) == [
        [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23],
        [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31],
    ]


def test_full_world_shard_uses_1d_mesh():
    usp_mesh = Mock()
    usp_mesh.shape = (1, 4)
    with (
        patch("angelspec.utils.distributed.get_usp_device_mesh", return_value=usp_mesh),
        patch("angelspec.utils.distributed.dist.get_world_size", return_value=4),
        patch("angelspec.utils.distributed.dist.device_mesh.DeviceMesh") as device_mesh,
    ):
        init_usp_hsdp_mesh(hsdp_shard_size=4)
    device_mesh.assert_called_once_with(
        "cuda",
        [0, 1, 2, 3],
        mesh_dim_names=("hsdp_shard",),
    )


def test_hsdp_shard_size_defaults_to_per_node_dp_width():
    assert (
        _resolve_usp_hsdp_shard_size(dp_size=8, sp_size=2, hsdp_shard_size=None, gpus_per_node=4)
        == 2
    )
    assert (
        _resolve_usp_hsdp_shard_size(dp_size=4, sp_size=2, hsdp_shard_size=None, gpus_per_node=8)
        == 4
    )
    assert (
        _resolve_usp_hsdp_shard_size(dp_size=8, sp_size=3, hsdp_shard_size=None, gpus_per_node=8)
        == 1
    )
    assert (
        _resolve_usp_hsdp_shard_size(dp_size=8, sp_size=2, hsdp_shard_size=2, gpus_per_node=4) == 2
    )


def test_flex_localize_passthrough_for_plain_tensor():
    from angelspec.models.ops.flex_attention import _localize_flex_tensor

    tensor = torch.randn(2, 3)
    assert _localize_flex_tensor(tensor, "query") is tensor


def _args():
    return SimpleNamespace(
        fsdp_strategy="FULL_SHARD",
        attention_backend="usp",
        sp_ulysses_size=SP_SIZE,
        sp_ring_size=1,
        bf16=False,
        fp16=False,
        fsdp_reduce_dtype="float32",
    )


def _model():
    torch.manual_seed(42)
    return (
        nn.Sequential(
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 8, bias=False),
        )
        .cuda()
        .bfloat16()
    )


def _batch(sp_rank: int, microbatch: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1000 + sp_rank * 100 + microbatch)
    x = torch.randn(4, 16, generator=gen, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(4, 8, generator=gen, device="cuda", dtype=torch.bfloat16)
    return x, y


def _local_loss(model, sp_rank: int, microbatch: int):
    x, y = _batch(sp_rank, microbatch)
    return (model(x) - y).square().sum() / (y.numel() * SP_SIZE * NUM_MICROBATCHES)


def _replicate_reference():
    model = _model()
    sp_rank = dist.get_rank() % SP_SIZE
    for microbatch in range(NUM_MICROBATCHES):
        _local_loss(model, sp_rank, microbatch).backward()

    dp_size = dist.get_world_size() // SP_SIZE
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(dp_size)
    return model


def _native_hsdp():
    model = apply_fsdp2(_model(), mesh=get_usp_hsdp_device_mesh(), args=_args())
    model.set_requires_all_reduce(False)
    sp_rank = dist.get_rank() % SP_SIZE
    for microbatch in range(NUM_MICROBATCHES):
        model.set_requires_all_reduce(microbatch == NUM_MICROBATCHES - 1)
        _local_loss(model, sp_rank, microbatch).backward()

    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(SP_SIZE)
    return model


def _full_tensors(model, attr: str):
    tensors = []
    for param in model.parameters():
        value = getattr(param, attr)
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        tensors.append(value.detach().float().cpu())
    return tensors


def _test_flex_dtensor_boundary():
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Replicate, Shard

    from angelspec.models.ops.flex_attention import (
        compile_friendly_create_block_mask,
        compile_friendly_flex_attention,
    )

    mesh = get_usp_hsdp_device_mesh()
    placements = [Replicate()] * mesh.ndim
    shape = (1, 2, 64, 16)
    local_tensors = [
        torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    dtensors = [
        DTensor.from_local(tensor, mesh, placements, run_check=False) for tensor in local_tensors
    ]

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = compile_friendly_create_block_mask(
        causal_mask,
        1,
        None,
        shape[2],
        shape[2],
        device=torch.device("cuda"),
        _compile=True,
    )
    output, _ = compile_friendly_flex_attention(
        *dtensors,
        block_mask=block_mask,
        return_lse=True,
    )
    if isinstance(output, DTensor):
        raise AssertionError("FlexAttention output must cross back to a local Tensor")
    output.float().square().mean().backward()
    for tensor in local_tensors:
        if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all()):
            raise AssertionError("DTensor-to-local FlexAttention backward produced invalid grads")

    sharded = DTensor.from_local(
        local_tensors[0].detach(),
        mesh,
        [Replicate(), Shard(2)],
        run_check=False,
        shape=shape,
        stride=local_tensors[0].stride(),
    )
    try:
        compile_friendly_flex_attention(
            sharded,
            dtensors[1],
            dtensors[2],
            block_mask=block_mask,
            return_lse=True,
        )
    except RuntimeError as exc:
        if "replicated DTensor inputs" not in str(exc):
            raise
    else:
        raise AssertionError("FlexAttention accepted a sharded DTensor input")


def _test_mtp_hsdp_flex():
    from torch.distributed.fsdp import register_fsdp_forward_method
    from torch.distributed.tensor import DTensor

    from angelspec.models.draft.mtp import MTPConfig, MTPDraftModel
    from angelspec.models.mtp import MTPModel

    config = MTPConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        qk_norm=True,
        vocab_size=128,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
        rope_theta=10000.0,
        mlp_type="dense",
        target_hidden_size=64,
        tie_lm_head=True,
    )
    torch.manual_seed(123)
    draft = MTPDraftModel(config, attention_backend="usp").cuda().bfloat16()
    draft.freeze_embedding()
    lm_head_weight = torch.randn(
        config.vocab_size,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    draft.set_lm_head_weight(lm_head_weight)
    model = MTPModel(
        draft,
        length=2,
        attention_backend="usp",
        gradient_checkpointing=True,
    ).cuda()
    units = [
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and ("midlayer" in name or "eh_proj" in name)
    ]
    model = apply_fsdp2(
        model,
        mesh=get_usp_hsdp_device_mesh(),
        args=_args(),
        modules_to_shard=units,
    )
    register_fsdp_forward_method(model, "warmup_usp_flex")
    model.warmup_usp_flex(
        seq_length=64,
        hidden_size=config.hidden_size,
        target_lm_head_weight=lm_head_weight,
    )

    batch_size, seq_len = 1, 128
    torch.manual_seed(456)
    input_ids = torch.randint(
        1,
        config.vocab_size,
        (batch_size, seq_len),
        device="cuda",
    )
    hidden_states = torch.randn(
        batch_size,
        seq_len,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    target_hidden_states = torch.randn_like(hidden_states)
    attention_mask = torch.ones(
        batch_size,
        seq_len,
        device="cuda",
        dtype=torch.bfloat16,
    )
    loss_mask = torch.ones(batch_size, seq_len, device="cuda")
    ce_losses, *_ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_hidden_states=target_hidden_states,
        target_lm_head_weight=lm_head_weight,
        loss_mask=loss_mask,
        hidden_states=hidden_states,
    )
    sum(ce_losses).backward()
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
        if not bool(torch.isfinite(local_grad).all()):
            raise AssertionError("MTP USP+HSDP produced non-finite gradients")


def test_resolve_hsdp_shard_size_from_dp_sp_knobs():
    # dp2 x sp16 (world 32).
    # Pure DP (sp defaults to 1): shard = hsdp_shard_dp.
    assert resolve_hsdp_shard_size(2, 16, hsdp_shard_dp=2) == 2
    assert resolve_hsdp_shard_size(2, 16, hsdp_shard_dp=1) == 1
    # Fold SP: shard = dp_size * hsdp_shard_sp (dp defaults to full dp_size).
    assert resolve_hsdp_shard_size(2, 16, hsdp_shard_sp=4) == 8
    assert resolve_hsdp_shard_size(2, 16, hsdp_shard_sp=16) == 32
    assert resolve_hsdp_shard_size(2, 16, hsdp_shard_dp=2, hsdp_shard_sp=8) == 16
    # Legacy int passthrough when no DP/SP knobs.
    assert resolve_hsdp_shard_size(2, 16, hsdp_shard_size=8) == 8
    # Conservative default (per-node) when nothing set.
    assert resolve_hsdp_shard_size(2, 16, gpus_per_node=8) == 1  # 8 % 16 != 0 → 1


def test_resolve_hsdp_shard_size_rejects_bad_combos():
    import pytest

    with pytest.raises(ValueError):  # dp_ways must divide dp_size
        resolve_hsdp_shard_size(2, 16, hsdp_shard_dp=3)
    with pytest.raises(ValueError):  # sp_ways must divide sp_size
        resolve_hsdp_shard_size(2, 16, hsdp_shard_sp=5)
    with pytest.raises(ValueError):  # SP folding needs full DP sharded first
        resolve_hsdp_shard_size(2, 16, hsdp_shard_dp=1, hsdp_shard_sp=4)


def _run_grad_param_parity(shard_label: str):
    reference = _replicate_reference()
    hsdp = _native_hsdp()

    reference_grads = _full_tensors(reference, "grad")
    hsdp_grads = _full_tensors(hsdp, "grad")
    if dist.get_rank() == 0:
        for index, (expected, actual) in enumerate(zip(reference_grads, hsdp_grads)):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=5e-3,
                atol=5e-3,
                msg=lambda msg, i=index: f"[{shard_label}] gradient {i} mismatch: {msg}",
            )

    ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
    hsdp_optim = torch.optim.SGD(hsdp.parameters(), lr=0.1)
    ref_optim.step()
    hsdp_optim.step()

    reference_params = _full_tensors(reference, "data")
    hsdp_params = _full_tensors(hsdp, "data")
    if dist.get_rank() == 0:
        for index, (expected, actual) in enumerate(zip(reference_params, hsdp_params)):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=5e-3,
                atol=5e-3,
                msg=lambda msg, i=index: f"[{shard_label}] parameter {i} mismatch: {msg}",
            )

    del reference, hsdp, ref_optim, hsdp_optim
    torch.cuda.empty_cache()


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if dist.get_world_size() != 4:
        raise RuntimeError(f"Expected 4 ranks, got {dist.get_world_size()}")

    init_usp_groups(sp_ulysses_size=SP_SIZE, sp_ring_size=1)

    # Parity must hold both for the default (shard ⊆ DP, SP replicated) and for a
    # shard that spans the SP dim (shard = world_size), which distributes the draft
    # across SP too. Grads/params must match the replicate reference either way.
    init_usp_hsdp_mesh()
    _run_grad_param_parity("default-shard")
    init_usp_hsdp_mesh(hsdp_shard_size=dist.get_world_size())
    _run_grad_param_parity("shard-spans-sp")

    # Restore the default mesh for the flex / MTP forward-backward smoke tests.
    init_usp_hsdp_mesh()
    _test_flex_dtensor_boundary()
    _test_mtp_hsdp_flex()

    if dist.get_rank() == 0:
        print(
            "PASSED: native HSDP parity (default + shard-spans-SP), DTensor "
            "FlexAttention boundary, and MTP USP warmup/forward/backward"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
