"""Target KV reuse and the MTP fusion under Ulysses SP match the single-rank latents, loss and
gradients of the DFly2 layout."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from angelspec.data.positions import build_position_ids
from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from angelspec.models.dspark import DSparkModel
from angelspec.models.ops.tilelang_dsa import tilelang_dsa_available
from angelspec.models.target.target_kv_latent import TargetKVLatent
from angelspec.utils import distributed as groups
from tests.test_dflash_target_index import (
    IMG_END,
    IMG_NL,
    IMG_PAD,
    IMG_START,
    INDEX_LAYER,
    RECIPE,
    _image,
    _random_indexer,
    _target_config,
)

WORLD = 2
V, H, L, BS, TOPK = 128, 128, 256, 4, 16
KV_LAYERS = [4, 5]
RANK, ROPE = 512, 64  # the sparse MLA kernel's fixed latent and rope widths
MROPE_SECTION = [12, 10, 10]


def _init(rank, root, use_cuda):
    torch.set_num_threads(1)
    if use_cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if use_cuda else "gloo",
        init_method=f"file://{root}/pg",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=300),
    )
    groups._SP_ULYSSES_GROUP = dist.group.WORLD
    return torch.device("cuda", rank) if use_cuda else torch.device("cpu")


def _kv_target_config(mrope):
    rope = {"rope_theta": 10000.0, "rope_type": "default"}
    if mrope:
        rope.update(mrope_section=MROPE_SECTION, mrope_interleaved=True)
    return _target_config(
        mrope=mrope,
        hidden_size=H,
        q_lora_rank=64,
        kv_lora_rank=RANK,
        qk_rope_head_dim=ROPE,
        rope_parameters=rope,
    )


def _random_latents(mrope):
    modules = []
    for seed, layer in enumerate(KV_LAYERS):
        module = TargetKVLatent(_kv_target_config(mrope), layer)
        g = torch.Generator().manual_seed(10 + seed)
        for param in module.parameters():
            param.data.copy_(torch.randn(param.shape, generator=g) * 0.5)
        modules.append(module)
    return modules


def _latent_worker(rank, root):
    _init(rank, root, use_cuda=False)
    try:
        g = torch.Generator().manual_seed(0)
        seq_len = 48
        local = slice(rank * seq_len // WORLD, (rank + 1) * seq_len // WORLD)
        for mrope in (False, True):
            module = _random_latents(mrope)[0]
            hidden = torch.randn(1, seq_len, H, generator=g)
            if mrope:
                input_ids = torch.tensor([[5] * 12 + _image(2, 3) + [5] * (seq_len - 22)])
                kw = dict(position_ids=build_position_ids(input_ids, RECIPE))
            else:
                input_ids = torch.randint(0, 100, (1, seq_len), generator=g)
                kw = dict(ctx_doc_ids=torch.tensor([[0] * 30 + [1] * (seq_len - 30)]))
            whole = module.latents(hidden, input_ids, **kw)
            sharded = module.latents(hidden[:, local], input_ids, sp_group=dist.group.WORLD, **kw)
            assert torch.equal(sharded, whole[:, local]), f"mrope={mrope}"
            with pytest.raises(ValueError, match="shard"):
                module.latents(hidden[:, :5], input_ids, sp_group=dist.group.WORLD, **kw)
        Path(root, f"rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


def _wrapper(backend, mrope, rank, sp_size):
    rope = {"rope_theta": 10000.0, "rope_type": "default"}
    images = {}
    if mrope:
        rope.update(mrope_section=MROPE_SECTION, mrope_interleaved=True)
        images = dict(
            image_start_token_id=IMG_START,
            image_end_token_id=IMG_END,
            image_newline_token_id=IMG_NL,
            image_token_id=IMG_PAD,
        )
    config = DSparkConfig(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=16,
        vocab_size=V,
        rms_norm_eps=1e-5,
        max_position_embeddings=4096,
        num_target_layers=2,
        target_hidden_size=H,
        target_num_hidden_layers=6,
        target_layer_ids=[layer - 1 for layer in KV_LAYERS],
        mask_token_id=V - 1,
        model_arch="dflash",
        attention_type="mla",
        q_lora_rank=64,
        kv_lora_rank=RANK,
        qk_nope_head_dim=64,
        qk_rope_head_dim=ROPE,
        v_head_dim=64,
        gated_mla=True,
        learnable_sink=True,
        rope_parameters=rope,
        target_index_layer=INDEX_LAYER,
        target_index_topk=TOPK,
        target_kv_layers=KV_LAYERS,
        mtp_fusion="slot0",
        mtp_fusion_init="random",
        dsa_backend=backend,
        enable_hidden_correction=True,
        markov_rank=0,
        enable_confidence_head=True,
        confidence_head_with_markov=False,
        **images,
    )
    torch.manual_seed(0)
    draft = DSparkDraftModel(config).to(torch.float32)
    draft.freeze_embedding()
    with torch.no_grad():  # off their identity inits, so every path carries a gradient
        for layer in draft.layers:
            layer.self_attn.learnable_sink_param.normal_()
        draft.hidden_correction.down_proj.weight.normal_(std=0.02)
        for name, param in draft.mtp_fusion.named_parameters():
            if "gate" in name:
                param.normal_()
    return DSparkModel(
        draft_model=draft,
        block_size=BS,
        num_anchors=32,
        loss_objective="dpace",
        ce_loss_alpha=0.1,
        kl_loss_weight=0.5,
        e2e_tv_loss_weight=1.0,
        confidence_head_alpha=1.0,
        causal_attn=True,
        sp_size=sp_size,
        sp_rank=rank,
        sp_group=dist.group.WORLD if sp_size > 1 else None,
    )


def _teacher_rows(last_hidden, anchors):
    """The trainer's per-anchor teacher rows: final hidden at anchor + k - 1."""
    offsets = torch.arange(BS, device=anchors.device).view(1, 1, -1)
    rows = (anchors.unsqueeze(-1) + offsets - 1).clamp(0, last_hidden.shape[1] - 1)
    gathered = last_hidden.gather(
        1, rows.flatten(1).unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1])
    )
    return gathered.view(*rows.shape, -1)


def _close(actual, expected, fp32, msg):
    a, b = actual.float().flatten(), expected.float().flatten()
    if fp32:
        scale = b.abs().max().item() + 1e-12
        assert (a - b).abs().max().item() <= 1e-4 * scale, msg
        return
    cos = F.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / (b.norm() + 1e-12)).item()
    assert cos >= 0.999 and rel <= 2e-2, f"{msg}: cos={cos:.5f} rel={rel:.4f}"


def _check_dspark(rank, device, backend, mrope, packed):
    g = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, 100, (1, L), generator=g)
    inputs = {}
    if mrope:
        image = _image(4, 6)  # positions [40, 70)
        input_ids[0, 40 : 40 + len(image)] = torch.tensor(image)
        inputs["position_ids"] = build_position_ids(input_ids, RECIPE)
        anchors = torch.cat([torch.arange(4, 36, 4), torch.linspace(72, L - BS - 1, 24).long()])
    else:
        anchors = torch.linspace(4, L - BS - 1, 32).long()
    if packed:
        split = 120
        inputs["ctx_doc_ids"] = torch.tensor([[0] * split + [1] * (L - split)])
        inputs["base_position_ids"] = torch.cat(
            [torch.arange(split), torch.arange(L - split)]
        ).unsqueeze(0)
        anchors = torch.cat(
            [
                torch.linspace(4, split - BS, 14).long(),
                torch.linspace(split, L - BS - 1, 18).long(),
            ]
        )
    loss_mask = torch.ones(1, L)
    loss_mask[:, :BS] = 0
    loss_mask[:, -10:] = 0
    hidden = [torch.randn(1, L, H, generator=g).to(device) for _ in range(2)]
    last_hidden = torch.randn(1, L, H, generator=g).to(device)
    norm = torch.nn.RMSNorm(H, eps=1e-5).requires_grad_(False)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.1 * torch.randn(H, generator=g))
    norm = norm.to(device)
    inputs.update(
        input_ids=input_ids,
        loss_mask=loss_mask,
        lm_head_weight=torch.randn(V, H, generator=g) * H**-0.5,
        injected_anchors=anchors.unsqueeze(0),
    )
    inputs = {name: t.to(device) for name, t in inputs.items()}
    indexer = _random_indexer(_target_config(mrope=mrope, hidden_size=H), topk=TOPK).to(device)
    latents = [module.to(device) for module in _random_latents(mrope)]

    reference = _wrapper(backend, mrope, rank=0, sp_size=1).to(device)
    usp = _wrapper(backend, mrope, rank=rank, sp_size=WORLD).to(device)
    usp.load_state_dict(reference.state_dict())
    for model in (reference, usp):
        model.attach_target_indexer(indexer)
        model.attach_target_kv_latents(latents)

    ref_out = reference(
        hidden_states_list=hidden, last_hidden_states=last_hidden, target_norm=norm, **inputs
    )
    ref_out[0].backward()
    local = slice(rank * L // WORLD, (rank + 1) * L // WORLD)
    out = usp(
        hidden_states_list=[h[:, local] for h in hidden],
        teacher_hidden_at_labels=_teacher_rows(
            last_hidden, inputs["injected_anchors"][:, rank::WORLD]
        ),
        target_norm=norm,
        context_pre_sharded=True,
        **inputs,
    )
    out[0].backward()

    case = f"{backend} mrope={mrope} packed={packed}"
    assert float(ref_out[5]["confidence_loss"]) > 0, case
    assert float(ref_out[5]["mtp_fusion_ratio"]) > 0, case
    # Each rank's loss is its share of the full-sequence mean, confidence term included.
    total = out[0].detach().clone()
    dist.all_reduce(total)
    fp32 = device.type == "cpu"
    torch.testing.assert_close(
        total, ref_out[0].detach(), rtol=1e-5 if fp32 else 1e-3, atol=0, msg=case
    )
    fused = 0
    for (name, p), (_, q) in zip(usp.named_parameters(), reference.named_parameters()):
        if q.grad is None:
            assert p.grad is None, f"{case} {name}"
            continue
        grad = p.grad.clone()
        dist.all_reduce(grad)
        _close(grad, q.grad, fp32, f"{case} {name}")
        fused += "mtp_fusion" in name
    assert fused, f"{case}: the MTP fusion got no gradient"


def _dspark_worker(rank, root, use_cuda, backend):
    device = _init(rank, root, use_cuda)
    try:
        for mrope, packed in ((False, True), (True, False)):
            _check_dspark(rank, device, backend, mrope, packed)
        Path(root, f"rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


def _spawn(tmp_path, worker, *args):
    mp.spawn(worker, args=(str(tmp_path), *args), nprocs=WORLD, join=True)
    assert all((tmp_path / f"rank{rank}.ok").is_file() for rank in range(WORLD))


def test_sharded_latents_match_the_whole_sequence(tmp_path):
    _spawn(tmp_path, _latent_worker)


def test_dfly2_layout_usp_matches_single_rank(tmp_path):
    _spawn(tmp_path, _dspark_worker, False, "torch")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_dfly2_layout_usp_matches_single_rank_nccl(tmp_path):
    if not tilelang_dsa_available():
        pytest.skip("TileLang DSA kernels are unavailable")
    _spawn(tmp_path, _dspark_worker, True, "tilelang")
