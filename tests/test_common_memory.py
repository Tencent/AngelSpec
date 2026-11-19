import copy

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("objective", ["decay", "dpace"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"kl_loss_weight": 0.6, "kl_topk": 5},
        {"lk_loss_weight": 0.7, "lk_loss_type": "hybrid"},
        {"lk_loss_weight": 1.0, "lk_loss_type": "alpha", "ce_loss_alpha": 0.0},
        {"l1_loss_alpha": 0.5},
        {"e2e_tv_loss_weight": 0.4},
        {"kl_loss_weight": 0.5, "lk_loss_weight": 0.6, "e2e_tv_loss_weight": 0.3},
    ],
)
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("selector", [False, True])
def test_dfly_loss_and_grad_parity(monkeypatch, objective, options, masked, selector):
    from angelspec.models.dfly import DFlyModel
    from angelspec.models.draft.dfly import DFlyDraftModel
    from tests.test_dfly import H, V, _config

    torch.manual_seed(41)
    draft = DFlyDraftModel(
        _config(enable_path_selector=selector, path_selector_top_k=4, path_selector_rank=8)
    )
    draft.freeze_embedding()
    with torch.no_grad():
        draft.hidden_correction.down_proj.weight.normal_(std=0.01)
    original = DFlyModel(draft, block_size=4, num_anchors=3, loss_objective=objective, **options)
    optimized = copy.deepcopy(original)
    optimized.loss_checkpoint_chunks = True
    optimized.draft_model.context_chunk_size = 5
    optimized.draft_model.context_checkpoint = True
    ids = torch.randint(0, V, (2, 16))
    data = dict(
        input_ids=ids,
        hidden_states_list=[torch.randn(2, 16, H) for _ in range(2)],
        loss_mask=torch.zeros(2, 16) if masked else torch.ones(2, 16),
        lm_head_weight=torch.randn(V, H) * 0.1,
        last_hidden_states=torch.randn(2, 16, H),
        injected_anchors=torch.tensor([[0, 5, 10], [0, 5, 10]]),
        injected_keep_mask=torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool),
        return_draft=True,
    )
    outputs = []
    for model, chunk in [(original, "0"), (optimized, "8")]:
        monkeypatch.setenv("ANGELSPEC_DFLASH_LOSS_CHUNK", chunk)
        out = model(**data)
        out[0].backward()
        outputs.append(out)
    for left, right in zip(outputs[0][:5], outputs[1][:5]):
        torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-5)
    for key in outputs[0][5]:
        torch.testing.assert_close(outputs[0][5][key], outputs[1][5][key], rtol=2e-4, atol=2e-5)
    assert torch.equal(outputs[0][6]["proposals"], outputs[1][6]["proposals"])
    assert all(not value.requires_grad for value in outputs[1][5].values())
    for (name, left), (_, right) in zip(original.named_parameters(), optimized.named_parameters()):
        if left.grad is not None:
            if right.grad is None:
                assert torch.count_nonzero(left.grad) == 0, name
            else:
                torch.testing.assert_close(left.grad, right.grad, rtol=3e-4, atol=3e-5, msg=name)
    for model in (original, optimized):
        torch.optim.SGD(model.parameters(), lr=0.001).step()
    for left, right in zip(original.parameters(), optimized.parameters()):
        torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-5)


def test_dfly_context_chunks_gradients():
    from functools import partial

    from angelspec.models.draft.dfly import DFlyDraftModel
    from angelspec.models.ops.memory_chunks import chunked_context_call
    from tests.test_dfly import H, _config

    torch.manual_seed(5)
    models = [DFlyDraftModel(_config())]
    models.append(copy.deepcopy(models[0]))
    data = torch.randn(2, 13, 2, H)
    losses = []
    for model, chunk in zip(models, [0, 3]):
        context = data.clone().requires_grad_()
        base = chunked_context_call(model._project_base_context, (context,), chunk, chunk > 0)
        parts = [
            chunked_context_call(
                partial(model._build_layer_context, layer_idx=i), (context, base), chunk, chunk > 0
            )
            for i in range(2)
        ]
        loss = sum(value.square().mean() for value in parts)
        loss.backward()
        losses.append((loss, context.grad))
    for left, right in zip(*losses):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-4)
    for left, right in zip(models[0].parameters(), models[1].parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("postnorm", [False, True])
def test_mtp_alignment_chunk_parity(postnorm):
    from angelspec.training.mtp_trainer import align_mtp_inputs

    torch.manual_seed(1)
    hs = torch.randn(2, 13, 8)
    ids = torch.arange(26).reshape(2, 13)
    mask = torch.ones_like(ids)
    docs = (ids % 13) // 4
    norm = torch.nn.RMSNorm(8)
    a = align_mtp_inputs(
        ids, hs, mask, verifier_norm=norm, draft_input_postnorm=postnorm, ctx_doc_ids=docs
    )
    b = align_mtp_inputs(
        ids,
        hs,
        mask,
        verifier_norm=norm,
        draft_input_postnorm=postnorm,
        ctx_doc_ids=docs,
        chunk_size=3,
    )
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("form", ["tv", "kl", "lk"])
def test_topk_gather_before_cast_parity(form):
    from angelspec.models.ops.loss import lk_tv_kl_per_pos

    torch.manual_seed(2)
    s = torch.randn(5, 17, dtype=torch.bfloat16, requires_grad=True)
    t = torch.randn_like(s)
    index = t.float().topk(5, dim=-1).indices
    a = lk_tv_kl_per_pos(s, t, form, topk=5)
    b = lk_tv_kl_per_pos(s.float().gather(-1, index), t.float().gather(-1, index), form)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)
    ga = torch.autograd.grad(a[0].sum(), s, retain_graph=True)[0]
    gb = torch.autograd.grad(b[0].sum(), s)[0]
    torch.testing.assert_close(ga, gb)


def test_dflash_chunk_checkpoint_saved_tensors(monkeypatch):
    from tests.test_dflash import _make_dflash_model

    torch.manual_seed(11)
    original = _make_dflash_model(V=4096)
    optimized = copy.deepcopy(original)
    optimized.loss_checkpoint_chunks = True
    saved_sizes = []
    data = dict(
        input_ids=torch.randint(0, 4096, (1, 16)),
        hidden_states_list=[torch.randn(1, 16, 64) for _ in range(2)],
        loss_mask=torch.ones(1, 16),
        lm_head_weight=torch.randn(4096, 64) * 0.1,
        injected_anchors=torch.tensor([[0, 4, 8, 10]]),
        injected_keep_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    for model in (original, optimized):
        monkeypatch.setenv("ANGELSPEC_DFLASH_LOSS_CHUNK", "4")
        saved = []

        def pack(tensor, saved=saved):
            if tensor.ndim >= 2 and tensor.shape[-1] == 4096:
                saved.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            output = model(**data)
        saved_sizes.append(sum(saved))
        del output
    assert saved_sizes[0] > 0 and saved_sizes[1] == 0


def _chunked_sp_worker(rank, rendezvous, output):
    from pathlib import Path
    from types import SimpleNamespace

    import torch.distributed as dist

    from angelspec.models.dflash import DFlashModel
    from angelspec.models.ops.dflash_chunked_loss import chunked_dflash_loss

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(4)
        head = torch.randn(13, 8) * 0.1
        all_hidden = torch.randn(1, 4, 4, 8)
        all_teacher = torch.randn_like(all_hidden)
        all_targets = torch.randint(0, 13, (1, 4, 4))
        results = []
        for world in (1, 2):
            n = 4 if world == 1 else 2
            sl = slice(None) if world == 1 else slice(rank * 2, rank * 2 + 2)
            hidden = all_hidden[:, sl].clone().reshape(1, n * 4, 8).requires_grad_()
            model = DFlashModel(
                SimpleNamespace(),
                block_size=4,
                kl_loss_weight=0.6,
                kl_topk=5,
                e2e_tv_loss_weight=0.2,
                sp_size=world,
                sp_rank=rank,
                sp_group=dist.group.WORLD if world == 2 else None,
            )
            model._compute_draft_logits = lambda h, w, previous, blocks, block_start=0: F.linear(
                h, w
            )
            indices = torch.arange(n * 4).reshape(1, n, 4)
            keep = torch.ones(1, n, dtype=torch.bool)
            if world == 2 and rank == 1:
                keep.zero_()
            if world == 1:
                keep[:, 2:] = False
            result = chunked_dflash_loss(
                model,
                hidden,
                all_targets[:, sl],
                all_targets[:, sl],
                keep,
                torch.ones_like(indices, dtype=torch.bool),
                indices,
                torch.ones(1, n * 4),
                head,
                n,
                indices[:, :, 0],
                None,
                all_teacher[:, sl],
                None,
                4,
                False,
            )
            result[0].backward()
            results.append((result[0].detach(), hidden.grad.clone()))
        loss = results[1][0].clone()
        dist.all_reduce(loss)
        torch.testing.assert_close(loss, results[0][0], rtol=1e-4, atol=1e-5)
        reference = results[0][1].reshape(1, 4, 4, 8)[:, rank * 2 : rank * 2 + 2].reshape(1, 8, 8)
        torch.testing.assert_close(results[1][1], reference, rtol=1e-4, atol=1e-5)
        Path(output, str(rank)).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_chunked_loss_sp_global_normalization(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _chunked_sp_worker,
        args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert (tmp_path / "0").read_text() == (tmp_path / "1").read_text() == "ok"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prealigned_teacher_and_eval(monkeypatch, dtype):
    from angelspec.models.dfly import DFlyModel
    from angelspec.models.draft.dfly import DFlyDraftModel
    from tests.test_dfly import H, V, _config

    torch.manual_seed(19)
    draft = DFlyDraftModel(_config()).to(dtype)
    model = DFlyModel(draft, block_size=4, num_anchors=2, kl_loss_weight=0.5, kl_topk=5)
    hidden = torch.randn(1, 8, H, dtype=dtype)
    anchors = torch.tensor([[0, 4]])
    keep = torch.ones(1, 2, dtype=torch.bool)
    model._draft_backbone = lambda *args, **kwargs: (hidden, anchors, keep, 2)
    model._compute_draft_logits = lambda h, w, p, n, block_start=0: F.linear(h, w)
    head = torch.randn(V, H, dtype=dtype) * 0.1
    ids = torch.randint(V, (1, 12))
    kwargs = dict(
        input_ids=ids,
        hidden_states_list=[torch.randn(1, 12, H, dtype=dtype) for _ in range(2)],
        loss_mask=torch.ones(1, 12),
        lm_head_weight=head,
        teacher_hidden_at_labels=torch.randn(1, 2, 4, H, dtype=dtype),
    )
    with torch.no_grad():
        monkeypatch.setenv("ANGELSPEC_DFLASH_LOSS_CHUNK", "0")
        left = model(**kwargs)
        monkeypatch.setenv("ANGELSPEC_DFLASH_LOSS_CHUNK", "4")
        model.loss_checkpoint_chunks = True
        right = model(**kwargs)
    for a, b in zip(left[:5], right[:5]):
        torch.testing.assert_close(a, b, rtol=0.02 if dtype == torch.bfloat16 else 1e-4, atol=1e-5)
    assert all(not value.requires_grad for value in right[5].values())
