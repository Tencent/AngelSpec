import pytest
import torch
import torch.nn.functional as F

from angelspec.models.ops.loss import (
    _kl_topk_variant_a,
    lk_tv_kl_per_pos,
    mtp_loss_from_hs,
)


def _reference_single_step(student_logits, teacher_logits, form, eta=3.0, topk=None):
    """lk_tv_kl_per_pos with autograd keeping every intermediate."""
    t_f32 = teacher_logits.float()
    if topk is not None:
        topk_idx = t_f32.topk(int(topk), dim=-1).indices
        t_f32 = t_f32.gather(-1, topk_idx)
        s_f32 = student_logits.gather(-1, topk_idx).float()
    else:
        s_f32 = student_logits.float()
    s_logp = F.log_softmax(s_f32, dim=-1)
    q = s_logp.exp()
    p = F.softmax(t_f32, dim=-1)
    tv = 0.5 * (p - q).abs().sum(dim=-1)
    if form == "tv":
        return tv, tv
    kl = (p * (p.clamp_min(1e-9).log() - s_logp)).sum(dim=-1)
    if form == "lk":
        lam = torch.exp(-eta * (1.0 - tv).detach().clamp(0.0, 1.0))
        return lam * kl + (1.0 - lam) * tv, tv
    return kl, tv


def _reference_variant_a(student_logits, teacher_logits, topk):
    t_f32 = teacher_logits.float()
    topk_idx = t_f32.topk(topk, dim=-1).indices
    t_logp = F.log_softmax(t_f32.gather(-1, topk_idx), dim=-1)
    s_logp = F.log_softmax(student_logits.gather(-1, topk_idx).float(), dim=-1)
    return (t_logp.exp() * (t_logp - s_logp)).sum(-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("topk", [None, 7])
@pytest.mark.parametrize("form", ["tv", "kl", "lk"])
def test_single_step_terms_are_bitwise_unchanged(form, topk, dtype):
    torch.manual_seed(3)
    student = torch.randn(9, 50).to(dtype)
    teacher = torch.randn(9, 50).to(dtype)
    probe = torch.randn(9)
    results = []
    for fn in (_reference_single_step, lk_tv_kl_per_pos):
        logits = student.clone().requires_grad_()
        ell, tv = fn(logits, teacher, form, topk=topk)
        ((ell + tv) * probe).sum().backward()
        results.append((ell.detach(), tv.detach(), logits.grad))
    for left, right in zip(*results):
        assert torch.equal(left, right)


def test_variant_a_kl_is_bitwise_unchanged():
    torch.manual_seed(4)
    teacher = torch.randn(9, 50).to(torch.bfloat16)
    results = []
    for fn in (_reference_variant_a, _kl_topk_variant_a):
        logits = torch.randn(9, 50, generator=torch.Generator().manual_seed(5))
        logits = logits.to(torch.bfloat16).requires_grad_()
        kl = fn(logits, teacher, 7)
        kl.sum().backward()
        results.append((kl.detach(), logits.grad))
    for left, right in zip(*results):
        assert torch.equal(left, right)


def test_single_step_keeps_no_vocab_sized_tensor_for_backward():
    logits = torch.randn(16, 4096, requires_grad=True)
    teacher = torch.randn(16, 4096)
    shapes = []

    def pack(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        ell, _ = lk_tv_kl_per_pos(logits, teacher, "tv", topk=32)
    assert (16, 32) in shapes and not any(4096 in shape for shape in shapes)
    ell.sum().backward()
    assert logits.grad.count_nonzero() <= 16 * 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled CE metric runs on CUDA")
def test_compiled_ce_metric_matches_eager():
    from angelspec.models.ops.loss import _ce_from_logits, _compiled_ce_from_logits

    torch.manual_seed(5)
    # A step's last chunk is shorter than the others.
    for rows in (1024, 333):
        logits = torch.randn(rows, 32000, device="cuda").to(torch.bfloat16)
        target = torch.randint(32000, (rows,), device="cuda")
        with torch.no_grad():
            torch.testing.assert_close(
                _compiled_ce_from_logits(logits, target), _ce_from_logits(logits, target)
            )


@pytest.mark.parametrize(
    "form,topk,variant",
    [
        (None, None, "a"),
        (None, 5, "a"),
        (None, 5, "b"),
        ("tv", 5, "a"),
        ("kl", 5, "a"),
        ("lk", None, "a"),
    ],
)
@pytest.mark.parametrize("ce_grad", [False, True])
def test_checkpoint_values_gradients_and_updates(monkeypatch, form, topk, variant, ce_grad):
    monkeypatch.setenv("ANGELSPEC_MTP_LOSS_CHUNK", "3")
    torch.manual_seed(49)
    hidden = torch.randn(11, 8)
    teacher = torch.randn_like(hidden)
    weight = torch.randn(17, 8)
    ids = torch.tensor([0, 2, 3, 5, 7, 8, 10])
    labels = torch.randint(17, (11,))
    outputs, gradients, updates = [], [], []
    for checkpointed in (False, True):
        hs = hidden.clone().requires_grad_()
        w = weight.clone().requires_grad_()
        result = mtp_loss_from_hs(
            hs,
            teacher,
            ids,
            w,
            weight,
            use_kl=True,
            kl_topk=topk,
            kl_variant=variant,
            gt_labels_flat=labels,
            ce_use_ground_truth=True,
            single_step_form=form,
            return_per_pos=form is not None,
            ce_requires_grad=ce_grad,
            checkpoint_chunks=checkpointed,
        )
        assert result[0].requires_grad == ce_grad
        loss = result[1] + (result[0] if ce_grad else 0)
        if form is not None:
            loss = loss + (result[5] * result[6]).sum()
        loss.backward()
        outputs.append([x.detach() for x in result])
        gradients.append((hs.grad.clone(), w.grad.clone()))
        torch.optim.SGD([hs, w], lr=0.001).step()
        updates.append((hs.detach(), w.detach()))
    for pair in (outputs, gradients, updates):
        for left, right in zip(*pair):
            torch.testing.assert_close(left, right)


def test_zero_weight_ce_keeps_metrics_and_tv_gradient(monkeypatch):
    monkeypatch.setenv("ANGELSPEC_MTP_LOSS_CHUNK", "2")
    torch.manual_seed(11)
    hidden = torch.randn(7, 8)
    teacher = torch.randn_like(hidden)
    head = torch.randn(13, 8)
    results = []
    for ce_grad in (True, False):
        hs = hidden.clone().requires_grad_()
        out = mtp_loss_from_hs(
            hs,
            teacher,
            torch.arange(7),
            head,
            head,
            single_step_form="tv",
            kl_topk=5,
            ce_requires_grad=ce_grad,
            checkpoint_chunks=not ce_grad,
        )
        (out[1] + out[0] * 0).backward()
        results.append((out[0].detach(), hs.grad))
    for left, right in zip(*results):
        torch.testing.assert_close(left, right)


def test_checkpoint_does_not_save_vocab_sized_intermediates(monkeypatch):
    monkeypatch.setenv("ANGELSPEC_MTP_LOSS_CHUNK", "4")
    hs = torch.randn(24, 8, requires_grad=True)
    teacher = torch.randn_like(hs)
    head = torch.randn(4096, 8)
    counts = []
    for checkpointed in (False, True):
        saved = []

        def pack(tensor, saved=saved):
            if tensor.ndim == 2 and tensor.shape[-1] == 4096:
                saved.append(tensor.numel() * tensor.element_size())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            out = mtp_loss_from_hs(
                hs,
                teacher,
                torch.arange(24),
                head,
                head,
                single_step_form="tv",
                kl_topk=32,
                return_per_pos=True,
                checkpoint_chunks=checkpointed,
            )
        counts.append(sum(saved))
        del out
    assert counts[0] > 0
    assert counts[1] == 0


def test_empty_and_eval(monkeypatch):
    monkeypatch.setenv("ANGELSPEC_MTP_LOSS_CHUNK", "2")
    hs = torch.randn(5, 8)
    head = torch.randn(13, 8)
    with torch.no_grad():
        for ids in (torch.empty(0, dtype=torch.long), torch.arange(5)):
            out = mtp_loss_from_hs(
                hs,
                hs,
                ids,
                head,
                head,
                single_step_form="tv",
                return_per_pos=True,
                checkpoint_chunks=True,
                ce_requires_grad=False,
            )
            assert out[3].item() == ids.numel()
            assert not any(t.requires_grad for t in out)


def test_ttt_checkpointed_loss_parity(monkeypatch):
    from angelspec.models.draft.mtp import MTPDraftModel
    from angelspec.models.mtp import MTPModel
    from tests.test_mtp import _tiny_config

    monkeypatch.setenv("ANGELSPEC_MTP_LOSS_CHUNK", "2")
    torch.manual_seed(18)
    cfg = _tiny_config()
    head = torch.randn(cfg.vocab_size, cfg.hidden_size)
    models = []
    for checkpointed in (False, True):
        draft = MTPDraftModel(cfg, attention_backend="sdpa")
        draft.set_lm_head_weight(head)
        models.append(
            MTPModel(
                draft,
                length=3,
                attention_backend="sdpa",
                gradient_checkpointing=True,
                on_policy=True,
                single_step_form="tv",
                kl_topk=5,
                e2e_weighting=True,
                ce_requires_grad=not checkpointed,
                loss_checkpoint_chunks=checkpointed,
            )
        )
    models[1].load_state_dict(models[0].state_dict())
    ids = torch.randint(1, cfg.vocab_size, (1, 8))
    hidden = torch.randn(1, 8, cfg.hidden_size)
    target = torch.randn_like(hidden)
    values = []
    for model in models:
        result = model(
            input_ids=ids,
            hidden_states=hidden,
            attention_mask=torch.ones_like(ids),
            target_hidden_states=target,
            target_lm_head_weight=head,
            loss_mask=torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0]]),
        )
        loss = result[6] + sum(result[0]) * 0
        loss.backward()
        values.append(loss.detach())
    torch.testing.assert_close(*values)
    for left, right in zip(models[0].parameters(), models[1].parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad)
