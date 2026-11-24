import copy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from angelspec.models.dflash import DFlashModel, _create_dflash_mask_mod
from angelspec.models.draft.dflare import DFlareDraftModel
from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from angelspec.models.draft.dfly import DFlyConfig, DFlyDraftModel
from angelspec.models.draft.mtp import MTPConfig, MTPDraftModel, MTPMLAAttention
from angelspec.models.mtp import MTPModel
from angelspec.models.ops.attention_mask import mtp_validity_mask
from angelspec.models.ops.sparse_indices import build_contiguous_causal_ranges


@pytest.fixture(scope="module", autouse=True)
def _compile_without_donated_buffers():
    # These tests take several gradients through one graph (retain_graph=True),
    # which a compiled backward with donated buffers rejects; recompile without.
    torch._dynamo.reset()
    with torch._functorch.config.patch(donated_buffer=False):
        yield
    torch._dynamo.reset()


def mtp_config(kind="gqa", topk=None):
    values = dict(
        hidden_size=32,
        target_hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        mlp_type="dense",
        max_position_embeddings=128,
    )
    if kind == "qwen35":
        values.update(rms_norm_type="qwen3_5", attn_output_gate=True, partial_rotary_factor=0.5)
    if kind == "mla":
        values.update(
            attention_type="mla",
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=8,
            index_topk=topk,
            index_head_dim=8,
            index_n_heads=2,
            gated_mla=True,
            learnable_sink=True,
        )
    return MTPConfig(**values)


def dense_mask(docs, valid):
    n = docs.shape[1]
    positions = torch.arange(n, device=docs.device)
    keep = positions[:, None] >= positions[None, :]
    keep = keep[None] & valid[:, None].bool()
    keep = keep & (docs[:, :, None] == docs[:, None, :]) & (docs[:, :, None] >= 0)
    return torch.zeros_like(keep[:, None], dtype=torch.float32).masked_fill(
        ~keep[:, None], -torch.inf
    )


@pytest.mark.parametrize("kind", ["gqa", "qwen35", "mla"])
@pytest.mark.parametrize("backend", ["sdpa", "fa", "flex_attention"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_mtp_wrapper_has_no_forbidden_input_gradient(kind, backend, checkpoint):
    torch.manual_seed(412)
    cfg = mtp_config(kind)
    draft = MTPDraftModel(cfg, attention_backend=backend)
    head = torch.randn(64, 32)
    draft.set_lm_head_weight(head)
    wrapper = MTPModel(
        draft,
        length=3,
        attention_backend=backend,
        gradient_checkpointing=checkpoint,
        on_policy=True,
    )
    inputs = torch.randn(2, 10, 32, requires_grad=True)
    docs = torch.tensor([[0] * 5 + [1] * 4 + [-1]] * 2)
    valid = docs >= 0
    outputs = []
    hook = draft.midlayer.self_attn.register_forward_hook(lambda m, a, out: outputs.append(out[0]))
    try:
        wrapper(
            input_ids=torch.randint(0, 64, (2, 10)),
            attention_mask=valid,
            target_hidden_states=torch.randn(2, 10, 32),
            target_lm_head_weight=head,
            loss_mask=valid.float(),
            hidden_states=inputs,
            ctx_doc_ids=docs,
        )
    finally:
        hook.remove()
    grad0 = torch.autograd.grad(outputs[-1][0, 1].sum(), inputs, retain_graph=True)[0]
    grad1 = torch.autograd.grad(outputs[-1][0, 6].sum(), inputs)[0]
    assert torch.count_nonzero(grad0[0, 2:]) == 0
    assert torch.count_nonzero(grad0[1]) == 0
    assert torch.count_nonzero(grad1[0, :5]) == 0
    assert torch.count_nonzero(grad1[0, 7:]) == 0
    assert torch.count_nonzero(grad0[0, :2]) > 0


@pytest.mark.parametrize("kind", ["gqa", "qwen35", "mla"])
@pytest.mark.parametrize("backend", ["fa", "flex_attention"])
def test_mtp_backend_loss_gradients_and_update_match(kind, backend):
    torch.manual_seed(51)
    cfg = mtp_config(kind)
    base = MTPDraftModel(cfg, attention_backend="sdpa")
    other = MTPDraftModel(cfg, attention_backend=backend)
    other.load_state_dict(base.state_dict())
    head = torch.randn(64, 32)
    docs = torch.tensor([[0] * 4 + [1] * 5 + [-1]] * 2)
    batch = dict(
        input_ids=torch.randint(0, 64, (2, 10)),
        attention_mask=docs >= 0,
        target_hidden_states=torch.randn(2, 10, 32),
        target_lm_head_weight=head,
        loss_mask=(docs >= 0).float(),
        hidden_states=torch.randn(2, 10, 32),
        ctx_doc_ids=docs,
    )
    losses = []
    for draft in (base, other):
        draft.set_lm_head_weight(head)
        wrapper = MTPModel(
            draft, length=3, attention_backend=draft.attention_backend, on_policy=True
        )
        loss = sum(wrapper(**batch)[0])
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1], atol=2e-5, rtol=2e-5)
    for p, q in zip(base.parameters(), other.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
    for draft in (base, other):
        torch.optim.SGD(draft.parameters(), lr=0.01).step()
    for p, q in zip(base.parameters(), other.parameters()):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("backend", ["sdpa", "fa", "flex_attention"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_mla_packed_short_documents_equal_dense(backend, cached, checkpoint):
    torch.manual_seed(13)
    cfg = mtp_config("mla", topk=4)
    sparse = MTPMLAAttention(cfg, backend)
    dense = copy.deepcopy(sparse)
    dense.use_indexer = False
    docs = torch.tensor([[0] * 3 + [1] * 3 + [2] * 3 + [3] + [-1] * 2] * 2)
    valid = docs >= 0
    mask = dense_mask(docs, valid)
    hs = torch.randn(2, 12, 32, requires_grad=True)
    ref_hs = hs.detach().clone().requires_grad_()
    positions = torch.arange(12)[None].expand(2, -1)
    with mock.patch.dict(
        "os.environ", {"ANGELSPEC_MTP_SPARSE_CHUNK_CHECKPOINT": str(int(checkpoint))}
    ):
        got, ck, cv, index = sparse(
            hs,
            attention_mask=mask,
            position_ids=positions,
            use_cache=cached,
            ctx_doc_ids=docs,
            return_index=True,
        )
        ref, rk, rv = dense(
            ref_hs, attention_mask=mask, position_ids=positions, use_cache=cached, ctx_doc_ids=docs
        )
        if cached:
            got, _, _ = sparse(
                hs,
                attention_mask=mask,
                position_ids=positions,
                use_cache=True,
                cache_keys=ck,
                cache_values=cv,
                ctx_doc_ids=docs,
                precomputed_index=index,
            )
            ref, _, _ = dense(
                ref_hs,
                attention_mask=mask,
                position_ids=positions,
                use_cache=True,
                cache_keys=rk,
                cache_values=rv,
                ctx_doc_ids=docs,
            )
        torch.testing.assert_close(got[valid], ref[valid], atol=2e-6, rtol=2e-5)
        seed = torch.randn_like(got) * valid.unsqueeze(-1)
        (got * seed).sum().backward()
        (ref * seed).sum().backward()
    torch.testing.assert_close(hs.grad, ref_hs.grad, atol=3e-6, rtol=3e-5)
    for (name, p), (_, q) in zip(sparse.named_parameters(), dense.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5, msg=name)
    for b in range(2):
        for row in range(12):
            selected = index[0][b, row]
            selected = selected[selected >= 0].long()
            assert bool((selected <= row).all())
            assert bool((docs[b, selected] == docs[b, row]).all())
            assert bool(valid[b, selected].all())


@pytest.mark.parametrize("sink", [False, True])
def test_mla_sparse_all_padding_is_finite(sink):
    cfg = mtp_config("mla", topk=4)
    cfg.learnable_sink = sink
    attn = MTPMLAAttention(cfg)
    hidden = torch.randn(1, 8, 32, requires_grad=True)
    mask = torch.full((1, 1, 8, 8), -torch.inf)
    out, _, _ = attn(
        hidden, attention_mask=mask, position_ids=torch.arange(8)[None], use_cache=True
    )
    out.sum().backward()
    assert bool(out.isfinite().all())
    assert torch.count_nonzero(out) == 0
    assert bool(hidden.grad.isfinite().all())


@pytest.mark.parametrize("backend", ["sdpa", "fa", "flex_attention"])
def test_absorbed_wrapper_preserves_validity(backend):
    cfg = mtp_config("mla", topk=4)
    cfg.kv_lora_rank = 512
    cfg.qk_rope_head_dim = 64
    cfg.index_head_dim = 64
    cfg.dsa_backend = "tilelang"
    draft = MTPDraftModel(cfg, attention_backend=backend)
    head = torch.randn(64, 32)
    draft.set_lm_head_weight(head)
    wrapper = MTPModel(draft, length=1, attention_backend=backend)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        torch.testing.assert_close(kwargs["valid_mask"], valid.bool())
        raise Captured()

    with mock.patch.object(draft.midlayer.self_attn.indexer, "sparse_indices", capture):
        with pytest.raises(Captured):
            wrapper(
                input_ids=torch.ones(2, 4).long(),
                attention_mask=valid,
                target_hidden_states=torch.randn(2, 4, 32),
                target_lm_head_weight=head,
                loss_mask=valid.float(),
                hidden_states=torch.randn(2, 4, 32),
            )


def test_indexer_ranges_handle_left_padding_and_reject_holes():
    starts, ends = build_contiguous_causal_ranges(5, valid_mask=torch.tensor([[0, 0, 1, 1, 0]]))
    assert starts.tolist() == [[0, 0, 2, 2, 0]]
    assert ends.tolist() == [[0, 0, 3, 4, 0]]
    with pytest.raises(ValueError, match="holes"):
        build_contiguous_causal_ranges(4, valid_mask=torch.tensor([[1, 0, 1, 1]]))


def test_mtp_rejects_invalid_backend_and_mask_contract():
    with pytest.raises(ValueError, match="attention_backend"):
        MTPDraftModel(mtp_config(), attention_backend="typo")
    with pytest.raises(ValueError, match="attention_backend"):
        MTPModel(SimpleNamespace(), attention_backend="typo")
    with pytest.raises(ValueError, match="cannot be flattened"):
        mtp_validity_mask(torch.zeros(1, 1, 4, 4), 1, 4, "cpu")
    with pytest.raises(ValueError, match="0 and 1"):
        mtp_validity_mask(torch.tensor([[0.0, -torch.inf]]), 1, 2, "cpu")


def dflash_draft(family):
    cfg_cls = DFlyConfig if family.startswith("dfly") else DFlashConfig
    cfg = cfg_cls(
        hidden_size=32,
        target_hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        num_target_layers=2,
        target_num_hidden_layers=12,
        mask_token_id=63,
        max_position_embeddings=128,
    )
    if family.endswith("mla"):
        for name, value in mtp_config("mla", topk=4).to_dict().items():
            if name in (
                "attention_type",
                "q_lora_rank",
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
                "index_topk",
                "index_head_dim",
                "index_n_heads",
                "gated_mla",
                "learnable_sink",
            ):
                setattr(cfg, name, value)
    cls = DFlyDraftModel if family.startswith("dfly") else DFlashDraftModel
    if family == "dflare":
        cls = DFlareDraftModel
    return cls(cfg)


@pytest.mark.parametrize("family", ["dflash", "dfly", "dflare", "dflash_mla", "dfly_mla"])
@pytest.mark.parametrize("causal", [False, True])
def test_dflash_full_backbone_mask_blocks_future_and_other_documents(family, causal):
    torch.manual_seed(84)
    draft = dflash_draft(family)
    wrapper = DFlashModel(draft, block_size=2, num_anchors=2, causal_attn=causal)
    hs = [torch.randn(2, 12, 32, requires_grad=True) for _ in range(2)]
    docs = torch.tensor([[0] * 6 + [1] * 6] * 2)
    out, *_ = wrapper._draft_backbone(
        torch.randint(0, 63, (2, 12)),
        hs,
        torch.ones(2, 12),
        ctx_doc_ids=docs,
        injected_anchors=torch.tensor([[2, 8]] * 2),
        injected_keep_mask=torch.ones(2, 2).bool(),
    )
    projection = torch.randn(32)
    grad0 = torch.autograd.grad((out[0, 1] * projection).sum(), hs[0], retain_graph=True)[0]
    grad1 = torch.autograd.grad((out[0, 3] * projection).sum(), hs[0])[0]
    assert torch.count_nonzero(grad0[0, 2:]) == 0
    assert torch.count_nonzero(grad0[1]) == 0
    assert torch.count_nonzero(grad1[0, :6]) == 0
    assert torch.count_nonzero(grad1[0, 8:]) == 0
    assert torch.count_nonzero(grad0[0, :2]) > 0


@pytest.mark.parametrize("causal", [False, True])
def test_dflash_sequence_parallelism_does_not_change_mask_semantics(causal):
    draft = dflash_draft("dflash")
    for sp in (1, 2, 4):
        wrapper = DFlashModel(draft, block_size=2, num_anchors=4, causal_attn=causal, sp_size=sp)
        assert wrapper.causal_attn is causal


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_dflash_mask_truth_table(causal, packed):
    from torch.nn.attention.flex_attention import create_mask

    anchors = torch.tensor([[2, 8, 0], [3, 7, 0]])
    active = torch.tensor([[True, True, False]] * 2)
    docs = torch.tensor([[0] * 6 + [1] * 4 + [-1] * 2] * 2) if packed else None
    mod = _create_dflash_mask_mod(anchors, active, 12, 2, docs, causal)
    actual = create_mask(mod, 2, 1, 6, 18, device="cpu")
    expected = torch.zeros_like(actual)
    for b in range(2):
        for q in range(6):
            anchor = int(anchors[b, q // 2])
            for key in range(18):
                context = key < anchor
                if context and packed:
                    context = docs[b, key] == docs[b, anchor] and docs[b, anchor] >= 0
                draft = key >= 12 and (key - 12) // 2 == q // 2
                if draft and causal:
                    draft = (key - 12) % 2 <= q % 2
                expected[b, 0, q, key] = bool(active[b, q // 2] and (context or draft))
    assert torch.equal(actual, expected)
