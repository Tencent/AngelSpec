"""GLM-5.3-Flash target reuse (models/target/glm53_target.py): the picks and latents against
their definition, vLLM's indexer kernels, the checkpoint, and a drafter trained on them."""

import os

import pytest
import torch
import torch.nn.functional as F

from angelspec.models.target import glm53_target
from angelspec.models.target.glm53_target import (
    Glm53TargetReuse,
    expand_pools,
    hadamard128,
    pool_keys,
    quantize_queries,
    top_pools,
)

GLM53_PATH = os.environ.get("GLM53_TARGET_MODEL", "zai-org/GLM-5.3-Flash")
# The hidden size a multiple of the activations' 128-wide quantization groups.
H, KPOOL, TOPK = 128, 4, 8


def _text(**kw):
    layer_types = ["linear_attention", "deepseek_sparse_attention"] * 3
    text = dict(
        model_type="glm5_next_text",
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
        hidden_size=H,
        q_lora_rank=32,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        rms_norm_eps=1e-5,
        index_topk=TOPK,
        index_kpool=KPOOL,
        index_n_heads=4,
        index_head_dim=128,
    )
    text.update(kw)
    return {"text_config": text}


def _reuse(seed=0, **kw):
    reuse = Glm53TargetReuse(_text(**kw), [1, 3], [1, 3], TOPK)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in reuse.named_parameters():
            values = torch.randn(param.shape, generator=g) * 0.2
            if name.endswith("input_norm"):
                values = values.abs() + 0.5
            param.copy_(values)
    return reuse


def _attention_input(reuse, layer, length, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = F.rms_norm(torch.randn(1, length, H, generator=g), (H,))
    return (x * reuse.parts[str(layer)].input_norm.float()).to(torch.bfloat16)


def test_short_rows_take_every_causal_token_and_long_rows_score_pools():
    reuse = _reuse()
    length = 40
    x = _attention_input(reuse, 3, length)
    anchors = torch.tensor([[5, 8, 37]])
    picks = reuse.candidates(3, x, anchors, None)
    assert picks.shape == (1, 3, TOPK + KPOOL - 1)
    # Rows 4 and 7 see 5 and 8 tokens, at most topk: every one of them.
    assert picks[0, 0][picks[0, 0] >= 0].tolist() == list(range(5))
    assert picks[0, 1][picks[0, 1] >= 0].tolist() == list(range(8))
    # Row 36 sees 37 tokens: 9 pools, the best 2 of them, then token 36 (37 % 4 = 1 tail).
    row = picks[0, 2]
    chosen = row[:TOPK][row[:TOPK] >= 0]
    assert len(chosen) == TOPK and int(chosen.max()) < 36
    assert row[TOPK:].tolist() == [36, -1, -1]
    part = reuse.parts["3"]
    k8, k_scale, first = reuse._pooled_keys(part, x[0], [(0, length)])
    q8, weights = reuse._queries(part, x[0, 36:37])
    logits = glm53_target.index_logits(
        q8, weights, k8, k_scale, torch.tensor([0]), torch.tensor([9])
    )[0]
    best = set(logits.topk(2).indices.tolist())
    assert {int(t) // KPOOL for t in chosen.tolist()} == best


def test_latents_are_the_normalized_kv_projection():
    reuse = _reuse()
    x = _attention_input(reuse, 1, 12)
    latent = reuse.latent(1, x)
    part = reuse.parts["1"]
    projected = F.linear(x, part.kv_a)
    want = glm53_target.rms_norm(projected, part.kv_norm, reuse.eps)
    torch.testing.assert_close(latent, want, rtol=0, atol=0)


def test_drives_a_grouped_drafter_through_the_wrapper():
    from angelspec.models.dflash import DFlashModel
    from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel

    reuse = _reuse()
    config = DFlashConfig(
        hidden_size=H,
        intermediate_size=128,
        num_hidden_layers=2,
        vocab_size=101,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
        num_target_layers=2,
        target_hidden_size=H,
        target_num_hidden_layers=6,
        target_layer_ids=[0, 2],
        mask_token_id=100,
        attention_type="glm53_mla",
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=512,
        qk_nope_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=6,
        context_local_window=2,
        target_index_layer=[1, 3],
        target_index_topk=TOPK,
        target_kv_layers=[1, 3],
        dsa_backend="grouped",
    )
    draft = DFlashDraftModel(config).float()
    draft.freeze_embedding()
    wrapper = DFlashModel(draft_model=draft, block_size=4, num_anchors=3, causal_attn=True)
    wrapper.attach_target_indexers(reuse.indexers())
    wrapper.attach_target_kv_latents(reuse.kv_modules())
    assert wrapper._sparse_kv_attention
    length = 40
    hidden = [_attention_input(reuse, layer, length, seed=layer).float() for layer in (1, 3)]
    g = torch.Generator().manual_seed(5)
    loss = wrapper(
        input_ids=torch.randint(0, 100, (1, length), generator=g),
        hidden_states_list=hidden,
        loss_mask=torch.ones(1, length),
        lm_head_weight=torch.randn(101, H, generator=g),
        injected_anchors=torch.tensor([[6, 20, 31]]),
    )[0]
    loss.backward()
    assert torch.isfinite(loss)
    missing = [n for n, p in wrapper.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_the_capture_hands_over_the_attention_input(monkeypatch):
    glm53 = pytest.importorskip("vllm.models.glm5next.common.model")
    from angelspec.inference.engine import vllm_runtime_patch as patch

    cls = glm53.Glm5NextModel

    def fake_forward(self, x):
        # Glm5NextModel.forward's captures: each listed layer's incoming stream, then the
        # end layer's.
        aux = []
        for idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            if idx in self.aux_hidden_state_layers:
                aux.append(x + 100 * idx)
            x = layer(x)
        if self.end_layer in self.aux_hidden_state_layers:
            aux.append(x - 1)
        return x, aux

    class Attn(torch.nn.Module):
        def forward(self, hidden_states, positions):
            return hidden_states + 1

    class Layer(torch.nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx, self.self_attn = idx, Attn()

        def forward(self, x):
            return self.self_attn(hidden_states=x * 2 + self.idx, positions=None)

    monkeypatch.setattr(cls, "forward", fake_forward)
    monkeypatch.setattr(cls, "_angelspec_attn_input", False, raising=False)
    assert patch.apply_glm53_attention_input_capture()
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.layers = torch.nn.ModuleList(Layer(i) for i in range(4))
    model.start_layer, model.end_layer, model.aux_hidden_state_layers = 0, 4, (1, 3, 4)
    x = torch.randn(3, 2)
    plain_out, plain = model(x)
    monkeypatch.setenv(patch.GLM53_ATTN_INPUT_AUX_ENV, "1,3")
    out, aux = model(x)
    torch.testing.assert_close(out, plain_out)
    stream_1 = Layer(0)(x)
    stream_3 = Layer(2)(Layer(1)(stream_1))
    torch.testing.assert_close(aux[0], stream_1 * 2 + 1)
    torch.testing.assert_close(aux[1], stream_3 * 2 + 3)
    torch.testing.assert_close(aux[2], plain[2])
    monkeypatch.setenv(patch.GLM53_ATTN_INPUT_AUX_ENV, "")
    for got, want in zip(model(x)[1], plain):
        torch.testing.assert_close(got, want)


cuda_vllm = pytest.mark.skipif(
    not torch.cuda.is_available() or glm53_target._kpool_kernels() is None,
    reason="needs CUDA and vLLM's GLM-5.3 kernels",
)


@cuda_vllm
def test_query_quantization_matches_vllm():
    g = torch.Generator().manual_seed(0)
    query = (torch.randn(300, 128, generator=g) * 3).to(torch.bfloat16).cuda()
    kernel_q, kernel_s = quantize_queries(query)
    rotated = hadamard128(query).to(torch.bfloat16).float()
    torch_q, torch_s = glm53_target._absmax_fp8(rotated)
    assert torch.equal(kernel_s, torch_s)
    assert torch.equal(kernel_q.view(torch.uint8), torch_q.view(torch.uint8))


@cuda_vllm
def test_pooling_matches_vllm():
    g = torch.Generator().manual_seed(1)
    keys = torch.randn(500, KPOOL, 128, generator=g).to(torch.bfloat16).cuda()
    gate = (torch.randn(500, KPOOL, 128, generator=g) * 2).to(torch.bfloat16).cuda()
    ape = (torch.randn(KPOOL, 128, generator=g)).cuda()
    kernel_k, kernel_s = pool_keys(keys, gate, ape)
    weights = torch.softmax(gate.float() + ape, dim=1)
    pooled = (weights * keys.float()).sum(1).to(torch.bfloat16).float()
    torch_k, torch_s = glm53_target._absmax_fp8(hadamard128(pooled).to(torch.bfloat16).float())
    # Triton's exp is not torch's: a pooled value can round to the next bf16.
    assert float((kernel_s != torch_s.squeeze(-1)).float().mean()) < 1e-2
    mismatch = (kernel_k.float() != torch_k.float()).float().mean()
    assert float(mismatch) < 1e-2


@cuda_vllm
def test_top_k_and_expansion_match_vllm():
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(64, 900, generator=g).cuda()
    starts = torch.randint(0, 100, (64,), generator=g).cuda()
    ends = starts + torch.randint(1, 800, (64,), generator=g).cuda()
    kernel = top_pools(logits, starts, ends, 128)
    cpu = top_pools(logits.cpu(), starts.cpu(), ends.cpu(), 128)
    for a, b in zip(kernel.cpu().tolist(), cpu.tolist()):
        assert sorted(a) == sorted(b)
    visible = (ends - starts) * KPOOL + torch.randint(0, KPOOL, (64,), generator=g).cuda()
    assert torch.equal(
        expand_pools(kernel, visible, KPOOL).cpu(),
        expand_pools(kernel.cpu(), visible.cpu(), KPOOL),
    )


@pytest.mark.skipif(not os.path.isdir(GLM53_PATH), reason="no GLM-5.3-Flash checkpoint")
def test_loads_the_dsa_layers_of_the_checkpoint():
    reuse = Glm53TargetReuse.from_pretrained(
        GLM53_PATH, [35, 39, 43], [35, 39, 43], 2048, device="cpu"
    )
    part = reuse.parts["43"]
    assert part.kv_a.shape == (512, 4096) and part.q_a.shape == (1536, 4096)
    assert part.wk_weights_proj.shape == (128 + 32, 4096) and part.ape.dtype == torch.float32
    assert float(part.input_norm.float().abs().mean()) > 0
    with pytest.raises(ValueError, match="linear_attention"):
        Glm53TargetReuse.from_config(GLM53_PATH, [34], [], None, device="cpu")
