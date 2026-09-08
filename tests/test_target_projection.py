"""Regression tests for supervised-row teacher projection with dense TTT lookup."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from angelspec.models.eagle3 import compute_target_p_padded


@pytest.mark.parametrize("layout", ["sparse", "all", "empty", "last"])
@pytest.mark.parametrize("chunk_size", [1, 4, 4096])
def test_precomputed_target_matches_dense_across_depths(layout, chunk_size):
    torch.manual_seed(7)
    batch, seq, hidden, vocab, depth = 2, 9, 8, 13, 4
    hs = torch.randn(batch, seq, hidden)
    teacher = torch.randn(vocab, hidden)
    t2d = torch.arange(vocab) % 3 != 0
    mask = torch.zeros(batch, seq)
    if layout == "all":
        mask.fill_(1)
    elif layout == "sparse":
        mask[0, [0, 3, 8]] = 1
        mask[1, [1, 5, 7]] = 1
    elif layout == "last":
        mask[:, -1] = 1

    linear = F.linear
    with patch("angelspec.models.eagle3.F.linear", wraps=linear) as project:
        result = compute_target_p_padded(hs, teacher, t2d, mask, depth, chunk_size)
    # Both vocabulary projections must operate only on supervised rows, in chunks.
    projected_rows = {}
    for call in project.call_args_list:
        x, weight = call.args
        if layout == "all" and x.ndim == 3:
            assert x.shape == hs.shape
        else:
            assert x.ndim == 2
            assert x.shape[0] <= chunk_size
        rows = x.numel() // hidden
        projected_rows[weight.shape[0]] = projected_rows.get(weight.shape[0], 0) + rows
    expected_rows = int(mask.sum())
    if expected_rows:
        assert projected_rows == {vocab: expected_rows, int(t2d.sum()): expected_rows}
    else:
        assert not project.called

    full_logits = linear(hs, teacher)
    dense_probs = F.softmax(linear(hs, teacher[t2d]).float(), dim=-1)
    expected_mask = mask * t2d[full_logits.argmax(-1)]
    torch.testing.assert_close(result.position_mask, expected_mask)
    assert result.target_p_padded.shape == (batch, seq + depth, int(t2d.sum()))
    assert result.target_p_padded.dtype == torch.float32
    assert not result.target_p_padded.requires_grad
    torch.testing.assert_close(
        result.target_p_padded[:, :seq][mask.bool()], dense_probs[mask.bool()]
    )
    assert torch.count_nonzero(result.target_p_padded[:, :seq][~mask.bool()]) == 0
    assert torch.count_nonzero(result.target_p_padded[:, seq:]) == 0

    if hasattr(result, "coverage_padded"):
        coverage = full_logits.softmax(-1)[..., t2d].sum(-1)
        torch.testing.assert_close(
            result.coverage_padded[:, :seq][mask.bool()], coverage[mask.bool()]
        )
        assert torch.count_nonzero(result.coverage_padded[:, :seq][~mask.bool()]) == 0

    # Shift the mask and target lookup together, as the TTT caller does.
    dense_padded = F.pad(dense_probs, (0, 0, 0, depth))
    shifted_mask = F.pad(expected_mask, (0, depth))
    student_hs = torch.randn(batch, seq, hidden, requires_grad=True)
    student_head = torch.randn(int(t2d.sum()), hidden, requires_grad=True)
    losses = []
    for probabilities in (dense_padded, result.target_p_padded):
        loss = (student_hs.sum() + student_head.sum()) * 0
        for i in range(depth):
            active = shifted_mask[:, i : i + seq].bool()
            logits = linear(student_hs[active], student_head)
            target = probabilities[:, i : i + seq][active]
            loss = loss - (target * logits.log_softmax(-1)).sum() / active.sum().clamp_min(1)
        losses.append(loss)
    torch.testing.assert_close(losses[0], losses[1])
    grads = [torch.autograd.grad(loss, (student_hs, student_head)) for loss in losses]
    for baseline, candidate in zip(*grads):
        torch.testing.assert_close(baseline, candidate)


@pytest.mark.parametrize("layout", ["sparse", "empty"])
def test_precomputed_target_preserves_ttt_parameter_gradients(layout):
    from transformers.models.llama.configuration_llama import LlamaConfig

    from angelspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
    from angelspec.models.eagle3 import Eagle3Model, PrecomputedTarget

    torch.manual_seed(13)
    batch, seq, hidden, vocab, draft_vocab, depth = 2, 12, 32, 64, 32, 4
    config = LlamaConfig(
        hidden_size=hidden,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        vocab_size=vocab,
        pad_token_id=0,
    )
    config.draft_vocab_size = draft_vocab
    draft = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
    model = Eagle3Model(draft, length=depth, attention_backend="sdpa")
    hs = torch.randn(batch, seq, hidden)
    teacher = torch.randn(vocab, hidden) * 0.1
    t2d = torch.arange(vocab) % 2 == 0
    mask = torch.zeros(batch, seq)
    if layout == "sparse":
        mask[0, [0, 4, 11]] = 1
        mask[1, [2, 8, 10]] = 1
    candidate = compute_target_p_padded(hs, teacher, t2d, mask, depth, chunk_size=3)
    dense = F.softmax(F.linear(hs, teacher[t2d]).float(), dim=-1)
    reference = PrecomputedTarget(F.pad(dense, (0, 0, 0, depth)), candidate.position_mask)
    if hasattr(candidate, "coverage_padded"):
        reference.coverage_padded = candidate.coverage_padded
    inputs = dict(
        input_ids=torch.randint(0, vocab, (batch, seq)),
        attention_mask=torch.ones(batch, seq, dtype=torch.long),
        loss_mask=mask,
        hidden_states=torch.randn(batch, seq, 3 * hidden),
    )
    outputs, gradients = [], []
    # This test checks the real multi-round model and autograd, independently
    # of compilation. Compiled loss paths have their own regression tests.
    with torch.compiler.set_stance("force_eager"):
        for target in (reference, candidate):
            model.zero_grad(set_to_none=True)
            result = model(target=target, **inputs)
            sum(result[0]).backward()
            outputs.append([loss.detach() for loss in result[0]])
            gradients.append(
                {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            )
    assert gradients[0].keys() == gradients[1].keys()
    for a, b in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(a, b)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name])
