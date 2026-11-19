"""dsa_index_share: later MTP TTT steps reuse step 0's DSA top-k, or select their own.

Block 0 of the TTT cache is the step-0 K/V, so a step that selects its own top-k
must score its queries against the step-0 indexer keys, not keys projected from
its own hidden states.
"""

from pathlib import Path

import pytest
import torch

from angelspec.models.draft.auto import AutoDraftModelConfig
from angelspec.models.draft.mtp import MTPConfig, MTPDraftModel, MTPMLAAttention
from angelspec.models.mtp import MTPModel
from tests.test_attention_mask_correctness import mtp_config

CONFIGS = Path(__file__).resolve().parents[1] / "angelspec/config"


def _capture_indexer_calls(attn):
    calls = []

    def hook(module, args, kwargs, output):
        calls.append((args[0], kwargs["precomputed_keys"], output))

    return calls, attn.indexer.register_forward_hook(hook, with_kwargs=True)


def test_index_share_defaults_on_and_reaches_the_attention():
    assert MTPConfig().dsa_index_share is True
    path = CONFIGS / "mtp_hy_v4_flash_draft_config.json"
    assert AutoDraftModelConfig.from_file(str(path)).dsa_index_share is True
    cfg = mtp_config("mla", topk=4)
    assert MTPMLAAttention(cfg).dsa_index_share is True
    cfg.dsa_index_share = False
    assert MTPMLAAttention(cfg).dsa_index_share is False


def test_unshared_steps_select_against_step0_keys():
    torch.manual_seed(3)
    cfg = mtp_config("mla", topk=4)
    cfg.dsa_index_share = False
    unshared = MTPMLAAttention(cfg).eval()
    shared = MTPMLAAttention(mtp_config("mla", topk=4)).eval()
    shared.load_state_dict(unshared.state_dict())
    seq = 16
    args = dict(attention_mask=torch.ones(1, seq), position_ids=torch.arange(seq)[None])
    hs = [torch.randn(1, seq, 32) for _ in range(3)]

    calls, handle = _capture_indexer_calls(unshared)
    try:
        out, ck, cv, keys0 = unshared(hs[0], use_cache=True, return_index=True, **args)
        ref, rk, rv, index0 = shared(hs[0], use_cache=True, return_index=True, **args)
        assert keys0.shape == (1, seq, cfg.index_head_dim)
        assert calls[0][1] is keys0
        assert torch.equal(calls[0][2][0], index0[0])
        torch.testing.assert_close(out, ref, atol=0, rtol=0)

        changed = False
        for step in (1, 2):
            out, ck, cv = unshared(
                hs[step],
                cache_keys=ck,
                cache_values=cv,
                use_cache=True,
                precomputed_index=keys0,
                **args,
            )
            hidden, keys, index = calls[step]
            assert hidden is hs[step]
            assert keys is keys0
            # Applying the step's own selection through the shared path must
            # reproduce it exactly; reusing step 0's must not.
            ref, next_rk, next_rv = shared(
                hs[step],
                cache_keys=rk,
                cache_values=rv,
                use_cache=True,
                precomputed_index=index,
                **args,
            )
            stale, _, _ = shared(
                hs[step],
                cache_keys=rk,
                cache_values=rv,
                use_cache=True,
                precomputed_index=index0,
                **args,
            )
            rk, rv = next_rk, next_rv
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
            changed |= not torch.equal(out, stale)
        assert len(calls) == 3
        assert changed
    finally:
        handle.remove()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_ttt_loop_runs_the_indexer_once_or_every_step(checkpoint):
    torch.manual_seed(5)
    head = torch.randn(64, 32)
    batch = dict(
        input_ids=torch.randint(0, 64, (1, 12)),
        attention_mask=torch.ones(1, 12),
        target_hidden_states=torch.randn(1, 12, 32),
        target_lm_head_weight=head,
        loss_mask=torch.ones(1, 12),
        hidden_states=torch.randn(1, 12, 32),
    )
    state = MTPDraftModel(mtp_config("mla", topk=4)).state_dict()
    losses = {}
    for share in (True, False):
        cfg = mtp_config("mla", topk=4)
        cfg.dsa_index_share = share
        draft = MTPDraftModel(cfg)
        draft.load_state_dict(state)
        draft.set_lm_head_weight(head)
        model = MTPModel(draft, length=3, gradient_checkpointing=checkpoint)
        calls, handle = _capture_indexer_calls(draft.midlayer.self_attn)
        try:
            loss = sum(model(**batch)[0])
            forward_calls = len(calls)
            loss.backward()
        finally:
            handle.remove()
        assert forward_calls == (1 if share else 3)
        if not share:
            assert all(keys is calls[0][1] for _, keys, _ in calls[1:forward_calls])
        grad = draft.midlayer.self_attn.q_b_proj.weight.grad
        assert grad is not None and bool(grad.isfinite().all())
        losses[share] = loss.detach()
    assert not torch.equal(losses[True], losses[False])
