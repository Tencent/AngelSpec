import copy
from unittest import mock

import pytest
import torch

from angelspec.data.positions import prepare_sample_positions
from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from angelspec.models.draft.hy4_mla import Hy4DraftMLAAttention
from angelspec.models.draft.mtp import _apply_rope_split, _interleaved_to_split_rope
from angelspec.models.ops.mrope import mrope_text_block_mask
from angelspec.training.dflash_trainer import DFlashTrainer
from tests.test_position_preparation import R, draft_config, image, sample


def prepared_model(block_size=4, num_anchors=1):
    data = prepare_sample_positions(sample([5] * 3 + image() + [6] * 16), R)
    model = DFlashModel(
        DFlashDraftModel(draft_config()), block_size=block_size, num_anchors=num_anchors
    )
    return data, model


def forward_kwargs(data):
    return dict(
        input_ids=data["input_ids"],
        hidden_states_list=list(data["hidden_states"].split(32, -1)),
        loss_mask=data["loss_mask"],
        lm_head_weight=torch.randn(128, 32),
        position_ids=data["position_ids"],
    )


@pytest.mark.parametrize("anchor", [0, 1, 2, 3, 4, 6, 11, 12, 26, -1, 29])
def test_visual_and_crossing_blocks_are_rejected(anchor):
    data, model = prepared_model()
    with pytest.raises(ValueError, match="consecutive text positions"):
        model(
            **forward_kwargs(data),
            injected_anchors=torch.tensor([[anchor]]),
            injected_keep_mask=torch.ones(1, 1, dtype=torch.bool),
        )


def test_visual_top_left_is_not_a_text_anchor_even_for_one_token_blocks():
    data, model = prepared_model(block_size=1)
    assert data["position_ids"][0, :, 4].tolist() == [4, 4, 4]
    assert not model._mrope_anchor_mask(data["input_ids"], data["position_ids"])[0, 4]


@pytest.mark.parametrize("anchor", [13, 15, 25])
def test_complete_text_blocks_after_image_are_allowed(anchor):
    data, model = prepared_model()
    out = model(
        **forward_kwargs(data),
        injected_anchors=torch.tensor([[anchor]]),
        injected_keep_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    out[0].backward()
    assert torch.isfinite(out[0])
    assert torch.isfinite(model.draft_model.layers[0].self_attn.q_b_proj.weight.grad).all()


def test_automatic_sampling_filters_blocks_without_changing_loss_mask():
    data, model = prepared_model(num_anchors=32)
    loss_mask = data["loss_mask"].clone()
    seen = {}
    original = model._sample_anchor_positions

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        seen["anchors"], seen["keep"] = result
        return result

    with mock.patch.object(model, "_sample_anchor_positions", side_effect=record):
        out = model(**forward_kwargs(data))
        out[0].backward()
    active = seen["anchors"][seen["keep"]]
    assert active.tolist() == list(range(13, 26))
    assert torch.equal(data["loss_mask"], loss_mask)


def test_inactive_visual_anchor_does_not_fail():
    data, model = prepared_model()
    out = model(
        **forward_kwargs(data),
        injected_anchors=torch.tensor([[4]]),
        injected_keep_mask=torch.zeros(1, 1, dtype=torch.bool),
    )
    out[0].backward()
    assert float(out[0].detach()) == 0.0


def test_usp_teacher_sampling_uses_the_same_eligibility():
    data, model = prepared_model(num_anchors=32)
    trainer = object.__new__(DFlashTrainer)
    trainer.dflash = model
    trainer.sp_group = None
    with (
        mock.patch.object(trainer, "_sp_rank", return_value=0),
        mock.patch.object(trainer, "_sp_size", return_value=2),
        mock.patch("angelspec.training.dflash_trainer.dist.get_global_rank", return_value=0),
        mock.patch("angelspec.training.dflash_trainer.dist.broadcast"),
    ):
        anchors, keep, local = trainer._sample_usp_anchors(
            data["input_ids"], data["loss_mask"], None, 32, data["position_ids"]
        )
    assert anchors[keep].tolist() == list(range(13, 26))
    assert torch.equal(local, anchors[:, ::2])


@pytest.mark.parametrize("block_size", [1, 2, 4, 8, 32])
def test_text_block_mask_matches_bruteforce(block_size):
    data, model = prepared_model(block_size=block_size)
    p = data["position_ids"]
    ids = data["input_ids"]
    markers = tuple(v for k, v in vars(R).items() if k.endswith("_token_id"))
    expected = torch.zeros_like(ids, dtype=torch.bool)
    for start in range(ids.shape[-1] - block_size + 1):
        coord = p[..., start : start + block_size]
        wanted = coord[:, :1, :1] + torch.arange(block_size)
        text = not any(int(token) in markers for token in ids[0, start : start + block_size])
        expected[:, start] = text and torch.equal(coord, wanted.expand(-1, 3, -1))
    assert torch.equal(model._mrope_anchor_mask(ids, p), expected)


def test_text_block_mask_rejects_resets_jumps_and_short_inputs():
    positions = torch.tensor([[[0, 1, 2, 10, 11, 0, 1, 2]]]).expand(-1, 3, -1)
    assert mrope_text_block_mask(positions, 3).tolist() == [
        [True, False, False, False, False, True, False, False]
    ]
    assert not mrope_text_block_mask(positions, 9).any()
    with pytest.raises(ValueError, match="block_size"):
        mrope_text_block_mask(positions, 0)


def test_video_markers_and_frame_tokens_cannot_be_drafted():
    data = prepare_sample_positions(sample([5, 105, 7] + image() + [106] + [6] * 8), R)
    model = DFlashModel(DFlashDraftModel(draft_config()), block_size=4, num_anchors=32)
    eligible = model._mrope_anchor_mask(data["input_ids"], data["position_ids"])
    assert eligible.nonzero(as_tuple=True)[1].tolist() == list(range(14, 19))


def test_dflash_three_axes_require_explicit_opt_in():
    plain = Hy4DraftMLAAttention(draft_config(False))
    with pytest.raises(ValueError, match="mrope_section"):
        plain._cos_sin(torch.zeros(1), torch.zeros(1, 3, 4, dtype=torch.long), 4)
    kwargs = draft_config().to_dict()
    kwargs["attention_type"] = "gqa"
    with pytest.raises(ValueError, match="MLA"):
        DFlashConfig(**kwargs)


@pytest.mark.parametrize("interleaved", [False, True])
def test_dflash_rotated_qk_match_independent_interleaved_pair_reference(interleaved):
    torch.manual_seed(23)
    cfg = draft_config()
    cfg.rope_parameters = copy.deepcopy(cfg.rope_parameters)
    cfg.rope_parameters["mrope_interleaved"] = interleaved
    attn = Hy4DraftMLAAttention(cfg)
    p = torch.randint(0, 4000, (2, 3, 37))
    q = torch.randn(2, 4, 37, 64)
    k = torch.randn(2, 1, 37, 64)
    inv = 1 / (cfg.rope_theta ** (torch.arange(0, 64, 2).float() / 64))
    axes = [0, 1, 2] * 10 + [0, 0] if interleaved else [0] * 12 + [1] * 10 + [2] * 10
    phase = torch.stack([p[:, axis] * inv[i] for i, axis in enumerate(axes)], -1)
    cos, sin = attn._cos_sin(q, p, p.shape[-1])
    for tensor in (q, k):
        actual = _apply_rope_split(_interleaved_to_split_rope(tensor), cos, sin)
        first, second = tensor[..., ::2], tensor[..., 1::2]
        expected = torch.stack(
            (
                first * phase.cos()[:, None] - second * phase.sin()[:, None],
                second * phase.cos()[:, None] + first * phase.sin()[:, None],
            ),
            -1,
        ).flatten(-2)
        torch.testing.assert_close(actual, _interleaved_to_split_rope(expected), atol=1e-6, rtol=0)


def test_automatic_sampling_without_text_blocks_returns_zero_loss():
    data = prepare_sample_positions(sample(image()), R)
    model = DFlashModel(DFlashDraftModel(draft_config()), block_size=4, num_anchors=4)
    out = model(**forward_kwargs(data))
    out[0].backward()
    assert float(out[0].detach()) == 0.0


def test_provided_positions_filter_coordinate_discontinuities():
    cfg = draft_config()
    cfg.position_provider = "provided"
    model = DFlashModel(DFlashDraftModel(cfg), block_size=3, num_anchors=2)
    p = torch.tensor([[[0, 1, 2, 10, 11, 0, 1, 2]]]).expand(-1, 3, -1)
    ids = torch.full((1, 8), 5)
    assert model._mrope_anchor_mask(ids, p).tolist() == [
        [True, False, False, False, False, True, False, False]
    ]
