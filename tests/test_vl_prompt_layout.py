"""Pre-tokenized VL prompt layouts: Hy4 flash newline slots and token-prompt expansion."""

from types import SimpleNamespace

import pytest
import torch

from angelspec.data import dataset
from angelspec.data.positions import (
    IMAGE_NEWLINE_SLOTS_KEY,
    PositionRecipe,
    build_position_ids,
)
from angelspec.data.utils import (
    HY4_IMAGE_END,
    HY4_IMAGE_NEWLINE,
    HY4_IMAGE_PAD,
    HY4_IMAGE_START,
    HY4_VIDEO_END,
    HY4_VIDEO_PAD,
    HY4_VIDEO_START,
)
from angelspec.inference.engine import vllm_runtime_patch

RECIPE = PositionRecipe()
TOKENIZER = SimpleNamespace(
    convert_tokens_to_ids={
        HY4_IMAGE_START: RECIPE.image_start_token_id,
        HY4_IMAGE_END: RECIPE.image_end_token_id,
        HY4_IMAGE_PAD: RECIPE.image_token_id,
        HY4_IMAGE_NEWLINE: RECIPE.image_newline_token_id,
        HY4_VIDEO_START: RECIPE.video_start_token_id,
        HY4_VIDEO_END: RECIPE.video_end_token_id,
        HY4_VIDEO_PAD: RECIPE.video_token_id,
    }.__getitem__
)


def _hy4_conversation_ids():
    s, e = RECIPE.image_start_token_id, RECIPE.image_end_token_id
    p, n = RECIPE.image_token_id, RECIPE.image_newline_token_id
    # Text, a two-row image of three pixels per row, text, a one-row image, text.
    return [1, 2, s, p, p, p, n, p, p, p, n, e, 3, s, p, p, n, e, 4]


def test_hy4_conversation_newlines_become_placeholders_with_slots():
    original = _hy4_conversation_ids()
    ids, slots = dataset._hy4_image_newlines_to_placeholder(original, TOKENIZER)
    assert RECIPE.image_newline_token_id not in ids.tolist()
    assert slots == [i for i, t in enumerate(original) if t == RECIPE.image_newline_token_id]
    assert all(ids[i] == RECIPE.image_token_id for i in slots)


def test_batched_hy4_conversation_uses_token_offsets_and_preserves_shape():
    original = torch.tensor([_hy4_conversation_ids()])
    ids, slots = dataset._hy4_image_newlines_to_placeholder(original, TOKENIZER)
    rewritten = torch.as_tensor(ids)
    expected_slots = [
        i
        for i, token in enumerate(original.reshape(-1).tolist())
        if token == RECIPE.image_newline_token_id
    ]

    assert rewritten.shape == original.shape
    assert slots == expected_slots
    assert torch.all(rewritten.reshape(-1)[slots] == RECIPE.image_token_id)
    assert torch.equal(
        build_position_ids(
            rewritten,
            RECIPE,
            multimodal_metadata={IMAGE_NEWLINE_SLOTS_KEY: slots},
        ),
        build_position_ids(original, RECIPE),
    )


def test_rewritten_layout_reconstructs_the_same_positions():
    original = _hy4_conversation_ids()
    ids, slots = dataset._hy4_image_newlines_to_placeholder(original, TOKENIZER)
    expected = build_position_ids(torch.tensor([original]), RECIPE)
    rebuilt = build_position_ids(
        torch.tensor([ids.tolist()]),
        RECIPE,
        multimodal_metadata={IMAGE_NEWLINE_SLOTS_KEY: slots},
    )
    assert torch.equal(rebuilt, expected)


def test_entry_metadata_carries_newline_slots():
    result = {
        "input_ids": [1, 2],
        "packed_loss_mask": "0,2",
        "formatted_prompt": "x",
        "multimodal_inputs": None,
        "image_newline_slots": [5, 9],
    }
    entry = dataset._build_entry("d0", result, defer_tokenization=False)
    assert entry["metadata"][IMAGE_NEWLINE_SLOTS_KEY] == [5, 9]
    result["image_newline_slots"] = []
    entry = dataset._build_entry("d1", result, defer_tokenization=False)
    assert IMAGE_NEWLINE_SLOTS_KEY not in entry["metadata"]


def test_token_prompts_are_treated_as_expanded_only_when_flagged(monkeypatch):
    class Processor:
        def _apply_hf_processor_main(
            self, prompt, mm_items, hf_kwargs, tok_kwargs, *, enable_hf_prompt_update
        ):
            return [1, 2, 3], {}, False

    vllm_runtime_patch._treat_token_prompts_as_expanded(Processor)

    def applied(prompt):
        return Processor()._apply_hf_processor_main(
            prompt, None, {}, {}, enable_hf_prompt_update=False
        )[2]

    monkeypatch.delenv(vllm_runtime_patch.PRETOKENIZED_MM_ENV, raising=False)
    assert applied([1, 2, 3]) is False
    monkeypatch.setenv(vllm_runtime_patch.PRETOKENIZED_MM_ENV, "1")
    assert applied([1, 2, 3]) is True
    assert applied("a text prompt") is False


def test_installed_vllm_processors_are_patched():
    pytest.importorskip("vllm")
    from vllm.multimodal.processing.processor import BaseMultiModalProcessor

    assert vllm_runtime_patch.apply_pretokenized_mm_prompts()
    assert BaseMultiModalProcessor.__dict__.get("_angelspec_pretokenized_mm")
