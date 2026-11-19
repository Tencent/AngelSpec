import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from angelspec.data.positions import (
    IMAGE_NEWLINE_SLOTS_KEY,
    POSITION_RECIPE_KEY,
    PositionRecipe,
    build_position_ids,
    position_recipe_from_config,
    prepare_sample_positions,
)
from angelspec.data.utils import (
    DataCollatorWithPadding,
    DFlashPackingCollator,
    MTPPackingCollator,
)
from angelspec.models.dflash import DFlashModel
from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from angelspec.models.draft.mtp import MTPConfig
from angelspec.training.data_fetcher import MooncakeDataset, TrainSample
from angelspec.training.position_config import configure_draft_positions
from tests.test_mtp_mrope import ROPE

R = PositionRecipe(
    image_start_token_id=100,
    image_end_token_id=101,
    image_newline_token_id=102,
    image_token_id=103,
    video_token_id=104,
    video_start_token_id=105,
    video_end_token_id=106,
)


def image(rows=2, cols=3):
    return [100] + ([103] * cols + [102]) * rows + [101]


def sample(tokens):
    n = len(tokens)
    return {
        "input_ids": torch.tensor([tokens], dtype=torch.long),
        "loss_mask": torch.ones(1, n),
        "hidden_states": torch.randn(1, n, 64),
        "last_hidden_states": torch.randn(1, n, 32),
    }


def draft_config(enabled=True):
    rope = {"rope_theta": 1e7, "rope_type": "default"}
    if enabled:
        rope.update(mrope_section=[12, 10, 10], mrope_interleaved=True)
    return DFlashConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        num_target_layers=2,
        target_hidden_size=32,
        target_num_hidden_layers=12,
        mask_token_id=127,
        max_position_embeddings=128,
        attention_type="mla",
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=64,
        v_head_dim=8,
        rope_parameters=rope,
        image_start_token_id=100,
        image_end_token_id=101,
        image_newline_token_id=102,
        image_token_id=103,
        video_token_id=104,
        video_start_token_id=105,
        video_end_token_id=106,
    )


def test_hand_worked_image_and_video_and_document_boundaries():
    tokens = [5, 5] + image() + [6, 6]
    p = build_position_ids(torch.tensor([tokens]), R)[0]
    assert p[0].tolist() == [0, 1, 2] + [3] * 8 + [7, 8, 9]
    assert p[1].tolist() == [0, 1, 2] + [3] * 4 + [4] * 4 + [7, 8, 9]
    assert p[2].tolist() == [0, 1, 2] + [3, 4, 5, 6] * 2 + [7, 8, 9]
    video = [5, 105, 7] + image() + [8, 8] + image() + [106, 6]
    actual = build_position_ids(torch.tensor([video]), R)
    assert actual.shape == (1, 3, len(video))
    ids = torch.tensor([tokens + video + [0, 0]])
    docs = torch.tensor([[0] * len(tokens) + [1] * len(video) + [-1, -1]])
    packed = build_position_ids(ids, R, doc_ids=docs)
    assert torch.equal(packed[0, :, : len(tokens)], p)
    assert torch.equal(packed[:, :, len(tokens) : -2], actual)
    assert packed[..., -2:].count_nonzero() == 0


def test_rewritten_image_rows_reconstruct_from_newline_slots():
    tokens = [5] + image() + [6] + image(rows=3, cols=2) + [7]
    slots = [i for i, t in enumerate(tokens) if t == 102]
    rewritten = torch.tensor([[103 if t == 102 else t for t in tokens]])
    expected = build_position_ids(torch.tensor([tokens]), R)
    meta = {"image_newline_slots": slots}
    assert torch.equal(build_position_ids(rewritten, R, multimodal_metadata=meta), expected)
    assert rewritten.eq(102).sum() == 0
    with pytest.raises(ValueError, match="image_newline_slots"):
        build_position_ids(rewritten, R)
    for bad in ([0], [len(tokens)], slots[:-1] + [slots[-1] + 1]):
        with pytest.raises(ValueError):
            build_position_ids(rewritten, R, multimodal_metadata={"image_newline_slots": bad})


@pytest.mark.parametrize(
    "tokens",
    [
        [100, 103, 102],
        [101],
        [100, 100, 103, 102, 101, 101],
        [100, 103, 103, 101],
        [100, 103, 102, 103, 103, 102, 101],
        [100, 103, 102, 103, 102, 103, 101],
        [100, 5, 102, 101],
        [103, 5],
        [105, 5, 106],
        [105] + image(),
        [100, 103, 105, 102, 101, 106],
    ],
)
def test_invalid_layouts_fail_closed(tokens):
    with pytest.raises(ValueError):
        build_position_ids(torch.tensor([tokens]), R)


def test_hy4_reconstruction_is_once_before_packing():
    data = sample([5] + image() + [6] * 6)
    with mock.patch(
        "angelspec.data.positions._hy4_document",
        wraps=__import__("angelspec.data.positions", fromlist=["_hy4_document"])._hy4_document,
    ) as build:
        prepare_sample_positions(data, R)
        first = data["position_ids"]
        prepare_sample_positions(data, R)
        assert build.call_count == 1
        assert data["position_ids"] is first
    for collator in (MTPPackingCollator(40), DFlashPackingCollator(40)):
        packed = collator([data, copy.deepcopy(data)])
        n = data["input_ids"].shape[-1]
        assert torch.equal(packed["position_ids"][..., :n], first)
        assert torch.equal(packed["position_ids"][..., n : 2 * n], first)
        assert packed["position_ids"][..., 2 * n :].count_nonzero() == 0
        assert packed["base_position_ids"][0, n] == 0


def test_provided_positions_extension_and_unknown_provider():
    r = replace(R, provider="provided")
    data = sample([5] * 8)
    with pytest.raises(ValueError, match="prepared position_ids"):
        prepare_sample_positions(data, r)
    data["position_ids"] = torch.arange(24).reshape(1, 3, 8)
    prepare_sample_positions(data, r)
    assert data[POSITION_RECIPE_KEY] == r.identity
    with pytest.raises(ValueError, match="Unsupported"):
        PositionRecipe(provider="qwen_unimplemented")
    with pytest.raises(ValueError, match="version"):
        replace(R, version=2)


def test_mooncake_transmits_only_original_tensors_and_prepares_final_tokens():
    from angelspec.models.target.eagle3_target_model import Eagle3TargetOutput

    data = sample([5] + image() + [6] * 6)
    ids = data["input_ids"][0]
    store = SimpleNamespace(
        get=mock.Mock(
            return_value=Eagle3TargetOutput(
                input_ids=ids, last_hidden_states=data["last_hidden_states"][0]
            )
        ),
        remove_tensors=mock.Mock(),
    )
    entry = TrainSample(
        "k",
        {
            "input_ids": tuple(ids.shape),
            "last_hidden_states": tuple(data["last_hidden_states"][0].shape),
        },
        packed_loss_mask="0," + str(ids.numel()),
    )
    dataset = MooncakeDataset(
        None, store, torch.device("cpu"), position_recipe=R, usp_local_shard=True
    )
    prepared, _ = dataset._prepare_one(entry, 0)
    assert prepared["position_ids"].shape == (1, 3, ids.numel())
    assert "position_ids" not in store.get.call_args.kwargs["shapes"]
    assert prepared["_mooncake_delete_key"] == ("k", True, False, False)
    placeholder = dataset._make_placeholder(entry)
    assert placeholder["position_ids"].shape == (1, 3, ids.numel())
    assert placeholder["loss_mask"].count_nonzero() == 0


def test_config_mtp_auto_dflash_opt_in_and_target_validation():
    target = {
        "model_type": "hy_v4_flash",
        "rope_parameters": ROPE,
        "qk_rope_head_dim": 64,
        "vision_config": {},
    }
    cfg = MTPConfig(attention_type="mla", qk_rope_head_dim=64)
    configure_draft_positions(cfg, target)
    assert position_recipe_from_config(cfg) == PositionRecipe()
    assert cfg.rope_parameters == ROPE
    plain = draft_config(False)
    configure_draft_positions(plain, target)
    assert position_recipe_from_config(plain) is None
    changed = {**target, "vision_config": {"cat_extra_token": 1}}
    with pytest.raises(ValueError, match="cat_extra_token"):
        configure_draft_positions(cfg, changed)
    other = MTPConfig(attention_type="mla", qk_rope_head_dim=64)
    with pytest.raises(ValueError, match="Unknown"):
        configure_draft_positions(other, {**target, "model_type": "qwen2_vl"})


def test_config_mtp_hy_v4_flash_uses_rope_of_the_mtp_attention_type():
    # Hy4 flash nests RoPE per layer type; the MTP block is a sparse-attention layer.
    target = {
        "model_type": "hy_v4_flash",
        "vision_config": {"use_image_newline_token": None},
        "text_config": {
            "qk_rope_head_dim": 64,
            "layer_types": [
                "deepseek_sparse_attention",
                "sliding_attention",
                "deepseek_sparse_attention",
            ],
            "rope_parameters": {
                "deepseek_sparse_attention": {**ROPE, "rope_theta": 10000000},
                "sliding_attention": {**ROPE, "rope_theta": 10000},
            },
        },
    }
    cfg = MTPConfig(attention_type="mla", qk_rope_head_dim=64)
    configure_draft_positions(cfg, target)
    assert cfg.rope_theta == 10000000
    assert cfg.rope_parameters["mrope_section"] == ROPE["mrope_section"]
    assert position_recipe_from_config(cfg) == PositionRecipe()


def test_dflash_receives_prepared_positions_never_reconstructs_in_forward():
    cfg = draft_config()
    draft = DFlashDraftModel(cfg)
    model = DFlashModel(draft, block_size=4, num_anchors=2)
    tokens = [5] * 3 + image() + [6] * 16
    prepared = prepare_sample_positions(sample(tokens), R)
    hidden = [prepared["hidden_states"][..., :32], prepared["hidden_states"][..., 32:]]
    kwargs = dict(
        input_ids=prepared["input_ids"],
        hidden_states_list=hidden,
        loss_mask=prepared["loss_mask"],
        lm_head_weight=torch.randn(128, 32),
        injected_anchors=torch.tensor([[15, 21]]),
        injected_keep_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="prepared position_ids"):
        model(**kwargs)
    seen = {}
    hook = draft.register_forward_pre_hook(lambda m, a, k: seen.update(k), with_kwargs=True)
    with mock.patch(
        "angelspec.data.positions._hy4_document", side_effect=AssertionError("recomputed")
    ):
        out = model(**kwargs, position_ids=prepared["position_ids"])
        out[0].backward()
    hook.remove()
    assert torch.equal(seen["context_position_ids"], prepared["position_ids"])
    assert seen["draft_position_ids"].shape == (1, 3, 8)
    assert draft.layers[0].self_attn.q_b_proj.weight.grad is not None


def test_shared_rotary_kernel_and_config_roundtrip():
    from angelspec.models.draft.hy4_mla import Hy4DraftMLAAttention
    from angelspec.models.draft.mtp import MTPMLAAttention

    a = Hy4DraftMLAAttention(draft_config())
    cfg = MTPConfig(
        attention_type="mla",
        q_lora_rank=16,
        hidden_size=32,
        num_attention_heads=4,
        qk_rope_head_dim=64,
        rope_parameters=ROPE,
    )
    b = MTPMLAAttention(cfg)
    p = build_position_ids(torch.tensor([[5] + image() + [6] * 4]), R)
    ca, sa = a._cos_sin(torch.zeros(1), p, p.shape[-1])
    cb, sb = b._cos_sin(torch.zeros(1), p, p.shape[-1])
    assert torch.equal(ca, cb) and torch.equal(sa, sb)
    restored = DFlashConfig(**draft_config().to_dict())
    assert position_recipe_from_config(restored).identity == R.identity


def test_eval_cache_invalidates_old_recipe_without_reusing_positions(tmp_path):
    cfg = draft_config()
    from angelspec.training.mtp_trainer import MTPTrainer

    trainer = object.__new__(MTPTrainer)
    trainer.args = SimpleNamespace(draft_model_config_obj=cfg)
    trainer.cache_rank = 0
    trainer.dp_rank = 0
    trainer._eval_cache_save_future = None
    path = tmp_path / "eval_rank_0.pt"
    data = prepare_sample_positions(sample([5] + image() + [6]), R)
    torch.save([data], path)
    assert trainer.load_eval_cache(str(tmp_path)) == 0
    torch.save({"position_recipe": "old", "samples": [data]}, path)
    assert trainer.load_eval_cache(str(tmp_path)) == 0
    torch.save({"position_recipe": R.identity, "samples": [data]}, path)
    assert trainer.load_eval_cache(str(tmp_path)) == 1
    with ThreadPoolExecutor(1) as pool:
        trainer._io_executor = pool
        trainer.save_eval_cache(str(tmp_path))
        trainer._wait_for_eval_cache_save()
    assert torch.load(path, weights_only=False)["position_recipe"] == R.identity


def test_padding_and_empty_samples():
    data = prepare_sample_positions(sample([]), R)
    assert data["position_ids"].shape == (1, 3, 0)
    a = prepare_sample_positions(sample([5] * 4), R)
    b = prepare_sample_positions(sample([6] * 7), R)
    batch = DataCollatorWithPadding()([a, b])
    assert batch["position_ids"].shape == (2, 3, 256)
    assert batch["position_ids"][0, :, 4:].count_nonzero() == 0


def test_dflash_text_outputs_match_plain_rope():
    torch.manual_seed(31)
    plain = DFlashDraftModel(draft_config(False))
    mr = DFlashDraftModel(draft_config(True))
    mr.load_state_dict(plain.state_dict())
    data = prepare_sample_positions(sample([5] * 20), R)
    kw = dict(
        input_ids=data["input_ids"],
        hidden_states_list=list(data["hidden_states"].split(32, -1)),
        loss_mask=data["loss_mask"],
        lm_head_weight=torch.randn(128, 32),
        injected_anchors=torch.tensor([[5, 11]]),
        injected_keep_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    a = DFlashModel(plain, block_size=4, num_anchors=2)(**kw)[0]
    b = DFlashModel(mr, block_size=4, num_anchors=2)(**kw, position_ids=data["position_ids"])[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_offline_rows_use_the_same_preparation():
    from angelspec.data.offline import OfflineBatchFetcher

    records = [{"seq_len": 18, "split": "train", "data_id": str(i)} for i in range(2)]
    data = sample([5] * 3 + image() + [6] * 5)
    data["attention_mask"] = torch.ones_like(data["input_ids"])

    def load(record, layer_ids=None):
        stored = copy.deepcopy(data)
        hidden = stored.pop("hidden_states")
        stored["layers"] = {
            1: hidden[..., :32],
            2: hidden[..., 32:],
            3: stored.pop("last_hidden_states"),
        }
        return stored

    dataset = SimpleNamespace(records=records, load=load)
    fetcher = OfflineBatchFetcher(
        dataset,
        rank=0,
        world_size=1,
        seed=1,
        max_length=40,
        hidden_size=32,
        aux_layer_ids=[1, 2],
        final_layer_id=3,
        vocab_size=128,
        min_loss_tokens=1,
        packing=True,
        model_family="dflash",
        position_recipe=R,
    )
    batch = fetcher.load_row(records)
    expected = build_position_ids(data["input_ids"], R)
    assert torch.equal(batch["position_ids"][..., :18], expected)
    assert torch.equal(batch["position_ids"][..., 18:36], expected)


def test_dflash_serving_refuses_unaligned_backend(tmp_path):
    from angelspec.inference.engine.eval_engine import EvalEngine

    (tmp_path / "config.json").write_text(json.dumps(draft_config().to_dict()))
    with pytest.raises(NotImplementedError, match="aligned backend"):
        EvalEngine._build_spec_config(SimpleNamespace(), "dflash", 4, str(tmp_path))


def test_mtp_reconstruction_does_not_require_vllm():
    from angelspec.train_entry import _validate_and_configure_mtp
    from tests.test_mtp_mrope import config

    for backend in ["hf", "sgl", "vllm"]:
        args = SimpleNamespace(inference_engine_type=backend, ttt_length=5)
        _validate_and_configure_mtp(args, config())
        assert not getattr(args, "mtp_capture_mrope", False)


def test_no_runtime_capture_hook_and_native_transport_unchanged():
    import inspect

    from angelspec.inference.engine import mrope_capture
    from angelspec.models.target.eagle3_target_model import Eagle3TargetOutput
    from angelspec.transfer.mooncake.draft_store import DraftMooncakeStore

    assert not hasattr(mrope_capture, "install_mrope_capture")
    assert "position_ids" not in inspect.signature(DraftMooncakeStore.put).parameters
    assert (
        "position_ids" not in inspect.signature(DraftMooncakeStore.put_kv_cache_chunks).parameters
    )
    assert "position_ids" not in Eagle3TargetOutput.__dataclass_fields__
    assert "_pos" not in DraftMooncakeStore.TENSOR_SUFFIXES


def test_diagnostic_compares_reconstruction_to_actual_positions():
    from angelspec.inference.engine.mrope_capture import compare_target_positions

    ids = torch.tensor([[5] + image() + [6]])
    target = build_position_ids(ids, R)
    assert compare_target_positions(ids, target, R)["max_error"] == 0
    with pytest.raises(ValueError, match="differ from target"):
        compare_target_positions(ids, target + 1, R)


def test_unknown_and_conflicting_layout_config_fail_before_training():
    cfg = MTPConfig(attention_type="mla", qk_rope_head_dim=64)
    target = {
        "model_type": "hy_v4_flash",
        "rope_parameters": ROPE,
        "vision_config": {"use_image_newline_token": False},
    }
    with pytest.raises(ValueError, match="newline"):
        configure_draft_positions(cfg, target)
    with pytest.raises(ValueError, match="conflicts"):
        DFlashConfig(
            attention_type="mla",
            qk_rope_head_dim=64,
            rope_parameters=ROPE,
            rope_theta=10000,
            position_provider="provided",
        )


def test_real_vllm_position_function_images_video_and_packing():
    import ast
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("vllm")
    if spec is None:
        pytest.skip("vLLM reference unavailable")
    path = Path(next(iter(spec.submodule_search_locations))) / "models/hy_v4/nvidia/vl_model.py"
    if not path.is_file():
        pytest.skip("Hy4 flash reference unavailable")
    from vllm.transformers_utils.processors.hy_v4 import vision_token_grid

    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "HYV4VLForConditionalGeneration"
    )
    func = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_mrope_input_positions"
    )
    ns = {
        "torch": torch,
        "vision_token_grid": vision_token_grid,
        "MultiModalFeatureSpec": type(
            "Features", (), {"gather_kwargs": staticmethod(lambda features, keys: features)}
        ),
    }
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), "exec"), ns)
    target = SimpleNamespace(
        image_token_id=103,
        video_token_id=104,
        image_start_token_id=100,
        image_end_token_id=101,
        video_start_token_id=105,
        video_end_token_id=106,
        vision_config=SimpleNamespace(
            spatial_merge_size=2,
            cat_extra_token=0,
            temporal_patch_size=1,
            spatial_patch_size=1,
            use_image_newline_token=True,
        ),
    )

    def served(tokens, pad):
        # vLLM's prompt: the vision tower fills each row's newline slot, so it holds the pad.
        return [pad if t in (102, 103) else t for t in tokens]

    text = [5] * 8
    native = [5] * 3 + image() + [5, 105, 5] + image(3, 2) + [6] * 3 + image(3, 2) + [106, 6, 6]
    video = native.index(105)
    visual = served(native[:video], 103) + served(native[video:], 104)
    features = {
        "image_grid_thw": [torch.tensor([1, 4, 6])],
        "video_grid_thw": [torch.tensor([2, 6, 4])],
    }
    expected, _ = ns["get_mrope_input_positions"](SimpleNamespace(config=target), visual, features)
    actual = build_position_ids(torch.tensor([native]), R)[0]
    assert torch.equal(actual, expected)
    slots = [i for i, t in enumerate(native) if t == 102]
    from_slots = build_position_ids(
        torch.tensor([visual]), R, multimodal_metadata={IMAGE_NEWLINE_SLOTS_KEY: slots}
    )[0]
    assert torch.equal(from_slots, expected)
    packed = build_position_ids(
        torch.tensor([text + native]),
        R,
        doc_ids=torch.tensor([[0] * len(text) + [1] * len(native)]),
    )
    assert torch.equal(packed[0, :, : len(text)], torch.arange(len(text)).expand(3, -1))
    assert torch.equal(packed[0, :, len(text) :], expected)


def test_dflash_mrope_rejects_visual_anchor():
    data = prepare_sample_positions(sample([5] * 3 + image() + [6] * 16), R)
    model = DFlashModel(DFlashDraftModel(draft_config()), block_size=4, num_anchors=1)
    with pytest.raises(ValueError, match="text positions"):
        model(
            input_ids=data["input_ids"],
            hidden_states_list=list(data["hidden_states"].split(32, -1)),
            loss_mask=data["loss_mask"],
            lm_head_weight=torch.randn(128, 32),
            position_ids=data["position_ids"],
            injected_anchors=torch.tensor([[6]]),
            injected_keep_mask=torch.ones(1, 1, dtype=torch.bool),
        )
