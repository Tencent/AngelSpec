import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import load_file, save_file

from angelspec.config.train_config import config_to_flat_args, load_config
from angelspec.controller import loop
from angelspec.controller.training_controller import AsyncTrainingController
from angelspec.data import offline
from angelspec.data.media_identity import MEDIA_DIGEST_VERSION, MULTIMODAL_SHA256_KEY
from angelspec.data.offline import (
    HiddenDataset,
    HiddenWriter,
    OfflineBatchFetcher,
    atomic_json,
    layer_recipe,
    tensor_digest,
    write_safetensors,
)
from angelspec.data.offline_assets import export_target_assets, read_target_components
from angelspec.data.positions import (
    IMAGE_NEWLINE_SLOTS_KEY,
    PositionRecipe,
    build_position_ids,
)
from angelspec.inference.online_dump import (
    OnlineDumpWriter,
    build_online_capture_recipe,
    dump_route,
    finalize_online_dump,
    materialize_native_capture,
    online_dump_status,
    validate_online_dump_args,
)
from angelspec.transfer.mooncake.store import MooncakeHiddenStateStore
from angelspec.utils.types import InferenceOutput


def test_online_dump_writer_defaults_and_limits(tmp_path):
    base = [
        "online_dump.enabled=true",
        f"online_dump.dump_path={tmp_path / 'hidden'}",
        f"online_dump.frozen_target_path={tmp_path / 'frozen'}",
        "inference.inference_engine_type=vllm",
    ]
    args = config_to_flat_args(load_config(cli_args=base))
    assert (args.online_dump_writer_threads, args.online_dump_reader_threads) == (16, 8)
    assert args.online_dump_writers is None
    assert args.online_dump_max_inflight_batches is None
    assert args.online_dump_max_pending_bytes == 32 * 1024**3
    for override in (
        "online_dump.reader_threads=0",
        "online_dump.max_inflight_batches=0",
        "online_dump.writers=0",
    ):
        with pytest.raises(ValueError, match="positive"):
            load_config(cli_args=[*base, override])


def test_online_dump_config_is_prefixed_and_rejects_unsupported_modes(tmp_path):
    config = load_config(
        cli_args=[
            "online_dump.enabled=true",
            f"online_dump.dump_path={tmp_path / 'hidden'}",
            f"online_dump.frozen_target_path={tmp_path / 'frozen'}",
            "online_dump.writer_node_ip=10.0.0.2",
            "online_dump.writer_env_vars.TEST_WRITER_ENV=ok",
            "inference.inference_engine_type=vllm",
        ]
    )
    args = config_to_flat_args(config)
    assert args.online_dump_enabled is True
    assert args.online_dump_dump_only is False
    assert args.online_dump_capture_epochs == 1
    assert args.online_dump_writer_node_ip == "10.0.0.2"
    assert args.online_dump_writer_env_vars["TEST_WRITER_ENV"] == "ok"
    with pytest.raises(ValueError, match="defer_tokenization"):
        load_config(
            cli_args=[
                "online_dump.enabled=true",
                f"online_dump.dump_path={tmp_path / 'hidden'}",
                f"online_dump.frozen_target_path={tmp_path / 'frozen'}",
                "inference.inference_engine_type=vllm",
                "dataset.defer_tokenization=true",
            ]
        )
    dump_only = config_to_flat_args(
        load_config(
            cli_args=[
                "online_dump.enabled=true",
                "online_dump.dump_only=true",
                f"online_dump.dump_path={tmp_path / 'hidden'}",
                f"online_dump.frozen_target_path={tmp_path / 'frozen'}",
                "inference.inference_engine_type=vllm",
            ]
        )
    )
    assert dump_only.online_dump_dump_only is True
    with pytest.raises(ValueError, match="dump_only requires"):
        load_config(cli_args=["online_dump.dump_only=true"])


def test_native_smoke_configs_load_without_tokenized_companion(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {
        "OFFLINE_HIDDEN_PATH": str(tmp_path / "hidden"),
        "OFFLINE_FROZEN_TARGET_PATH": str(tmp_path / "frozen"),
        "OFFLINE_OUTPUT_DIR": str(tmp_path / "offline-run"),
        "ONLINE_DUMP_PATH": str(tmp_path / "online-hidden"),
        "ONLINE_DUMP_FROZEN_TARGET_PATH": str(tmp_path / "online-frozen"),
        "ONLINE_RUN_OUTPUT": str(tmp_path / "online-run"),
        "ONLINE_RUN_CACHE": str(tmp_path / "online-cache"),
    }
    with mock.patch.dict(os.environ, env):
        hy4 = load_config(str(root / "configs/offline_dfly_hy_v4_flash_smoke.yaml"))
        qwen = load_config(str(root / "configs/offline_dfly_qwen3_8b_smoke.yaml"))
    assert hy4.offline.format == "native"
    assert hy4.offline.tokenized_path is None
    assert qwen.offline.format == "native"


def test_online_dump_recipe_and_runtime_guardrails(tmp_path):
    from angelspec.models.draft.dflash import DFlashConfig

    draft = DFlashConfig(
        hidden_size=2,
        target_hidden_size=2,
        vocab_size=32,
        num_target_layers=2,
        target_layer_ids=[1, 3],
        target_num_hidden_layers=4,
    )
    args = SimpleNamespace(
        online_dump_enabled=True,
        online_dump_dump_path=str(tmp_path / "hidden"),
        online_dump_frozen_target_path=str(tmp_path / "frozen"),
        online_dump_validation_fraction=0.1,
        online_dump_capture_epochs=1,
        target_model_path=str(tmp_path / "target"),
        inference_engine_type="vllm",
        train_with_decode=False,
        defer_tokenization=False,
        vllm_enable_multimodal=False,
        sglang_enable_multimodal=False,
        attention_backend="sdpa",
        usp_local_shard=False,
        store_last_hidden_states=True,
        last_hidden_states_prenorm=True,
        max_seq_length=32,
        chat_template="qwen",
        last_turn_loss_only=False,
        min_loss_tokens=2,
        seed=5,
    )
    validate_online_dump_args(args, draft)
    recipe = build_online_capture_recipe(
        args,
        draft,
        {"hidden_size": 2, "vocab_size": 32, "num_hidden_layers": 4},
        {"config.json": "identity"},
    )
    assert recipe["capture_aux_layer_order"] == [1]
    assert recipe["layer_ids"] == [1, 3]
    assert recipe["layer_tensors"] == {"1": "hidden_layer_1", "3": "hidden_layer_3"}
    assert recipe["layer_semantics"] == "decoder_layer_output"
    assert recipe["final_layer_id"] == 3
    assert "capture_code" not in recipe
    assert "media_digest" not in recipe
    nested = build_online_capture_recipe(
        args,
        draft,
        {
            "text_config": {
                "hidden_size": 2,
                "vocab_size": 32,
                "num_hidden_layers": 4,
            },
            "vision_config": {"hidden_size": 3},
        },
        {"config.json": "nested"},
    )
    assert nested["hidden_size"] == 2
    args.last_hidden_states_prenorm = False
    with pytest.raises(ValueError, match="pre-final-norm"):
        validate_online_dump_args(args, draft)
    args.last_hidden_states_prenorm = True
    args.vllm_enable_multimodal = True
    with mock.patch(
        "angelspec.inference.online_dump.position_recipe_from_config",
        return_value=PositionRecipe(),
    ):
        validate_online_dump_args(args, draft)
        vl_recipe = build_online_capture_recipe(
            args,
            draft,
            {"hidden_size": 2, "vocab_size": 32, "num_hidden_layers": 4},
            {"config.json": "vl"},
        )
        assert vl_recipe["media_digest"] == MEDIA_DIGEST_VERSION
        args.inference_engine_type = "sgl"
        with pytest.raises(ValueError, match="vllm"):
            validate_online_dump_args(args, draft)


def test_vl_target_assets_use_language_subconfig(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    config = {
        "model_type": "composite_vl",
        "text_config": {"hidden_size": 2, "vocab_size": 4, "num_hidden_layers": 2},
        "vision_config": {"hidden_size": 3},
    }
    atomic_json(target / "config.json", config)
    save_file(
        {
            "model.embed_tokens.weight": torch.ones(4, 2),
            "lm_head.weight": torch.ones(4, 2),
            "model.norm.weight": torch.ones(2),
        },
        str(target / "model.safetensors"),
    )
    loaded, identity, _ = read_target_components(target)
    assert loaded == config
    frozen = tmp_path / "frozen"
    assert export_target_assets(target, frozen) == identity
    assert (frozen / "config.json").is_file()


def _recipe(*, mtp=False, multimodal=False):
    position = PositionRecipe() if multimodal else None
    aux = [] if mtp else [1]
    return {
        **layer_recipe([*aux, 3], target_num_hidden_layers=4, hidden_size=2),
        "capture_aux_layer_order": aux,
        "schema_version": 2 if multimodal else 1,
        "capture_backend": "angelspec-online-test",
        "online_capture": True,
        "model_family": "mtp" if mtp else "dflash",
        "target_metadata_sha256": {"config.json": "test"},
        "vocab_size": 120832 if multimodal else 32,
        "storage_dtype": "bfloat16",
        "capture_epochs": 1,
        "multimodal": multimodal,
        "position_recipe_fingerprint": position.identity if position else None,
        "position_recipe": (
            {
                "provider": position.provider,
                "version": position.version,
                "image_start_token_id": position.image_start_token_id,
                "image_end_token_id": position.image_end_token_id,
                "image_newline_token_id": position.image_newline_token_id,
                "image_token_id": position.image_token_id,
                "video_token_id": position.video_token_id,
                "video_start_token_id": position.video_start_token_id,
                "video_end_token_id": position.video_end_token_id,
            }
            if position
            else None
        ),
        **({"media_digest": MEDIA_DIGEST_VERSION} if multimodal else {}),
    }


class _Store:
    def __init__(self, values):
        self.values = values
        self.closed = False
        self.reads = []
        self.removed = []

    def get_into(self, key, tensors):
        self.reads.append((key, sorted(tensors)))
        for name, tensor in tensors.items():
            tensor.copy_(self.values[key][name].to(tensor.dtype).reshape(tensor.shape))

    def remove_tensors(self, key, **flags):
        self.removed.append(key)

    def close(self):
        self.closed = True


def _result(key, ids, *, include_hidden=True, metadata=None):
    shapes = {
        "input_ids": tuple(ids.shape),
        "last_hidden_states": (ids.numel(), 2),
    }
    dtypes = {
        "input_ids": torch.int64,
        "last_hidden_states": torch.bfloat16,
    }
    if include_hidden:
        shapes["hidden_states"] = (ids.numel(), 2)
        dtypes["hidden_states"] = torch.bfloat16
    return InferenceOutput(
        data_id=f"source#{key}",
        mooncake_key=key,
        tensor_shapes=shapes,
        tensor_dtypes=dtypes,
        packed_loss_mask=f"1,{ids.numel() - 1}",
        metadata=metadata or {},
    )


def _values(ids):
    length = ids.numel()
    return {
        "input_ids": ids.clone(),
        "hidden_states": torch.arange(length * 2, dtype=torch.bfloat16).reshape(length, 2),
        "last_hidden_states": torch.ones(length, 2, dtype=torch.bfloat16),
    }


def test_online_writer_roundtrip_deduplicates_and_preserves_provenance(tmp_path):
    ids = torch.tensor([1, 2, 3, 4])
    store = _Store({"k": _values(ids)})
    writer = OnlineDumpWriter(
        None,
        tmp_path,
        _recipe(),
        validation_fraction=1e-12,
        seed=7,
        min_loss_tokens=1,
        max_pending_bytes=1024,
        store=store,
    )
    result = _result("k", ids)
    placement = {"dispatch": 3, "dp_rank": 1, "row_id": 7, "row_position": 0}
    with pytest.raises(ValueError, match="align"):
        writer.stage_batch([result], placements=[])
    assert writer.stage_batch([result], placements=[placement])["staged"] == 1
    store.reads.clear()
    assert writer.stage_batch([result])["staged"] == 0
    assert store.reads == [("k", ["input_ids"])]
    assert store.removed == []
    assert writer.release([result]) == 1
    assert store.removed == ["k"]
    status = writer.finalize()
    assert status["published"] == 1
    assert status["verify"] == "header"
    assert status["duplicates"] == 1
    assert store.closed
    dataset = HiddenDataset(tmp_path)
    assert dataset.verify() == 1
    assert dataset.records[0]["provenance"]["data_id"] == "source#k"
    assert dataset.records[0]["provenance"]["placement"] == placement
    sessions = (tmp_path / "sessions.jsonl").read_text()
    assert sessions.count("capture_code") == 1
    assert '"mode": "online"' in sessions
    tensors = load_file(str(tmp_path / dataset.records[0]["file"]))
    assert set(tensors) == {"input_ids", "loss_mask", "hidden_layer_1", "hidden_layer_3"}
    torch.testing.assert_close(tensors["hidden_layer_1"], _values(ids)["hidden_states"])
    torch.testing.assert_close(tensors["hidden_layer_3"], _values(ids)["last_hidden_states"])
    batch = next(
        OfflineBatchFetcher(
            dataset,
            rank=0,
            world_size=1,
            seed=7,
            max_length=16,
            hidden_size=2,
            aux_layer_ids=[1],
            final_layer_id=3,
            vocab_size=32,
            min_loss_tokens=1,
        )
    )
    assert batch["input_ids"].shape[0] == 1
    assert batch["input_ids"][0, :4].tolist() == [1, 2, 3, 4]


def test_materialize_native_capture_owns_buffers_and_checks_tokens():
    ids = torch.tensor([1, 2, 3, 4])
    captured = SimpleNamespace(
        input_ids=ids.clone(),
        hidden_states=torch.ones(4, 2, dtype=torch.bfloat16),
        last_hidden_states=torch.ones(4, 2, dtype=torch.bfloat16),
    )
    store = _Store({"k": vars(captured)})
    result = {
        "mooncake_key": "k",
        "tensor_shapes": {"input_ids": [4], "hidden_states": [4, 2], "last_hidden_states": [4, 2]},
        "tensor_dtypes": {
            "input_ids": "int64",
            "hidden_states": "bfloat16",
            "last_hidden_states": "bfloat16",
        },
    }
    tokens = {"input_ids": ids, "loss_mask": torch.tensor([0, 1, 1, 1])}
    tensors = materialize_native_capture(result, tokens, store, _recipe())
    captured.hidden_states.zero_()
    assert set(tensors) == {"input_ids", "loss_mask", "hidden_layer_1", "hidden_layer_3"}
    assert bool(tensors["hidden_layer_1"].all())
    captured.input_ids[0] = 7
    with pytest.raises(ValueError, match="different token"):
        materialize_native_capture(result, tokens, store, _recipe())


def test_online_writer_mtp_ignores_unused_aux_hidden(tmp_path):
    ids = torch.tensor([4, 3, 2, 1])
    writer = OnlineDumpWriter(
        None,
        tmp_path,
        _recipe(mtp=True),
        validation_fraction=0.25,
        seed=3,
        min_loss_tokens=1,
        max_pending_bytes=1024,
        store=_Store({"mtp": _values(ids)}),
        mode="dump_only",
    )
    writer.stage_batch([_result("mtp", ids)])
    writer.finalize()
    dataset = HiddenDataset(tmp_path)
    tensors = load_file(str(tmp_path / dataset.records[0]["file"]))
    assert set(tensors) == {"input_ids", "loss_mask", "hidden_layer_3"}
    assert dataset.verify() == 1
    assert '"mode": "dump_only"' in (tmp_path / "sessions.jsonl").read_text()


def test_online_vl_metadata_reconstructs_identical_mrope(tmp_path):
    position = PositionRecipe()
    ids = torch.tensor(
        [
            10,
            position.image_start_token_id,
            position.image_token_id,
            position.image_token_id,
            position.image_end_token_id,
            11,
        ]
    )
    metadata = {IMAGE_NEWLINE_SLOTS_KEY: [3]}
    writer = OnlineDumpWriter(
        None,
        tmp_path,
        _recipe(multimodal=True),
        validation_fraction=1e-12,
        seed=9,
        min_loss_tokens=1,
        max_pending_bytes=1024,
        store=_Store({"vl": _values(ids)}),
    )
    writer.stage_batch(
        [_result("vl", ids, metadata={**metadata, MULTIMODAL_SHA256_KEY: "a" * 64})]
    )
    writer.finalize()
    dataset = HiddenDataset(tmp_path)
    assert dataset.records[0]["sample_metadata"] == metadata
    fetcher = OfflineBatchFetcher(
        dataset,
        rank=0,
        world_size=1,
        seed=9,
        max_length=32,
        hidden_size=2,
        aux_layer_ids=[1],
        final_layer_id=3,
        vocab_size=120832,
        min_loss_tokens=1,
        position_recipe=position,
    )
    batch = next(fetcher)
    expected = build_position_ids(ids.reshape(1, -1), position, multimodal_metadata=metadata)
    torch.testing.assert_close(batch["position_ids"][..., : ids.numel()], expected)


def test_online_vl_sample_id_covers_media_content(tmp_path):
    position = PositionRecipe()
    ids = torch.tensor(
        [
            10,
            position.image_start_token_id,
            position.image_token_id,
            position.image_token_id,
            position.image_end_token_id,
            11,
        ]
    )
    text_ids = torch.tensor([12, 13, 14, 15])
    slots = {IMAGE_NEWLINE_SLOTS_KEY: [3]}
    first, second = "a" * 64, "b" * 64
    store = _Store({key: _values(ids) for key in ("first", "second", "again")})
    store.values["text"] = _values(text_ids)
    writer = OnlineDumpWriter(
        None,
        tmp_path,
        _recipe(multimodal=True),
        validation_fraction=1e-12,
        seed=9,
        min_loss_tokens=1,
        max_pending_bytes=1024,
        store=store,
    )
    staged = writer.stage_batch(
        [
            _result("first", ids, metadata={**slots, MULTIMODAL_SHA256_KEY: first}),
            _result("second", ids, metadata={**slots, MULTIMODAL_SHA256_KEY: second}),
            _result("again", ids, metadata={**slots, MULTIMODAL_SHA256_KEY: first}),
            _result("text", text_ids, metadata={MULTIMODAL_SHA256_KEY: None}),
        ]
    )
    assert staged["staged"] == 3
    status = writer.finalize()
    assert status["published"] == 3
    assert status["duplicates"] == 1
    provenance = {
        record["provenance"]["data_id"]: record["provenance"]
        for record in HiddenDataset(tmp_path).records
    }
    assert set(provenance) == {"source#first", "source#second", "source#text"}
    assert provenance["source#first"][MULTIMODAL_SHA256_KEY] == first
    assert provenance["source#second"][MULTIMODAL_SHA256_KEY] == second
    assert MULTIMODAL_SHA256_KEY not in provenance["source#text"]


def test_online_dump_rejects_missing_or_unexpected_media_digests(tmp_path):
    ids = torch.tensor([1, 2, 3, 4])
    cases = [
        (True, {}, "media digest for every sample"),
        (True, {MULTIMODAL_SHA256_KEY: "not-a-digest"}, "Invalid media digest"),
        (False, {MULTIMODAL_SHA256_KEY: "a" * 64}, "multimodal capture is disabled"),
    ]
    for index, (multimodal, metadata, message) in enumerate(cases):
        writer = OnlineDumpWriter(
            None,
            tmp_path / str(index),
            _recipe(multimodal=multimodal),
            validation_fraction=0.25,
            seed=1,
            min_loss_tokens=1,
            max_pending_bytes=1024,
            store=_Store({"k": _values(ids)}),
        )
        with pytest.raises(ValueError, match=message):
            writer.stage_batch([_result("k", ids, metadata=metadata)])
        writer.abort()
        assert not (tmp_path / str(index) / "manifest.json").exists()


def test_online_writer_backpressures_and_propagates_failure(tmp_path):
    first_ids = torch.tensor([1, 2, 3, 4])
    second_ids = torch.tensor([5, 6, 7, 8])
    writer = OnlineDumpWriter(
        None,
        tmp_path,
        _recipe(),
        validation_fraction=0.25,
        seed=1,
        min_loss_tokens=1,
        max_pending_bytes=1,
        store=_Store({"a": _values(first_ids), "b": _values(second_ids)}),
    )
    release = threading.Event()
    original = writer.writer.write

    def slow_write(*args, **kwargs):
        release.wait(timeout=5)
        return original(*args, **kwargs)

    writer.writer.write = slow_write
    writer.stage_batch([_result("a", first_ids)])
    thread = threading.Thread(target=lambda: writer.stage_batch([_result("b", second_ids)]))
    thread.start()
    time.sleep(0.05)
    assert thread.is_alive()
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    writer.finalize()

    failed = OnlineDumpWriter(
        None,
        tmp_path / "failed",
        _recipe(),
        validation_fraction=0.25,
        seed=1,
        min_loss_tokens=1,
        max_pending_bytes=1024,
        store=_Store({"a": _values(first_ids)}),
    )
    failed.writer.write = mock.Mock(side_effect=OSError("disk full"))
    with pytest.raises(RuntimeError, match="disk full"):
        failed.stage_batch([_result("a", first_ids)])
        failed.finalize()
    failed.abort()
    assert not (tmp_path / "failed" / "manifest.json").exists()


def test_streamed_safetensors_file_is_exact_and_deterministic(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import load as load_bytes

    tensors = {
        "input_ids": torch.tensor([5, 1, 4], dtype=torch.int64),
        "loss_mask": torch.tensor([0, 1, 1], dtype=torch.uint8),
        "hidden_layer_8": torch.randn(3, 4).to(torch.bfloat16),
        "hidden_layer_16": torch.randn(3, 4).to(torch.bfloat16),
    }
    metadata = {"format": "angelspec_hidden_v2", "id": "abc", "hidden_size": "4"}
    path = tmp_path / "sample.safetensors"
    with open(path, "wb") as stream:
        size, sha256, digests = write_safetensors(stream, tensors, metadata)
    payload = path.read_bytes()
    assert size == len(payload)
    assert sha256 == __import__("hashlib").sha256(payload).hexdigest()
    assert digests == {name: tensor_digest(tensor) for name, tensor in tensors.items()}
    loaded = load_bytes(payload)
    assert all(torch.equal(loaded[name], tensor) for name, tensor in tensors.items())
    with safe_open(str(path), framework="pt") as reader:
        assert reader.metadata() == metadata
    again = tmp_path / "again.safetensors"
    with open(again, "wb") as stream:
        write_safetensors(
            stream, dict(reversed(tensors.items())), dict(reversed(metadata.items()))
        )
    assert again.read_bytes() == payload


class _ConcurrentStore(_Store):
    def __init__(self, values, delays=None):
        super().__init__(values)
        self.delays = delays or {}
        self.lock = threading.Lock()
        self.active = self.peak = 0

    def get_into(self, key, tensors):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delays.get(key, 0.02))
            return super().get_into(key, tensors)
        finally:
            with self.lock:
                self.active -= 1


def test_online_writer_reads_in_parallel_and_keeps_the_first_duplicate(tmp_path):
    first_ids = torch.tensor([1, 2, 3, 4])
    others = {key: torch.tensor([index + 5, 2, 3, 4]) for index, key in enumerate("bcd")}
    values = {"first": _values(first_ids), "again": _values(first_ids)}
    values.update({key: _values(ids) for key, ids in others.items()})
    store = _ConcurrentStore(values, delays={"first": 0.2, "again": 0.0})
    writer = OnlineDumpWriter(
        None,
        tmp_path,
        _recipe(),
        validation_fraction=0.25,
        seed=1,
        min_loss_tokens=1,
        max_pending_bytes=1 << 20,
        reader_threads=4,
        store=store,
    )
    results = [_result("first", first_ids), _result("again", first_ids)]
    results += [_result(key, ids) for key, ids in others.items()]
    assert writer.stage_batch(results)["staged"] == 4
    assert store.peak > 1
    status = writer.finalize()
    assert (status["published"], status["duplicates"]) == (4, 1)
    assert status["read_bytes"] == status["written_bytes"] == 4 * 4 * (9 + 2 * 2 * 2)
    data_ids = {record["provenance"]["data_id"] for record in HiddenDataset(tmp_path).records}
    assert data_ids == {"source#first", "source#b", "source#c", "source#d"}


def test_online_writer_gives_each_reading_thread_its_own_client(tmp_path):
    from angelspec.config.mooncake_config import MooncakeConfig

    ids = {f"k{index}": torch.tensor([index + 1, 2, 3, 4]) for index in range(6)}
    clients = []

    class Client(_ConcurrentStore):
        def __init__(self, config):
            super().__init__({key: _values(value) for key, value in ids.items()})
            self.config = config
            self.threads = set()
            clients.append(self)

        def setup(self, device):
            assert device == torch.device("cpu")

        def get_into(self, key, tensors):
            self.threads.add(threading.current_thread().name)
            return super().get_into(key, tensors)

    recipe = {**_recipe(), "max_seq_length": 16}
    results = [_result(key, value) for key, value in ids.items()]
    with mock.patch("angelspec.transfer.mooncake.draft_store.DraftMooncakeStore", Client):
        writer = OnlineDumpWriter(
            MooncakeConfig(),
            tmp_path,
            recipe,
            validation_fraction=0.25,
            seed=1,
            min_loss_tokens=1,
            max_pending_bytes=1 << 20,
            reader_threads=3,
        )
        assert writer.stage_batch(results)["staged"] == 6
        writer.release(results)
        assert writer.finalize()["published"] == 6
    actor_client, *reading = clients
    assert reading and all(len(client.threads) == 1 for client in reading)
    assert len({name for client in reading for name in client.threads}) == len(reading)
    assert not actor_client.threads and actor_client.removed == list(ids)
    assert all(client.closed for client in clients)
    assert {client.config.global_segment_size for client in clients} == {0}
    assert {client.config.local_buffer_size for client in clients} == {64 << 20}


def test_cpu_mooncake_client_does_not_create_cuda_stream():
    config = SimpleNamespace(
        protocol="tcp",
        device_name="",
        local_hostname="localhost",
        metadata_server="http://localhost/metadata",
        global_segment_size=0,
        local_buffer_size=1,
        master_server_address="localhost:1",
        async_put_pool_size=0,
        enable_gpu_direct=False,
        apply_env_defaults=lambda: None,
    )
    store = MooncakeHiddenStateStore(config)
    distributed = mock.Mock()
    distributed.setup.return_value = 0
    with (
        mock.patch(
            "angelspec.transfer.mooncake.store.MooncakeDistributedStore",
            return_value=distributed,
        ),
        mock.patch.object(store, "_verify_force_delete"),
        mock.patch.object(store, "_build_replicate_config"),
        mock.patch("angelspec.transfer.mooncake.store.torch.cuda.is_available", return_value=True),
        mock.patch("angelspec.transfer.mooncake.store.torch.cuda.Stream") as stream,
    ):
        store.setup(torch.device("cpu"))
    stream.assert_not_called()


def test_incomplete_online_dump_can_be_inspected_and_finalized(tmp_path):
    writer = HiddenWriter(tmp_path, _recipe(), hold_lock=True)
    tensors = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "loss_mask": torch.tensor([0, 1, 1, 1], dtype=torch.uint8),
        "hidden_layer_1": torch.ones(4, 2, dtype=torch.bfloat16),
        "hidden_layer_3": torch.ones(4, 2, dtype=torch.bfloat16),
    }
    writer.write("sample", tensors, provenance={"data_id": "raw#1"})
    running = online_dump_status(tmp_path, verify=True)
    assert running["samples"] == 1
    assert running["layer_ids"] == [1, 3]
    writer.close()
    status = finalize_online_dump(tmp_path)
    assert status["complete"] is True
    assert status["verified"] == 1


class _Queue:
    def __init__(self, *args, **kwargs):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _Remote:
    def __init__(self, fn):
        self.remote = fn


def _controller(*, packing=False, sp_size=2, **extra):
    args = SimpleNamespace(
        **extra,
        attention_backend="usp" if sp_size > 1 else "sdpa",
        sp_ulysses_size=sp_size,
        sp_ring_size=1,
        sp_leader_mooncake_fetch=False,
        per_dp_rank_batch_size=1,
        dflash_packing=packing,
        mtp_packing=False,
        max_seq_length=8,
        packing_min_fill_ratio=0.0,
        packing_max_wait_seconds=0.0,
        packing_balance_dp=False,
        seed=1,
        shuffle_dataset=False,
        online_dump_capture_epochs=1,
    )
    with mock.patch("angelspec.controller.training_controller.Queue", _Queue):
        return AsyncTrainingController.__ray_actor_class__(args, dp_size=2)


def _fill_controller(controller, count=2):
    results = []
    for index in range(count):
        result = InferenceOutput(
            data_id=f"d{index}",
            mooncake_key=f"k{index}",
            tensor_shapes={"input_ids": (4,)},
        )
        results.append(result)
        controller.sample_pool.append(result)
        controller._sample_bytes[result.mooncake_key] = 4
        controller._pool_bytes += 4
    return results


def test_controller_stages_once_before_dp_sp_fanout():
    controller = _controller()
    _fill_controller(controller)
    calls, released = [], []

    def stage(batch, capture_epoch, placements):
        assert not controller.sample_pool
        calls.append(([item.mooncake_key for item in batch], capture_epoch, placements))
        return {"staged": len(batch)}

    def release(batch):
        released.append([item.mooncake_key for item in batch])
        return len(batch)

    controller._online_dump_writers = [
        SimpleNamespace(stage_batch=_Remote(stage), release=_Remote(release))
    ]
    assert controller.try_dispatch_batch()
    assert calls == [
        (
            ["k0", "k1"],
            0,
            [
                {"dispatch": 0, "dp_rank": 0, "slot": 0},
                {"dispatch": 0, "dp_rank": 1, "slot": 0},
            ],
        )
    ]
    assert controller._pool_bytes == 0
    assert (controller._dump_inflight_bytes, controller._dump_inflight_samples) == (8, 2)
    queued = [sample for queue in controller.train_queues for sample in queue.items]
    assert len(queued) == 4
    assert all(sample.retain_for_dump for sample in queued)
    assert released == []
    controller.release_online_dump_consumed()
    assert released == [["k0", "k1"]]
    assert controller._dump_unreleased == []


def test_controller_dump_window_pauses_dispatch_and_surfaces_writer_errors():
    tc = __import__("angelspec.controller.training_controller", fromlist=["ray"])
    controller = _controller()
    controller._dump_max_inflight = 1
    _fill_controller(controller, count=4)
    controller._online_dump_writers = [
        SimpleNamespace(stage_batch=_Remote(lambda *args, **kwargs: "ref"))
    ]
    assert controller.try_dispatch_batch()
    with mock.patch.object(tc.ray, "wait", return_value=([], ["ref"])):
        assert controller.try_dispatch_batch() is False
        assert controller.get_pool_size() == 4
    with (
        mock.patch.object(tc.ray, "wait", return_value=(["ref"], [])),
        mock.patch.object(tc.ray, "get", side_effect=lambda value: value),
    ):
        assert controller.try_dispatch_batch()
    with (
        mock.patch.object(tc.ray, "wait", return_value=(["ref"], [])),
        mock.patch.object(tc.ray, "get", side_effect=OSError("disk full")),
    ):
        with pytest.raises(RuntimeError, match="disk full"):
            controller.try_dispatch_batch()
    assert controller._dump_inflight_samples == 0


def test_controller_dump_window_covers_one_optimizer_step():
    assert _controller(draft_accumulation_steps=6)._dump_max_inflight == 6
    explicit = _controller(draft_accumulation_steps=6, online_dump_max_inflight_batches=2)
    assert explicit._dump_max_inflight == 2
    assert _controller(online_dump_dump_only=True)._dump_max_inflight == 2
    dump_only = _controller(online_dump_dump_only=True, draft_accumulation_steps=6)
    assert dump_only._dump_max_inflight == 6
    single = _controller(online_dump_dump_only=True, online_dump_max_inflight_batches=1)
    assert single._dump_max_inflight == 1


def test_one_writer_per_inference_node_off_the_driver():
    from angelspec.controller import setup

    nodes = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.2"]

    def placed(writers=None, node_ip=None, inference=nodes):
        args = SimpleNamespace(online_dump_writers=writers, online_dump_writer_node_ip=node_ip)
        return setup.writer_node_ips(args, inference)

    with mock.patch.object(setup.ray.util, "get_node_ip_address", return_value="10.0.0.1"):
        assert placed() == placed(node_ip="auto") == ["10.0.0.2", "10.0.0.3"]
        assert placed(writers=3) == ["10.0.0.2", "10.0.0.3", "10.0.0.2"]
        assert placed(inference=["10.0.0.1"]) == placed(inference=[]) == [None]
        assert placed(writers=2, inference=["10.0.0.1"]) == [None, None]
        assert placed(node_ip="10.0.0.9") == ["10.0.0.9"]
        assert placed(writers=2, node_ip="10.0.0.9") == ["10.0.0.9", "10.0.0.9"]


def _routed(key, mask, metadata=None):
    return InferenceOutput(
        data_id=f"source#{key}",
        mooncake_key=key,
        tensor_shapes={"input_ids": (sum(mask),)},
        packed_loss_mask=",".join(str(length) for length in mask),
        metadata=metadata or {},
    )


def test_dump_route_keeps_equal_samples_on_one_writer():
    metadata = {IMAGE_NEWLINE_SLOTS_KEY: [3], MULTIMODAL_SHA256_KEY: "a" * 64}
    first = _routed("first", [2, 5], metadata)
    again = _routed("again", [2, 5], dict(metadata))
    again.packed_loss_mask = [2, 5]
    assert all(dump_route(first, count) == dump_route(again, count) for count in (2, 3, 8))
    assert dump_route(first, 1) == 0
    spread = {dump_route(_routed(f"s{index}", [index, 9]), 4) for index in range(64)}
    assert spread == {0, 1, 2, 3}


def _dump_writer(name, calls):
    def finalize(verify="header", publish=True, records=()):
        calls.append((name, "finalize", publish, list(records)))
        status = {
            "complete": publish,
            "staged": 4,
            "written": 4,
            "published": 8 * publish,
            "elapsed_seconds": 1.0 + publish,
            "error": None,
        }
        return status if publish else {**status, "records": [{"id": name}]}

    return SimpleNamespace(
        stage_batch=_Remote(
            lambda batch, capture_epoch, placements: calls.append(
                (name, "stage", [item.mooncake_key for item in batch], placements)
            )
            or name
        ),
        release=_Remote(
            lambda batch: calls.append((name, "release", [item.mooncake_key for item in batch]))
            or name
        ),
        finalize=_Remote(finalize),
    )


def test_controller_splits_dump_batches_and_publishes_on_the_first_writer():
    tc = __import__("angelspec.controller.training_controller", fromlist=["ray"])
    controller = _controller()
    calls = []
    controller._online_dump_writers = [_dump_writer("w0", calls), _dump_writer("w1", calls)]
    results = [_routed(f"k{index}", [index, 4]) for index in range(8)]
    placements = [{"dispatch": 0, "slot": index} for index in range(8)]
    routes = [dump_route(result, 2) for result in results]
    assert set(routes) == {0, 1}
    with (
        mock.patch.object(tc.ray, "wait", side_effect=lambda refs, **kwargs: (list(refs), [])),
        mock.patch.object(tc.ray, "get", side_effect=lambda value: value),
    ):
        assert controller._stage_online_dump(results, placements, 80)
        assert len(controller._dump_inflight) == 1
        controller.release_online_dump_consumed()
        status = controller.finalize_online_dump()
    shares = {
        name: [result.mooncake_key for result, route in zip(results, routes) if route == index]
        for index, name in enumerate(("w0", "w1"))
    }
    assert calls[:2] == [
        (name, "stage", share, [placements[int(key[1:])] for key in share])
        for name, share in shares.items()
    ]
    assert calls[2:4] == [(name, "release", share) for name, share in shares.items()]
    assert calls[4:] == [("w1", "finalize", False, []), ("w0", "finalize", True, [{"id": "w1"}])]
    assert "records" not in status
    assert (status["published"], status["written"], status["writers"]) == (8, 8, 2)
    assert status["elapsed_seconds"] == 2.0 and status["complete"] is True
    assert controller._online_dump_writers == []


_read_json = offline.read_json


def _no_receipt_reads(path):
    assert "receipts" not in str(path), f"reread {path}"
    return _read_json(path)


def test_other_writers_resume_from_receipt_names(tmp_path):
    ids = torch.tensor([1, 2, 3, 4])
    store = _Store({"k": _values(ids)})
    options = dict(validation_fraction=1e-12, seed=7, min_loss_tokens=1, max_pending_bytes=1024)
    first = OnlineDumpWriter(None, tmp_path, _recipe(), store=store, **options)
    assert first.stage_batch([_result("k", ids)])["staged"] == 1
    first.abort()
    with mock.patch.object(offline, "read_json", side_effect=_no_receipt_reads):
        second = OnlineDumpWriter(
            None, tmp_path, _recipe(), store=store, writer_index=1, writer_count=2, **options
        )
    assert second.stage_batch([_result("k", ids)])["staged"] == 0
    assert second.status()["duplicates"] == 1
    second.abort()


def test_writers_share_one_dump_and_the_first_publishes_it(tmp_path):
    keys = {
        "a": torch.tensor([1, 2, 3, 4]),
        "b": torch.tensor([5, 6, 7]),
        "c": torch.tensor([8, 9]),
    }
    store = _Store({key: _values(ids) for key, ids in keys.items()})
    writers = [
        OnlineDumpWriter(
            None,
            tmp_path,
            _recipe(),
            validation_fraction=1e-12,
            seed=7,
            min_loss_tokens=1,
            max_pending_bytes=1024,
            store=store,
            writer_index=index,
            writer_count=2,
        )
        for index in range(2)
    ]
    assert (tmp_path / ".writer.lock").exists() and (tmp_path / ".writer-1.lock").exists()
    assert writers[0].stage_batch([_result("a", keys["a"])])["staged"] == 1
    assert (
        writers[1].stage_batch([_result("b", keys["b"]), _result("c", keys["c"])])["staged"] == 2
    )
    for writer, staged in zip(writers, (1, 2)):
        deadline = time.monotonic() + 30
        while writer.status()["written"] < staged:
            assert time.monotonic() < deadline
            time.sleep(0.01)
    running = online_dump_status(tmp_path)
    assert (running["samples"], running["writers"]) == (3, 2)
    others = writers[1].finalize(publish=False)
    assert (others["published"], len(others["records"])) == (0, 2)
    assert not (tmp_path / "manifest.json").exists()
    with mock.patch.object(offline, "read_json", side_effect=_no_receipt_reads):
        assert writers[0].finalize(records=others["records"])["published"] == 3
    assert HiddenDataset(tmp_path).verify() == 3
    with pytest.raises(ValueError, match="writer index"):
        OnlineDumpWriter(
            None,
            tmp_path / "other",
            _recipe(),
            validation_fraction=0.5,
            seed=7,
            min_loss_tokens=1,
            max_pending_bytes=1024,
            store=store,
            writer_index=2,
            writer_count=2,
        )


def test_low_free_space_reading_must_persist(tmp_path):
    from angelspec.inference import online_dump

    writer = OnlineDumpWriter(
        None,
        tmp_path,
        _recipe(),
        validation_fraction=0.5,
        seed=7,
        min_loss_tokens=1,
        max_pending_bytes=1024,
        store=_Store({}),
    )
    readings = iter([10, 1 << 40])
    with (
        mock.patch.object(
            online_dump.shutil,
            "disk_usage",
            side_effect=lambda root: SimpleNamespace(free=next(readings)),
        ),
        mock.patch.object(online_dump.time, "sleep") as sleep,
    ):
        writer._require_space(100)
        writer._require_space(100)
    assert sleep.call_count == 1
    writer._space_checked = float("-inf")
    with (
        mock.patch.object(online_dump.shutil, "disk_usage", return_value=SimpleNamespace(free=10)),
        mock.patch.object(online_dump.time, "sleep") as sleep,
    ):
        with pytest.raises(RuntimeError, match="reports 10 free"):
            writer._require_space(100)
    assert sleep.call_count == online_dump._SPACE_CONFIRMATIONS - 1
    writer.abort()


def test_controller_counts_only_dispatch_time_held_back_by_the_writer():
    tc = __import__("angelspec.controller.training_controller", fromlist=["ray"])
    controller = _controller()
    controller._dump_max_inflight = 1
    _fill_controller(controller, count=4)
    controller._online_dump_writers = [
        SimpleNamespace(stage_batch=_Remote(lambda *args, **kwargs: "ref"))
    ]
    assert controller.try_dispatch_batch()
    assert controller.get_online_dump_wait_seconds() == 0.0
    with mock.patch.object(tc.ray, "wait", return_value=([], ["ref"])):
        assert controller.try_dispatch_batch() is False
        time.sleep(0.05)
        assert controller.get_online_dump_wait_seconds() >= 0.05
    with (
        mock.patch.object(tc.ray, "wait", return_value=(["ref"], [])),
        mock.patch.object(tc.ray, "get", side_effect=lambda value: value),
    ):
        assert controller.try_dispatch_batch()
    waited = controller.get_online_dump_wait_seconds()
    time.sleep(0.02)
    assert controller.get_online_dump_wait_seconds() == waited


def test_retained_samples_are_left_for_the_dump_writer():
    from angelspec.training.data_fetcher import (
        MOONCAKE_DELETE_KEY_FIELD,
        MooncakeDataset,
        TrainSample,
    )

    fetcher = object.__new__(MooncakeDataset)
    fetcher.mooncake_store = mock.Mock()
    shapes = {"input_ids": (4,), "last_hidden_states": (4, 2)}
    for usp_local_shard in (False, True):
        fetcher.usp_local_shard = usp_local_shard
        result = {}
        fetcher._cleanup_mooncake_data(TrainSample("kept", shapes, retain_for_dump=True), result)
        assert MOONCAKE_DELETE_KEY_FIELD not in result
    fetcher.mooncake_store.remove_tensors.assert_not_called()
    fetcher.usp_local_shard = False
    fetcher._cleanup_mooncake_data(TrainSample("freed", shapes), {})
    fetcher.mooncake_store.remove_tensors.assert_called_once()


def test_packed_controller_stages_original_samples_before_rows():
    controller = _controller(packing=True, sp_size=1)
    results = _fill_controller(controller)
    for result in results:
        result.tensor_shapes["input_ids"] = (5,)
    calls = []

    def stage(batch, capture_epoch, placements):
        assert not controller.sample_pool
        calls.append(([item.mooncake_key for item in batch], placements))
        return {"staged": len(batch)}

    controller._online_dump_writers = [SimpleNamespace(stage_batch=_Remote(stage))]
    assert controller.try_dispatch_batch()
    assert calls == [
        (
            ["k0", "k1"],
            [
                {"dispatch": 0, "dp_rank": 0, "row_id": 0, "row_position": 0},
                {"dispatch": 0, "dp_rank": 1, "row_id": 1, "row_position": 0},
            ],
        )
    ]
    assert [item.mooncake_key for item in controller._dump_unreleased] == ["k0", "k1"]
    queued = [
        (rank, sample.mooncake_key, sample.row_id, sample.retain_for_dump)
        for rank, queue in enumerate(controller.train_queues)
        for sample in queue.items
    ]
    assert queued == [(0, "k0", 0, True), (1, "k1", 1, True)]


def test_training_loop_publishes_only_on_success():
    args = SimpleNamespace(online_dump_enabled=True)
    calls = []
    controller = SimpleNamespace(
        flush_online_dump_staging=_Remote(lambda: calls.append("flush")),
        finalize_online_dump=_Remote(lambda: calls.append("finalize") or {"complete": True}),
        abort_online_dump=_Remote(lambda: calls.append("abort") or {"complete": False}),
    )
    inference_manager = SimpleNamespace(run=_Remote(lambda: "run"))
    with (
        mock.patch.object(loop, "training_loop", return_value=9),
        mock.patch.object(
            loop, "_safe_training_cleanup", side_effect=lambda **kwargs: calls.append("cleanup")
        ),
        mock.patch.object(loop.ray, "get", side_effect=lambda value: value),
    ):
        assert loop.run_training_loop(args, controller, inference_manager, object()) == 9
    assert calls == ["flush", "cleanup", "finalize"]

    calls.clear()
    with (
        mock.patch.object(loop, "training_loop", side_effect=RuntimeError("train failed")),
        mock.patch.object(loop, "_safe_training_cleanup"),
        mock.patch.object(loop.ray, "get", side_effect=lambda value: value),
    ):
        with pytest.raises(RuntimeError, match="train failed"):
            loop.run_training_loop(args, controller, inference_manager, object())
    assert calls == ["abort"]


def test_dump_only_drain_stages_then_frees_pooled_samples():
    tc = __import__("angelspec.controller.training_controller", fromlist=["ray"])
    controller = _controller()
    _fill_controller(controller, count=3)
    calls = []

    def stage(batch, capture_epoch, placements):
        calls.append(("stage", [item.mooncake_key for item in batch], capture_epoch, placements))
        return "staged"

    def release(batch):
        calls.append(("release", [item.mooncake_key for item in batch]))
        return "released"

    controller._online_dump_writers = [
        SimpleNamespace(stage_batch=_Remote(stage), release=_Remote(release))
    ]
    with (
        mock.patch.object(tc.ray, "wait", side_effect=lambda refs, **kwargs: (list(refs), [])),
        mock.patch.object(tc.ray, "get", side_effect=lambda value: value),
    ):
        assert controller.drain_to_online_dump(2) == 2
        assert controller.drain_to_online_dump(2) == 1
        assert controller.drain_to_online_dump(2) == 0
    assert calls == [
        ("stage", ["k0", "k1"], 0, None),
        ("release", ["k0", "k1"]),
        ("stage", ["k2"], 0, None),
        ("release", ["k2"]),
    ]
    assert controller._pool_bytes == 0
    assert controller._epoch_consumed_samples == 3
    assert all(not queue.items for queue in controller.train_queues)

    windowed = _controller()
    _fill_controller(windowed, count=2)
    windowed._dump_max_inflight = 1
    windowed._online_dump_writers = [
        SimpleNamespace(
            stage_batch=_Remote(lambda *args, **kwargs: "ref"),
            release=_Remote(lambda batch: "freed"),
        )
    ]
    with mock.patch.object(tc.ray, "wait", side_effect=lambda refs, **kwargs: ([], list(refs))):
        assert windowed.drain_to_online_dump(1) == 1
        assert windowed.drain_to_online_dump(1) == 0
    windowed._inference_error = "engine died"
    with pytest.raises(RuntimeError, match="engine died"):
        windowed.drain_to_online_dump(1)


def _dump_only_controller(drains, statuses, calls):
    return SimpleNamespace(
        submit_training_dataset=_Remote(lambda epoch, skip: calls.append(("submit", epoch, skip))),
        drain_to_online_dump=_Remote(lambda max_samples: next(drains)),
        get_full_status=_Remote(lambda: next(statuses)),
        reload_dataset=_Remote(lambda: calls.append("reload")),
    )


def _dump_status(consumed=0, pool=0):
    return {"epoch_consumed_samples": consumed, "prompt_buffer_size": 0, "sample_pool_size": pool}


def _manager_status(queued=0, pending=0, in_transit=0):
    return {"prompt_buffer_size": queued, "pending_tasks": pending, "in_transit": in_transit}


def _dump_only(drains, statuses, managers, *, dataset_size, epochs=1, stopped=False):
    drains, statuses, managers = iter(drains), iter(statuses), iter(managers)
    calls = []
    args = SimpleNamespace(online_dump_capture_epochs=epochs, inference_batch_size=2)
    controller = _dump_only_controller(drains, statuses, calls)
    manager = SimpleNamespace(get_status=_Remote(lambda: next(managers)))
    with (
        mock.patch.object(loop.ray, "get", side_effect=lambda value: value),
        mock.patch.object(
            loop.ray, "wait", return_value=(["run"], []) if stopped else ([], ["run"])
        ),
        mock.patch.object(loop.time, "sleep"),
        mock.patch.object(loop, "_DUMP_CHECK_SECONDS", 0.0),
    ):
        loop.dump_only_loop(args, controller, manager, "run", dataset_size)
    assert next(drains, None) is None
    assert next(statuses, None) is None and next(managers, None) is None
    return calls


def test_dump_only_loop_reloads_until_capture_epochs():
    calls = _dump_only(
        [2, 1, 0, 2, 1, 0],
        [_dump_status(), _dump_status(3), _dump_status(), _dump_status(3)],
        [_manager_status(), _manager_status()],
        dataset_size=3,
        epochs=2,
    )
    assert calls == [("submit", 0, 0), "reload"]


def test_dump_only_waits_for_requests_in_flight_and_results_in_transit():
    calls = _dump_only(
        [2, 0, 0, 0, 1, 0],
        [
            _dump_status(),
            _dump_status(),
            _dump_status(),
            _dump_status(pool=1),
            _dump_status(),
            _dump_status(3),
        ],
        [
            _manager_status(pending=1),
            _manager_status(in_transit=1),
            _manager_status(),
            _manager_status(),
        ],
        dataset_size=3,
    )
    assert calls == [("submit", 0, 0)]


def test_dump_only_epoch_ends_once_inference_drops_samples():
    calls = _dump_only(
        [2, 0], [_dump_status(), _dump_status(2)], [_manager_status()], dataset_size=3
    )
    assert calls == [("submit", 0, 0)]


def test_dump_only_hands_each_reader_thread_two_samples():
    sizes, drains = [], iter([3, 0])
    controller = SimpleNamespace(
        submit_training_dataset=_Remote(lambda epoch, skip: None),
        drain_to_online_dump=_Remote(
            lambda max_samples: sizes.append(max_samples) or next(drains)
        ),
        get_full_status=_Remote(lambda: _dump_status(3)),
        reload_dataset=_Remote(lambda: None),
    )
    manager = SimpleNamespace(get_status=_Remote(lambda: _manager_status()))
    args = SimpleNamespace(
        online_dump_capture_epochs=1, inference_batch_size=2, online_dump_reader_threads=6
    )
    with (
        mock.patch.object(loop.ray, "get", side_effect=lambda value: value),
        mock.patch.object(loop.ray, "wait", return_value=([], ["run"])),
        mock.patch.object(loop.time, "sleep"),
        mock.patch.object(loop, "_DUMP_CHECK_SECONDS", 0.0),
    ):
        loop.dump_only_loop(args, controller, manager, "run", 3)
    assert sizes == [12, 12]


def test_dump_only_fails_when_the_inference_manager_stops():
    with pytest.raises(RuntimeError, match="Inference manager stopped"):
        _dump_only([0], [], [], dataset_size=3, stopped=True)


def test_dump_only_run_publishes_only_on_success():
    args = SimpleNamespace(online_dump_enabled=True)
    calls = []
    controller = SimpleNamespace(
        flush_online_dump_staging=_Remote(lambda: calls.append("flush")),
        finalize_online_dump=_Remote(lambda: calls.append("finalize") or {"complete": True}),
        abort_online_dump=_Remote(lambda: calls.append("abort") or {"complete": False}),
    )
    inference_manager = SimpleNamespace(run=_Remote(lambda: "run"))

    def cleanup(**kwargs):
        calls.append("cleanup")

    with (
        mock.patch.object(loop, "dump_only_loop", side_effect=lambda *a: calls.append("dump")),
        mock.patch.object(loop, "_safe_training_cleanup", side_effect=cleanup),
        mock.patch.object(loop.ray, "get", side_effect=lambda value: value),
    ):
        loop.run_dump_only_loop(args, controller, inference_manager, [], 3)
    assert calls == ["dump", "flush", "cleanup", "finalize"]

    calls.clear()
    with (
        mock.patch.object(loop, "dump_only_loop", side_effect=RuntimeError("engine died")),
        mock.patch.object(loop, "_safe_training_cleanup", side_effect=cleanup),
        mock.patch.object(loop.ray, "get", side_effect=lambda value: value),
    ):
        with pytest.raises(RuntimeError, match="engine died"):
            loop.run_dump_only_loop(args, controller, inference_manager, [], 3)
    assert calls == ["abort", "cleanup"]
