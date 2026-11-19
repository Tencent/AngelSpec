import dataclasses
import hashlib
import json
import os
import queue
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import torch

from angelspec.data.media_identity import MEDIA_DIGEST_VERSION, MULTIMODAL_SHA256_KEY
from angelspec.data.offline import (
    HiddenDataset,
    HiddenWriter,
    atomic_json,
    file_hash,
    layer_recipe,
    layer_tensor_name,
    object_hash,
    read_json,
)
from angelspec.data.offline_assets import (
    export_target_assets,
    mtp_init_prefix,
    read_target_components,
    target_weight_prefixes,
    tensor_hash,
)
from angelspec.data.positions import (
    IMAGE_NEWLINE_SLOTS_KEY,
    position_recipe_from_config,
)
from angelspec.data.utils import deserialize_packed_loss_mask, unpack_loss_mask
from angelspec.utils.logging import logger

CAPTURE_CODE = (
    "data/offline.py",
    "data/media_identity.py",
    "inference/online_dump.py",
    "controller/training_controller.py",
)
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_SPACE_CHECK_SECONDS = 1.0
_SPACE_CONFIRMATIONS = 5
_SUMMED_STATUS = (
    "staged",
    "written",
    "duplicates",
    "skipped",
    "pending_bytes",
    "read_bytes",
    "read_seconds",
    "written_bytes",
    "write_seconds",
    "published",
)


def dump_route(result, writers):
    """Index of the writer that copies ``result``; equal samples share a writer.

    A sample ID hashes the token IDs, the loss mask and the position and media
    metadata. Token IDs are only known after a Mooncake read, so the route
    hashes the rest, which equal samples share as well.
    """
    if writers == 1:
        return 0
    result = _result_dict(result)
    metadata = result.get("metadata") or {}
    packed = result.get("packed_loss_mask")
    if isinstance(packed, str):
        packed = deserialize_packed_loss_mask(packed)
    key = json.dumps(
        [
            [int(length) for length in packed or ()],
            [int(size) for size in (result.get("tensor_shapes") or {}).get("input_ids") or ()],
            metadata.get(IMAGE_NEWLINE_SLOTS_KEY),
            metadata.get(MULTIMODAL_SHA256_KEY),
        ]
    )
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % writers


def merge_dump_status(statuses):
    """One status for several writers of the same dump."""
    if len(statuses) == 1:
        return statuses[0]
    merged = dict(statuses[0])
    for key in _SUMMED_STATUS:
        if key in merged:
            merged[key] = sum(status.get(key, 0) for status in statuses)
    merged["elapsed_seconds"] = max(status.get("elapsed_seconds", 0) for status in statuses)
    merged["error"] = next((status["error"] for status in statuses if status.get("error")), None)
    merged["writers"] = len(statuses)
    return merged


def materialize_native_capture(result, tokens, store, recipe, *, allow_extra=False):
    """Read one Mooncake capture straight into owned per-layer native CPU tensors.

    Mooncake carries the auxiliary layers concatenated in
    ``capture_aux_layer_order`` and the final layer separately; both are split
    into one named tensor per layer. Token IDs and the final layer are read in
    place; the interleaved auxiliary layers take one copy to split.
    """
    aux_order = list(recipe["capture_aux_layer_order"])
    final = recipe["final_layer_id"]
    if sorted(aux_order + [final]) != recipe["layer_ids"]:
        raise ValueError("Recipe capture order differs from its stored layers")
    shapes = result["tensor_shapes"]
    required = {"input_ids", "last_hidden_states"} | ({"hidden_states"} if aux_order else set())
    if not required.issubset(shapes) or (not allow_extra and set(shapes) != required):
        raise ValueError(f"Unexpected capture tensor keys: {sorted(shapes)}")
    dtypes = {
        key: getattr(torch, value.removeprefix("torch.")) if isinstance(value, str) else value
        for key, value in result["tensor_dtypes"].items()
    }
    if any(dtypes.get(key) != torch.bfloat16 for key in required - {"input_ids"}):
        raise ValueError("Native capture expects BF16 hidden states")
    width = recipe["hidden_size"]
    expected_input_ids = tokens.get("input_ids")
    length = int(
        expected_input_ids.shape[-1]
        if isinstance(expected_input_ids, torch.Tensor)
        else tokens["loss_mask"].shape[-1]
    )
    expected_numel = {
        "input_ids": length,
        "last_hidden_states": length * width,
    }
    if aux_order:
        expected_numel["hidden_states"] = length * width * len(aux_order)

    for key in required:
        shape = shapes[key]
        if not shape or any(not isinstance(size, int) or size < 1 for size in shape):
            raise ValueError(f"Invalid captured shape for {key}")
        if torch.Size(shape).numel() != expected_numel[key]:
            raise ValueError(f"Captured tensor size differs from recipe for {key}")
    captured = {
        key: torch.empty(shapes[key], dtype=torch.int64 if key == "input_ids" else dtypes[key])
        for key in required
    }
    store.get_into(result["mooncake_key"], captured)
    tensors = {
        "input_ids": captured["input_ids"].reshape(length),
        "loss_mask": tokens["loss_mask"].reshape(length).to(torch.uint8).clone(),
        layer_tensor_name(final): captured["last_hidden_states"].reshape(length, width),
    }
    if aux_order:
        hidden = captured["hidden_states"].reshape(length, len(aux_order), width)
        for position, layer in enumerate(aux_order):
            tensors[layer_tensor_name(layer)] = hidden[:, position].contiguous()
    if expected_input_ids is not None and not torch.equal(
        tensors["input_ids"], expected_input_ids.reshape(length)
    ):
        raise ValueError("Inference returned different token IDs")
    return tensors


def validate_online_dump_args(args, draft_config) -> None:
    if not getattr(args, "online_dump_enabled", False):
        return
    from angelspec.models.draft.dflash import DFlashConfig
    from angelspec.models.draft.mtp import MTPConfig

    if not isinstance(draft_config, (DFlashConfig, MTPConfig)):
        raise ValueError("Online dump supports the DFlash family and MTP only")
    if getattr(args, "train_with_decode", False):
        raise ValueError("Online dump v1 does not support train_with_decode")
    if getattr(args, "defer_tokenization", False):
        raise ValueError("Online dump v1 requires defer_tokenization=false")
    multimodal = getattr(args, "vllm_enable_multimodal", False) or getattr(
        args, "sglang_enable_multimodal", False
    )
    if multimodal:
        if getattr(args, "inference_engine_type", None) != "vllm":
            raise ValueError("Online VL dump currently requires the vllm backend")
        position_recipe = position_recipe_from_config(draft_config)
        if position_recipe is None or position_recipe.provider != "hy_v4_flash":
            raise ValueError("Online VL dump requires a configured Hy4 flash position recipe")
    if getattr(args, "attention_backend", None) == "usp" and not getattr(
        args, "usp_local_shard", False
    ):
        raise ValueError("Online dump v1 requires usp_local_shard=true for USP")
    if getattr(args, "inference_engine_type", None) not in ("vllm", "sgl"):
        raise ValueError("Online dump v1 requires the vllm or sgl inference backend")
    if not getattr(args, "store_last_hidden_states", False):
        raise ValueError("Online dump requires inference.store_last_hidden_states=true")
    if getattr(args, "last_hidden_states_prenorm", None) is not True:
        raise ValueError(
            "Online dump requires verified pre-final-norm last hidden states; "
            "set inference.last_hidden_states_prenorm=true only for a backend that provides them"
        )
    draft_vocab = getattr(draft_config, "draft_vocab_size", None)
    if draft_vocab not in (None, draft_config.vocab_size):
        raise ValueError("Online dump does not support draft vocabulary pruning")
    dump = Path(args.online_dump_dump_path).resolve()
    frozen = Path(args.online_dump_frozen_target_path).resolve()
    target = Path(args.target_model_path).resolve()
    if dump == frozen or dump.is_relative_to(frozen) or frozen.is_relative_to(dump):
        raise ValueError("Online dump and frozen target directories must be separate")
    if dump == target or dump.is_relative_to(target):
        raise ValueError("Online dump output must not be inside the target model")


def build_online_capture_recipe(args, draft_config, target, target_identity):
    from angelspec.models.draft.mtp import MTPConfig
    from angelspec.models.draft.offline_family import offline_family, offline_layer_ids

    text_target = target.get("text_config", target)
    final = text_target["num_hidden_layers"] - 1
    if isinstance(draft_config, MTPConfig):
        aux_order = []
        family = "mtp"
    else:
        aux_order = [layer for layer in offline_layer_ids(draft_config) if layer != final]
        if not aux_order:
            raise ValueError("Online DFlash-family dump requires an auxiliary hidden layer")
        family = offline_family(draft_config)
    if draft_config.target_hidden_size != text_target["hidden_size"]:
        raise ValueError("Draft target hidden width differs from the online target")
    if draft_config.vocab_size != text_target["vocab_size"]:
        raise ValueError("Draft vocabulary differs from the online target")
    target_layers = getattr(
        draft_config, "target_num_hidden_layers", text_target["num_hidden_layers"]
    )
    if target_layers != text_target["num_hidden_layers"]:
        raise ValueError("Draft target layer count differs from the online target")
    position_recipe = position_recipe_from_config(draft_config)
    multimodal = bool(
        getattr(args, "vllm_enable_multimodal", False)
        or getattr(args, "sglang_enable_multimodal", False)
    )
    return {
        **layer_recipe(
            aux_order + [final],
            target_num_hidden_layers=text_target["num_hidden_layers"],
            hidden_size=text_target["hidden_size"],
        ),
        "capture_aux_layer_order": aux_order,
        "schema_version": 2 if multimodal else 1,
        "capture_backend": f"angelspec-online-{args.inference_engine_type}",
        "online_capture": True,
        "multimodal": multimodal,
        "model_family": family,
        "target_metadata_sha256": target_identity,
        "vocab_size": text_target["vocab_size"],
        "max_seq_length": args.max_seq_length,
        **(
            {"min_seq_length": int(args.min_seq_length)}
            if getattr(args, "min_seq_length", 0)
            else {}
        ),
        "storage_dtype": "bfloat16",
        "tokenizer": target_identity,
        "mask_policy": {
            "source": "online_training",
            "chat_template": getattr(args, "chat_template", None),
            "last_turn_loss_only": getattr(args, "last_turn_loss_only", False),
            "min_loss_tokens": getattr(args, "min_loss_tokens", 0),
        },
        "validation_fraction": args.online_dump_validation_fraction,
        "seed": args.seed,
        "capture_epochs": args.online_dump_capture_epochs,
        "position_recipe_fingerprint": (
            position_recipe.identity if position_recipe is not None else None
        ),
        "position_recipe": (
            dataclasses.asdict(position_recipe) if position_recipe is not None else None
        ),
        "sample_metadata_keys": [IMAGE_NEWLINE_SLOTS_KEY] if multimodal else [],
        **({"media_digest": MEDIA_DIGEST_VERSION} if multimodal else {}),
    }


def prepare_online_dump_assets(args, draft_config):
    """Export the frozen target and pin an online capture recipe before actors start."""
    validate_online_dump_args(args, draft_config)
    identity = export_target_assets(
        args.target_model_path,
        args.online_dump_frozen_target_path,
        init_prefix=mtp_init_prefix(args, draft_config),
        weight_prefixes=target_weight_prefixes(args.target_model_path, draft_config),
    )
    target, current_identity, components = read_target_components(args.target_model_path)
    del components
    if current_identity != identity:
        raise ValueError("Target changed while preparing online dump assets")
    return build_online_capture_recipe(args, draft_config, target, identity)


def _result_dict(result):
    if isinstance(result, dict):
        return result
    return {
        name: getattr(result, name)
        for name in (
            "data_id",
            "mooncake_key",
            "tensor_shapes",
            "tensor_dtypes",
            "packed_loss_mask",
            "metadata",
        )
    }


def _dtype(value):
    if isinstance(value, str):
        return getattr(torch, value.removeprefix("torch."))
    return value


def _capture_bytes(recipe, seq_len):
    """Bytes of one stored sample: int64 tokens, a uint8 mask and every BF16 layer."""
    return int(seq_len) * (9 + 2 * int(recipe["hidden_size"]) * len(recipe["layer_ids"]))


class OnlineDumpWriter:
    """Byte-bounded online capture worker with parallel reads and durable writes.

    ``stage_batch`` copies every tensor it keeps into owned CPU memory before it
    returns. With ``retained=True`` trainers leave the Mooncake keys alone and
    this writer deletes them once copied, so the controller does not have to
    wait for staging before dispatching the batch.
    """

    _STOP = object()

    def __init__(
        self,
        mooncake_config,
        root,
        recipe,
        *,
        validation_fraction,
        seed,
        min_loss_tokens,
        max_pending_bytes,
        writer_threads=16,
        reader_threads=8,
        store=None,
        mode="online",
        writer_index=0,
        writer_count=1,
    ):
        self.root = Path(root).resolve()
        self.recipe = recipe
        self.mode = mode
        self.validation_fraction = float(validation_fraction)
        self.seed = int(seed)
        self.min_loss_tokens = int(min_loss_tokens)
        self.max_pending_bytes = int(max_pending_bytes)
        self.writer_index, self.writer_count = int(writer_index), int(writer_count)
        if not 0 < self.validation_fraction < 1 or self.max_pending_bytes < 1:
            raise ValueError("Invalid online dump split or pending-byte limit")
        if int(writer_threads) < 1 or int(reader_threads) < 1:
            raise ValueError("online_dump.writer_threads and reader_threads must be positive")
        if not 0 <= self.writer_index < self.writer_count:
            raise ValueError("Invalid online dump writer index")
        self.writer = HiddenWriter(
            self.root,
            recipe,
            hold_lock=True,
            lock_name=(
                ".writer.lock" if self.writer_index == 0 else f".writer-{self.writer_index}.lock"
            ),
        )
        self._progress_path = self.root / (
            "progress.json" if self.writer_count == 1 else f"progress-{self.writer_index}.json"
        )
        self._space_lock = threading.Lock()
        self._free_bytes = 0
        self._space_checked = float("-inf")
        self._condition = threading.Condition()
        self._store_closed = False
        self._shared_store = store
        self._stores = []
        self._local = threading.local()
        if store is None:
            # Every reading thread gets its own client: Mooncake carries one
            # client's transfers one after another, so parallel reads need parallel
            # clients. Reads land straight in this writer's tensors, so a client
            # needs no sample-sized receive buffer.
            self._client_config = dataclasses.replace(
                mooncake_config,
                global_segment_size=0,
                async_put_pool_size=0,
                enable_gpu_direct=False,
                local_buffer_size=64 << 20,
            )
        # Connect now so an unreachable Mooncake fails at startup; releases use this client.
        self._thread_store()
        self._readers = ThreadPoolExecutor(
            max_workers=int(reader_threads), thread_name_prefix="angelspec-online-dump-read"
        )
        if self.writer_index == 0:
            existing = self.writer.committed_records(verify="header")
            self._known_ids = {record["id"] for record in existing}
        else:
            # The first writer checks the committed receipts; deduplication only
            # needs their IDs, which name the receipt files.
            existing = []
            self._known_ids = {path.stem for path in (self.root / "receipts").glob("*.json")}
        self._session_records = []
        self._jobs = queue.Queue()
        self._pending_bytes = 0
        self._error = None
        self._closed = False
        self._staged = 0
        self._written = len(existing)
        self._duplicates = 0
        self._skipped = 0
        self._read_bytes = self._written_bytes = 0
        self._read_seconds = self._write_seconds = 0.0
        self._started = time.monotonic()
        self._last_progress = 0.0
        self._last_log = self._started
        self._threads = [
            threading.Thread(
                target=self._write_loop, name=f"angelspec-online-dump-{index}", daemon=True
            )
            for index in range(int(writer_threads))
        ]
        for thread in self._threads:
            thread.start()
        self._record_session()
        self._write_progress(force=True)

    def _record_session(self):
        code_root = Path(__file__).resolve().parents[1]
        entry = {
            "started": time.time(),
            "mode": self.mode,
            "writer": self.writer_index,
            "capture_code": {name: file_hash(code_root / name) for name in CAPTURE_CODE},
        }
        with open(self.root / "sessions.jsonl", "a") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _thread_store(self):
        """This thread's Mooncake client, or the injected store shared by all threads."""
        if self._shared_store is not None:
            return self._shared_store
        store = getattr(self._local, "store", None)
        if store is None:
            from angelspec.transfer.mooncake.draft_store import DraftMooncakeStore

            store = DraftMooncakeStore(dataclasses.replace(self._client_config))
            store.setup(device=torch.device("cpu"))
            with self._condition:
                self._stores.append(store)
            self._local.store = store
        return store

    def _status_unlocked(self, *, complete=False):
        return {
            "complete": complete,
            "staged": self._staged,
            "written": self._written,
            "duplicates": self._duplicates,
            "skipped": self._skipped,
            "pending_bytes": self._pending_bytes,
            # Thread-seconds: divided by the elapsed time they give the average
            # number of reads or writes in flight.
            "read_bytes": self._read_bytes,
            "read_seconds": round(self._read_seconds, 3),
            "written_bytes": self._written_bytes,
            "write_seconds": round(self._write_seconds, 3),
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
            "error": str(self._error) if self._error is not None else None,
            "recipe_sha256": object_hash(self.recipe),
        }

    def _write_progress(self, *, complete=False, force=False):
        with self._condition:
            now = time.monotonic()
            if not force and now - self._last_progress < 2.0:
                return None
            self._last_progress = now
            value = self._status_unlocked(complete=complete)
            log = now - self._last_log >= 60.0
            if log:
                self._last_log = now
        atomic_json(self._progress_path, value)
        if log:
            elapsed = max(value["elapsed_seconds"], 1e-9)
            logger.info(
                f"Online dump: {value['written']} samples written; since start "
                f"{value['read_bytes'] / elapsed / 2**30:.2f} GiB/s read and "
                f"{value['written_bytes'] / elapsed / 2**30:.2f} GiB/s written; "
                f"{value['pending_bytes'] / 2**30:.1f} GiB waiting for a file writer"
            )
        return value

    def _raise_if_failed(self):
        with self._condition:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Online dump writer failed: {error}") from error

    def _reserve(self, size):
        with self._condition:
            while self._pending_bytes and self._pending_bytes + size > self.max_pending_bytes:
                if self._error is not None:
                    raise RuntimeError(
                        f"Online dump writer failed: {self._error}"
                    ) from self._error
                self._condition.wait(timeout=0.1)
            if self._error is not None:
                raise RuntimeError(f"Online dump writer failed: {self._error}") from self._error
            self._pending_bytes += size

    def _write_loop(self):
        while True:
            job = self._jobs.get()
            if job is self._STOP:
                self._jobs.task_done()
                return
            sample_id, tensors, split, provenance, sample_metadata, size = job
            try:
                started = time.monotonic()
                record = self.writer.write(
                    sample_id,
                    tensors,
                    split=split,
                    provenance=provenance,
                    sample_metadata=sample_metadata,
                )
                with self._condition:
                    self._session_records.append(record)
                    self._written += 1
                    self._written_bytes += size
                    self._write_seconds += time.monotonic() - started
            except BaseException as exc:  # noqa: B036 - the owner re-raises it
                with self._condition:
                    if self._error is None:
                        self._error = exc
            finally:
                with self._condition:
                    self._pending_bytes -= size
                    self._condition.notify_all()
                self._jobs.task_done()
                self._write_progress()

    def _release(self, result):
        shapes = result.get("tensor_shapes") or {}
        self._thread_store().remove_tensors(
            result["mooncake_key"],
            has_last_hidden_states="last_hidden_states" in shapes,
            has_target="target" in shapes,
            has_hidden_states="hidden_states" in shapes,
        )

    def _read_input_ids(self, result):
        input_ids = torch.empty(result["tensor_shapes"]["input_ids"], dtype=torch.int64)
        self._thread_store().get_into(result["mooncake_key"], {"input_ids": input_ids})
        return input_ids.reshape(-1)

    def _identify(self, result):
        """Check one result and read its token IDs; None if it has too few loss tokens."""
        packed = result.get("packed_loss_mask")
        if packed is None:
            raise ValueError("Online dump requires packed_loss_mask for every sample")
        seq_len = int(result["tensor_shapes"]["input_ids"][-1])
        loss_mask = unpack_loss_mask(packed).reshape(-1).to(torch.uint8)
        if loss_mask.shape != (seq_len,):
            raise ValueError("Online dump loss mask length differs from input_ids")
        if int(loss_mask.sum()) < self.min_loss_tokens:
            with self._condition:
                self._skipped += 1
            return None
        raw_metadata = result.get("metadata") or {}
        if raw_metadata.get("multimodal_pruning"):
            raise ValueError("Online VL dump does not support multimodal pruning")
        sample_metadata = {}
        slots = raw_metadata.get(IMAGE_NEWLINE_SLOTS_KEY)
        if slots is not None:
            if not self.recipe.get("multimodal", False):
                raise ValueError(
                    "Received VL position metadata but online multimodal capture is disabled"
                )
            if not isinstance(slots, (list, tuple)) or any(
                type(slot) is not int or slot < 0 for slot in slots
            ):
                raise ValueError("Invalid image_newline_slots metadata")
            sample_metadata[IMAGE_NEWLINE_SLOTS_KEY] = list(slots)
        media = raw_metadata.get(MULTIMODAL_SHA256_KEY)
        if self.recipe.get("multimodal", False):
            if MULTIMODAL_SHA256_KEY not in raw_metadata:
                raise ValueError(
                    "Online VL dump requires the engine's media digest for every sample"
                )
        elif media is not None:
            raise ValueError("Received media but online multimodal capture is disabled")
        if media is not None and (not isinstance(media, str) or not SHA256_HEX.fullmatch(media)):
            raise ValueError("Invalid media digest")
        input_ids = self._read_input_ids(result)
        input_hash = tensor_hash(input_ids.reshape(1, -1))
        sample_identity = {"input": input_hash, "mask": tensor_hash(loss_mask.reshape(1, -1))}
        if sample_metadata:
            sample_identity["sample_metadata"] = sample_metadata
        if media is not None:
            sample_identity["media"] = media
        return {
            "id": object_hash(sample_identity),
            "result": result,
            "tokens": {"input_ids": input_ids, "loss_mask": loss_mask},
            "input_hash": input_hash,
            "sample_metadata": sample_metadata or None,
            "media": media,
        }

    def _require_space(self, size):
        """Fail before copying a sample the dump file system cannot hold.

        Free space is read at most once a second. statvfs on a busy FUSE mount
        can briefly report almost nothing free, so a low reading has to persist
        for a few seconds before it fails the run.
        """
        needed = size * 1.1
        with self._space_lock:
            fresh = time.monotonic() - self._space_checked < _SPACE_CHECK_SECONDS
            if fresh and self._free_bytes >= needed:
                self._free_bytes -= size
                return
            for attempt in range(_SPACE_CONFIRMATIONS):
                if attempt:
                    time.sleep(1.0)
                free = shutil.disk_usage(self.root).free
                if free >= needed:
                    self._free_bytes, self._space_checked = free - size, time.monotonic()
                    return
        raise RuntimeError(
            f"Insufficient online dump space; need approximately {size} new bytes, "
            f"the file system reports {free} free"
        )

    def _copy(self, sample, capture_epoch, placement):
        """Read one claimed sample's layers into owned memory and queue its file."""
        result = sample["result"]
        size = _capture_bytes(self.recipe, sample["tokens"]["input_ids"].numel())
        self._require_space(size)
        self._reserve(size)
        try:
            started = time.monotonic()
            tensors = materialize_native_capture(
                result, sample["tokens"], self._thread_store(), self.recipe, allow_extra=True
            )
            elapsed = time.monotonic() - started
        except BaseException:
            with self._condition:
                self._pending_bytes -= size
                self._condition.notify_all()
            raise
        split_key = object_hash({"input": sample["input_hash"], "seed": self.seed})
        split = (
            "validation" if int(split_key[:16], 16) / 2**64 < self.validation_fraction else "train"
        )
        provenance = {
            "data_id": str(result["data_id"]),
            "capture_epoch": int(capture_epoch),
        }
        if sample["media"] is not None:
            provenance[MULTIMODAL_SHA256_KEY] = sample["media"]
        if placement is not None:
            provenance["placement"] = dict(placement)
        self._jobs.put((sample["id"], tensors, split, provenance, sample["sample_metadata"], size))
        with self._condition:
            self._staged += 1
            self._read_bytes += size
            self._read_seconds += elapsed

    def _in_parallel(self, function, *columns):
        """Call ``function`` row by row on the reader threads; raise the first failure."""
        futures = [self._readers.submit(function, *row) for row in zip(*columns)]
        wait(futures)
        return [future.result() for future in futures]

    def stage_batch(self, results, capture_epoch=0, placements=None):
        if self._closed or self._store_closed:
            raise RuntimeError("Online dump writer is closed")
        self._raise_if_failed()
        if placements is not None and len(placements) != len(results):
            raise ValueError("Online dump placements must align with the staged results")
        if int(capture_epoch) >= int(self.recipe["capture_epochs"]):
            return {"staged": 0, "skipped_epoch": len(results)}
        if placements is None:
            placements = [None] * len(results)
        try:
            samples = self._in_parallel(self._identify, [_result_dict(raw) for raw in results])
            claimed = []
            with self._condition:
                # In batch order, so the first of several equal samples is the one kept.
                for sample, placement in zip(samples, placements):
                    if sample is None:
                        continue
                    if sample["id"] in self._known_ids:
                        self._duplicates += 1
                        continue
                    self._known_ids.add(sample["id"])
                    claimed.append((sample, placement))
            self._in_parallel(
                self._copy,
                [sample for sample, _ in claimed],
                [capture_epoch] * len(claimed),
                [placement for _, placement in claimed],
            )
        except BaseException as exc:
            # The batch may already be training, so a partial copy must never publish.
            with self._condition:
                if self._error is None:
                    self._error = exc
            raise
        self._raise_if_failed()
        return {"staged": len(claimed), "skipped_epoch": 0}

    def release(self, results):
        """Free retained samples' Mooncake keys.

        The caller must submit this after the samples' ``stage_batch`` call and
        only once every trainer has read them.
        """
        if self._store_closed:
            return 0
        for raw in results:
            self._release(_result_dict(raw))
        return len(results)

    def status(self):
        self._raise_if_failed()
        return self._write_progress(force=True)

    def _close_store(self):
        if self._store_closed:
            return
        self._store_closed = True
        self._readers.shutdown(wait=True)
        stores = [self._shared_store] if self._shared_store is not None else self._stores
        closable = [store for store in stores if hasattr(store, "close")]
        with ThreadPoolExecutor(max_workers=max(1, len(closable))) as pool:
            list(pool.map(lambda store: store.close(), closable))

    def flush_staging(self):
        """Close the Mooncake clients; earlier stage_batch calls have all returned."""
        self._raise_if_failed()
        self._close_store()
        return self._write_progress(force=True)

    def _drain_and_close(self):
        if self._closed:
            return
        self._jobs.join()
        for _ in self._threads:
            self._jobs.put(self._STOP)
        self._jobs.join()
        for thread in self._threads:
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError("Online dump writer thread did not stop")
        self._close_store()
        self._closed = True
        self._raise_if_failed()

    def finalize(self, verify="header", publish=True, records=()):
        """Drain pending writes and publish; ``verify="full"`` re-reads every file.

        Of several writers sharing a dump only the first publishes, given the
        ``records`` the others returned when they finalized with ``publish=False``.
        Receipts written in this session are taken from memory rather than reread.
        """
        self._drain_and_close()
        published = (
            self.writer.publish_committed(
                verify=verify, records=[*self._session_records, *records]
            )
            if publish
            else 0
        )
        self.writer.close()
        result = self._write_progress(complete=publish, force=True)
        result.update(published=published, verify=verify or "none")
        if not publish:
            result["records"] = list(self._session_records)
        return result

    def abort(self):
        if not self._closed:
            self._jobs.join()
            for _ in self._threads:
                self._jobs.put(self._STOP)
            self._jobs.join()
            for thread in self._threads:
                thread.join(timeout=30)
            self._close_store()
            self._closed = True
        self.writer.close()
        return self._write_progress(force=True)


def online_dump_status(root, *, verify=False):
    root = Path(root).resolve()
    if (root / "manifest.json").exists():
        dataset = HiddenDataset(root, format="native", verify_hashes=verify)
        result = {
            "complete": True,
            "samples": len(dataset.records),
            "train": sum(record["split"] == "train" for record in dataset.records),
            "validation": sum(record["split"] == "validation" for record in dataset.records),
            "layer_ids": dataset.layer_ids,
        }
        if verify:
            result["verified"] = dataset.verify()
        return result
    recipe = read_json(root / "recipe.json")
    writer = HiddenWriter(root, recipe, read_only=True)
    records = writer.committed_records(verify="full" if verify else None)
    progress_files = sorted(root.glob("progress*.json"))
    progress = (
        merge_dump_status([read_json(path) for path in progress_files]) if progress_files else {}
    )
    return {
        **progress,
        "complete": False,
        "samples": len(records),
        "train": sum(record["split"] == "train" for record in records),
        "validation": sum(record["split"] == "validation" for record in records),
        "layer_ids": recipe.get("layer_ids"),
    }


def finalize_online_dump(root, *, verify="full"):
    root = Path(root).resolve()
    if (root / "manifest.json").exists():
        return online_dump_status(root, verify=True)
    receipt_ids = {path.stem for path in (root / "receipts").glob("*.json")}
    dump_ids = {path.stem for path in (root / "dumps").glob("*.safetensors")}
    if dump_ids != receipt_ids:
        raise ValueError(
            "Online dump contains incomplete tensor/receipt pairs; quarantine them before finalize"
        )
    writer = HiddenWriter(root, read_json(root / "recipe.json"))
    published = writer.publish_committed(verify=verify)
    return {
        **online_dump_status(root),
        "verify": verify,
        "verified": published if verify == "full" else None,
    }
