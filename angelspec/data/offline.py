import fcntl
import hashlib
import json
import math
import os
import random
import re
import stat
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load as load_bytes
from safetensors.torch import load_file

FORMAT = "angelspec_hidden_v2"
LEGACY_FORMAT = "angelspec_hidden_v1"
LEGACY_KEYS = {"input_ids", "loss_mask", "attention_mask", "hidden_states", "last_hidden_states"}
# Residual stream after decoder layer i runs (0-based), before the model's final norm.
LAYER_SEMANTICS = "decoder_layer_output"
SAMPLE_ID = re.compile(r"[A-Za-z0-9_-]+")
VERIFY_MODES = ("full", "header", None)
STORAGE_DTYPES = ("bfloat16", "float16", "float32")


@contextmanager
def writer_lock(root):
    with open(Path(root) / ".writer.lock", "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_prefix(tensor):
    return object_hash({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}).encode()


def tensor_digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(_digest_prefix(tensor))
    digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()


_SAFETENSORS_DTYPES = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
}
_WRITE_CHUNK_BYTES = 64 * 1024 * 1024


def write_safetensors(stream, tensors, metadata):
    """Write contiguous CPU ``tensors`` to ``stream`` as one safetensors file.

    The bytes go out straight from the tensors' memory, so no file-sized copy is
    built and other threads keep running while this one hashes and writes.
    Tensors are laid out widest element first, then by name, so each starts
    aligned to its element size; metadata keys are sorted, so equal content gives
    equal files. Returns ``(size, file sha256, {name: tensor_digest})``.
    """
    names = sorted(tensors, key=lambda name: (-tensors[name].element_size(), name))
    header = {"__metadata__": dict(sorted(metadata.items()))}
    offset = 0
    for name in names:
        tensor = tensors[name]
        size = tensor.numel() * tensor.element_size()
        header[name] = {
            "dtype": _SAFETENSORS_DTYPES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    prefix = len(encoded).to_bytes(8, "little") + encoded
    file_digest = hashlib.sha256(prefix)
    stream.write(prefix)
    digests = {}
    for name in names:
        tensor = tensors[name]
        digest = hashlib.sha256(_digest_prefix(tensor))
        data = memoryview(tensor.view(torch.uint8).numpy()).cast("B")
        for start in range(0, len(data), _WRITE_CHUNK_BYTES):
            chunk = data[start : start + _WRITE_CHUNK_BYTES]
            file_digest.update(chunk)
            digest.update(chunk)
            stream.write(chunk)
        digests[name] = digest.hexdigest()
    return len(prefix) + offset, file_digest.hexdigest(), digests


def read_json(path):
    return json.loads(Path(path).read_text())


def child_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Artifact path escapes its root: {relative}")
    return path


def _discard(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def atomic_json(path, value):
    path = Path(path)
    try:
        fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        _discard(temporary)
        raise


def layer_tensor_name(layer_id):
    return f"hidden_layer_{int(layer_id)}"


def layer_recipe(layer_ids, *, target_num_hidden_layers, hidden_size):
    """Recipe fields naming every stored layer and what its tensor holds."""
    layers = sorted(int(layer) for layer in layer_ids)
    if (
        not layers
        or len(set(layers)) != len(layers)
        or layers[0] < 0
        or layers[-1] >= int(target_num_hidden_layers)
    ):
        raise ValueError(
            f"Invalid stored layer IDs {list(layer_ids)} for {target_num_hidden_layers} layers"
        )
    return {
        "format": FORMAT,
        "layer_ids": layers,
        "layer_tensors": {str(layer): layer_tensor_name(layer) for layer in layers},
        "layer_semantics": LAYER_SEMANTICS,
        "layer_index_base": 0,
        "final_norm_applied": False,
        "target_num_hidden_layers": int(target_num_hidden_layers),
        "final_layer_id": int(target_num_hidden_layers) - 1,
        "hidden_size": int(hidden_size),
    }


def validate_layer_recipe(recipe):
    if recipe.get("format") != FORMAT:
        raise ValueError(f"Native writer and reader need recipe format {FORMAT}")
    if recipe.get("storage_dtype", "bfloat16") not in STORAGE_DTYPES:
        raise ValueError(f"Unsupported storage_dtype {recipe.get('storage_dtype')!r}")
    expected = layer_recipe(
        recipe.get("layer_ids") or [],
        target_num_hidden_layers=recipe.get("target_num_hidden_layers", 0),
        hidden_size=recipe.get("hidden_size", 0),
    )
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("Recipe layer description is inconsistent")


def file_metadata(sample_id, recipe, recipe_hash):
    return {
        "format": FORMAT,
        "id": sample_id,
        "recipe_sha256": recipe_hash,
        "layer_ids": ",".join(str(layer) for layer in recipe["layer_ids"]),
        "layer_semantics": recipe["layer_semantics"],
        "final_norm_applied": "false",
        "target_num_hidden_layers": str(recipe["target_num_hidden_layers"]),
        "hidden_size": str(recipe["hidden_size"]),
        "storage_dtype": recipe.get("storage_dtype", "bfloat16"),
    }


def _header_metadata(payload):
    size = int.from_bytes(payload[:8], "little")
    return json.loads(payload[8 : 8 + size]).get("__metadata__", {})


_FINITE_CHECK_ROWS = 16384
# Inf and NaN are exactly the floats whose exponent bits are all set.
_EXPONENT_BITS = {
    torch.bfloat16: (torch.int16, 0x7F80),
    torch.float16: (torch.int16, 0x7C00),
    torch.float32: (torch.int32, 0x7F800000),
}


def _all_finite(tensor):
    """``torch.isfinite(tensor).all()`` in one chunk-sized buffer.

    A fresh temporary per chunk leaves concurrent writer threads mostly
    faulting in new pages, so float storage reuses one buffer for its
    exponent bits.
    """
    if tensor.numel() == 0:
        return True
    if tensor.dtype not in _EXPONENT_BITS:
        return all(
            bool(torch.isfinite(tensor[start : start + _FINITE_CHECK_ROWS]).all())
            for start in range(0, tensor.shape[0], _FINITE_CHECK_ROWS)
        )
    bits_dtype, mask = _EXPONENT_BITS[tensor.dtype]
    bits = tensor.view(bits_dtype)
    buffer = torch.empty(
        (min(_FINITE_CHECK_ROWS, bits.shape[0]), *bits.shape[1:]), dtype=bits_dtype
    )
    for start in range(0, bits.shape[0], _FINITE_CHECK_ROWS):
        chunk = bits[start : start + _FINITE_CHECK_ROWS]
        exponent = buffer[: chunk.shape[0]]
        torch.bitwise_and(chunk, mask, out=exponent)
        if int(exponent.max()) == mask:
            return False
    return True


def validate_sample(tensors, seq_len, recipe, layer_ids=None, check_finite=True):
    """Check native tensors: [L] tokens/mask plus one [L, H] tensor per layer."""
    layers = recipe["layer_ids"] if layer_ids is None else list(layer_ids)
    storage_dtype = recipe.get("storage_dtype", "bfloat16")
    expected = {"input_ids", "loss_mask", *(layer_tensor_name(layer) for layer in layers)}
    if set(tensors) != expected:
        raise ValueError(f"Expected tensor keys {sorted(expected)}, got {sorted(tensors)}")
    ids, mask = tensors["input_ids"], tensors["loss_mask"]
    if tuple(ids.shape) != (seq_len,) or tuple(mask.shape) != (seq_len,):
        raise ValueError("input_ids and loss_mask must have shape [seq_len]")
    if ids.dtype != torch.int64 or mask.dtype != torch.uint8:
        raise ValueError("input_ids must be int64 and loss_mask uint8")
    vocab_size = recipe.get("vocab_size")
    if bool((ids < 0).any()) or (vocab_size is not None and bool((ids >= vocab_size).any())):
        raise ValueError("input_ids outside target vocabulary")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("loss_mask must be binary")
    for layer in layers:
        tensor = tensors[layer_tensor_name(layer)]
        if tuple(tensor.shape) != (seq_len, recipe["hidden_size"]):
            raise ValueError(f"Invalid layer {layer} shape: {tuple(tensor.shape)}")
        if tensor.dtype != getattr(torch, storage_dtype):
            raise ValueError(f"Layer {layer} must be stored as {storage_dtype}")
        if check_finite and not _all_finite(tensor):
            raise ValueError(f"Non-finite layer {layer}")


def verify_native_file(root, record, recipe, recipe_hash):
    """Read one native file once and check bytes, header, structure and tensor digests."""
    payload = child_path(root, record["file"]).read_bytes()
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {record['id']}")
    if _header_metadata(payload) != file_metadata(record["id"], recipe, recipe_hash):
        raise ValueError(f"Tensor metadata mismatch: {record['id']}")
    tensors = load_bytes(payload)
    validate_sample(tensors, record["seq_len"], recipe)
    if {name: tensor_digest(tensor) for name, tensor in tensors.items()} != record[
        "tensor_sha256"
    ]:
        raise ValueError(f"Tensor checksum mismatch: {record['id']}")
    return tensors


def validate_tensors(tensors, seq_len, hidden_size, context_layers, vocab_size=None):
    """Validate the legacy concatenated layout (v1 native and DFly imports)."""
    expected_keys = LEGACY_KEYS if context_layers else LEGACY_KEYS - {"hidden_states"}
    if set(tensors) != expected_keys:
        raise ValueError(f"Expected tensor keys {sorted(expected_keys)}, got {sorted(tensors)}")
    for key in ("input_ids", "loss_mask", "attention_mask"):
        tensor = tensors[key]
        if tuple(tensor.shape) != (1, seq_len):
            raise ValueError(f"Invalid {key} shape: {tuple(tensor.shape)}")
    if tensors["input_ids"].dtype != torch.int64:
        raise ValueError("input_ids must be int64")
    ids = tensors["input_ids"]
    if bool((ids < 0).any()) or (vocab_size is not None and bool((ids >= vocab_size).any())):
        raise ValueError("input_ids outside target vocabulary")
    for key in ("loss_mask", "attention_mask"):
        if not bool(((tensors[key] == 0) | (tensors[key] == 1)).all()):
            raise ValueError(f"{key} must be binary")
    if not bool(tensors["attention_mask"].all()):
        raise ValueError("Persist unpadded single samples, not padded batches")
    for key, width in (
        ("hidden_states", hidden_size * context_layers),
        ("last_hidden_states", hidden_size),
    ):
        if key == "hidden_states" and not context_layers:
            continue
        tensor = tensors[key]
        if tuple(tensor.shape) != (1, seq_len, width):
            raise ValueError(f"Invalid {key} shape: {tuple(tensor.shape)}")
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Invalid {key} dtype")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite {key}")


class HiddenDataset:
    """Read-only view of a published dump as per-layer teacher features.

    Native v2 files name every layer. Legacy v1 and DFly imports keep the final
    layer unnamed in ``last_hidden_states``; bind the target's final layer ID
    before loading them.

    ``HiddenWriter`` rejects non-finite layers when it writes a native v2 file, so
    loading one re-scans the layers for non-finite values only together with the
    tensor digests (``verify_hashes``); legacy and imported dumps are always scanned.
    """

    def __init__(
        self, root, *, verify_hashes=True, format="auto", tokenized_path=None, final_layer_id=None
    ):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        self.verify_hashes = verify_hashes
        self.recipe = self.manifest["recipe"]
        self.recipe_hash = object_hash(self.recipe)
        if not self.manifest.get("complete") or self.manifest["recipe_sha256"] != self.recipe_hash:
            raise ValueError("Incomplete manifest or recipe hash mismatch")
        manifest_format = self.manifest.get("format")
        if manifest_format in (FORMAT, LEGACY_FORMAT):
            detected = "native"
        elif manifest_format is None:
            detected = "dfly"
        else:
            raise ValueError(f"Unsupported artifact format: {manifest_format}")
        if format not in ("auto", detected):
            raise ValueError(f"Requested {format}, found {detected}")
        self.format = detected
        self.legacy = manifest_format != FORMAT
        self.records = self.manifest["records"]
        ids = [record["id"] for record in self.records]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Empty dataset or duplicate sample IDs")
        for record in self.records:
            if not SAMPLE_ID.fullmatch(record["id"]):
                raise ValueError("Unsafe sample ID")
            child_path(self.root, record["file"])
            if record["split"] not in ("train", "validation") or record["seq_len"] < 1:
                raise ValueError("Invalid sample split or length")
            sample_metadata = record.get("sample_metadata")
            if sample_metadata is not None and not isinstance(sample_metadata, dict):
                raise ValueError("Offline sample_metadata must be a mapping")
        if detected == "dfly":
            if tokenized_path is None:
                raise ValueError("DFly import requires offline.tokenized_path")
            self._validate_dfly(Path(tokenized_path))
        self.layer_ids = self.final_layer_id = None
        self.hidden_size = self.recipe.get("hidden_size")
        if self.legacy:
            if self.recipe.get("last_hidden_states_pre_final_norm") is not True:
                raise ValueError("Only pre-final-norm teacher hidden is supported")
        else:
            validate_layer_recipe(self.recipe)
            self.layer_ids = list(self.recipe["layer_ids"])
            self.final_layer_id = self.recipe["final_layer_id"]
            for record in self.records:
                if record.get("layer_ids") != self.layer_ids or set(
                    record.get("tensor_sha256") or ()
                ) != {
                    "input_ids",
                    "loss_mask",
                    *(layer_tensor_name(layer) for layer in self.layer_ids),
                }:
                    raise ValueError(
                        f"Record layer description differs from recipe: {record['id']}"
                    )
        if final_layer_id is not None:
            self.bind_final_layer(final_layer_id)
        self.identity = object_hash(self.manifest)

    def bind_final_layer(self, final_layer_id):
        final_layer_id = int(final_layer_id)
        if not self.legacy:
            if final_layer_id != self.final_layer_id:
                raise ValueError(
                    f"Dump final layer {self.final_layer_id} differs from target {final_layer_id}"
                )
            return
        captured = (
            self.recipe["capture_post_layer_ids"]
            if self.format == "dfly"
            else self.recipe["target_layer_ids"]
        )
        aux = [layer for layer in captured if layer != final_layer_id]
        if self.format == "native" and aux != self.recipe["hidden_layer_ids"]:
            raise ValueError("Legacy recipe disagrees with the target's final layer")
        if final_layer_id in aux or any(layer > final_layer_id for layer in aux):
            raise ValueError("Legacy capture layers exceed the target's final layer")
        self._legacy_aux = aux
        self.final_layer_id = final_layer_id
        self.layer_ids = sorted(aux + [final_layer_id])

    def _validate_dfly(self, pilot_root):
        pilot = read_json(pilot_root / "manifest.json")
        if not pilot.get("complete") or object_hash(pilot["recipe"]) != pilot["recipe_sha256"]:
            raise ValueError("Incomplete or invalid DFly tokenized manifest")
        if self.recipe.get("pilot_identity") == "recipe-v1":
            matches = pilot["recipe_sha256"] == self.recipe["pilot_recipe_sha256"]
        else:
            matches = (
                file_hash(pilot_root / "manifest.json") == self.recipe["pilot_manifest_sha256"]
            )
        if not matches:
            raise ValueError("DFly tokenized/dump identity mismatch")
        validation = read_json(self.root / "validation.json")
        if not validation.get("complete") or validation.get("status") != "passed":
            raise ValueError("DFly dump must pass independent verification")
        if validation["recipe_sha256"] != self.recipe_hash:
            raise ValueError("DFly verification recipe mismatch")
        ids = {record["id"] for record in self.records}
        if ids != {r["id"] for r in pilot["records"]} or ids != {
            r["id"] for r in validation["records"]
        }:
            raise ValueError("DFly sample sets differ")
        pilots = {r["id"]: r for r in pilot["records"]}
        self.dfly_validation = {r["id"]: r for r in validation["records"]}
        for record in self.records:
            other = pilots[record["id"]]
            if (record["seq_len"], record["split"]) != (other["seq_len"], other["split"]):
                raise ValueError("DFly split/length mismatch")
        if self.recipe.get("last_hidden_states_pre_final_norm") is not True:
            raise ValueError("DFly teacher hidden must be pre-final-norm")

    def _check_receipt(self, record):
        path = child_path(self.root, record["file"])
        receipt = (
            read_json(self.root / "receipts" / f"{record['id']}.json")
            if self.format == "dfly"
            else record
        )
        if receipt["id"] != record["id"] or receipt["recipe_sha256"] != self.recipe_hash:
            raise ValueError("Receipt identity mismatch")
        if receipt.get("file") != record["file"] or receipt["seq_len"] != record["seq_len"]:
            raise ValueError("Receipt path/length mismatch")
        if path.stat().st_size != receipt["bytes"]:
            raise ValueError("Artifact size mismatch")
        if self.format == "dfly":
            verified = self.dfly_validation[record["id"]]
            if verified["sha256"] != receipt["sha256"] or verified["bytes"] != receipt["bytes"]:
                raise ValueError("Receipt differs from independently verified artifact")
        return path, receipt

    def _load_legacy(self, record):
        path, receipt = self._check_receipt(record)
        if self.verify_hashes and file_hash(path) != receipt["sha256"]:
            raise ValueError("Artifact checksum mismatch")
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            metadata = reader.metadata() or {}
        if self.format == "dfly":
            metadata = json.loads(metadata["offline"])
            if (
                metadata.get("seq_len") != record["seq_len"]
                or metadata.get("split") != record["split"]
                or metadata.get("target_layers") != self.recipe["capture_post_layer_ids"]
                or metadata.get("last_hidden_states_pre_final_norm") is not True
            ):
                raise ValueError("DFly tensor feature contract mismatch")
        if metadata.get("id") != record["id"] or metadata.get("recipe_sha256") != self.recipe_hash:
            raise ValueError("Tensor metadata mismatch")
        return load_file(str(path), device="cpu")

    def load(self, record, layer_ids=None):
        """Return input_ids/loss_mask/attention_mask [1, L] and layers {id: [1, L, H]}."""
        if self.layer_ids is None:
            raise ValueError("Bind the target's final layer before loading a legacy dump")
        layers = list(self.layer_ids if layer_ids is None else layer_ids)
        missing = sorted(set(layers) - set(self.layer_ids))
        if missing:
            raise ValueError(f"Dump does not store layers {missing}; it stores {self.layer_ids}")
        length = record["seq_len"]
        if self.legacy:
            tensors = self._load_legacy(record)
            validate_tensors(
                tensors,
                length,
                self.hidden_size,
                len(self._legacy_aux),
                self.recipe.get("vocab_size"),
            )
            width = self.hidden_size
            stored = {self.final_layer_id: tensors["last_hidden_states"]}
            for position, layer in enumerate(self._legacy_aux):
                stored[layer] = tensors["hidden_states"][
                    ..., position * width : (position + 1) * width
                ]
            return {
                "input_ids": tensors["input_ids"],
                "loss_mask": tensors["loss_mask"],
                "attention_mask": tensors["attention_mask"],
                "layers": {layer: stored[layer] for layer in layers},
            }
        path, _ = self._check_receipt(record)
        names = ["input_ids", "loss_mask", *(layer_tensor_name(layer) for layer in layers)]
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            if (reader.metadata() or {}) != file_metadata(
                record["id"], self.recipe, self.recipe_hash
            ):
                raise ValueError("Tensor metadata mismatch")
            tensors = {name: reader.get_tensor(name) for name in names}
        if self.verify_hashes:
            for name, tensor in tensors.items():
                if tensor_digest(tensor) != record["tensor_sha256"][name]:
                    raise ValueError(f"Tensor checksum mismatch: {record['id']}/{name}")
        validate_sample(tensors, length, self.recipe, layers, check_finite=self.verify_hashes)
        return {
            "input_ids": tensors["input_ids"].reshape(1, length),
            "loss_mask": tensors["loss_mask"].reshape(1, length),
            "attention_mask": torch.ones(1, length, dtype=torch.bool),
            "layers": {
                layer: tensors[layer_tensor_name(layer)].reshape(1, length, -1) for layer in layers
            },
        }

    def verify(self, hidden_size=None, context_layers=None, vocab_size=None):
        if not self.legacy:
            for record in self.records:
                verify_native_file(self.root, record, self.recipe, self.recipe_hash)
            return len(self.records)
        for record in self.records:
            validate_tensors(
                self._load_legacy(record),
                record["seq_len"],
                hidden_size,
                context_layers,
                vocab_size,
            )
        return len(self.records)


# Receipt checks mostly wait on file-system metadata; a full check also holds a
# whole sample file in memory per thread.
_CHECK_THREADS = {None: 32, "header": 32, "full": 8}


def _in_threads(function, items, threads):
    """``[function(item) for item in items]`` on up to ``threads`` threads.

    Raises the failure of the earliest failing item, as a sequential loop would.
    """
    if threads <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(threads, len(items))) as pool:
        futures = [pool.submit(function, item) for item in items]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


class HiddenWriter:
    """Writes native v2 samples; hashes are computed from memory, never by re-reading.

    ``hold_lock=True`` keeps the single-writer lock for the writer's lifetime so
    several threads of the owning process can publish distinct samples at once.
    Writers of one capture that never write the same sample may each hold their
    own ``lock_name``. ``read_only=True`` only inspects receipts and never takes
    the lock.
    """

    def __init__(
        self, root, recipe, *, hold_lock=False, read_only=False, lock_name=".writer.lock"
    ):
        self.root = Path(root).resolve()
        validate_layer_recipe(recipe)
        if not recipe.get("target_metadata_sha256"):
            raise ValueError("Writer requires a frozen target identity")
        self.recipe = recipe
        self.recipe_hash = object_hash(recipe)
        self.read_only = read_only
        self._lock_stream = None
        if read_only:
            if read_json(self.root / "recipe.json") != recipe:
                raise ValueError("Recipe file differs from the requested recipe")
            return
        self.root.mkdir(parents=True, exist_ok=True)
        if hold_lock:
            self._lock_stream = open(self.root / lock_name, "a")
            fcntl.flock(self._lock_stream, fcntl.LOCK_EX)
        recipe_file = self.root / "recipe.json"
        with self._locked():
            if (self.root / "manifest.json").exists():
                self.close()
                raise FileExistsError("A published dataset is immutable; use a new directory")
            if recipe_file.exists() and read_json(recipe_file) != recipe:
                self.close()
                raise ValueError("Cannot resume a different capture recipe")
            if not recipe_file.exists():
                atomic_json(recipe_file, recipe)
            for name in ("dumps", "receipts"):
                (self.root / name).mkdir(exist_ok=True)

    @contextmanager
    def _locked(self):
        if self._lock_stream is not None:
            yield
        else:
            with writer_lock(self.root):
                yield

    def close(self):
        if self._lock_stream is not None:
            fcntl.flock(self._lock_stream, fcntl.LOCK_UN)
            self._lock_stream.close()
            self._lock_stream = None

    def write(self, sample_id, tensors, *, split="train", provenance=None, sample_metadata=None):
        if self.read_only:
            raise PermissionError("Read-only writer cannot write samples")
        with self._locked():
            if (self.root / "manifest.json").exists():
                raise FileExistsError("Published datasets are immutable")
            return self._write(
                sample_id,
                tensors,
                split=split,
                provenance=provenance,
                sample_metadata=sample_metadata,
            )

    def _write(self, sample_id, tensors, *, split, provenance=None, sample_metadata=None):
        if not SAMPLE_ID.fullmatch(sample_id):
            raise ValueError("Unsafe sample ID")
        if split not in ("train", "validation"):
            raise ValueError("Unknown split")
        if provenance is not None:
            json.dumps(provenance, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if sample_metadata is not None:
            if not isinstance(sample_metadata, dict):
                raise ValueError("sample_metadata must be a mapping")
            json.dumps(sample_metadata, sort_keys=True, ensure_ascii=False, allow_nan=False)
        tensors = {key: value.detach().cpu().contiguous() for key, value in tensors.items()}
        length = tensors["input_ids"].shape[0]
        validate_sample(tensors, length, self.recipe)
        relative = f"dumps/{sample_id}.safetensors"
        path = self.root / relative
        receipt_path = self.root / "receipts" / f"{sample_id}.json"
        if path.exists() or receipt_path.exists():
            if not path.exists() or not receipt_path.exists():
                raise ValueError("Incomplete existing sample; quarantine it before retrying")
            record = read_json(receipt_path)
            if (
                record["recipe_sha256"] != self.recipe_hash
                or record["split"] != split
                or record["id"] != sample_id
                or record["file"] != relative
                or record["seq_len"] != length
                or record["bytes"] != path.stat().st_size
                or record.get("sample_metadata") != sample_metadata
            ):
                raise ValueError("Existing sample identity differs")
            digests = {name: tensor_digest(tensor) for name, tensor in tensors.items()}
            if record.get("tensor_sha256") != digests:
                raise ValueError("Same sample ID has different tensors")
            return record
        fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                size, sha256, digests = write_safetensors(
                    stream, tensors, file_metadata(sample_id, self.recipe, self.recipe_hash)
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            _discard(temporary)
            raise
        record = {
            "id": sample_id,
            "split": split,
            "seq_len": length,
            "file": relative,
            "recipe_sha256": self.recipe_hash,
            "bytes": size,
            "sha256": sha256,
            "layer_ids": list(self.recipe["layer_ids"]),
            "tensor_sha256": digests,
        }
        if provenance is not None:
            record["provenance"] = provenance
        if sample_metadata is not None:
            record["sample_metadata"] = sample_metadata
        atomic_json(receipt_path, record)
        return record

    def _check_record(self, record, verify):
        if verify not in VERIFY_MODES:
            raise ValueError(f"Unknown verification mode {verify!r}")
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not SAMPLE_ID.fullmatch(sample_id):
            raise ValueError(f"Invalid receipt identity: {sample_id}")
        if record.get("recipe_sha256") != self.recipe_hash:
            raise ValueError(f"Receipt recipe mismatch: {sample_id}")
        if record.get("split") not in ("train", "validation"):
            raise ValueError(f"Invalid receipt split: {sample_id}")
        expected_names = {
            "input_ids",
            "loss_mask",
            *(layer_tensor_name(layer) for layer in self.recipe["layer_ids"]),
        }
        if (
            record.get("layer_ids") != self.recipe["layer_ids"]
            or set(record.get("tensor_sha256") or ()) != expected_names
        ):
            raise ValueError(f"Receipt layer description mismatch: {sample_id}")
        sample_metadata = record.get("sample_metadata")
        if sample_metadata is not None and not isinstance(sample_metadata, dict):
            raise ValueError(f"Invalid receipt sample metadata: {sample_id}")
        path = child_path(self.root, record.get("file", ""))
        try:
            info = path.stat()
        except OSError:
            info = None
        if info is None or not stat.S_ISREG(info.st_mode) or info.st_size != record.get("bytes"):
            raise ValueError(f"Receipt artifact size mismatch: {sample_id}")
        if verify == "full":
            verify_native_file(self.root, record, self.recipe, self.recipe_hash)
        elif verify == "header":
            with safe_open(str(path), framework="pt", device="cpu") as reader:
                if (reader.metadata() or {}) != file_metadata(
                    sample_id, self.recipe, self.recipe_hash
                ):
                    raise ValueError(f"Receipt tensor metadata mismatch: {sample_id}")
                if set(reader.keys()) != expected_names:
                    raise ValueError(f"Receipt tensor names mismatch: {sample_id}")

    def _write_manifest(self, records):
        if not records:
            raise ValueError("Cannot publish an empty offline dataset")
        atomic_json(
            self.root / "manifest.json",
            {
                "format": FORMAT,
                "complete": True,
                "recipe": self.recipe,
                "recipe_sha256": self.recipe_hash,
                "records": records,
            },
        )

    def publish(self, expected_ids, *, verify="full"):
        if self.read_only:
            raise PermissionError("Read-only writer cannot publish")
        expected_ids = list(expected_ids)
        if not expected_ids or len(set(expected_ids)) != len(expected_ids):
            raise ValueError("Expected IDs must be nonempty and unique")
        if not all(SAMPLE_ID.fullmatch(sample_id) for sample_id in expected_ids):
            raise ValueError("Unsafe sample ID")

        def load(sample_id):
            record = read_json(self.root / "receipts" / f"{sample_id}.json")
            if record.get("id") != sample_id:
                raise ValueError("Receipt identity mismatch")
            self._check_record(record, verify)
            return record

        with self._locked():
            if (self.root / "manifest.json").exists():
                raise FileExistsError("Published datasets are immutable")
            self._write_manifest(_in_threads(load, expected_ids, _CHECK_THREADS.get(verify, 1)))

    def committed_records(self, *, verify="full", known=None):
        """Return resumable receipt records in deterministic sample-ID order.

        Records in ``known``, receipts this process wrote, are not reread: only
        their files' sizes are checked, unless ``verify="full"``.
        """
        receipts = self.root / "receipts"
        if not receipts.exists():
            return []
        known = {} if verify == "full" else {record["id"]: record for record in known or ()}

        def load(receipt_path):
            record = known.get(receipt_path.stem)
            if record is not None:
                self._check_record(record, None)
                return record
            record = read_json(receipt_path)
            if receipt_path.name != f"{record.get('id')}.json":
                raise ValueError(f"Invalid receipt identity: {receipt_path.name}")
            self._check_record(record, verify)
            return record

        return _in_threads(load, sorted(receipts.glob("*.json")), _CHECK_THREADS.get(verify, 1))

    def publish_committed(self, *, verify="header", records=None):
        """Publish every receipt of an interrupted or finished capture.

        ``records`` are receipts written in this session, see ``committed_records``.
        """
        if self.read_only:
            raise PermissionError("Read-only writer cannot publish")
        with self._locked():
            if (self.root / "manifest.json").exists():
                raise FileExistsError("Published datasets are immutable")
            records = self.committed_records(verify=verify, known=records)
            self._write_manifest(records)
            return len(records)


class OfflineRowPlan:
    def __init__(self, records, *, max_length, packing=False, shuffle=True, seed=0, row_plan=None):
        if not records or max_length < 1:
            raise ValueError("A row plan requires records and a positive max_length")
        self.records = records
        self.max_length, self.packing = max_length, packing
        self.shuffle, self.seed = shuffle, seed
        for record in records:
            length = record["seq_len"]
            if length < 1 or length > max_length or (not packing and length == max_length):
                raise ValueError(
                    "Offline input exceeds max_seq_length; do not crop teacher features"
                )
        self.explicit_rows = None if row_plan is None else self._resolve_row_plan(row_plan)

    def _resolve_row_plan(self, row_plan):
        if self.shuffle:
            raise ValueError(
                "An explicit row plan fixes the order; set dataset.shuffle_dataset=false"
            )
        index = {record["id"]: position for position, record in enumerate(self.records)}
        rows, seen = [], set()
        for row in row_plan:
            if not row or (not self.packing and len(row) != 1):
                raise ValueError(
                    "Row plan rows must be nonempty, with one sample each without packing"
                )
            for sample_id in row:
                if sample_id not in index:
                    raise ValueError(f"Row plan sample is not in the training split: {sample_id}")
                if sample_id in seen:
                    raise ValueError(f"Row plan repeats sample {sample_id}")
                seen.add(sample_id)
            indices = [index[sample_id] for sample_id in row]
            if sum(self.records[i]["seq_len"] for i in indices) > self.max_length:
                raise ValueError("Row plan row overflows max_seq_length")
            rows.append(indices)
        if not rows:
            raise ValueError("Row plan is empty")
        return rows

    def rows_for_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        if self.explicit_rows is not None:
            return [list(row) for row in self.explicit_rows]
        order = list(range(len(self.records)))
        if self.shuffle:
            random.Random(self.seed + epoch).shuffle(order)
        if not self.packing:
            return [[index] for index in order]
        rows, current, used = [], [], 0
        for index in order:
            length = self.records[index]["seq_len"]
            if current and used + length > self.max_length:
                rows.append(current)
                current, used = [], 0
            current.append(index)
            used += length
        if current:
            rows.append(current)
        return rows

    def row_budget(self, epochs):
        if not math.isfinite(epochs) or epochs <= 0:
            raise ValueError("num_epochs must be finite and positive")
        whole = math.floor(epochs)
        if self.explicit_rows is not None:
            return math.floor(epochs * len(self.explicit_rows))
        if not self.packing:
            return math.floor(epochs * len(self.records))
        if not self.shuffle:
            return math.floor(epochs * len(self.rows_for_epoch(0)))
        budget = sum(len(self.rows_for_epoch(epoch)) for epoch in range(whole))
        if epochs > whole:
            budget += math.floor((epochs - whole) * len(self.rows_for_epoch(whole)))
        return budget


class OfflineBatchFetcher:
    """Assembles trainer batches from the layers a draft needs.

    ``aux_layer_ids`` become the concatenated ``hidden_states`` context in that
    order; ``final_layer_id`` becomes ``last_hidden_states``, the teacher.
    Only those layers are read from each file.
    """

    def __init__(
        self,
        dataset,
        *,
        rank,
        world_size,
        seed,
        max_length,
        hidden_size,
        aux_layer_ids,
        final_layer_id,
        vocab_size,
        min_loss_tokens,
        shuffle=True,
        packing=False,
        model_family="dflash",
        position_recipe=None,
        row_plan=None,
    ):
        from angelspec.data.utils import (
            DataCollatorWithPadding,
            DFlashPackingCollator,
            MTPPackingCollator,
        )

        self.dataset = dataset
        self.records = [r for r in dataset.records if r["split"] == "train"]
        if not self.records or not 0 <= rank < world_size:
            raise ValueError("Empty training split or invalid data-parallel rank")
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.shuffle = shuffle
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.aux_layer_ids = [int(layer) for layer in aux_layer_ids]
        self.final_layer_id = int(final_layer_id)
        if self.final_layer_id in self.aux_layer_ids or len(set(self.aux_layer_ids)) != len(
            self.aux_layer_ids
        ):
            raise ValueError("Auxiliary layers must be unique and exclude the final layer")
        self.layer_ids = self.aux_layer_ids + [self.final_layer_id]
        self.vocab_size, self.min_loss_tokens = vocab_size, min_loss_tokens
        self.packing = packing
        self.model_family = model_family
        self.position_recipe = position_recipe
        self.plan = OfflineRowPlan(
            self.records,
            max_length=max_length,
            packing=packing,
            shuffle=shuffle,
            seed=seed,
            row_plan=row_plan,
        )
        packing_collator = MTPPackingCollator if model_family == "mtp" else DFlashPackingCollator
        self.collator = (
            packing_collator(max_length) if packing else DataCollatorWithPadding(usp_enabled=False)
        )
        self.position = rank
        self.epoch = 0
        self.epoch_start = 0
        self.rows = self.plan.rows_for_epoch(0)

    def seek(self, global_sample_offset):
        if global_sample_offset < 0:
            raise ValueError("Sample offset must be nonnegative")
        self.position = global_sample_offset + self.rank
        self.epoch, self.epoch_start = 0, 0
        self.rows = self.plan.rows_for_epoch(0)

    def __iter__(self):
        return self

    def __next__(self):
        while self.position >= self.epoch_start + len(self.rows):
            self.epoch_start += len(self.rows)
            self.epoch += 1
            self.rows = self.plan.rows_for_epoch(self.epoch)
        indices = self.rows[self.position - self.epoch_start]
        batch = self.load_row([self.records[index] for index in indices])
        self.position += self.world_size
        return batch

    def load_row(self, records):
        from angelspec.data.positions import prepare_sample_positions

        if not records or (not self.packing and len(records) != 1):
            raise ValueError("Invalid offline row")
        if self.packing and sum(r["seq_len"] for r in records) > self.max_length:
            raise ValueError("Packed row overflows; never let the collator discard samples")
        samples, features = [], []
        for record in records:
            sample = self.dataset.load(record, self.layer_ids)
            layers = sample.pop("layers")
            if any(tensor.shape[-1] != self.hidden_size for tensor in layers.values()):
                raise ValueError("Stored layer width differs from the target hidden size")
            if int(sample["loss_mask"].sum()) < self.min_loss_tokens:
                raise ValueError("Offline sample has too few supervised tokens")
            prepare_sample_positions(
                sample,
                self.position_recipe,
                multimodal_metadata=record.get("sample_metadata"),
            )
            samples.append(sample)
            features.append(layers)
        if not self.packing:
            sample, layers = samples[0], features[0]
            sample["last_hidden_states"] = layers[self.final_layer_id]
            if self.aux_layer_ids:
                sample["hidden_states"] = torch.cat(
                    [layers[layer] for layer in self.aux_layer_ids], dim=-1
                )
            return self.collator([sample])
        # The collator packs everything but the layer features, which go straight
        # from the files into the packed row instead of being concatenated twice.
        batch = self.collator(samples)
        batch.update(self._packed_features(features))
        return batch

    def _packed_features(self, features):
        width = self.hidden_size
        first = features[0][self.final_layer_id]
        last = first.new_empty(1, self.max_length, width)
        hidden = None
        if self.aux_layer_ids and self.model_family != "mtp":
            hidden = first.new_empty(1, self.max_length, len(self.aux_layer_ids) * width)
        offset = 0
        for layers in features:
            length = layers[self.final_layer_id].shape[1]
            last[:, offset : offset + length] = layers[self.final_layer_id]
            if hidden is not None:
                for slot, layer in enumerate(self.aux_layer_ids):
                    hidden[:, offset : offset + length, slot * width : (slot + 1) * width] = (
                        layers[layer]
                    )
            offset += length
        last[:, offset:] = 0
        if hidden is not None:
            hidden[:, offset:] = 0
        return {"last_hidden_states": last, "hidden_states": hidden}

    def iter_step_batches(self, num_rows):
        if not self.packing or num_rows < 1:
            raise ValueError("iter_step_batches requires packing and positive num_rows")
        for _ in range(num_rows):
            yield next(self)


def prefetch_offline(fetcher, depth):
    from angelspec.training.data_fetcher import PrefetchedDataFetcher

    class OfflinePrefetchedFetcher(PrefetchedDataFetcher):
        def iter_step_batches(self, num_rows):
            if not self.inner.packing or num_rows < 1:
                raise ValueError("iter_step_batches requires packing and positive num_rows")
            iterator = iter(self)
            for _ in range(num_rows):
                yield next(iterator)

    return OfflinePrefetchedFetcher(fetcher, prefetch_depth=depth, target_device=None)
