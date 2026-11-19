import hashlib
import json

import numpy as np
import torch

# Inference-result metadata: SHA-256 over the decoded media of one request, or
# None for a request without media. Set by the vLLM engine under online dump.
MULTIMODAL_SHA256_KEY = "multimodal_sha256"
MEDIA_DIGEST_VERSION = "decoded-media-sha256-v1"


def _header(digest, *fields):
    digest.update(json.dumps(fields).encode())


def _update(digest, value):
    from PIL import Image

    if isinstance(value, Image.Image):
        _header(digest, "image", value.mode, *value.size)
        digest.update(value.tobytes())
    elif isinstance(value, torch.Tensor):
        _update(digest, value.detach().cpu().contiguous().numpy())
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Cannot digest an object array")
        value = np.ascontiguousarray(value)
        _header(digest, "array", value.dtype.str, *value.shape)
        digest.update(value)
    elif isinstance(value, np.generic):
        _update(digest, value.item())
    elif isinstance(value, (bytes, bytearray)):
        _header(digest, "bytes", len(value))
        digest.update(value)
    elif isinstance(value, (list, tuple)):
        _header(digest, "sequence", len(value))
        for item in value:
            _update(digest, item)
    elif isinstance(value, dict):
        _header(digest, "mapping", len(value))
        for key in sorted(value, key=str):
            _header(digest, "key", str(key))
            _update(digest, value[key])
    elif value is None or isinstance(value, (bool, int, float, str)):
        _header(digest, "scalar", value)
    else:
        raise TypeError(f"Cannot digest media of type {type(value).__name__}")


def media_digest(multi_modal_data):
    """SHA-256 over the decoded media of one vLLM prompt, or None without media.

    Hashes the pixels and video frames handed to the engine, so the digest does
    not depend on which file, patch offset or encoding the dataset referenced.
    """
    if not multi_modal_data:
        return None
    digest = hashlib.sha256(MEDIA_DIGEST_VERSION.encode())
    for modality in sorted(multi_modal_data):
        items = multi_modal_data[modality]
        _header(digest, "modality", modality)
        _update(digest, items if isinstance(items, list) else [items])
    return digest.hexdigest()
