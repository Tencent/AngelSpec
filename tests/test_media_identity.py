import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from angelspec.data.media_identity import MULTIMODAL_SHA256_KEY, media_digest


def _image(color, size=(4, 4)):
    return Image.new("RGB", size, color)


def test_media_digest_follows_decoded_content():
    red = _image((255, 0, 0))
    blue = _image((0, 0, 255))
    assert media_digest(None) is None
    assert media_digest({}) is None
    assert media_digest({"image": red}) == media_digest({"image": [_image((255, 0, 0))]})
    edited = _image((255, 0, 0))
    edited.putpixel((0, 0), (254, 0, 0))
    assert media_digest({"image": edited}) != media_digest({"image": red})
    assert media_digest({"image": _image((255, 0, 0), (2, 8))}) != media_digest({"image": red})
    assert media_digest({"image": [red, blue]}) != media_digest({"image": [blue, red]})

    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    video = (frames, {"fps": 1.0, "frames_indices": [0, 1]})
    same = (frames.copy(), {"frames_indices": [np.int64(0), np.int64(1)], "fps": np.float64(1.0)})
    assert media_digest({"video": video}) == media_digest({"video": same})
    assert media_digest({"video": video}) != media_digest({"image": video})
    assert media_digest({"video": video}) != media_digest(
        {"video": (frames, {"fps": 2.0, "frames_indices": [0, 1]})}
    )
    shifted = frames.copy()
    shifted[1, 0, 0, 0] = 1
    assert media_digest({"video": (shifted, video[1])}) != media_digest({"video": video})
    assert media_digest({"video": torch.from_numpy(frames)}) == media_digest({"video": frames})
    with pytest.raises(TypeError, match="Cannot digest"):
        media_digest({"image": object()})


def test_online_dump_engine_tags_each_request_with_its_media_digest(tmp_path):
    pytest.importorskip("vllm")
    from angelspec.inference.engine.vllm_engine import VllmEngine

    encoded = []
    for color in ((255, 0, 0), (0, 255, 0)):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
        encoded.append(buffer.getvalue())
    blobs = [encoded[0], encoded[1], encoded[0]]
    offsets = np.cumsum([0, *map(len, blobs)]).tolist()
    patch = tmp_path / "images.patch"
    patch.write_bytes(b"".join(blobs))

    def ref(index, size=(16, 16)):
        return {
            "source_type": "bytearray",
            "bytearray": {
                "patch": str(patch),
                "start_num": offsets[index],
                "size": len(blobs[index]),
            },
            "target_size": list(size),
        }

    engine = VllmEngine.__new__(VllmEngine)
    engine.rank = 0
    engine.args = SimpleNamespace(online_dump_enabled=True)
    engine._engine = SimpleNamespace(
        generate=lambda prompts, params, use_tqdm: [
            SimpleNamespace(prompt_token_ids=[1, 2], kv_transfer_params={"mooncake_key": f"k{i}"})
            for i in range(len(prompts))
        ]
    )
    media = [
        {"images": [ref(0)]},
        {"images": [ref(1)]},
        {"images": [ref(2)]},
        {"images": [ref(0, (8, 8))]},
        None,
    ]
    ids = [torch.tensor([1, 2])] * len(media)
    results = engine._generate_impl(
        [f"d{i}" for i in range(len(media))], ids, multimodal_inputs=media
    )
    digests = [result["metadata"][MULTIMODAL_SHA256_KEY] for result in results]
    assert digests[0] == digests[2]
    assert len({digests[0], digests[1], digests[3]}) == 3
    assert digests[4] is None

    engine.args.online_dump_enabled = False
    results = engine._generate_impl(["d0"], ids[:1], multimodal_inputs=media[:1])
    assert "metadata" not in results[0]
