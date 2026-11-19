import os
import struct
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from angelspec.data.megatron_bin import (
    _load_image_sidecar,
    _pack_loss_mask_np,
    _parse_bin_data_list,
    _rewrite_image_newlines,
    _sample_conversations,
    _unpack_bin_range,
    load_megatron_bin_blend,
    load_megatron_bin_dataset,
    resolve_image_newline_rewrite,
)
from angelspec.data.utils import pack_loss_mask, unpack_loss_mask


class TestImageNewlineRewrite(unittest.TestCase):
    def test_only_newlines_inside_image_spans_become_placeholders(self):
        ids = np.array([5, 120118, 120120, 120121, 120120, 120121, 120119, 120121, 7])
        out, slots = _rewrite_image_newlines(ids, 120118, 120119, 120121, 120120)
        self.assertEqual(
            out.tolist(), [5, 120118, 120120, 120120, 120120, 120120, 120119, 120121, 7]
        )
        self.assertEqual(slots, [3, 5])

    def test_loaded_sample_keeps_positions_of_the_original_rows(self):
        import json

        from angelspec.data.positions import PositionRecipe, prepare_sample_positions

        image = [120118] + [120120, 120120, 120120, 120121] * 2 + [120119]
        conversation = [300001, *image, 2, 3]
        with tempfile.TemporaryDirectory() as root:
            prefix = os.path.join(root, "train_text_sentence")
            np.asarray(conversation + [120026], dtype=np.int32).tofile(prefix + ".bin")
            with open(prefix + ".idx", "wb") as f:
                f.write(b"MMIDIDX\x00\x00")
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<B", 4))
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<Q", 1))
                f.write(np.asarray([len(conversation) + 1], dtype=np.int32).tobytes())
            np.save(
                prefix + "_img.npy",
                {
                    "img_path": {0: [{"patch": "/tmp/image.patch", "start_num": 0, "size": 1}]},
                    "img_target_size": {0: [(32, 32)]},
                },
            )
            target = os.path.join(root, "target")
            os.makedirs(target)
            with open(os.path.join(target, "config.json"), "w") as f:
                json.dump({"model_type": "hy_v4_flash", "vision_config": {}}, f)
            (prompt,) = load_megatron_bin_dataset(
                SimpleNamespace(
                    train_data_path=prefix + ".idx",
                    cache_dir=os.path.join(root, "cache"),
                    target_model_path=target,
                    max_seq_length=64,
                    min_loss_tokens=1,
                    megatron_loss_offset=300000,
                    megatron_doc_sep_id=120026,
                    megatron_pad_id=120002,
                )
            )

        original = torch.tensor([1, *image, 2, 3])
        self.assertEqual(int((prompt["input_ids"] == 120121).sum()), 0)
        self.assertEqual(prompt["metadata"]["image_newline_slots"], [5, 9])
        recipe = PositionRecipe()
        expected = prepare_sample_positions({"input_ids": original}, recipe)["position_ids"]
        actual = prepare_sample_positions(
            {"input_ids": prompt["input_ids"]}, recipe, multimodal_metadata=prompt["metadata"]
        )["position_ids"]
        self.assertTrue(torch.equal(actual, expected))
        with self.assertRaisesRegex(ValueError, "image_newline_slots"):
            prepare_sample_positions({"input_ids": prompt["input_ids"]}, recipe)

    def test_video_frames_take_the_video_placeholder(self):
        from angelspec.data.positions import PositionRecipe, prepare_sample_positions

        frame = [120118, 120120, 120120, 120121, 120120, 120120, 120121, 120119]
        image = [120118, 120120, 120121, 120119]
        ids = np.array([1, *image, 120122, 7, *frame, 8, *frame, 120123, 2])
        out, slots = _rewrite_image_newlines(
            ids, 120118, 120119, 120121, 120120, video=(120122, 120123, 120124)
        )
        served = [120118, *[120124] * 6, 120119]
        self.assertEqual(
            out.tolist(),
            [1, 120118, 120120, 120120, 120119, 120122, 7, *served, 8, *served, 120123, 2],
        )
        self.assertEqual(slots, [3, 10, 13, 19, 22])
        recipe = PositionRecipe()
        expected = prepare_sample_positions({"input_ids": torch.tensor(ids)}, recipe)
        actual = prepare_sample_positions(
            {"input_ids": torch.tensor(out)},
            recipe,
            multimodal_metadata={"image_newline_slots": slots},
        )
        self.assertTrue(torch.equal(actual["position_ids"], expected["position_ids"]))

    def test_rewrite_follows_the_target_vision_tower(self):
        import json

        hy4 = (120121, 120120, (120122, 120123, 120124))
        cases = [
            ("hy_v4_flash", {"use_image_newline_token": None}, hy4),
            ("hy_v4_flash", {}, hy4),
            ("hy_v4_flash", {"use_image_newline_token": False}, None),
            ("qwen3_vl", {}, None),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for model_type, vision, expected in cases:
                with open(os.path.join(tmp, "config.json"), "w") as f:
                    json.dump(
                        {
                            "model_type": model_type,
                            "vision_config": vision,
                            "image_token_id": 120120,
                            "image_newline_token_id": 120121,
                        },
                        f,
                    )
                args = SimpleNamespace(target_model_path=tmp)
                self.assertEqual(resolve_image_newline_rewrite(args), expected, model_type)
            args = SimpleNamespace(
                target_model_path=tmp, megatron_image_newline_as_placeholder=True
            )
            self.assertEqual(resolve_image_newline_rewrite(args), hy4)


class TestPackLossMaskNumpy(unittest.TestCase):
    def test_matches_torch_implementation(self):
        rng = np.random.default_rng(0)
        cases = [
            np.array([], dtype=bool),
            np.array([False], dtype=bool),
            np.array([True], dtype=bool),
            np.array([False, False, True, True, False], dtype=bool),
            np.array([True, True, False, True], dtype=bool),
        ]
        cases.extend(rng.integers(0, 2, size=n, dtype=np.int8).astype(bool) for n in range(2, 256))
        for mask in cases:
            expected = pack_loss_mask(torch.from_numpy(mask.astype(np.int64)))
            self.assertEqual(_pack_loss_mask_np(mask), expected)


class TestSampleConversations(unittest.TestCase):
    def test_seeded_subset_keeps_file_order(self):
        prompts = [{"data_id": i} for i in range(100)]
        args = SimpleNamespace(megatron_sample_size=10, megatron_sample_seed=3)
        ids = [p["data_id"] for p in _sample_conversations(args, prompts)]
        self.assertEqual(len(set(ids)), 10)
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(ids, [p["data_id"] for p in _sample_conversations(args, prompts)])
        everything = SimpleNamespace(megatron_sample_size=-1)
        self.assertIs(_sample_conversations(everything, prompts), prompts)


class TestMegatronBinParallel(unittest.TestCase):
    def _write_dataset(self, root):
        prefix = os.path.join(root, "train_text_sentence")
        bins = [
            [300001, 2, 3, 120026, 300004, 5, 6, 120026],
            [300007, 300008, 120026, 9, 10, 11, 120026, 120002],
            [300012, 13, 14, 15, 120026],
        ]
        sizes = np.asarray([len(x) for x in bins], dtype=np.int32)
        np.concatenate([np.asarray(x, dtype=np.int32) for x in bins]).tofile(prefix + ".bin")
        with open(prefix + ".idx", "wb") as f:
            f.write(b"MMIDIDX\x00\x00")
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<B", 4))
            f.write(struct.pack("<Q", len(bins)))
            f.write(struct.pack("<Q", len(bins)))
            f.write(sizes.tobytes())
        return prefix + ".idx"

    def _args(self, data_path, cache_dir, workers):
        return SimpleNamespace(
            train_data_path=data_path,
            cache_dir=cache_dir,
            max_seq_length=64,
            min_loss_tokens=1,
            megatron_loss_offset=300000,
            megatron_doc_sep_id=120026,
            megatron_pad_id=120002,
            megatron_max_bins=-1,
            megatron_num_proc=workers,
        )

    def test_parallel_matches_serial_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as root:
            data_path = self._write_dataset(root)
            serial = load_megatron_bin_dataset(
                self._args(data_path, os.path.join(root, "serial"), 1)
            )
            parallel = load_megatron_bin_dataset(
                self._args(data_path, os.path.join(root, "parallel"), 4)
            )
            self.assertEqual(
                [(x["data_id"], x["packed_loss_mask"]) for x in serial],
                [(x["data_id"], x["packed_loss_mask"]) for x in parallel],
            )
            for a, b in zip(serial, parallel):
                self.assertTrue(torch.equal(a["input_ids"], b["input_ids"]))

    def test_bin_window_selects_only_those_bins(self):
        with tempfile.TemporaryDirectory() as root:
            data_path = self._write_dataset(root)
            cache = os.path.join(root, "cache")
            full = load_megatron_bin_dataset(self._args(data_path, cache, 2))

            def window(start, count):
                args = self._args(data_path, cache, 2)
                args.megatron_bin_start, args.megatron_max_bins = start, count
                return load_megatron_bin_dataset(args)

            def bins(samples):
                return [x["data_id"].split("_")[1] for x in samples]

            self.assertEqual(bins(window(1, 1)), ["bin1"])
            tail = window(1, -1)
            self.assertEqual(bins(tail), ["bin1", "bin2"])
            expected = [x for x in full if x["data_id"].split("_")[1] != "bin0"]
            for a, b in zip(tail, expected, strict=True):
                self.assertTrue(torch.equal(a["input_ids"], b["input_ids"]))
                self.assertEqual(a["packed_loss_mask"], b["packed_loss_mask"])
            with self.assertRaises(ValueError):
                window(4, 1)

    def test_eval_dataset_reads_megatron_bins(self):
        from unittest import mock

        from angelspec.controller.training_controller import AsyncTrainingController
        from tests.test_sp_leader_mooncake import (
            FakeRayQueue,
            TestSPLeaderControllerQueues,
        )

        with tempfile.TemporaryDirectory() as root:
            data_path = self._write_dataset(root)
            with mock.patch("angelspec.controller.training_controller.Queue", FakeRayQueue):
                controller = AsyncTrainingController.__ray_actor_class__(
                    TestSPLeaderControllerQueues.make_args(True), dp_size=1
                )
            args = self._args("train.jsonl", os.path.join(root, "cache"), 1)
            args.eval_data_path = data_path
            args.data_format = "jsonl"
            args.megatron_bin_start = 1
            direct = self._args(data_path, args.cache_dir, 1)
            direct.megatron_bin_start = 1
            expected = [x["data_id"] for x in load_megatron_bin_dataset(direct)]
            self.assertEqual(controller.load_eval_dataset(args), len(expected))
            self.assertEqual([x["data_id"] for x in controller._stored_eval_dataset], expected)

    def test_corrupt_cache_is_rebuilt_atomically(self):
        with tempfile.TemporaryDirectory() as root:
            data_path = self._write_dataset(root)
            args = self._args(data_path, os.path.join(root, "cache"), 2)
            expected = load_megatron_bin_dataset(args)
            cache_dir = os.path.join(args.cache_dir, "tokenized_dataset")
            cache_path = next(
                os.path.join(cache_dir, name)
                for name in os.listdir(cache_dir)
                if name.endswith(".pt")
            )
            with open(cache_path, "wb") as f:
                f.write(b"corrupt")
            actual = load_megatron_bin_dataset(args)
            self.assertEqual(len(actual), len(expected))
            self.assertTrue(any(".corrupt." in name for name in os.listdir(cache_dir)))
            self.assertFalse(any(".tmp." in name for name in os.listdir(cache_dir)))

    def test_min_seq_length_drops_short_conversations(self):
        with tempfile.TemporaryDirectory() as root:
            data_path = self._write_dataset(root)
            args = self._args(data_path, os.path.join(root, "cache"), 2)
            cache_dir = os.path.join(args.cache_dir, "tokenized_dataset")
            full = load_megatron_bin_dataset(args)
            self.assertEqual([len(x["input_ids"]) for x in full], [3, 3, 3, 4])
            caches = sorted(os.listdir(cache_dir))

            args.min_seq_length = 0
            self.assertEqual(len(load_megatron_bin_dataset(args)), len(full))
            self.assertEqual(sorted(os.listdir(cache_dir)), caches)

            args.min_seq_length = 4
            kept = load_megatron_bin_dataset(args)
            self.assertEqual([x["input_ids"].tolist() for x in kept], [[12, 13, 14, 15]])
            self.assertEqual(len(os.listdir(cache_dir)), 2)


class TestMegatronBinMultimodal(unittest.TestCase):
    def test_bin_list_resolves_relative_prefixes(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "bin_data_list.txt")
            with open(path, "w") as f:
                f.write("1.0 data/train_text_sentence llm\n")
            sources = _parse_bin_data_list(path)
            self.assertEqual(
                sources,
                [
                    (
                        1.0,
                        os.path.join(root, "data", "train_text_sentence"),
                        "llm",
                    )
                ],
            )

    def test_skips_visual_tokens_outside_image_spans_and_keeps_image_order(self):
        sep = 120026
        media = (120118, 120119, 120120, 120121, 120122, 120123, 120124)
        raw = np.asarray(
            [
                300001,
                120118,
                120120,
                120121,
                120119,
                120120,
                2,
                3,
                sep,
                300001,
                120118,
                120120,
                120121,
                120119,
                2,
                3,
                sep,
            ],
            dtype=np.int32,
        )
        entries, dropped, skipped, stray = _unpack_bin_range(
            raw,
            np.asarray([0, len(raw)]),
            0,
            1,
            300000,
            sep,
            120002,
            64,
            1,
            120118,
            {
                0: [
                    {"bytearray": {"patch": "first"}, "is_video": False},
                    {"bytearray": {"patch": "second"}, "is_video": False},
                ]
            },
            {0: [(32, 32), (32, 32)]},
            media,
        )
        self.assertEqual((dropped, skipped, stray, len(entries)), (0, 0, 1, 1))
        self.assertEqual(entries[0][3]["images"][0]["bytearray"]["patch"], "second")

    def test_masks_all_hy_v4_flash_media_tokens_and_checks_sidecar_count(self):
        sep = 120026
        media = [120118, 120120, 120121, 120119, 120122, 120124, 120123]
        raw = np.asarray(
            [
                300001,
                120118,
                120120,
                120121,
                120119,
                120122,
                120118,
                120120,
                120121,
                120119,
                120123,
                2,
                sep,
            ],
            dtype=np.int32,
        )
        entries, dropped, skipped, _ = _unpack_bin_range(
            raw,
            np.asarray([0, len(raw)]),
            0,
            1,
            300000,
            sep,
            120002,
            64,
            1,
            120118,
            {
                0: [
                    {"bytearray": {}, "is_video": False},
                    {"bytearray": {}, "is_video": True},
                ]
            },
            {0: [(32, 32), (32, 32)]},
            tuple(media),
        )
        self.assertEqual((dropped, skipped), (0, 0))
        ids = entries[0][1]
        mask = unpack_loss_mask(entries[0][2])
        self.assertEqual(int(mask[torch.isin(ids, torch.tensor(media))].sum()), 0)

        with self.assertRaisesRegex(ValueError, "sidecar contains 0"):
            _unpack_bin_range(
                raw,
                np.asarray([0, len(raw)]),
                0,
                1,
                300000,
                sep,
                120002,
                64,
                1,
                120118,
                {0: []},
                {0: []},
                tuple(media),
            )

    def test_groups_video_frames_from_sidecar_and_token_stream(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "train_text_sentence_img.npy")
            np.save(
                path,
                {
                    "img_path": {
                        0: [
                            {
                                "patch": "/tmp/image.patch",
                                "start_num": 0,
                                "size": 1,
                            },
                            {
                                "patch": "/tmp/frame.patch",
                                "start_num": 0,
                                "size": 1,
                            },
                            {
                                "patch": "/tmp/frame.patch",
                                "start_num": 1,
                                "size": 1,
                            },
                        ]
                    },
                    "img_target_size": {0: [(32, 32), (64, 32), (64, 32)]},
                    "is_video": {0: [0, 1, 1]},
                    "video_idx": {0: [0, 1, 1]},
                },
            )
            refs, sizes = _load_image_sidecar(path)
            raw = np.asarray(
                [
                    300001,
                    120118,
                    120120,
                    120121,
                    120119,
                    120122,
                    10,
                    120118,
                    120120,
                    120121,
                    120119,
                    11,
                    120118,
                    120120,
                    120121,
                    120119,
                    120123,
                    2,
                    120026,
                ],
                dtype=np.int32,
            )
            entries, dropped, skipped, _ = _unpack_bin_range(
                raw,
                np.asarray([0, len(raw)]),
                0,
                1,
                300000,
                120026,
                120002,
                64,
                1,
                120118,
                refs,
                sizes,
                (120118, 120119, 120120, 120121, 120122, 120123, 120124),
            )
            self.assertEqual((dropped, skipped), (0, 0))
            mm = entries[0][3]
            self.assertEqual(len(mm["images"]), 1)
            self.assertEqual(len(mm["videos"]), 1)
            self.assertEqual(len(mm["videos"][0]["frames"]), 2)
            self.assertEqual(mm["videos"][0]["frame_prefix_token_ids"], [[10], [11]])
            self.assertEqual(mm["videos"][0]["suffix_token_ids"], [])

    def test_allows_sidecar_without_target_sizes(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "train_text_sentence_img.npy")
            np.save(
                path,
                {
                    "img_path": {
                        0: [
                            {
                                "patch": "/tmp/image.patch",
                                "start_num": 0,
                                "size": 1,
                            }
                        ]
                    },
                },
            )
            images, sizes = _load_image_sidecar(path)
            self.assertEqual(len(images[0]), 1)
            self.assertEqual(sizes[0], [])

    def test_remaps_absolute_patch_paths(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "train_text_sentence_img.npy")
            np.save(
                path,
                {
                    "img_path": {
                        0: [
                            {
                                "patch": "/old/root/patch-00000.patch",
                                "start_num": 2,
                                "size": 3,
                            }
                        ]
                    },
                    "img_target_size": {0: [(32, 32)]},
                },
            )
            images, _ = _load_image_sidecar(path, path_replacements="/old/root:/new/root")
            self.assertEqual(
                images[0][0]["bytearray"]["patch"],
                "/new/root/patch-00000.patch",
            )

    def test_token_weighted_blend_preserves_expected_token_ratio(self):
        from unittest import mock

        def sample(kind, index, length):
            return {
                "data_id": f"{kind}-{index}",
                "metadata": {"source": kind},
                "input_ids": torch.zeros(length, dtype=torch.long),
            }

        long = [sample("llm", i, 100) for i in range(100)]
        short = [sample("vlm", i, 10) for i in range(10)]
        args = SimpleNamespace(
            bin_data_list="/tmp/list",
            seed=1,
            megatron_blend_unit="tokens",
        )
        with (
            mock.patch(
                "angelspec.data.megatron_bin._parse_bin_data_list",
                return_value=[(1.0, "llm", "llm"), (1.0, "vlm", "vlm")],
            ),
            mock.patch(
                "angelspec.data.megatron_bin._load_megatron_source",
                side_effect=[long, short],
            ),
        ):
            blended = load_megatron_bin_blend(args)
        counts = {
            kind: sum(item["metadata"]["source"] == kind for item in blended)
            for kind in ("llm", "vlm")
        }
        llm_tokens = counts["llm"] * 100
        vlm_tokens = counts["vlm"] * 10
        self.assertLess(
            abs(llm_tokens / (llm_tokens + vlm_tokens) - 0.5),
            0.02,
        )
