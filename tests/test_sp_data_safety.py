import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from angelspec.controller.loop import _validate_train_results
from angelspec.data.utils import DataCollatorWithPadding
from angelspec.training.data_fetcher import MooncakeDataset, TrainSample
from angelspec.training.dflash_trainer import DFlashTrainer
from angelspec.training.trainer import _with_lookahead_islast


class TestQueueFailures(unittest.TestCase):
    def dataset(self, packing):
        return MooncakeDataset(
            ray_queue=None,
            mooncake_store=None,
            device=torch.device("cpu"),
            packing_enabled=packing,
            max_seq_length=16,
        )

    def test_queue_and_collective_errors_propagate(self):
        for packing in (False, True):
            for error in (RuntimeError("broadcast failed"), queue.Empty("timeout")):
                with self.subTest(packing=packing, error=type(error)):
                    dataset = self.dataset(packing)
                    with patch.object(dataset, "_get_next_item", side_effect=error):
                        with self.assertRaises(type(error)) as caught:
                            list(dataset)
                    self.assertIs(caught.exception, error)

    def test_empty_sentinel_ends_dataset_normally(self):
        for packing in (False, True):
            dataset = self.dataset(packing)
            with patch.object(dataset, "_get_next_item", return_value=None):
                self.assertEqual(list(dataset), [])

    def test_pre_sharded_queue_errors_propagate(self):
        dataset = self.dataset(False)
        dataset.usp_enabled = True
        dataset.ray_queue = Mock()
        dataset.ray_queue.get.side_effect = RuntimeError("queue failed")
        with self.assertRaisesRegex(RuntimeError, "queue failed"):
            list(dataset)

    def test_partial_packed_row_is_not_yielded(self):
        dataset = self.dataset(True)
        sample = TrainSample("key", {}, row_id=0)
        for ending in (None, RuntimeError("broadcast failed")):
            with self.subTest(ending=ending):
                with (
                    patch.object(dataset, "_get_next_item", side_effect=[sample, ending]),
                    patch.object(dataset, "_prepare_one", return_value=({"input_ids": 1}, 0)),
                ):
                    with self.assertRaises(RuntimeError):
                        next(iter(dataset))

    def test_complete_packed_row_is_yielded(self):
        dataset = self.dataset(True)
        sample = TrainSample("key", {}, row_id=0, last_in_row=True)
        with (
            patch.object(dataset, "_get_next_item", side_effect=[sample, None]),
            patch.object(dataset, "_prepare_one", return_value=({"input_ids": 1}, 0)),
        ):
            self.assertEqual(list(dataset), [[{"input_ids": 1}]])

    def test_incomplete_step_raises(self):
        for count in (0, 1):
            with self.subTest(count=count):
                with self.assertRaisesRegex(RuntimeError, f"{count}/2"):
                    list(_with_lookahead_islast(range(count), known_count=2))

    def test_step_count_has_no_lookahead(self):
        consumed = []

        def batches():
            for i in range(2):
                consumed.append(i)
                yield i

        iterator = _with_lookahead_islast(batches(), known_count=2)
        self.assertEqual(next(iterator), (0, False))
        self.assertEqual(consumed, [0])
        self.assertEqual(next(iterator), (1, True))
        self.assertEqual(list(iterator), [])

    def test_failed_worker_results_cannot_advance_step(self):
        for results in (
            [],
            [{}],
            [{"loss": 0}, {}],
            [None, None],
            ["invalid", {"loss": 1}],
        ):
            with self.subTest(results=results):
                with self.assertRaises(RuntimeError):
                    _validate_train_results(results, 2, 900)
        _validate_train_results([{"loss": 0}, {"loss": 1}], 2, 900)


class TestShardedEval(unittest.TestCase):
    @staticmethod
    def samples():
        return [
            {
                "input_ids": torch.arange(n).view(1, n),
                "loss_mask": torch.ones(1, n),
                "hidden_states": torch.arange(n // 2, n).view(1, n // 2, 1).float(),
            }
            for n in (256, 512)
        ]

    def test_sharded_collator_rejects_rebatching(self):
        collator = DataCollatorWithPadding(usp_sequence_sharded=True, sp_size=2)
        with self.assertRaisesRegex(ValueError, "collated individually"):
            collator(self.samples())

    def test_eval_preserves_each_sample_shard_offset(self):
        trainer = DFlashTrainer.__new__(DFlashTrainer)
        trainer.args = SimpleNamespace(eval_micro_batch_size=2, micro_batch_size=1)
        trainer._eval_cache = self.samples()
        trainer._eval_collator = DataCollatorWithPadding(usp_sequence_sharded=True, sp_size=2)
        trainer.model = Mock()
        trainer._defer_batch_h2d = lambda: True
        batches = []
        trainer.eval_forward = lambda batch: batches.append(batch) or {"loss": 0}
        trainer._aggregate_eval_metrics = lambda metrics: {"samples": len(metrics)}
        self.assertEqual(trainer.eval_from_cache(), {"samples": 2})
        self.assertEqual([b["input_ids"].shape[1] for b in batches], [256, 512])
        self.assertEqual([b["hidden_states"][0, 0, 0].item() for b in batches], [128, 256])
        trainer.model.train.assert_called_once()

    def test_unsharded_eval_retains_requested_batch_size(self):
        trainer = DFlashTrainer.__new__(DFlashTrainer)
        trainer.args = SimpleNamespace(eval_micro_batch_size=2, micro_batch_size=1)
        trainer._eval_cache = self.samples()
        trainer._eval_collator = DataCollatorWithPadding()
        trainer.model = Mock()
        trainer._defer_batch_h2d = lambda: True
        batches = []
        trainer.eval_forward = lambda batch: batches.append(batch) or {"loss": 0}
        trainer._aggregate_eval_metrics = lambda metrics: {"batches": len(metrics)}
        self.assertEqual(trainer.eval_from_cache(), {"batches": 1})
        self.assertEqual(batches[0]["input_ids"].shape[0], 2)


class TestScatterDevices(unittest.TestCase):
    def dataset(self):
        dataset = MooncakeDataset.__new__(MooncakeDataset)
        dataset.device = torch.device("cpu")
        dataset._sp_gloo_group = object()
        dataset._sp_rank = 0
        dataset._sp_world_size = 2
        dataset._leader_scatter_logged = True
        dataset.mooncake_store = Mock()
        dataset.usp_local_shard = True
        return dataset

    def test_store_output_is_normalized_before_status(self):
        dataset = self.dataset()
        sample = TrainSample(
            "key",
            {"input_ids": (4,), "hidden_states": (4, 2)},
            {"input_ids": "int64", "hidden_states": "bfloat16"},
        )
        source = {
            name: Mock(shape=shape, device=torch.device("cuda:0"))
            for name, shape in sample.tensor_shapes.items()
        }
        for name, tensor in source.items():
            tensor.to.return_value = torch.ones(
                tensor.shape, dtype=getattr(torch, sample.tensor_dtypes[name])
            )
        dataset.mooncake_store.get.return_value.to_tensor_dict.return_value = dict(source)

        def status(payload, **kwargs):
            self.assertEqual(payload, [(True, False, None)])
            for name, tensor in source.items():
                tensor.to.assert_called_once_with(
                    device=torch.device("cpu"),
                    dtype=getattr(torch, sample.tensor_dtypes[name]),
                )

        def broadcast(tensor, **kwargs):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertIs(kwargs["group"], dataset._sp_gloo_group)

        def scatter(tensor, **kwargs):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertIs(kwargs["group"], dataset._sp_gloo_group)
            for shard in kwargs["scatter_list"]:
                self.assertEqual(shard.device.type, "cpu")
            tensor.copy_(kwargs["scatter_list"][0])

        with (
            patch("angelspec.training.data_fetcher.dist.get_global_rank", return_value=8),
            patch(
                "angelspec.training.data_fetcher.dist.broadcast_object_list", side_effect=status
            ),
            patch("angelspec.training.data_fetcher.dist.broadcast", side_effect=broadcast),
            patch("angelspec.training.data_fetcher.dist.scatter", side_effect=scatter),
        ):
            result = dataset._load_from_mooncake_leader_scatter(sample)
        self.assertEqual(result["hidden_states"].shape, (128, 2))

    def test_conversion_error_is_broadcast_before_collectives(self):
        dataset = self.dataset()
        sample = TrainSample("key", {"input_ids": (4,)})
        tensor = Mock(shape=(4,))
        tensor.to.side_effect = RuntimeError("device conversion failed")
        dataset.mooncake_store.get.return_value.to_tensor_dict.return_value = {"input_ids": tensor}
        with (
            patch("angelspec.training.data_fetcher.dist.get_global_rank", return_value=8),
            patch("angelspec.training.data_fetcher.dist.broadcast_object_list") as status,
            patch("angelspec.training.data_fetcher.dist.broadcast") as broadcast,
            patch("angelspec.training.data_fetcher.dist.scatter") as scatter,
        ):
            with self.assertRaisesRegex(RuntimeError, "device conversion failed"):
                dataset._load_from_mooncake_leader_scatter(sample)
        self.assertFalse(status.call_args.args[0][0][0])
        broadcast.assert_not_called()
        scatter.assert_not_called()
