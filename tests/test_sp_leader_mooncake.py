import math
import os
import queue as std_queue
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from angelspec.controller.training_controller import AsyncTrainingController
from angelspec.data.utils import DFlashPackingCollator
from angelspec.training.data_fetcher import (
    MooncakeDataset,
    PrefetchedDataFetcher,
    TrainSample,
    _LeaderReadAhead,
)
from angelspec.training.trainer import Trainer
from angelspec.transfer.mooncake.draft_store import MooncakeSampleUnavailable


class FakeQueue:
    def __init__(self, item=None, error=None):
        self.item = item
        self.error = error
        self.get_calls = 0

    def get(self, block=True, timeout=None):
        self.get_calls += 1
        if self.error is not None:
            raise self.error
        return self.item


class SequenceQueue(FakeQueue):
    def __init__(self, items):
        super().__init__()
        self.items = iter(items)

    def get(self, block=True, timeout=None):
        self.get_calls += 1
        return next(self.items)


class FakeTensors:
    def __init__(self, tensors):
        self.tensors = tensors

    def to_tensor_dict(self):
        return self.tensors


class FakeStore:
    def __init__(self):
        self.get_calls = 0

    def get(self, key, shapes, dtypes, device):
        self.get_calls += 1
        return FakeTensors(
            {
                name: torch.arange(
                    torch.tensor(shape).prod().item(), dtype=dtypes[name], device=device
                ).reshape(shape)
                for name, shape in shapes.items()
            }
        )


class FakeRayQueue:
    def __init__(self, *args, **kwargs):
        self.items = []
        self.options = kwargs.get("actor_options")

    def put(self, item):
        self.items.append(item)


def make_sample():
    return TrainSample(
        mooncake_key="k0",
        tensor_shapes={"input_ids": (4,), "last_hidden_states": (4, 2)},
        tensor_dtypes={"input_ids": "int64", "last_hidden_states": "bfloat16"},
        packed_loss_mask="2,2",
    )


def keyed_sample(index, **fields):
    sample = make_sample()
    sample.mooncake_key = f"k{index}"
    for name, value in fields.items():
        setattr(sample, name, value)
    return sample


def keyed_tensors(key, shapes, dtypes):
    offset = 10 * int(key[1:])
    return {
        name: (torch.arange(math.prod(shape)) + offset).to(dtypes[name]).reshape(shape)
        for name, shape in shapes.items()
    }


class KeyedStore:
    """Per-key tensors; records the thread of every read."""

    def __init__(self, failures=None):
        self.failures = failures or {}
        self.threads = []

    @property
    def get_calls(self):
        return len(self.threads)

    def get(self, key, shapes, dtypes, device):
        self.threads.append(threading.current_thread())
        if key in self.failures:
            raise self.failures[key]
        return FakeTensors(
            {
                name: tensor.to(device)
                for name, tensor in keyed_tensors(key, shapes, dtypes).items()
            }
        )


class AliasingStore(KeyedStore):
    """Returns views of one buffer that every read overwrites, like host-buffer reads."""

    def __init__(self):
        super().__init__()
        self.buffers = {}

    def get(self, key, shapes, dtypes, device):
        tensors = super().get(key, shapes, dtypes, device).to_tensor_dict()
        views = {}
        for name, tensor in tensors.items():
            buffer = self.buffers.setdefault(name, torch.empty_like(tensor))
            buffer.copy_(tensor)
            views[name] = buffer.view(tensor.shape)
        return FakeTensors(views)


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


def run_distributed_fetch(rank, world_size, init_file, packing=False):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        gloo_group = dist.new_group(backend="gloo")
        sample = make_sample()
        if packing:
            sample.tensor_shapes["hidden_states"] = (4, 2)
            sample.tensor_dtypes["hidden_states"] = "bfloat16"
            sample.row_id = 0
            second = make_sample()
            second.mooncake_key = "k1"
            second.tensor_shapes["hidden_states"] = (4, 2)
            second.tensor_dtypes["hidden_states"] = "bfloat16"
            second.row_id = 0
            second.last_in_row = True
        queue = FakeQueue(
            sample if rank == 0 else None,
            error=None if rank == 0 else AssertionError(),
        )
        if packing and rank == 0:
            queue = SequenceQueue([sample, second])
        store = FakeStore() if rank == 0 else None
        with (
            patch(
                "angelspec.training.data_fetcher.get_draft_sp_group",
                return_value=dist.group.WORLD,
            ),
            patch("angelspec.training.data_fetcher.get_sp_ring_group", return_value=None),
            patch(
                "angelspec.training.data_fetcher.get_draft_sp_gloo_group",
                return_value=gloo_group,
            ),
        ):
            dataset = MooncakeDataset(
                ray_queue=queue,
                mooncake_store=store,
                device=torch.device("cpu"),
                usp_local_shard=True,
                sp_leader_mooncake_fetch=True,
                packing_enabled=packing,
                max_seq_length=10 if packing else None,
            )
        broadcast = dist.broadcast
        broadcast_objects = dist.broadcast_object_list

        def checked_broadcast(*args, **kwargs):
            assert kwargs["group"] is gloo_group
            return broadcast(*args, **kwargs)

        def checked_objects(*args, **kwargs):
            assert kwargs["group"] is gloo_group
            assert kwargs["device"] == torch.device("cpu")
            return broadcast_objects(*args, **kwargs)

        with (
            patch("angelspec.training.data_fetcher.dist.broadcast", side_effect=checked_broadcast),
            patch(
                "angelspec.training.data_fetcher.dist.broadcast_object_list",
                side_effect=checked_objects,
            ),
        ):
            if packing:
                row = next(iter(dataset))
                result = DFlashPackingCollator(10)(row)
                assert result["input_ids"].tolist() == [[0, 1, 2, 3, 0, 1, 2, 3, 0, 0]]
                assert result["ctx_doc_ids"].tolist() == [[0, 0, 0, 0, 1, 1, 1, 1, -1, -1]]
                expected = torch.arange(8, dtype=torch.bfloat16).reshape(1, 4, 2)
                assert torch.equal(result["hidden_states"][:, :4], expected)
                assert torch.equal(result["hidden_states"][:, 4:8], expected)
                assert result["hidden_states"].device.type == "cpu"
            else:
                item = dataset._get_next_item()
                result = dataset._load_from_mooncake(item)
                assert item.mooncake_key == "k0"
                assert torch.equal(result["input_ids"], torch.arange(4))
                assert torch.equal(
                    result["last_hidden_states"],
                    torch.arange(8, dtype=torch.bfloat16).reshape(4, 2),
                )
        counts = torch.tensor(
            [queue.get_calls, store.get_calls if store is not None else 0],
            dtype=torch.int64,
        )
        dist.all_reduce(counts)
        assert counts.tolist() == ([2, 2] if packing else [1, 1])
    finally:
        dist.destroy_process_group()


def run_distributed_read_ahead(rank, world_size, init_file, packing):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        gloo_group = dist.new_group(backend="gloo")
        samples = [keyed_sample(i) for i in range(4)]
        if packing:
            for i, sample in enumerate(samples):
                sample.tensor_shapes["hidden_states"] = (4, 2)
                sample.tensor_dtypes["hidden_states"] = "bfloat16"
                sample.row_id = i // 2
                sample.last_in_row = i % 2 == 1
        store = None
        if rank == 0:
            source = std_queue.Queue()
            for sample in samples + [None]:
                source.put(sample)
            store = KeyedStore()
        else:
            source = FakeQueue(error=AssertionError("only the SP leader reads the queue"))
        with (
            patch(
                "angelspec.training.data_fetcher.get_draft_sp_group",
                return_value=dist.group.WORLD,
            ),
            patch("angelspec.training.data_fetcher.get_sp_ring_group", return_value=None),
            patch(
                "angelspec.training.data_fetcher.get_draft_sp_gloo_group",
                return_value=gloo_group,
            ),
        ):
            dataset = MooncakeDataset(
                ray_queue=source,
                mooncake_store=store,
                device=torch.device("cpu"),
                usp_local_shard=True,
                sp_leader_mooncake_fetch=True,
                packing_enabled=packing,
                max_seq_length=10 if packing else None,
                leader_read_ahead_items=0 if packing else 2,
                leader_read_ahead_tokens=8 if packing else 0,
            )
        main = threading.main_thread()

        def on_main_thread(collective):
            def checked(*args, **kwargs):
                assert threading.current_thread() is main
                return collective(*args, **kwargs)

            return checked

        with (
            patch(
                "angelspec.training.data_fetcher.dist.broadcast",
                side_effect=on_main_thread(dist.broadcast),
            ),
            patch(
                "angelspec.training.data_fetcher.dist.broadcast_object_list",
                side_effect=on_main_thread(dist.broadcast_object_list),
            ),
        ):
            if packing:
                rows = list(dataset._iter_rows_by_id())
                assert len(rows) == 2
                for index, row in enumerate(rows):
                    batch = DFlashPackingCollator(10)(row)
                    first, second = 20 * index, 20 * index + 10
                    assert batch["input_ids"].tolist() == [
                        [first + i for i in range(4)] + [second + i for i in range(4)] + [0, 0]
                    ]
                    hidden = torch.arange(8, dtype=torch.bfloat16).reshape(1, 4, 2)
                    assert torch.equal(batch["hidden_states"][:, :4], hidden + first)
                    assert torch.equal(batch["hidden_states"][:, 4:8], hidden + second)
            else:
                keys = []
                while (item := dataset._get_next_item()) is not None:
                    result = dataset._load_from_mooncake(item)
                    keys.append(item.mooncake_key)
                    expected = keyed_tensors(
                        item.mooncake_key,
                        item.tensor_shapes,
                        dataset._resolve_dtypes(item.tensor_dtypes),
                    )
                    for name, tensor in expected.items():
                        assert torch.equal(result[name], tensor), name
                assert keys == ["k0", "k1", "k2", "k3"]
        dataset.close()
        if rank == 0:
            assert store.get_calls == 4
            assert all(thread is not main for thread in store.threads)
    finally:
        dist.destroy_process_group()


class TestSPLeaderMooncake(unittest.TestCase):
    def make_dataset(self, rank, queue=None, store=None):
        with (
            patch("angelspec.training.data_fetcher.get_draft_sp_group", return_value=object()),
            patch("angelspec.training.data_fetcher.dist.get_world_size", return_value=2),
            patch("angelspec.training.data_fetcher.dist.get_rank", return_value=rank),
            patch(
                "angelspec.training.data_fetcher.get_draft_sp_gloo_group",
                return_value=object(),
            ),
        ):
            return MooncakeDataset(
                ray_queue=queue,
                mooncake_store=store,
                device=torch.device("cpu"),
                usp_local_shard=True,
                sp_leader_mooncake_fetch=True,
            )

    def test_only_leader_reads_ray_queue(self):
        sample = make_sample()
        queue = FakeQueue(sample)
        leader = self.make_dataset(0, queue=queue)
        follower = self.make_dataset(1, queue=queue)
        with patch.object(
            leader, "_broadcast_object_from_sp_leader", return_value=(True, sample, None)
        ):
            self.assertIs(leader._get_next_item(), sample)
        with patch.object(
            follower, "_broadcast_object_from_sp_leader", return_value=(True, sample, None)
        ):
            self.assertIs(follower._get_next_item(), sample)
        self.assertEqual(queue.get_calls, 1)

    def test_only_leader_reads_mooncake(self):
        sample = make_sample()
        store = FakeStore()
        leader = self.make_dataset(0, store=store)
        follower = self.make_dataset(1)
        dtypes = {"input_ids": torch.int64, "last_hidden_states": torch.bfloat16}
        with (
            patch.object(
                leader, "_broadcast_object_from_sp_leader", return_value=(True, False, None)
            ),
            patch("angelspec.training.data_fetcher.dist.get_global_rank", return_value=0),
            patch("angelspec.training.data_fetcher.dist.broadcast"),
        ):
            leader_result = leader._load_from_mooncake_sp_leader(sample, dtypes)
        tensors = iter(leader_result.values())

        def fill(tensor, **kwargs):
            tensor.copy_(next(tensors))

        with (
            patch.object(
                follower, "_broadcast_object_from_sp_leader", return_value=(True, False, None)
            ),
            patch("angelspec.training.data_fetcher.dist.get_global_rank", return_value=0),
            patch("angelspec.training.data_fetcher.dist.broadcast", side_effect=fill),
        ):
            follower_result = follower._load_from_mooncake_sp_leader(sample, dtypes)
        self.assertEqual(store.get_calls, 1)
        for name in leader_result:
            self.assertTrue(torch.equal(leader_result[name], follower_result[name]))

    def test_leader_failure_is_forwarded(self):
        sample = make_sample()
        leader = self.make_dataset(0)
        broadcasts = []

        def capture(value):
            broadcasts.append(value)
            return value

        with (
            patch.object(leader, "_broadcast_object_from_sp_leader", side_effect=capture),
            patch("angelspec.training.data_fetcher.dist.get_global_rank", return_value=0),
        ):
            with self.assertRaisesRegex(RuntimeError, "SP leader failed to load Mooncake key k0"):
                leader._load_from_mooncake_sp_leader(
                    sample,
                    {"input_ids": torch.int64, "last_hidden_states": torch.bfloat16},
                )
        self.assertFalse(broadcasts[0][0])

    def test_broadcast_context_follows_configured_device(self):
        dataset = self.make_dataset(0)
        for configured, expected in (("cpu", "cpu"), ("cuda:3", "cuda:3"), (3, "cuda:3")):
            with self.subTest(device=configured):
                dataset.device = configured
                expected_group = dataset._sp_gloo_group if expected == "cpu" else dataset._sp_group
                with (
                    patch(
                        "angelspec.training.data_fetcher.dist.get_global_rank", return_value=8
                    ) as root,
                    patch(
                        "angelspec.training.data_fetcher.dist.broadcast_object_list"
                    ) as broadcast,
                ):
                    dataset._broadcast_object_from_sp_leader("value")
                root.assert_called_once_with(expected_group, 0)
                self.assertIs(broadcast.call_args.kwargs["group"], expected_group)
                self.assertEqual(broadcast.call_args.kwargs["src"], 8)
                self.assertEqual(broadcast.call_args.kwargs["device"], torch.device(expected))

    def test_cpu_path_requires_sp_gloo_group(self):
        dataset = self.make_dataset(0)
        dataset._sp_gloo_group = None
        with self.assertRaisesRegex(RuntimeError, "not initialized for device cpu"):
            dataset._sp_broadcast_context()

    def test_leader_normalizes_tensors_before_success_status(self):
        sample = make_sample()
        dtypes = {"input_ids": torch.int64, "last_hidden_states": torch.bfloat16}
        source = {
            name: Mock(shape=shape, device=torch.device("cuda:0"))
            for name, shape in sample.tensor_shapes.items()
        }
        for name, tensor in source.items():
            tensor.to.return_value = torch.ones(sample.tensor_shapes[name], dtype=dtypes[name])
        store = Mock()
        store.get.return_value = FakeTensors(dict(source))
        dataset = self.make_dataset(0, store=store)

        def status(value):
            for name, tensor in source.items():
                tensor.to.assert_called_once_with(device=torch.device("cpu"), dtype=dtypes[name])
            return value

        with (
            patch.object(dataset, "_broadcast_object_from_sp_leader", side_effect=status),
            patch("angelspec.training.data_fetcher.dist.get_global_rank", return_value=8),
            patch("angelspec.training.data_fetcher.dist.broadcast") as broadcast,
        ):
            result = dataset._load_from_mooncake_sp_leader(sample, dtypes)
        for call in broadcast.call_args_list:
            self.assertIs(call.kwargs["group"], dataset._sp_gloo_group)
            self.assertEqual(call.kwargs["src"], 8)
            self.assertEqual(call.args[0].device.type, "cpu")
        self.assertEqual(set(result), set(sample.tensor_shapes))

    def test_conversion_failure_is_forwarded_before_tensor_broadcast(self):
        sample = make_sample()
        tensor = Mock(shape=(4,))
        tensor.to.side_effect = RuntimeError("conversion failed")
        store = Mock()
        store.get.return_value = FakeTensors({"input_ids": tensor})
        dataset = self.make_dataset(0, store=store)
        with (
            patch.object(
                dataset, "_broadcast_object_from_sp_leader", side_effect=lambda value: value
            ),
            patch("angelspec.training.data_fetcher.dist.get_global_rank", return_value=0),
            patch("angelspec.training.data_fetcher.dist.broadcast") as broadcast,
        ):
            with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                dataset._load_from_mooncake_sp_leader(sample, {})
        broadcast.assert_not_called()

    def test_mode_requires_usp_local_shard(self):
        with self.assertRaisesRegex(ValueError, "requires usp_local_shard=true"):
            MooncakeDataset(
                ray_queue=None,
                mooncake_store=None,
                device=torch.device("cpu"),
                usp_local_shard=False,
                sp_leader_mooncake_fetch=True,
            )

    def test_distributed_gloo_fetches_once(self):
        self._run_distributed_fetch(packing=False)

    def test_distributed_gloo_packing_fetches_once(self):
        self._run_distributed_fetch(packing=True)

    def _run_distributed_fetch(self, packing):
        if os.environ.get("RUN_DISTRIBUTED_TESTS") != "1":
            self.skipTest("set RUN_DISTRIBUTED_TESTS=1")
        fd, init_file = tempfile.mkstemp()
        os.close(fd)
        os.unlink(init_file)
        try:
            mp.spawn(run_distributed_fetch, args=(2, init_file, packing), nprocs=2, join=True)
        finally:
            if os.path.exists(init_file):
                os.unlink(init_file)


class TestSPLeaderControllerQueues(unittest.TestCase):
    @staticmethod
    def make_args(enabled):
        return type(
            "Args",
            (),
            {
                "sp_ulysses_size": 4,
                "sp_ring_size": 1,
                "attention_backend": "usp",
                "usp_local_shard": True,
                "sp_leader_mooncake_fetch": enabled,
                "per_dp_rank_batch_size": 1,
                "dflash_packing": False,
                "mtp_packing": False,
                "max_seq_length": 16,
                "packing_min_fill_ratio": 0.0,
                "packing_max_wait_seconds": 5.0,
                "seed": 1,
                "shuffle_dataset": False,
            },
        )()

    def test_queue_count_is_dp_in_leader_mode(self):
        cls = AsyncTrainingController.__ray_actor_class__
        with patch("angelspec.controller.training_controller.Queue", FakeRayQueue):
            controller = cls(self.make_args(True), dp_size=3)
        self.assertEqual(controller.queue_count, 3)
        self.assertEqual([q.options for q in controller.train_queues], [{"num_cpus": 0}] * 3)

    def test_legacy_queue_count_is_dp_times_sp(self):
        cls = AsyncTrainingController.__ray_actor_class__
        with patch("angelspec.controller.training_controller.Queue", FakeRayQueue):
            controller = cls(self.make_args(False), dp_size=3)
        self.assertEqual(controller.queue_count, 12)

    def test_dispatch_puts_once_per_dp_group(self):
        cls = AsyncTrainingController.__ray_actor_class__
        with patch("angelspec.controller.training_controller.Queue", FakeRayQueue):
            controller = cls(self.make_args(True), dp_size=2)
        samples = [
            type(
                "Output",
                (),
                {
                    "mooncake_key": f"k{i}",
                    "tensor_shapes": {"input_ids": (4,)},
                    "tensor_dtypes": {"input_ids": "int64"},
                    "packed_loss_mask": "2,2",
                    "metadata": {},
                },
            )()
            for i in range(2)
        ]
        controller._dispatch_to_queues(samples, controller.train_queues)
        self.assertEqual([len(q.items) for q in controller.train_queues], [1, 1])


class TestSPLeaderReadAhead(unittest.TestCase):
    dtypes = {"input_ids": torch.int64, "last_hidden_states": torch.bfloat16}

    def make_dataset(self, source, store, items=2, tokens=0, timeout=None):
        with patch("angelspec.training.data_fetcher.get_draft_sp_group", return_value=None):
            dataset = MooncakeDataset(
                ray_queue=source,
                mooncake_store=store,
                device=torch.device("cpu"),
                timeout=timeout,
                usp_local_shard=True,
                sp_leader_mooncake_fetch=True,
                leader_read_ahead_items=items,
                leader_read_ahead_tokens=tokens,
            )
        self.addCleanup(dataset.close)
        return dataset

    def filled_queue(self, count):
        source = std_queue.Queue()
        for index in range(count):
            source.put(keyed_sample(index))
        source.put(None)
        return source

    def test_reads_ahead_up_to_the_budget(self):
        for items, tokens in ((2, 0), (0, 8)):
            with self.subTest(items=items, tokens=tokens):
                store = KeyedStore()
                dataset = self.make_dataset(self.filled_queue(6), store, items, tokens)
                self.assertEqual(dataset._get_next_item().mooncake_key, "k0")
                # Two samples staged, one more read and waiting for room.
                wait_for(lambda store=store: store.get_calls == 4)
                time.sleep(0.2)
                self.assertEqual(store.get_calls, 4)
                keys = ["k0"]
                while (item := dataset._get_next_item()) is not None:
                    keys.append(item.mooncake_key)
                self.assertEqual(keys, [f"k{i}" for i in range(6)])

    def test_loads_staged_tensors_in_order(self):
        store = KeyedStore()
        dataset = self.make_dataset(self.filled_queue(3), store)
        while (item := dataset._get_next_item()) is not None:
            result = dataset._load_from_mooncake(item)
            expected = keyed_tensors(item.mooncake_key, item.tensor_shapes, self.dtypes)
            for name, tensor in expected.items():
                self.assertTrue(torch.equal(result[name], tensor), name)
        self.assertEqual(store.get_calls, 3)
        self.assertNotIn(threading.main_thread(), store.threads)

    def test_staged_tensors_do_not_alias_the_store_buffer(self):
        store = AliasingStore()
        dataset = self.make_dataset(self.filled_queue(2), store)
        first = dataset._get_next_item()
        wait_for(lambda: store.get_calls == 2)
        result = dataset._load_from_mooncake_sp_leader(first, self.dtypes)
        expected = keyed_tensors("k0", first.tensor_shapes, self.dtypes)
        for name, tensor in expected.items():
            self.assertTrue(torch.equal(result[name], tensor), name)

    def test_staged_unavailable_sample_raises_at_load(self):
        store = KeyedStore({"k0": MooncakeSampleUnavailable("evicted")})
        dataset = self.make_dataset(self.filled_queue(2), store)
        item = dataset._get_next_item()
        with self.assertRaisesRegex(MooncakeSampleUnavailable, "evicted"):
            dataset._load_from_mooncake_sp_leader(item, self.dtypes)
        nxt = dataset._get_next_item()
        self.assertEqual(nxt.mooncake_key, "k1")
        self.assertIn("input_ids", dataset._load_from_mooncake_sp_leader(nxt, self.dtypes))

    def test_staged_unavailable_status_is_broadcast(self):
        store = KeyedStore({"k0": MooncakeSampleUnavailable("evicted")})
        dataset = self.make_dataset(self.filled_queue(1), store)
        item = dataset._get_next_item()
        dataset._sp_world_size = 2
        broadcasts = []

        def capture(value):
            broadcasts.append(value)
            return value

        with (
            patch.object(dataset, "_broadcast_object_from_sp_leader", side_effect=capture),
            patch.object(dataset, "_sp_broadcast_context", return_value=(None, "cpu", 0)),
            patch("angelspec.training.data_fetcher.dist.broadcast") as broadcast,
        ):
            with self.assertRaises(MooncakeSampleUnavailable):
                dataset._load_from_mooncake_sp_leader(item, self.dtypes)
        self.assertEqual(broadcasts[0][:2], (False, True))
        broadcast.assert_not_called()

    def test_queue_timeout_is_reported(self):
        dataset = self.make_dataset(std_queue.Queue(), KeyedStore(), timeout=0.3)
        with self.assertRaisesRegex(RuntimeError, "failed to read training queue: Empty"):
            dataset._get_next_item()

    def test_close_stops_a_reader_waiting_on_an_empty_queue(self):
        source = std_queue.Queue()
        dataset = self.make_dataset(source, KeyedStore())
        read_ahead = _LeaderReadAhead(dataset, max_items=2)
        read_ahead.close()
        read_ahead._thread.join(timeout=3 * _LeaderReadAhead._POLL_SEC)
        self.assertFalse(read_ahead._thread.is_alive())
        source.put(keyed_sample(0))
        self.assertEqual(source.qsize(), 1)

    def test_requires_sp_leader_fetch(self):
        with self.assertRaisesRegex(ValueError, "requires sp_leader_mooncake_fetch"):
            MooncakeDataset(
                ray_queue=None,
                mooncake_store=None,
                device=torch.device("cpu"),
                usp_local_shard=True,
                leader_read_ahead_items=2,
            )

    def test_distributed_gloo_read_ahead(self):
        self._run_distributed(packing=False)

    def test_distributed_gloo_packing_read_ahead(self):
        self._run_distributed(packing=True)

    def _run_distributed(self, packing):
        fd, init_file = tempfile.mkstemp()
        os.close(fd)
        os.unlink(init_file)
        try:
            mp.spawn(run_distributed_read_ahead, args=(2, init_file, packing), nprocs=2, join=True)
        finally:
            if os.path.exists(init_file):
                os.unlink(init_file)


class TestTrainerLeaderReadAhead(unittest.TestCase):
    def wire(self, store=None, **args):
        defaults = dict(
            attention_backend="usp",
            usp_local_shard=True,
            sp_leader_mooncake_fetch=True,
            prefetch_depth=3,
            max_seq_length=64,
            draft_accumulation_steps=2,
            sp_ulysses_size=2,
        )
        defaults.update(args)
        trainer = SimpleNamespace(
            args=SimpleNamespace(**defaults),
            sp_group=None,
            mooncake_store=store if store is not None else object(),
            dp_rank=0,
            assistant_header_ids=None,
            end_token_ids=None,
            dynamic_loss_mask=False,
            last_turn_loss_only=False,
            skip_after_header=0,
            _sequence_shard_after_leader_fetch=lambda: False,
            _defer_batch_h2d=lambda: False,
            _position_recipe=lambda: None,
        )
        with (
            patch("angelspec.training.trainer.MooncakeDataFetcher") as fetcher,
            patch("angelspec.training.trainer.torch.cuda.current_device", return_value=0),
        ):
            Trainer.set_train_queue(trainer, object(), None, per_dp_rank_batch_size=2)
        return trainer, fetcher.call_args.kwargs

    def test_packing_reads_ahead_whole_steps_of_tokens(self):
        trainer, kwargs = self.wire(mtp_packing=True)
        self.assertEqual(kwargs["leader_read_ahead_tokens"], 3 * 2 * 64)
        self.assertEqual(kwargs["leader_read_ahead_items"], 0)
        self.assertNotIsInstance(trainer.data_fetcher, PrefetchedDataFetcher)
        self.assertEqual(kwargs["device"], 0)

    def test_unpacked_reads_ahead_batches(self):
        _, kwargs = self.wire()
        self.assertEqual(kwargs["leader_read_ahead_items"], 3 * 2)
        self.assertEqual(kwargs["leader_read_ahead_tokens"], 0)

    def test_no_read_ahead_with_gpu_direct(self):
        store = SimpleNamespace(config=SimpleNamespace(enable_gpu_direct=True))
        trainer, kwargs = self.wire(store=store)
        self.assertEqual(kwargs["leader_read_ahead_items"], 0)
        self.assertEqual(kwargs["leader_read_ahead_tokens"], 0)
        self.assertNotIsInstance(trainer.data_fetcher, PrefetchedDataFetcher)

    def test_no_read_ahead_without_prefetch_or_leader_fetch(self):
        for args in ({"prefetch_depth": 0}, {"sp_leader_mooncake_fetch": False}):
            with self.subTest(**args):
                _, kwargs = self.wire(**args)
                self.assertEqual(kwargs["leader_read_ahead_items"], 0)
                self.assertEqual(kwargs["leader_read_ahead_tokens"], 0)
