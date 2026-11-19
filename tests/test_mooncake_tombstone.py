import ctypes
import unittest
from unittest import mock

import torch

from angelspec.transfer.mooncake.draft_store import (
    DraftMooncakeStore,
    MooncakeSampleUnavailable,
)

BASE_KEY = "k2404-b98a9bbd"
TOMBSTONE = BASE_KEY + DraftMooncakeStore.FAILED_SUFFIX


class _FakeBuffer:
    def __init__(self, nbytes):
        self._storage = (ctypes.c_byte * nbytes)()

    def size(self):
        return len(self._storage)

    def ptr(self):
        return ctypes.addressof(self._storage)


class _FakeStore:
    def __init__(self, present=(), tombstones=()):
        self.present = dict(present)
        self.tombstones = set(tombstones)
        self.puts = {}
        self.removed = []
        self.get_calls = 0

    def batch_get_buffer(self, keys):
        self.get_calls += 1
        return [self.present.get(key) for key in keys]

    def is_exist(self, key):
        return 1 if key in self.tombstones else 0

    def put(self, key, value):
        self.puts[key] = value
        self.tombstones.add(key)
        return 0

    def remove(self, key, force=False):
        self.removed.append(key)
        self.tombstones.discard(key)
        return 0


def _make_store(fake_store):
    store = DraftMooncakeStore.__new__(DraftMooncakeStore)
    store._store = fake_store
    store._ensure_initialized = lambda: None
    return store


class TestTombstonePublishing(unittest.TestCase):
    def test_failed_keys_publish_markers(self):
        fake = _FakeStore()
        _make_store(fake).mark_put_failed({BASE_KEY, "other"})
        self.assertEqual(
            set(fake.puts),
            {TOMBSTONE, "other" + DraftMooncakeStore.FAILED_SUFFIX},
        )

    def test_publish_failure_is_swallowed(self):
        fake = _FakeStore()
        fake.put = mock.Mock(side_effect=RuntimeError("store down"))
        _make_store(fake).mark_put_failed({BASE_KEY})

    def test_tensor_suffixes_collapse_to_base_keys(self):
        store = _make_store(_FakeStore())
        store._async_put_manager = mock.Mock()
        store._async_put_manager.pop_failed_keys.return_value = {
            f"{BASE_KEY}_hs",
            f"{BASE_KEY}_ids",
            "other_lhs",
        }
        self.assertEqual(store.pop_failed_put_keys(), {BASE_KEY, "other"})


class TestTombstoneConsumption(unittest.TestCase):
    def setUp(self):
        self.config = mock.Mock(
            get_retry_wait_seconds=0.01,
            get_retry_log_interval_seconds=10.0,
            get_retry_max_wait_seconds=0.05,
        )

    def _fetch(self, fake):
        store = _make_store(fake)
        store.config = self.config
        return store._get_tensors_via_host_buffer(
            [f"{BASE_KEY}_ids"],
            [("input_ids", (2,), torch.int64)],
            torch.device("cpu"),
            BASE_KEY,
        )

    def test_tombstone_aborts_without_waiting(self):
        fake = _FakeStore(tombstones=[TOMBSTONE])
        with self.assertRaises(MooncakeSampleUnavailable):
            self._fetch(fake)
        self.assertEqual(fake.get_calls, 2)
        self.assertEqual(fake.removed, [TOMBSTONE])

    def test_republished_tensors_win_over_tombstone(self):
        fake = _FakeStore(tombstones=[TOMBSTONE])

        def reappear(keys):
            fake.get_calls += 1
            return [None] if fake.get_calls == 1 else [_FakeBuffer(16)]

        fake.batch_get_buffer = reappear
        result = self._fetch(fake)
        self.assertIn("input_ids", result)
        self.assertEqual(fake.removed, [])

    def test_timeout_raises_typed_exception(self):
        with self.assertRaises(MooncakeSampleUnavailable):
            self._fetch(_FakeStore())

    def test_cpu_result_survives_mooncake_buffer_reuse(self):
        buf = _FakeBuffer(16)
        source = (ctypes.c_int64 * 2).from_address(buf.ptr())
        source[:] = [123, 456]
        result = self._fetch(_FakeStore(present={f"{BASE_KEY}_ids": buf}))
        # The allocator may recycle the buffer immediately after the get returns.
        source[:] = [-1, -1]
        self.assertEqual(result["input_ids"].tolist(), [123, 456])
