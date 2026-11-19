import unittest
from unittest import mock

from angelspec.transfer.mooncake import buffers
from angelspec.transfer.mooncake.buffers import PUT_RETRY_ATTEMPTS, AsyncPutManager


class _FakeStore:
    def __init__(self, script):
        self._script = list(script)
        self.put_calls = 0
        self.removed_batches = []

    def batch_put_from(self, keys, buffer_ptrs, sizes, config=None):
        self.put_calls += 1
        codes = self._script.pop(0) if self._script else 0
        return [codes] * len(keys) if isinstance(codes, int) else list(codes)

    def batch_remove(self, keys, force=False):
        self.removed_batches.append(list(keys))


class TestAsyncPutRetry(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(buffers.time, "sleep")
        self.addCleanup(patcher.stop)
        patcher.start()

    def _run_put(self, store, keys=("k_hs", "k_ids")):
        manager = AsyncPutManager(store)
        self.addCleanup(manager.shutdown)
        manager._do_put(list(keys), [0, 1], [8, 8])
        return manager

    def test_transient_failure_is_retried_and_succeeds(self):
        store = _FakeStore([-800, 0])
        manager = self._run_put(store)
        self.assertEqual(store.put_calls, 2)
        self.assertEqual(manager.pop_failed_keys(), set())

    def test_retry_republishes_whole_batch_after_cleanup(self):
        store = _FakeStore([[0, -800], 0])
        self._run_put(store)
        self.assertEqual(store.removed_batches, [["k_hs", "k_ids"]])
        self.assertEqual(store.put_calls, 2)

    def test_persistent_failure_records_keys_and_raises(self):
        store = _FakeStore([-800] * PUT_RETRY_ATTEMPTS)
        manager = AsyncPutManager(store)
        self.addCleanup(manager.shutdown)
        with self.assertRaises(RuntimeError):
            manager._do_put(["k_hs", "k_ids"], [0, 1], [8, 8])
        self.assertEqual(store.put_calls, PUT_RETRY_ATTEMPTS)
        self.assertEqual(manager.pop_failed_keys(), {"k_hs", "k_ids"})
        self.assertEqual(manager.pop_failed_keys(), set())

    def test_failed_keys_surface_through_submit(self):
        store = _FakeStore([-800] * PUT_RETRY_ATTEMPTS)
        manager = AsyncPutManager(store)
        self.addCleanup(manager.shutdown)
        manager.submit(["k_hs"], [0], [8], owner_buffer_ptr=0)
        manager.drain()
        self.assertEqual(manager.pop_failed_keys(), {"k_hs"})

    def test_failure_does_not_raise_into_the_next_put_on_its_buffer(self):
        store = _FakeStore([-800] * PUT_RETRY_ATTEMPTS + [0])
        manager = AsyncPutManager(store)
        self.addCleanup(manager.shutdown)
        manager.submit(["a_hs"], [0], [8], owner_buffer_ptr=0)
        manager.wait_for_buffer(0)  # the next sample reuses buffer 0
        manager.submit(["b_hs"], [0], [8], owner_buffer_ptr=0)
        manager.drain()
        self.assertEqual(manager.pop_failed_keys(), {"a_hs"})

    def test_store_exception_records_keys(self):
        class _Raising(_FakeStore):
            def batch_put_from(self, *args, **kwargs):
                raise ConnectionError("transfer engine down")

        manager = AsyncPutManager(_Raising([]))
        self.addCleanup(manager.shutdown)
        manager.submit(["k_hs", "k_ids"], [0, 1], [8, 8], owner_buffer_ptr=0)
        manager.drain()
        self.assertEqual(manager.pop_failed_keys(), {"k_hs", "k_ids"})
