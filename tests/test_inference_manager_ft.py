"""Inference-side replica fault tolerance: quarantine, requeue, degrade, escalate."""

import asyncio
from types import SimpleNamespace

import pytest
from ray.exceptions import RayActorError

from angelspec.controller.inference_manager import (
    AsyncInferenceManager,
    EnginePool,
    NoHealthyEngine,
)
from angelspec.utils.types import InferenceInput


def _make_ray_actor_error() -> RayActorError:
    try:
        return RayActorError()
    except Exception:

        class _E(RayActorError):
            def __init__(self):
                pass

        return _E()


class _Remote:
    def __init__(self, coro_factory):
        self._coro_factory = coro_factory

    def remote(self, *args, **kwargs):
        return self._coro_factory(*args, **kwargs)


class FakeEngine:
    """Engine whose generate.remote() returns a coroutine per the configured mode."""

    def __init__(self, mode="ok", delay=0.0):
        self.mode = mode
        self.delay = delay
        self.generate = _Remote(self._run)
        self.generate_with_decode = _Remote(self._run)

    async def _run(self, *, data_id, **kwargs):
        if self.mode == "actor_error":
            raise _make_ray_actor_error()
        if self.mode == "value_error":
            raise ValueError("bad batch")
        if self.mode == "timeout":
            await asyncio.sleep(self.delay)
        return [{"data_id": did, "mooncake_key": f"k-{did}"} for did in data_id]


class FakeController:
    def __init__(self):
        self.error = None
        self.eval_prompts = []
        self.set_inference_error = _Remote(self._set_error)
        self.get_eval_prompts = _Remote(self._get_eval_prompts)

    async def _set_error(self, msg):
        self.error = msg
        return None

    async def _get_eval_prompts(self, n):
        taken, self.eval_prompts = self.eval_prompts[:n], self.eval_prompts[n:]
        return taken


def _make_manager(engines, **overrides):
    args = SimpleNamespace(
        inference_buffer_threshold=32,
        inference_fetch_batch=1,
        inference_batch_size=1,
        max_sample_pool_size=0,
        max_sample_pool_bytes=0,
        defer_tokenization=True,
        compute_logits_in_trainer=True,
        train_with_decode=False,
        enable_perf_metrics=False,
        inference_ft_enabled=True,
        inference_generate_timeout_s=0.0,
        inference_quarantine_max_consec_failures=3,
        inference_min_active_engines=1,
    )
    for k, v in overrides.items():
        setattr(args, k, v)

    meta = getattr(AsyncInferenceManager, "__ray_metadata__", None)
    cls = getattr(meta, "modified_class", None) if meta else None
    if cls is None:
        pytest.skip("cannot unwrap @ray.remote AsyncInferenceManager on this Ray version")
    controller = FakeController()
    mgr = cls(args, controller, inference_engines=engines, max_concurrent_batches=1)
    return mgr, controller


def _entries(n):
    return [InferenceInput(data_id=f"d{i}", formatted_prompt=f"p{i}") for i in range(n)]


@pytest.mark.parametrize(
    "dp_size,batch_size,threshold", [(1, 1, 2), (1, 8, 2), (2, 8, 2), (2, 8, 32)]
)
def test_small_refill_threshold_still_dispatches_complete_batch(dp_size, batch_size, threshold):
    async def run():
        mgr, controller = _make_manager(
            [FakeEngine() for _ in range(dp_size)],
            inference_batch_size=batch_size,
            inference_buffer_threshold=threshold,
            vllm_data_parallel_size=dp_size,
        )
        fetched = []

        async def get_prompts(n):
            fetched.append(n)
            return _entries(n)

        controller.get_prompts = _Remote(get_prompts)
        await mgr._refill_buffer()
        await mgr._dispatch_ready_batches()
        assert len(mgr._pending_tasks) == 1
        results = await asyncio.gather(*mgr._pending_tasks)
        assert len(results[0]) == batch_size * dp_size
        assert fetched == [max(threshold, batch_size * dp_size)]

    asyncio.run(run())


# -- EnginePool (pure, no Ray) -------------------------------------------------


def test_pool_pick_skips_quarantined():
    pool = EnginePool([FakeEngine(), FakeEngine(), FakeEngine()])
    dead = pool._engines[1]
    pool.quarantine(dead, "test")
    assert pool.active_count == 2
    picks = {id(pool.pick()) for _ in range(12)}
    assert id(dead) not in picks


def test_pool_all_quarantined_raises():
    pool = EnginePool([FakeEngine(), FakeEngine()])
    for e in list(pool._engines):
        pool.quarantine(e, "test")
    assert pool.active_count == 0
    with pytest.raises(NoHealthyEngine):
        pool.pick()


def test_pool_failure_counters():
    pool = EnginePool([FakeEngine()])
    e = pool._engines[0]
    assert pool.record_failure(e) == 1
    assert pool.record_failure(e) == 2
    pool.record_success(e)
    assert pool.consec_failures[0] == 0


def test_pool_acquire_reuses_free_engine_without_queueing_behind_busy_peer():
    async def run():
        engines = [FakeEngine(), FakeEngine()]
        pool = EnginePool(engines, max_concurrent_per_engine=1)
        first = await pool.acquire()
        second = await pool.acquire()
        assert first is engines[0]
        assert second is engines[1]

        waiter = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0)
        assert not waiter.done()

        await pool.release(second)
        assert await asyncio.wait_for(waiter, timeout=1) is engines[1]
        await pool.release(first)
        await pool.release(engines[1])

    asyncio.run(run())


# -- _dispatch_batch fault handling -------------------------------------------


def test_dispatch_actor_error_quarantines_and_requeues():
    engines = [FakeEngine("actor_error"), FakeEngine("ok")]
    mgr, controller = _make_manager(engines)
    entries = _entries(2)
    result = asyncio.run(mgr._dispatch_batch(entries))
    assert result == []
    assert mgr._engines.active_count == 1
    assert [e.data_id for e in mgr._prompt_buffer] == ["d0", "d1"]
    assert controller.error is None  # still 1 replica alive -> not fatal


def test_dispatch_escalates_below_min_active():
    mgr, controller = _make_manager([FakeEngine("actor_error")])
    asyncio.run(mgr._dispatch_batch(_entries(1)))
    assert mgr._engines.active_count == 0
    assert mgr._running is False
    assert controller.error is not None


def test_dispatch_timeout_quarantines():
    engines = [FakeEngine("timeout", delay=5.0), FakeEngine("ok")]
    mgr, _ = _make_manager(engines, inference_generate_timeout_s=0.05)
    result = asyncio.run(mgr._dispatch_batch(_entries(1)))
    assert result == []
    assert mgr._engines.active_count == 1


def test_ft_disabled_preserves_fatal():
    mgr, controller = _make_manager([FakeEngine("actor_error")], inference_ft_enabled=False)
    with pytest.raises(RayActorError):
        asyncio.run(mgr._dispatch_batch(_entries(1)))
    assert mgr._running is False
    assert controller.error is not None


def test_generic_exception_does_not_drain_pool():
    engines = [FakeEngine("value_error"), FakeEngine("ok")]
    mgr, controller = _make_manager(engines, inference_quarantine_max_consec_failures=3)
    result = asyncio.run(mgr._dispatch_batch(_entries(1)))
    assert len(result) == 1 and isinstance(result[0][1], ValueError)
    assert mgr._engines.active_count == 2  # below threshold -> no quarantine
    assert controller.error is None
