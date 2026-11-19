import asyncio
from types import SimpleNamespace

import pytest

from angelspec.utils.types import InferenceInput
from tests.test_inference_manager_ft import FakeEngine, _entries, _make_manager, _Remote


def _eval_entries(n):
    return [InferenceInput(data_id=f"eval_e{i}", formatted_prompt=f"q{i}") for i in range(n)]


def _fill_pool(controller, size):
    async def pool_size():
        return size

    async def pool_bytes():
        return 0

    controller.get_pool_size = _Remote(pool_size)
    controller.get_pool_bytes = _Remote(pool_bytes)


def test_eval_prompts_skip_full_pool_and_buffered_training():
    async def run():
        manager, controller = _make_manager([FakeEngine(), FakeEngine()], max_sample_pool_size=4)
        _fill_pool(controller, 4)
        started = []

        async def dispatch(entries):
            started.extend(entry.data_id for entry in entries)
            return []

        manager._dispatch_batch = dispatch
        manager._prompt_buffer.extend(_entries(3))
        controller.eval_prompts = _eval_entries(1)
        dispatching = asyncio.create_task(manager._dispatch_ready_batches())
        await asyncio.sleep(0.05)
        assert started == ["eval_e0"]
        assert [entry.data_id for entry in manager._prompt_buffer] == ["d0", "d1", "d2"]
        manager._running = False
        await asyncio.wait_for(dispatching, timeout=2)

    asyncio.run(run())


def test_pool_wait_yields_to_new_eval_prompts():
    async def run():
        manager, controller = _make_manager([FakeEngine()], max_sample_pool_size=4)
        _fill_pool(controller, 4)
        waiting = asyncio.create_task(manager._await_pool_capacity())
        await asyncio.sleep(0.1)
        assert not waiting.done()
        controller.eval_prompts = _eval_entries(2)
        assert await asyncio.wait_for(waiting, timeout=2) is False
        # One batch per free engine slot is staged; the rest stays with the controller.
        assert [entry.data_id for entry in manager._eval_buffer] == ["eval_e0"]
        assert [entry.data_id for entry in controller.eval_prompts] == ["eval_e1"]

    asyncio.run(run())


def test_pool_wait_yields_to_finished_dispatches():
    async def run():
        manager, controller = _make_manager([FakeEngine()], max_sample_pool_size=4)
        _fill_pool(controller, 4)
        manager._pending_tasks = {asyncio.create_task(asyncio.sleep(0.1, result=[]))}
        assert await asyncio.wait_for(manager._await_pool_capacity(), timeout=2) is False
        await asyncio.gather(*manager._pending_tasks)

    asyncio.run(run())


def test_requeue_returns_eval_and_training_prompts_to_their_own_buffers():
    manager, _ = _make_manager([FakeEngine()])
    manager._prompt_buffer.extend(_entries(3)[2:])
    manager._requeue(_entries(2) + _eval_entries(1))
    assert [entry.data_id for entry in manager._prompt_buffer] == ["d0", "d1", "d2"]
    assert [entry.data_id for entry in manager._eval_buffer] == ["eval_e0"]


def test_collect_does_not_block_idle_replica_after_refill():
    async def run():
        manager, _ = _make_manager([FakeEngine(), FakeEngine()], inference_buffer_threshold=1)
        release = asyncio.Event()
        started = []
        forwarded = []

        async def dispatch(entries):
            started.append(entries[0].data_id)
            await release.wait()
            return entries

        async def forward(entries):
            forwarded.extend(entries)

        manager._dispatch_batch = dispatch
        manager._forward_results = forward
        manager._prompt_buffer.extend(_entries(1))
        await manager._dispatch_ready_batches()
        await asyncio.sleep(0)
        manager._prompt_buffer.extend(_entries(2)[1:])
        await asyncio.wait_for(manager._collect_completed(), timeout=0.5)
        await manager._dispatch_ready_batches()
        await asyncio.sleep(0)
        assert started == ["d0", "d1"]
        release.set()
        while manager._pending_tasks:
            await manager._collect_completed()
        assert sorted(entry.data_id for entry in forwarded) == ["d0", "d1"]

    asyncio.run(run())


def test_status_counts_prompts_and_results_in_transit():
    async def run():
        manager, controller = _make_manager([FakeEngine()])
        release = asyncio.Event()
        keys = ("prompt_buffer_size", "pending_tasks", "in_transit")
        seen = []

        async def snapshot():
            status = await manager.get_status()
            seen.append(tuple(status[k] for k in keys))
            await release.wait()

        async def get_prompts(n):
            await snapshot()
            return _entries(1)

        async def forward(results):
            await snapshot()

        controller.get_prompts = _Remote(get_prompts)
        manager._forward_results = forward
        manager._pending_tasks = {asyncio.create_task(asyncio.sleep(0, result=[]))}
        for step in (manager._refill_buffer(), manager._collect_completed()):
            release.clear()
            moving = asyncio.create_task(step)
            await asyncio.sleep(0.05)
            release.set()
            await asyncio.wait_for(moving, timeout=2)
        assert seen == [(0, 1, 1), (1, 0, 1)]
        assert (await manager.get_status())["in_transit"] == 0

    asyncio.run(run())


def test_dp_wave_still_waits_for_completion():
    async def run():
        manager, _ = _make_manager([FakeEngine()], vllm_data_parallel_size=2)
        manager._pending_tasks = {asyncio.create_task(asyncio.sleep(0.1, result=[]))}
        collecting = asyncio.create_task(manager._collect_completed())
        await asyncio.sleep(0.02)
        assert not collecting.done()
        collecting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await collecting
        await asyncio.gather(*manager._pending_tasks)

    asyncio.run(run())


def test_dp_worlds_take_waves_independently():
    async def run():
        heads = [FakeEngine() for _ in range(4)]
        manager, _ = _make_manager(heads, vllm_data_parallel_size=2)
        release = {0: asyncio.Event(), 1: asyncio.Event()}
        started = []
        forwarded = []

        async def wave(world_heads, shards):
            world = heads.index(world_heads[0]) // 2
            started.append((world, sorted(e.data_id for shard in shards for e in shard)))
            await release[world].wait()
            return [(e, {}) for shard in shards for e in shard]

        async def forward(results):
            forwarded.extend(e.data_id for e, _ in results)

        manager._run_dp_wave = wave
        manager._forward_results = forward
        manager._prompt_buffer.extend(_entries(6))
        await manager._dispatch_ready_batches()
        await asyncio.sleep(0)
        # Each world takes a wave of dp_size * inference_batch_size prompts.
        assert started == [(0, ["d0", "d1"]), (1, ["d2", "d3"])]
        release[1].set()
        await asyncio.wait_for(manager._collect_completed(), timeout=0.5)
        await manager._dispatch_ready_batches()
        await asyncio.sleep(0)
        # The world that finished takes the next wave while the other still runs.
        assert started[-1] == (1, ["d4", "d5"])
        release[0].set()
        while manager._pending_tasks:
            await manager._collect_completed()
        assert sorted(forwarded) == [f"d{i}" for i in range(6)]

    asyncio.run(run())


def test_dp_eval_waves_spread_over_worlds_without_training_prompts():
    async def run():
        manager, controller = _make_manager(
            [FakeEngine() for _ in range(4)],
            vllm_data_parallel_size=2,
            inference_batch_size=4,
            inference_eval_batch_size=1,
        )
        release = asyncio.Event()
        started = []

        async def wave(world_heads, shards):
            started.append(sorted(e.data_id for shard in shards for e in shard))
            await release.wait()
            return []

        manager._run_dp_wave = wave
        manager._prompt_buffer.extend(_entries(8))
        controller.eval_prompts = _eval_entries(3)
        await manager._dispatch_ready_batches()
        await asyncio.sleep(0)
        assert started == [["eval_e0", "eval_e1"], ["eval_e2"]]
        assert [entry.data_id for entry in manager._prompt_buffer] == [f"d{i}" for i in range(8)]
        release.set()
        await asyncio.gather(*manager._pending_tasks)

    asyncio.run(run())


def test_independent_target_length():
    from angelspec.inference.engine.vllm_engine import _target_max_model_len

    args = SimpleNamespace(max_seq_length=1048576, vllm_max_model_len=262144)
    assert _target_max_model_len(args) == 262144
    args.vllm_max_model_len = None
    assert _target_max_model_len(args) == 1048576
    for value in (0, True, 1.5):
        args.vllm_max_model_len = value
        with pytest.raises(ValueError):
            _target_max_model_len(args)
