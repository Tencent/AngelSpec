"""vLLM data-parallel worlds: the inference GPUs split into worlds of dp_size*tp_size ranks."""

from types import SimpleNamespace

import pytest

from angelspec.inference import factory


class _Call:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeActor:
    def __init__(self, options, kwargs):
        self.options = options
        self.kwargs = kwargs
        self.dist_init_addr = None
        self.find_free_port = _Call(lambda **_: 12000 + kwargs["rank"])
        self.get_node_ip = _Call(lambda: f"10.0.0.{kwargs['node_rank']}")
        self.init = _Call(self._init)
        self.set_tp_group = _Call(lambda workers: workers)

    def _init(self, mooncake_config=None, dist_init_addr=None):
        self.dist_init_addr = dist_init_addr
        return self.kwargs["rank"]


class _FakeActorClass:
    def __init__(self):
        self.actors = []

    def options(self, **options):
        return _Call(lambda **kwargs: self._spawn(options, kwargs))

    def _spawn(self, options, kwargs):
        self.actors.append(_FakeActor(options, kwargs))
        return self.actors[-1]


@pytest.fixture
def actor_class(monkeypatch):
    actor_class = _FakeActorClass()
    monkeypatch.setattr(factory.ray, "remote", lambda cls: actor_class)
    monkeypatch.setattr(factory.ray, "get", lambda ref, timeout=None: ref)
    monkeypatch.setattr(factory, "PlacementGroupSchedulingStrategy", lambda **kwargs: kwargs)
    monkeypatch.setattr(factory, "get_angelspec_env_vars", lambda: {})
    monkeypatch.setattr(factory, "_vllm_config_env_vars", lambda args: {})
    monkeypatch.setattr(factory, "_alive_worker_engines", [])
    return actor_class


def _args(dp_size, tp_size, num_gpus):
    return SimpleNamespace(
        vllm_data_parallel_size=dp_size,
        vllm_tp_size=tp_size,
        inference_num_gpus=num_gpus,
        inference_num_gpus_per_node=8,
    )


@pytest.mark.parametrize("tp_size", [1, 2])
def test_each_node_is_its_own_world(actor_class, tp_size):
    dp_size = 8 // tp_size
    # Two inference nodes after 16 training bundles.
    bundles = [16 + g for g in range(16)]
    gpu_ids = list(range(8)) * 2
    heads, _ = factory._prepare_vllm_dp_engines(
        _args(dp_size, tp_size, 16), (None, bundles, gpu_ids), engine_group=1
    )
    actors = actor_class.actors
    assert [a.kwargs["rank"] for a in actors] == list(range(16))
    for g, actor in enumerate(actors):
        world, r = divmod(g, 8)
        assert actor.kwargs["world_size"] == 8
        assert actor.kwargs["world_rank"] == r
        assert actor.kwargs["node_rank"] == world
        assert actor.kwargs["local_rank"] == gpu_ids[g]
        assert (actor.kwargs["dp_rank"], actor.kwargs["tp_rank"]) == divmod(r, tp_size)
        assert actor.options["scheduling_strategy"]["placement_group_bundle_index"] == bundles[g]
        # Every world meets at its own rank0, on its own node.
        assert actor.dist_init_addr == f"10.0.0.{world}:{12000 + 8 * world}"
    ports = {a.options["runtime_env"]["env_vars"]["VLLM_PORT"] for a in actors}
    assert len(ports) == 16
    # One head per DP group, world by world.
    assert heads == actors[::tp_size]
    assert len(factory._alive_worker_engines) == 16 - len(heads)


def test_inference_gpus_must_fill_whole_worlds(actor_class):
    with pytest.raises(ValueError, match="not a multiple"):
        factory._prepare_vllm_dp_engines(
            _args(8, 1, 12), (None, list(range(12)), list(range(8)) + list(range(4)))
        )
    assert not actor_class.actors


def _engine(dp_size, tp_size, world_size, num_gpus=1, **args):
    from angelspec.inference.engine.vllm_engine import VllmEngine

    engine = VllmEngine.__new__(VllmEngine)
    engine.args = SimpleNamespace(**args)
    engine.dp_size, engine.dp_tp_size, engine.world_size = dp_size, tp_size, world_size
    engine.num_gpus_per_engine = num_gpus
    return engine


@pytest.mark.parametrize("tp_size", [1, 2])
def test_a_dp_rank_spans_its_tp_group(tp_size):
    # As the factory builds it: one GPU an actor, the world dp_size * tp_size ranks.
    engine = _engine(8 // tp_size, tp_size, 8)
    assert engine._tp_size(pp_size=1, nnodes=1) == tp_size
    with pytest.raises(ValueError, match="DP world's ranks"):
        engine._tp_size(pp_size=2, nnodes=1)


def test_a_single_replica_needs_its_whole_allocation():
    engine = _engine(1, None, None, num_gpus=8, vllm_tp_size=8)
    assert engine._tp_size(pp_size=1, nnodes=1) == 8
    engine.args.vllm_tp_size = 4
    with pytest.raises(ValueError, match="allocated GPUs"):
        engine._tp_size(pp_size=1, nnodes=1)
