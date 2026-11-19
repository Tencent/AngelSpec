from argparse import Namespace
from unittest import mock

import pytest

from angelspec.config.eval_config import named_eval_datasets
from angelspec.config.train_config import config_to_flat_args, load_config
from angelspec.controller.eval import run_eval, setup_eval


def eval_args(tmp_path, cache=True):
    return Namespace(
        eval_datasets={"alpha": str(tmp_path / "a.jsonl"), "beta": str(tmp_path / "b.jsonl")},
        eval_data_path=None,
        eval_interval=5,
        cache_eval_hidden_states=cache,
        dp_size=2,
        checkpoint_dir=None,
        cache_dir=str(tmp_path / "cache"),
        inference_batch_size=2,
        eval_seed=42,
        micro_batch_size=1,
    )


class Remote:
    def __init__(self, function):
        self.remote = function


class SimController:
    def __init__(self, infos):
        self.infos = infos
        self.active = None
        self.get_named_eval_datasets = Remote(lambda: infos)
        self.select_eval_dataset = Remote(self.select)
        self.submit_eval_chunk = Remote(lambda start, end: end - start)
        self.try_dispatch_eval_batch = Remote(lambda: True)
        self.finalize_eval_dispatch = Remote(lambda: None)
        self.try_dispatch_batch = Remote(lambda: True)

    def select(self, name):
        self.active = name
        return next(info["dispatch_samples"] for info in self.infos if info["name"] == name)


class SimTrainGroup:
    def __init__(self, controller, cache_hit=False, fail=None):
        self.controller = controller
        self.cache_hit = cache_hit
        self.count = 0
        self.clear_calls = 0
        self.loads, self.saves, self.runs = [], [], []
        self.fail = fail

    def clear_eval_cache(self):
        self.clear_calls += 1
        self.count = 0

    def load_eval_cache(self, path):
        self.loads.append(path)
        if self.cache_hit:
            info = next(
                info for info in self.controller.infos if info["name"] == self.controller.active
            )
            self.count = info["dispatch_samples"] // 2
            return [self.count] * 2
        return [0, 0]

    def cache_eval_samples(self, count):
        self.count += count
        return [self.count] * 2

    def async_save_eval_cache(self, path):
        self.saves.append(path)

    def run_named_eval(self, seed):
        self.runs.append((self.controller.active, seed))
        if self.controller.active == self.fail:
            raise RuntimeError("intentional failure")
        value = {"alpha": 1.0, "beta": 3.0}[self.controller.active]
        return [
            {
                "eval/avg_loss": value,
                "eval/avg_acc": value / 4,
                "eval/simulated_acc_len": value / 2,
            }
        ] * 2


def infos_for(args):
    return [
        {
            "name": name,
            "path": path,
            "samples": n,
            "dispatch_samples": n + n % 2,
            "padding_samples": n % 2,
        }
        for (name, path), n in zip(named_eval_datasets(args), (1, 3))
    ]


def test_config_relative_paths_names_and_legacy(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    file = config_dir / "run.yaml"
    file.write_text(
        "dataset:\n  eval_datasets:\n    数学.v1: ./math.jsonl\n    code: org/dataset\n"
    )
    monkeypatch.chdir(tmp_path)
    args = config_to_flat_args(load_config(str(file)))
    assert dict(named_eval_datasets(args)) == {
        "数学.v1": str(config_dir / "math.jsonl"),
        "code": "org/dataset",
    }
    args = config_to_flat_args(load_config(cli_args=["dataset.eval_datasets.code=./code.jsonl"]))
    assert named_eval_datasets(args) == (("code", str(tmp_path / "code.jsonl")),)
    assert named_eval_datasets(config_to_flat_args(load_config())) == ()
    assert named_eval_datasets(Namespace(eval_data_path="/tmp/eval.jsonl")) == ()


@pytest.mark.parametrize(
    "names", [{"../x": "file"}, {"": "file"}, {"a/b": "file"}, {"a": ""}, {"a": 1}, ["file"]]
)
def test_invalid_names_and_paths(names):
    with pytest.raises(ValueError):
        named_eval_datasets(Namespace(eval_datasets=names))


def test_conflicting_and_offline_configuration():
    with pytest.raises(ValueError, match="only one"):
        load_config(
            cli_args=[
                "dataset.eval_data_path=/tmp/a.jsonl",
                "dataset.eval_datasets.a=/tmp/a.jsonl",
            ]
        )
    with pytest.raises(ValueError, match="online"):
        load_config(
            cli_args=[
                "offline.manifest_path=/tmp/manifest",
                "dataset.eval_datasets.a=/tmp/a.jsonl",
            ]
        )


@pytest.mark.parametrize("cache,hit", [(True, False), (True, True), (False, False)])
def test_sequential_eval_named_metrics_cache_and_macro_mean(tmp_path, cache, hit):
    args = eval_args(tmp_path, cache)
    infos = infos_for(args)
    controller = SimController(infos)
    group = SimTrainGroup(controller, hit)
    with (
        mock.patch("angelspec.controller.eval.ray.get", side_effect=lambda value: value),
        mock.patch("angelspec.controller.eval.wandb.run", None),
        mock.patch("angelspec.controller.eval.swanlab", None),
        mock.patch("angelspec.utils.logging.get_tb_writer", return_value=None),
    ):
        state = setup_eval(controller, group, args, 6)
        assert state.named_datasets and not state.eval_cache_loaded
        result = run_eval(10, group, True, controller=controller, eval_state=state, args=args)
    assert result["eval/alpha/avg_loss"] == 1
    assert result["eval/beta/avg_loss"] == 3
    assert result["eval/avg_loss"] == 2
    assert result["eval/alpha/samples"] == 1
    assert result["eval/beta/samples"] == 3
    assert result["eval/dataset_count"] == 2
    assert result["eval/step"] == 10
    assert [name for name, _ in group.runs] == ["alpha", "beta"]
    assert group.runs[0][1] != group.runs[1][1]
    assert group.count == 0
    assert len(group.loads) == (2 if cache else 0)
    assert len(group.saves) == (2 if cache and not hit else 0)
    if cache:
        assert group.loads[0] != group.loads[1]


def test_logging_names_and_checkpoint_metadata(tmp_path):
    from angelspec.controller.eval import update_checkpoint_eval_meta

    args = eval_args(tmp_path)
    controller = SimController(infos_for(args))
    group = SimTrainGroup(controller, cache_hit=True)
    writer = mock.Mock()
    with (
        mock.patch("angelspec.controller.eval.ray.get", side_effect=lambda value: value),
        mock.patch("angelspec.controller.eval.wandb") as wandb,
        mock.patch("angelspec.controller.eval.swanlab", None),
        mock.patch("angelspec.utils.logging.get_tb_writer", return_value=writer),
    ):
        state = setup_eval(controller, group, args, 6)
        result = run_eval(4, group, True, controller=controller, eval_state=state, args=args)
    assert wandb.log.call_args.args[0]["eval/beta/avg_loss"] == 3
    assert mock.call("eval/alpha/avg_loss", 1, 4) in writer.add_scalar.call_args_list
    checkpoint = tmp_path / "checkpoints"
    with (
        mock.patch(
            "angelspec.controller.eval._read_checkpoint_metadata", return_value={"step": 5}
        ),
        mock.patch("angelspec.controller.eval._write_checkpoint_metadata") as write,
    ):
        checkpoint.mkdir()
        best = update_checkpoint_eval_meta(str(checkpoint), 4, result, -1)
    assert best == result["eval/simulated_acc_len"]
    assert write.call_args_list[0].args[1]["eval/beta/avg_loss"] == 3
