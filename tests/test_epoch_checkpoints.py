from types import SimpleNamespace
from unittest import mock

import pytest
from omegaconf import OmegaConf

from angelspec.config.train_config import Config, _validate_keep_epoch_checkpoints
from angelspec.controller import loop

STEPS_PER_EPOCH = 2


def remote(fn):
    return SimpleNamespace(remote=fn)


def run_epochs(checkpoint_dir, num_steps, save_interval=0, keep=True):
    """Drive training_loop over epochs of STEPS_PER_EPOCH one-batch steps, the dataset
    draining at each epoch end; return the iterations kept as epoch ends."""
    remaining = STEPS_PER_EPOCH

    def try_dispatch_batch():
        nonlocal remaining
        if remaining == 0:
            return False
        remaining -= 1
        return True

    def reload_dataset():
        nonlocal remaining
        remaining = STEPS_PER_EPOCH

    def full_status():
        return {
            "inference_speed": 0.0,
            "sample_pool_size": 0,
            "dispatch_batch_size": 8,
            "can_dispatch": remaining > 0,
            "prompt_buffer_size": 0,
            "elapsed_seconds": 0.0,
            "avg_inference_speed": 0.0,
            "avg_training_speed": 0.0,
        }

    controller = SimpleNamespace(
        get_dataset_size=remote(lambda: 64),
        get_eval_dataset_size=remote(lambda: 0),
        submit_training_dataset=remote(lambda **kw: None),
        try_dispatch_batch=remote(try_dispatch_batch),
        get_full_status=remote(full_status),
        reload_dataset=remote(reload_dataset),
    )
    actor = SimpleNamespace(
        get_global_step=remote(lambda: 0),
        train_from_queue=remote(lambda step, num_batches: {"train/avg_loss": 0.0}),
    )
    train_group = SimpleNamespace(_actor_handlers=[actor], save_model=lambda step, **kw: None)
    inference_manager = SimpleNamespace(flush_metrics=remote(lambda: {}))
    args = SimpleNamespace(
        num_train_steps=num_steps,
        draft_accumulation_steps=1,
        global_batch_size=8,
        per_dp_rank_batch_size=8,
        dp_size=1,
        steps_per_epoch=STEPS_PER_EPOCH,
        num_epochs=-(-num_steps // STEPS_PER_EPOCH),
        save_interval=save_interval,
        save_per_epoch=True,
        keep_epoch_checkpoints=keep,
        checkpoint_dir=str(checkpoint_dir),
        max_checkpoints=0,
        online_dump_enabled=False,
        enable_perf_metrics=False,
        epoch_exhaustion_grace_seconds=0.0,
    )
    eval_state = SimpleNamespace(
        eval_enabled=False,
        named_datasets=(),
        cache_eval_hidden_states=False,
        eval_cache_loaded=False,
        eval_interval=0,
        best_eval_score=None,
    )
    with (
        mock.patch.object(loop.ray, "get", side_effect=lambda value: value),
        mock.patch.object(loop.time, "sleep"),
        mock.patch.object(loop, "wandb", SimpleNamespace(run=None)),
        mock.patch.object(loop, "swanlab", None),
        mock.patch.object(loop, "get_tb_writer", return_value=None),
        mock.patch.object(loop, "setup_eval", return_value=eval_state),
        mock.patch.object(loop, "run_eval", return_value={}),
        mock.patch.object(loop, "update_checkpoint_eval_meta", return_value=None),
        mock.patch.object(loop, "infer_model_arch", return_value="dflash"),
        mock.patch.object(loop, "setup_online_eval", return_value=None),
        mock.patch.object(loop, "maybe_run_online_eval", return_value=None),
        mock.patch.object(loop, "drain_online_eval", return_value=None),
        mock.patch.object(loop, "_record_data_position"),
    ):
        loop.training_loop(args, controller, inference_manager, train_group)
    kept = checkpoint_dir / loop.EPOCH_CHECKPOINTS_FILE
    return [int(token) for token in kept.read_text().split()] if kept.exists() else []


@pytest.mark.parametrize(
    "num_steps, save_interval, kept",
    [
        (4, 0, [3, 5]),  # both epoch ends; the second is the final save
        (4, 2, [3, 5]),  # interval saves land on both epoch ends
        (3, 0, [3]),  # the final save at step 3 ends no epoch
        (2, 0, [3]),  # stopping after one epoch: the final save is its end
    ],
)
def test_epoch_end_checkpoints_are_kept(tmp_path, num_steps, save_interval, kept):
    assert run_epochs(tmp_path, num_steps, save_interval) == kept


def test_nothing_is_kept_without_the_flag(tmp_path):
    assert run_epochs(tmp_path, num_steps=4, keep=False) == []


def make_iterations(base_dir, iterations):
    for iteration in iterations:
        (base_dir / f"iter_{iteration:07d}").mkdir()


def remaining_iterations(base_dir):
    return sorted(int(d.name[len("iter_") :]) for d in base_dir.iterdir() if d.is_dir())


def test_rotation_neither_counts_nor_deletes_kept_epoch_ends(tmp_path):
    (tmp_path / loop.EPOCH_CHECKPOINTS_FILE).write_text("3862\n")
    make_iterations(tmp_path, [3001, 3501, 3862, 4001])
    loop._cleanup_old_checkpoints(str(tmp_path), 2)
    assert remaining_iterations(tmp_path) == [3501, 3862, 4001]
    make_iterations(tmp_path, [4501, 5001])
    loop._cleanup_old_checkpoints(str(tmp_path), 2)
    assert remaining_iterations(tmp_path) == [3862, 4501, 5001]


def test_rotation_deletes_nothing_when_the_kept_list_is_unreadable(tmp_path):
    make_iterations(tmp_path, [1, 2, 3])
    (tmp_path / loop.EPOCH_CHECKPOINTS_FILE).write_text("2\nnot-a-step\n")
    loop._cleanup_old_checkpoints(str(tmp_path), 1)
    assert remaining_iterations(tmp_path) == [1, 2, 3]


def test_keeping_an_epoch_end_twice_lists_it_once(tmp_path):
    loop._keep_epoch_checkpoint(str(tmp_path), 3861)
    loop._keep_epoch_checkpoint(str(tmp_path), 3861)
    assert (tmp_path / loop.EPOCH_CHECKPOINTS_FILE).read_text() == "3862\n"


def test_keep_epoch_checkpoints_needs_save_per_epoch():
    config = OmegaConf.structured(Config)
    config.training.keep_epoch_checkpoints = True
    config.training.save_per_epoch = True
    _validate_keep_epoch_checkpoints(config)
    config.training.save_per_epoch = False
    with pytest.raises(ValueError, match="save_per_epoch"):
        _validate_keep_epoch_checkpoints(config)
