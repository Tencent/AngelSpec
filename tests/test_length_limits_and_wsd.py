"""Per-request engine length limits, prepared eval-set filtering and WSD validation."""

from types import SimpleNamespace

import pytest
import torch

from angelspec.controller.training_controller import _drop_overlong_eval_samples
from angelspec.inference.engine.vllm_engine import VllmEngine
from angelspec.training.lr_scheduler import LRSchedulerWithWarmup


def _engine(**args):
    engine = VllmEngine.__new__(VllmEngine)
    engine.rank = 0
    engine.args = SimpleNamespace(vllm_max_model_len=None, **args)
    return engine


def test_training_requests_keep_the_training_limit_when_eval_is_longer():
    engine = _engine(max_seq_length=100, eval_max_seq_length=200)
    ids = ["data_1", "eval_data_2", "data_3"]
    # 150-token prompts fit the eval limit, not the training one.
    assert engine._requests_within_length_limit(ids, [150, 150, 50]) == [1, 2]


def test_eval_requests_use_the_training_limit_without_a_longer_eval_limit():
    engine = _engine(max_seq_length=100)
    assert engine._requests_within_length_limit(["eval_a", "b"], [150, 99]) == [1]


def test_prepared_eval_samples_the_engine_would_skip_are_dropped():
    samples = [
        {"input_ids": torch.zeros(1, 99)},
        {"input_ids": torch.zeros(1, 100)},
        {"input_ids": list(range(10))},
        {"formatted_prompt": "no ids"},
    ]
    kept = _drop_overlong_eval_samples("alpha", samples, 100)
    assert [id(s) for s in kept] == [id(samples[i]) for i in (0, 2, 3)]
    assert _drop_overlong_eval_samples("alpha", samples, None) is samples


def _scheduler(style):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    return LRSchedulerWithWarmup(
        optimizer,
        max_lr=1.0,
        total_steps=10,
        decay_style="WSD",
        wsd_decay_steps=2,
        wsd_decay_style=style,
    )


def test_wsd_decay_style_is_validated_at_construction():
    with pytest.raises(ValueError, match="Unknown WSD decay style"):
        _scheduler(None)
    scheduler = _scheduler("cosine")
    for _ in range(10):
        scheduler.step()
