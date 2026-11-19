import pytest

from angelspec.utils.packing_balance import balance_packed_rows


def test_balances_across_dispatches_without_repacking():
    lengths = [8, 2]
    rows = [[[0]], [[1]]]
    first, loads = balance_packed_rows(rows, lengths, [0, 0])
    assert first == rows and loads == [8, 2]
    second, loads = balance_packed_rows(rows, lengths, loads)
    assert second == [[[1]], [[0]]] and loads == [10, 10]
    assert rows == [[[0]], [[1]]]


def test_preserves_rows_samples_order_and_count():
    rows = [[[0, 1], [2]], [[3, 4], [5]]]
    lengths = [2, 3, 7, 1, 4, 9]
    first = balance_packed_rows(rows, lengths, [10, 0])
    assert first == balance_packed_rows(rows, lengths, [10, 0])
    assigned, loads = first
    assert all(len(group) == 2 for group in assigned)
    assert sorted(tuple(row) for group in assigned for row in group) == sorted(
        tuple(row) for group in rows for row in group
    )
    assert sum(loads) == 10 + sum(lengths)


@pytest.mark.parametrize(
    "rows,lengths,loads",
    [
        ([], [], []),
        ([[[0]], []], [1], [0, 0]),
        ([[[0]], [[0]]], [1], [0, 0]),
        ([[[0]]], [0], [0]),
        ([[[0]]], [1], [-1]),
    ],
)
def test_invalid_input(rows, lengths, loads):
    with pytest.raises(ValueError):
        balance_packed_rows(rows, lengths, loads)


def test_online_controller_balances_and_resets_at_step_boundary():
    from collections import deque
    from threading import Lock
    from types import SimpleNamespace

    from angelspec.controller.training_controller import AsyncTrainingController

    cls = AsyncTrainingController.__ray_metadata__.modified_class
    controller = cls.__new__(cls)
    controller.args = SimpleNamespace(draft_accumulation_steps=2)
    controller.dp_size = 2
    controller._pool_lock = Lock()
    controller.packing_max_seq = 8
    controller.packing_min_fill_ratio = 0.0
    controller.packing_max_wait_seconds = 5.0
    controller._packing_low_fill_since = None
    controller._packing_wait_seconds_total = 0
    controller._last_dispatch_log_time = 0
    controller.packing_balance_dp = True
    controller._packing_rank_load = [0, 0]
    controller._packing_dispatches_in_step = 0
    controller._row_id_counter = 0
    controller._epoch_consumed_samples = 0
    controller.batch_id = 0
    controller._online_dump_writers = []
    controller._training_monitor = SimpleNamespace(record=lambda n: None)
    destinations = []
    controller._enqueue_sample = lambda sample, rank, **kwargs: destinations.append(
        (sample.mooncake_key, rank, kwargs)
    )
    for dispatch in range(3):
        controller.sample_pool = deque(
            [
                SimpleNamespace(mooncake_key="long", tensor_shapes={"input_ids": (8,)}),
                SimpleNamespace(mooncake_key="short", tensor_shapes={"input_ids": (2,)}),
            ]
        )
        controller._sample_bytes = {"long": 8, "short": 2}
        controller._pool_bytes = 10
        assert controller._try_dispatch_packed()
        assert controller._pool_bytes == 0
        ranks = {key: rank for key, rank, _ in destinations[-2:]}
        assert ranks["long"] == dispatch % 2
    assert controller._packing_rank_load == [8, 2]
    assert controller._packing_dispatches_in_step == 1
    assert all(meta["last_in_row"] for _, _, meta in destinations)


def test_row_reassignment_preserves_aggregate_loss_and_gradient():
    import torch

    rows = [[[0, 1]], [[2]]]
    lengths = [2, 3, 4]
    assigned, _ = balance_packed_rows(rows, lengths, [100, 0])
    torch.manual_seed(11)
    samples = [torch.randn(length, 3) for length in lengths]
    parameters = [torch.randn(3, requires_grad=True)]
    parameters.append(parameters[0].detach().clone().requires_grad_())
    values = []
    for assignment, parameter in zip([rows, assigned], parameters):
        value = (
            sum(
                (torch.cat([samples[i] for i in row]) @ parameter).square().mean()
                for group in assignment
                for row in group
            )
            / 2
        )
        value.backward()
        values.append(value)
    torch.testing.assert_close(values[0], values[1])
    torch.testing.assert_close(parameters[0].grad, parameters[1].grad)
