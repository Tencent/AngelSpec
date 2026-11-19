from unittest import mock

import torch

from angelspec.data.utils import DataCollatorWithPadding
from angelspec.training.data_fetcher import MooncakeDataset
from angelspec.training.dflash_trainer import DFlashTrainer
from angelspec.utils.usp import slice_usp_sequence_shard


def test_sequence_shard_slices_before_device_transfer():
    tensor = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    rank0 = slice_usp_sequence_shard(
        tensor,
        sp_rank=0,
        sp_size=2,
        padded_seq_len=6,
    )
    rank1 = slice_usp_sequence_shard(
        tensor,
        sp_rank=1,
        sp_size=2,
        padded_seq_len=6,
    )
    torch.testing.assert_close(rank0, tensor[:, :3])
    torch.testing.assert_close(rank1[:, :2], tensor[:, 3:5])
    assert torch.equal(rank1[:, 2], torch.zeros_like(rank1[:, 2]))
    assert rank0.is_contiguous()
    assert rank1.is_contiguous()


def test_leader_scatter_sequence_slice_handles_padding():
    tensor = torch.arange(5 * 3).reshape(5, 3)
    rank0 = MooncakeDataset._slice_sequence_tensor(
        tensor,
        seq_dim=0,
        start=0,
        shard_len=3,
    )
    rank1 = MooncakeDataset._slice_sequence_tensor(
        tensor,
        seq_dim=0,
        start=3,
        shard_len=3,
    )
    torch.testing.assert_close(rank0, tensor[:3])
    torch.testing.assert_close(rank1[:2], tensor[3:5])
    assert torch.equal(rank1[2], torch.zeros_like(rank1[2]))


def test_leader_scatter_collator_keeps_full_metadata_and_local_hidden():
    collator = DataCollatorWithPadding(
        usp_sequence_sharded=True,
        sp_size=2,
    )
    feature = {
        "input_ids": torch.arange(5).unsqueeze(0),
        "loss_mask": torch.ones(1, 5, dtype=torch.long),
        "hidden_states": torch.ones(1, 128, 6),
        "last_hidden_states": torch.ones(1, 128, 2),
    }

    batch = collator([feature])

    assert batch["input_ids"].shape == (1, 256)
    assert batch["loss_mask"].shape == (1, 256)
    assert batch["hidden_states"].shape == (1, 128, 6)
    assert batch["last_hidden_states"].shape == (1, 128, 2)
    assert batch["_usp_sequence_sharded"] is True


def test_teacher_rows_all_to_all_uses_local_hidden_shards():
    trainer = object.__new__(DFlashTrainer)
    trainer.block_size = 2
    trainer.sp_group = object()
    trainer._sp_size = lambda: 2
    trainer._sp_rank = lambda: 0
    full_hidden = torch.arange(1 * 8 * 3).reshape(1, 8, 3)
    global_anchors = torch.tensor([[1, 3, 5, 7]])

    def all_to_all(output, input, **kwargs):
        output.copy_(input[: output.shape[0]])

    with (mock.patch("torch.distributed.all_to_all_single", side_effect=all_to_all),):
        rows = trainer._gather_sharded_teacher_rows(
            full_hidden[:, :4],
            global_anchors,
            padded_seq_len=8,
        )

    assert rows.shape == (1, 2, 2, 3)
