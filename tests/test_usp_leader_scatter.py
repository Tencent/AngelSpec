"""The SP leader packs a DFlash row once and hands every rank its sequence shard."""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from angelspec.data.utils import MOONCAKE_DELETE_KEYS_FIELD, DFlashPackingCollator
from angelspec.training.data_fetcher import MooncakeDataset, TrainSample
from angelspec.utils import distributed as dmod


class _Tensors:
    def __init__(self, values):
        self.values = values

    def to_tensor_dict(self):
        return self.values


class _Queue:
    def __init__(self, error):
        self.error = error
        self.get_calls = 0

    def get(self, block=True, timeout=None):
        self.get_calls += 1
        raise self.error


_ROW_LENGTHS = (6, 5)
_ROW_MAX_SEQ = 16


def _row_features(index, seq_len):
    offset = 100 * index
    ids = torch.arange(offset, offset + seq_len, dtype=torch.long)
    hidden = (torch.arange(seq_len * 4) % 97 + 1 + offset).to(torch.bfloat16).reshape(seq_len, 4)
    last = (torch.arange(seq_len * 2) % 89 + 1 + offset).to(torch.bfloat16).reshape(seq_len, 2)
    return ids, hidden, last


class _RowStore:
    def __init__(self):
        self.get_calls = 0

    def get(self, key, shapes, dtypes, device):
        self.get_calls += 1
        ids, hidden, last = _row_features(int(key[-1]), shapes["input_ids"][-1])
        return _Tensors({"input_ids": ids, "hidden_states": hidden, "last_hidden_states": last})


class _RowQueue:
    def __init__(self, items):
        self.items = list(items)
        self.get_calls = 0

    def get(self, block=True, timeout=None):
        self.get_calls += 1
        return self.items.pop(0) if self.items else None


def _row_sample(index, seq_len, last_in_row):
    return TrainSample(
        mooncake_key=f"sample{index}",
        tensor_shapes={
            "input_ids": (seq_len,),
            "hidden_states": (seq_len, 4),
            "last_hidden_states": (seq_len, 2),
        },
        tensor_dtypes={
            "input_ids": "int64",
            "hidden_states": "bfloat16",
            "last_hidden_states": "bfloat16",
        },
        packed_loss_mask=f"0,{seq_len}",
        row_id=0,
        last_in_row=last_in_row,
    )


def _packed_row_worker(rank, world_size, path):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.new_group(list(range(world_size)), backend="gloo")
    dmod._DRAFT_SP_GROUP = group
    dmod._DRAFT_SP_GLOO_GROUP = group
    samples = [
        _row_sample(i, n, last_in_row=i == len(_ROW_LENGTHS) - 1)
        for i, n in enumerate(_ROW_LENGTHS)
    ]
    queue = _RowQueue(samples) if rank == 0 else _Queue(AssertionError("peer read queue"))
    store = _RowStore() if rank == 0 else None
    dataset = MooncakeDataset(
        ray_queue=queue,
        mooncake_store=store,
        device=torch.device("cpu"),
        usp_local_shard=True,
        sp_leader_mooncake_fetch=True,
        sequence_shard_after_leader_fetch=True,
        packing_enabled=True,
        max_seq_length=_ROW_MAX_SEQ,
    )
    assert dataset.scatters_packed_rows()
    packed_bin = next(dataset._iter_rows_by_id())
    assert len(packed_bin) == (len(_ROW_LENGTHS) if rank == 0 else 0)
    row = dataset.scatter_packed_row(DFlashPackingCollator(_ROW_MAX_SEQ), packed_bin)

    counts = torch.tensor(
        [queue.get_calls, store.get_calls if store is not None else 0], dtype=torch.long
    )
    dist.all_reduce(counts)
    assert counts.tolist() == [len(_ROW_LENGTHS), len(_ROW_LENGTHS)]

    features = [_row_features(i, n) for i, n in enumerate(_ROW_LENGTHS)]
    used = sum(_ROW_LENGTHS)
    pad = _ROW_MAX_SEQ - used
    full_ids = torch.cat([f[0] for f in features] + [torch.zeros(pad, dtype=torch.long)])
    full_hidden = torch.cat([f[1] for f in features] + [torch.zeros(pad, 4, dtype=torch.bfloat16)])
    full_last = torch.cat([f[2] for f in features] + [torch.zeros(pad, 2, dtype=torch.bfloat16)])
    shard = _ROW_MAX_SEQ // world_size
    local = slice(rank * shard, (rank + 1) * shard)

    assert row["_usp_sequence_sharded"] is True
    torch.testing.assert_close(row["input_ids"][0], full_ids)
    torch.testing.assert_close(row["hidden_states"][0], full_hidden[local])
    torch.testing.assert_close(row["last_hidden_states"][0], full_last[local])
    assert row["loss_mask"][0].tolist() == [1] * used + [0] * pad
    assert row["ctx_doc_ids"][0].tolist() == [0] * 6 + [1] * 5 + [-1] * pad
    delete_keys = row.get(MOONCAKE_DELETE_KEYS_FIELD)
    assert (len(delete_keys) if delete_keys else 0) == (len(_ROW_LENGTHS) if rank == 0 else 0)
    dist.destroy_process_group()


def test_packed_rows_are_collated_once_and_scattered():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "pg")
        mp.spawn(_packed_row_worker, args=(2, path), nprocs=2, join=True)
