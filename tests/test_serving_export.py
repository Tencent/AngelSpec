"""Serving export has one writer even when several ranks share dp_rank 0 (USP)."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from angelspec.training.trainer import Trainer


class _Draft(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.embed_tokens.weight.requires_grad_(False)
        self.proj = nn.Linear(4, 4)


def _export_worker(rank, world_size, init_file, out_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    real_save = torch.save
    saves = []

    def recording_save(obj, path, *args, **kwargs):
        saves.append(path)
        return real_save(obj, path, *args, **kwargs)

    try:
        torch.manual_seed(0)
        torch.save = recording_save
        # Every rank of the first USP sequence-parallel group reports dp_rank 0.
        Trainer.save_draft_model_for_serving(
            SimpleNamespace(draft_model=_Draft(), dp_rank=0), out_dir
        )
    finally:
        torch.save = real_save
        dist.destroy_process_group()
    real_save(saves, os.path.join(out_dir, f"saves_rank{rank}.pt"))


def test_only_global_rank_zero_writes_the_export(tmp_path):
    out_dir = tmp_path / "hf_final"
    out_dir.mkdir()
    mp.spawn(_export_worker, args=(2, str(tmp_path / "pg"), str(out_dir)), nprocs=2, join=True)
    assert torch.load(out_dir / "saves_rank1.pt") == []
    assert len(torch.load(out_dir / "saves_rank0.pt")) == 1
    state = torch.load(out_dir / "pytorch_model.bin")
    assert set(state) == {"proj.weight", "proj.bias"}
    assert not [name for name in os.listdir(out_dir) if ".tmp." in name]
