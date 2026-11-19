"""Frozen target embeddings are neither saved to nor restored from DCP checkpoints.

They are copies of the current target's ``embed_tokens`` and are reloaded from the
target on every start; restoring them would bring back an old target's embedding
after the target changes.
"""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn

from angelspec.training import checkpoint

_TMPDIR = None


def setUpModule():
    global _TMPDIR
    if not dist.is_initialized():
        _TMPDIR = tempfile.TemporaryDirectory()
        dist.init_process_group(
            backend="gloo", init_method=f"file://{_TMPDIR.name}/pg", world_size=1, rank=0
        )


def tearDownModule():
    if _TMPDIR is not None:
        dist.destroy_process_group()
        _TMPDIR.cleanup()


class _Draft(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.embed_tokens.weight.requires_grad = False
        self.proj = nn.Linear(8, 8)


class _TrainingWrapper(nn.Module):
    """Trainers checkpoint the wrapper (e.g. ``MTPModel``), not the bare draft."""

    def __init__(self):
        super().__init__()
        self.draft_model = _Draft()


def _actor(model, checkpoint_dir, load_path=None):
    args = Namespace(
        checkpoint_dir=str(checkpoint_dir), load_path=load_path, continual_training=False
    )
    return Namespace(model=model, args=args, global_step=0)


def _saved_keys(model_dir: Path) -> set[str]:
    return set(dcp.FileSystemReader(str(model_dir)).read_metadata().state_dict_metadata)


@unittest.skipUnless(torch.cuda.is_available(), "checkpoint.save/load synchronize CUDA")
class TestCheckpointTargetEmbedding(unittest.TestCase):
    def test_only_frozen_embeddings_are_target_copies(self):
        model = _TrainingWrapper()
        self.assertEqual(
            checkpoint.frozen_target_embedding_keys(model), {"draft_model.embed_tokens.weight"}
        )
        model.draft_model.embed_tokens.weight.requires_grad = True
        self.assertEqual(checkpoint.frozen_target_embedding_keys(model), frozenset())

    def test_save_omits_frozen_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint.save(_actor(_TrainingWrapper(), tmp), step=0)
            keys = _saved_keys(Path(tmp) / "iter_0000001" / "model")
        self.assertFalse(any(k.endswith("embed_tokens.weight") for k in keys))
        self.assertTrue(any(k.endswith("draft_model.proj.weight") for k in keys))

    def test_load_keeps_current_embedding_and_restores_the_rest(self):
        torch.manual_seed(0)
        saved = _TrainingWrapper()
        resumed = _TrainingWrapper()
        current_embedding = resumed.draft_model.embed_tokens.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint.save(_actor(saved, tmp), step=0)
            payload = checkpoint.load(_actor(resumed, tmp, load_path=tmp))
        self.assertIsNotNone(payload)
        torch.testing.assert_close(resumed.draft_model.proj.weight, saved.draft_model.proj.weight)
        torch.testing.assert_close(resumed.draft_model.embed_tokens.weight, current_embedding)

    def test_load_ignores_embedding_in_legacy_checkpoint(self):
        torch.manual_seed(0)
        saved = _TrainingWrapper()
        resumed = _TrainingWrapper()
        current_embedding = resumed.draft_model.embed_tokens.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "iter_0000001" / "model"
            model_dir.mkdir(parents=True)
            # Checkpoints written before this change still carry the embedding.
            dcp.save({"model_state": checkpoint.ModelState(saved)}, checkpoint_id=str(model_dir))
            self.assertTrue(any(k.endswith("embed_tokens.weight") for k in _saved_keys(model_dir)))
            checkpoint.load(_actor(resumed, tmp, load_path=str(model_dir.parent)))
        torch.testing.assert_close(resumed.draft_model.proj.weight, saved.draft_model.proj.weight)
        torch.testing.assert_close(resumed.draft_model.embed_tokens.weight, current_embedding)


if __name__ == "__main__":
    unittest.main()
