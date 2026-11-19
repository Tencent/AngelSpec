"""Training flags left unset resolve from the run's setup; explicit values are kept."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from angelspec.models.ops.loss import mtp_loss_chunk_rows
from angelspec.train_entry import _resolve_default_flags

ROOT = Path(__file__).resolve().parents[1]
MTP = str(ROOT / "angelspec/config/mtp_hy_v4_flash_draft_config.json")
EAGLE3 = str(ROOT / "configs/draft_models/qwen3_8b_eagle3.json")


def _resolved(**overrides):
    args = SimpleNamespace(
        attention_backend="usp",
        inference_engine_type="vllm",
        offline_manifest_path=None,
        draft_model_config=MTP,
        usp_local_shard=None,
        sp_leader_mooncake_fetch=None,
        gradient_checkpointing=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    _resolve_default_flags(args)
    return args.usp_local_shard, args.sp_leader_mooncake_fetch, args.gradient_checkpointing


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, (True, True, True)),
        (dict(inference_engine_type="sgl"), (False, False, True)),
        (dict(attention_backend="sdpa"), (False, False, True)),
        (dict(offline_manifest_path="/data/manifest.json"), (True, False, True)),
        (dict(draft_model_config=EAGLE3), (True, False, False)),
        (dict(draft_model_config=None), (True, False, False)),
        (dict(usp_local_shard=False), (False, False, True)),
    ],
)
def test_unset_flags_follow_the_run(overrides, expected):
    assert _resolved(**overrides) == expected


def test_explicit_flags_are_kept():
    assert _resolved(sp_leader_mooncake_fetch=False, gradient_checkpointing=False) == (
        True,
        False,
        False,
    )
    assert _resolved(draft_model_config=EAGLE3, gradient_checkpointing=True)[2] is True


def test_mtp_loss_chunk_defaults_to_4096_rows(monkeypatch):
    monkeypatch.delenv("ANGELSPEC_MTP_LOSS_CHUNK", raising=False)
    assert mtp_loss_chunk_rows() == 4096
    monkeypatch.setenv("ANGELSPEC_MTP_LOSS_CHUNK", "0")
    assert mtp_loss_chunk_rows() == 0
