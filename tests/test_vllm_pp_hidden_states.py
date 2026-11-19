import os
from types import SimpleNamespace
from unittest import mock

import torch

from angelspec.inference.engine.mooncake_hidden_states_connector import (
    MooncakeHiddenStatesConnector,
)
from angelspec.inference.engine.vllm_engine import VllmEngine
from angelspec.inference.engine.vllm_runtime_patch import (
    _build_enhanced_classes,
    _hyv4_flash_inner_forward,
)


class _HC:
    """An iHC boundary over a two-channel stream whose pre-gate sums the channels."""

    def prepare_input(self, hidden):
        return hidden if hidden.dim() == 3 else hidden.unsqueeze(1).repeat(1, 2, 1)

    def hc_pre(self, stream):
        return stream.sum(1), None


class _Layer:
    def __init__(self):
        self.hc_attn_layer = _HC()
        self.engram_layer = lambda stream, hashes: stream + 100
        self.self_attn = SimpleNamespace(mla_attn=SimpleNamespace(layer_name="layers.0.attn"))

    def __call__(self, positions, hidden, residual):
        return self.hc_attn_layer.prepare_input(hidden) + 1, None


class _Flash:
    enable_ihc = True
    engram_hash = None
    config = SimpleNamespace(engram_layer_ids=None)
    hc_head = staticmethod(lambda stream: stream.sum(1) * 10)
    norm = staticmethod(lambda x: -x)

    def __init__(self, aux):
        self.aux_hidden_state_layers = aux
        self.layers = [_Layer(), _Layer(), _Layer()]

    @staticmethod
    def embed_input_ids(input_ids):
        return input_ids.float().unsqueeze(-1)


def test_hy_v4_flash_captures_pre_gate_inputs_and_hc_head_in_requested_order():
    hidden, aux = _hyv4_flash_inner_forward(_Flash((3, 0, 2)), torch.tensor([1, 2]), None)
    # Layers add 1 to both channels; the final boundary is hc_head, the output its norm.
    torch.testing.assert_close(aux[0], torch.tensor([[80.0], [100.0]]))
    torch.testing.assert_close(aux[1], torch.tensor([[1.0], [2.0]]))
    torch.testing.assert_close(aux[2], torch.tensor([[6.0], [8.0]]))
    torch.testing.assert_close(hidden, torch.tensor([[-80.0], [-100.0]]))


def test_hy_v4_flash_capture_precedes_the_engram_residual():
    model = _Flash((1, 3))
    model.config = SimpleNamespace(engram_layer_ids=[2])
    model.engram_hash = SimpleNamespace(
        offsets=torch.zeros(1, 3), prepare_hashes_with_output=mock.Mock()
    )
    ids = torch.tensor([1, 2])
    hidden, aux = _hyv4_flash_inner_forward(model, ids, None, lookback_token_ids="lookback")
    torch.testing.assert_close(aux[0], torch.tensor([[4.0], [6.0]]))
    torch.testing.assert_close(aux[1], torch.tensor([[2080.0], [2100.0]]))
    torch.testing.assert_close(hidden, -aux[1])
    call = model.engram_hash.prepare_hashes_with_output.call_args.args
    assert call[0] is ids and call[1] == "lookback" and call[2] == "layers.0.attn"
    assert tuple(call[3].shape) == (1, 2, 3)


def test_enhanced_class_swaps_model_cls_and_keeps_the_causal_forward():
    class Model(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.make_empty_intermediate_tensors = None

    class ForCausalLM(torch.nn.Module):
        model_cls = Model

        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.model = self.model_cls(vllm_config=vllm_config, prefix=prefix)

        def forward(self, input_ids, positions, lookback_token_ids=None):
            return self.model(input_ids, positions, lookback_token_ids)

    module = SimpleNamespace(Model=Model, ForCausalLM=ForCausalLM)
    enhanced = _build_enhanced_classes(
        module, "ForCausalLM", "Model", lambda self, *args: args[-1], keep_causal_forward=True
    )
    assert enhanced.forward is ForCausalLM.forward
    lm = enhanced(vllm_config=SimpleNamespace(speculative_config=None))
    assert type(lm.model).__name__ == "Eagle3Model"
    assert lm(1, 2, lookback_token_ids="lookback") == "lookback"


def test_non_last_pp_connector_skips_missing_cache_only_layer():
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._vllm_config = object()
    connector._ensure_mooncake_store = mock.Mock()
    pp = SimpleNamespace(is_last_rank=False)
    with (
        mock.patch(
            "angelspec.inference.engine.mooncake_hidden_states_connector."
            "get_layers_from_vllm_config",
            return_value={},
        ),
        mock.patch("vllm.distributed.get_pp_group", return_value=pp),
    ):
        connector.register_kv_caches({})
    assert connector.cache_layers == []
    connector._ensure_mooncake_store.assert_not_called()


def test_vllm_env_vars_are_exported_for_ray_workers():
    engine = VllmEngine.__new__(VllmEngine)
    engine.rank = 0
    engine.args = SimpleNamespace(
        vllm_env_vars={
            "ANGELSPEC_VLLM_HS_CHUNK_TOKENS": 16384,
            "VLLM_ENGINE_READY_TIMEOUT_S": 4800,
        }
    )
    with mock.patch.dict(os.environ, {}, clear=True):
        engine._apply_env_vars()
        assert os.environ["ANGELSPEC_VLLM_HS_CHUNK_TOKENS"] == "16384"
        assert os.environ["VLLM_ENGINE_READY_TIMEOUT_S"] == "4800"
        assert os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == "ANGELSPEC_VLLM_HS_CHUNK_TOKENS"


def test_the_scheduler_ignores_extract_hidden_states_drafts_across_stages(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    from angelspec.inference.engine import vllm_runtime_patch as patch

    taken = []
    monkeypatch.setattr(
        Scheduler, "update_draft_token_ids", lambda self, drafts: taken.append(drafts)
    )
    monkeypatch.setattr(Scheduler, "_angelspec_pp_no_drafts", False, raising=False)
    assert patch.apply_extract_hidden_states_pp_no_drafts()

    def scheduler(method, pp):
        return SimpleNamespace(
            vllm_config=SimpleNamespace(
                speculative_config=SimpleNamespace(method=method),
                parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
            )
        )

    Scheduler.update_draft_token_ids(scheduler("extract_hidden_states", 2), "drafts")
    assert taken == []
    Scheduler.update_draft_token_ids(scheduler("extract_hidden_states", 1), "one stage")
    Scheduler.update_draft_token_ids(scheduler("eagle3", 2), "eagle3")
    assert taken == ["one stage", "eagle3"]
