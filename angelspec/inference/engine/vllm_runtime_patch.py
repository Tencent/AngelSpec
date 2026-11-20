"""Runtime injection of EAGLE3 / extract_hidden_states support into vLLM models.

vLLM's ``extract_hidden_states`` speculative method requires the target model's
inner ``Model`` class to inherit ``EagleModelMixin`` and call
``_maybe_add_hidden_state`` inside its decoder-layer loop, and the outer
``ForCausalLM`` class to satisfy the ``SupportsEagle3`` protocol. Upstream
achieves this by editing each model file. To avoid patching the installed vLLM
source, this module re-registers an enhanced subclass at runtime.

The injection runs inside **every** vLLM process (driver, engine core, and—via
``vllm.general_plugins`` entry point—each spawned worker) because the Eagle3
capability check (``gpu_model_runner._maybe_setup_eagle3``) executes in the
worker after the model is built.

Design notes:
  - The inner Model is decorated with ``@support_torch_compile``. Subclassing it
    and overriding ``forward`` is safe: the decorator compiles ``self.forward``
    (see ``compilation/decorators.py`` ``__call__``), so the subclass override
    becomes the traced function while the wrapped ``__init__`` / ``__call__`` are
    inherited unchanged.
  - The outer ForCausalLM subclass mixes in ``SupportsEagle`` / ``SupportsEagle3``
    (runtime_checkable protocols, attribute-based isinstance) and propagates the
    ``(hidden_states, aux_hidden_states)`` tuple returned by the enhanced Model.
"""

import inspect
import multiprocessing
import os
from collections.abc import Callable, Iterable

from angelspec.utils.hf_config_registry import register_extra_hf_configs
from angelspec.utils.logging import logger


def _running_in_vllm_worker() -> bool:
    """Best-effort detection of a spawned vLLM TP/PP worker process.

    We only need this to AVOID the destructive ``rebuild_dataclass(VllmConfig)``
    in workers; false negatives (treating a worker as non-worker) merely
    reintroduce the original behavior, so this is safe-by-default.

    Two backends:
      - ``mp`` executor spawns child processes named ``VllmWorker-*`` (engine
        core ``EngineCore*``, driver ``MainProcess``) — caught by the name check.
      - ``ray`` executor spawns each TP worker as a *Ray actor*. So is our
        ``VllmEngine`` engine-core/driver actor, so "is a ray worker actor"
        alone can't tell them apart. The engine actor marks its own process with
        ``ANGELSPEC_VLLM_ENGINE_DRIVER``; a ray-worker-mode vLLM process WITHOUT
        that marker is therefore a spawned TP worker.
    """
    try:
        name = multiprocessing.current_process().name or ""
    except Exception:
        name = ""
    if name.startswith("VllmWorker") or "Worker" in name or name.startswith("EngineCore"):
        return True

    # ray backend: our engine-core/driver actor sets this marker in its own
    # process before importing vllm; TP-worker ray actors never run VllmEngine
    # and so lack it.
    if os.environ.get("ANGELSPEC_VLLM_ENGINE_DRIVER") == "1":
        return False
    try:
        import ray

        if ray.is_initialized() and ray.get_runtime_context().worker.mode == ray.WORKER_MODE:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Per-model EAGLE3 forward overrides
# ---------------------------------------------------------------------------
# IMPORTANT: this injection is NOT a generic "works on any model" transform.
# Each target model's inner ``Model.forward`` has its own calling conventions
# (decoder-layer signature, fused vs. plain final norm, aux-capture start index,
# embedding helper name, ...). A forward body copied from one model will, at
# best, raise on another and, at worst, *silently* capture the wrong residual
# stream — which corrupts training data without any error.
#
# Therefore each model that does NOT already ship EAGLE3 support (most models in
# vLLM >= 0.10 already do; those are detected and skipped) must register an
# explicit forward override here, faithfully mirroring its own upstream forward
# plus the two ``_maybe_add_hidden_state`` calls and the aux-tuple return.
#
# To add a new model:
#   1. Read the model's inner ``Model.forward`` in vLLM source.
#   2. Write a ``<model>_inner_forward(self, ...)`` mirroring it, inserting
#      ``self._maybe_add_hidden_state([], <start>, hs, residual)`` before the
#      layer loop and ``self._maybe_add_hidden_state(aux, <idx>+1, hs, residual)``
#      inside it, and returning ``(hidden_states, aux)`` when aux is non-empty.
#   3. Add an entry to ``_EAGLE3_TARGETS`` pointing ``forward`` at it.
#   4. Add the HF ``model_type`` to ``_EXTRA_AUX_MODEL_TYPES``.


def _hyv3_inner_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    """EAGLE3 forward for ``HYV3Model`` (Hy3 / hy_v3).

    Mirrors ``vllm.model_executor.models.hy_v3.HYV3Model.forward`` exactly,
    with auxiliary hidden-state capture added. Hy3 specifics preserved:
      - decoder layer takes an ``idx=`` keyword argument
      - final norm: the vLLM fork's ``HYV3Model`` switches on
        ``VLLM_PRECISION_MODE``. In ``"HF"`` mode ``self.norm`` is a plain
        single-input RMSNorm and the residual is added manually beforehand
        (the path this override mirrors); in ``"default"`` mode it is the fused
        two-input ``self.norm(hidden, residual)``. We force ``"HF"`` for hy_v3
        targets in ``vllm_engine.py``, so
        the HF branch below is the authoritative one. The ``"default"`` branch
        is mirrored too for safety, in case the env is set elsewhere.

    NOTE: the fork's forward also has a ``get_tpsp_ctx()`` sequence-parallel
    chunk/all-gather path. AngelSpec hidden-state extraction runs the target at
    tp=1 (see ``configs/vllm_hy3_mtp.yaml`` / dflash configs), where
    ``get_tpsp_ctx()`` is ``None`` and that path is a no-op, so it is omitted
    here. If a hy_v3 target is ever served at tp>1 for extraction, this body
    must be extended to mirror the tpsp chunk/all-gather.
    """
    from itertools import islice

    from vllm import envs
    from vllm.distributed import get_pp_group
    from vllm.sequence import IntermediateTensors

    pp = get_pp_group()
    if pp.is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    aux_by_layer = _restore_pp_aux_hidden_states(self, intermediate_tensors)
    if self.start_layer in self.aux_hidden_state_layers:
        aux_by_layer[self.start_layer] = (
            hidden_states + residual if residual is not None else hidden_states
        )

    layers = list(islice(self.layers, self.start_layer, self.end_layer))
    for idx, layer in enumerate(layers):
        hidden_states, residual = layer(positions, hidden_states, residual, idx=idx)
        cap = self.start_layer + idx + 1
        if cap in self.aux_hidden_state_layers and (idx + 1 < len(layers) or pp.is_last_rank):
            aux_by_layer[cap] = hidden_states + residual if residual is not None else hidden_states

    if not pp.is_last_rank:
        tensors = {"hidden_states": hidden_states, "residual": residual}
        _add_pp_aux_hidden_states(tensors, self, aux_by_layer)
        return IntermediateTensors(tensors)

    if getattr(envs, "VLLM_PRECISION_MODE", "HF") == "default":
        hidden_states, residual = self.norm(hidden_states, residual)
    else:
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

    if self.aux_hidden_state_layers:
        return hidden_states, _ordered_pp_aux_hidden_states(self, aux_by_layer)
    return hidden_states


_PP_AUX_PREFIX = "angelspec_aux_hidden_"


def _pp_aux_key(layer_id: int) -> str:
    return f"{_PP_AUX_PREFIX}{layer_id}"


def _restore_pp_aux_hidden_states(self, intermediate_tensors) -> dict[int, object]:
    if intermediate_tensors is None:
        return {}
    return {
        layer_id: intermediate_tensors[_pp_aux_key(layer_id)]
        for layer_id in self.aux_hidden_state_layers
        if _pp_aux_key(layer_id) in intermediate_tensors.tensors
    }


def _add_pp_aux_hidden_states(tensors: dict, self, aux_by_layer: dict[int, object]) -> None:
    for layer_id in self.aux_hidden_state_layers:
        if layer_id in aux_by_layer:
            tensors[_pp_aux_key(layer_id)] = aux_by_layer[layer_id]


def _ordered_pp_aux_hidden_states(self, aux_by_layer: dict[int, object]) -> list:
    missing = [
        layer_id for layer_id in self.aux_hidden_state_layers if layer_id not in aux_by_layer
    ]
    if missing:
        raise RuntimeError(
            f"Missing auxiliary hidden states for layers {missing}; "
            f"captured={sorted(aux_by_layer)} stage=[{self.start_layer},{self.end_layer})"
        )
    return [aux_by_layer[layer_id] for layer_id in self.aux_hidden_state_layers]


def _hyv4_flash_inner_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors=None,
    inputs_embeds=None,
    lookback_token_ids=None,
):
    """EAGLE3 forward for ``HYV4FlashModel`` (Hy4 flash, ``hy_v4_flash``).

    Mirrors ``vllm.models.hy_v4_flash.nvidia.model.HYV4FlashModel.forward``, Engram hashing
    included, and the single-stage ``HYV4Model.forward`` it defers to without Engram (Flash
    runs no pipeline parallelism), with auxiliary hidden-state capture added.

    Under iHC the residual stream is ``[num_tokens, hc_mult, hidden]``. A capture at the input
    of layer L is the 2D activation that layer's iHC pre-gate makes of the stream
    (``hc_pre(prepare_input(h))``), taken before L's Engram residual is added; the capture
    after the last layer is ``hc_head``'s output. Without iHC a capture is ``hidden +
    residual``.
    """
    import torch

    want = self.aux_hidden_state_layers
    hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
    residual = None
    aux = {0: hidden} if 0 in want else {}

    hashes = None
    hash_indices = {}
    if self.engram_hash is not None:
        if input_ids is None:
            raise ValueError("Flash Engram requires original input token IDs")
        hashes = input_ids.new_empty(
            (
                len(self.config.engram_layer_ids),
                input_ids.numel(),
                self.engram_hash.offsets.shape[1],
            ),
            dtype=torch.int32,
        )
        self.engram_hash.prepare_hashes_with_output(
            input_ids,
            lookback_token_ids,
            self.layers[0].self_attn.mla_attn.layer_name,
            hashes,
        )
        hash_indices = {
            layer_id - 1: index for index, layer_id in enumerate(self.config.engram_layer_ids)
        }

    num_layers = len(self.layers)
    for index, layer in enumerate(self.layers):
        if index in hash_indices:
            hidden = layer.hc_attn_layer.prepare_input(hidden)
            hidden = layer.engram_layer(hidden, hashes[hash_indices[index]])
        hidden, residual = layer(positions, hidden, residual)
        boundary = index + 1
        if boundary in want and boundary < num_layers:
            if self.enable_ihc:
                hc = self.layers[boundary].hc_attn_layer
                aux[boundary], _ = hc.hc_pre(hc.prepare_input(hidden))
            else:
                aux[boundary] = hidden + residual

    hidden = self.hc_head(hidden) if self.enable_ihc else hidden + residual
    if num_layers in want:
        aux[num_layers] = hidden
    hidden = self.norm(hidden)
    if want:
        return hidden, [aux[layer_id] for layer_id in want]
    return hidden


def _hyv4_inner_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors=None,
    inputs_embeds=None,
):
    """EAGLE3 forward for ``HYV4Model`` (Hy4, ``hy_v4``).

    Mirrors ``vllm.models.hy_v4.nvidia.model.HYV4Model.forward``, pipeline parallelism
    included, with auxiliary hidden-state capture added. Captures follow the Hy4 flash
    override: the input of layer L is the 2D activation L's iHC pre-gate makes of the stream,
    and the capture after the last layer is ``hc_head``'s output. A stage takes the capture at
    its own first layer, whose pre-gate it holds, from the stream the previous stage sent.
    """
    from itertools import islice

    from vllm.distributed import get_pp_group
    from vllm.models.hy_v4.nvidia import model as hy_v4_model
    from vllm.sequence import IntermediateTensors

    if hy_v4_model.is_sequence_parallel_active(
        self.ihc_sequence_parallel, positions.shape[-1], hy_v4_model._IHC_SP_MIN_TOKENS
    ):
        raise RuntimeError("Hy4 hidden-state capture does not support iHC sequence parallelism")

    want = self.aux_hidden_state_layers
    pp = get_pp_group()
    if pp.is_first_rank:
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden = intermediate_tensors["hidden_states"]
        residual = None if self.enable_ihc else intermediate_tensors["residual"]
        hy_v4_model.copy_pp_buffers(intermediate_tensors, self.pp_buffers)

    def layer_input(layer_id):
        if not self.enable_ihc:
            return hidden if residual is None else hidden + residual
        hc = self.layers[layer_id].hc_attn_layer
        return hc.pre(hc.prepare_input(hidden))[0]

    if self.start_layer < self.end_layer:
        hy_v4_model.set_token_rope(
            self.layers[self.start_layer].self_attn.rotary_emb, positions, hidden
        )
    aux = _restore_pp_aux_hidden_states(self, intermediate_tensors)
    if self.start_layer in want:
        aux[self.start_layer] = hidden if self.start_layer == 0 else layer_input(self.start_layer)
    for index, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer), start=self.start_layer
    ):
        hidden, residual = layer(positions, hidden, residual)
        if index + 1 in want and index + 1 < self.end_layer:
            aux[index + 1] = layer_input(index + 1)

    if not pp.is_last_rank:
        buffers = {k: v[: positions.shape[0]] for k, v in self.pp_buffers.items()}
        if self.enable_ihc:
            tensors = {"hidden_states": hidden.flatten(1), **buffers}
        else:
            tensors = {"hidden_states": hidden, "residual": residual, **buffers}
        _add_pp_aux_hidden_states(tensors, self, aux)
        return IntermediateTensors(tensors)

    hidden = self.hc_head(hidden) if self.enable_ihc else hidden + residual
    if self.end_layer in want:
        aux[self.end_layer] = hidden
    hidden = self.norm(hidden)
    if want:
        return hidden, _ordered_pp_aux_hidden_states(self, aux)
    return hidden


# Map of model architecture -> injection spec. Each spec carries the candidate
# module path(s), the inner Model / outer ForCausalLM class names, and a
# model-specific ``forward`` override (see the note above). ``modules`` is an
# ordered list of candidate import paths: the first that imports wins.
_EAGLE3_TARGETS: dict[str, dict] = {
    "HYV3ForCausalLM": {
        "modules": [
            "vllm.model_executor.models.hunyuan_v3",
            "vllm.model_executor.models.hy_v3",
        ],
        "causal_lm": "HYV3ForCausalLM",
        "inner_model": "HYV3Model",
        "forward": _hyv3_inner_forward,
    },
    "HYV4FlashForCausalLM": {
        "modules": [
            "vllm.models.hy_v4_flash.nvidia.model",
        ],
        "causal_lm": "HYV4FlashForCausalLM",
        "inner_model": "HYV4FlashModel",
        "forward": _hyv4_flash_inner_forward,
        # Flash's own ForCausalLM.forward passes the Engram lookback tokens through.
        "keep_causal_forward": True,
    },
    # vLLM loads Hy4 flash checkpoints as this multimodal wrapper, which builds its text tower
    # from the registry as ``HYV4FlashForCausalLM`` (above) and returns its output verbatim.
    "HYV4FlashVLForConditionalGeneration": {
        "modules": [
            "vllm.models.hy_v4_flash.nvidia.vl_model",
        ],
        "causal_lm": "HYV4FlashVLForConditionalGeneration",
        "wrapper": True,
        "language_model_arch": "HYV4FlashForCausalLM",
    },
    "HYV4ForCausalLM": {
        "modules": [
            "vllm.models.hy_v4.nvidia.model",
        ],
        "causal_lm": "HYV4ForCausalLM",
        "inner_model": "HYV4Model",
        "forward": _hyv4_inner_forward,
    },
    # Hy4 checkpoints load as this multimodal wrapper around ``HYV4ForCausalLM``.
    "HYV4VLForConditionalGeneration": {
        "modules": [
            "vllm.models.hy_v4.nvidia.vl_model",
        ],
        "causal_lm": "HYV4VLForConditionalGeneration",
        "wrapper": True,
        "language_model_arch": "HYV4ForCausalLM",
    },
    # Qwen3-Omni thinker: vLLM maps ``Qwen3OmniMoeForConditionalGeneration`` to
    # the thinker-only class, whose text tower is ``Qwen3MoeLLMForCausalLM`` (a
    # ``Qwen3MoeForCausalLM`` subclass, already natively ``SupportsEagle3``). The
    # thinker forward returns the tower output verbatim, so we only need to mark
    # the outer wrapper ``SupportsEagle3``.
    "Qwen3OmniMoeForConditionalGeneration": {
        "modules": [
            "vllm.model_executor.models.qwen3_omni_moe_thinker",
        ],
        "causal_lm": "Qwen3OmniMoeThinkerForConditionalGeneration",
        "wrapper": True,
        "language_model_arch": "Qwen3MoeForCausalLM",
    },
}

_injected_archs: set[str] = set()

# model_type values to whitelist for extract_hidden_states / eagle3 / dflash.
# vLLM hardcodes a supported list inside SpeculativeConfig._verify_args; we add
# ours at runtime instead of patching vLLM source.
_EXTRA_AUX_MODEL_TYPES: set[str] = {
    "hy_v3",
    "hy_v4",
    "hy_v4_flash",
    "glm_moe_dsa",
    "qwen3_omni_moe_text",
}
_spec_validator_patched = False
_extract_hidden_states_pp_patched = False


def _defer_worker_vllm_schema_rebuild():
    from vllm.config import VllmConfig

    original = VllmConfig.with_hf_config
    if getattr(original, "_angelspec_schema_rebuild", False):
        return
    rebuilt = False

    def with_hf_config(self, hf_config, architectures=None):
        nonlocal rebuilt
        spec = self.speculative_config
        target = getattr(spec, "target_model_config", None)
        model_type = getattr(getattr(target, "hf_text_config", None), "model_type", None)
        if (
            not rebuilt
            and getattr(spec, "method", None) in ("eagle3", "extract_hidden_states", "dflash")
            and model_type in _EXTRA_AUX_MODEL_TYPES
        ):
            import pydantic.dataclasses as pdc

            pdc.rebuild_dataclass(VllmConfig, force=True)
            rebuilt = True
        return original(self, hf_config, architectures=architectures)

    with_hf_config._angelspec_schema_rebuild = True
    VllmConfig.with_hf_config = with_hf_config


def apply_speculative_model_type_whitelist() -> bool:
    """Allow extra model_types through SpeculativeConfig's extract_hidden_states
    validator.

    vLLM 0.10's ``SpeculativeConfig._verify_args`` raises if the target
    ``model_type`` is not in a hardcoded allow-list. This wraps that validator
    so that, when the only failure is the unsupported-model-type check for a
    model_type we know is supported (via our Eagle3 injection), it is allowed.
    Idempotent; safe in every process.
    """
    global _spec_validator_patched
    if _spec_validator_patched:
        return True
    try:
        from vllm.config.speculative import SpeculativeConfig

        # SpeculativeConfig is a pydantic *dataclass*; its validators are
        # compiled into __pydantic_validator__ at class-creation time. Patching
        # the bound method is not enough — we must replace the registered
        # validator func in __pydantic_decorators__ and rebuild the dataclass
        # schema so pydantic re-reads it.
        decorators = SpeculativeConfig.__pydantic_decorators__
        mv = decorators.model_validators.get("_verify_args")
        if mv is None:
            logger.warning(
                "apply_speculative_model_type_whitelist: _verify_args validator not found; skipping."
            )
            return False

        original_verify = mv.func

        def _verify_args(self):
            if (
                getattr(self, "method", None) == "extract_hidden_states"
                and getattr(self, "draft_parallel_config", None) is not None
                and self.draft_parallel_config.pipeline_parallel_size > 1
            ):
                from vllm.config.utils import replace

                self.draft_parallel_config = replace(
                    self.draft_parallel_config,
                    pipeline_parallel_size=1,
                )
            try:
                return original_verify(self)
            except ValueError as e:
                msg = str(e)
                if "is only supported for" not in msg:
                    raise
                tgt = getattr(self, "target_model_config", None)
                model_type = getattr(tgt.hf_text_config, "model_type", None) if tgt else None
                if model_type not in _EXTRA_AUX_MODEL_TYPES:
                    raise
                logger.info(
                    "apply_speculative_model_type_whitelist: allowing "
                    "method=%s for model_type=%s (runtime-injected aux support).",
                    getattr(self, "method", None),
                    model_type,
                )
                # Re-run the trailing validation the original would have done
                # after the unsupported-model check.
                self.verify_equal_vocab_size_if_draft_model()
                return self

        mv.func = _verify_args

        import pydantic.dataclasses as _pdc

        _pdc.rebuild_dataclass(SpeculativeConfig, force=True)

        # VllmConfig embeds SpeculativeConfig's validation schema; rebuilding it
        # lets the nested re-validation during VllmConfig construction pick up
        # the patched validator. BUT force-rebuilding VllmConfig's pydantic
        # schema inside spawned TP **worker** processes corrupts the config
        # (de)serialization those workers rely on, and the model-runner's
        # memory-profiling dummy forward then hangs forever (GPU idle, only
        # "No available shared memory broadcast" repeats). SpeculativeConfig is
        # only ever constructed/validated in the engine-core/driver process, so
        # the VllmConfig rebuild is unnecessary in workers. Skip it there.
        # ``VLLM_DP_RANK``/multiproc workers don't build the spec config; detect
        # the worker role via the process name and skip.
        if not _running_in_vllm_worker():
            try:
                from vllm.config import VllmConfig

                _pdc.rebuild_dataclass(VllmConfig, force=True)
            except Exception:
                logger.warning(
                    "apply_speculative_model_type_whitelist: could not rebuild "
                    "VllmConfig; nested re-validation may still reject the "
                    "model_type."
                )
        else:
            _defer_worker_vllm_schema_rebuild()
            logger.info(
                "apply_speculative_model_type_whitelist: worker process — "
                "deferring VllmConfig rebuild until VL text-config construction."
            )

        _spec_validator_patched = True
        logger.info(
            "apply_speculative_model_type_whitelist: patched SpeculativeConfig._verify_args for %s.",
            sorted(_EXTRA_AUX_MODEL_TYPES),
        )
        return True
    except Exception:
        logger.exception("apply_speculative_model_type_whitelist: failed to patch validator")
        return False


def apply_extract_hidden_states_pp_compat() -> bool:
    global _extract_hidden_states_pp_patched
    if _extract_hidden_states_pp_patched:
        return True
    try:
        from vllm.distributed import get_pp_group
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        # Only the V2 runner left, and it builds the drafter on the last stage alone.
        return False
    try:
        original = GPUModelRunner.initialize_kv_cache

        def wrapped(self, kv_cache_config, *args, **kwargs):
            spec_config = getattr(self, "speculative_config", None)
            if (
                spec_config is not None
                and spec_config.uses_extract_hidden_states()
                and not get_pp_group().is_last_rank
            ):
                self.speculative_config = None
                try:
                    return original(self, kv_cache_config, *args, **kwargs)
                finally:
                    self.speculative_config = spec_config
            return original(self, kv_cache_config, *args, **kwargs)

        GPUModelRunner.initialize_kv_cache = wrapped
        _extract_hidden_states_pp_patched = True
        return True
    except Exception:
        logger.exception("apply_extract_hidden_states_pp_compat: failed")
        return False


def apply_extract_hidden_states_pp_no_drafts() -> bool:
    """Ignore extract_hidden_states' drafts in the scheduler under pipeline parallelism.

    Its drafts are placeholders AngelSpec never verifies (prompts run with max_tokens=1). Over
    pipeline stages vLLM's scheduler can take a request's drafts before the output of the step
    that made them, and schedule it again with a draft but not the token it follows, which the
    V2 model runner rejects (``num_scheduled_tokens >= num_logits`` in prepare_inputs). Without
    drafts the request waits for its output and finishes, as on one stage.
    """
    try:
        import functools

        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        return False
    if getattr(Scheduler, "_angelspec_pp_no_drafts", False):
        return True
    original = Scheduler.update_draft_token_ids

    @functools.wraps(original)
    def update_draft_token_ids(self, draft_token_ids):
        spec = self.vllm_config.speculative_config
        if (
            spec is not None
            and spec.method == "extract_hidden_states"
            and self.vllm_config.parallel_config.pipeline_parallel_size > 1
        ):
            return None
        return original(self, draft_token_ids)

    Scheduler.update_draft_token_ids = update_draft_token_ids
    Scheduler._angelspec_pp_no_drafts = True
    return True


def _build_enhanced_classes(
    module,
    causal_lm_name: str,
    inner_model_name: str,
    inner_forward: Callable,
    keep_causal_forward: bool = False,
):
    """Create EAGLE3-enhanced subclasses of the inner Model and outer ForCausalLM.

    ``inner_forward`` is the model-specific forward override registered in
    ``_EAGLE3_TARGETS`` (see the note at the top of this module). It is bound as
    the enhanced inner model's ``forward``; this module does NOT attempt to
    synthesize a forward that works across model families. ``keep_causal_forward``
    keeps the ForCausalLM's own forward, for models whose forward already returns the
    inner model's output verbatim but passes it extra arguments.
    """
    from vllm.model_executor.models.interfaces import (
        EagleModelMixin,
        SupportsEagle,
        SupportsEagle3,
    )

    OrigModel = getattr(module, inner_model_name)
    OrigCausalLM = getattr(module, causal_lm_name)

    class EagleInnerModel(OrigModel, EagleModelMixin):
        """Inner model that emits auxiliary hidden states for EAGLE3.

        Captures the residual stream at the model input and after each decoder
        layer, returning the aux list alongside the final hidden states when any
        capture layer is configured. The actual forward body is the registered
        model-specific override.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            base_make_empty = self.make_empty_intermediate_tensors

            def make_empty_intermediate_tensors(batch_size, dtype, device):
                import torch

                result = base_make_empty(batch_size, dtype, device)
                for layer_id in self.aux_hidden_state_layers:
                    if layer_id < self.start_layer:
                        result.tensors[_pp_aux_key(layer_id)] = torch.zeros(
                            (batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device,
                        )
                return result

            self.make_empty_intermediate_tensors = make_empty_intermediate_tensors

    EagleInnerModel.forward = inner_forward
    EagleInnerModel.__name__ = f"Eagle3{inner_model_name}"
    EagleInnerModel.__qualname__ = EagleInnerModel.__name__

    class EagleCausalLM(OrigCausalLM, SupportsEagle, SupportsEagle3):
        """ForCausalLM that builds the EAGLE3-enhanced inner model and
        propagates the (hidden_states, aux_hidden_states) tuple."""

        def __init__(self, *, vllm_config, prefix: str = ""):
            # Swap the inner Model class for our enhanced subclass *only*
            # during this constructor, so the original __init__ wires up
            # everything against EagleInnerModel without us reimplementing it.
            original = getattr(module, inner_model_name)
            setattr(module, inner_model_name, EagleInnerModel)
            try:
                super().__init__(vllm_config=vllm_config, prefix=prefix)
            finally:
                setattr(module, inner_model_name, original)
            spec_config = getattr(vllm_config, "speculative_config", None)
            draft_config = getattr(spec_config, "draft_model_config", None)
            hf_config = getattr(draft_config, "hf_config", None)
            aux_layers = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
            if aux_layers:
                self.set_aux_hidden_state_layers(tuple(aux_layers))

        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors=None,
            inputs_embeds=None,
        ):
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

        def load_weights(self, weights):
            # A VL checkpoint loaded as its text tower (no ``visual`` submodule)
            # still ships vision-encoder tensors; drop them so AutoWeightsLoader
            # doesn't reject the unexpected keys. No-op for text-only checkpoints.
            return super().load_weights(
                (n, w) for n, w in weights if not (n.startswith("vit.") or n.startswith("visual."))
            )

    if hasattr(OrigCausalLM, "model_cls"):
        EagleCausalLM.model_cls = EagleInnerModel
    if keep_causal_forward:
        del EagleCausalLM.forward
    EagleCausalLM.__name__ = causal_lm_name
    EagleCausalLM.__qualname__ = causal_lm_name

    return EagleCausalLM


def _build_enhanced_vl_class(module, vl_name: str):
    """Enhance a multimodal wrapper (e.g. ``HYV4FlashVLForConditionalGeneration``).

    The wrapper builds its text tower via ``init_vllm_registered_model`` and its
    ``forward`` returns that tower's output verbatim, so once the inner
    ``ForCausalLM`` is EAGLE3-enhanced (its ``forward`` returns the
    ``(hidden_states, aux)`` tuple) the wrapper propagates the tuple unchanged.
    All that is missing is the ``SupportsEagle3`` marker: its default
    ``set_aux_hidden_state_layers`` already descends via ``get_language_model()``
    → ``language_model.model`` (an ``EagleModelMixin`` after inner injection).
    """
    from vllm.model_executor.models.interfaces import SupportsEagle, SupportsEagle3

    OrigVL = getattr(module, vl_name)

    class EagleVL(OrigVL, SupportsEagle, SupportsEagle3):
        pass

    EagleVL.__name__ = vl_name
    EagleVL.__qualname__ = vl_name
    return EagleVL


def apply_eagle3_mixin(arch: str = "HYV3ForCausalLM") -> bool:
    """Re-register *arch* with an EAGLE3-enhanced subclass.

    Idempotent and safe to call in every process. Returns True if injection
    happened (or was already done), False if the architecture is unknown or
    vLLM is unavailable.
    """
    if arch in _injected_archs:
        return True

    spec = _EAGLE3_TARGETS.get(arch)
    if spec is None:
        logger.warning("apply_eagle3_mixin: unknown architecture %s", arch)
        return False

    try:
        import importlib

        from vllm import ModelRegistry
        from vllm.model_executor.models.interfaces import supports_eagle3

        candidates = spec.get("modules") or [spec["module"]]
        module = None
        for mod_path in candidates:
            try:
                module = importlib.import_module(mod_path)
                break
            except ModuleNotFoundError as e:
                # Only treat this as "candidate absent" if the *candidate module
                # itself* is what's missing (rename across vLLM versions). A
                # transitive ModuleNotFoundError (e.g. an optional dep imported
                # inside the model file) is a real problem — let it propagate so
                # it surfaces instead of silently disabling injection.
                missing = e.name or ""
                if missing == mod_path or mod_path.startswith(missing + "."):
                    continue
                raise
        if module is None:
            # None of the candidate modules ship in this vLLM build. Nothing to
            # inject; skip quietly (fail-closed: an unregistered model simply
            # won't be offered extract_hidden_states support).
            logger.info(
                "apply_eagle3_mixin: none of %s present in this vLLM build; skipping %s.",
                candidates,
                arch,
            )
            _injected_archs.add(arch)
            return False

        orig_causal = getattr(module, spec["causal_lm"])
        if supports_eagle3(orig_causal):
            # Upstream already supports it (e.g. a patched / newer vLLM).
            _injected_archs.add(arch)
            logger.info("apply_eagle3_mixin: %s already supports EAGLE3, skipping.", arch)
            return True

        if spec.get("wrapper"):
            # Multimodal wrapper: capture lives in the nested text tower, so
            # enhance that arch first, then mark the wrapper SupportsEagle3.
            lm_arch = spec["language_model_arch"]
            if lm_arch not in _injected_archs:
                apply_eagle3_mixin(lm_arch)
            enhanced = _build_enhanced_vl_class(module, spec["causal_lm"])
            ModelRegistry.register_model(arch, enhanced)
            _injected_archs.add(arch)
            logger.info(
                "apply_eagle3_mixin: registered EAGLE3-enhanced wrapper %s (text tower: %s).",
                arch,
                lm_arch,
            )
            return True

        inner_forward = spec.get("forward")
        if inner_forward is None:
            logger.error(
                "apply_eagle3_mixin: %s has no registered forward override; "
                "cannot inject EAGLE3 safely. Add one to _EAGLE3_TARGETS.",
                arch,
            )
            return False

        enhanced = _build_enhanced_classes(
            module,
            spec["causal_lm"],
            spec["inner_model"],
            inner_forward,
            keep_causal_forward=spec.get("keep_causal_forward", False),
        )
        ModelRegistry.register_model(arch, enhanced)
        _injected_archs.add(arch)
        logger.info(
            "apply_eagle3_mixin: registered EAGLE3-enhanced %s (%s).",
            arch,
            enhanced.__module__,
        )
        return True
    except Exception:
        logger.exception("apply_eagle3_mixin: failed to inject %s", arch)
        return False


_qwen3_mtp_patched = False


def apply_qwen3_mtp() -> bool:
    """Register the dense-Qwen3 single-head MTP draft into vLLM at runtime.

    vLLM ships MTP model classes for hy_v3 / deepseek / qwen3_next / … but none
    for a *dense* Qwen3 target. AngelSpec trains exactly that (see
    ``angelspec/inference/engine/qwen3_mtp.py``). Rather than patch vLLM source,
    we inject three things at plugin-load time (same mechanism as the EAGLE3
    injection above), so an upgrade/reinstall of vLLM is unaffected:

      1. Register ``Qwen3MTP`` under arch ``Qwen3MTPForCausalLM`` so the drafter
         resolves to it.
      2. Wrap ``SpeculativeConfig.hf_config_override`` so a served checkpoint with
         ``model_type="qwen3"`` **and** ``num_nextn_predict_layers>0`` promotes the
         *draft* config to ``qwen3_mtp`` / ``Qwen3MTPForCausalLM`` (mirrors the
         built-in ``qwen3_next``/``nemotron_h`` branches). The guard means a plain
         Qwen3 checkpoint is untouched; only the draft config copy is mutated, so
         the main model still loads as ``Qwen3ForCausalLM``.
      3. Re-register ``Qwen3ForCausalLM`` with a subclass whose weight loader skips
         the co-located ``mtp.*`` keys. The MTP head lives in the SAME dir as the
         target (method="mtp" loads the drafter from the target path), and the
         native Qwen3 loader raises on unexpected keys — ``qwen3_next`` avoids this
         with ``skip_prefixes=["mtp."]``; we do the same. Harmless when no ``mtp.*``
         keys are present, so it is safe for ordinary Qwen3 serving.

    Idempotent; safe to call in every vLLM process.
    """
    global _qwen3_mtp_patched
    if _qwen3_mtp_patched:
        return True
    try:
        from vllm import ModelRegistry
        from vllm.config.speculative import SpeculativeConfig
        from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

        from angelspec.inference.engine.qwen3_mtp import Qwen3MTP

        # (1) drafter arch
        ModelRegistry.register_model("Qwen3MTPForCausalLM", Qwen3MTP)

        # (2) spec detection: qwen3 (+ nextn head) -> Qwen3MTPForCausalLM (draft)
        _orig_override = SpeculativeConfig.hf_config_override

        def _override(hf_config):
            hf_config = _orig_override(hf_config)
            if (
                getattr(hf_config, "model_type", None) == "qwen3"
                and getattr(hf_config, "num_nextn_predict_layers", 0) > 0
            ):
                n_predict = hf_config.num_nextn_predict_layers
                # "mtp" (generic) is the model_type value SpeculativeConfig's
                # method-detection accepts (it is in MTPModelTypes); "qwen3_mtp"
                # is not, so vLLM would raise "Unsupported speculative method".
                # architectures routes model construction to our registered class.
                hf_config.model_type = "mtp"
                hf_config.update(
                    {"n_predict": n_predict, "architectures": ["Qwen3MTPForCausalLM"]}
                )
            return hf_config

        SpeculativeConfig.hf_config_override = staticmethod(_override)

        # (3) let the main Qwen3 target loader ignore the co-located mtp.* weights
        class Qwen3ForCausalLMSkipMTP(Qwen3ForCausalLM):
            def load_weights(self, weights):
                return super().load_weights((n, w) for n, w in weights if not n.startswith("mtp."))

        Qwen3ForCausalLMSkipMTP.__name__ = "Qwen3ForCausalLM"
        Qwen3ForCausalLMSkipMTP.__qualname__ = "Qwen3ForCausalLM"
        ModelRegistry.register_model("Qwen3ForCausalLM", Qwen3ForCausalLMSkipMTP)

        _qwen3_mtp_patched = True
        logger.info(
            "apply_qwen3_mtp: registered Qwen3MTPForCausalLM + spec detection + mtp-skip Qwen3 target loader."
        )
        return True
    except Exception:
        logger.exception("apply_qwen3_mtp: failed to inject dense-Qwen3 MTP support")
        return False


def apply_vit_rotary_emb_fix() -> bool:
    """Route ``ApplyRotaryEmb.forward_cuda`` to ``forward_native``.

    vLLM's ViT rotary custom op (used by the Qwen2.5/Qwen3-VL vision tower)
    calls an unbound ``apply_rotary_emb`` in ``forward_cuda`` — the flash-rotary
    import is commented out, so the CUDA path raises ``NameError`` during the
    vision profiling run. ``forward_native`` is numerically equivalent and has an
    identical signature, so we bind it over ``forward_cuda`` at plugin load.
    """
    try:
        from vllm.model_executor.layers.rotary_embedding import common as _rope_common

        cls = getattr(_rope_common, "ApplyRotaryEmb", None)
        if cls is None:
            return False
        if getattr(cls, "_angelspec_rotary_native_fix", False):
            return True
        cls.forward_cuda = cls.forward_native
        cls._angelspec_rotary_native_fix = True
        logger.info(
            "apply_vit_rotary_emb_fix: routed ApplyRotaryEmb.forward_cuda -> forward_native"
        )
        return True
    except Exception:
        logger.exception("apply_vit_rotary_emb_fix: failed to patch ViT rotary op")
        return False


def apply_qwen3_omni_thinker_aux() -> bool:
    """Add EAGLE3 aux-hidden capture to the Qwen3-Omni thinker text model.

    ``Qwen3MoeLLMModel`` (the thinker's text tower) overrides ``forward`` to add
    deepstack fusion but drops the base ``Qwen3MoeModel`` aux-capture, so it only
    ever returns a single tensor. With the outer wrapper marked SupportsEagle3,
    vLLM then tries to unpack ``hidden, aux = outputs`` and fails. Re-bind a
    forward that keeps the deepstack fusion *and* collects aux hidden states.
    """
    try:
        from vllm.distributed import get_pp_group
        from vllm.model_executor.models import qwen3_omni_moe_thinker as _mod
        from vllm.sequence import IntermediateTensors

        cls = getattr(_mod, "Qwen3MoeLLMModel", None)
        if cls is None:
            return False
        if getattr(cls, "_angelspec_aux_forward", False):
            return True

        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors=None,
            inputs_embeds=None,
            deepstack_input_embeds=None,
        ):
            if get_pp_group().is_first_rank:
                hidden_states = (
                    inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
                )
                residual = None
            else:
                assert intermediate_tensors is not None
                hidden_states = intermediate_tensors["hidden_states"]
                residual = intermediate_tensors["residual"]

            aux_hidden_states = self._maybe_add_hidden_state(
                [], self.start_layer, hidden_states, residual
            )
            for layer_idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
                layer_idx = layer_idx + self.start_layer
                hidden_states, residual = layer(positions, hidden_states, residual)
                if deepstack_input_embeds is not None and layer_idx in range(
                    0, len(deepstack_input_embeds)
                ):
                    hidden_states = (
                        hidden_states
                        + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                    )
                self._maybe_add_hidden_state(
                    aux_hidden_states, layer_idx + 1, hidden_states, residual
                )

            if not get_pp_group().is_last_rank:
                return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
            hidden_states, _ = self.norm(hidden_states, residual)
            if len(aux_hidden_states) > 0:
                return hidden_states, aux_hidden_states
            return hidden_states

        cls.forward = forward
        cls._angelspec_aux_forward = True
        logger.info(
            "apply_qwen3_omni_thinker_aux: patched Qwen3MoeLLMModel.forward with aux capture"
        )
        return True
    except Exception:
        logger.exception("apply_qwen3_omni_thinker_aux: failed to patch thinker text model")
        return False


def apply_kv_cache_no_pad_cache_only() -> bool:
    """Give the extract-mode "cache-only" hidden-states layer its OWN KV-cache group
    instead of being summed into the single uniform group.

    In extract-hidden-states mode the cache-only layer (head_size=hidden_size, ~786KB
    page) is an MLAAttentionSpec with the same block_size as the real MLA/indexer/full
    layers, so ``UniformTypeKVCacheSpecs.is_uniform_type`` (kv_cache_interface.py:667)
    merges all groups into ONE whose ``page_size_bytes`` is the SUM of per-layer pages
    (kv_cache_interface.py:656-657). ``num_blocks = avail // summed_page`` then collapses
    GPU KV capacity by the number of layers sharing the group. (This is why patching
    ``unify_kv_cache_spec_page_size`` did nothing: the uniform-type branch returns before
    it is ever reached.)

    We make ``is_uniform_type`` return False when a cache-only layer is mixed with real
    attention layers, so vLLM falls through to its heterogeneous-page (HMA) grouping:
    the cache-only layer gets its own big page while MLA(576)/indexer(132)/full keep
    their small pages -> capacity recovers toward the plain-serve figure.

    Precise, name-independent detection: the cache-only layer is the only AttentionSpec
    with head_size == hidden_size (>=1024); real attention heads are <=576.

    Gated by ANGELSPEC_KV_SEGREGATE_CACHE_ONLY (default off). No-op for non-extract
    paths (no cache-only layer). Set the env to 0 to revert instantly.
    """
    import os

    if os.environ.get("ANGELSPEC_KV_SEGREGATE_CACHE_ONLY", "0") != "1":
        return False
    try:
        from vllm.v1 import kv_cache_interface as _kci

        cls = _kci.UniformTypeKVCacheSpecs
        if getattr(cls, "_angelspec_split_cacheonly", False):
            return True
        _orig = cls.is_uniform_type.__func__  # unwrap classmethod

        def _is_cache_only(s) -> bool:
            # extract cache-only hidden-states layer: head_size == hidden_size (>=1024),
            # from extract_hidden_states.py:364-368. Real attn heads (MLA 576 / DSA
            # indexer 132 / full <=256) never reach 1024. Name/prefix independent.
            return getattr(s, "head_size", 0) >= 1024

        def is_uniform_type(cls_, kv_cache_specs):
            specs = list(kv_cache_specs.values())
            has_co = any(_is_cache_only(s) for s in specs)
            if has_co and not all(_is_cache_only(s) for s in specs):
                if not getattr(cls_, "_angelspec_split_logged", False):
                    logger.info(
                        "apply_kv_cache_no_pad_cache_only: cache-only layer mixed with %d "
                        "real-attn layers -> is_uniform_type=False (split into own KV group)",
                        sum(1 for s in specs if not _is_cache_only(s)),
                    )
                    cls_._angelspec_split_logged = True
                return False
            return _orig(cls_, kv_cache_specs)

        cls.is_uniform_type = classmethod(is_uniform_type)
        cls._angelspec_split_cacheonly = True
        logger.info(
            "apply_kv_cache_no_pad_cache_only: patched UniformTypeKVCacheSpecs.is_uniform_type "
            "(ANGELSPEC_KV_SEGREGATE_CACHE_ONLY=1)"
        )
        return True
    except Exception:
        logger.exception(
            "apply_kv_cache_no_pad_cache_only: failed to patch "
            "UniformTypeKVCacheSpecs.is_uniform_type"
        )
        return False


def apply_kv_block_zeroer_skip_cache_only() -> bool:
    """Skip the extract cache-only layer in ``KVBlockZeroer.init_meta``.

    ``init_meta`` (vllm/v1/worker/utils.py) precomputes the Triton zero-block-ids
    kernel meta by walking *all* ``FullAttentionSpec`` layers and asserting they
    share one page size (Mamba layers are skipped). The extract cache-only layer
    is a FullAttentionSpec with head_size==hidden_size, so its page differs from
    the model's real attention layers. On a non-hybrid target both happen to match
    (same block_size, and the cache-only page equals the attention page); on a
    hybrid mamba target the real full-attention page is padded up to the
    mamba page, so the cache-only page no longer matches -> ``Non-uniform page
    sizes`` assert. The cache-only layer only stores extracted hidden states and is
    never block-zeroed, so we add it to ``runner_only_attn_layers`` (already the
    skip mechanism inside init_meta) instead of letting it break the uniform check.

    Same detection as :func:`apply_kv_cache_no_pad_cache_only` (head_size >= 1024),
    gated by the same ``ANGELSPEC_KV_SEGREGATE_CACHE_ONLY`` env.
    """
    import os

    if os.environ.get("ANGELSPEC_KV_SEGREGATE_CACHE_ONLY", "0") != "1":
        return False
    try:
        from vllm.v1.worker import utils as _wu

        cls = _wu.KVBlockZeroer
        if getattr(cls, "_angelspec_skip_cache_only", False):
            return True
        _orig = cls.init_meta

        def init_meta(
            self,
            attn_groups_iter,
            kernel_block_sizes,
            cache_dtype,
            runner_only_attn_layers,
            static_forward_context,
        ):
            groups = list(attn_groups_iter)
            extra_skip: set[str] = set()
            for group in groups:
                spec = getattr(group, "kv_cache_spec", None)
                if spec is not None and getattr(spec, "head_size", 0) >= 1024:
                    extra_skip.update(group.layer_names)
            if extra_skip and not getattr(cls, "_angelspec_skip_logged", False):
                logger.info(
                    "apply_kv_block_zeroer_skip_cache_only: skipping %d cache-only "
                    "layer(s) in KVBlockZeroer.init_meta (head_size>=hidden_size)",
                    len(extra_skip),
                )
                cls._angelspec_skip_logged = True
            merged = set(runner_only_attn_layers) | extra_skip
            return _orig(
                self, iter(groups), kernel_block_sizes, cache_dtype, merged, static_forward_context
            )

        cls.init_meta = init_meta
        cls._angelspec_skip_cache_only = True
        logger.info(
            "apply_kv_block_zeroer_skip_cache_only: patched KVBlockZeroer.init_meta "
            "(ANGELSPEC_KV_SEGREGATE_CACHE_ONLY=1)"
        )
        return True
    except Exception:
        logger.exception(
            "apply_kv_block_zeroer_skip_cache_only: failed to patch KVBlockZeroer.init_meta"
        )
        return False


def apply_kv_cache_diag() -> bool:
    """Diagnostic: log the exact group breakdown + intermediates that
    ``get_max_concurrency_for_kv_cache_config`` uses to derive GPU KV cache size,
    so we can see precisely how the 8,511-token capacity arises. Gated by
    ANGELSPEC_KV_DIAG (default off); pure logging, does not change behavior.
    """
    import os

    if os.environ.get("ANGELSPEC_KV_DIAG", "0") != "1":
        return False
    try:
        from vllm.v1.core import kv_cache_utils as _kv

        if getattr(_kv, "_angelspec_kv_diag", False):
            return True
        _orig = _kv.get_max_concurrency_for_kv_cache_config

        def _wrapped(vllm_config, kv_cache_config):
            try:
                groups = kv_cache_config.kv_cache_groups
                brk = [
                    (len(g.layer_names), getattr(g.kv_cache_spec, "page_size_bytes", None))
                    for g in groups
                ]
                nlpg = max(len(g.layer_names) for g in groups) if groups else 0
                g0 = groups[0].kv_cache_spec.page_size_bytes if groups else None
                logger.info(
                    "KV_DIAG: num_groups=%d num_blocks=%s num_layer_per_group=%d "
                    "groups(nlayers,page_bytes)=%s groups0_page=%s",
                    len(groups),
                    getattr(kv_cache_config, "num_blocks", None),
                    nlpg,
                    brk,
                    g0,
                )
            except Exception:
                logger.exception("KV_DIAG: introspection failed")
            res = _orig(vllm_config, kv_cache_config)
            logger.info("KV_DIAG: max_concurrency=%s", res)
            return res

        _kv.get_max_concurrency_for_kv_cache_config = _wrapped
        _kv._angelspec_kv_diag = True
        logger.info("apply_kv_cache_diag: instrumented get_max_concurrency_for_kv_cache_config")
        return True
    except Exception:
        logger.exception("apply_kv_cache_diag: failed to instrument")
        return False


HY4_FRAME_PREFIXES_KEY = "angelspec_frame_prefix_token_ids"
HY4_VIDEO_SUFFIX_KEY = "angelspec_video_suffix_token_ids"


def _hy_v4_pretokenized_video_tokens(processor, grid_thw, metadata):
    """A megatron video as its prompt holds it: video start, each frame's packed prefix
    then the frame (image start, video pads with each row's newline slot, image end),
    the packed suffix, video end."""
    from vllm.transformers_utils.processors.hy_v4 import (
        vision_token_count,
        vision_token_grid,
    )

    vision = processor.config.vision_config
    frames, _, _ = vision_token_grid(grid_thw, vision, "video")
    prefixes = metadata[HY4_FRAME_PREFIXES_KEY]
    if len(prefixes) != frames:
        raise ValueError(
            f"Hy4 video has {len(prefixes)} pretokenized frames but the processor "
            f"produced {frames}"
        )
    pads = [processor.video_token_id] * (vision_token_count(grid_thw, vision, "video") // frames)
    tokens = [processor.video_start_token_id]
    for prefix in prefixes:
        tokens += [*prefix, processor.image_start_token_id, *pads, processor.image_end_token_id]
    tokens += [*metadata.get(HY4_VIDEO_SUFFIX_KEY, []), processor.video_end_token_id]
    return tokens


def _video_with_metadata(item):
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
        return item
    return item, {}


def apply_hy_v4_flash_pretokenized_video() -> bool:
    """Lay out megatron Hy4 flash videos with the frame prefixes they were packed with.

    A megatron video keeps the tokens before each frame (and after the last) as packed,
    while vLLM's Hy4 processor writes ``<t seconds>`` stamps of its own, so it could not
    locate the video in the token prompt. For a video whose metadata names those
    prefixes (``VllmEngine._load_patch_video``), the replacement uses them; its frames
    arrive at their packed size, so they are not resized again. Other media keep vLLM's
    layout, which AngelSpec's own rendering already matches.
    """
    try:
        import numpy as np
        from vllm.models.hy_v4.nvidia.mm_preprocess import (
            HYV4VLMultiModalProcessor as Proc,
        )
        from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
    except Exception:
        return False
    if getattr(Proc, "_angelspec_pretokenized_video", False):
        return True
    original_call = Proc._call_hf_processor
    original_updates = Proc._get_prompt_updates

    def _call_hf_processor(self, hf_data, hf_kwargs):
        videos = hf_data.get("videos")
        if videos:
            cleaned, packed_pixels = [], 0
            for item in videos:
                frames, metadata = _video_with_metadata(item)
                if HY4_FRAME_PREFIXES_KEY in metadata:
                    packed_pixels = max(packed_pixels, int(np.prod(np.shape(frames)[:3])))
                    metadata = {
                        k: v for k, v in metadata.items() if not k.startswith("angelspec_")
                    }
                    item = (frames, metadata)
                cleaned.append(item)
            hf_data = {**hf_data, "videos": cleaned}
            if packed_pixels:
                videos_kwargs = dict(hf_kwargs.get("videos_kwargs") or {})
                budget = max(packed_pixels, int(hf_kwargs.get("max_pixels") or 0))
                videos_kwargs.update(do_resize=False, max_pixels=budget)
                hf_kwargs = {**hf_kwargs, "videos_kwargs": videos_kwargs}
        return original_call(self, hf_data, hf_kwargs)

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        updates = original_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        videos = mm_items.get("video") if "video" in mm_items else None
        metadata = [_video_with_metadata(item)[1] for item in (videos or ())]
        if not any(HY4_FRAME_PREFIXES_KEY in m for m in metadata):
            return updates
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def packed(update):
            native = update.replacement

            def replacement(index):
                if HY4_FRAME_PREFIXES_KEY not in metadata[index]:
                    return native(index)
                grid = out_mm_kwargs["video"][index]["video_grid_thw"].data
                tokens = _hy_v4_pretokenized_video_tokens(processor, grid, metadata[index])
                return PromptUpdateDetails.select_token_id(tokens, processor.video_token_id)

            return PromptReplacement(
                modality=update.modality, target=update.target, replacement=replacement
            )

        return [packed(u) if u.modality == "video" else u for u in updates]

    Proc._call_hf_processor = _call_hf_processor
    Proc._get_prompt_updates = _get_prompt_updates
    Proc._angelspec_pretokenized_video = True
    return True


PRETOKENIZED_MM_ENV = "ANGELSPEC_PRETOKENIZED_MM"


def _treat_token_prompts_as_expanded(cls) -> None:
    import functools
    import inspect

    original = cls.__dict__["_apply_hf_processor_main"]
    if "prompt" not in inspect.signature(original).parameters:
        # vLLM 0.32 applies prompt updates outside this method, which now only runs
        # the HF processor on the media.
        _locate_expanded_placeholders(cls)
        return

    @functools.wraps(original)
    def wrapped(self, prompt, *args, **kwargs):
        prompt_ids, mm_data, is_applied = original(self, prompt, *args, **kwargs)
        if not isinstance(prompt, str) and os.environ.get(PRETOKENIZED_MM_ENV) == "1":
            is_applied = True
        return prompt_ids, mm_data, is_applied

    cls._apply_hf_processor_main = wrapped
    cls._angelspec_pretokenized_mm = True


def _locate_expanded_placeholders(cls) -> None:
    """vLLM 0.32 tokenizes text prompts before multimodal processing and then always
    applies the prompt updates, so it cannot tell a token prompt AngelSpec already
    expanded; VllmEngine sets PRETOKENIZED_MM_ENV per batch to say which it sends."""
    import functools

    original = cls._maybe_apply_prompt_updates

    @functools.wraps(original)
    def wrapped(self, mm_items, mm_res):
        if os.environ.get(PRETOKENIZED_MM_ENV) != "1":
            return original(self, mm_items, mm_res)
        counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_res.kwargs, counts)
        self._validate_mm_updates(mm_res.prompt_updates, counts)
        placeholders = self._find_mm_placeholders(mm_res.prompt_ids, mm_res.prompt_updates)
        self._validate_mm_placeholders(placeholders, counts)
        return mm_res.prompt_ids, placeholders

    cls._maybe_apply_prompt_updates = wrapped
    cls._angelspec_pretokenized_mm = True


def apply_pretokenized_mm_prompts() -> bool:
    """Locate, not re-insert, media placeholders in AngelSpec's token prompts.

    Without deferred tokenization the data loader expands every image
    placeholder to the vision tower's token count before tokenizing (see
    ``expand_image_tokens``). vLLM marks token-id prompts as not yet updated and
    expands them again: each image's first placeholder grows into another full
    run, the features land in the wrong slots and every later token shifts.
    VllmEngine sets ``ANGELSPEC_PRETOKENIZED_MM=1``; token prompts are then
    treated as expanded, so vLLM finds the placeholders and fails loudly when
    their count disagrees with the image. Megatron Hy4 flash videos keep their packed
    frame prefixes (``apply_hy_v4_flash_pretokenized_video``).
    """
    try:
        from vllm.multimodal.processing.processor import BaseMultiModalProcessor
    except Exception:
        logger.exception("apply_pretokenized_mm_prompts: failed")
        return False
    targets = [BaseMultiModalProcessor]
    try:
        # Qwen2.5-Omni (and Qwen3-Omni, which inherits it) reimplement the method.
        from vllm.model_executor.models.qwen2_5_omni_thinker import (
            Qwen2_5OmniThinkerMultiModalProcessor,
        )

        targets.append(Qwen2_5OmniThinkerMultiModalProcessor)
    except Exception:
        pass
    for cls in targets:
        if "_apply_hf_processor_main" in cls.__dict__ and not cls.__dict__.get(
            "_angelspec_pretokenized_mm", False
        ):
            _treat_token_prompts_as_expanded(cls)
    return True


def apply_hy_v4_flash_batch_invariance() -> bool:
    """Admit Hy4 flash to vLLM's batch-invariant mode, which OPD's teacher engines run in.

    vLLM refuses the mode for attention backends that do not declare it, and Hy4 flash's
    do not: window and sparse attention read each query's own top-k rows, and the
    IndexPool scores every row and pools every group on its own, so their results do not
    depend on what else is in the batch. The recall's tie order, the one batch-dependent
    choice, is fixed by ``opd_vllm.configure_teacher``. The claim is checked end to end by
    ``scripts/hy_v4_flash_batch_invariance.py``. Only applied in processes started with
    ``VLLM_BATCH_INVARIANT=1``.
    """
    try:
        import vllm.envs as envs
        from vllm.models.hy_v4_flash.nvidia.indexpool_cache import (
            IndexPoolBackend,
            IndexPoolTailBackend,
        )
        from vllm.models.hy_v4_flash.nvidia.swa import (
            FlashSparseBackend,
            FlashSWABackend,
        )
    except Exception:
        return False
    if not envs.VLLM_BATCH_INVARIANT:
        return False
    for backend in (FlashSparseBackend, FlashSWABackend, IndexPoolBackend, IndexPoolTailBackend):
        backend.supports_batch_invariance = classmethod(lambda cls: True)
    return True


def apply_hy_v4_flash_extract_compat() -> bool:
    """Let Hy4 flash run vLLM's ``extract_hidden_states`` method.

    Flash's ``validate_runtime`` admits DFly as the only speculative method and no KV
    transfer. Hidden-state extraction drafts nothing: it submits prefill-only requests
    (``max_tokens=1``) and hands the captured states to AngelSpec's KV connector. Every
    other check still runs, and other speculative methods stay rejected.
    """
    try:
        import functools

        from vllm.models.hy_v4_flash.nvidia import model as flash_model
        from vllm.models.hy_v4_flash.nvidia import vl_model as flash_vl_model
    except Exception:
        return False
    original = flash_model.validate_runtime
    if getattr(original, "_angelspec_extract_compat", False):
        return True

    @functools.wraps(original)
    def validate_runtime(vllm_config):
        spec = vllm_config.speculative_config
        if spec is None or getattr(spec, "method", None) != "extract_hidden_states":
            return original(vllm_config)
        kv_transfer = vllm_config.kv_transfer_config
        object.__setattr__(vllm_config, "speculative_config", None)
        object.__setattr__(vllm_config, "kv_transfer_config", None)
        try:
            return original(vllm_config)
        finally:
            object.__setattr__(vllm_config, "speculative_config", spec)
            object.__setattr__(vllm_config, "kv_transfer_config", kv_transfer)

    validate_runtime._angelspec_extract_compat = True
    flash_model.validate_runtime = validate_runtime
    flash_vl_model.validate_runtime = validate_runtime
    return True


def apply_ray_worker_init_compat() -> bool:
    try:
        from vllm.v1.executor.multiproc_executor import WorkerProc
        from vllm.v1.executor.ray_executor_v2 import RayWorkerProc

        if getattr(RayWorkerProc, "_angelspec_init_compat", False):
            return True

        import functools

        original = RayWorkerProc.initialize_worker
        accepted = set(inspect.signature(WorkerProc.__init__).parameters) - {
            "self",
            "local_rank",
        }

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            self._init_kwargs = {
                key: value for key, value in self._init_kwargs.items() if key in accepted
            }
            return original(self, *args, **kwargs)

        RayWorkerProc.initialize_worker = wrapped
        RayWorkerProc._angelspec_init_compat = True
        return True
    except Exception:
        logger.exception("apply_ray_worker_init_compat: failed")
        return False


def apply_unquantized_moe_swiglu_clamp_compat() -> bool:
    """Thread ``FusedMoE.swiglu_limit`` into the unquantized quant config.

    vLLM's ``UnquantizedFusedMoEMethod.get_fused_moe_quant_config`` returns a
    config with ``gemm1_clamp_limit=None`` (the quantized methods set it), so the
    batched (EP/DP) MoE kernel asserts under a ``swigluclamp`` activation. Mirror
    the quantized behaviour without mutating the shared unquantized config.
    """
    try:
        import dataclasses
        import functools

        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )

        if getattr(UnquantizedFusedMoEMethod, "_angelspec_swiglu_clamp", False):
            return True

        original = UnquantizedFusedMoEMethod.get_fused_moe_quant_config

        @functools.wraps(original)
        def wrapped(self, layer):
            cfg = original(self, layer)
            limit = getattr(layer, "swiglu_limit", None)
            if (
                cfg is not None
                and limit is not None
                and getattr(cfg, "gemm1_clamp_limit", None) is None
            ):
                cfg = dataclasses.replace(cfg, gemm1_clamp_limit=float(limit))
            return cfg

        UnquantizedFusedMoEMethod.get_fused_moe_quant_config = wrapped
        UnquantizedFusedMoEMethod._angelspec_swiglu_clamp = True
        return True
    except Exception:
        logger.exception("apply_unquantized_moe_swiglu_clamp_compat: failed")
        return False


def apply_all(archs: Iterable[str] | None = None) -> None:
    """Entry point invoked by vLLM's ``vllm.general_plugins`` mechanism.

    Runs in every vLLM process. Injects EAGLE3 support for all known target
    architectures (or the provided subset).
    """
    from angelspec.inference.engine.deep_gemm_compat import (
        apply_deep_gemm_fp8_fp4_compat,
    )

    targets = list(archs) if archs is not None else list(_EAGLE3_TARGETS)
    apply_deep_gemm_fp8_fp4_compat()
    apply_ray_worker_init_compat()
    register_extra_hf_configs()
    apply_speculative_model_type_whitelist()
    apply_extract_hidden_states_pp_compat()
    apply_extract_hidden_states_pp_no_drafts()
    apply_vit_rotary_emb_fix()
    apply_qwen3_omni_thinker_aux()
    apply_qwen3_mtp()
    apply_kv_cache_no_pad_cache_only()
    apply_kv_block_zeroer_skip_cache_only()
    apply_kv_cache_diag()
    apply_hy_v4_flash_pretokenized_video()
    apply_pretokenized_mm_prompts()
    apply_hy_v4_flash_extract_compat()
    apply_hy_v4_flash_batch_invariance()
    from angelspec.inference.engine.opd_cache import install_cache_bound

    install_cache_bound()
    apply_unquantized_moe_swiglu_clamp_compat()
    for arch in targets:
        apply_eagle3_mixin(arch)
