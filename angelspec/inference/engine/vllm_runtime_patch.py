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

import multiprocessing
import os
from collections.abc import Callable, Iterable

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
    if name.startswith("VllmWorker") or "Worker" in name:
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

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
        hidden_states, residual = layer(positions, hidden_states, residual, idx=idx)
        self._maybe_add_hidden_state(aux_hidden_states, idx + 1, hidden_states, residual)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

    if getattr(envs, "VLLM_PRECISION_MODE", "HF") == "default":
        hidden_states, residual = self.norm(hidden_states, residual)
    else:
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states


# Map of model architecture -> injection spec. Each spec carries the candidate
# module path(s), the inner Model / outer ForCausalLM class names, and a
# model-specific ``forward`` override (see the note above). ``modules`` is an
# ordered list of candidate import paths: the first that imports wins.
_EAGLE3_TARGETS: dict[str, dict] = {
    "HYV3ForCausalLM": {
        "modules": [
            "vllm.model_executor.models.hy_v3",
        ],
        "causal_lm": "HYV3ForCausalLM",
        "inner_model": "HYV3Model",
        "forward": _hyv3_inner_forward,
    },
}

_injected_archs: set[str] = set()

# model_type values to whitelist for extract_hidden_states / eagle3 / dflash.
# vLLM hardcodes a supported list inside SpeculativeConfig._verify_args; we add
# ours at runtime instead of patching vLLM source.
_EXTRA_AUX_MODEL_TYPES: set[str] = {"hy_v3"}
_spec_validator_patched = False


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
            logger.info(
                "apply_speculative_model_type_whitelist: worker process — "
                "skipping VllmConfig rebuild to avoid profiling-stage hang."
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


def _build_enhanced_classes(
    module,
    causal_lm_name: str,
    inner_model_name: str,
    inner_forward: Callable,
):
    """Create EAGLE3-enhanced subclasses of the inner Model and outer ForCausalLM.

    ``inner_forward`` is the model-specific forward override registered in
    ``_EAGLE3_TARGETS`` (see the note at the top of this module). It is bound as
    the enhanced inner model's ``forward``; this module does NOT attempt to
    synthesize a forward that works across model families.
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

        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors=None,
            inputs_embeds=None,
        ):
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    EagleCausalLM.__name__ = causal_lm_name
    EagleCausalLM.__qualname__ = causal_lm_name

    return EagleCausalLM


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

        inner_forward = spec.get("forward")
        if inner_forward is None:
            logger.error(
                "apply_eagle3_mixin: %s has no registered forward override; "
                "cannot inject EAGLE3 safely. Add one to _EAGLE3_TARGETS.",
                arch,
            )
            return False

        enhanced = _build_enhanced_classes(
            module, spec["causal_lm"], spec["inner_model"], inner_forward
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


def apply_all(archs: Iterable[str] | None = None) -> None:
    """Entry point invoked by vLLM's ``vllm.general_plugins`` mechanism.

    Runs in every vLLM process. Injects EAGLE3 support for all known target
    architectures (or the provided subset).
    """
    targets = list(archs) if archs is not None else list(_EAGLE3_TARGETS)
    apply_speculative_model_type_whitelist()
    apply_qwen3_mtp()
    for arch in targets:
        apply_eagle3_mixin(arch)
