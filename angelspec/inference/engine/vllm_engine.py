# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
VLLM Ray actor engine for distributed deployment.

Uses vLLM's ``extract_hidden_states`` speculative decoding method with a
custom ``MooncakeHiddenStatesConnector`` KV Connector to capture intermediate
hidden states and store them directly to Mooncake via RDMA.

This replaces the previous worker-extension approach that monkey-patched
``model.forward``.  The new approach uses only public vLLM APIs
(``speculative_config`` + ``kv_transfer_config``) and is compatible with
MRV2, CUDA graphs, and ``torch.compile``.
"""

import gc
import os
import socket
from typing import Any

import ray
import torch
from omegaconf import DictConfig, OmegaConf

from angelspec.inference.engine.base import InferenceEngine
from angelspec.ray.ray_actor import RayActor
from angelspec.utils.logging import logger, setup_file_logging
from angelspec.utils.misc import get_default_eagle3_aux_layer_ids

_PROTECTION_ENGINE_KEYS = frozenset(
    {
        "model",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "nnodes",
        "node_rank",
        "master_addr",
        "master_port",
        "speculative_config",
        "kv_transfer_config",
    }
)


class VllmEngine(InferenceEngine, RayActor):
    """Ray actor wrapper for vLLM LLM engine with distributed deployment support.

    Uses vLLM's ``extract_hidden_states`` speculative method with a
    ``MooncakeHiddenStatesConnector`` to capture hidden states from selected
    model layers and write them directly to Mooncake.
    """

    def __init__(
        self,
        args,
        rank: int,
        base_gpu_id: int | None = None,
        num_gpus_per_engine: int = 1,
        node_rank: int = 0,
        engine_group: int = 0,
    ):
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.num_gpus_per_engine = num_gpus_per_engine
        self.node_rank = node_rank
        self._engine = None
        self._mooncake_config = None
        self._hidden_size = None
        self.local_gpu_id = None

        # Mark this process as the vLLM engine-core/driver so the runtime patch
        # can tell it apart from ray-spawned TP workers (both are ray actors under
        # the ray executor backend). Set before any `import vllm` in this process.
        os.environ["ANGELSPEC_VLLM_ENGINE_DRIVER"] = "1"

        setup_file_logging("inference", self.rank, group=engine_group)

    def init(
        self,
        mooncake_config=None,
        dist_init_addr: str | None = None,
        pre_allocated_port: int | None = None,
    ) -> None:
        if self.base_gpu_id is not None:
            self.local_gpu_id = self.setup_gpu(self.base_gpu_id)
            logger.info(
                f"VllmEngine rank {self.rank}: base_gpu_id={self.base_gpu_id}, using local GPU {self.local_gpu_id}"
            )

        self._mooncake_config = mooncake_config

        if mooncake_config is not None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = "localhost"
                logger.warning(
                    f"VllmEngine rank {self.rank}: failed to get local IP, using localhost"
                )

            mooncake_config.local_hostname = local_ip
            # Export env vars so worker processes (and the connector) can
            # initialize their own Mooncake stores via MooncakeConfig.from_env().
            mooncake_config.export_env()

            from angelspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address,
                mooncake_config.metadata_server,
            )

        mem_fraction = getattr(self.args, "vllm_mem_fraction_static", None)
        pp_size = getattr(self.args, "vllm_pp_size", 1)

        if self.args.aux_hidden_states_layers is not None:
            self.aux_hidden_state_layer_ids = list(self.args.aux_hidden_states_layers)
        else:
            self.aux_hidden_state_layer_ids = get_default_eagle3_aux_layer_ids(
                self.args.target_model_path
            )
            if self.rank == 0:
                logger.info(
                    f"Using default aux hidden state layer ids: {self.aux_hidden_state_layer_ids}"
                )

        from transformers import AutoConfig as _AC

        _cfg = _AC.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        _cfg = getattr(_cfg, "text_config", _cfg)

        # hy_v3 (HYV3) on the vLLM fork: HYV3Model.forward switches its final norm
        # on VLLM_PRECISION_MODE. The runtime EAGLE3 override (_hyv3_inner_forward)
        # is authoritative in the "HF" branch (manual residual add + single-input
        # norm); pin "HF" so the injected forward matches the un-instrumented
        # path exactly. Old vLLM (vllm-src 0.10) has no such env and ignores it.
        if getattr(_cfg, "model_type", None) == "hy_v3":
            os.environ.setdefault("VLLM_PRECISION_MODE", "HF")
            if self.rank == 0:
                logger.info(
                    "hy_v3 target: set VLLM_PRECISION_MODE=%s",
                    os.environ["VLLM_PRECISION_MODE"],
                )

        # Layer IDs use post-layer semantics: "capture the residual stream
        # after layer N runs".  vllm's capture hook fires at the INPUT of each
        # listed layer (= output of the previous layer), so we shift by +1 to
        # align with sglang's convention.
        # vllm's `_maybe_add_hidden_state` is called with `layer_idx + 1`
        # *after* each layer runs, so valid capture indices are
        # [0, num_hidden_layers]; we keep ids up to num_hidden_layers
        # (the pre-`norm` slot, see final-layer block below).
        num_layers = _cfg.num_hidden_layers
        self.aux_hidden_state_layer_ids = [
            lid + 1 for lid in self.aux_hidden_state_layer_ids if lid < num_layers
        ]
        if self.rank == 0:
            logger.info(
                f"Shifted aux layer ids +1 for vllm (post-layer → pre-next-layer): {self.aux_hidden_state_layer_ids}"
            )

        # Append the model's final layer to capture last_hidden_states
        # (pre-norm) for target logit computation.  Index `num_hidden_layers`
        # is vllm's reserved post-last-layer / pre-`norm` slot, so training
        # can apply the model's final norm itself on top of this.
        final_layer_id = num_layers
        if final_layer_id not in self.aux_hidden_state_layer_ids:
            self.aux_hidden_state_layer_ids.append(final_layer_id)
            if self.rank == 0:
                logger.info(
                    f"Appended final layer {final_layer_id} for last_hidden_states: {self.aux_hidden_state_layer_ids}"
                )

        nnodes = getattr(self.args, "vllm_nnodes", 1)
        backend = getattr(self.args, "vllm_distributed_executor_backend", None) or (
            "ray" if nnodes > 1 else "mp"
        )
        # ray: one actor drives all workers across nodes, so the factory passes the
        # full cross-node TP width as num_gpus_per_engine. mp: per-node width × nnodes.
        tp_size = (
            self.num_gpus_per_engine if backend == "ray" else nnodes * self.num_gpus_per_engine
        )

        logger.info(
            f"VllmEngine rank {self.rank}: BEFORE init - "
            f"base_gpu_id={self.base_gpu_id}, num_gpus={self.num_gpus_per_engine}, "
            f"tp_size={tp_size}, pp_size={pp_size}, nnodes={nnodes}, node_rank={self.node_rank}, "
            f"aux_hidden_state_layer_ids={self.aux_hidden_state_layer_ids}"
        )

        self._init_engine(
            tp_size, pp_size, nnodes, mem_fraction, dist_init_addr, pre_allocated_port
        )

        self._hidden_size = self._get_hidden_size_from_engine()

        logger.info(
            f"VllmEngine rank {self.rank}: initialized from {self.args.target_model_path} "
            f"(tp_size={tp_size}, aux_layers={self.aux_hidden_state_layer_ids}, hidden_size={self._hidden_size})"
        )

    def _init_engine(
        self,
        tp_size: int,
        pp_size: int,
        nnodes: int,
        mem_fraction: float | None,
        dist_init_addr: str | None,
        pre_allocated_port: int | None = None,
    ) -> None:
        """Initialize LLM with extract_hidden_states speculative config."""
        backend = getattr(self.args, "vllm_distributed_executor_backend", None) or (
            "ray" if nnodes > 1 else "mp"
        )
        # Pin vLLM's mp workers to the correct physical GPUs.  Without this,
        # all engines on a node see every GPU and their workers collide on
        # devices 0..tp_size-1.  Under the ray backend, vLLM's RayDistributedExecutor
        # sets each worker's CUDA_VISIBLE_DEVICES itself (from the placement group);
        # a driver-side value spanning non-local ids would corrupt device selection.
        if backend == "mp" and self.base_gpu_id is not None:
            gpu_ids = [str(self.base_gpu_id + i) for i in range(self.num_gpus_per_engine)]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            logger.info(
                f"VllmEngine rank {self.rank}: set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
            )

        from vllm import LLM

        # Packed tree-forward scoring engine (half on-policy MTP OPD): an
        # independent encoder-only prefill engine (is_causal=False → every layer
        # is EncoderOnlyAttention, no KV cache) on the FlexAttention backend so the
        # custom tree block_mask applies. No extract connector / speculative config.
        if getattr(self.args, "vllm_score_engine", False):
            engine_kwargs = {
                "model": self.args.target_model_path,
                "tensor_parallel_size": tp_size,
                "trust_remote_code": getattr(self.args, "trust_remote_code", True),
                "distributed_executor_backend": "mp",
                "disable_custom_all_reduce": True,
                "runner": "pooling",
                "attention_backend": "FLEX_ATTENTION",
                "hf_overrides": {"is_causal": False},
                "worker_extension_cls": (
                    "angelspec.inference.engine.score_worker_ext.ScorePackedWorkerExt"
                ),
                "enforce_eager": True,
                # Packed tree = trunk + all branches, much longer than the training
                # max_seq_length — size the encoder-only context generously.
                "max_model_len": int(getattr(self.args, "opd_score_max_model_len", 16384)),
            }
        else:
            engine_kwargs = {
                "model": self.args.target_model_path,
                "tensor_parallel_size": tp_size,
                "trust_remote_code": getattr(self.args, "trust_remote_code", True),
                "distributed_executor_backend": backend,
                "disable_custom_all_reduce": True,
                "speculative_config": {
                    "method": "extract_hidden_states",
                    "num_speculative_tokens": 1,
                    "draft_model_config": {
                        "hf_config": {
                            "eagle_aux_hidden_state_layer_ids": list(
                                self.aux_hidden_state_layer_ids
                            )
                        }
                    },
                },
                "kv_transfer_config": {
                    "kv_connector": "MooncakeHiddenStatesConnector",
                    "kv_connector_module_path": (
                        "angelspec.inference.engine.mooncake_hidden_states_connector"
                    ),
                    "kv_role": "kv_producer",
                },
            }

        # Score engine: minimal encoder-only pooling engine. Skip the whole
        # extract/generate common tail (max_num_seqs, prefix_caching, cudagraph
        # compilation, vllm_extra_args) — those conflict with the pooling runner's
        # SchedulerConfig. Only gpu_memory_utilization + timeout are relevant.
        if getattr(self.args, "vllm_score_engine", False):
            if mem_fraction is not None:
                engine_kwargs["gpu_memory_utilization"] = mem_fraction
            engine_kwargs["distributed_timeout_seconds"] = (
                getattr(self.args, "distributed_timeout_minutes", 10) * 60
            )
            self._engine = LLM(**engine_kwargs)
            logger.info(
                "VllmEngine rank %s: initialized encoder-only score engine "
                "(FlexAttention, pooling, is_causal=False, max_model_len=%s)",
                self.rank,
                engine_kwargs.get("max_model_len"),
            )
            return

        extra_args = getattr(self.args, "vllm_extra_args", None)
        if extra_args:
            if isinstance(extra_args, DictConfig):
                extra = OmegaConf.to_container(extra_args, resolve=True)
            else:
                extra = dict(extra_args) if not isinstance(extra_args, dict) else extra_args
            blocked = extra.keys() & _PROTECTION_ENGINE_KEYS
            if blocked:
                logger.warning(
                    f"vllm extra_args contains protected keys that will be ignored: "
                    f"{sorted(blocked)}. These are managed internally by AngelSpec."
                )
                extra = {k: v for k, v in extra.items() if k not in _PROTECTION_ENGINE_KEYS}
            engine_kwargs.update(extra)

        inference_batch_size = getattr(self.args, "inference_batch_size", None)
        if inference_batch_size is not None:
            if "max_num_seqs" not in engine_kwargs:
                engine_kwargs["max_num_seqs"] = inference_batch_size
                logger.info(
                    f"VllmEngine rank {self.rank}: defaulting "
                    f"max_num_seqs={inference_batch_size} from inference_batch_size"
                )

        engine_kwargs["enable_prefix_caching"] = False

        # Prefill-only: no decode graphs needed and piecewise adds launch
        # overhead not worthwhile for variable-length prefills.
        # Cap capture sizes when the user overrides the mode via extra_args.
        comp_cfg = engine_kwargs.get("compilation_config", {})
        if isinstance(comp_cfg, dict):
            if "cudagraph_mode" not in comp_cfg:
                comp_cfg["cudagraph_mode"] = "NONE"
            elif inference_batch_size is not None and "cudagraph_capture_sizes" not in comp_cfg:
                comp_cfg["cudagraph_capture_sizes"] = [
                    2**i
                    for i in range(inference_batch_size.bit_length())
                    if 2**i <= inference_batch_size
                ]
            engine_kwargs["compilation_config"] = comp_cfg

        if "distributed_timeout_seconds" not in engine_kwargs:
            timeout_min = getattr(self.args, "distributed_timeout_minutes", 10)
            engine_kwargs["distributed_timeout_seconds"] = timeout_min * 60

        max_seq_length = getattr(self.args, "max_seq_length", None)
        if max_seq_length and not getattr(self.args, "vllm_score_engine", False):
            engine_kwargs["max_model_len"] = max_seq_length

        # Multi-node rendezvous is only for the legacy mp per-node-LLM path.
        # Under the ray backend a single EngineCore owns all workers and vLLM's
        # RayDistributedExecutor handles placement/rendezvous itself — passing
        # master_addr/nnodes/node_rank there is wrong (and node_rank triggers the
        # follower EngineCore that asserts on collective_rpc).
        if backend == "mp":
            if dist_init_addr:
                host, port_str = dist_init_addr.rsplit(":", 1)
                engine_kwargs["master_addr"] = host
                engine_kwargs["master_port"] = int(port_str)
            elif pre_allocated_port is not None:
                engine_kwargs["master_port"] = pre_allocated_port

            if nnodes > 1:
                engine_kwargs["nnodes"] = nnodes
                engine_kwargs["node_rank"] = self.node_rank

        if mem_fraction is not None:
            engine_kwargs["gpu_memory_utilization"] = mem_fraction
        else:
            # vLLM's profiler doesn't account for the connector's runtime
            # allocations (extracting hidden states from KV cache during
            # save_kv_layer).  Auto-compute a utilization that reserves room.
            engine_kwargs["gpu_memory_utilization"] = self._compute_mem_fraction(engine_kwargs)

        self._engine = LLM(**engine_kwargs)
        logger.info(
            f"VllmEngine rank {self.rank}: initialized extract_hidden_states mode "
            f"with layers={self.aux_hidden_state_layer_ids}"
        )

    _VLLM_DEFAULT_GPU_MEMORY_UTILIZATION = 0.9

    def _compute_mem_fraction(self, engine_kwargs: dict) -> float:
        """Auto-compute gpu_memory_utilization with connector overhead reserved.

        Starts from vLLM's default (0.9) and subtracts the estimated peak
        memory of the MooncakeHiddenStatesConnector's ``save_kv_layer``,
        which creates temporary GPU tensors that vLLM's profiler doesn't
        account for.
        """
        base = self._VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
        max_len = engine_kwargs.get("max_model_len")
        if max_len is None:
            return base

        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        hf_cfg = getattr(hf_cfg, "text_config", hf_cfg)
        hidden_size = hf_cfg.hidden_size
        num_aux_layers = len(self.aux_hidden_state_layer_ids)

        # Peak per-request: _extract_from_kv_cache materialises
        # (seq_len, num_aux_layers * hidden_size) in bf16.
        connector_bytes = max_len * num_aux_layers * hidden_size * 2
        # 2x safety for PyTorch allocator fragmentation + small extras
        reserved_bytes = int(connector_bytes * 2)

        total_gpu_bytes = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory
        overhead_frac = reserved_bytes / total_gpu_bytes
        adjusted = base - overhead_frac
        adjusted = max(adjusted, 0.4)

        logger.info(
            f"VllmEngine rank {self.rank}: auto gpu_memory_utilization={adjusted:.3f} "
            f"(reserving {reserved_bytes / (1 << 30):.1f} GiB for connector: "
            f"{max_len} tokens × {num_aux_layers} layers × {hidden_size} hidden × bf16 × 2x safety)"
        )
        return adjusted

    def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str | None] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict]:
        """Generate hidden states for training data.

        Hidden states are captured by vLLM's ``extract_hidden_states``
        speculative method and stored to Mooncake by the
        ``MooncakeHiddenStatesConnector``.  Metadata comes back in
        ``output.kv_transfer_params``.
        """
        if self._engine is None:
            raise RuntimeError("VllmEngine not initialized. Call init() first.")

        if (input_ids_ref is None) == (formatted_prompts is None):
            raise ValueError("Exactly one of input_ids_ref or formatted_prompts must be set")

        use_prompts = formatted_prompts is not None
        input_ids_list: list[torch.Tensor] | None = None

        if use_prompts:
            batch_size = len(formatted_prompts)
        else:
            if isinstance(input_ids_ref, ray.ObjectRef):
                input_ids_list = ray.get(input_ids_ref)
            else:
                input_ids_list = input_ids_ref
            if input_ids_list is None:
                raise ValueError("input_ids_ref resolved to None")
            batch_size = len(input_ids_list)

        if isinstance(data_id, str):
            data_ids = [f"{data_id}_{i}" for i in range(batch_size)]
        elif len(data_id) == batch_size:
            data_ids = list(data_id)
        else:
            raise ValueError(
                f"data_id length {len(data_id)} does not match batch size {batch_size}"
            )

        # Skip samples whose prompt leaves no room for the >=1 token vLLM must
        # generate to extract hidden states: vLLM rejects the whole request when
        # prompt_len + 1 > max_model_len (== max_seq_length), and the offending
        # request can stall the run. SglEngine guards the same way. The dataset
        # reserves 1 token (max_seq_length - 1), so only pathological samples
        # (e.g. media-expanded prompts) reach here.
        max_seq_length = getattr(self.args, "max_seq_length", None)
        active_indices = list(range(batch_size))
        if max_seq_length:
            prompt_lengths = self._prompt_token_lengths(
                use_prompts, formatted_prompts, input_ids_list
            )
            active_indices = []
            for i, prompt_len in enumerate(prompt_lengths):
                if prompt_len is not None and prompt_len + 1 > max_seq_length:
                    logger.warning(
                        f"VllmEngine rank {self.rank}: skipping data_id={data_ids[i]} "
                        f"because prompt_tokens={prompt_len} leaves no room for the "
                        f"generated token within max_seq_length={max_seq_length}"
                    )
                    continue
                active_indices.append(i)

        if not active_indices:
            return []

        data_ids = [data_ids[i] for i in active_indices]
        active_formatted_prompts = (
            [formatted_prompts[i] for i in active_indices] if use_prompts else None
        )
        active_input_ids_list = (
            [input_ids_list[i] for i in active_indices] if not use_prompts else None
        )
        active_multimodal_inputs = (
            [multimodal_inputs[i] for i in active_indices]
            if multimodal_inputs is not None
            else None
        )
        active_packed_loss_mask_list = (
            [packed_loss_mask_list[i] for i in active_indices]
            if packed_loss_mask_list is not None
            else None
        )
        batch_size = len(active_indices)

        prompts = self._build_prompts(
            formatted_prompts=active_formatted_prompts,
            input_ids_list=active_input_ids_list,
            multimodal_inputs=active_multimodal_inputs,
            batch_size=batch_size,
        )

        from vllm import SamplingParams

        sampling_params = SamplingParams(max_tokens=1, temperature=0)

        # Build packed_loss_mask_map for result assembly
        packed_loss_mask_map: dict[str, str | None] = {}
        if active_packed_loss_mask_list is not None:
            for i, did in enumerate(data_ids):
                if i < len(active_packed_loss_mask_list):
                    packed_loss_mask_map[did] = active_packed_loss_mask_list[i]

        outputs = self._engine.generate(prompts, sampling_params, use_tqdm=False)

        results = []
        for i, output in enumerate(outputs):
            seq_len = len(output.prompt_token_ids)
            did = data_ids[i]

            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is None:
                logger.error(
                    f"VllmEngine rank {self.rank}: No kv_transfer_params for data_id={did}. "
                    f"The MooncakeHiddenStatesConnector may not have stored this request."
                )
                continue

            mooncake_key = kv_params.get("mooncake_key", did)
            tensor_shapes = kv_params.get("tensor_shapes", {})
            tensor_dtypes = kv_params.get("tensor_dtypes", {})

            result: dict[str, Any] = {
                "mooncake_key": mooncake_key,
                "tensor_shapes": tensor_shapes,
                "tensor_dtypes": tensor_dtypes,
                "data_id": did,
                "seq_len": seq_len,
            }

            packed_loss_mask = packed_loss_mask_map.get(did)
            if packed_loss_mask is not None:
                result["packed_loss_mask"] = packed_loss_mask

            input_ids_from_kv = kv_params.get("input_ids_list")
            if input_ids_from_kv is not None:
                result["input_ids_list"] = input_ids_from_kv
            else:
                result["input_ids_list"] = list(output.prompt_token_ids)

            results.append(result)

        logger.debug(
            f"VllmEngine rank {self.rank}: generated {len(results)} mooncake results for data_ids={data_ids}"
        )
        return results

    def _normalize_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 2 and input_ids.shape[0] == 1:
            return input_ids.squeeze(0)
        if input_ids.dim() == 1:
            return input_ids
        raise ValueError(f"Unexpected input_ids shape: {input_ids.shape}")

    @staticmethod
    def _resolve_media(items: list, fetch_fn) -> list:
        """Resolve URL strings to loaded objects via *fetch_fn*, drop None entries."""
        resolved = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, str):
                resolved.append(fetch_fn(item))
            else:
                resolved.append(item)
        return resolved

    @staticmethod
    def _to_vllm_multi_modal_data(mm_input: dict | None) -> dict | None:
        """Convert AngelSpec multimodal payload to vLLM ``multi_modal_data``.

        AngelSpec shape:  ``{"images": [...], "videos": [...]}``
        vLLM shape:       ``{"image": <PIL.Image|list>, "video": ...}``

        URL strings are resolved to PIL Images / video objects via
        ``vllm.multimodal.utils.fetch_image`` / ``fetch_video`` so that
        ``LLM.generate()`` receives the data types it expects.
        None entries (from incomplete media blocks) are filtered out.
        """
        if not mm_input:
            return None
        mm_data: dict = {}
        images = mm_input.get("images")
        if images:
            from vllm.multimodal.utils import fetch_image

            loaded = VllmEngine._resolve_media(images, fetch_image)
            if loaded:
                mm_data["image"] = loaded[0] if len(loaded) == 1 else loaded
        videos = mm_input.get("videos")
        if videos:
            try:
                from vllm.multimodal.utils import fetch_video

                loaded = VllmEngine._resolve_media(videos, fetch_video)
            except ImportError:
                loaded = [v for v in videos if v is not None]
            if loaded:
                mm_data["video"] = loaded[0] if len(loaded) == 1 else loaded
        return mm_data or None

    def _prompt_token_lengths(
        self,
        use_prompts: bool,
        formatted_prompts: list[str] | None,
        input_ids_list: list[torch.Tensor] | None,
    ) -> list[int | None]:
        """Token length per request, for the pre-generate length guard.

        Returns None for a request whose length can't be determined (e.g. the
        tokenizer is unavailable); callers treat None as "don't skip".
        """
        if not use_prompts:
            assert input_ids_list is not None
            return [int(self._normalize_input_ids(ids).numel()) for ids in input_ids_list]

        try:
            tokenizer = self._engine.get_tokenizer()
        except Exception:
            tokenizer = None
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            return [None for _ in formatted_prompts or []]

        lengths: list[int | None] = []
        for prompt in formatted_prompts or []:
            try:
                try:
                    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                except TypeError:
                    token_ids = tokenizer.encode(prompt)
                lengths.append(len(token_ids))
            except Exception as exc:
                logger.warning(
                    f"VllmEngine rank {self.rank}: failed to estimate prompt token "
                    f"length before generate ({exc!r}); letting vLLM tokenize it."
                )
                lengths.append(None)
        return lengths

    def _build_prompts(
        self,
        formatted_prompts: list[str] | None,
        input_ids_list: list[torch.Tensor] | None,
        multimodal_inputs: list[dict | None] | None,
        batch_size: int,
    ) -> list:
        """Assemble per-request vLLM prompt dicts, attaching multimodal data when present."""
        if multimodal_inputs is not None and len(multimodal_inputs) != batch_size:
            raise ValueError(
                f"multimodal_inputs length {len(multimodal_inputs)} does not match batch size {batch_size}"
            )

        prompts: list = []
        for i in range(batch_size):
            if formatted_prompts is not None:
                prompt_dict: dict = {"prompt": formatted_prompts[i]}
            else:
                prompt_dict = {
                    "prompt_token_ids": self._normalize_input_ids(input_ids_list[i]).tolist()
                }

            if multimodal_inputs is not None:
                mm_data = self._to_vllm_multi_modal_data(multimodal_inputs[i])
                if mm_data is not None:
                    prompt_dict["multi_modal_data"] = mm_data

            prompts.append(prompt_dict)
        return prompts

    def health_check(self, timeout: float = 5.0) -> bool:
        return self._engine is not None

    def shutdown(self) -> None:
        if self._engine is not None:
            try:
                llm_engine = getattr(self._engine, "llm_engine", None)
                engine_core = getattr(llm_engine, "engine_core", None)

                if engine_core is not None and hasattr(engine_core, "shutdown"):
                    engine_core.shutdown()
                elif llm_engine is not None and hasattr(llm_engine, "shutdown"):
                    llm_engine.shutdown()
                elif hasattr(self._engine, "close"):
                    self._engine.close()
            except Exception as e:
                logger.warning(f"VllmEngine rank {self.rank}: Error during engine shutdown: {e}")
            finally:
                self._engine = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(f"VllmEngine rank {self.rank}: shutdown complete")

    def get_status(self) -> dict:
        return {
            "rank": self.rank,
            "initialized": self._engine is not None,
            "base_gpu_id": self.base_gpu_id,
            "hidden_size": self._hidden_size,
        }

    def score_packed(
        self, packed_ids, positions, doc_ids, anchor_of, score_index, trunk_doc_of=None
    ):
        """Packed tree-forward scoring (requires vllm_score_engine=True).

        Runs the encoder-only target forward over one packed row (one OR many
        sequences' [trunk | branches] trees) with the tree mask and returns the
        selected pre-lm_head hidden as a nested list (M, H). ``trunk_doc_of``
        enables multi-sequence batching (cross-sequence isolation). Args are plain
        int lists (collective_rpc mangles tensors).
        """
        out = self._engine.collective_rpc(
            "score_packed",
            args=(packed_ids, positions, doc_ids, anchor_of, score_index, trunk_doc_of),
        )
        return out[0]

    def _get_hidden_size_from_engine(self) -> int:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"Could not determine hidden_size from model config: {self.args.target_model_path}"
            )
        return hidden_size
