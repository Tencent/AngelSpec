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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""KV Connector that writes hidden states directly to Mooncake.

vLLM discovers this connector via ``kv_connector_module_path`` in the
``kv_transfer_config`` dict -- no registration in vLLM's factory needed.

Architecture note: vLLM creates separate connector instances for the scheduler
process and each worker process.  Scheduler-side methods (``build_connector_meta``,
``request_finished``) run on one instance; worker-side methods (``save_kv_layer``,
``wait_for_save``) run on another.  They do NOT share state.  Metadata returned
by ``request_finished`` must therefore be pre-computed on the scheduler side.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)

HIDDEN_STATES_DTYPE_STR = "bfloat16"


def _sanitize_mooncake_key(key: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    if sanitized and sanitized[0].isdigit():
        sanitized = "k" + sanitized
    return sanitized


def _extract_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Extract data from KV cache.

    Assumes kv_cache shape: (num_pages, page_size, num_heads, head_size)
    """
    padded_kv = kv_cache.flatten(0, 1)[slot_mapping]
    return padded_kv[:num_tokens]


def _slot_mapping_from_block_ids(
    block_ids: list[int],
    page_size: int,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute slot_mapping from accumulated block_ids instead of attn_metadata.

    We cannot use ``attn_metadata.slot_mapping`` for two reasons:
    1. Chunked prefill: attn_metadata only has the current chunk's slots, but
       we need the full sequence's mapping (block_ids are accumulated across
       chunks in ``build_connector_meta``).
    2. HMA models: ``cache_config.block_size`` differs from the CacheOnly
       group's actual page_size.  Reading ``page_size`` from
       ``kv_layer.shape[1]`` handles both cases correctly.
    """
    block_ids_gpu = torch.tensor(block_ids, dtype=torch.int64, device=device)
    offsets = torch.arange(page_size, dtype=torch.int64, device=device)
    return (block_ids_gpu.unsqueeze(1) * page_size + offsets).flatten()[:num_tokens]


@dataclass
class _ReqMeta:
    req_id: str
    token_ids: torch.Tensor
    block_ids: list[int] = field(default_factory=list)


@dataclass
class MooncakeConnectorMetadata(KVConnectorMetadata):
    requests: list[_ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        req_id: str,
        token_ids: list[int],
        block_ids: list[int],
    ) -> None:
        self.requests.append(
            _ReqMeta(
                req_id=req_id,
                token_ids=torch.tensor(token_ids),
                block_ids=list(block_ids),
            )
        )


class MooncakeHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """KV Connector that stores extracted hidden states directly to Mooncake.

    Must be used with vLLM's ``extract_hidden_states`` speculative method.
    Mooncake connection parameters are read from environment variables
    (exported by AngelSpec's VllmEngine before creating the LLM instance).
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self.cache_layers: list[str] = []
        self._cache_layer_group_id: int = self._find_cache_layer_group_id(kv_cache_config)

        assert self._vllm_config.speculative_config is not None, (
            "MooncakeHiddenStatesConnector requires 'extract_hidden_states' speculative method"
        )
        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self._layer_ids = list(getattr(spec_config, "eagle_aux_hidden_state_layer_ids", []))
        self.num_hidden_states = len(self._layer_ids)
        self._hidden_size = vllm_config.model_config.get_hidden_size()

        # The last aux layer is the model's final layer (appended by
        # VllmEngine for last_hidden_states capture).  Training hidden
        # states use the remaining layers.
        self._num_training_layers = max(self.num_hidden_states - 1, 1)

        # Scheduler-side state: track requests and pre-computed metadata
        self._active_requests: dict[str, Any] = {}
        self._req_blocks: dict[str, list[int]] = {}
        self._req_metadata: dict[str, dict[str, Any]] = {}

        self._mooncake_store = None
        self._mooncake_setup_done = False
        self._tp_rank: int | None = None

    @staticmethod
    def _find_cache_layer_group_id(kv_cache_config) -> int:
        """Find the KV cache group that contains the CacheOnlyAttentionLayer.

        When ``kv_cache_config`` is available (worker side), we look for the
        group whose layer name contains ``cache_only_layers``.  On the
        scheduler side (``kv_cache_config is None``) we return ``None``
        and resolve it lazily in ``build_connector_meta`` by picking the
        group with ``ceil(num_tokens / block_size)`` blocks.
        """
        if kv_cache_config is None:
            return None  # type: ignore[return-value]
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                if "cache_only_layers" in name:
                    logger.info(
                        f"Cache-only layer found in KV group {gid}: {name} "
                        f"(total groups={len(kv_cache_config.kv_cache_groups)})"
                    )
                    return gid
        logger.warning(
            f"Cache-only layer NOT found in KV cache groups "
            f"(groups={[g.layer_names for g in kv_cache_config.kv_cache_groups]})"
        )
        return None  # type: ignore[return-value]

    def _get_tp_rank(self) -> int:
        if self._tp_rank is None:
            try:
                from vllm.distributed import get_tensor_model_parallel_rank

                self._tp_rank = get_tensor_model_parallel_rank()
            except Exception:
                self._tp_rank = 0
        return self._tp_rank

    def _ensure_mooncake_store(self) -> bool:
        if self._mooncake_setup_done:
            return self._mooncake_store is not None

        if self._get_tp_rank() != 0:
            self._mooncake_setup_done = True
            return False

        if not os.environ.get("MOONCAKE_MASTER_SERVER") and not os.environ.get(
            "MOONCAKE_MASTER_HOST"
        ):
            logger.warning(
                "MooncakeHiddenStatesConnector: no MOONCAKE_MASTER_SERVER env var; hidden states will NOT be stored."
            )
            self._mooncake_setup_done = True
            return False

        try:
            from angelspec.config.mooncake_config import MooncakeConfig
            from angelspec.transfer.mooncake.eagle_store import EagleMooncakeStore

            config = MooncakeConfig.from_env()
            self._mooncake_store = EagleMooncakeStore(config)

            device: torch.device | None = None
            if torch.cuda.is_initialized():
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self._mooncake_store.setup(device=device)
            self._mooncake_setup_done = True
            logger.info(
                f"MooncakeHiddenStatesConnector: store initialized (master={config.master_server_address})"
            )
            return True
        except Exception:
            logger.exception("MooncakeHiddenStatesConnector: failed to init store")
            self._mooncake_setup_done = True
            return False

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self):
        if self._mooncake_store is not None:
            self._mooncake_store.flush()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )

        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(self.cache_layers)}"
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name not in self.cache_layers:
            return

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )

        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata)

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, MooncakeConnectorMetadata)

        if not self._ensure_mooncake_store():
            if self._get_tp_rank() == 0:
                logger.warning("save_kv_layer: Mooncake store not available, skipping")
            return

        # Use tensor's actual page_size, not cache_config.block_size
        # (they differ on HMA models).
        page_size = kv_layer.shape[1]

        for request in connector_metadata.requests:
            num_tokens = request.token_ids.shape[0]
            mooncake_key = _sanitize_mooncake_key(request.req_id)

            # Recompute from accumulated block_ids — attn_metadata.slot_mapping
            # only covers the current chunk, not the full sequence.
            slot_mapping = _slot_mapping_from_block_ids(
                request.block_ids,
                page_size,
                num_tokens,
                device=kv_layer.device,
            )
            num_slots = slot_mapping.shape[0]

            # With chunked prefill, save_kv_layer is called per chunk.
            # Skip partial chunks — only store when all blocks are allocated.
            if num_slots < num_tokens:
                continue

            hidden_states_3d = _extract_from_kv_cache(kv_layer, slot_mapping, num_tokens)

            all_hidden = hidden_states_3d.reshape(num_tokens, -1)

            # Split: first N-1 aux layers → draft model input,
            # last aux layer (final model layer) → target logit computation
            split_at = self._num_training_layers * self._hidden_size
            hidden_states = all_hidden[:, :split_at]
            last_hidden_states = all_hidden[:, -self._hidden_size :]

            input_ids = request.token_ids.to(hidden_states.device)

            try:
                self._mooncake_store.put(
                    key=mooncake_key,
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    last_hidden_states=last_hidden_states,
                    target=None,
                )
            except Exception:
                logger.exception(
                    f"save_kv_layer: failed to store to Mooncake for {request.req_id}"
                )

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert num_external_tokens == 0, "This connector is store-only"

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            group_sizes = [len(g) for g in new_req.block_ids]
            gid = self._cache_layer_group_id
            if gid is None:
                # On the scheduler side kv_cache_config is unavailable, so we
                # pick the group with the most blocks.  The CacheOnly group
                # uses the smallest page_size and therefore always has the
                # highest block count for a given token count.
                gid = max(range(len(new_req.block_ids)), key=lambda i: len(new_req.block_ids[i]))
                self._cache_layer_group_id = gid
                logger.warning(
                    f"Resolved cache-only KV group id={gid} (group_sizes={group_sizes})"
                )
            meta.add_request(
                new_req.req_id,
                token_ids=token_ids,
                block_ids=new_req.block_ids[gid],
            )
            logger.debug(
                "build_connector_meta: req_id=%s key=%s num_tokens=%d gid=%d group_sizes=%s chosen_blocks=%d",
                new_req.req_id,
                _sanitize_mooncake_key(new_req.req_id),
                len(token_ids),
                gid,
                group_sizes,
                len(new_req.block_ids[gid]),
            )
            self._active_requests[new_req.req_id] = new_req
            self._req_blocks[new_req.req_id] = list(new_req.block_ids[gid])

            seq_len = len(token_ids)
            training_hidden_size = self._num_training_layers * self._hidden_size
            mooncake_key = _sanitize_mooncake_key(new_req.req_id)
            self._req_metadata[new_req.req_id] = {
                "mooncake_key": mooncake_key,
                "tensor_shapes": {
                    "hidden_states": (seq_len, training_hidden_size),
                    "input_ids": (seq_len,),
                    "last_hidden_states": (seq_len, self._hidden_size),
                },
                "tensor_dtypes": {
                    "hidden_states": HIDDEN_STATES_DTYPE_STR,
                    "input_ids": "int64",
                    "last_hidden_states": HIDDEN_STATES_DTYPE_STR,
                },
                "num_layers": self.num_hidden_states,
                "input_ids_list": token_ids,
            }

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._active_requests:
                continue

            new_block_ids = cached_reqs.new_block_ids[i]
            if new_block_ids is None:
                continue

            cached_req = self._active_requests[req_id]
            req_block_ids = self._req_blocks[req_id]

            block_ids = new_block_ids[self._cache_layer_group_id]
            req_block_ids.extend(block_ids)

            meta.add_request(
                req_id=req_id,
                token_ids=cached_req.prompt_token_ids or [],
                block_ids=req_block_ids,
            )

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        _ = self._active_requests.pop(req_id, None)
        _ = self._req_blocks.pop(req_id, None)

        mooncake_meta = self._req_metadata.pop(req_id, None)
        return False, mooncake_meta

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        group0_ids = block_ids[0] if block_ids else []
        return self.request_finished(request, group0_ids)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        return "NHD"
