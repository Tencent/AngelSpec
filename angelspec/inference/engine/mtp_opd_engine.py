"""Hy4 flash decoder prefill / branch verification, reusing the current VL pipeline."""

import os

import torch
from omegaconf import DictConfig, OmegaConf

from angelspec.inference.engine.opd_utils import prefix_media
from angelspec.inference.engine.vllm_engine import (
    _PROTECTION_ENGINE_KEYS as _BASE_PROTECTION_ENGINE_KEYS,
)
from angelspec.inference.engine.vllm_engine import VllmEngine, one_request_per_step
from angelspec.utils.logging import logger

_HIDDEN_CAPTURE_EXT = "angelspec.inference.engine.hidden_capture_worker_ext.HiddenCaptureWorkerExt"
_PROTECTION_ENGINE_KEYS = _BASE_PROTECTION_ENGINE_KEYS | {
    "model",
    "tensor_parallel_size",
    "worker_extension_cls",
    "enforce_eager",
    "enable_prefix_caching",
    "max_model_len",
    "runner",
    "speculative_config",
    "kv_transfer_config",
    "distributed_executor_backend",
}


class MTPOPDEngine(VllmEngine):
    def _init_engine(
        self, tp_size, pp_size, nnodes, mem_fraction, dist_init_addr, pre_allocated_port=None
    ):
        import vllm.envs as envs
        from vllm import LLM

        if not envs.VLLM_BATCH_INVARIANT:
            raise ValueError("MTP OPD prefill and verifier require VLLM_BATCH_INVARIANT=1")
        if pp_size != 1 or nnodes != 1:
            raise ValueError("OPD target engines require single-node TP, PP=1")
        self._engine_mode = (
            "verify" if getattr(self.args, "mtp_opd_verify_engine", False) else "hidden_capture"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(self.base_gpu_id + i) for i in range(tp_size)
        )
        self._init_hidden_capture_engine(
            LLM, tp_size, nnodes, mem_fraction, dist_init_addr, pre_allocated_port
        )

    def _branch_prompt(self, base_ids, prefix, tail, media, media_cache=None):
        mm = prefix_media(base_ids, prefix, media)
        prompt = {"prompt_token_ids": base_ids[:prefix] + list(tail)}
        if mm:
            key = (len(mm.get("images") or []), len(mm.get("videos") or []))
            if media_cache is not None and key in media_cache:
                resolved = media_cache[key]
            else:
                resolved = self._to_vllm_multi_modal_data(mm)
                if media_cache is not None:
                    media_cache[key] = resolved
            if resolved:
                prompt["multi_modal_data"] = resolved
        return prompt

    def _init_hidden_capture_engine(
        self, LLM, tp_size, nnodes, mem_fraction, dist_init_addr, pre_allocated_port
    ) -> None:
        verify = self._engine_mode == "verify"
        max_seq_length = int(getattr(self.args, "max_seq_length", 32768))
        num_steps = int(getattr(self.args, "mtp_opd_block_size", 5))
        engine_kwargs = {
            "model": self.args.target_model_path,
            "tensor_parallel_size": tp_size,
            "trust_remote_code": getattr(self.args, "trust_remote_code", True),
            "distributed_executor_backend": "mp",
            "disable_custom_all_reduce": True,
            # The capture wraps the eager model forward (no CUDA graphs).
            "enforce_eager": True,
            "worker_extension_cls": _HIDDEN_CAPTURE_EXT,
            # verify: branch prompts share the trunk prefix; prefill: every
            # position must actually be computed to be captured.
            "enable_prefix_caching": verify,
            "max_model_len": max_seq_length + num_steps,
            "gpu_memory_utilization": mem_fraction if mem_fraction is not None else 0.85,
            "distributed_timeout_seconds": getattr(self.args, "distributed_timeout_minutes", 10)
            * 60,
        }
        if getattr(self.args, "vllm_enable_multimodal", False):
            for attr, key in (
                ("vllm_mm_processor_kwargs", "mm_processor_kwargs"),
                ("vllm_limit_mm_per_prompt", "limit_mm_per_prompt"),
            ):
                value = getattr(self.args, attr, None)
                if value:
                    engine_kwargs[key] = (
                        OmegaConf.to_container(value, resolve=True)
                        if isinstance(value, DictConfig)
                        else dict(value)
                    )
        if not verify:
            inference_batch_size = getattr(self.args, "inference_batch_size", None)
            if inference_batch_size is not None:
                engine_kwargs["max_num_seqs"] = int(inference_batch_size)
        extra_args = getattr(self.args, "vllm_extra_args", None)
        if extra_args:
            extra = (
                OmegaConf.to_container(extra_args, resolve=True)
                if isinstance(extra_args, DictConfig)
                else dict(extra_args)
            )
            engine_kwargs.update(
                {k: v for k, v in extra.items() if k not in _PROTECTION_ENGINE_KEYS}
            )
        if verify:
            # vllm_extra_args tune the prefill engines; verify batches are many
            # short requests over cached prefixes.
            engine_kwargs["max_num_seqs"] = int(
                getattr(self.args, "mtp_opd_verify_max_num_seqs", 1024)
            )
            engine_kwargs["max_num_batched_tokens"] = int(
                getattr(self.args, "mtp_opd_verify_max_num_batched_tokens", 65536)
            )
            # Every branch re-hashes its whole (up to max_seq_length) prompt.
            engine_kwargs["prefix_caching_hash_algo"] = "xxhash"
        if one_request_per_step(self.args.target_model_path):
            engine_kwargs["max_num_seqs"] = 1
        if dist_init_addr:
            host, port_str = dist_init_addr.rsplit(":", 1)
            engine_kwargs["master_addr"] = host
            engine_kwargs["master_port"] = int(port_str)
        elif pre_allocated_port is not None:
            engine_kwargs["master_port"] = pre_allocated_port
        self._engine = LLM(**engine_kwargs)
        self._capture_last_n = (num_steps - 1) if verify else -1
        self._engine.collective_rpc("hc_configure", args=(self._capture_last_n,))
        logger.info(
            f"VllmEngine rank {self.rank}: {self._engine_mode} engine ready "
            f"(capture_last_n={self._capture_last_n}, prefix_caching={verify}, "
            f"max_model_len={engine_kwargs['max_model_len']})"
        )

    def _generate_hidden_capture(self, prompts, data_ids, packed_loss_mask_map) -> list[dict]:
        """Prefill engine: capture every prompt position and write it to Mooncake."""
        import re
        import uuid

        from vllm import SamplingParams

        outputs = self._engine.generate(
            prompts, SamplingParams(max_tokens=1, temperature=0, detokenize=False), use_tqdm=False
        )
        items = []
        for did, out in zip(data_ids, outputs):
            key = re.sub(r"[^a-zA-Z0-9_-]", "_", f"hc_{did}_{uuid.uuid4().hex[:8]}")
            items.append((out.request_id, key, list(out.prompt_token_ids)))
        meta = self._engine.collective_rpc("hc_flush_to_mooncake", args=(items,))[0]
        results = []
        for did, (_, key, ids) in zip(data_ids, items):
            info = meta.get(key)
            if info is None:
                raise RuntimeError(f"Incomplete OPD prefill hidden capture for {did}")
            result = {
                "mooncake_key": key,
                "tensor_shapes": info["shapes"],
                "tensor_dtypes": info["dtypes"],
                "data_id": did,
                "seq_len": len(ids),
                "input_ids_list": ids,
            }
            if packed_loss_mask_map.get(did) is not None:
                result["packed_loss_mask"] = packed_loss_mask_map[did]
            results.append(result)
        return results

    def opd_warm(self, base_ids: list[int], media=None) -> float:
        """Verify engine: prefill the trunk once so branch requests hit the prefix cache.

        Returns the elapsed seconds.
        """
        import time

        from vllm import SamplingParams

        t0 = time.time()
        self._engine.generate(
            [self._branch_prompt(base_ids, len(base_ids), [], media)],
            SamplingParams(max_tokens=1, temperature=0, detokenize=False),
            use_tqdm=False,
        )
        self._engine.collective_rpc("hc_discard", args=(None,))
        return time.time() - t0

    def opd_verify(
        self,
        base_ids: list[int],
        prefix_lens: list[int],
        tails: list[list[int]],
        after=None,
        require_complete: bool = True,
        media=None,
    ) -> dict:
        """Verify engine: score branch prompts ``base_ids[:p] + tail``.

        Returns the target post-norm hidden at every captured tail position:
        ``{"branch": int32 [R], "pos": int64 [R] (absolute), "rows": bf16 [R, H]}``.
        Tail positions served from the prefix cache are absent until repaired.
        ``after`` is unused; callers pass the ``opd_warm`` ObjectRef so Ray runs
        this only once the trunk is in the prefix cache.

        With require_complete, repair cache-skipped rows with one uncached retry
        of only the incomplete branches. Never silently shorten the OPD block.
        """
        if require_complete:
            return self._opd_verify_complete(base_ids, prefix_lens, tails, media=media)
        return self._opd_verify_once(base_ids, prefix_lens, tails, media=media)

    def _opd_verify_complete(self, base_ids, prefix_lens, tails, media=None):
        import time

        if len(prefix_lens) != len(tails) or any(
            p < 1 or p > len(base_ids) or not tail for p, tail in zip(prefix_lens, tails)
        ):
            raise ValueError(
                "Complete OPD verification requires aligned nonempty branches and valid prefixes"
            )

        def collect(result, prefixes, branches):
            row_map = {}
            rows, indices, positions = result["rows"], result["branch"], result["pos"]
            if (
                rows.ndim != 2
                or rows.shape[1] != self._hidden_size
                or indices.ndim != 1
                or positions.ndim != 1
            ):
                raise RuntimeError("Invalid complete OPD teacher tensor shapes")
            if len(rows) != len(indices) or len(rows) != len(positions):
                raise RuntimeError("Misaligned complete OPD teacher rows")
            for i, pos, row in zip(indices.tolist(), positions.tolist(), rows):
                if (
                    i < 0
                    or i >= len(branches)
                    or not prefixes[i] <= pos < prefixes[i] + len(branches[i])
                ):
                    raise RuntimeError(
                        f"Invalid complete OPD teacher position: branch={i}, pos={pos}"
                    )
                if (i, pos) in row_map:
                    raise RuntimeError(
                        f"Duplicate complete OPD teacher position: branch={i}, pos={pos}"
                    )
                row_map[i, pos] = row
            return row_map

        t0 = time.time()
        result = self._opd_verify_once(base_ids, prefix_lens, tails, media=media)
        rows = collect(result, prefix_lens, tails)
        missing = [
            i
            for i, (p, tail) in enumerate(zip(prefix_lens, tails))
            if any((i, pos) not in rows for pos in range(p, p + len(tail)))
        ]
        repair_time = 0.0
        if missing:
            repair_start = time.time()
            prefixes = [prefix_lens[i] for i in missing]
            branches = [tails[i] for i in missing]
            retry = self._opd_verify_once(
                base_ids,
                prefixes,
                branches,
                skip_reading_prefix_cache=True,
                media=media,
            )
            repaired = collect(retry, prefixes, branches)
            rows.update({(missing[i], pos): row for (i, pos), row in repaired.items()})
            repair_time = time.time() - repair_start
        expected = [
            (i, pos)
            for i, (p, tail) in enumerate(zip(prefix_lens, tails))
            for pos in range(p, p + len(tail))
        ]
        absent = [key for key in expected if key not in rows]
        if absent:
            raise RuntimeError(f"Incomplete OPD teacher after uncached retry: {absent[:8]}")
        result.update(
            branch=torch.tensor([i for i, _ in expected], dtype=torch.int32),
            pos=torch.tensor([p for _, p in expected], dtype=torch.int64),
            rows=torch.stack([rows[key] for key in expected]) if expected else result["rows"],
            time=time.time() - t0,
            repair_branches=len(missing),
            repair_time=repair_time,
        )
        return result

    def _opd_verify_once(
        self, base_ids, prefix_lens, tails, *, skip_reading_prefix_cache=False, media=None
    ):
        import time

        import numpy as np
        from vllm import SamplingParams

        t0 = time.time()
        media_cache = {}
        prompts = [
            self._branch_prompt(base_ids, p, t, media, media_cache)
            for p, t in zip(prefix_lens, tails)
        ]
        extra = {"skip_reading_prefix_cache": True} if skip_reading_prefix_cache else {}
        outs = self._engine.generate(
            prompts,
            SamplingParams(max_tokens=1, temperature=0, detokenize=False, **extra),
            use_tqdm=False,
        )
        got = self._engine.collective_rpc("hc_pop", args=([o.request_id for o in outs],))[0]
        H = self._hidden_size
        branch, pos, rows = [], [], []
        for i, (o, p) in enumerate(zip(outs, prefix_lens)):
            pos_b, rows_b, n = got.get(o.request_id, (b"", b"", 0))
            if n == 0:
                continue
            ps = np.frombuffer(pos_b, dtype=np.int64)
            rs = torch.frombuffer(bytearray(rows_b), dtype=torch.bfloat16).view(n, H)
            keep = ps >= p
            if not keep.any():
                continue
            pos.append(ps[keep])
            branch.append(np.full(int(keep.sum()), i, dtype=np.int32))
            rows.append(rs[torch.from_numpy(keep)])
        if rows:
            out = {
                "branch": torch.from_numpy(np.concatenate(branch)),
                "pos": torch.from_numpy(np.concatenate(pos)),
                "rows": torch.cat(rows),
            }
        else:
            out = {
                "branch": torch.zeros(0, dtype=torch.int32),
                "pos": torch.zeros(0, dtype=torch.int64),
                "rows": torch.zeros(0, H, dtype=torch.bfloat16),
            }
        out["time"] = time.time() - t0
        return out
