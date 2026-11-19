"""Branch-request OPD verification for the DFly/DSpark drafter."""

import os

import torch
from omegaconf import DictConfig, OmegaConf

from angelspec.inference.engine.opd_utils import prefix_media
from angelspec.inference.engine.vllm_engine import (
    _PROTECTION_ENGINE_KEYS,
    VllmEngine,
    one_request_per_step,
)
from angelspec.utils.logging import logger

_HIDDEN_CAPTURE_EXT = "angelspec.inference.engine.hidden_capture_worker_ext.HiddenCaptureWorkerExt"
# Managed by this engine; vllm.extra_args must not override them.
_PROTECTED = _PROTECTION_ENGINE_KEYS | frozenset(
    {
        "model",
        "tensor_parallel_size",
        "distributed_executor_backend",
        "max_num_seqs",
        "max_num_batched_tokens",
        "pipeline_parallel_size",
        "gpu_memory_utilization",
        "worker_extension_cls",
        "enforce_eager",
        "enable_prefix_caching",
        "max_model_len",
        "runner",
        "speculative_config",
        "kv_transfer_config",
        "nnodes",
        "node_rank",
        "master_addr",
        "master_port",
    }
)


class DFlyOPDEngine(VllmEngine):
    """Decoder engine that scores DFlash proposal branches and returns hidden rows."""

    def _init_engine(
        self, tp_size, pp_size, nnodes, mem_fraction, dist_init_addr, pre_allocated_port=None
    ):
        import vllm.envs as envs
        from vllm import LLM

        if not envs.VLLM_BATCH_INVARIANT:
            # The GT-prefix shortcut reads teacher rows from the PREFILL engine and
            # the diverged suffix from this one; both must agree numerically.
            raise ValueError("DFly branch OPD verification requires VLLM_BATCH_INVARIANT=1")
        if pp_size != 1 or nnodes != 1:
            raise ValueError("DFly OPD verify engines require single-node TP, PP=1")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(self.base_gpu_id + i) for i in range(tp_size)
        )

        block_size = int(getattr(self.args, "dflash_block_size", 8))
        if block_size < 3:
            raise ValueError(f"DFly branch OPD needs dflash_block_size >= 3, got {block_size}")
        # A branch prompt is base_ids[:anchor+first_slot] + the tokens of slots
        # first_slot..B-2 (the trainer scores every drafted token). With the GT
        # shortcut the first slot is at least 1 (slot 0's teacher comes from the
        # prefill engine), so the longest tail is B-2 tokens; without it slot 0 is
        # included too and the tail reaches B-1.
        self._capture_last_n = block_size - (
            2 if bool(getattr(self.args, "dflash_opd_gt_shortcut", True)) else 1
        )
        max_model_len = int(getattr(self.args, "max_seq_length", 32769)) + block_size

        engine_kwargs = {
            "model": self.args.target_model_path,
            "tensor_parallel_size": tp_size,
            "trust_remote_code": getattr(self.args, "trust_remote_code", True),
            "distributed_executor_backend": "mp",
            "disable_custom_all_reduce": True,
            # The capture wraps the eager model forward (no CUDA graphs).
            "enforce_eager": True,
            "worker_extension_cls": _HIDDEN_CAPTURE_EXT,
            # Branch prompts share the trunk; the whole design rests on this.
            "enable_prefix_caching": True,
            # Every branch re-hashes its whole (up to max_model_len) prompt.
            "prefix_caching_hash_algo": "xxhash",
            "max_model_len": max_model_len,
            "gpu_memory_utilization": mem_fraction if mem_fraction is not None else 0.85,
            "distributed_timeout_seconds": (
                getattr(self.args, "distributed_timeout_minutes", 10) * 60
            ),
            "max_num_seqs": int(getattr(self.args, "dflash_opd_verify_max_num_seqs", 1024)),
            "max_num_batched_tokens": int(
                getattr(self.args, "dflash_opd_verify_max_num_batched_tokens", 65536)
            ),
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
        extra_args = getattr(self.args, "vllm_extra_args", None)
        if extra_args:
            extra = (
                OmegaConf.to_container(extra_args, resolve=True)
                if isinstance(extra_args, DictConfig)
                else dict(extra_args)
            )
            engine_kwargs.update({k: v for k, v in extra.items() if k not in _PROTECTED})
        if one_request_per_step(self.args.target_model_path):
            engine_kwargs["max_num_seqs"] = 1
        if dist_init_addr:
            host, port_str = dist_init_addr.rsplit(":", 1)
            engine_kwargs["master_addr"] = host
            engine_kwargs["master_port"] = int(port_str)
        elif pre_allocated_port is not None:
            engine_kwargs["master_port"] = pre_allocated_port

        self._engine = LLM(**engine_kwargs)
        self._engine.collective_rpc("hc_configure", args=(self._capture_last_n,))
        contracts = self._engine.collective_rpc("opd_teacher_contract")
        if any(c != contracts[0] for c in contracts):
            raise RuntimeError(f"OPD verify TP teacher contracts differ: {contracts}")
        self._opd_teacher_contract = contracts[0]
        self._hidden_size = self._get_hidden_size_from_engine()
        logger.info(
            "DFlyOPDEngine rank %s: verify engine ready (capture_last_n=%s, "
            "draft_tokens=%s, block_size=%s, max_model_len=%s, tp=%s, hidden_size=%s)",
            self.rank,
            self._capture_last_n,
            block_size - 1,
            block_size,
            max_model_len,
            tp_size,
            self._hidden_size,
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

    def opd_warm(self, base_ids: list[int], media=None) -> float:
        """Prefill the trunk once so branch requests hit the prefix cache."""
        import time

        from vllm import SamplingParams

        t0 = time.time()
        self._engine.collective_rpc("hc_configure", args=(0,))
        self._engine.generate(
            [self._branch_prompt(base_ids, len(base_ids), [], media)],
            SamplingParams(max_tokens=1, temperature=0, detokenize=False),
            use_tqdm=False,
        )
        self._engine.collective_rpc("hc_configure", args=(self._capture_last_n,))
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
        """Score branch prompts ``base_ids[:p] + tail``."""
        if require_complete:
            return self._opd_verify_complete(base_ids, prefix_lens, tails, media=media)
        return self._opd_verify_once(base_ids, prefix_lens, tails, media=media)

    def _opd_verify_complete(self, base_ids, prefix_lens, tails, media=None):
        import time

        if len(prefix_lens) != len(tails) or any(
            p < 0 or p > len(base_ids) or not tail for p, tail in zip(prefix_lens, tails)
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
                    or not (prefixes[i] <= pos < prefixes[i] + len(branches[i]))
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
        if missing:
            raise RuntimeError(
                f"OPD capture-bound invariant failed: missing teacher rows in "
                f"{len(missing)} branches (first {missing[:8]}). Refusing uncached repair."
            )
        expected = [
            (i, pos)
            for i, (p, tail) in enumerate(zip(prefix_lens, tails))
            for pos in range(p, p + len(tail))
        ]
        result.update(
            branch=torch.tensor([i for i, _ in expected], dtype=torch.int32),
            pos=torch.tensor([p for _, p in expected], dtype=torch.int64),
            rows=torch.stack([rows[key] for key in expected]) if expected else result["rows"],
            time=time.time() - t0,
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
        from angelspec.inference.engine.opd_cache import CAPTURE_START

        prepare_time = time.time() - t0
        params = [
            SamplingParams(
                max_tokens=1,
                temperature=0,
                detokenize=False,
                skip_reading_prefix_cache=skip_reading_prefix_cache,
                extra_args={CAPTURE_START: int(p)},
            )
            for p in prefix_lens
        ]
        tg = time.time()
        outs = self._engine.generate(
            prompts,
            params,
            use_tqdm=False,
        )
        generate_time = time.time() - tg
        got = self._engine.collective_rpc("hc_pop", args=([o.request_id for o in outs],))[0]
        hidden_size = self._hidden_size
        branch, pos, rows = [], [], []
        for i, (o, p) in enumerate(zip(outs, prefix_lens)):
            pos_b, rows_b, n = got.get(o.request_id, (b"", b"", 0))
            if n == 0:
                continue
            ps = np.frombuffer(pos_b, dtype=np.int64)
            rs = torch.frombuffer(bytearray(rows_b), dtype=torch.bfloat16).view(n, hidden_size)
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
                "rows": torch.zeros(0, hidden_size, dtype=torch.bfloat16),
            }
        out["time"] = time.time() - t0
        out["prepare_time"] = prepare_time
        out["generate_time"] = generate_time
        out["capture_time"] = out["time"] - prepare_time - generate_time
        out["cached_tokens"] = sum(int(getattr(o, "num_cached_tokens", 0) or 0) for o in outs)
        out["computed_tokens"] = sum(len(o.prompt_token_ids) for o in outs) - out["cached_tokens"]
        return out
