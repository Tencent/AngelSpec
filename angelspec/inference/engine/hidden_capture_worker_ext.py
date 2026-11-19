"""Capture postnorm target hidden states from the eager vLLM decoder.

Prefill engines retain all prompt rows on TP rank 0 for Mooncake transfer;
verifiers return only the final draft-token rows through Ray. Chunked prefill
and preemption are merged by logical token offset, independent of MRoPE.
Prefix-cache hits produce no rows and are repaired by MTPOPDEngine.
"""

import numpy as np
import torch


def _external_id(internal: str) -> str:
    head, sep, tail = internal.rpartition("-")
    return head if sep and len(tail) == 8 else internal


def _skip_per_request_prompt_scans(mr) -> None:
    """Drop the O(prompt length) Python work vLLM does for every new request.

    Verify branches are short extensions of a long cached prompt, so per-request
    scans of the whole prompt dominate the engine step. These engines never use
    sampling penalties (no prompt-token bitmask needed), and text-only prompts
    have plain ``arange`` M-RoPE positions.
    """
    batch = getattr(mr, "input_batch", None)
    if hasattr(batch, "_init_penalty_token_stats"):
        batch._init_penalty_token_stats = lambda *args, **kwargs: None

    model = mr.get_model()
    orig = getattr(model, "get_mrope_input_positions", None)
    if orig is None:
        return

    def get_mrope_input_positions(input_tokens, mm_features):
        if mm_features:
            return orig(input_tokens, mm_features)
        n = len(input_tokens)
        return torch.arange(n).unsqueeze(0).expand(3, n), 0

    model.get_mrope_input_positions = get_mrope_input_positions


class HiddenCaptureWorkerExt:
    """Mixin providing ``hc_*`` RPCs on the vLLM worker."""

    def opd_teacher_contract(self):
        return self._hc_teacher_config

    def hc_configure(self, last_n: int) -> bool:
        from vllm.distributed import get_tensor_model_parallel_rank

        self._hc_last_n = int(last_n)
        self._hc_rows: dict[str, list] = {}
        self._hc_is_rank0 = get_tensor_model_parallel_rank() == 0
        mr = self.model_runner
        if getattr(mr, "_hc_wrapped", False):
            return True
        from angelspec.inference.engine.opd_vllm import configure_teacher

        self._hc_teacher_config = configure_teacher(mr)
        if hasattr(mr, "_model_forward"):
            orig = mr._model_forward

            def _model_forward(*args, **kwargs):
                out = orig(*args, **kwargs)
                if self._hc_is_rank0 and self._hc_last_n != 0:
                    self._hc_record(out)
                return out

            mr._model_forward = _model_forward
        else:
            # Model runner V2 builds each step's batch in prepare_inputs and then calls
            # the eager model, whose output the hook reads against that batch.
            prepare = mr.prepare_inputs

            def prepare_inputs(*args, **kwargs):
                mr._hc_batch = prepare(*args, **kwargs)
                return mr._hc_batch

            def record(_module, _inputs, out):
                if self._hc_is_rank0 and self._hc_last_n != 0:
                    self._hc_record(out)

            mr.prepare_inputs = prepare_inputs
            mr.get_model().register_forward_hook(record)
        mr._hc_wrapped = True
        _skip_per_request_prompt_scans(mr)
        return True

    def _hc_step(self):
        """``(req_ids, query_start_loc, num_computed_tokens, num_prompt_tokens)`` of the
        step being run."""
        mr = self.model_runner
        batch = getattr(mr, "_hc_batch", None)
        if batch is None:
            req_ids = mr.input_batch.req_ids
            n = len(req_ids)
            return (
                req_ids,
                mr.query_start_loc.cpu[: n + 1].tolist(),
                mr.input_batch.num_computed_tokens_cpu[:n],
                mr.input_batch.num_prompt_tokens[:n],
            )
        n = batch.num_reqs
        return (
            batch.req_ids[:n],
            batch.query_start_loc_np[: n + 1].tolist(),
            batch.num_computed_tokens_np[:n],
            mr.req_states.prompt_len.np[batch.idx_mapping_np[:n]],
        )

    def _hc_record(self, out) -> None:
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(hidden, torch.Tensor):
            return
        req_ids, qsl, computed, n_prompt = self._hc_step()
        if not req_ids:
            return
        n_req = len(req_ids)
        last_n = self._hc_last_n
        # Logical token offsets are independent of multimodal RoPE coordinates.
        pos_cpu = torch.cat(
            [
                torch.arange(int(computed[i]), int(computed[i]) + qsl[i + 1] - qsl[i])
                for i in range(n_req)
            ]
        )
        qsl_t = torch.tensor(qsl)
        counts = qsl_t[1:] - qsl_t[:-1]
        hi = torch.as_tensor(np.asarray(n_prompt[:n_req]), dtype=pos_cpu.dtype).repeat_interleave(
            counts
        )
        lo = torch.zeros_like(hi) if last_n < 0 else hi - last_n
        keep = (pos_cpu >= lo) & (pos_cpu < hi)
        idx = torch.nonzero(keep, as_tuple=False).flatten()
        if idx.numel() == 0:
            return
        rows = hidden.index_select(0, idx.to(hidden.device)).to(torch.bfloat16)
        # Full-trunk captures go straight to Mooncake from the GPU; short
        # verify captures leave via Ray and are staged on the host.
        if last_n > 0:
            rows = rows.cpu()
        owner = torch.searchsorted(qsl_t[1:], idx, right=True)
        per_req = torch.bincount(owner, minlength=n_req).tolist()
        kept_pos = pos_cpu[idx]
        start = 0
        for i, n in enumerate(per_req):
            if n:
                self._hc_rows.setdefault(req_ids[i], []).append(
                    (kept_pos[start : start + n].clone(), rows[start : start + n])
                )
                start += n

    def _hc_take(self, external_ids):
        by_ext = {}
        for rid in list(self._hc_rows):
            by_ext.setdefault(_external_id(rid), []).append(rid)
        out = {}
        for ext in external_ids:
            chunks = []
            for rid in by_ext.get(ext, []):
                chunks.extend(self._hc_rows.pop(rid))
            out[ext] = chunks
        return out

    @staticmethod
    def _hc_merge(chunks):
        """Concatenate chunks; later chunks win on duplicate positions (preemption)."""
        if not chunks:
            return np.zeros((0,), dtype=np.int64), None
        pos = torch.cat([c[0] for c in chunks]).long()
        rows = torch.cat([c[1] for c in chunks])
        order = torch.argsort(pos, stable=True)
        pos, rows = pos[order], rows[order]
        if pos.numel() > 1:
            last = torch.ones_like(pos, dtype=torch.bool)
            last[:-1] = pos[1:] != pos[:-1]
            pos, rows = pos[last], rows[last]
        return pos.numpy(), rows

    def hc_pop(self, external_ids):
        """Return ``{ext: (positions int64 bytes, rows bf16 bytes, n_rows)}`` (rank 0 only)."""
        if not self._hc_is_rank0:
            return None
        result = {}
        for ext, chunks in self._hc_take(external_ids).items():
            pos, rows = self._hc_merge(chunks)
            if rows is None:
                result[ext] = (b"", b"", 0)
                continue
            result[ext] = (
                pos.astype(np.int64).tobytes(),
                rows.cpu().contiguous().view(torch.uint8).numpy().tobytes(),
                int(pos.shape[0]),
            )
        return result

    def hc_discard(self, external_ids=None) -> bool:
        if external_ids is None:
            self._hc_rows = {}
        else:
            self._hc_take(external_ids)
        return True

    def hc_flush_to_mooncake(self, items):
        """Write full-trunk captures to Mooncake.

        ``items``: list of ``(external_id, mooncake_key, input_ids list)``. Every
        position 0..len-1 must have been captured (prefill engines run without
        prefix caching). Returns ``{key: {"shapes", "dtypes"} | None}`` on rank 0;
        ``None`` marks an incomplete capture (the engine raises).
        """
        if not self._hc_is_rank0:
            return None
        store = self._hc_mooncake_store()
        taken = self._hc_take([ext for ext, _, _ in items])
        meta = {}
        for ext, key, input_ids in items:
            pos, rows = self._hc_merge(taken.get(ext, []))
            n = len(input_ids)
            if rows is None or pos.shape[0] != n or int(pos[0]) != 0 or int(pos[-1]) != n - 1:
                meta[key] = None
                continue
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            rows = rows.to(device)
            ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            info = store.put(
                key=key,
                hidden_states=None,
                input_ids=ids,
                last_hidden_states=rows,
                target=None,
            )
            meta[key] = {
                "shapes": {k: list(v) for k, v in info["shapes"].items()},
                "dtypes": {k: str(v).replace("torch.", "") for k, v in info["dtypes"].items()},
            }
        store.flush()
        failed = store.pop_failed_put_keys()
        if failed:
            store.mark_put_failed(failed)
            raise RuntimeError(f"OPD prefill transfer failed for {len(failed)} samples")
        return meta

    def _hc_mooncake_store(self):
        store = getattr(self, "_hc_store", None)
        if store is None:
            from angelspec.config.mooncake_config import MooncakeConfig
            from angelspec.transfer.mooncake.eagle_store import EagleMooncakeStore

            store = EagleMooncakeStore(MooncakeConfig.from_env())
            store.setup(device=torch.device(f"cuda:{torch.cuda.current_device()}"))
            self._hc_store = store
        return store
