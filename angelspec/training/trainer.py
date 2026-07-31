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

import abc
import concurrent.futures
import dataclasses
import itertools
import logging
import os
import time
from argparse import Namespace
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh

from angelspec.config.mooncake_config import MooncakeConfig
from angelspec.data.utils import (
    DataCollatorWithPadding,
    DFlashPackingCollator,
    MTPPackingCollator,
)
from angelspec.training import checkpoint
from angelspec.training.data_fetcher import MooncakeDataFetcher, PrefetchedDataFetcher
from angelspec.training.fsdp import init_empty_weights
from angelspec.training.optimizer import BF16Optimizer
from angelspec.transfer.mooncake.eagle_store import EagleMooncakeStore
from angelspec.utils.distributed import get_usp_device_mesh, get_usp_grad_sync_mesh
from angelspec.utils.logging import logger
from angelspec.utils.metrics import token_weighted_loss_scale
from angelspec.utils.processing import get_assistant_token_ids
from angelspec.utils.profiling import TrainProfiler
from angelspec.utils.train_dump import extract_gradients, extract_model_weights
from angelspec.utils.usp import usp_dp_average_factor


def _with_lookahead_islast(iterable, known_count: Optional[int] = None):
    """Yield ``(item, is_last)`` pairs, marking the final item without a count upfront.

    Holds one item ahead so the packed equal-weight path streams rows lazily (peak
    1-2 rows) instead of materializing the step just to find the last micro-batch.
    ``known_count`` is a fast path (non-packing case) that skips the buffering.
    """
    if known_count is not None:
        for i, item in enumerate(iterable):
            yield item, (i == known_count - 1)
        return

    it = iter(iterable)
    try:
        prev = next(it)
    except StopIteration:
        return
    for cur in it:
        yield prev, False
        prev = cur
    yield prev, True


class Trainer(abc.ABC):
    """Base trainer for async training pipeline.

    Provides shared infrastructure: device mesh, data fetching, training loop
    skeleton, checkpointing, and profiling. Subclasses implement model-specific
    logic via ``init_model``, ``_train_step``, and ``_aggregate_metrics``.
    """

    def __init__(self, args: Namespace):
        self.args = args

        self._setup_device_mesh()
        torch.manual_seed(getattr(args, "seed", 42))

        self.fsdp_cpu_offload = getattr(args, "fsdp_cpu_offload", False)

        self.global_step = 0
        self.model = None
        self.draft_model = None
        self.optimizer: Optional[BF16Optimizer] = None
        self.lr_scheduler = None
        self.data_fetcher: Optional[MooncakeDataFetcher] = None
        self.train_queue = None
        self.mooncake_store: Optional[EagleMooncakeStore] = None
        self._eval_cache: list[dict] = []
        # List of Ray handles to the encoder-only score engine(s) (half on-policy
        # OPD), round-robined by the DFlash step. Set by the controller via
        # set_score_engine; None when OPD is off.
        self.score_engine = None

        self.prof = TrainProfiler(args)

        self.dynamic_loss_mask = getattr(args, "dynamic_loss_mask", False)
        self.last_turn_loss_only = getattr(args, "last_turn_loss_only", False)
        self.assistant_header_ids, self.end_token_ids, self.skip_after_header = (
            get_assistant_token_ids(self.args)
        )

        self.save_debug_train_data = getattr(args, "save_debug_train_data", None)
        self.max_dump_steps = getattr(args, "max_dump_steps", 5)

        self._enable_perf_metrics = getattr(args, "enable_perf_metrics", True)

        self._io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._eval_cache_save_future: Optional[concurrent.futures.Future] = None

    # ------------------------------------------------------------------
    # Device mesh
    # ------------------------------------------------------------------

    def _setup_device_mesh(self) -> None:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        self.cache_rank = rank

        usp_mesh = None
        if getattr(self.args, "attention_backend", None) == "usp":
            usp_mesh = get_usp_device_mesh()

        if usp_mesh is not None:
            self.mesh = usp_mesh
            self.dp_size = getattr(self.args, "dp_size", world_size)
            self.dp_mesh = usp_mesh["draft_dp"]
            self.grad_sync_mesh = get_usp_grad_sync_mesh()
            if self.grad_sync_mesh is None:
                raise RuntimeError("USP grad sync mesh has not been initialized")
            self.dp_group = usp_mesh.get_group("draft_dp")
            self.dp_rank = dist.get_rank(self.dp_group)
            logger.info(
                f"[Rank {rank}] Device mesh (USP): world_size={world_size}, dp_size={self.dp_size}, "
                f"dp_rank={self.dp_rank}, grad_sync_size={world_size}"
            )
            return

        self.dp_size = world_size
        self.dp_rank = rank

        self.mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size,), mesh_dim_names=("dp",))
        self.dp_group = self.mesh.get_group("dp")
        self.dp_mesh = self.mesh
        self.grad_sync_mesh = self.dp_mesh

        logger.info(
            f"[Rank {rank}] Device mesh (1D): world_size={world_size}, dp_size={self.dp_size}"
        )

    def _get_init_weight_context_manager(self):
        """Meta-device context for non-rank-0 processes to save memory."""

        def cpu_init_weights():
            return torch.device("cpu")

        if dist.get_rank() != 0:
            return init_empty_weights
        return cpu_init_weights

    # ------------------------------------------------------------------
    # Mooncake store
    # ------------------------------------------------------------------

    def init_mooncake_store(
        self,
        mooncake_config: Optional[MooncakeConfig] = None,
    ) -> EagleMooncakeStore:
        if mooncake_config is None:
            mooncake_config = MooncakeConfig.from_flat_args(self.args)

        mooncake_config = dataclasses.replace(
            mooncake_config,
            global_segment_size=0,
            async_put_pool_size=0,
        )

        store = EagleMooncakeStore(mooncake_config)
        store.setup(device=torch.cuda.current_device())
        self.mooncake_store = store
        logger.info(f"[Rank {self.dp_rank}] EagleMooncakeStore initialized")
        return store

    def set_score_engine(self, score_engine) -> None:
        """Inject the encoder-only score engine handle(s) (half on-policy OPD).

        ``score_engine`` is a list of Ray handles (one TP=1 engine per score GPU).
        The DFlash trainer round-robins ``score_packed.remote(...)`` across them
        mid-step to have the target score the draft's packed proposal tree.
        """
        self.score_engine = score_engine
        logger.info(
            "Score engine handle set on trainer (%d engine(s)).",
            len(score_engine) if score_engine else 0,
        )

    # ------------------------------------------------------------------
    # Data queue
    # ------------------------------------------------------------------

    def set_train_queue(
        self,
        queue,
        mooncake_config: Optional[MooncakeConfig] = None,
        per_dp_rank_batch_size: int = 1,
    ) -> None:
        self.train_queue = queue
        self.per_dp_rank_batch_size = per_dp_rank_batch_size
        usp_enabled = getattr(self.args, "attention_backend", None) == "usp"
        if usp_enabled and per_dp_rank_batch_size != 1:
            raise ValueError("USP requires per_dp_rank_batch_size=1")
        # Data-path USP pre-shards into {key}_usp{rank}; local-shard USP fans the whole
        # sample to all SP ranks and slices in-forward (only the former is special-cased).
        usp_sharded_data = usp_enabled and not getattr(self.args, "usp_local_shard", False)
        if mooncake_config is not None and self.mooncake_store is None:
            self.init_mooncake_store(mooncake_config)

        dflash_packing = getattr(self.args, "dflash_packing", False)
        mtp_packing = getattr(self.args, "mtp_packing", False)
        if dflash_packing and mtp_packing:
            raise ValueError(
                "dflash_packing and mtp_packing are mutually exclusive (different collators)."
            )
        if dflash_packing or mtp_packing:
            if usp_sharded_data:
                raise NotImplementedError(
                    "sequence packing is incompatible with USP sharded-data pre-shard; "
                    "use usp_local_shard=true so each rank slices the packed row in-forward."
                )
            packing_enabled = True
            max_seq_length = getattr(self.args, "max_seq_length", None)
            collator = (
                MTPPackingCollator(max_seq_length=max_seq_length)
                if mtp_packing
                else DFlashPackingCollator(max_seq_length=max_seq_length)
            )
        else:
            packing_enabled = False
            collator = DataCollatorWithPadding(usp_enabled=usp_sharded_data)

        prefetch_depth = getattr(self.args, "prefetch_depth", 0)
        gpu_device = torch.cuda.current_device()

        # When prefetching, stage data on CPU to avoid GPU contention between
        # background Mooncake TCP transfers and forward/backward compute.
        fetch_device = "cpu" if prefetch_depth > 0 else gpu_device

        inner_fetcher = MooncakeDataFetcher(
            queue=self.train_queue,
            mooncake_store=self.mooncake_store,
            collator=collator,
            device=fetch_device,
            batch_size=per_dp_rank_batch_size,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            dynamic_loss_mask=self.dynamic_loss_mask,
            last_turn_loss_only=self.last_turn_loss_only,
            skip_after_header=self.skip_after_header,
            min_loss_tokens=getattr(self.args, "min_loss_tokens", 0),
            usp_enabled=usp_sharded_data,
            ttt_length=getattr(self.args, "ttt_length", 1),
            max_seq_length=getattr(self.args, "max_seq_length", None),
            usp_local_shard=getattr(self.args, "usp_local_shard", False),
            packing_enabled=packing_enabled,
        )

        if prefetch_depth > 0:
            if packing_enabled:
                # Packing: prefetch blocks per-step rows (prefetch_depth counts
                # STEPS, not batches). rows_per_step == draft_accumulation_steps
                # (the loop dispatches that many rows per step; see
                # controller/loop.py). At most prefetch_depth * rows_per_step rows
                # stay resident on CPU.
                rows_per_step = getattr(self.args, "draft_accumulation_steps", 1)
                self.data_fetcher = PrefetchedDataFetcher(
                    inner_fetcher,
                    prefetch_depth=prefetch_depth,
                    target_device=gpu_device,
                    packing_enabled=True,
                    rows_per_step=rows_per_step,
                )
                hidden_dim = getattr(self.args, "hidden_size", 0) or 0
                n_layers = len(getattr(self.args, "aux_hidden_states_layers", []) or []) or 1
                max_seq = getattr(self.args, "max_seq_length", 0) or 0
                approx_bytes = prefetch_depth * rows_per_step * max_seq * hidden_dim * n_layers * 2
                logger.info(
                    f"[Rank {self.dp_rank}] Prefetched packing data fetcher initialized "
                    f"(prefetch_depth(steps)={prefetch_depth}, rows_per_step={rows_per_step}, "
                    f"max_resident≈{prefetch_depth * rows_per_step} rows, "
                    f"hidden≈{approx_bytes / 1e9:.2f}GB CPU staging, target={gpu_device})"
                )
            else:
                self.data_fetcher = PrefetchedDataFetcher(
                    inner_fetcher,
                    prefetch_depth=prefetch_depth,
                    target_device=gpu_device,
                )
                logger.info(
                    f"[Rank {self.dp_rank}] Prefetched data fetcher initialized "
                    f"(batch_size={per_dp_rank_batch_size}, prefetch_depth={prefetch_depth}, "
                    f"staging=CPU, target={gpu_device})"
                )
        else:
            self.data_fetcher = inner_fetcher
            logger.info(
                f"[Rank {self.dp_rank}] Data fetcher initialized with batch_size={per_dp_rank_batch_size}"
            )

    # ------------------------------------------------------------------
    # Eval queue & CPU cache
    # ------------------------------------------------------------------

    def set_eval_queue(
        self,
        queue,
        mooncake_config: Optional[MooncakeConfig] = None,
        per_dp_rank_batch_size: int = 1,
    ) -> None:
        usp_enabled = getattr(self.args, "attention_backend", None) == "usp"
        usp_sharded_data = usp_enabled and not getattr(self.args, "usp_local_shard", False)
        if mooncake_config is not None and self.mooncake_store is None:
            self.init_mooncake_store(mooncake_config)

        collator = DataCollatorWithPadding(usp_enabled=usp_sharded_data)

        self._eval_data_fetcher = MooncakeDataFetcher(
            queue=queue,
            mooncake_store=self.mooncake_store,
            collator=collator,
            device=torch.cuda.current_device(),
            batch_size=per_dp_rank_batch_size,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            dynamic_loss_mask=self.dynamic_loss_mask,
            last_turn_loss_only=self.last_turn_loss_only,
            skip_after_header=self.skip_after_header,
            min_loss_tokens=getattr(self.args, "min_loss_tokens", 0),
            usp_enabled=usp_sharded_data,
            ttt_length=getattr(self.args, "ttt_length", 1),
            max_seq_length=getattr(self.args, "max_seq_length", None),
            usp_local_shard=getattr(self.args, "usp_local_shard", False),
        )
        self._eval_collator = collator
        self._eval_cache: list[dict] = []
        logger.info(
            f"[Rank {self.dp_rank}] Eval data fetcher initialized with batch_size={per_dp_rank_batch_size}"
        )

    def cache_eval_samples(self, count: int) -> int:
        for sample in itertools.islice(self._eval_data_fetcher, count):
            cpu_sample = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sample.items()
            }
            self._eval_cache.append(cpu_sample)
        return len(self._eval_cache)

    def save_eval_cache(self, cache_dir: str) -> None:
        if not getattr(self, "_eval_cache", None):
            return
        self._wait_for_eval_cache_save()

        cache_snapshot = list(self._eval_cache)
        rank = self.cache_rank

        def _save() -> None:
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, f"eval_rank_{rank}.pt")
            tmp_path = path + ".tmp"
            torch.save(cache_snapshot, tmp_path)
            os.replace(tmp_path, path)
            logger.info(f"[Rank {rank}] Saved {len(cache_snapshot)} eval batches to {path}")

        self._eval_cache_save_future = self._io_executor.submit(_save)

    def _wait_for_eval_cache_save(self) -> None:
        fut = self._eval_cache_save_future
        if fut is not None:
            fut.result()
            self._eval_cache_save_future = None

    def load_eval_cache(self, cache_dir: str) -> int:
        # Safe guard to wait for eval cache save to complete.
        self._wait_for_eval_cache_save()
        path = os.path.join(cache_dir, f"eval_rank_{self.cache_rank}.pt")
        if not os.path.exists(path):
            return 0
        try:
            self._eval_cache = torch.load(path, weights_only=False, mmap=True)
        except Exception as e:
            logger.warning(f"[Rank {self.dp_rank}] Corrupt eval cache at {path}, ignoring: {e}")
            return 0
        logger.info(
            f"[Rank {self.dp_rank}] Loaded {len(self._eval_cache)} eval batches from {path}"
        )
        return len(self._eval_cache)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_from_queue(self, step: int, num_batches: int) -> dict:
        if self.data_fetcher is None:
            raise RuntimeError("Data fetcher not initialized. Call set_train_queue first.")
        perf = self._enable_perf_metrics
        if perf:
            t0 = time.time()
        metrics = self._train_core_from_queue(step=step, num_batches=num_batches)
        if perf:
            # _aggregate_metrics already synced via .item() — wall-clock is accurate
            step_time = time.time() - t0
            metrics["perf/step_time"] = step_time
            # Effective throughput: real (non-pad) tokens forwarded per second.
            if step_time > 0 and "data/seq_tokens" in metrics:
                metrics["perf/effective_tokens_per_sec"] = metrics["data/seq_tokens"] / step_time
        self.prof.step(step=step)
        return metrics

    def _train_core_from_queue(self, step: int, num_batches: int) -> dict:
        """Training loop skeleton.

        Calls ``_train_step`` for each micro-batch, wrapped with debug logging
        and optional dump extraction.  One optimizer step is performed after
        all micro-batches.  ``global_step`` counts optimizer steps.
        """
        self.model.train()
        accumulation_steps = num_batches

        all_step_metrics: list[dict] = []
        grad_norm = None

        perf = self._enable_perf_metrics
        if perf:
            data_time = 0.0
            compute_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            t_data_start = time.time()

        # Gradient sync control for micro-batch accumulation.
        # FSDP2 fully_shard: use set_requires_gradient_sync(bool)
        # replicate (DDP): use no_sync() context manager
        _model = getattr(self.model, "_orig_mod", self.model)
        _set_grad_sync = getattr(_model, "set_requires_gradient_sync", None)
        _no_sync = getattr(_model, "no_sync", None) if _set_grad_sync is None else None

        # USP: draft attention fires Ulysses all-to-all during backward. If DDP's
        # in-backward grad all-reduce runs in the same backward, the two collective
        # streams race across ranks → deadlock. So suppress in-backward grad sync and
        # all-reduce grads manually once after backward.
        # NOTE: composable `replicate` (torch 2.11) has set_requires_gradient_sync but
        # NOT no_sync(), so no_sync() leaves the hook live — must disable it explicitly.
        usp_manual_grad = getattr(self.args, "attention_backend", None) == "usp"
        if usp_manual_grad and _set_grad_sync is not None:
            _set_grad_sync(False)

        dflash_packing = bool(getattr(self.args, "dflash_packing", False))
        mtp_packing = bool(getattr(self.args, "mtp_packing", False))
        packing = dflash_packing or mtp_packing
        # This weighting option is a DFlash recipe knob. MTP packing shares the
        # fixed-row data protocol but keeps its existing equal-row loss semantics.
        token_weighted = dflash_packing and bool(
            getattr(self.args, "dflash_packing_token_weighted_loss", False)
        )
        if packing:
            # Fixed-rows packing: controller emits exactly ONE row per rank per
            # dispatch, num_batches dispatches → equal micro-batch counts across DP
            # ranks (DP-safe under FSDP resharding). Rows assembled by row_id.
            num_micro = num_batches
            if token_weighted:
                # Token-weighted: needs the step's global supervised-token total, so
                # materialize all rows first; each weighted by row_tokens/step_tokens.
                step_batches = list(self.data_fetcher.iter_step_batches(num_micro))
                device = torch.device("cuda", torch.cuda.current_device())
                ready = torch.tensor(
                    int(len(step_batches) == num_micro),
                    dtype=torch.int32,
                    device=device,
                )
                dist.all_reduce(ready, op=dist.ReduceOp.SUM, group=self.dp_group)
                dp_world_size = dist.get_world_size(self.dp_group)
                if int(ready.item()) != dp_world_size:
                    raise RuntimeError(
                        f"step {step}: token-weighted packing rows are inconsistent "
                        f"across DP ranks (this rank has {len(step_batches)}/{num_micro}; "
                        f"{int(ready.item())}/{dp_world_size} ranks are ready). "
                        "Aborting instead of entering asymmetric collectives or "
                        "discarding differently sized queue prefixes."
                    )

                local_step_tokens = torch.stack(
                    [
                        b["loss_mask"].sum().to(device=device, dtype=torch.float32)
                        for b in step_batches
                    ]
                ).sum()
                global_step_tokens = local_step_tokens.clone()
                dist.all_reduce(
                    global_step_tokens,
                    op=dist.ReduceOp.SUM,
                    group=self.dp_group,
                )
                denom = global_step_tokens.clamp_min(1.0)
                for b in step_batches:
                    # DDP/FSDP averages gradients across DP ranks. Multiplying by
                    # dp_world_size makes that AVG reconstruct the global token mean.
                    b["_loss_scale"] = token_weighted_loss_scale(
                        b["loss_mask"].sum().to(device=device, dtype=torch.float32),
                        denom,
                        dp_world_size,
                    )
                batch_iter = _with_lookahead_islast(iter(step_batches), known_count=num_micro)
            else:
                # Equal-weight: each row's loss divided by fixed num_batches in
                # _backward; rows stream lazily, last marked without look-ahead.
                batch_iter = _with_lookahead_islast(
                    self.data_fetcher.iter_step_batches(num_micro), known_count=num_micro
                )
            batches = self.prof.iterate_train_actor(batch_iter)
        else:
            num_micro = num_batches
            batches = self.prof.iterate_train_actor(
                _with_lookahead_islast(
                    self._iter_batches_from_queue(num_batches), known_count=num_batches
                )
            )

        # Token accounting for throughput (covers packed + pad-to-longest):
        # tok_seq = non-pad tokens forwarded, tok_sup = supervised, tok_slots = padded
        # slots; fill_ratio = tok_seq/tok_slots = fraction of forwarded compute non-pad.
        tok_seq = 0
        tok_sup = 0
        tok_slots = 0
        rows_seen = 0

        for batch_idx, (batch, is_last) in enumerate(batches):
            rows_seen += 1

            if "attention_mask" in batch:
                tok_seq += int(batch["attention_mask"].sum().item())
                tok_slots += int(batch["attention_mask"].numel())
            if "loss_mask" in batch:
                tok_sup += int(batch["loss_mask"].sum().item())

            if perf:
                data_time += time.time() - t_data_start
                evt_start = torch.cuda.Event(enable_timing=True)
                evt_end = torch.cuda.Event(enable_timing=True)
                evt_start.record()

            if logger.isEnabledFor(logging.DEBUG):
                self._log_batch_debug(batch, step, batch_idx, num_batches)

            if usp_manual_grad:
                # In-backward grad sync already disabled above (the comm hook must
                # not race the Ulysses all-to-all); grads reduced manually post-backward.
                ctx = nullcontext()
            elif _set_grad_sync is not None:
                _set_grad_sync(is_last)
                ctx = nullcontext()
            else:
                ctx = _no_sync() if (_no_sync is not None and not is_last) else nullcontext()

            with ctx:
                step_metrics = self._train_step(
                    batch=batch,
                    accumulation_steps=accumulation_steps,
                    step=step,
                    batch_idx=batch_idx,
                    num_batches=num_batches,
                )

            if is_last:
                self._maybe_dump(batch, step_metrics, step, batch_idx)
                _usp_dbg = os.environ.get("ANGELSPEC_USP_COLLECTIVE_DEBUG") == "1"
                if usp_manual_grad:
                    # All all-to-all drained; reduce grads across the SP group once.
                    # SUM over grad_sync_mesh: each rank holds its seq shard's grad, so
                    # summing reconstructs the full-sequence gradient.
                    if _usp_dbg:
                        logger.info(
                            f"[USP-COLL rank{dist.get_rank()}] step={step} manual_grad_allreduce BEGIN"
                        )
                    self._usp_manual_grad_allreduce()
                    if _usp_dbg:
                        logger.info(
                            f"[USP-COLL rank{dist.get_rank()}] step={step} manual_grad_allreduce END"
                        )
                _evt_opt_s = torch.cuda.Event(enable_timing=True)
                _evt_opt_e = torch.cuda.Event(enable_timing=True)
                _evt_opt_s.record()
                if _usp_dbg:
                    logger.info(
                        f"[USP-COLL rank{dist.get_rank()}] step={step} optimizer.step BEGIN"
                    )
                grad_norm = self.optimizer.step()
                if _usp_dbg:
                    logger.info(
                        f"[USP-COLL rank{dist.get_rank()}] step={step} optimizer.step END grad_norm={float(grad_norm):.4f}"
                    )
                _evt_opt_e.record()
                step_metrics["_opt_events"] = (_evt_opt_s, _evt_opt_e)

            if perf:
                evt_end.record()
                compute_events.append((evt_start, evt_end))

            all_step_metrics.append(step_metrics)

            if perf:
                t_data_start = time.time()

        if rows_seen == 0:
            # Queue starved (or all keys were dropped); optimizer never stepped, so
            # skip cleanly rather than increment global_step and aggregate no metrics.
            logger.warning(f"step {step}: data queue produced 0 batches; skipping step.")
            return {}

        self.global_step += 1

        if os.environ.get("ANGELSPEC_USP_COLLECTIVE_DEBUG") == "1":
            logger.info(
                f"[USP-COLL rank{dist.get_rank()}] step={step} aggregate_metrics BEGIN (n_metrics={len(all_step_metrics)})"
            )
        metrics = self._aggregate_metrics(all_step_metrics, step, grad_norm=grad_norm)
        if os.environ.get("ANGELSPEC_USP_COLLECTIVE_DEBUG") == "1":
            logger.info(f"[USP-COLL rank{dist.get_rank()}] step={step} aggregate_metrics END")

        # Per-rank throughput metrics; fill_ratio = non-pad fraction of forwarded slots.
        metrics["data/seq_tokens"] = tok_seq
        metrics["data/supervised_tokens"] = tok_sup
        metrics["data/fill_ratio"] = (tok_seq / tok_slots) if tok_slots else 0.0
        if packing:
            logger.info(
                f"step {step}: packing {rows_seen} rows, seq_tokens={tok_seq} "
                f"fill={(tok_seq / tok_slots) if tok_slots else 0.0:.1%} "
                f"(slots {tok_slots}), supervised_tokens={tok_sup}"
            )
        # _aggregate_metrics calls .item() which syncs CUDA —
        # all recorded events are now completed, safe to query without extra sync
        if perf:
            compute_time_ms = sum(s.elapsed_time(e) for s, e in compute_events)
            metrics["perf/data_time"] = data_time
            metrics["perf/compute_time"] = compute_time_ms / 1000.0
            # Optimizer timing (only recorded in last micro-batch)
            opt_ms = 0.0
            for m in all_step_metrics:
                if "_opt_events" in m:
                    opt_ms += m["_opt_events"][0].elapsed_time(m["_opt_events"][1])
            metrics["perf/optimizer_time"] = opt_ms / 1000.0

        return metrics

    def _iter_batches_from_queue(self, num_batches: int):
        # Guard the data-path contract: the fetcher must yield dict batches, not a
        # DataLoader / (dataloader, dataset, collator) tuple — a routing regression
        # otherwise surfaces as an opaque 'DataLoader' not subscriptable in _forward.
        for batch in itertools.islice(self.data_fetcher, num_batches):
            if not isinstance(batch, dict):
                raise TypeError(
                    f"data fetcher yielded {type(batch).__name__}, expected a dict "
                    "batch — likely a data-path routing regression (e.g. an "
                    "un-unpacked create_mooncake_dataloader tuple)."
                )
            yield batch

    def _usp_manual_grad_allreduce(self) -> None:
        """Reduce USP grads with SP-SUM and DP-AVG semantics.

        Used only on the USP path, where DDP's in-backward grad reduction is
        suppressed (no_sync) to avoid racing the Ulysses all-to-all collectives.
        Called once per optimizer step, after backward has fully drained, so this
        is the only gradient collective and it runs in identical order on every
        rank. Sequence shards are summed to reconstruct each DP replica's full
        sequence gradient; independent DP replicas are then averaged.

        Reduces over the FULL trainable param set in a stable order, materializing
        a zero grad where a rank produced none (e.g. an MoE expert unrouted on that
        rank's shard). Every rank must reduce the identical tensor set or the
        collective desyncs, so a per-rank `p.grad is None` cannot be skipped.
        """
        mesh = getattr(self, "grad_sync_mesh", None)
        group = mesh.get_group() if mesh is not None else None
        model = getattr(self.model, "_orig_mod", self.model)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return
        grads = []
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        group_world_size = dist.get_world_size(group)
        sp_size = int(getattr(self.args, "sp_ulysses_size", 1)) * int(
            getattr(self.args, "sp_ring_size", 1)
        )
        flat.div_(usp_dp_average_factor(group_world_size, sp_size))
        for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)

    # ------------------------------------------------------------------
    # Checkpointing & persistence
    # ------------------------------------------------------------------

    def save_model(self, step: int, force_sync: bool = False) -> None:
        if not self.args.checkpoint_dir:
            return
        checkpoint.save(self, step=step)

    def save_draft_model_for_serving(self, output_dir: str) -> None:
        """Save draft model in HuggingFace format for serving update."""
        os.makedirs(output_dir, exist_ok=True)

        model = self.draft_model
        if hasattr(model, "module"):
            model = model.module

        try:
            state_dict = get_model_state_dict(
                self.draft_model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )

            if self.dp_rank == 0:
                if hasattr(model, "save_pretrained"):
                    if hasattr(model, "config"):
                        model.config.save_pretrained(output_dir)
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(f"[Rank {self.dp_rank}] Saved draft model to {output_dir}")
                else:
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model state dict to {output_dir}"
                    )

        except Exception as e:
            logger.warning(
                f"[Rank {self.dp_rank}] Failed to save with FSDP2 state dict, trying fallback: {e}"
            )
            if self.dp_rank == 0:
                if hasattr(model, "save_pretrained"):
                    model.save_pretrained(output_dir)
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model using save_pretrained to {output_dir}"
                    )
                else:
                    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    logger.info(
                        f"[Rank {self.dp_rank}] Saved draft model state dict to {output_dir}"
                    )

        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint(self) -> dict | None:
        return checkpoint.load(self)

    def close(self) -> None:
        self._wait_for_eval_cache_save()
        if self.data_fetcher is not None and hasattr(self.data_fetcher, "close"):
            self.data_fetcher.close()
        self._io_executor.shutdown(wait=True)
        if self.mooncake_store is not None and hasattr(self.mooncake_store, "close"):
            self.mooncake_store.close()
            logger.info(f"[Rank {self.dp_rank}] EagleMooncakeStore closed")

    # ------------------------------------------------------------------
    # Debug logging & dump helpers
    # ------------------------------------------------------------------

    def _log_batch_debug(self, batch: dict, step: int, batch_idx: int, num_batches: int) -> None:
        batch_size = batch["input_ids"].shape[0]
        seq_len = batch["input_ids"].shape[1]
        hs_shape = (
            tuple(batch["hidden_states"].shape) if batch.get("hidden_states") is not None else None
        )
        logger.debug(
            f"step={step} batch={batch_idx}/{num_batches} | "
            f"batch_size={batch_size}, seq_len={seq_len}, "
            f"input_ids={tuple(batch['input_ids'].shape)}, hidden_states={hs_shape}"
        )

    def _should_dump_step(self) -> bool:
        return bool(self.save_debug_train_data and (self.global_step + 1) <= self.max_dump_steps)

    def _maybe_dump(self, batch: dict, step_metrics: dict, step: int, batch_idx: int) -> None:
        if not self._should_dump_step():
            return

        self._save_dump_data(
            batch=batch,
            step_metrics=step_metrics,
            gradients=extract_gradients(self.model),
            model_weights=extract_model_weights(self.model),
            step=step,
            batch_idx=batch_idx,
        )

    def _save_dump_data(  # noqa: B027 - optional hook with a default no-op body
        self,
        *,
        batch: dict,
        step_metrics: dict,
        gradients: dict,
        model_weights: dict,
        step: int,
        batch_idx: int,
    ) -> None:
        """Save debug dump data. Override in subclass for model-specific dumps."""

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def init_model(self, *args, **kwargs) -> int:
        """Initialize model, optimizer, and load checkpoint.

        Returns:
            Start step.
        """
        ...

    @abc.abstractmethod
    def _train_step(
        self,
        batch: dict,
        accumulation_steps: int,
        step: int,
        batch_idx: int,
        num_batches: int,
    ) -> dict:
        """Run forward + backward for a single micro-batch.

        The optimizer step is handled by the base class after the last
        micro-batch — do NOT call ``self.optimizer.step()`` here.

        Returns:
            Dict of step-level metrics for later aggregation.
        """
        ...

    @abc.abstractmethod
    def _aggregate_metrics(
        self, all_step_metrics: list[dict], step: int, *, grad_norm: torch.Tensor = None
    ) -> dict:
        """Aggregate per-step metrics into a single metrics dict.

        Called once per optimizer step after all micro-batches.
        """
        ...
