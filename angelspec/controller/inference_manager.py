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

"""Watermark-buffered, backpressure-aware batch dispatcher for inference engines.

Sits between the training controller and Ray inference engine actors.
Refills its prompt buffer when it drops below a low-watermark, dispatches
fixed-size batches to engines in round-robin order, and forwards the
resulting mooncake keys back to the controller's sample pool.

NOTE: The inference engine stores tensors in mooncake.
This dispatcher only receives keys and forwards them to training.

Data flow:
  Controller.prompt_buffer → dispatcher._prompt_buffer → Engine.generate()
  → Engine stores tensors in mooncake → returns dict with mooncake_key
  → dispatcher → Controller.sample_pool

Backpressure:
  When sample_pool exceeds _max_pool_size, pauses dispatching until
  training catches up and consumes data.
"""

import asyncio
import time
from collections import deque
from typing import Any

import ray
from ray.exceptions import RayActorError

from angelspec.utils.logging import logger
from angelspec.utils.types import InferenceInput, InferenceOutput

MOONCAKE_BACKPRESSURE_POLL_INTERVAL = 0.5  # seconds
MOONCAKE_BACKPRESSURE_LOG_INTERVAL = 5.0  # seconds
EVAL_PROMPT_POLL_INTERVAL = 0.05  # seconds


def _is_eval_entry(entry: InferenceInput) -> bool:
    return str(entry.data_id).startswith("eval_")


def _parse_bytes(value: int | str | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip().upper()
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    for unit in ("TB", "GB", "MB", "KB", "B"):
        if text.endswith(unit):
            return int(float(text[: -len(unit)].strip()) * units[unit])
    return int(text)


class NoHealthyEngine(RuntimeError):
    """Raised when every inference replica has been quarantined."""


class EnginePool:
    """Round-robin engine selection with per-replica concurrency control.

    A replica whose TP group has died (RayActorError / generate timeout / repeated
    failures) is quarantined and skipped, so the remaining replicas keep serving at
    reduced throughput instead of aborting the whole run.
    """

    def __init__(self, engines: list, max_concurrent_per_engine: int = 1):
        if not engines:
            raise ValueError("engines must be non-empty")
        self._engines = list(engines)
        self._quarantined: set[int] = set()
        self._consec_failures: dict[int, int] = {i: 0 for i in range(len(self._engines))}
        self._in_flight: dict[int, int] = {i: 0 for i in range(len(self._engines))}
        self._max_concurrent_per_engine = max_concurrent_per_engine
        self._available = asyncio.Condition()
        self._next_idx = 0
        self._max_concurrent = max_concurrent_per_engine * len(self._engines)
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    async def acquire(self):
        async with self._available:
            while True:
                if self.active_count == 0:
                    raise NoHealthyEngine(
                        f"all {len(self._engines)} inference replicas quarantined"
                    )
                n = len(self._engines)
                for _ in range(n):
                    idx = self._next_idx
                    self._next_idx = (self._next_idx + 1) % n
                    if (
                        idx not in self._quarantined
                        and self._in_flight[idx] < self._max_concurrent_per_engine
                    ):
                        self._in_flight[idx] += 1
                        return self._engines[idx]
                await self._available.wait()

    async def release(self, engine) -> None:
        idx = self._index_of(engine)
        if idx is None:
            return
        async with self._available:
            if self._in_flight[idx] > 0:
                self._in_flight[idx] -= 1
            self._available.notify_all()

    def pick(self):
        """Return the next healthy engine in round-robin order.

        Raises NoHealthyEngine when all replicas are quarantined.
        """
        n = len(self._engines)
        for _ in range(n):
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % n
            if idx not in self._quarantined:
                return self._engines[idx]
        raise NoHealthyEngine(f"all {n} inference replicas quarantined")

    def _index_of(self, engine) -> int | None:
        for i, e in enumerate(self._engines):
            if e is engine:
                return i
        return None

    def quarantine(self, engine, reason: str) -> None:
        """Remove a replica from rotation. Idempotent."""
        idx = self._index_of(engine)
        if idx is None or idx in self._quarantined:
            return
        self._quarantined.add(idx)
        logger.error(
            "Quarantined inference replica %d (%s); %d/%d replicas alive",
            idx,
            reason,
            self.active_count,
            len(self._engines),
        )

    def record_failure(self, engine) -> int:
        """Increment consecutive-failure count for a replica, return new value."""
        idx = self._index_of(engine)
        if idx is None:
            return 0
        self._consec_failures[idx] += 1
        return self._consec_failures[idx]

    def record_success(self, engine) -> None:
        idx = self._index_of(engine)
        if idx is not None:
            self._consec_failures[idx] = 0

    @property
    def active_count(self) -> int:
        return len(self._engines) - len(self._quarantined)

    @property
    def consec_failures(self) -> dict[int, int]:
        return dict(self._consec_failures)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    def __len__(self) -> int:
        return len(self._engines)


class MetricsCollector:
    """Collects per-sample metrics from engine outputs and aggregates on flush."""

    # engine output key → buffer key
    _KEYS = {
        "e2e_latency": "e2e_latencies",
        "prompt_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "spec_accept_rate": "spec_accept_rates",
        "spec_accept_length": "spec_accept_lengths",
    }

    # buffer key → (wandb metric prefix, include min/max)
    _AGGREGATIONS = {
        "e2e_latencies": ("inference/e2e_latency", True),
        "prompt_tokens": ("inference/prompt_tokens", False),
        "completion_tokens": ("inference/completion_tokens", False),
        "spec_accept_rates": ("inference/spec_accept_rate", False),
        "spec_accept_lengths": ("inference/spec_accept_length", False),
    }

    def __init__(self):
        self._buffer: dict[str, list] = {k: [] for k in self._AGGREGATIONS}

    def record(self, output: dict) -> None:
        """Record metrics from a single engine output dict."""
        for key, buffer_key in self._KEYS.items():
            if key in output:
                self._buffer[buffer_key].append(output[key])

    def flush(self) -> dict:
        """Return aggregated metrics and reset."""
        metrics = {}
        for buffer_key, (prefix, with_minmax) in self._AGGREGATIONS.items():
            values = self._buffer[buffer_key]
            if not values:
                continue
            metrics[f"{prefix}_avg"] = sum(values) / len(values)
            if with_minmax:
                metrics[f"{prefix}_min"] = min(values)
                metrics[f"{prefix}_max"] = max(values)
        self._buffer = {k: [] for k in self._AGGREGATIONS}
        return metrics


@ray.remote
class AsyncInferenceManager:
    """Dispatches inference batches to distributed Ray engine actors.

    Main loop::

        Controller ──get_prompts──→ [_prompt_buffer]
                                         │
                        ┌────────────────┘
                        ▼
                  _dispatch_ready_batches ──→ EnginePool ──→ engine.generate()
                        │                        ▲
                        │ (backpressure:          │ semaphore
                        │  pool full → wait)      │
                        ▼                         │
                  _collect_completed ◄─── _pending_tasks
                        │
                        ▼
                  _forward_results ──→ Controller.push_inference_results()

    Public API (called via Ray remote):
      run(), stop(), flush_metrics(), get_status(), get_pool_status()
    """

    def __init__(
        self,
        args,
        controller,
        inference_engines: list | None = None,
        max_concurrent_batches: int = 1,
    ):
        self.args = args
        self.controller = controller

        self._engines = EnginePool(inference_engines or [], max_concurrent_batches)
        self._metrics = MetricsCollector()

        # DP-wave dispatch: with data_parallel_size>1 the inference_engines ARE the
        # DP-group heads of external_launcher worlds, world by world, dp_size heads a
        # world. vLLM's DP coordinator steps a world's DP ranks in lockstep, so a head
        # with no work hangs the busy ones (verified via skewed dispatch). We therefore
        # dispatch a "wave" to ALL heads of a world at once, one wave in flight per
        # world; idle heads get a duplicate of a real prompt as padding -- its
        # connector store overwrites the same mooncake_key (no junk keys, no
        # connector change), and duplicate results are de-duped by data_id.
        self._dp_size = int(getattr(args, "vllm_data_parallel_size", 1) or 1)
        self._dp_mode = self._dp_size > 1
        self._dp_busy: set[int] = set()

        self._prompt_buffer: deque[InferenceInput] = deque()
        # Eval prompts skip both the buffered training prompts and the training pool's
        # backpressure: their results go to the eval pool, which the trainers drain while
        # the training loop waits on the eval, so holding them back only stalls the eval.
        self._eval_buffer: deque[InferenceInput] = deque()
        self._next_eval_poll = 0.0
        self._buffer_low_watermark = getattr(args, "inference_buffer_threshold", 32)
        self._refill_size = getattr(args, "inference_fetch_batch", 1)
        self._batch_size = getattr(args, "inference_batch_size", 8)
        self._eval_batch_size = (
            getattr(args, "inference_eval_batch_size", None) or self._batch_size
        )
        minimum_watermark = self._batch_size * (self._dp_size if self._dp_mode else 1)
        if self._buffer_low_watermark < minimum_watermark:
            logger.warning(
                "Raising inference_buffer_threshold from %s to %s to fit one dispatch",
                self._buffer_low_watermark,
                minimum_watermark,
            )
            self._buffer_low_watermark = minimum_watermark
        self._max_pool_size = getattr(args, "max_sample_pool_size", 0)
        self._max_pool_bytes = _parse_bytes(getattr(args, "max_sample_pool_bytes", 0))

        self._running = True
        self._pending_tasks: set[asyncio.Task] = set()
        # Prompts on their way from the controller and results of finished dispatches
        # on their way back are in neither _prompt_buffer nor _pending_tasks; this
        # counts those transfers until they land.
        self._in_transit = 0
        self._source_dry = False

        self._ft_enabled = getattr(args, "inference_ft_enabled", True)
        self._generate_timeout_s = getattr(args, "inference_generate_timeout_s", 0.0)
        self._quarantine_max_consec_failures = getattr(
            args, "inference_quarantine_max_consec_failures", 3
        )
        self._min_active_engines = getattr(args, "inference_min_active_engines", 1)

        self._defer_tokenization = getattr(args, "defer_tokenization", False)
        self._return_hidden_states = getattr(args, "compute_logits_in_trainer", True)
        self._train_with_decode = getattr(args, "train_with_decode", False)

        self._enable_perf_metrics = getattr(args, "enable_perf_metrics", True)
        self._perf_window_seconds: float = 10.0
        # Each entry: (num_samples, elapsed_seconds, finish_timestamp)
        self._batch_times: deque[tuple[int, float, float]] = deque(maxlen=50)

        self._last_pool_full_log_time = 0.0
        self._last_pool_wait_log_time = 0.0

        if self._max_pool_size > 0:
            logger.info(f"Flow control enabled: max_pool_size={self._max_pool_size} samples")
        else:
            logger.warning("Flow control disabled: max_pool_size=0 (unlimited generation)")
        if self._max_pool_bytes > 0:
            logger.info(
                "Byte flow control enabled: max_pool_bytes=%.1f GiB",
                self._max_pool_bytes / 1024**3,
            )

    # -- Public API ----------------------------------------------------------

    async def run(self) -> None:
        """Main loop: refill runs in background, dispatch → collect in foreground."""
        logger.info(
            f"AsyncInferenceManager starting ({len(self._engines)} engines, "
            f"max_concurrent={self._engines.max_concurrent})"
        )

        refill_task = asyncio.create_task(self._continuous_refill())

        while self._running:
            await self._dispatch_ready_batches()

            if not self._pending_tasks:
                await asyncio.sleep(0.01)
                continue

            await self._collect_completed()

        refill_task.cancel()
        try:
            await refill_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass  # a real refill failure was already reported via set_inference_error
        await self._drain()
        logger.info("AsyncInferenceManager stopped")

    async def _continuous_refill(self) -> None:
        """Background task: continuously refill prompt buffer."""
        while self._running:
            try:
                await self._refill_buffer()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                import traceback

                logger.critical(
                    f"Prompt refill task failed, terminating inference manager: {e}\n"
                    f"{traceback.format_exc()}"
                )
                self._running = False
                await self.controller.set_inference_error.remote(str(e))
                raise
            await asyncio.sleep(0.001)  # yield to event loop

    def stop(self) -> None:
        """Signal the main loop to stop after current iteration."""
        self._running = False

    def flush_metrics(self) -> dict:
        """Return aggregated metrics and reset the buffer."""
        metrics = self._metrics.flush()
        if self._enable_perf_metrics:
            metrics.update(self._compute_perf_metrics())
        return metrics

    def _compute_perf_metrics(self) -> dict:
        """Compute throughput metrics over a sliding time window."""
        if not self._batch_times:
            return {}

        cutoff = time.time() - self._perf_window_seconds
        recent = [
            (samples, elapsed)
            for samples, elapsed, finished_at in self._batch_times
            if finished_at > cutoff
        ]
        if not recent:
            return {}

        total_samples = sum(samples for samples, _ in recent)
        total_time = sum(elapsed for _, elapsed in recent)
        if total_time <= 0.001:
            return {}

        per_slot_rate = total_samples / total_time
        return {
            "perf/infer_capacity": per_slot_rate * self._engines.max_concurrent,
            "perf/infer_batch_time": total_time / len(recent),
        }

    async def get_status(self) -> dict:
        """Get current status of inference manager."""
        status = {
            "mode": f"distributed_engines ({len(self._engines)})",
            "prompt_buffer_size": len(self._prompt_buffer),
            "running": self._running,
            "pending_tasks": len(self._pending_tasks),
            "in_transit": self._in_transit,
            "engine_semaphore_available": getattr(self._engines.semaphore, "_value", None),
            "active_engines": self._engines.active_count,
            "quarantined_engines": len(self._engines) - self._engines.active_count,
            "engine_consec_failures": self._engines.consec_failures,
        }
        status.update(await self.get_pool_status())
        return status

    async def get_pool_status(self) -> dict:
        """Get current sample pool status."""
        try:
            pool_size = await self.controller.get_pool_size.remote()
        except Exception:
            pool_size = -1

        return {
            "max_pool_size": self._max_pool_size,
            "current_pool_size": pool_size,
            "pool_usage_percent": (
                int(pool_size * 100 / self._max_pool_size)
                if self._max_pool_size > 0 and pool_size >= 0
                else 0
            ),
            "flow_control_enabled": self._max_pool_size > 0,
        }

    # -- Main loop internals -------------------------------------------------

    async def _dispatch_ready_batches(self) -> None:
        """Dispatch batches from buffer, up to the engine concurrency limit."""
        if self._dp_mode:
            await self._dispatch_dp_waves()
            return
        while len(self._pending_tasks) < self._engines.max_concurrent and (
            await self._pull_eval_prompts()
        ):
            entries = [
                self._eval_buffer.popleft()
                for _ in range(min(self._eval_batch_size, len(self._eval_buffer)))
            ]
            self._pending_tasks.add(asyncio.create_task(self._dispatch_batch(entries)))

        while (
            len(self._prompt_buffer) >= self._batch_size
            and len(self._pending_tasks) < self._engines.max_concurrent
        ):
            if not await self._await_pool_capacity():
                return
            entries = self._take_batch(self._batch_size)
            task = asyncio.create_task(self._dispatch_batch(entries))
            self._pending_tasks.add(task)

        if (
            self._source_dry
            and 0 < len(self._prompt_buffer) < self._batch_size
            and len(self._pending_tasks) < self._engines.max_concurrent
        ):
            if not await self._await_pool_capacity():
                return
            entries = self._take_batch(self._batch_size)
            self._pending_tasks.add(asyncio.create_task(self._dispatch_batch(entries)))

    async def _pull_eval_prompts(self) -> bool:
        """Stage pending eval prompts from the controller; True if any are staged."""
        now = time.monotonic()
        if not self._eval_buffer and now >= self._next_eval_poll:
            self._next_eval_poll = now + EVAL_PROMPT_POLL_INTERVAL
            entries = await self.controller.get_eval_prompts.remote(
                self._batch_size * max(1, self._engines.max_concurrent)
            )
            self._eval_buffer.extend(entries)
        return bool(self._eval_buffer)

    def _entry_len(self, e) -> int:
        ids = getattr(e, "input_ids", None)
        if ids is not None:
            try:
                return len(ids)
            except Exception:
                pass
        s = getattr(e, "formatted_prompt", None) or getattr(e, "prompt", None) or ""
        return len(s)

    def _length_balanced_shards(self, entries: list, n: int) -> list[list]:
        """Split entries into n length-balanced shards (one per DP head).

        Lockstep wall-clock is the slowest shard, so greedily place the longest
        remaining prompt onto the currently-lightest shard.
        """
        shards: list[list] = [[] for _ in range(n)]
        load = [0] * n
        for e in sorted(entries, key=self._entry_len, reverse=True):
            i = min(range(n), key=lambda k: load[k])
            shards[i].append(e)
            load[i] += self._entry_len(e) + 1
        return shards

    def _dp_worlds(self) -> list[list]:
        """The DP heads grouped by world (the factory returns them world by world)."""
        heads = list(self._engines._engines)
        return [heads[i : i + self._dp_size] for i in range(0, len(heads), self._dp_size)]

    async def _dispatch_dp_waves(self) -> None:
        """DP mode: dispatch a wave to ALL DP heads of each idle world in lockstep.

        Every head of a world must enter the generate collective each wave or the
        busy heads hang on DP coordination. Idle heads (tail/drain) get a duplicate
        of a real prompt as padding; its store is idempotent (same mooncake_key) and
        its duplicate result is de-duped by data_id.
        """
        wave_size = self._dp_size * max(1, self._batch_size)
        if await self._pull_eval_prompts():
            self._prompt_buffer.extendleft(reversed(self._eval_buffer))
            self._eval_buffer.clear()
        # One wave in flight per world: a world's heads are coupled, so overlapping
        # waves would just contend on the same engines. Worlds run independently.
        for world, heads in enumerate(self._dp_worlds()):
            if world in self._dp_busy:
                continue
            if not (
                self._prompt_buffer
                and (
                    len(self._prompt_buffer) >= wave_size
                    or self._source_dry
                    or _is_eval_entry(self._prompt_buffer[0])
                )
            ):
                return
            if _is_eval_entry(self._prompt_buffer[0]):
                # Eval waves hold eval prompts only, inference_eval_batch_size a head, so the
                # eval spreads over the worlds instead of riding one world's training wave.
                n = 0
                limit = min(
                    self._dp_size * max(1, self._eval_batch_size), len(self._prompt_buffer)
                )
                while n < limit and _is_eval_entry(self._prompt_buffer[n]):
                    n += 1
                entries = self._take_batch(n)
            elif await self._await_pool_capacity():
                entries = self._take_batch(min(wave_size, len(self._prompt_buffer)))
            else:
                return
            if not entries:
                return
            shards = self._length_balanced_shards(entries, len(heads))
            self._dp_busy.add(world)
            self._pending_tasks.add(
                asyncio.create_task(self._dispatch_dp_wave(heads, shards, world))
            )

    async def _dispatch_dp_wave(
        self, heads: list, shards: list[list], world: int = 0
    ) -> list[tuple[InferenceInput, dict | Exception]]:
        """Fire generate on all DP heads of a world at once, await the wave, align+dedup."""
        try:
            return await self._run_dp_wave(heads, shards)
        finally:
            self._dp_busy.discard(world)

    async def _run_dp_wave(
        self, heads: list, shards: list[list]
    ) -> list[tuple[InferenceInput, dict | Exception]]:
        async with self._engines.semaphore:
            real_entries = [e for shard in shards for e in shard]
            if not real_entries:
                return []
            pad = real_entries[0]  # idle head steps with a dup real prompt (no junk key)
            futs = []
            for shard in shards:
                kwargs = self._prepare_engine_inputs(shard if shard else [pad])
                head = heads[len(futs)]
                gen = (
                    head.generate_with_decode if self._train_with_decode else head.generate
                ).remote(
                    data_id=kwargs["data_ids"],
                    input_ids_ref=kwargs["input_ids_ref"],
                    packed_loss_mask_list=kwargs["packed_loss_mask_list"],
                    formatted_prompts=kwargs["formatted_prompts"],
                    return_last_hidden_states=kwargs["return_last_hidden_states"],
                    return_logits=kwargs["return_logits"],
                    multimodal_inputs=kwargs["multimodal_inputs"],
                )
                futs.append(gen)
            try:
                if self._generate_timeout_s and self._generate_timeout_s > 0:
                    head_outputs = await asyncio.wait_for(
                        asyncio.gather(*futs), timeout=self._generate_timeout_s
                    )
                else:
                    head_outputs = await asyncio.gather(*futs)
            except (RayActorError, asyncio.TimeoutError) as e:
                # DP world is coupled — a stuck/failed head kills the whole wave.
                await self._escalate_fatal(f"DP-wave generate failed/timed out: {e}")
                self._requeue(real_entries)
                return []
            for head in heads:
                self._engines.record_success(head)
            output_by_id: dict = {}
            for outs in head_outputs:
                for o in outs:
                    if isinstance(o, dict) and o.get("data_id") is not None:
                        output_by_id.setdefault(o["data_id"], o)
            aligned = []
            for e in real_entries:
                o = output_by_id.get(e.data_id)
                if o is None:
                    o = ValueError(f"DP-wave: no result for data_id={e.data_id}")
                aligned.append((e, o))
            return aligned

    async def _collect_completed(self) -> None:
        """Wait for at least one dispatch to finish and forward results."""
        # One DP world takes its next wave only once this one is back; with several,
        # the idle ones take waves while the others run.
        single_world = self._dp_mode and len(self._dp_worlds()) == 1
        done, self._pending_tasks = await asyncio.wait(
            self._pending_tasks,
            timeout=None if single_world else 0.01,
            return_when=asyncio.FIRST_COMPLETED,
        )
        self._in_transit += len(done)
        try:
            for task in done:
                await self._forward_results(task.result())
        finally:
            self._in_transit -= len(done)

    async def _drain(self) -> None:
        # Currently drain is only called on shutdown, so we don't need to await pool.
        if self._prompt_buffer or self._eval_buffer:
            logger.info(
                f"Drain: discarding {len(self._prompt_buffer) + len(self._eval_buffer)} "
                "buffered prompts on shutdown"
            )
            self._prompt_buffer.clear()
            self._eval_buffer.clear()
        if self._pending_tasks:
            done, _ = await asyncio.wait(self._pending_tasks, return_when=asyncio.ALL_COMPLETED)
            for task in done:
                await self._forward_results(task.result())
            self._pending_tasks.clear()

    # -- Buffer management ---------------------------------------------------

    async def _refill_buffer(self) -> None:
        """Fetch prompts from controller if buffer is below watermark.

        Requests enough prompts to fill the buffer up to the watermark in one
        call, avoiding multiple round-trips to the controller.
        """
        if len(self._prompt_buffer) >= self._buffer_low_watermark:
            return

        if self._max_pool_size > 0:
            pool_size = await self.controller.get_pool_size.remote()
            if pool_size >= self._max_pool_size:
                now = time.monotonic()
                if now - self._last_pool_full_log_time >= 2.0:
                    self._last_pool_full_log_time = now
                    logger.debug(
                        f"Skipping prompt fetch: pool full ({pool_size}/{self._max_pool_size})"
                    )
                return
        if self._max_pool_bytes > 0:
            pool_bytes = await self.controller.get_pool_bytes.remote()
            if pool_bytes >= self._max_pool_bytes:
                return

        need = self._buffer_low_watermark - len(self._prompt_buffer)
        fetch_size = max(need, self._refill_size)
        self._in_transit += 1
        try:
            entries = await self.controller.get_prompts.remote(fetch_size)
            if entries:
                self._prompt_buffer.extend(entries)
                self._source_dry = False
                logger.debug(
                    f"Fetched {len(entries)} prompts, buffer_size={len(self._prompt_buffer)}"
                )
            else:
                self._source_dry = True
        finally:
            self._in_transit -= 1

    def _take_batch(self, n: int) -> list[InferenceInput]:
        """Pop up to n entries from internal buffer."""
        n = min(n, len(self._prompt_buffer))
        return [self._prompt_buffer.popleft() for _ in range(n)]

    # -- Backpressure --------------------------------------------------------

    async def _await_pool_capacity(self) -> bool:
        """Block until sample pool has capacity.

        Returns False when eval prompts arrive or an in-flight dispatch finishes during
        the wait, so the caller hands the loop back to dispatch the eval prompts and
        forward the finished results. Holding finished results back would starve the
        eval: its next prompts are only submitted once its current results arrive.
        """
        if self._max_pool_size <= 0 and self._max_pool_bytes <= 0:
            return True

        pool_size = await self.controller.get_pool_size.remote()
        pool_bytes = await self.controller.get_pool_bytes.remote()
        size_full = self._max_pool_size > 0 and pool_size >= self._max_pool_size
        bytes_full = self._max_pool_bytes > 0 and pool_bytes >= self._max_pool_bytes
        if not size_full and not bytes_full:
            return True

        wait_start = time.time()
        last_log_time = wait_start

        if wait_start - self._last_pool_wait_log_time >= MOONCAKE_BACKPRESSURE_LOG_INTERVAL:
            self._last_pool_wait_log_time = wait_start
            logger.warning(
                "Sample pool full, pausing generation: samples=%s/%s bytes=%.1f/%.1f GiB",
                pool_size,
                self._max_pool_size or "unlimited",
                pool_bytes / 1024**3,
                self._max_pool_bytes / 1024**3 if self._max_pool_bytes else 0.0,
            )

        while self._running:
            size_full = self._max_pool_size > 0 and pool_size >= self._max_pool_size
            bytes_full = self._max_pool_bytes > 0 and pool_bytes >= self._max_pool_bytes
            if not size_full and not bytes_full:
                break
            if await self._pull_eval_prompts() or any(t.done() for t in self._pending_tasks):
                return False
            await asyncio.sleep(MOONCAKE_BACKPRESSURE_POLL_INTERVAL)
            pool_size = await self.controller.get_pool_size.remote()
            pool_bytes = await self.controller.get_pool_bytes.remote()

            now = time.time()
            if now - last_log_time >= MOONCAKE_BACKPRESSURE_LOG_INTERVAL:
                wait_duration = now - wait_start
                logger.info(
                    "Still waiting for pool capacity (waited %.1fs): samples=%s/%s "
                    "bytes=%.1f/%.1f GiB",
                    wait_duration,
                    pool_size,
                    self._max_pool_size or "unlimited",
                    pool_bytes / 1024**3,
                    self._max_pool_bytes / 1024**3 if self._max_pool_bytes else 0.0,
                )
                last_log_time = now

        wait_duration = time.time() - wait_start
        if not self._running:
            logger.info(f"Pool wait aborted (shutdown requested) after {wait_duration:.1f}s")
            return True
        logger.info(
            f"Pool capacity available after {wait_duration:.1f}s, pool_size={pool_size}/{self._max_pool_size}"
        )
        return True

    # -- Engine dispatch -----------------------------------------------------

    def _prepare_engine_inputs(self, entries: list[InferenceInput]) -> dict:
        """Pack entries into engine.generate() kwargs."""
        packed_loss_mask_list = [e.packed_loss_mask for e in entries]
        data_ids = [e.data_id for e in entries]
        multimodal_inputs = [e.multimodal_inputs for e in entries]
        has_multimodal = any(m is not None for m in multimodal_inputs)

        if self._defer_tokenization:
            input_ids_ref = None
            packed_loss_mask_list = None
            assert all(
                e.formatted_prompt is not None for e in entries
            ), "formatted_prompt is required when defer_tokenization is True"
            formatted_prompts = [e.formatted_prompt for e in entries]
        else:
            input_ids_ref = ray.put([e.input_ids for e in entries])
            formatted_prompts = None

        return {
            "data_ids": data_ids,
            "input_ids_ref": input_ids_ref,
            "packed_loss_mask_list": packed_loss_mask_list,
            "formatted_prompts": formatted_prompts,
            "return_last_hidden_states": self._return_hidden_states,
            "return_logits": not self._return_hidden_states,
            "multimodal_inputs": multimodal_inputs if has_multimodal else None,
        }

    async def _escalate_fatal(self, msg: str) -> None:
        """Stop the manager and signal the controller (outer restart takes over)."""
        logger.critical("Inference fatal, terminating manager: %s", msg)
        self._running = False
        await self.controller.set_inference_error.remote(msg)

    def _requeue(self, entries: list[InferenceInput]) -> None:
        """Put undispatched entries back at the head of their own buffer."""
        self._eval_buffer.extendleft(reversed([e for e in entries if _is_eval_entry(e)]))
        self._prompt_buffer.extendleft(reversed([e for e in entries if not _is_eval_entry(e)]))

    async def _quarantine_and_requeue(
        self, engine, reason: str, entries: list[InferenceInput]
    ) -> None:
        """Drop a dead replica, put its in-flight entries back at the buffer head,
        and escalate to a fatal error only if too few replicas remain."""
        self._engines.quarantine(engine, reason)
        self._requeue(entries)
        if self._engines.active_count < self._min_active_engines:
            await self._escalate_fatal(
                f"{self._engines.active_count} inference replicas alive "
                f"(< min {self._min_active_engines}); last: {reason}"
            )

    async def _dispatch_batch(
        self, entries: list[InferenceInput]
    ) -> list[tuple[InferenceInput, dict | Exception]]:
        """Send a batch to the next engine and return (entry, output) pairs."""
        engine = None
        try:
            engine = await self._engines.acquire()
        except NoHealthyEngine as e:
            await self._escalate_fatal(str(e))
            self._requeue(entries)
            return []

        try:
            kwargs = self._prepare_engine_inputs(entries)
            try:
                if self._enable_perf_metrics:
                    t0 = time.time()
                generate_method = (
                    engine.generate_with_decode if self._train_with_decode else engine.generate
                )
                gen = generate_method.remote(
                    data_id=kwargs["data_ids"],
                    input_ids_ref=kwargs["input_ids_ref"],
                    packed_loss_mask_list=kwargs["packed_loss_mask_list"],
                    formatted_prompts=kwargs["formatted_prompts"],
                    return_last_hidden_states=kwargs["return_last_hidden_states"],
                    return_logits=kwargs["return_logits"],
                    multimodal_inputs=kwargs["multimodal_inputs"],
                )
                if self._generate_timeout_s and self._generate_timeout_s > 0:
                    outputs = await asyncio.wait_for(gen, timeout=self._generate_timeout_s)
                else:
                    outputs = await gen
                if self._enable_perf_metrics:
                    now = time.time()
                    self._batch_times.append((len(entries), now - t0, now))
                self._engines.record_success(engine)
                output_by_id = {
                    output.get("data_id"): output
                    for output in outputs
                    if isinstance(output, dict) and output.get("data_id") is not None
                }
                aligned = []
                for entry in entries:
                    output = output_by_id.get(entry.data_id)
                    if output is None:
                        output = ValueError(
                            f"engine returned no result for data_id={entry.data_id}"
                        )
                    aligned.append((entry, output))
                if len(output_by_id) != len(entries):
                    logger.warning(
                        "Engine returned %d/%d data-id-aligned results",
                        len(output_by_id),
                        len(entries),
                    )
                return aligned
            except (RayActorError, asyncio.TimeoutError) as e:
                reason = (
                    f"generate timeout after {self._generate_timeout_s}s"
                    if isinstance(e, asyncio.TimeoutError)
                    else f"actor died: {e}"
                )
                if not self._ft_enabled:
                    logger.critical(f"Engine actor died, terminating inference manager: {e}")
                    if isinstance(e, RayActorError):
                        self._running = False
                        await self.controller.set_inference_error.remote(str(e))
                        raise
                    return [(entry, e) for entry in entries]
                await self._quarantine_and_requeue(engine, reason, entries)
                return []
            except Exception as e:
                import traceback

                logger.error(f"Engine dispatch failed: {e}\n{traceback.format_exc()}")
                if self._ft_enabled:
                    count = self._engines.record_failure(engine)
                    if count >= self._quarantine_max_consec_failures:
                        await self._quarantine_and_requeue(
                            engine, f"{count} consecutive failures: {e}", []
                        )
                return [(entry, e) for entry in entries]
        finally:
            await self._engines.release(engine)

    # -- Result processing ---------------------------------------------------

    def _parse_engine_output(self, entry: InferenceInput, output: dict) -> InferenceOutput | None:
        """Convert engine output dict to InferenceOutput."""
        if not isinstance(output, dict) or "mooncake_key" not in output:
            logger.debug(
                f"Skipping invalid engine output for data_id={entry.data_id}: {type(output)}"
            )
            return None

        self._metrics.record(output)

        metadata = dict(entry.metadata or {})
        metadata.update(output.get("metadata", {}) or {})

        return InferenceOutput(
            data_id=entry.data_id,
            mooncake_key=output["mooncake_key"],
            tensor_shapes=output.get("tensor_shapes", {}),
            tensor_dtypes=output.get("tensor_dtypes", {}),
            packed_loss_mask=output.get("packed_loss_mask", entry.packed_loss_mask),
            metadata=metadata,
        )

    async def _forward_results(self, results: list[tuple[InferenceInput, Any | Exception]]) -> int:
        """Parse results and forward to controller. Returns success count."""
        inference_results = []

        for entry, result in results:
            if isinstance(result, Exception):
                continue

            inference_result = self._parse_engine_output(entry, result)
            if inference_result is not None:
                inference_results.append(inference_result)

        if inference_results:
            await self.controller.push_inference_results.remote(inference_results)
            logger.debug(f"Forwarded {len(inference_results)} results to controller")

        return len(inference_results)
