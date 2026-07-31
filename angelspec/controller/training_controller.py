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

"""
Async Training Controller for decoupled inference and training.

Data flow:
  load_dataset(args) → Stored Dataset → Prompt Buffer → Inference Manager
    → Sample Pool → Train Queues → Training Workers
  (Stored dataset is retained for epoch reloads and vocab mapping computation.)

Controller manages the tokenized dataset (for epoch reloads and vocab mapping),
prompt metadata, and mooncake keys. Actual inference tensor data is stored in
mooncake; the controller only tracks keys and byte sizes for backpressure.

Batch Size Design:
  micro_batch_size                   # Samples per GPU per dispatch (user config)
  per_dp_rank_batch_size             # = micro_batch_size * sp_size (derived)
  dispatch_batch_size            # = per_dp_rank_batch_size * dp_size (samples per dispatch)
  global_batch_size                  # = dispatch_batch_size * accumulation_steps (per optimizer step)

  Example with micro_batch_size=2, sp_size=1, dp_size=4, accumulation_steps=2:
    - per_dp_rank_batch_size = 2 * 1 = 2
    - dispatch_batch_size = 2 * 4 = 8
    - global_batch_size = 8 * 2 = 16

  Data flow per optimizer step (with accumulation_steps=2):
    1. Controller dispatches 8 samples per dispatch, 2 dispatches per optimizer step
    2. Each DP rank receives 2 samples per dispatch (4 total per optimizer step)
    3. Each train actor calls train_from_queue(num_batches=accumulation_steps)
    4. Forward/backward for each micro-batch, optimizer step after last one
"""

import copy
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import ray
from ray.util.queue import Queue

from angelspec.training.data_fetcher import TrainSample
from angelspec.utils.logging import logger
from angelspec.utils.memory import estimate_tensor_bytes
from angelspec.utils.types import InferenceInput, InferenceOutput

_estimate_bytes = estimate_tensor_bytes


def validate_packing_candidates(candidates: list[tuple[str, int]], max_seq: int) -> None:
    """Reject samples that can never fit a packed row.

    Waiting for more supply cannot make these samples packable, so treating this
    as a configuration/data error avoids a permanently wedged controller pool.
    """
    invalid = [(key, length) for key, length in candidates if length <= 0 or length > max_seq]
    if not invalid:
        return
    preview = ", ".join(f"{key}:{length}" for key, length in invalid[:5])
    raise ValueError(
        "Packing received samples that can never fit a row: "
        f"{preview}. training.max_seq_length={max_seq}. "
        "Align the inference generation/model length with the training packing "
        "limit. Refusing to wait forever or silently drop data."
    )


def packed_rows_fill_ratio(
    rank_rows: list[list[list[int]]],
    lengths: list[int],
    max_seq: int,
) -> float:
    num_rows = sum(len(rows) for rows in rank_rows)
    if num_rows <= 0 or max_seq <= 0:
        return 0.0
    packed_tokens = sum(lengths[idx] for rows in rank_rows for row in rows for idx in row)
    return packed_tokens / (num_rows * max_seq)


def should_wait_for_packing_fill(
    *,
    fill_ratio: float,
    min_fill_ratio: float,
    waited_seconds: float,
    max_wait_seconds: float,
) -> bool:
    """Whether a low-fill candidate should wait for more inference supply."""
    return fill_ratio < min_fill_ratio and waited_seconds < max_wait_seconds


def pack_into_rows(
    lengths: list[int],
    dp_size: int,
    rows_per_rank: int,
    max_seq: int,
) -> tuple[list[list[list[int]]], int] | None:
    """Bin-pack samples into exactly ``dp_size * rows_per_rank`` packed rows.

    Pure, deterministic, no I/O — the single source of truth for how samples map
    to packed rows (trainer does NOT re-pack; it just groups by the row_id the
    controller stamps). Given each sample's token length, greedily fills rows in
    FIFO order up to ``max_seq``, then assigns whole rows to DP ranks by
    longest-processing-time so per-rank total tokens are balanced.

    Args:
        lengths: token length of each candidate sample (index = sample position).
        dp_size: number of DP ranks.
        rows_per_rank: R — rows each rank must get (== grad-accum micro-batches).
        max_seq: max tokens per packed row.

    Returns:
        ``(rank_rows, num_consumed)`` where ``rank_rows[r]`` is a list of exactly
        ``rows_per_rank`` rows, each row a list of sample indices (into
        ``lengths``); ``num_consumed`` is how many of the leading samples were
        actually placed (a prefix — callers pop exactly this many from the pool).
        Returns ``None`` if there are not enough non-empty rows yet (caller should
        wait for more supply). Rows may be partially filled when the available
        supply ends; callers should monitor the resulting fill ratio. A single
        sample longer than ``max_seq`` is skipped when this pure helper is called
        directly; the controller rejects such samples with a configuration error
        before packing so they cannot wedge the live pool.
    """
    need_rows = dp_size * rows_per_rank
    if need_rows <= 0 or max_seq <= 0:
        return None

    # Greedy bin packing over the FIFO pool (must track which leading prefix is
    # consumed): append to the open row, close it on overflow, and stop at
    # need_rows closed rows.
    rows: list[list[int]] = []  # closed rows (each a list of sample idx)
    row_tokens: list[int] = []  # token count per closed row
    cur: list[int] = []
    cur_len = 0
    consumed = 0

    for idx, s in enumerate(lengths):
        if len(rows) >= need_rows:
            break
        consumed = idx  # exclusive prefix length before placing this sample
        if s <= 0:
            consumed = idx + 1
            continue
        if s > max_seq:
            # Impossible to pack; skip it (still consume so pool advances).
            consumed = idx + 1
            continue
        if cur and cur_len + s > max_seq:
            rows.append(cur)
            row_tokens.append(cur_len)
            cur = [idx]
            cur_len = s
        else:
            cur.append(idx)
            cur_len += s
        consumed = idx + 1

    # Close the trailing open row only if supply ended before need_rows (else the
    # open remainder is left unconsumed for the next dispatch).
    if len(rows) < need_rows and cur:
        rows.append(cur)
        row_tokens.append(cur_len)

    if len(rows) < need_rows:
        return None  # not enough to fill all rows yet — wait for supply

    # Consumed = last sample index in the kept rows (open row discarded above).
    rows = rows[:need_rows]
    row_tokens = row_tokens[:need_rows]
    consumed = max(r[-1] for r in rows) + 1

    # Assign whole rows to ranks by longest-processing-time (balance per-rank
    # tokens). Sort rows by token count desc, drop each into the currently
    # lightest rank that still has < rows_per_rank rows.
    order = sorted(range(need_rows), key=lambda i: row_tokens[i], reverse=True)
    rank_rows: list[list[list[int]]] = [[] for _ in range(dp_size)]
    rank_load = [0] * dp_size
    for ri in order:
        # eligible ranks not yet full
        cand = [r for r in range(dp_size) if len(rank_rows[r]) < rows_per_rank]
        r = min(cand, key=lambda r: rank_load[r])
        rank_rows[r].append(rows[ri])
        rank_load[r] += row_tokens[ri]

    return rank_rows, consumed


@dataclass
class SpeedMonitor:
    """Tracks throughput over a sliding time window."""

    window_seconds: float = 10.0
    _events: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _total_count: int = 0

    def record(self, count: int = 1) -> None:
        """Record count entries at current time."""
        now = time.time()
        with self._lock:
            self._events.append((now, count))
            self._total_count += count
            self._prune_old_events(now)

    def _prune_old_events(self, now: float) -> None:
        """Remove events outside the window."""
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def get_speed(self) -> float:
        """Get current speed in entries/sec over the window."""
        now = time.time()
        with self._lock:
            self._prune_old_events(now)
            if not self._events:
                return 0.0

            window_count = sum(count for _, count in self._events)
            oldest_time = self._events[0][0]
            elapsed = now - oldest_time
            if elapsed < 0.001:
                return 0.0
            return window_count / elapsed

    def get_total_count(self) -> int:
        """Get total count since start."""
        return self._total_count


@ray.remote
class AsyncTrainingController:
    """Central controller for async training pipeline.

    Responsibilities:
      - Loads and stores tokenized datasets for training and eval
      - Computes vocab mappings for draft model pruning
      - Manages prompt buffer (samples waiting for inference)
      - Manages sample pool (completed inferences waiting for training)
      - Dispatches batches to per-DP training queues when pool is full
      - Monitors inference and training throughput
    """

    def __init__(self, args, dp_size: int):
        self.args = args
        self.dp_size = dp_size
        self.sp_size = (
            getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
            if getattr(args, "attention_backend", None) == "usp"
            else 1
        )
        self.queue_count = dp_size * self.sp_size

        self.prompt_buffer: deque[InferenceInput] = deque()
        self._prompt_lock = threading.Lock()

        self.sample_pool: deque[InferenceOutput] = deque()
        self._pool_lock = threading.Lock()
        self._pool_bytes = 0
        self._sample_bytes: dict[str, int] = {}

        self.train_queues = [Queue() for _ in range(self.queue_count)]

        # Eval: separate pool and queues so eval data never mixes with training
        self.eval_pool: deque[InferenceOutput] = deque()
        self._eval_pool_lock = threading.Lock()
        self._eval_data_ids: set[str] = set()
        self._eval_expected_count: int = 0
        self._eval_dispatched_samples: int = 0
        self.eval_queues = [Queue() for _ in range(self.queue_count)]

        self.batch_id = 0
        self.dispatch_batch_size = args.per_dp_rank_batch_size * dp_size
        self.eval_dispatch_batch_size = dp_size
        self._data_id_counter = 0

        # Fixed-rows DFlash packing: each dispatch packs the pool into exactly dp_size
        # rows (one per rank) via pack_into_rows with a monotonic row_id, so every rank
        # gets the same rows/step and DP stays lock-step under FSDP resharding.
        # Off → legacy per-sample dispatch (row_id=None).
        self.packing_enabled = bool(
            getattr(args, "dflash_packing", False) or getattr(args, "mtp_packing", False)
        )
        self.packing_max_seq = int(getattr(args, "max_seq_length", 0) or 0)
        if self.packing_enabled and self.packing_max_seq <= 0:
            raise ValueError(
                "sequence packing (dflash_packing/mtp_packing) requires max_seq_length > 0."
            )
        self.packing_min_fill_ratio = float(getattr(args, "packing_min_fill_ratio", 0.0) or 0.0)
        self.packing_max_wait_seconds = float(getattr(args, "packing_max_wait_seconds", 5.0))
        if not 0.0 <= self.packing_min_fill_ratio <= 1.0:
            raise ValueError(
                f"packing_min_fill_ratio must be in [0, 1], got {self.packing_min_fill_ratio}"
            )
        if self.packing_max_wait_seconds < 0.0:
            raise ValueError(
                f"packing_max_wait_seconds must be >= 0, got {self.packing_max_wait_seconds}"
            )
        self._row_id_counter = 0
        self._packing_low_fill_since: float | None = None
        self._packing_low_fill_flushes = 0
        self._packing_wait_seconds_total = 0.0
        self._packing_last_dispatch_fill_ratio = 0.0

        self._stored_dataset: list | None = None
        self._stored_eval_dataset: list | None = None
        self._dataset_epoch: int = 0
        self._dataset_seed: int = getattr(args, "seed", 42)
        self._shuffle_dataset: bool = getattr(args, "shuffle_dataset", True)

        self._start_time = time.time()
        self._inference_monitor = SpeedMonitor(window_seconds=10.0)
        self._training_monitor = SpeedMonitor(window_seconds=10.0)
        self._last_dispatch_log_time = 0.0
        self._inference_error: str | None = None

    def _generate_data_id(self) -> str:
        self._data_id_counter += 1
        return f"data_{self._data_id_counter}"

    # ─────────────────────────────────────────────────────────────
    # Dataset Loading
    # ─────────────────────────────────────────────────────────────

    def add_dataset(self, dataset: list) -> int:
        with self._prompt_lock:
            for sample in dataset:
                if isinstance(sample, dict):
                    data_id = sample.get("data_id") or self._generate_data_id()
                    input_ids = sample.get("input_ids")
                    packed_loss_mask = sample.get("packed_loss_mask")
                    if input_ids is not None and packed_loss_mask is None:
                        raise ValueError(
                            f"packed_loss_mask is required when input_ids is provided "
                            f"(data_id={data_id}). Use defer_tokenization=True to skip "
                            f"tokenization entirely."
                        )
                    entry = InferenceInput(
                        data_id=data_id,
                        prompt=sample.get("prompt", sample),
                        input_ids=input_ids,
                        packed_loss_mask=packed_loss_mask,
                        formatted_prompt=sample.get("formatted_prompt"),
                        metadata=sample.get("metadata", {}),
                        multimodal_inputs=sample.get("multimodal_inputs"),
                    )
                else:
                    entry = InferenceInput(
                        data_id=self._generate_data_id(),
                        prompt=sample,
                    )
                self.prompt_buffer.append(entry)
            return len(dataset)

    def load_dataset(self, args) -> int:
        """Load and store dataset on the controller for later use."""
        from angelspec.data.dataset import load_conversation_dataset

        self._stored_dataset = load_conversation_dataset(args)
        if not self._stored_dataset:
            raise ValueError(
                f"Training dataset is empty after processing. "
                f"Check train_data_path='{args.train_data_path}', "
                f"max_seq_length={getattr(args, 'max_seq_length', None)}, "
                f"and chat_template settings."
            )
        logger.info(f"Controller loaded dataset: {len(self._stored_dataset)} samples")
        return len(self._stored_dataset)

    def _prepare_dataset(self, skip: int = 0) -> list:
        """Return dataset for the current epoch, optionally shuffled.

        When shuffle is enabled the ordering is deterministic from
        (seed + epoch), so resume can reconstruct the same epoch ordering
        and approximately skip samples consumed by completed optimizer
        steps.  This is best-effort only because async prompt/result
        buffers may still contain in-flight samples.
        """
        data = list(self._stored_dataset)
        if self._shuffle_dataset:
            import random

            rng = random.Random(self._dataset_seed + self._dataset_epoch)
            rng.shuffle(data)

        if skip > 0:
            skip = min(skip, len(data))
            data = data[skip:]

        shuffle_tag = (
            f"seed {self._dataset_seed}+{self._dataset_epoch}"
            if self._shuffle_dataset
            else "shuffle disabled"
        )
        logger.info(
            f"Prepared dataset ({len(data)} samples, {shuffle_tag}"
            + (f", skipped {skip})" if skip > 0 else ")")
        )
        return data

    def submit_training_dataset(self, epoch: int = 0, skip: int = 0) -> int:
        """Submit the stored training dataset to the prompt buffer for inference.

        Args:
            epoch: Current epoch number (for deterministic shuffle seed).
            skip: Number of samples to skip from the start (for resume mid-epoch).
        """
        assert self._stored_dataset is not None, "No stored dataset to submit"
        self._dataset_epoch = epoch
        return self.add_dataset(self._prepare_dataset(skip=skip))

    def reload_dataset(self) -> int:
        """Re-add the stored dataset to the prompt buffer (epoch reload)."""
        assert self._stored_dataset is not None, "No stored dataset to reload"
        self._dataset_epoch += 1
        return self.add_dataset(self._prepare_dataset())

    def load_eval_dataset(self, args) -> int:
        """Load eval dataset on the controller and store it. Returns size (0 if none)."""
        eval_data_path = getattr(args, "eval_data_path", None)
        if not eval_data_path:
            return 0

        from angelspec.data.dataset import load_conversation_dataset

        eval_args = copy.copy(args)
        eval_args.train_data_path = eval_data_path
        eval_prompt_key = getattr(args, "eval_prompt_key", None)
        if eval_prompt_key:
            eval_args.prompt_key = eval_prompt_key
        raw_dataset = load_conversation_dataset(eval_args)
        raw_count = len(raw_dataset)
        # Truncate to a multiple of dp_size so every dispatch is a full batch
        usable = (raw_count // self.dp_size) * self.dp_size
        if usable < raw_count:
            logger.info(
                f"Eval dataset truncated from {raw_count} to {usable} samples (dp_size={self.dp_size})"
            )
        self._stored_eval_dataset = raw_dataset[:usable] if usable > 0 else []
        count = len(self._stored_eval_dataset)
        logger.info(f"Controller loaded eval dataset: {count} samples from {eval_data_path}")
        return count

    def get_dataset_size(self) -> int:
        if self._stored_dataset is None:
            raise RuntimeError(
                "get_dataset_size() called but no dataset has been loaded. Call load_dataset() first."
            )
        return len(self._stored_dataset)

    def get_eval_dataset_size(self) -> int:
        return len(self._stored_eval_dataset) if self._stored_eval_dataset is not None else 0

    def compute_vocab_mapping(self, target_vocab_size: int, draft_vocab_size: int) -> tuple:
        """Generate vocab mapping on the controller using the stored dataset.

        Requires the dataset to have been loaded with defer_tokenization=False,
        since vocab mapping needs input_ids.
        """
        from angelspec.data.preprocessing import generate_vocab_mapping

        assert self._stored_dataset is not None, "No stored dataset for vocab mapping"
        assert (
            "input_ids" in self._stored_dataset[0]
        ), "compute_vocab_mapping requires input_ids in dataset. Set defer_tokenization=False to enable tokenization."
        return generate_vocab_mapping(
            prompts=self._stored_dataset,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
        )

    # ─────────────────────────────────────────────────────────────
    # Interface for Inference Manager
    # ─────────────────────────────────────────────────────────────

    def get_prompts(self, num_prompts: int) -> list[InferenceInput]:
        """Inference manager gets prompts with data_ids.

        Args:
            num_prompts: Maximum number of prompts to fetch.

        Returns:
            List of InferenceInput objects.
        """
        with self._prompt_lock:
            entries = []
            for _ in range(min(num_prompts, len(self.prompt_buffer))):
                entries.append(self.prompt_buffer.popleft())
            return entries

    def push_inference_results(self, results: list[InferenceOutput]) -> int:
        """Inference sends back (data_id, mooncake_key) pairs.

        Controller stores the keys and tracks exact bytes for backpressure.
        Eval results (identified by data_id) are routed to the eval pool.

        Args:
            results: List of InferenceOutput containing data_id, mooncake_key,
                    tensor_shapes, and tensor_dtypes.

        Returns:
            Current pool bytes after adding results. This allows inference manager
            to implement Mooncake backpressure.
        """
        eval_results = []
        train_results = []
        for result in results:
            if result.data_id in self._eval_data_ids:
                eval_results.append(result)
            else:
                train_results.append(result)

        if eval_results:
            with self._eval_pool_lock:
                self.eval_pool.extend(eval_results)

        pool_bytes = 0
        if train_results:
            with self._pool_lock:
                for result in train_results:
                    sample_bytes = estimate_tensor_bytes(
                        result.tensor_shapes or {},
                        result.tensor_dtypes or {},
                    )
                    self._sample_bytes[result.mooncake_key] = sample_bytes
                    self._pool_bytes += sample_bytes
                self.sample_pool.extend(train_results)
                pool_bytes = self._pool_bytes

        self._inference_monitor.record(len(results))
        return pool_bytes

    # ─────────────────────────────────────────────────────────────
    # Interface for Training
    # ─────────────────────────────────────────────────────────────

    def get_train_queues(self) -> list[Queue]:
        """Get the per-DP training queues."""
        return self.train_queues

    def get_pool_size(self) -> int:
        """Total mooncake-resident samples (training + eval) for backpressure.

        Always includes eval pool so that backpressure accounts for mooncake
        segment capacity used by eval data.  Without this, eval data occupies
        mooncake outside backpressure awareness and the segment overflows.
        """
        train_size = len(self.sample_pool)
        with self._eval_pool_lock:
            return train_size + len(self.eval_pool)

    def get_pool_bytes(self) -> int:
        """Get current bytes in sample pool (for Mooncake backpressure)."""
        with self._pool_lock:
            return self._pool_bytes

    # ─────────────────────────────────────────────────────────────
    # Dispatch Logic
    # ─────────────────────────────────────────────────────────────

    def set_inference_error(self, msg: str) -> None:
        """Called by the inference manager when a fatal error occurs."""
        self._inference_error = msg

    def try_dispatch_batch(self) -> bool:
        """Try to dispatch one batch to training queues.

        Only dispatches when sample pool has enough samples (>= dispatch_batch_size).
        Dispatches TrainSample objects that MooncakeDataFetcher can consume.
        Subtracts bytes from pool tracking when dispatching.

        Returns:
            True if a batch was dispatched, False if not enough samples.

        Raises:
            RuntimeError: If the inference manager has reported a fatal error.
        """
        if self._inference_error is not None:
            raise RuntimeError(f"Inference engine failed: {self._inference_error}")

        if self.packing_enabled:
            return self._try_dispatch_packed()

        with self._pool_lock:
            pool_size = len(self.sample_pool)
            now = time.time()
            should_log = (now - self._last_dispatch_log_time) >= 2.0
            if pool_size < self.dispatch_batch_size:
                if should_log:
                    self._last_dispatch_log_time = now
                    logger.debug(
                        f"try_dispatch_batch: pool_size={pool_size} < dispatch_batch_size={self.dispatch_batch_size}, not dispatching"
                    )
                return False

            if should_log:
                self._last_dispatch_log_time = now
                logger.debug(
                    f"try_dispatch_batch: pool_size={pool_size} >= dispatch_batch_size={self.dispatch_batch_size}, dispatching batch {self.batch_id}"
                )

            batch_results = []
            for _ in range(self.dispatch_batch_size):
                result = self.sample_pool.popleft()
                sample_bytes = self._sample_bytes.pop(result.mooncake_key, 0)
                self._pool_bytes -= sample_bytes
                batch_results.append(result)

        self._dispatch_to_queues(batch_results, self.train_queues)

        self._training_monitor.record(self.dispatch_batch_size)
        logger.debug(
            f"Dispatched batch {self.batch_id} with {self.dispatch_batch_size} samples "
            f"to {self.dp_size} queues at t={time.time():.3f}"
        )
        self.batch_id += 1
        return True

    def _try_dispatch_packed(self) -> bool:
        """Fixed-rows packing dispatch: emit ONE packed row per DP rank per call.

        Mirrors the non-packing contract (one micro-batch per rank per dispatch),
        so the training loop's ``accumulation_steps`` dispatch iterations produce
        exactly ``accumulation_steps`` rows per rank — every rank runs the SAME
        number of micro-batches, DP-safe under standard FSDP resharding.

        Peeks the pool (does not pop) to run pack_into_rows for ONE row per rank;
        only if it yields all dp_size rows do we pop exactly ``consumed`` leading
        samples and enqueue them grouped by row (row_id stamped, a row's samples
        enqueued consecutively). Otherwise returns False → the loop waits for more
        inference supply (no partial batch reaches a rank → DP stays in lock-step).
        """
        need_rows = self.dp_size  # one row per rank per dispatch
        with self._pool_lock:
            validate_packing_candidates(
                [
                    (sample.mooncake_key, int(sample.tensor_shapes["input_ids"][-1]))
                    for sample in self.sample_pool
                ],
                self.packing_max_seq,
            )

            pool_size = len(self.sample_pool)
            if pool_size < need_rows:
                now = time.time()
                if (now - self._last_dispatch_log_time) >= 2.0:
                    self._last_dispatch_log_time = now
                    logger.debug(
                        f"_try_dispatch_packed: pool_size={pool_size} < need_rows={need_rows}, waiting"
                    )
                return False

            samples = list(self.sample_pool)
            lengths = [int(s.tensor_shapes["input_ids"][-1]) for s in samples]
            packed = pack_into_rows(lengths, self.dp_size, 1, self.packing_max_seq)
            if packed is None:
                now = time.time()
                if (now - self._last_dispatch_log_time) >= 2.0:
                    self._last_dispatch_log_time = now
                    logger.debug(
                        f"_try_dispatch_packed: pool_size={pool_size} cannot fill "
                        f"{need_rows} rows yet, waiting for supply"
                    )
                return False

            rank_rows, consumed = packed
            fill_ratio = packed_rows_fill_ratio(
                rank_rows,
                lengths,
                self.packing_max_seq,
            )
            now = time.time()
            if fill_ratio < self.packing_min_fill_ratio:
                if self._packing_low_fill_since is None:
                    self._packing_low_fill_since = now
                waited = now - self._packing_low_fill_since
                if should_wait_for_packing_fill(
                    fill_ratio=fill_ratio,
                    min_fill_ratio=self.packing_min_fill_ratio,
                    waited_seconds=waited,
                    max_wait_seconds=self.packing_max_wait_seconds,
                ):
                    if (now - self._last_dispatch_log_time) >= 2.0:
                        self._last_dispatch_log_time = now
                        logger.info(
                            "Packing candidate fill %.1f%% < minimum %.1f%%; waiting for more supply (%.1f/%.1fs)",
                            fill_ratio * 100,
                            self.packing_min_fill_ratio * 100,
                            waited,
                            self.packing_max_wait_seconds,
                        )
                    return False
                self._packing_low_fill_flushes += 1
                self._packing_wait_seconds_total += waited
                logger.warning(
                    "Packing low-fill timeout: dispatching %.1f%% full rows after "
                    "%.1fs (minimum %.1f%%) to avoid deadlock.",
                    fill_ratio * 100,
                    waited,
                    self.packing_min_fill_ratio * 100,
                )
            elif self._packing_low_fill_since is not None:
                self._packing_wait_seconds_total += now - self._packing_low_fill_since

            self._packing_low_fill_since = None
            self._packing_last_dispatch_fill_ratio = fill_ratio
            # Pop exactly the consumed leading prefix from the FIFO pool.
            popped = [self.sample_pool.popleft() for _ in range(consumed)]
            for s in popped:
                self._pool_bytes -= self._sample_bytes.pop(s.mooncake_key, 0)

        # Enqueue outside the pool lock: stamp each row a fresh row_id and enqueue its
        # samples consecutively, marking the last so the trainer flushes without look-ahead.
        for dp_rank, rows in enumerate(rank_rows):
            for row in rows:
                row_id = self._row_id_counter
                self._row_id_counter += 1
                for j, idx in enumerate(row):
                    self._enqueue_sample(
                        popped[idx],
                        dp_rank,
                        row_id=row_id,
                        last_in_row=(j == len(row) - 1),
                    )

        self._training_monitor.record(consumed)
        logger.debug(
            f"Dispatched packed batch {self.batch_id}: {consumed} samples -> "
            f"{need_rows} rows (1/rank) across {self.dp_size} ranks, "
            f"fill={self._packing_last_dispatch_fill_ratio:.1%}"
        )
        self.batch_id += 1
        return True

    def _enqueue_sample(
        self,
        result: InferenceOutput,
        dp_rank: int,
        row_id: int | None = None,
        last_in_row: bool = False,
        queues: list[Queue] | None = None,
    ) -> None:
        """Build a TrainSample from an InferenceOutput and push to rank's queue(s).

        Shared by the packing (row_id set) and non-packing (row_id None) paths and
        by the USP fan-out. Single place that constructs TrainSample so the two
        paths never diverge on field population.

        ``queues`` selects the target queue set (defaults to the training queues;
        eval dispatch passes ``self.eval_queues``).
        """
        queues = queues if queues is not None else self.train_queues
        metadata = getattr(result, "metadata", {}) or {}
        last_turn_loss_only = metadata.get("has_thinking")
        sample = TrainSample(
            mooncake_key=result.mooncake_key,
            tensor_shapes=result.tensor_shapes,
            tensor_dtypes=result.tensor_dtypes,
            packed_loss_mask=result.packed_loss_mask,
            last_turn_loss_only=last_turn_loss_only,
            metadata=metadata,
            row_id=row_id,
            last_in_row=last_in_row,
        )
        if self.sp_size > 1 and len(queues) == self.queue_count:
            start = dp_rank * self.sp_size
            for rank in range(start, start + self.sp_size):
                queues[rank].put(sample)
        else:
            queues[dp_rank].put(sample)

    def _partition_results(self, results: list[InferenceOutput]) -> list[list[InferenceOutput]]:
        """Partition InferenceOutputs across DP ranks (non-packing, round-robin)."""
        partitions: list[list[InferenceOutput]] = [[] for _ in range(self.dp_size)]
        for i, result in enumerate(results):
            partitions[i % self.dp_size].append(result)
        return partitions

    def _dispatch_to_queues(
        self,
        batch_results: list[InferenceOutput],
        queues: list[Queue],
    ) -> None:
        """Non-packing dispatch: round-robin partition, one sample per micro-batch."""
        partitioned = self._partition_results(batch_results)
        for dp_rank, results in enumerate(partitioned):
            for result in results:
                self._enqueue_sample(result, dp_rank, row_id=None, queues=queues)

    # ─────────────────────────────────────────────────────────────
    # Eval Pipeline
    # ─────────────────────────────────────────────────────────────

    def _build_eval_entries(self, dataset: list) -> list[InferenceInput]:
        eval_entries: list[InferenceInput] = []
        for sample in dataset:
            if isinstance(sample, dict):
                raw_id = sample.get("data_id") or self._generate_data_id()
                data_id = f"eval_{raw_id}"
                self._eval_data_ids.add(data_id)
                entry = InferenceInput(
                    data_id=data_id,
                    prompt=sample.get("prompt", sample),
                    input_ids=sample.get("input_ids"),
                    packed_loss_mask=sample.get("packed_loss_mask"),
                    formatted_prompt=sample.get("formatted_prompt"),
                    metadata=sample.get("metadata", {}),
                    multimodal_inputs=sample.get("multimodal_inputs"),
                )
            else:
                data_id = f"eval_{self._generate_data_id()}"
                self._eval_data_ids.add(data_id)
                entry = InferenceInput(data_id=data_id, prompt=sample)
            eval_entries.append(entry)
        return eval_entries

    def submit_eval_chunk(self, start: int, end: int) -> int:
        """Submit a slice of the stored eval dataset for inference."""
        assert self._stored_eval_dataset is not None, "No stored eval dataset"
        chunk = self._stored_eval_dataset[start:end]
        if not chunk:
            return 0

        if start == 0:
            self._eval_expected_count = len(self._stored_eval_dataset)
            self._eval_dispatched_samples = 0

        eval_entries = self._build_eval_entries(chunk)

        with self._prompt_lock:
            self.prompt_buffer.extendleft(reversed(eval_entries))
        logger.info(
            f"Eval: submitted chunk [{start}:{end}] ({len(chunk)} samples, total_expected={self._eval_expected_count})"
        )
        return len(chunk)

    def get_eval_pool_size(self) -> int:
        with self._eval_pool_lock:
            return len(self.eval_pool)

    def get_eval_queues(self) -> list[Queue]:
        return self.eval_queues

    def try_dispatch_eval_batch(self) -> bool:
        """Dispatch one eval batch from the pool if enough samples are available."""
        bs = self.eval_dispatch_batch_size
        with self._eval_pool_lock:
            if len(self.eval_pool) < bs:
                return False
            batch_results = [self.eval_pool.popleft() for _ in range(bs)]

        self._dispatch_to_queues(batch_results, self.eval_queues)
        self._eval_dispatched_samples += bs
        logger.debug(
            f"Eval: dispatched batch ({self._eval_dispatched_samples}/{self._eval_expected_count} samples)"
        )
        return True

    def finalize_eval_dispatch(self) -> None:
        """Assert all eval batches were dispatched, then clean up tracking state.

        Raises AssertionError if not all expected samples have arrived or
        undispatched full batches remain in the pool.
        """
        with self._eval_pool_lock:
            arrived = self._eval_dispatched_samples + len(self.eval_pool)
            pool_remaining = len(self.eval_pool)

        assert self._eval_expected_count > 0 and arrived >= self._eval_expected_count, (
            f"finalize_eval_dispatch called before all samples arrived "
            f"(arrived={arrived}, expected={self._eval_expected_count})"
        )
        assert pool_remaining < self.eval_dispatch_batch_size, (
            f"finalize_eval_dispatch called with undispatched full batches "
            f"(pool={pool_remaining}, batch_size={self.eval_dispatch_batch_size})"
        )

        with self._eval_pool_lock:
            if pool_remaining > 0:
                logger.info(
                    f"Eval: dropping {pool_remaining} leftover samples that didn't fill a batch"
                )
                self.eval_pool.clear()

        self._eval_data_ids.clear()
        self._eval_expected_count = 0
        self._eval_dispatched_samples = 0
        logger.info("Eval: dispatch finalized, tracking state cleared")

    # ─────────────────────────────────────────────────────────────
    # Status and Monitoring
    # ─────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get current status of controller."""
        return {
            "prompt_buffer_size": len(self.prompt_buffer),
            "sample_pool_size": len(self.sample_pool),
            "batches_dispatched": self.batch_id,
            "dispatch_batch_size": self.dispatch_batch_size,
            "packing_last_dispatch_fill_ratio": self._packing_last_dispatch_fill_ratio,
            "packing_low_fill_flushes": self._packing_low_fill_flushes,
            "packing_wait_seconds_total": round(self._packing_wait_seconds_total, 3),
        }

    def get_speeds(self) -> dict:
        """Get current throughput speeds in entries/sec."""
        elapsed = time.time() - self._start_time
        return {
            "inference_speed": round(self._inference_monitor.get_speed(), 2),
            "training_speed": round(self._training_monitor.get_speed(), 2),
            "inference_total": self._inference_monitor.get_total_count(),
            "training_total": self._training_monitor.get_total_count(),
            "elapsed_seconds": round(elapsed, 1),
            "avg_inference_speed": round(
                self._inference_monitor.get_total_count() / max(elapsed, 0.001), 2
            ),
            "avg_training_speed": round(
                self._training_monitor.get_total_count() / max(elapsed, 0.001), 2
            ),
        }

    def get_full_status(self) -> dict:
        """Get complete status including speeds."""
        status = self.get_status()
        status.update(self.get_speeds())
        return status

    def shutdown(self) -> None:
        """Signal training workers to stop by sending None to queues."""
        for q in self.train_queues:
            q.put(None)
        logger.info("Controller shutdown: sent stop signals to training queues")
