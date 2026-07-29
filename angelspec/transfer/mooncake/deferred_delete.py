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

"""Deferred deletion manager for Mooncake objects with lease TTL."""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from angelspec.utils.logging import logger


@dataclass
class DeleteTask:
    """A deferred deletion task."""

    keys: List[str]  # List of Mooncake keys to delete
    base_key: str  # Original base key for logging
    enqueue_time: float  # When this was queued
    attempts: int = 0  # Number of delete attempts
    last_attempt_time: float = 0.0  # Last attempt timestamp
    max_attempts: int = 3  # Maximum retry attempts


class DeferredDeleteManager:
    """Manages deferred deletions with TTL awareness and retry logic.

    Mooncake objects have a lease TTL during which they cannot be deleted.
    This manager queues deletions and processes them after the TTL expires.
    """

    def __init__(
        self,
        store: Any,
        ttl_seconds: float = 5.0,
        ttl_buffer_seconds: float = 0.5,
        check_interval: float = 1.0,
        max_queue_size: int = 10000,
        retry_interval: float = 2.0,
    ):
        """Initialize the deferred delete manager.

        Args:
            store: Mooncake store instance with remove() method
            ttl_seconds: Mooncake lease TTL duration
            ttl_buffer_seconds: Extra buffer to add to TTL before deletion
            check_interval: How often to check for deletions (seconds)
            max_queue_size: Maximum pending deletions
            retry_interval: Wait time between retries (seconds)
        """
        self.store = store
        self.ttl_seconds = ttl_seconds
        self.ttl_buffer_seconds = ttl_buffer_seconds
        self.check_interval = check_interval
        self.retry_interval = retry_interval

        self.delete_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.retry_queue: List[DeleteTask] = []  # Tasks to retry

        # Statistics
        self.stats = {
            "enqueued": 0,
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "retried": 0,
            "abandoned": 0,  # Exceeded max attempts
        }
        self.stats_lock = threading.Lock()

        # Background thread
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        logger.info(
            "Manager started: ttl=%.1fs, buffer=%.1fs, check_interval=%.1fs",
            ttl_seconds,
            ttl_buffer_seconds,
            check_interval,
        )

    def enqueue_delete(
        self,
        keys: List[str],
        base_key: str,
        max_attempts: int = 3,
    ) -> bool:
        """Queue a deletion task.

        Args:
            keys: List of Mooncake keys to delete
            base_key: Original base key for logging
            max_attempts: Maximum retry attempts

        Returns:
            True if enqueued successfully, False if queue is full
        """
        task = DeleteTask(
            keys=keys,
            base_key=base_key,
            enqueue_time=time.time(),
            max_attempts=max_attempts,
        )

        try:
            self.delete_queue.put_nowait(task)
            with self.stats_lock:
                self.stats["enqueued"] += 1
            logger.debug("Enqueued deletion for %s (%d keys)", base_key, len(keys))
            return True
        except queue.Full:
            logger.error("Queue full! Cannot enqueue deletion for %s", base_key)
            return False

    def _worker_loop(self):
        """Background worker that processes deletions."""
        logger.info("Worker thread started")

        while not self._stop_event.is_set():
            try:
                # Process retry queue first (these are already past TTL)
                self._process_retry_queue()

                # Check for new tasks from main queue
                try:
                    task = self.delete_queue.get(timeout=self.check_interval)
                    self._process_task(task)
                except queue.Empty:
                    pass  # No tasks, continue loop

            except Exception as e:
                logger.error("Worker error: %s", e, exc_info=True)
                time.sleep(1.0)  # Brief pause on error

        logger.info("Worker thread stopped")

    def _process_task(self, task: DeleteTask):
        """Process a single deletion task."""
        current_time = time.time()
        elapsed = current_time - task.enqueue_time
        wait_time = self.ttl_seconds + self.ttl_buffer_seconds

        # Check if TTL has expired
        if elapsed < wait_time:
            # Not ready yet, wait
            remaining = wait_time - elapsed
            logger.debug(
                "Waiting %.1fs for TTL on %s",
                remaining,
                task.base_key,
            )
            time.sleep(remaining)

        # Attempt deletion
        self._attempt_delete(task)

    def _process_retry_queue(self):
        """Process tasks in the retry queue."""
        if not self.retry_queue:
            return

        current_time = time.time()
        still_waiting = []

        for task in self.retry_queue:
            # Check if enough time has passed since last attempt
            if current_time - task.last_attempt_time >= self.retry_interval:
                self._attempt_delete(task)
            else:
                still_waiting.append(task)

        self.retry_queue = still_waiting

    def _attempt_delete(self, task: DeleteTask):
        """Attempt to delete keys for a task."""
        task.attempts += 1
        task.last_attempt_time = time.time()

        with self.stats_lock:
            self.stats["attempted"] += 1
            if task.attempts > 1:
                self.stats["retried"] += 1

        logger.debug(
            "Attempting delete for %s (attempt %d/%d)",
            task.base_key,
            task.attempts,
            task.max_attempts,
        )

        removed_count = 0
        failed_keys = []

        for key in task.keys:
            try:
                result = self.store.remove(key)

                # remove() returns 0 on success, -704 (OBJECT_NOT_FOUND) if
                # the key is already gone — both count as success.
                if result is None or result == 0 or result == -704:
                    removed_count += 1
                    logger.debug("Successfully removed %s", key)
                else:
                    logger.warning(
                        "Remove returned error code %s for %s",
                        result,
                        key,
                    )
                    failed_keys.append((key, f"error_code_{result}"))

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.warning("Exception removing %s: %s", key, error_msg)
                failed_keys.append((key, error_msg))

        # Check results
        if removed_count == len(task.keys):
            # Complete success
            with self.stats_lock:
                self.stats["succeeded"] += 1
            logger.debug(
                "Successfully deleted %s (all %d keys removed, attempt %d)",
                task.base_key,
                len(task.keys),
                task.attempts,
            )
        elif failed_keys:
            # Some or all keys failed
            if task.attempts < task.max_attempts:
                # Retry
                logger.warning(
                    "Partial failure for %s: %d/%d keys failed, will retry (attempt %d/%d)",
                    task.base_key,
                    len(failed_keys),
                    len(task.keys),
                    task.attempts,
                    task.max_attempts,
                )
                # Update task to only retry failed keys
                task.keys = [key for key, _ in failed_keys]
                self.retry_queue.append(task)
            else:
                # Exceeded max attempts
                with self.stats_lock:
                    self.stats["failed"] += 1
                    self.stats["abandoned"] += 1
                logger.error(
                    "Abandoned deletion for %s after %d attempts: %d keys still failed",
                    task.base_key,
                    task.attempts,
                    len(failed_keys),
                )
                # Log failed keys
                for key, error in failed_keys[:5]:  # Show first 5
                    logger.error("Failed key: %s (%s)", key, error)

    def get_stats(self) -> Dict[str, int]:
        """Get deletion statistics."""
        with self.stats_lock:
            return self.stats.copy()

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.delete_queue.qsize() + len(self.retry_queue)

    def stop(self):
        """Stop the worker thread."""
        logger.info("Stopping worker thread...")
        self._stop_event.set()
        self._worker_thread.join(timeout=10.0)

        # Log final statistics
        stats = self.get_stats()
        logger.info(
            "Final stats: enqueued=%d, succeeded=%d, failed=%d, abandoned=%d, queue_size=%d",
            stats["enqueued"],
            stats["succeeded"],
            stats["failed"],
            stats["abandoned"],
            self.get_queue_size(),
        )

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_worker_thread") and self._worker_thread.is_alive():
            self.stop()
