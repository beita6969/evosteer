"""Bounded CPU preparation, outside the actor's next-turn dependency chain.

Queued immutable histories and retained prepared edges have separate byte limits.
The former applies backpressure; the latter fails the batch if exceeded (waiting
for a terminal artifact while refusing its remaining edges would deadlock).
Shared objects are conservatively counted again across separate queue entries.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import fields, is_dataclass

from skillev.policy import PolicyBackbone
from skillev.policy.interface import ModelInputWindow
from skillev.policy.versions import TrainableVersions
from skillev.rollout import RolloutArtifact
from skillev.rollout.provisional import ProvisionalStep
from skillev.scoring.edge_plan import PreparedEdgePlan

from .provisional_math import PreparedProvisionalStep, prepare_provisional_step


def retained_bytes(value: object) -> int:
    """Account Python storage, not token count or a nominal number of entries."""
    seen: set[int] = set()

    def size(item: object) -> int:
        if id(item) in seen:
            return 0
        seen.add(id(item))
        total = sys.getsizeof(item)
        if is_dataclass(item) and not isinstance(item, type):
            total += sum(size(getattr(item, f.name)) for f in fields(item))
        elif isinstance(item, Mapping):
            total += sum(size(k) + size(v) for k, v in item.items())
        elif isinstance(item, tuple | list):
            total += sum(size(v) for v in item)
        return total

    return size(value)


class ProvisionalPreparation:
    def __init__(
        self,
        backbone: PolicyBackbone,
        versions: TrainableVersions,
        executor: ThreadPoolExecutor,
        publish: Callable[[int, PreparedProvisionalStep, dict[str, object]], None],
        fail: Callable[[BaseException], None],
        *,
        queue_bytes: int,
        prepared_bytes: int,
    ) -> None:
        self.backbone, self.versions, self.executor = backbone, versions, executor
        self.publish, self.fail = publish, fail
        self.queue_limit, self.prepared_limit = queue_bytes, prepared_bytes
        self.pending_bytes = self.ready_bytes = self.peak_pending = self.peak_ready = 0
        self._sizes: dict[int, int] = {}
        self._futures: dict[int, list[Future[PreparedProvisionalStep]]] = {}
        self._inputs: dict[int, list[object]] = {}
        self._first: dict[int, ProvisionalStep] = {}
        self._lock = threading.RLock()
        self._wake = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._error: BaseException | None = None

    async def accept(self, position: int, value: ProvisionalStep) -> None:
        self._loop = asyncio.get_running_loop()
        count = retained_bytes(value)
        if count > self.queue_limit:
            raise MemoryError("one provisional history exceeds the declared CPU queue budget")
        while True:
            with self._lock:
                if self._error is not None:
                    raise RuntimeError("provisional preparation aborted") from self._error
                self._wake.clear()
                if self.pending_bytes + count <= self.queue_limit:
                    previous = self._inputs.setdefault(position, [])
                    if (
                        value.step.index != len(previous) + 1
                        or tuple(previous) != value.previous_steps
                    ):
                        raise ValueError("provisional history is missing, repeated or changed")
                    first = self._first.setdefault(position, value)
                    if any(
                        getattr(first, k) != getattr(value, k)
                        for k in (
                            "trajectory_id",
                            "task_id",
                            "policy",
                            "library_version",
                            "query",
                            "initial_text",
                            "input_window",
                        )
                    ):
                        raise ValueError("provisional history changed its fixed identity")
                    previous.append(value.step)
                    self.pending_bytes += count
                    self.peak_pending = max(self.peak_pending, self.pending_bytes)
                    chain = self._futures.setdefault(position, [])
                    predecessor = chain[-1] if chain else None
                    chain.append(
                        self.executor.submit(
                            self._prepare, position, value, count, predecessor, time.perf_counter()
                        )
                    )
                    break
            await self._wake.wait()

    def _prepare(
        self,
        position: int,
        value: ProvisionalStep,
        count: int,
        predecessor: Future[PreparedProvisionalStep] | None,
        queued: float,
    ) -> PreparedProvisionalStep:
        try:
            if predecessor is not None:
                predecessor.result()  # Same-trajectory publication order, not completion order.
            with self._lock:
                if self._error is not None:
                    raise RuntimeError("preparation cancelled") from self._error
            started = time.perf_counter()
            edge = prepare_provisional_step(self.backbone, value, self.versions)
            size = retained_bytes(edge)
            with self._lock:
                if self.ready_bytes + size > self.prepared_limit:
                    raise MemoryError(
                        "retained provisional plans exceed their declared byte budget"
                    )
                self.ready_bytes += size
                self._sizes[position] = self._sizes.get(position, 0) + size
                self.peak_ready = max(self.peak_ready, self.ready_bytes)
            self.publish(
                position,
                edge,
                {
                    "provisional_prepare_queue_seconds": started - queued,
                    "provisional_prepare_seconds": time.perf_counter() - started,
                    "provisional_step_prepared": value.step.index,
                },
            )
            return edge
        finally:
            # Failure propagates unchanged to the Future. Publish cancellation
            # before releasing capacity: a done callback runs too late to stop
            # another actor from entering this failed batch.
            error = sys.exception()
            if error is not None and self.cancel_pending(error):
                self.fail(error)
            with self._lock:
                self.pending_bytes -= count
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._wake.set)

    def plan(self, position: int, artifact: RolloutArtifact) -> PreparedEdgePlan:
        with self._lock:
            futures = tuple(self._futures[position])
            first = self._first[position]
            steps = tuple(self._inputs[position])
        edges = tuple(f.result() for f in futures)
        if (
            steps != artifact.record.steps
            or artifact.initial_context.text != first.initial_text
            or artifact.record.initial_context.query != first.query
            or artifact.record.tokenizer_id != first.policy.tokenizer_id
            or ModelInputWindow.from_meta(artifact.record.initial_context.meta)
            != first.input_window
        ):
            raise ValueError("terminal artifact differs from the accepted provisional history")
        return PreparedEdgePlan(
            artifact.record,
            first.initial_text,
            first.policy.tokenizer_id,
            tuple(e.forward for e in edges) + tuple(e.backward for e in edges),
        )

    def forget_completed(self, position: int) -> None:
        with self._lock:
            self.ready_bytes -= self._sizes.pop(position, 0)
            self._futures.pop(position, None)
            self._inputs.pop(position, None)
            self._first.pop(position, None)

    def cancel_pending(self, error: BaseException) -> bool:
        with self._lock:
            first_failure = self._error is None
            self._error = self._error or error
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._wake.set)
        return first_failure

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "queued_history_bytes": self.pending_bytes,
                "retained_plan_bytes": self.ready_bytes,
                "peak_queued_history_bytes": self.peak_pending,
                "peak_retained_plan_bytes": self.peak_ready,
                "queue_limit_bytes": self.queue_limit,
                "plan_limit_bytes": self.prepared_limit,
            }
