"""Hash-chained, per-record durable journal for Gate 4c."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from scripts.gate4c_durable_io import append_line_durable
from skillev.contracts import canonical_json
from skillev.experiments import (
    Gate4cCudaMemorySnapshot,
    Gate4cStage,
    Gate4cStageEvent,
    Gate4cStageState,
)


def read_gate4c_journal(path: Path) -> tuple[Gate4cStageEvent, ...]:
    events = tuple(
        Gate4cStageEvent.from_value(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    previous: str | None = None
    active: Gate4cStage | None = None
    for expected_ordinal, event in enumerate(events, start=1):
        if event.ordinal != expected_ordinal or event.previous_event_hash != previous:
            raise ValueError("Gate 4c stage journal hash chain is invalid")
        if event.state is Gate4cStageState.STARTED:
            if active is not None:
                raise ValueError("Gate 4c stage journal contains overlapping stages")
            active = event.stage
        elif active is not event.stage:
            raise ValueError("Gate 4c stage completion has no matching start")
        else:
            active = None
        previous = event.content_hash
    if active is not None:
        raise ValueError("Gate 4c stage journal ends inside an unclosed stage")
    return events


class Gate4cJournal:
    def __init__(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        self._path = path
        self._ordinal = 0
        self._previous_hash: str | None = None
        self._current_stage = Gate4cStage.PROCESS_START

    @property
    def path(self) -> Path:
        return self._path

    @property
    def current_stage(self) -> Gate4cStage:
        return self._current_stage

    @staticmethod
    def _memory() -> Gate4cCudaMemorySnapshot | None:
        if not torch.cuda.is_initialized():
            return None
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Gate 4c telemetry requires exactly one visible CUDA device")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return Gate4cCudaMemorySnapshot(
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            max_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_reserved_bytes=torch.cuda.max_memory_reserved(),
            free_bytes=free_bytes,
            total_bytes=total_bytes,
        )

    def append(self, stage: Gate4cStage, state: Gate4cStageState) -> Gate4cStageEvent:
        self._ordinal += 1
        event = Gate4cStageEvent(
            ordinal=self._ordinal,
            stage=stage,
            state=state,
            monotonic_ns=time.monotonic_ns(),
            previous_event_hash=self._previous_hash,
            cuda_memory=self._memory(),
        )
        append_line_durable(self._path, canonical_json(event.to_value()))
        self._previous_hash = event.content_hash
        self._current_stage = stage
        return event

    @contextmanager
    def stage(self, stage: Gate4cStage) -> Iterator[None]:
        self.append(stage, Gate4cStageState.STARTED)
        try:
            yield
        except BaseException:
            self.append(stage, Gate4cStageState.FAILED)
            raise
        else:
            self.append(stage, Gate4cStageState.COMPLETED)


__all__ = ["Gate4cJournal", "read_gate4c_journal"]
