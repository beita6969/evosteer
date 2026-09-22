"""Durable private generation journal for answer-separated replay."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

from skillev.evaluation.direct_baseline.runner import RawGenerationRecord


def require_fresh_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("formal attempt output directory must be empty")
    path.mkdir(parents=True, exist_ok=True)


def write_attempt_marker(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


@dataclass(slots=True)
class PrivateGenerationJournal:
    path: Path
    append_existing: bool = False
    _queue: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue, init=False)
    _writer_task: asyncio.Task[None] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError("private journal path must be absolute")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self) -> Self:
        if self._writer_task is not None:
            raise RuntimeError("private generation journal is already open")
        self._writer_task = asyncio.create_task(self._writer())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        await self._queue.put(None)
        if self._writer_task is not None:
            await self._writer_task
        self._writer_task = None

    async def append(self, attempt: RawGenerationRecord) -> None:
        if self._writer_task is None:
            raise RuntimeError("private generation journal must be opened as an async context")
        record = {
            "task_id": attempt.task_id,
            "benchmark": attempt.benchmark.value,
            "raw_text": attempt.raw_text,
            "reasoning_text": attempt.reasoning_text,
            "finish_reason": attempt.finish_reason,
            "prompt_tokens": attempt.prompt_tokens,
            "completion_tokens": attempt.completion_tokens,
            "response_model": attempt.response_model,
            "response_id": attempt.response_id,
            "service_instance_id": attempt.service_instance_id,
            "panel_index": attempt.panel_index,
            "service_slot": attempt.service_slot,
            "prompt_profile_id": attempt.prompt_profile_id,
            "parser_profile_id": attempt.parser_profile_id,
            "decoding_profile_id": attempt.decoding_profile_id,
            "population_id": attempt.population_id,
            "run_seed": attempt.run_seed,
            "request_attempt": attempt.request_attempt,
            "infrastructure_error": attempt.infrastructure_error,
        }
        await self._queue.put(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    async def _writer(self) -> None:
        mode = "a" if self.append_existing else "x"
        with self.path.open(mode, encoding="utf-8") as stream:
            while (line := await self._queue.get()) is not None:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())


def public_journal_projection(record: dict[str, object]) -> dict[str, object]:
    """Return metadata safe for aggregate/public reporting."""

    private_fields = {"raw_text", "reasoning_text", "parsed_value"}
    return {key: value for key, value in record.items() if key not in private_fields}
