"""Fail-closed private subprocess bindings for Protocol 10 official graders."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from skillev.contracts import JsonValue, normalize_json

from .protocol_v10_official import HealthBenchGrade, SpreadsheetBenchGrade

if TYPE_CHECKING:
    from skillev.training import AsyncResourceLimiter

    from .protocol_v10_population import PrivateBenchmarkPopulation
    from .protocol_v10_sessions import ProtocolV10PrivateRecord

_HEALTHBENCH_WORKER = Path(__file__).with_name("protocol_v10_healthbench_worker.py")
_SPREADSHEET_WORKER = Path(__file__).with_name("protocol_v10_spreadsheet_worker.py")


class ProtocolV10WorkerError(RuntimeError):
    """An official grader process failed; never convert this to reward zero."""


class PrivateWorkerTransport(Protocol):
    """Bounded asynchronous byte transport for one private worker request."""

    async def request(
        self,
        *,
        command: tuple[str, ...],
        working_directory: Path,
        encoded: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> bytes: ...


async def _stop_process(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int] | None,
) -> None:
    if process.returncode is not None or (wait_task is not None and wait_task.done()):
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        waiter = wait_task if wait_task is not None else asyncio.create_task(process.wait())
        await asyncio.wait_for(asyncio.shield(waiter), timeout=1.0)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        waiter = wait_task if wait_task is not None else asyncio.create_task(process.wait())
        await asyncio.shield(waiter)


@dataclass(frozen=True, slots=True)
class AsyncSubprocessWorkerTransport:
    """Native asyncio subprocess transport with cancellation-safe cleanup."""

    async def request(
        self,
        *,
        command: tuple[str, ...],
        working_directory: Path,
        encoded: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> bytes:
        process: asyncio.subprocess.Process | None = None
        wait_task: asyncio.Task[int] | None = None
        try:
            async with asyncio.timeout(timeout_seconds):
                environment = os.environ.copy()
                environment.pop("PYTHONPATH", None)
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=working_directory,
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                if process.stdin is None or process.stdout is None:  # pragma: no cover
                    raise ProtocolV10WorkerError("official grader pipes are unavailable")
                wait_task = asyncio.create_task(process.wait())
                try:
                    process.stdin.write(encoded)
                    await process.stdin.drain()
                    process.stdin.close()
                    await process.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                try:
                    response = await process.stdout.readexactly(max_response_bytes + 1)
                except asyncio.IncompleteReadError as error:
                    response = error.partial
                if len(response) > max_response_bytes:
                    raise ProtocolV10WorkerError(
                        "official grader returned an invalid response size"
                    )
                return_code = await asyncio.shield(wait_task)
                if return_code != 0:
                    raise ProtocolV10WorkerError("official grader process returned failure")
                return response
        except asyncio.CancelledError:
            if process is not None:
                await asyncio.shield(_stop_process(process, wait_task))
            raise
        except ProtocolV10WorkerError:
            if process is not None:
                await _stop_process(process, wait_task)
            raise
        except (OSError, TimeoutError) as error:
            if process is not None:
                await _stop_process(process, wait_task)
            raise ProtocolV10WorkerError("official grader process did not complete") from error


InMemoryWorkerResponder = Callable[[bytes], bytes | Awaitable[bytes]]


@dataclass(slots=True)
class InMemoryWorkerTransport:
    """Injectable worker fake that exercises parsing and resource limits without spawning."""

    responder: InMemoryWorkerResponder
    requests: list[bytes] = field(default_factory=list)

    async def request(
        self,
        *,
        command: tuple[str, ...],
        working_directory: Path,
        encoded: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> bytes:
        del command, working_directory
        self.requests.append(encoded)
        try:
            async with asyncio.timeout(timeout_seconds):
                result = self.responder(encoded)
                response = await result if inspect.isawaitable(result) else result
        except TimeoutError as error:
            raise ProtocolV10WorkerError("official grader process did not complete") from error
        if not isinstance(response, bytes):
            raise TypeError("in-memory worker response must be bytes")
        if len(response) > max_response_bytes:
            raise ProtocolV10WorkerError("official grader returned an invalid response size")
        return response


@dataclass(frozen=True, slots=True)
class PrivateJSONWorker:
    """One-request/one-response JSON worker with bounded private I/O."""

    command: tuple[str, ...]
    working_directory: Path
    timeout_seconds: float
    max_response_bytes: int = 1024 * 1024
    process_limiter: AsyncResourceLimiter | None = None
    transport: PrivateWorkerTransport | None = None

    def __post_init__(self) -> None:
        if not self.command or any(not part for part in self.command):
            raise ValueError("official worker command must be non-empty")
        if not self.working_directory.is_absolute():
            raise ValueError("official worker directory must be absolute")
        if self.timeout_seconds <= 0 or self.max_response_bytes <= 0:
            raise ValueError("official worker limits must be positive")

    async def request(self, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        normalized = normalize_json(value)
        if not isinstance(normalized, dict):  # pragma: no cover - input annotation
            raise TypeError("official worker request must be an object")
        encoded = json.dumps(normalized, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        if self.process_limiter is None:
            return await self._request_async(encoded)
        async with self.process_limiter.lease():
            return await self._request_async(encoded)

    async def _request_async(self, encoded: bytes) -> dict[str, JsonValue]:
        transport = self.transport or AsyncSubprocessWorkerTransport()
        response = await transport.request(
            command=self.command,
            working_directory=self.working_directory,
            encoded=encoded,
            timeout_seconds=self.timeout_seconds,
            max_response_bytes=self.max_response_bytes,
        )
        if not response:
            raise ProtocolV10WorkerError("official grader returned an invalid response size")
        try:
            value = normalize_json(json.loads(response))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ProtocolV10WorkerError("official grader returned malformed JSON") from error
        if not isinstance(value, dict):
            raise ProtocolV10WorkerError("official grader response must be an object")
        return value


@dataclass(frozen=True, slots=True)
class OfficialHealthBenchProcessGrader:
    """Invoke a pinned simple-evals worker that owns the rubric and grader client."""

    worker: PrivateJSONWorker
    verifier_version: str
    private_cases: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        cases = normalize_json(dict(self.private_cases))
        if not isinstance(cases, dict) or not cases:
            raise ValueError("HealthBench private cases are unavailable")
        object.__setattr__(self, "private_cases", cases)

    async def grade(self, task_id: str, candidate_answer: str) -> HealthBenchGrade:
        try:
            private_case = self.private_cases[task_id]
        except KeyError as error:
            raise ProtocolV10WorkerError("HealthBench task is not privately routed") from error
        result = await self.worker.request(
            {
                "candidate_answer": candidate_answer,
                "operation": "grade",
                "private_case": private_case,
                "task_id": task_id,
            }
        )
        if set(result) != {"official_rubric_score", "triggered_negative_rubric_count"}:
            raise ProtocolV10WorkerError("HealthBench worker response fields differ")
        score = result["official_rubric_score"]
        negative = result["triggered_negative_rubric_count"]
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or type(negative) is not int
        ):
            raise ProtocolV10WorkerError("HealthBench worker response types differ")
        return HealthBenchGrade(float(score), negative)


@dataclass(frozen=True, slots=True)
class HealthBenchProcessDeployment:
    """Private binding for the pinned OpenAI simple-evals HealthBench grader."""

    interpreter_path: Path
    official_source_root: Path
    grader_model: str = "gpt-4.1-2025-04-14"
    api_key_environment: str = "OPENAI_API_KEY"
    api_base_url: str | None = None
    request_timeout_seconds: float = 90.0
    worker_timeout_seconds: float = 900.0
    max_parse_attempts: int = 3
    verifier_version: str = "openai-simple-evals-healthbench-gpt-4.1-2025-04-14@1"

    def __post_init__(self) -> None:
        if not self.interpreter_path.is_absolute() or not self.interpreter_path.is_file():
            raise ValueError("HealthBench interpreter must be an absolute file")
        if not self.official_source_root.is_absolute() or not self.official_source_root.is_dir():
            raise ValueError("HealthBench simple-evals root must be an absolute directory")
        if (
            not self.grader_model.strip()
            or not self.api_key_environment.strip()
            or (self.api_base_url is not None and not self.api_base_url.strip())
            or self.request_timeout_seconds <= 0
            or self.worker_timeout_seconds <= self.request_timeout_seconds
            or self.max_parse_attempts <= 0
            or not self.verifier_version.strip()
        ):
            raise ValueError("HealthBench process deployment is invalid")

    def build(
        self,
        population: PrivateBenchmarkPopulation,
        records: tuple[ProtocolV10PrivateRecord, ...],
        *,
        process_limiter: AsyncResourceLimiter | None = None,
    ) -> OfficialHealthBenchProcessGrader:
        by_source = {record.source_id: record.private_payload for record in records}
        source_ids = {item.source_id for item in population.items}
        if set(by_source) != source_ids:
            raise ValueError("HealthBench public and private routes differ")
        command = [
            str(self.interpreter_path),
            str(_HEALTHBENCH_WORKER),
            "--official-source-root",
            str(self.official_source_root),
            "--grader-model",
            self.grader_model,
            "--api-key-environment",
            self.api_key_environment,
            "--request-timeout-seconds",
            str(self.request_timeout_seconds),
            "--max-parse-attempts",
            str(self.max_parse_attempts),
        ]
        if self.api_base_url is not None:
            command.extend(("--api-base-url", self.api_base_url))
        return OfficialHealthBenchProcessGrader(
            worker=PrivateJSONWorker(
                command=tuple(command),
                working_directory=self.official_source_root.parent,
                timeout_seconds=self.worker_timeout_seconds,
                process_limiter=process_limiter,
            ),
            verifier_version=self.verifier_version,
            private_cases=by_source,
        )


@dataclass(frozen=True, slots=True)
class OfficialSpreadsheetBenchProcessOJ:
    """Invoke the official OJ/LibreOffice worker in its private environment."""

    worker: PrivateJSONWorker
    verifier_version: str

    async def grade(
        self,
        task_id: str,
        submitted_workbook: JsonValue,
    ) -> SpreadsheetBenchGrade:
        result = await self.worker.request(
            {
                "operation": "grade",
                "submitted_workbook": submitted_workbook,
                "task_id": task_id,
            }
        )
        if set(result) != {"passed_case_count", "total_case_count"}:
            raise ProtocolV10WorkerError("SpreadsheetBench worker response fields differ")
        passed = result["passed_case_count"]
        total = result["total_case_count"]
        if type(passed) is not int or type(total) is not int:
            raise ProtocolV10WorkerError("SpreadsheetBench worker response types differ")
        return SpreadsheetBenchGrade(passed, total)


@dataclass(frozen=True, slots=True)
class SpreadsheetBenchProcessDeployment:
    """Private path binding for the pinned official SpreadsheetBench OJ."""

    interpreter_path: Path
    official_source_root: Path
    training_archive: Path
    verified_root: Path
    libreoffice_path: Path
    workspace_root: Path
    temporary_root: Path
    timeout_seconds: float = 120.0
    verifier_version: str = "spreadsheetbench-v1-official-oj-libreoffice@2"

    def __post_init__(self) -> None:
        for path, label in (
            (self.interpreter_path, "SpreadsheetBench interpreter"),
            (self.training_archive, "spreadsheet training archive"),
            (self.libreoffice_path, "LibreOffice executable"),
        ):
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"{label} must be an absolute file")
        for path, label in (
            (self.official_source_root, "SpreadsheetBench source root"),
            (self.verified_root, "SpreadsheetBench verified root"),
            (self.workspace_root, "spreadsheet workspace root"),
            (self.temporary_root, "spreadsheet temporary root"),
        ):
            if not path.is_absolute() or not path.is_dir():
                raise ValueError(f"{label} must be an absolute directory")
        if self.timeout_seconds <= 0 or not self.verifier_version.strip():
            raise ValueError("SpreadsheetBench process deployment limits are invalid")

    def build(
        self,
        *,
        process_limiter: AsyncResourceLimiter | None = None,
    ) -> OfficialSpreadsheetBenchProcessOJ:
        return OfficialSpreadsheetBenchProcessOJ(
            worker=PrivateJSONWorker(
                command=(
                    str(self.interpreter_path),
                    str(_SPREADSHEET_WORKER),
                    "--official-source-root",
                    str(self.official_source_root),
                    "--training-archive",
                    str(self.training_archive),
                    "--verified-root",
                    str(self.verified_root),
                    "--libreoffice",
                    str(self.libreoffice_path),
                    "--workspace-root",
                    str(self.workspace_root),
                    "--temporary-root",
                    str(self.temporary_root),
                    "--timeout-seconds",
                    str(self.timeout_seconds),
                ),
                working_directory=self.official_source_root,
                timeout_seconds=self.timeout_seconds + 30.0,
                process_limiter=process_limiter,
            ),
            verifier_version=self.verifier_version,
        )


__all__ = [
    "AsyncSubprocessWorkerTransport",
    "HealthBenchProcessDeployment",
    "InMemoryWorkerTransport",
    "OfficialHealthBenchProcessGrader",
    "OfficialSpreadsheetBenchProcessOJ",
    "PrivateJSONWorker",
    "PrivateWorkerTransport",
    "ProtocolV10WorkerError",
    "SpreadsheetBenchProcessDeployment",
]
