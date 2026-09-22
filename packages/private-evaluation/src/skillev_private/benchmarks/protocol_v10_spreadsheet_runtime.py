"""Isolated editable-workbook runtime for Protocol 10 SpreadsheetBench.

The rollout sees one copied input workbook and a small Python editing tool.
Private targets remain owned by the official OJ process and are never mounted
in the editing sandbox.  The concrete sandbox uses bubblewrap so it can run on
the GPU host without requiring a privileged or system Docker daemon.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol

from skillev.contracts import JsonValue
from skillev.rollout import RolloutTask
from skillev.runtime import (
    ActionKind,
    BudgetVector,
    EnvironmentObservation,
    StructuredAction,
)

from .protocol_v10_official import SpreadsheetBenchGrade, SpreadsheetBenchOfficialOJ


@dataclass(frozen=True, slots=True)
class SpreadsheetWorkbookAssets:
    """Private input assets required to create one answer-free workspace."""

    input_workbook: Path

    def __post_init__(self) -> None:
        if not self.input_workbook.is_absolute() or not self.input_workbook.is_file():
            raise ValueError("spreadsheet input workbook must be an absolute file")


class SpreadsheetWorkbookAssetResolver(Protocol):
    def resolve(
        self,
        task: RolloutTask,
        private_payload: dict[str, JsonValue],
    ) -> SpreadsheetWorkbookAssets: ...


@dataclass(slots=True)
class ProtocolV10SpreadsheetAssetResolver:
    """Resolve training ZIP and Verified-400 inputs without exposing targets."""

    training_archive: Path
    verified_root: Path
    extraction_cache_root: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.training_archive.is_absolute() or not self.training_archive.is_file():
            raise ValueError("spreadsheet training archive must be an absolute file")
        for path, label in (
            (self.verified_root, "SpreadsheetBench verified root"),
            (self.extraction_cache_root, "spreadsheet extraction cache"),
        ):
            if not path.is_absolute() or not path.is_dir():
                raise ValueError(f"{label} must be an absolute directory")

    def resolve(
        self,
        task: RolloutTask,
        private_payload: dict[str, JsonValue],
    ) -> SpreadsheetWorkbookAssets:
        del task
        training_route = private_payload.get("spreadsheet_task_route")
        verified_route = private_payload.get("spreadsheet_relative_path")
        if type(training_route) is str and verified_route is None:
            return SpreadsheetWorkbookAssets(self._training_input(training_route))
        if type(verified_route) is str and training_route is None:
            return SpreadsheetWorkbookAssets(self._verified_input(verified_route))
        raise ValueError("spreadsheet private payload has no unique workbook route")

    @staticmethod
    def _relative_route(value: str) -> PurePosixPath:
        route = PurePosixPath(value)
        if (
            not value.strip()
            or route.is_absolute()
            or ".." in route.parts
            or route.as_posix() != value
        ):
            raise ValueError("spreadsheet workbook route is not normalized")
        return route

    def _training_input(self, value: str) -> Path:
        route = self._relative_route(value)
        destination = (self.extraction_cache_root / Path(*route.parts) / "input.xlsx").resolve()
        if not destination.is_relative_to(self.extraction_cache_root.resolve()):
            raise ValueError("spreadsheet training cache escaped its private root")
        with self._lock:
            if destination.is_file():
                return destination
            member = f"{route.as_posix()}/input.xlsx"
            with zipfile.ZipFile(self.training_archive) as archive:
                try:
                    workbook = archive.read(member)
                except KeyError as error:
                    raise ValueError(
                        "spreadsheet training input is absent from its archive"
                    ) from error
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(workbook)
            temporary.replace(destination)
        return destination

    def _verified_input(self, value: str) -> Path:
        route = self._relative_route(value)
        directory = (self.verified_root / Path(*route.parts)).resolve()
        if not directory.is_relative_to(self.verified_root.resolve()) or not directory.is_dir():
            raise ValueError("SpreadsheetBench verified workbook route is absent")
        # Most verified tasks use ``<task>_init.xlsx``; five released tasks
        # instead use the unprefixed ``initial.xlsx`` name.
        candidates = tuple(directory.glob("*_init.xlsx"))
        unprefixed = directory / "initial.xlsx"
        if unprefixed.is_file():
            candidates += (unprefixed,)
        if len(candidates) != 1:
            raise ValueError("SpreadsheetBench route has no unique initial workbook")
        return candidates[0]


@dataclass(frozen=True, slots=True)
class SpreadsheetExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def __post_init__(self) -> None:
        if type(self.exit_code) is not int or type(self.timed_out) is not bool:
            raise TypeError("spreadsheet execution result fields are invalid")


class SpreadsheetExecutionSandbox(Protocol):
    async def execute(self, workspace: Path, code: str) -> SpreadsheetExecutionResult: ...


_PYTHON_WRAPPER = """\
import sys

WORKBOOK_PATH = "/workspace/input.xlsx"
source = sys.stdin.read()
namespace = {"WORKBOOK_PATH": WORKBOOK_PATH, "__name__": "__main__"}
exec(compile(source, "<spreadsheet.execute>", "exec"), namespace, namespace)
"""


@dataclass(frozen=True, slots=True)
class BubblewrapSpreadsheetExecutor:
    """Run model-authored workbook edits in a networkless user namespace."""

    bubblewrap_path: Path
    prlimit_path: Path
    python_environment: Path
    python_package_roots: tuple[Path, ...]
    runtime_readonly_paths: tuple[Path, ...]
    timeout_seconds: float = 60.0
    address_space_mib: int = 4_096
    process_limit: int | None = None
    file_size_mib: int = 64
    output_tail_chars: int = 32_768

    def __post_init__(self) -> None:
        for path, label in (
            (self.bubblewrap_path, "bubblewrap"),
            (self.prlimit_path, "prlimit"),
        ):
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"{label} executable must be an absolute file")
        if not self.python_environment.is_absolute() or not self.python_environment.is_dir():
            raise ValueError("spreadsheet Python environment must be an absolute directory")
        if any(not path.is_absolute() or not path.is_dir() for path in self.python_package_roots):
            raise ValueError("spreadsheet Python package roots must be absolute directories")
        if not self.runtime_readonly_paths or any(
            not path.is_absolute() or not path.exists() for path in self.runtime_readonly_paths
        ):
            raise ValueError("spreadsheet sandbox runtime paths must exist and be absolute")
        if self.timeout_seconds <= 0:
            raise ValueError("spreadsheet sandbox timeout must be positive")
        if (
            min(
                self.address_space_mib,
                self.file_size_mib,
                self.output_tail_chars,
            )
            <= 0
        ):
            raise ValueError("spreadsheet sandbox limits must be positive")
        if self.process_limit is not None and self.process_limit <= 0:
            raise ValueError("spreadsheet sandbox process limit must be positive")

    async def execute(self, workspace: Path, code: str) -> SpreadsheetExecutionResult:
        if not workspace.is_absolute() or not workspace.is_dir():
            raise ValueError("spreadsheet workspace must be an absolute directory")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("spreadsheet code must be non-empty text")
        task = asyncio.create_task(asyncio.to_thread(self._execute_sync, workspace, code))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    def _execute_sync(self, workspace: Path, code: str) -> SpreadsheetExecutionResult:
        command = [
            str(self.prlimit_path),
            f"--as={self.address_space_mib * 1024 * 1024}",
            f"--cpu={max(1, int(self.timeout_seconds))}",
            f"--fsize={self.file_size_mib * 1024 * 1024}",
            "--nofile=256",
        ]
        if self.process_limit is not None:
            command.append(f"--nproc={self.process_limit}")
        command.extend(
            (
                str(self.bubblewrap_path),
                "--unshare-all",
                "--die-with-parent",
                "--new-session",
            )
        )
        for path in self.runtime_readonly_paths:
            command.extend(("--ro-bind", str(path), str(path)))
        command.extend(
            (
                "--ro-bind",
                str(self.python_environment),
                "/opt/skillev-python",
                "--bind",
                str(workspace),
                "/workspace",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",  # noqa: S108 - private tmpfs inside the sandbox namespace
                "--chdir",
                "/workspace",
                "--setenv",
                "HOME",
                "/tmp",  # noqa: S108 - same private sandbox tmpfs
                "--setenv",
                "PYTHONNOUSERSITE",
                "1",
            )
        )
        package_paths = []
        for index, path in enumerate(self.python_package_roots):
            destination = f"/opt/skillev-packages/{index}"
            command.extend(("--ro-bind", str(path), destination))
            package_paths.append(destination)
        if package_paths:
            command.extend(("--setenv", "PYTHONPATH", ":".join(package_paths)))
        command.extend(
            (
                "/opt/skillev-python/bin/python3",
                "-s",
                "-c",
                _PYTHON_WRAPPER,
            )
        )
        try:
            completed = subprocess.run(  # noqa: S603 - deployment-pinned sandbox executables
                tuple(command),
                input=code,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return SpreadsheetExecutionResult(
                exit_code=124,
                stdout=_timeout_output(error.stdout)[-self.output_tail_chars :],
                stderr=_timeout_output(error.stderr)[-self.output_tail_chars :],
                timed_out=True,
            )
        return SpreadsheetExecutionResult(
            exit_code=completed.returncode,
            stdout=completed.stdout[-self.output_tail_chars :],
            stderr=completed.stderr[-self.output_tail_chars :],
        )


def _timeout_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


@dataclass(slots=True)
class IsolatedSpreadsheetWorkspace:
    task: RolloutTask
    directory: Path
    sandbox: SpreadsheetExecutionSandbox
    oj: SpreadsheetBenchOfficialOJ
    private_payload: dict[str, JsonValue]
    _steps: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def verifier_version(self) -> str:
        return self.oj.verifier_version

    @property
    def environment_id(self) -> str:
        return self.task.environment_id

    @property
    def task_family(self) -> str:
        return self.task.task_family

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        if self._closed:
            raise RuntimeError("spreadsheet workspace is closed")
        if step_index <= self._steps:
            raise ValueError("spreadsheet steps must be strictly increasing")
        self._steps = step_index
        if action.kind is ActionKind.SKILL and action.skill_id is not None:
            return EnvironmentObservation(
                public_value={"status": "skill-invoked"},
                observation_status="success",
                invoked_skill_ids=(action.skill_id,),
                budget_usage=BudgetVector(tool_calls=1),
            )
        if (
            action.kind is not ActionKind.TOOL
            or action.resource_id != "spreadsheet"
            or action.name != "execute"
            or not isinstance(action.arguments, dict)
            or set(action.arguments) != {"code"}
            or type(action.arguments["code"]) is not str
        ):
            return EnvironmentObservation(
                public_value={"error": "unsupported_spreadsheet_action"},
                observation_status="tool_error",
                budget_usage=BudgetVector(tool_calls=1),
            )
        result = await self.sandbox.execute(self.directory, action.arguments["code"])
        return EnvironmentObservation(
            public_value={
                "exit_code": result.exit_code,
                "stderr": result.stderr,
                "stdout": result.stdout,
                "timed_out": result.timed_out,
            },
            observation_status="success" if result.exit_code == 0 else "tool_error",
            budget_usage=BudgetVector(tool_calls=1),
        )

    def validate_completion(self, submission: JsonValue) -> bool:
        expected: JsonValue = (
            {"submit": True}
            if self.task.action_surface is not None
            else {"workspace_id": self.task.task_id}
        )
        return submission == expected

    async def grade(
        self,
        task_id: str,
        submitted_workbook: JsonValue,
    ) -> SpreadsheetBenchGrade:
        if task_id != self.task.task_id or not self.validate_completion(submitted_workbook):
            raise ValueError("spreadsheet submission belongs to another workspace")
        workbook = self.directory / "input.xlsx"
        if not workbook.is_file():
            raise RuntimeError("spreadsheet workspace no longer contains its workbook")
        return await self.oj.grade(
            task_id,
            {
                "private_payload": self.private_payload,
                "submitted_workbook_path": str(workbook),
                "workspace_id": self.task.task_id,
            },
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # ShareStore can briefly report an NFS directory as non-empty after
        # its last file was removed. Cleanup must not erase an already-scored
        # episode, so retry the best-effort removal without turning this
        # storage race into a candidate outcome.
        for attempt in range(5):
            shutil.rmtree(self.directory, ignore_errors=True)
            if not self.directory.exists():
                return
            await asyncio.sleep(0.05 * (attempt + 1))


@dataclass(frozen=True, slots=True)
class IsolatedSpreadsheetWorkspaceFactory:
    workspace_root: Path
    assets: SpreadsheetWorkbookAssetResolver
    sandbox: SpreadsheetExecutionSandbox
    oj: SpreadsheetBenchOfficialOJ

    def __post_init__(self) -> None:
        if not self.workspace_root.is_absolute() or not self.workspace_root.is_dir():
            raise ValueError("spreadsheet workspace root must be an absolute directory")
        if not callable(getattr(self.assets, "resolve", None)):
            raise TypeError("spreadsheet workspace requires an asset resolver")
        if not callable(getattr(self.sandbox, "execute", None)):
            raise TypeError("spreadsheet workspace requires an execution sandbox")
        if not callable(getattr(self.oj, "grade", None)):
            raise TypeError("spreadsheet workspace requires the official OJ")

    def create(
        self,
        task: RolloutTask,
        private_payload: dict[str, JsonValue],
    ) -> IsolatedSpreadsheetWorkspace:
        resolved = self.assets.resolve(task, private_payload)
        directory = Path(tempfile.mkdtemp(prefix="spreadsheet-", dir=self.workspace_root)).resolve()
        try:
            shutil.copy2(resolved.input_workbook, directory / "input.xlsx")
        except BaseException:
            shutil.rmtree(directory)
            raise
        return IsolatedSpreadsheetWorkspace(
            task=task,
            directory=directory,
            sandbox=self.sandbox,
            oj=self.oj,
            private_payload=private_payload,
        )


__all__ = [
    "BubblewrapSpreadsheetExecutor",
    "IsolatedSpreadsheetWorkspace",
    "IsolatedSpreadsheetWorkspaceFactory",
    "ProtocolV10SpreadsheetAssetResolver",
    "SpreadsheetExecutionResult",
    "SpreadsheetExecutionSandbox",
    "SpreadsheetWorkbookAssetResolver",
    "SpreadsheetWorkbookAssets",
]
