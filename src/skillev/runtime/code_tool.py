"""Offline, resource-bounded Python execution as a local tool backend."""

from __future__ import annotations

import ast
import json
import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from skillev.contracts.canonical import canonical_json, normalize_json

from .sandbox import SandboxPolicy, SandboxWorkspace
from .tools import ToolRequest, ToolResult

SANDBOXED_PYTHON_ACTION: Final = "execute_python"
SANDBOXED_PYTHON_RESULT_SCHEMA: Final = "skillev.sandboxed-python-result.v1"
OUTER_NETWORK_POLICY_ENV: Final = "SKILLEV_SANDBOX_NETWORK"
_OUTER_OFFLINE_VALUE: Final = "disabled"
_RESULT_MARKER: Final = "\x1eSKILLEV_SANDBOX_RESULT_V1 "
_READY_MARKER: Final = "skillev-sandbox-ready"
_ALLOWED_PROGRAM_IMPORTS: Final = frozenset(
    {
        "__future__",
        "bisect",
        "collections",
        "decimal",
        "fractions",
        "functools",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "os",
        "re",
        "socket",
        "statistics",
        "sys",
    }
)


class PythonSandboxStrategy(StrEnum):
    """Explicit isolation mode; modes never silently fall back to one another."""

    BUBBLEWRAP = "bubblewrap"
    OUTER_OFFLINE_CONTAINER = "outer-offline-container"


class SandboxRunnerState(StrEnum):
    READY = "ready"
    RUNNER_NOT_FOUND = "runner-not-found"
    PYTHON_NOT_FOUND = "python-not-found"
    ISOLATION_PROBE_FAILED = "isolation-probe-failed"
    OUTER_OFFLINE_POLICY_MISSING = "outer-offline-policy-missing"


@dataclass(frozen=True, slots=True)
class SandboxRunnerAvailability:
    strategy: PythonSandboxStrategy
    state: SandboxRunnerState

    @property
    def available(self) -> bool:
        return self.state is SandboxRunnerState.READY


class SandboxRunnerUnavailableError(RuntimeError):
    """The selected process isolation mechanism is not available."""

    def __init__(self, availability: SandboxRunnerAvailability) -> None:
        self.availability = availability
        super().__init__(
            f"Python sandbox {availability.strategy.value!r} is unavailable "
            f"({availability.state.value})"
        )


@dataclass(frozen=True, slots=True)
class PythonSandboxLimits:
    """Per-invocation time, memory, file, and byte ceilings."""

    wall_time_seconds: float = 5.0
    cpu_time_seconds: int = 3
    memory_bytes: int = 256 * 1024 * 1024
    max_output_bytes: int = 64 * 1024
    max_source_bytes: int = 64 * 1024
    max_input_bytes: int = 256 * 1024
    max_file_bytes: int = 8 * 1024 * 1024
    max_open_files: int = 64
    scratch_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        numeric_limits = (
            self.wall_time_seconds,
            self.cpu_time_seconds,
            self.memory_bytes,
            self.max_output_bytes,
            self.max_source_bytes,
            self.max_input_bytes,
            self.max_file_bytes,
            self.max_open_files,
            self.scratch_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int | float) or value <= 0
            for value in numeric_limits
        ):
            raise ValueError("Python sandbox limits must be positive numbers")
        if self.max_output_bytes < 1024:
            raise ValueError("Output limit must leave room for the result envelope")
        if self.max_open_files < 16:
            raise ValueError("Open-file limit is too small to start Python")


@dataclass(slots=True)
class SandboxedPythonTool:
    """Execute ``main(payload)`` inside an explicitly offline process.

    ``BUBBLEWRAP`` constructs namespaces itself. ``OUTER_OFFLINE_CONTAINER`` is
    for a launcher that already removed network access and is accepted only
    when the launcher sets the neutral network-policy environment marker.
    """

    strategy: PythonSandboxStrategy = PythonSandboxStrategy.BUBBLEWRAP
    limits: PythonSandboxLimits = field(default_factory=PythonSandboxLimits)
    policy: SandboxPolicy = field(default_factory=SandboxPolicy)
    python_executable: str = "/usr/bin/python3"
    bubblewrap_executable: str = "/usr/bin/bwrap"
    read_only_runtime_roots: tuple[str, ...] = (
        "/usr",
        "/usr/local",
        "/lib",
        "/lib64",
    )
    scratch_root: Path | None = None
    _availability: SandboxRunnerAvailability | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.policy.network_enabled or self.policy.package_install_enabled:
            raise ValueError("Sandboxed Python must use an offline immutable policy")
        if not self.read_only_runtime_roots or any(
            not Path(root).is_absolute() for root in self.read_only_runtime_roots
        ):
            raise ValueError("Read-only runtime roots must be absolute paths")
        if self.scratch_root is not None:
            self.scratch_root.mkdir(parents=True, exist_ok=True)

    def availability(self, *, refresh: bool = False) -> SandboxRunnerAvailability:
        if self._availability is not None and not refresh:
            return self._availability
        if not self._executable_exists(self.python_executable):
            availability = SandboxRunnerAvailability(
                self.strategy,
                SandboxRunnerState.PYTHON_NOT_FOUND,
            )
        elif self.strategy is PythonSandboxStrategy.OUTER_OFFLINE_CONTAINER:
            availability = self._outer_container_availability()
        else:
            availability = self._bubblewrap_availability()
        self._availability = availability
        return availability

    def invoke(self, request: ToolRequest) -> ToolResult:
        if request.action != SANDBOXED_PYTHON_ACTION:
            raise PermissionError("Sandboxed Python received an unsupported action")
        source, input_bytes = self._normalize_request(request)
        source_failure = self._validate_source(source)
        if source_failure is not None:
            return source_failure

        refresh = self.strategy is PythonSandboxStrategy.OUTER_OFFLINE_CONTAINER
        availability = self.availability(refresh=refresh)
        if not availability.available:
            raise SandboxRunnerUnavailableError(availability)

        scratch_parent = None if self.scratch_root is None else str(self.scratch_root)
        with tempfile.TemporaryDirectory(prefix="skillev-python-", dir=scratch_parent) as root:
            workspace_path = Path(root)
            workspace = SandboxWorkspace(
                workspace_path,
                max_file_bytes=max(
                    self.limits.max_source_bytes,
                    self.limits.max_input_bytes,
                    len(_SANDBOX_RUNNER_SOURCE.encode("utf-8")),
                ),
            )
            (workspace_path / "tmp").mkdir(mode=0o700)
            workspace.write_bytes("program.py", source.encode("utf-8"))
            workspace.write_bytes("input.json", input_bytes)
            workspace.write_bytes("runner.py", _SANDBOX_RUNNER_SOURCE.encode("utf-8"))
            return self._run(
                command=self._command(workspace_path),
                cwd=workspace_path,
                environment=self._environment(workspace_path),
            )

    def _normalize_request(
        self,
        request: ToolRequest,
    ) -> tuple[str, bytes]:
        normalized = normalize_json(dict(request.arguments))
        if not isinstance(normalized, dict) or set(normalized) != {"input", "source"}:
            raise ValueError("Sandboxed Python requires source and input arguments")
        source = normalized["source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Sandboxed Python source must be a non-empty string")
        source_bytes = source.encode("utf-8")
        if len(source_bytes) > self.limits.max_source_bytes:
            raise ValueError("Sandboxed Python source exceeds its byte limit")
        payload = normalized["input"]
        input_bytes = canonical_json(payload).encode("utf-8")
        if len(input_bytes) > self.limits.max_input_bytes:
            raise ValueError("Sandboxed Python input exceeds its byte limit")
        return source, input_bytes

    def _run(
        self,
        *,
        command: list[str],
        cwd: Path,
        environment: dict[str, str],
    ) -> ToolResult:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(  # noqa: S603 - argv is constructed by this module
                command,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                start_new_session=True,
            )
            try:
                process.wait(timeout=self.limits.wall_time_seconds)
            except subprocess.TimeoutExpired:
                self._kill_process_group(process)
                return self._failure("timeout")
            stdout.seek(0)
            raw_stdout = stdout.read(self.limits.max_output_bytes + 1)
            stderr.seek(0)
            raw_stderr = stderr.read(self.limits.max_output_bytes + 1)

        if (
            len(raw_stdout) > self.limits.max_output_bytes
            or len(raw_stderr) > self.limits.max_output_bytes
        ):
            return self._failure("output_limit")
        envelope = self._parse_envelope(raw_stdout)
        if envelope is not None:
            if envelope.get("status") == "completed":
                return ToolResult(
                    value={
                        "result": normalize_json(envelope.get("result")),
                        "schema_id": SANDBOXED_PYTHON_RESULT_SCHEMA,
                        "status": "completed",
                        "stderr": self._bounded_text(envelope.get("stderr")),
                        "stdout": self._bounded_text(envelope.get("stdout")),
                    }
                )
            return ToolResult(
                value={
                    "error_type": self._bounded_text(envelope.get("error_type")),
                    "failure_kind": self._bounded_text(envelope.get("failure_kind")),
                    "schema_id": SANDBOXED_PYTHON_RESULT_SCHEMA,
                    "status": "failed",
                    "stderr": self._bounded_text(envelope.get("stderr")),
                    "stdout": self._bounded_text(envelope.get("stdout")),
                },
                completed=False,
            )
        if self.strategy is PythonSandboxStrategy.BUBBLEWRAP and raw_stderr.startswith(b"bwrap:"):
            unavailable = SandboxRunnerAvailability(
                self.strategy,
                SandboxRunnerState.ISOLATION_PROBE_FAILED,
            )
            self._availability = unavailable
            raise SandboxRunnerUnavailableError(unavailable)
        return self._failure(self._exit_failure_kind(process.returncode))

    def _command(self, workspace: Path) -> list[str]:
        runner_arguments = [
            "runner.py",
            "program.py",
            "input.json",
            str(self.limits.cpu_time_seconds),
            str(self.limits.memory_bytes),
            str(self.limits.max_file_bytes),
            str(self.limits.max_open_files),
            str(self.limits.max_output_bytes),
            canonical_json(list(self.read_only_runtime_roots)),
        ]
        if self.strategy is PythonSandboxStrategy.OUTER_OFFLINE_CONTAINER:
            return [self.python_executable, "-I", "-S", *runner_arguments]
        command = self._bubblewrap_prefix()
        for name in ("runner.py", "program.py", "input.json"):
            command.extend(("--ro-bind", str(workspace / name), f"/work/{name}"))
        command.extend(
            (
                "--chdir",
                "/work",
                self.python_executable,
                "-I",
                "-S",
                *runner_arguments,
            )
        )
        return command

    def _bubblewrap_prefix(self) -> list[str]:
        command = [
            self.bubblewrap_executable,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--unshare-user",
            "--disable-userns",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--setenv",
            "CUDA_VISIBLE_DEVICES",
            "",
            "--setenv",
            "NVIDIA_VISIBLE_DEVICES",
            "void",
            "--setenv",
            "HOME",
            "/work",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "PYTHONHASHSEED",
            "0",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "TMPDIR",
            "/tmp",  # noqa: S108 - private tmpfs created below
            "--setenv",
            "TZ",
            "UTC",
        ]
        for name in self.policy.inherited_environment:
            value = os.environ.get(name)
            if value is not None and "\x00" not in value:
                command.extend(("--setenv", name, value))
        for root in self.read_only_runtime_roots:
            if Path(root).exists():
                command.extend(("--ro-bind", root, root))
        command.extend(
            (
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--size",
                str(self.limits.scratch_bytes),
                "--tmpfs",
                "/tmp",  # noqa: S108 - private size-capped tmpfs
                "--dir",
                "/work",
            )
        )
        return command

    def _environment(self, workspace: Path) -> dict[str, str]:
        if self.strategy is PythonSandboxStrategy.BUBBLEWRAP:
            environment = {
                "CUDA_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "void",
                "PATH": "/usr/bin:/bin",
            }
        else:
            environment = {
                "CUDA_VISIBLE_DEVICES": "",
                "HOME": str(workspace),
                "LANG": "C.UTF-8",
                "NVIDIA_VISIBLE_DEVICES": "void",
                OUTER_NETWORK_POLICY_ENV: _OUTER_OFFLINE_VALUE,
                "PATH": "/usr/bin:/bin",
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": str(workspace / "tmp"),
                "TZ": "UTC",
            }
        for name in self.policy.inherited_environment:
            value = os.environ.get(name)
            if value is not None and "\x00" not in value:
                environment[name] = value
        return environment

    def _bubblewrap_availability(self) -> SandboxRunnerAvailability:
        if not self._executable_exists(self.bubblewrap_executable):
            return SandboxRunnerAvailability(
                self.strategy,
                SandboxRunnerState.RUNNER_NOT_FOUND,
            )
        probe = [
            *self._bubblewrap_prefix(),
            "--chdir",
            "/work",
            self.python_executable,
            "-I",
            "-S",
            "-c",
            f"print({_READY_MARKER!r})",
        ]
        try:
            completed = subprocess.run(  # noqa: S603 - fixed isolation probe argv
                probe,
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "NVIDIA_VISIBLE_DEVICES": "void",
                    "PATH": "/usr/bin:/bin",
                },
                timeout=min(self.limits.wall_time_seconds, 5.0),
            )
        except (OSError, subprocess.TimeoutExpired):
            return SandboxRunnerAvailability(
                self.strategy,
                SandboxRunnerState.ISOLATION_PROBE_FAILED,
            )
        state = (
            SandboxRunnerState.READY
            if completed.returncode == 0
            and completed.stdout.strip() == _READY_MARKER.encode("utf-8")
            else SandboxRunnerState.ISOLATION_PROBE_FAILED
        )
        return SandboxRunnerAvailability(self.strategy, state)

    def _outer_container_availability(self) -> SandboxRunnerAvailability:
        state = (
            SandboxRunnerState.READY
            if os.environ.get(OUTER_NETWORK_POLICY_ENV) == _OUTER_OFFLINE_VALUE
            else SandboxRunnerState.OUTER_OFFLINE_POLICY_MISSING
        )
        return SandboxRunnerAvailability(self.strategy, state)

    def _parse_envelope(self, output: bytes) -> dict[str, object] | None:
        marker = _RESULT_MARKER.encode("utf-8")
        marker_index = output.rfind(marker)
        if marker_index < 0:
            return None
        payload = output[marker_index + len(marker) :].strip()
        try:
            parsed = cast(object, json.loads(payload))
            normalized = normalize_json(parsed)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        if (
            not isinstance(normalized, dict)
            or normalized.get("schema_id") != SANDBOXED_PYTHON_RESULT_SCHEMA
        ):
            return None
        return cast(dict[str, object], normalized)

    def _failure(self, kind: str) -> ToolResult:
        return ToolResult(
            value={
                "failure_kind": kind,
                "schema_id": SANDBOXED_PYTHON_RESULT_SCHEMA,
                "status": "failed",
            },
            completed=False,
        )

    def _validate_source(self, source: str) -> ToolResult | None:
        try:
            tree = ast.parse(source, filename="<sandboxed-python>", mode="exec")
        except SyntaxError:
            return self._failure("syntax_error")
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    return self._failure("forbidden_import")
                imported = ((node.module or "").partition(".")[0],)
            if imported and any(name not in _ALLOWED_PROGRAM_IMPORTS for name in imported):
                return self._failure("forbidden_import")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"__import__", "compile", "eval", "exec"}
            ):
                return self._failure("dynamic_code")
        return None

    @staticmethod
    def _bounded_text(value: object) -> str:
        return value if isinstance(value, str) else ""

    @staticmethod
    def _exit_failure_kind(returncode: int) -> str:
        if returncode in {-signal.SIGXCPU, 128 + signal.SIGXCPU}:
            return "cpu_limit"
        if returncode in {-signal.SIGXFSZ, 128 + signal.SIGXFSZ}:
            return "file_limit"
        if returncode < 0 or returncode >= 128:
            return "resource_limit"
        return "program_exit"

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    @staticmethod
    def _executable_exists(executable: str) -> bool:
        if os.path.sep in executable:
            return Path(executable).is_file() and os.access(executable, os.X_OK)
        return shutil.which(executable) is not None


_SANDBOX_RUNNER_SOURCE = r"""from __future__ import annotations

import contextlib
import io
import json
import os
import resource
import sys

SCHEMA_ID = "skillev.sandboxed-python-result.v1"
MARKER = "\x1eSKILLEV_SANDBOX_RESULT_V1 "


class LimitedText(io.TextIOBase):
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.size = 0

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8")
        remaining = max(0, self.limit - self.size)
        if remaining:
            accepted = encoded[:remaining].decode("utf-8", errors="ignore")
            self.parts.append(accepted)
            self.size += len(accepted.encode("utf-8"))
        return len(value)

    def value(self) -> str:
        return "".join(self.parts)


def set_limits() -> int:
    cpu_seconds = int(sys.argv[3])
    memory_bytes = int(sys.argv[4])
    max_file_bytes = int(sys.argv[5])
    max_open_files = int(sys.argv[6])
    max_output_bytes = int(sys.argv[7])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (max_open_files, max_open_files))
    return max_output_bytes


def install_offline_audit_hook(workspace: str, read_only_roots: list[str]) -> None:
    writable_roots = (os.path.realpath(workspace), os.path.realpath(os.environ["TMPDIR"]))
    readable_roots = (*writable_roots, *(os.path.realpath(root) for root in read_only_roots))

    def within_roots(path: object, roots: tuple[str, ...]) -> bool:
        if isinstance(path, int):
            return True
        try:
            resolved = os.path.realpath(os.fspath(path))
        except TypeError:
            return False
        return any(
            resolved == root or resolved.startswith(root + os.sep)
            for root in roots
        )

    def audit(event: str, arguments: tuple[object, ...]) -> None:
        if event == "open" and arguments:
            path = arguments[0]
            mode = arguments[1] if len(arguments) > 1 else "r"
            flags = arguments[2] if len(arguments) > 2 else 0
            writing = (
                isinstance(mode, str) and any(character in mode for character in "wax+")
            ) or (
                isinstance(flags, int)
                and bool(
                    flags
                    & (
                        os.O_WRONLY
                        | os.O_RDWR
                        | os.O_CREAT
                        | os.O_TRUNC
                        | os.O_APPEND
                    )
                )
            )
            if writing and not within_roots(path, writable_roots):
                raise PermissionError("sandbox write outside workspace")
            if not writing and not within_roots(path, readable_roots):
                raise PermissionError("sandbox read outside visible roots")
        if event in {"os.listdir", "os.scandir"} and arguments:
            if not within_roots(arguments[0], readable_roots):
                raise PermissionError("sandbox listing outside visible roots")
        if event.startswith("socket."):
            raise PermissionError("sandbox network is disabled")
        if event in {
            "ctypes.dlopen",
            "os.exec",
            "os.fork",
            "os.forkpty",
            "os.posix_spawn",
            "os.spawn",
            "os.system",
            "subprocess.Popen",
        }:
            raise PermissionError("sandbox process expansion is disabled")

    sys.addaudithook(audit)


def emit(envelope: dict[str, object]) -> None:
    payload = json.dumps(
        envelope,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    print(MARKER + payload, flush=True)


def main() -> None:
    max_output_bytes = set_limits()
    log_limit = max(128, max_output_bytes // 8)
    stdout = LimitedText(log_limit)
    stderr = LimitedText(log_limit)
    try:
        with open(sys.argv[1], encoding="utf-8") as stream:
            source = stream.read()
        with open(sys.argv[2], encoding="utf-8") as stream:
            payload = json.load(stream)
        read_only_roots = json.loads(sys.argv[8])
        if not isinstance(read_only_roots, list) or not all(
            isinstance(root, str) for root in read_only_roots
        ):
            raise TypeError("sandbox read roots must be a string list")
        install_offline_audit_hook(os.getcwd(), read_only_roots)
        namespace: dict[str, object] = {"__name__": "__skillev_sandbox_program__"}
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(source, "<sandboxed-python>", "exec"), namespace)
            function = namespace.get("main")
            if not callable(function):
                raise TypeError("source must define callable main(payload)")
            result = function(payload)
        result_payload = json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(result_payload.encode("utf-8")) > max_output_bytes // 2:
            emit(
                {
                    "failure_kind": "output_limit",
                    "schema_id": SCHEMA_ID,
                    "status": "failed",
                    "stderr": stderr.value(),
                    "stdout": stdout.value(),
                }
            )
            return
        emit(
            {
                "result": json.loads(result_payload),
                "schema_id": SCHEMA_ID,
                "status": "completed",
                "stderr": stderr.value(),
                "stdout": stdout.value(),
            }
        )
    except BaseException as error:
        emit(
            {
                "error_type": type(error).__name__,
                "failure_kind": (
                    "memory_limit" if isinstance(error, MemoryError) else "program_error"
                ),
                "schema_id": SCHEMA_ID,
                "status": "failed",
                "stderr": stderr.value(),
                "stdout": stdout.value(),
            }
        )


if __name__ == "__main__":
    main()
"""
