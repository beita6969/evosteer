"""Fixed-resource, out-of-process HumanEval execution backend.

Only the child interpreter compiles or executes a candidate.  The parent
process receives one content-free status and treats failure to obtain that
status as evaluator infrastructure failure rather than a benchmark miss.
"""

from __future__ import annotations

import json
import platform
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from skillev.evaluation.current_iid.protocol13.evaluator_contracts import (
    CodeExecutorContract,
)

from .code_math import (
    CodeExecutionInfrastructureError,
    CodeExecutionRequest,
    CodeExecutionResult,
    CodeExecutionStatus,
)

_MAX_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_RESULT_BYTES = 4096
_CHILD_STARTED = b"skillev-humaneval-started\n"


@dataclass(frozen=True, slots=True)
class HumanEvalResourceLimits:
    """Pinned limits applied before the isolated interpreter starts."""

    wall_timeout_seconds: float = 3.0
    cpu_seconds: int = 2
    address_space_bytes: int = 512 * 1024 * 1024
    process_count: int = 1
    file_size_bytes: int = 1024 * 1024
    open_file_count: int = 32


HUMANEVAL_RESOURCE_LIMITS = HumanEvalResourceLimits()


def current_humaneval_executor_contract() -> CodeExecutorContract:
    """Resolve the interpreter and fixed limits used by this process."""

    limits = HUMANEVAL_RESOURCE_LIMITS
    return CodeExecutorContract(
        profile_id="humaneval-python-isolated@2",
        benchmark="humaneval",
        python_version=platform.python_version(),
        wall_timeout_seconds=str(limits.wall_timeout_seconds),
        cpu_seconds=limits.cpu_seconds,
        address_space_bytes=limits.address_space_bytes,
        process_count=limits.process_count,
        file_size_bytes=limits.file_size_bytes,
        open_file_count=limits.open_file_count,
        test_profile="openai-humaneval-original-tests@1",
    )


_CHILD_RUNNER = r"""
import contextlib
import json
import os
import resource
import sys

resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def emit(value):
    data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def outcome(status, stage, error=None):
    emit({"status": status, "native_result": status if error is None else
          type(error).__name__ + ": " + str(error)[:1024],
          "exception_type": None if error is None else type(error).__name__, "stage": stage})
    raise SystemExit(0)


try:
    payload = json.loads(sys.stdin.buffer.read())
    if set(payload) != {"completion", "entry_point", "prompt", "test_source"}:
        raise ValueError("invalid child request")
    if not all(isinstance(payload[name], str) for name in payload):
        raise TypeError("invalid child request")
except BaseException:
    emit({"infrastructure_error": True})
    raise SystemExit(0)

candidate_source = payload["prompt"] + payload["completion"]
os.write(2, b"skillev-humaneval-started\n")
try:
    candidate_code = compile(candidate_source, "<candidate>", "exec")
except SyntaxError as error:
    outcome("syntax-error", "compile", error)
except BaseException:
    emit({"infrastructure_error": True})
    raise SystemExit(0)

namespace = {"__name__": "__humaneval_candidate__"}
try:
    with open(os.devnull, "w", encoding="utf-8") as null:
        with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            exec(candidate_code, namespace, namespace)
except MemoryError as error:
    outcome("resource-limit", "run", error)
except BaseException as error:
    outcome("runtime-error", "run", error)

candidate = namespace.get(payload["entry_point"])
if not callable(candidate):
    outcome("runtime-error", "run")

try:
    test_code = compile(payload["test_source"], "<private-tests>", "exec")
    exec(test_code, namespace, namespace)
    check = namespace["check"]
    if not callable(check):
        raise TypeError("private check is not callable")
except BaseException:
    emit({"infrastructure_error": True})
    raise SystemExit(0)

try:
    with open(os.devnull, "w", encoding="utf-8") as null:
        with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            check(candidate)
except AssertionError as error:
    outcome("test-failure", "assertion", error)
except MemoryError as error:
    outcome("resource-limit", "run", error)
except BaseException as error:
    outcome("runtime-error", "run", error)

outcome("passed", "run")
"""


@dataclass(frozen=True, slots=True)
class IsolatedHumanEvalExecutionBackend:
    """Execute one candidate serially in a resource-limited interpreter."""

    # Evaluation may add its filesystem/network namespace. Legacy callers
    # retain the same resource-bounded child and exact native test harness.
    command_prefix: tuple[str, ...] = ()

    async def run(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        if not isinstance(request, CodeExecutionRequest):
            raise TypeError("HumanEval backend requires CodeExecutionRequest")
        return _run_isolated(request, command_prefix=self.command_prefix)


def _run_isolated(
    request: CodeExecutionRequest, *, command_prefix: tuple[str, ...] = ()
) -> CodeExecutionResult:
    payload = json.dumps(
        {
            "completion": request.completion,
            "entry_point": request.entry_point,
            "prompt": request.prompt,
            "test_source": request.test_source,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(payload) > _MAX_REQUEST_BYTES:
        raise CodeExecutionInfrastructureError(
            "isolated code execution request exceeds its fixed transport limit"
        )

    with tempfile.TemporaryDirectory(prefix="skillev-humaneval-") as directory:
        try:
            completed = subprocess.run(  # noqa: S603 - fixed interpreter argv
                [*command_prefix, sys.executable, "-I", "-S", "-c", _CHILD_RUNNER],
                input=payload,
                cwd=directory,
                env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONHASHSEED": "0",
                },
                capture_output=True,
                start_new_session=True,
                timeout=HUMANEVAL_RESOURCE_LIMITS.wall_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            if not (error.stderr or b"").startswith(_CHILD_STARTED):
                raise CodeExecutionInfrastructureError(
                    "isolated interpreter did not start before its deadline"
                ) from error
            return CodeExecutionResult(CodeExecutionStatus.TIMEOUT, "timed out", stage="timeout")
        except (OSError, subprocess.SubprocessError) as error:
            raise CodeExecutionInfrastructureError(
                "isolated code execution process could not start"
            ) from error

    if not completed.stderr.startswith(_CHILD_STARTED):
        raise CodeExecutionInfrastructureError("isolated interpreter did not start the evaluator")
    if completed.returncode < 0:
        return _signal_result(-completed.returncode)
    if completed.returncode != 0:
        return CodeExecutionResult(CodeExecutionStatus.RUNTIME_ERROR)
    stdout = completed.stdout
    if len(stdout) > _MAX_RESULT_BYTES:
        raise CodeExecutionInfrastructureError(
            "isolated code execution returned an invalid verdict"
        )
    try:
        value = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CodeExecutionInfrastructureError(
            "isolated code execution returned an invalid verdict"
        ) from error
    if value == {"infrastructure_error": True}:
        raise CodeExecutionInfrastructureError(
            "isolated code execution could not evaluate trusted tests"
        )
    if not isinstance(value, dict) or set(value) not in (
        {"status"},
        {"status", "native_result", "exception_type", "stage"},
    ):
        raise CodeExecutionInfrastructureError(
            "isolated code execution returned an invalid verdict"
        )
    try:
        status = CodeExecutionStatus(value["status"])
    except (TypeError, ValueError) as error:
        raise CodeExecutionInfrastructureError(
            "isolated code execution returned an invalid verdict"
        ) from error
    if any(
        value.get(key) is not None and not isinstance(value[key], str)
        for key in ("native_result", "exception_type", "stage")
    ):
        raise CodeExecutionInfrastructureError("isolated native diagnostics are malformed")
    return CodeExecutionResult(
        status, value.get("native_result"), value.get("exception_type"), value.get("stage")
    )


def _signal_result(signal_number: int) -> CodeExecutionResult:
    resource_signals = {
        signal.SIGABRT,
        signal.SIGBUS,
        signal.SIGKILL,
        signal.SIGSEGV,
        signal.SIGXCPU,
        signal.SIGXFSZ,
    }
    if signal_number in resource_signals:
        return CodeExecutionResult(CodeExecutionStatus.RESOURCE_LIMIT)
    return CodeExecutionResult(CodeExecutionStatus.RUNTIME_ERROR)


__all__ = [
    "HUMANEVAL_RESOURCE_LIMITS",
    "HumanEvalResourceLimits",
    "IsolatedHumanEvalExecutionBackend",
    "current_humaneval_executor_contract",
]
