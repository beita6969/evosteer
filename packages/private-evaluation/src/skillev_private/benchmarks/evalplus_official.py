"""Sealed EvalPlus pass@1 worker for MBPP+ and HumanEval+.

The materialized public task contains only the model-visible prompt.  This
module loads the corresponding private EvalPlus row, computes the official
reference outputs, and runs the submitted completion in a separate,
resource-limited interpreter.  The child returns only base/plus verdicts; test
inputs, reference outputs, and canonical solutions never cross the evaluator
boundary into a rollout observation or terminal payload.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from skillev.contracts import JsonValue, normalize_json, validate_sha256
from skillev.experiments import Benchmark
from skillev.rollout import RolloutTask

from .external_completion import (
    ExternalCompletionInfrastructureError,
    ExternalCompletionResult,
    ExternalCompletionSessionFactory,
)

EVALPLUS_VERIFIER_VERSION = "evalplus-base-plus-pass-at-1@1"

_SUPPORTED_BENCHMARKS = frozenset({Benchmark.MBPP_PLUS, Benchmark.HUMANEVAL_PLUS})
_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_MAX_RESPONSE_BYTES = 4096
_WALL_TIMEOUT_SECONDS = 90.0


class EvalPlusEvaluationInfrastructureError(ExternalCompletionInfrastructureError):
    """The frozen EvalPlus case or isolated evaluator could not produce a verdict."""


@dataclass(frozen=True, slots=True)
class EvalPlusEvaluationCase:
    """One answer-bearing EvalPlus row keyed by its answer-free task identity."""

    benchmark: Benchmark
    task_id: str
    official_row: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if self.benchmark not in _SUPPORTED_BENCHMARKS:
            raise ValueError("EvalPlus case benchmark is unsupported")
        if type(self.task_id) is not str or not self.task_id.strip() or "\x00" in self.task_id:
            raise ValueError("EvalPlus task_id must be non-empty text without NUL")
        row = normalize_json(self.official_row)
        if not isinstance(row, dict):
            raise TypeError("EvalPlus official_row must be an object")
        expected_fields = (
            {
                "code",
                "prompt",
                "source_file",
                "task_id",
                "test",
                "test_imports",
                "test_list",
            }
            if self.benchmark is Benchmark.MBPP_PLUS
            else {"canonical_solution", "entry_point", "prompt", "task_id", "test"}
        )
        if set(row) != expected_fields:
            raise ValueError("EvalPlus official_row has incompatible evaluator fields")
        text_fields = (
            ("code", "prompt", "source_file")
            if self.benchmark is Benchmark.MBPP_PLUS
            else ("canonical_solution", "entry_point", "prompt", "test")
        )
        for field_name in text_fields:
            value = row[field_name]
            if type(value) is not str or not value.strip() or "\x00" in value:
                raise ValueError(f"EvalPlus {field_name} must be non-empty text without NUL")
        if self.benchmark is Benchmark.MBPP_PLUS:
            if not isinstance(row["test_imports"], list) or any(
                type(item) is not str or not item.strip() for item in row["test_imports"]
            ):
                raise TypeError("EvalPlus test_imports must be a text array")
            if not isinstance(row["test_list"], list) or any(
                type(item) is not str or not item.strip() for item in row["test_list"]
            ):
                raise TypeError("EvalPlus test_list must be a text array")
            test = row["test"]
            if not (
                (type(test) is str and bool(test.strip()))
                or (
                    isinstance(test, list)
                    and bool(test)
                    and all(type(item) is str and bool(item.strip()) for item in test)
                )
            ):
                raise TypeError("EvalPlus test must be non-empty test source")


def _manifest_record(value: object) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != {"official_row", "task_id"}:
        raise ValueError("EvalPlus private manifest record has incompatible fields")
    if type(normalized["task_id"]) is not str:
        raise TypeError("EvalPlus private task_id must be text")
    if not isinstance(normalized["official_row"], dict):
        raise TypeError("EvalPlus private official_row must be an object")
    return cast(dict[str, JsonValue], normalized)


def load_evalplus_cases(
    private_manifest: Path,
    *,
    benchmark: Benchmark,
) -> tuple[EvalPlusEvaluationCase, ...]:
    """Load private official rows without projecting any answer-bearing field."""

    if not private_manifest.is_file():
        raise FileNotFoundError(private_manifest)
    cases: list[EvalPlusEvaluationCase] = []
    with private_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("EvalPlus private manifest contains an empty record")
            record = _manifest_record(json.loads(line))
            cases.append(
                EvalPlusEvaluationCase(
                    benchmark=benchmark,
                    task_id=cast(str, record["task_id"]),
                    official_row=cast(dict[str, JsonValue], record["official_row"]),
                )
            )
    if not cases or len({case.task_id for case in cases}) != len(cases):
        raise ValueError("EvalPlus private cases must be non-empty and task-unique")
    return tuple(cases)


def load_evalplus_session_factory(
    *,
    benchmark: Benchmark,
    tasks: tuple[RolloutTask, ...],
    private_manifest_path: Path,
    expected_sha256: str,
) -> ExternalCompletionSessionFactory:
    """Build an exact production session factory from one frozen private manifest."""

    if benchmark not in _SUPPORTED_BENCHMARKS:
        raise ValueError("EvalPlus factory supports MBPP+ and HumanEval+ only")
    validate_sha256(expected_sha256)
    manifest_wire = private_manifest_path.read_bytes()
    if f"sha256:{hashlib.sha256(manifest_wire).hexdigest()}" != expected_sha256:
        raise ValueError("EvalPlus private manifest differs from its frozen identity")
    if not tasks or any(
        not isinstance(task.public_context, dict)
        or task.public_context.get("benchmark_id") != benchmark.value
        for task in tasks
    ):
        raise ValueError("EvalPlus public tasks differ from the selected benchmark")
    worker = EvalPlusOfficialWorker(
        benchmark=benchmark,
        cases=load_evalplus_cases(private_manifest_path, benchmark=benchmark),
    )
    return ExternalCompletionSessionFactory(tasks=tasks, worker=worker)


_CHILD_RUNNER = r"""
import contextlib
import json
import os
import resource
import signal
import sys

resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
resource.setrlimit(resource.RLIMIT_AS, (4294967296, 4294967296))
resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def emit(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":"), sort_keys=True))
    sys.stdout.flush()


def finish(base, plus, status):
    emit({"base_passed": bool(base), "plus_passed": bool(plus), "status": status})
    raise SystemExit(0)


try:
    payload = json.loads(sys.stdin.buffer.read())
    if set(payload) != {"benchmark", "official_row", "submission"}:
        raise ValueError
    benchmark = payload["benchmark"]
    row = payload["official_row"]
    submission = payload["submission"]
    if benchmark not in {"mbpp-plus", "humaneval-plus"}:
        raise ValueError
    if not isinstance(row, dict) or not isinstance(submission, str):
        raise TypeError
except BaseException:
    emit({"infrastructure_error": True})
    raise SystemExit(0)


class CallTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    del signum, frame
    raise CallTimeout


signal.signal(signal.SIGALRM, timeout_handler)


def guarded_exec(source, namespace):
    signal.setitimer(signal.ITIMER_REAL, 10.0)
    try:
        with open(os.devnull, "w", encoding="utf-8") as null:
            with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                exec(compile(source, "<evalplus>", "exec"), namespace, namespace)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


def source_list(value):
    if isinstance(value, str) and value.strip():
        return [value]
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item.strip() for item in value)
    ):
        return value
    raise TypeError("invalid official test source")


def run_humaneval(completion):
    prompt = row["prompt"]
    entry_point = row["entry_point"]
    test_source = row["test"]
    values = (prompt, completion, entry_point, test_source)
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise TypeError
    namespace = {"__name__": "__evalplus_candidate__"}
    guarded_exec(prompt + completion, namespace)
    candidate = namespace.get(entry_point)
    if not callable(candidate):
        raise TypeError("entry point is absent")
    guarded_exec(test_source, namespace)
    check = namespace.get("check")
    if not callable(check):
        raise TypeError("official check is absent")
    signal.setitimer(signal.ITIMER_REAL, 10.0)
    try:
        with open(os.devnull, "w", encoding="utf-8") as null:
            with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                check(candidate)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    return True


def run_mbpp(code, tests):
    if not isinstance(code, str) or not code.strip():
        raise TypeError
    imports = row["test_imports"]
    if not isinstance(imports, list) or not all(
        isinstance(item, str) and item.strip() for item in imports
    ):
        raise TypeError
    assertions = source_list(tests)
    namespace = {"__name__": "__evalplus_candidate__"}
    guarded_exec("\n".join(imports) + "\n" + code, namespace)
    for assertion in assertions:
        guarded_exec(assertion, namespace)
    return True


if benchmark == "humaneval-plus":
    try:
        run_humaneval(row["canonical_solution"])
    except BaseException:
        emit({"infrastructure_error": True})
        raise SystemExit(0)
    try:
        passed = run_humaneval(submission)
    except BaseException:
        passed = False
    finish(passed, passed, "passed" if passed else "failed")

try:
    run_mbpp(row["code"], row["test_list"])
    run_mbpp(row["code"], row["test"])
except BaseException:
    emit({"infrastructure_error": True})
    raise SystemExit(0)

try:
    base_passed = run_mbpp(submission, row["test_list"])
except BaseException:
    base_passed = False
try:
    plus_passed = base_passed and run_mbpp(submission, row["test"])
except BaseException:
    plus_passed = False
finish(base_passed, plus_passed, "passed" if plus_passed else "failed")
"""


async def _isolated_evalplus_verdict(
    *,
    benchmark: Benchmark,
    row: dict[str, JsonValue],
    submission: str,
) -> tuple[bool, bool, str]:
    payload = json.dumps(
        {
            "benchmark": benchmark.value,
            "official_row": row,
            "submission": submission,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(payload) > _MAX_REQUEST_BYTES:
        raise EvalPlusEvaluationInfrastructureError("EvalPlus case exceeds worker transport")
    with tempfile.TemporaryDirectory(prefix="skillev-evalplus-") as directory:
        try:
            process = await asyncio.create_subprocess_exec(
                # EvalPlus' augmented tests legitimately import evaluator
                # dependencies such as NumPy.  ``-I`` still isolates user
                # environment variables and user site packages; ``-S`` would
                # incorrectly hide the pinned environment's official deps.
                sys.executable,
                "-I",
                "-c",
                _CHILD_RUNNER,
                cwd=directory,
                env={
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "PYTHONHASHSEED": "0",
                },
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise EvalPlusEvaluationInfrastructureError(
                "EvalPlus isolated evaluator could not start"
            ) from error
        communication = asyncio.create_task(process.communicate(payload))
        try:
            stdout, _ = await asyncio.wait_for(
                asyncio.shield(communication),
                timeout=_WALL_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            if process.returncode is None:
                process.kill()
            await communication
            return False, False, "timeout"
        except asyncio.CancelledError:
            await communication
            raise
    return_code = process.returncode
    if return_code is None:
        raise EvalPlusEvaluationInfrastructureError("EvalPlus worker did not terminate")
    if return_code < 0:
        if -return_code in {
            signal.SIGABRT,
            signal.SIGBUS,
            signal.SIGKILL,
            signal.SIGSEGV,
            signal.SIGXCPU,
            signal.SIGXFSZ,
        }:
            return False, False, "resource-limit"
        return False, False, "runtime-error"
    if return_code != 0 or len(stdout) > _MAX_RESPONSE_BYTES:
        raise EvalPlusEvaluationInfrastructureError("EvalPlus worker returned no verdict")
    try:
        value = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvalPlusEvaluationInfrastructureError(
            "EvalPlus worker returned no verdict"
        ) from error
    if value == {"infrastructure_error": True}:
        raise EvalPlusEvaluationInfrastructureError(
            "EvalPlus trusted reference could not be evaluated"
        )
    if not isinstance(value, dict) or set(value) != {"base_passed", "plus_passed", "status"}:
        raise EvalPlusEvaluationInfrastructureError("EvalPlus worker returned no verdict")
    if type(value["base_passed"]) is not bool or type(value["plus_passed"]) is not bool:
        raise EvalPlusEvaluationInfrastructureError("EvalPlus worker returned no verdict")
    if type(value["status"]) is not str or not value["status"]:
        raise EvalPlusEvaluationInfrastructureError("EvalPlus worker returned no verdict")
    return value["base_passed"], value["plus_passed"], value["status"]


@dataclass(frozen=True, slots=True)
class EvalPlusOfficialWorker:
    """Evaluate one completion against both original and EvalPlus test inputs."""

    benchmark: Benchmark
    cases: tuple[EvalPlusEvaluationCase, ...]
    verifier_version: str = EVALPLUS_VERIFIER_VERSION

    def __post_init__(self) -> None:
        if self.benchmark not in _SUPPORTED_BENCHMARKS:
            raise ValueError("EvalPlus worker supports MBPP+ and HumanEval+ only")
        if not self.cases or len({case.task_id for case in self.cases}) != len(self.cases):
            raise ValueError("EvalPlus worker cases must be non-empty and task-unique")
        if type(self.verifier_version) is not str or not self.verifier_version.strip():
            raise ValueError("EvalPlus verifier_version must be non-empty text")

    @property
    def benchmark_id(self) -> str:
        return self.benchmark.value

    async def evaluate(self, *, task_id: str, submission: str) -> ExternalCompletionResult:
        matches = tuple(case for case in self.cases if case.task_id == task_id)
        if len(matches) != 1:
            raise EvalPlusEvaluationInfrastructureError(
                "EvalPlus task is absent from the frozen private manifest"
            )
        base_passed, plus_passed, status = await _isolated_evalplus_verdict(
            benchmark=self.benchmark,
            row=matches[0].official_row,
            submission=submission,
        )
        passed = base_passed and plus_passed
        return ExternalCompletionResult(
            reward_value=float(passed),
            success=passed,
            native_metric_name="pass@1",
            public_metrics={
                "base_passed": base_passed,
                "plus_passed": plus_passed,
                "status": status,
            },
            verifier_version=self.verifier_version,
        )

    def no_submission_result(self, *, task_id: str) -> ExternalCompletionResult:
        matches = tuple(case for case in self.cases if case.task_id == task_id)
        if len(matches) != 1:
            raise EvalPlusEvaluationInfrastructureError(
                "EvalPlus task is absent from the frozen private manifest"
            )
        return ExternalCompletionResult(
            reward_value=0.0,
            success=False,
            native_metric_name="pass@1",
            public_metrics={
                "base_passed": False,
                "plus_passed": False,
                "status": "no-submission",
            },
            verifier_version=self.verifier_version,
        )


__all__ = [
    "EVALPLUS_VERIFIER_VERSION",
    "EvalPlusEvaluationCase",
    "EvalPlusEvaluationInfrastructureError",
    "EvalPlusOfficialWorker",
    "load_evalplus_cases",
    "load_evalplus_session_factory",
]
