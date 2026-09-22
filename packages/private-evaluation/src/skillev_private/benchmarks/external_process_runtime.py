"""Official interactive workers for AppWorld and SkillFlow-Bench.

Both benchmarks own state that must survive across rollout steps.  Their
dependencies are intentionally kept in a child interpreter: the training
process sees only public observations and the final scalar reward.  AppWorld
uses its official Python environment; SkillFlow-Bench uses a prebuilt image
for the exact official task and injects private tests only after completion.
"""

from __future__ import annotations

import json
import math
import os
import select
import subprocess
from asyncio import sleep
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from skillev.contracts import JsonValue, SuccessRule, TerminalReward, normalize_json
from skillev.experiments import Benchmark
from skillev.rollout import (
    NoTerminalSubmission,
    RolloutTask,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import ActionKind, BudgetVector, EnvironmentObservation, StructuredAction
from skillev.training import RolloutSessionBundle

from .catalog import PrivateSessionFactory
from .terminal_inputs import no_submission_reward, submitted_value

_WORKER = Path(__file__).with_name("external_process_worker.py")
_PROTOCOL = "skillev-external-process-worker@1"
_MAX_MESSAGE_BYTES = 8 * 1024 * 1024


def _absolute_file(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} must be an absolute file")
    return path


def _absolute_directory(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{label} must be an absolute directory")
    return path


def _positive_float(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


@dataclass(frozen=True, slots=True)
class ExternalProcessRuntime:
    """Deployment-only child runtime for one official process benchmark."""

    benchmark: Benchmark
    interpreter_path: Path
    source_root: Path
    state_root: Path
    request_timeout_seconds: float
    container_runtime_path: Path | None = None
    container_storage_root: Path | None = None
    image_prefix: str | None = None

    def __post_init__(self) -> None:
        if self.benchmark not in {Benchmark.APPWORLD, Benchmark.SKILLFLOW_BENCH}:
            raise ValueError("external training runtime benchmark is unsupported")
        object.__setattr__(
            self,
            "interpreter_path",
            _absolute_file(self.interpreter_path, label="external interpreter"),
        )
        object.__setattr__(
            self,
            "source_root",
            _absolute_directory(self.source_root, label="external source root"),
        )
        object.__setattr__(
            self,
            "state_root",
            _absolute_directory(self.state_root, label="external state root"),
        )
        object.__setattr__(
            self,
            "request_timeout_seconds",
            _positive_float(self.request_timeout_seconds, label="external worker timeout"),
        )
        if self.benchmark is Benchmark.APPWORLD:
            if any(
                value is not None
                for value in (
                    self.container_runtime_path,
                    self.container_storage_root,
                    self.image_prefix,
                )
            ):
                raise ValueError("AppWorld runtime cannot carry container fields")
            return
        if self.container_runtime_path is None or self.container_storage_root is None:
            raise ValueError("SkillFlow-Bench requires an explicit container deployment")
        object.__setattr__(
            self,
            "container_runtime_path",
            _absolute_file(self.container_runtime_path, label="container runtime"),
        )
        object.__setattr__(
            self,
            "container_storage_root",
            _absolute_directory(self.container_storage_root, label="container storage root"),
        )
        if type(self.image_prefix) is not str or not self.image_prefix.strip():
            raise ValueError("SkillFlow-Bench image prefix must be non-empty")


@dataclass(slots=True)
class _ExternalWorkerClient:
    runtime: ExternalProcessRuntime
    task: RolloutTask
    _process: subprocess.Popen[bytes] = field(init=False, repr=False)
    _response_fd: int = field(init=False, repr=False)
    _request_id: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _executor: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="skillev-private-environment",
        )
        read_fd, write_fd = os.pipe()
        self._process = subprocess.Popen(  # noqa: S603 - deployment-pinned interpreter
            (
                str(self.runtime.interpreter_path),
                str(_WORKER),
                "--response-fd",
                str(write_fd),
            ),
            cwd=self.runtime.source_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(write_fd,),
        )
        os.close(write_fd)
        self._response_fd = read_fd
        self.request(
            "initialize",
            {
                "benchmark": self.runtime.benchmark.value,
                "container_runtime_path": (
                    None
                    if self.runtime.container_runtime_path is None
                    else self.runtime.container_runtime_path.as_posix()
                ),
                "container_storage_root": (
                    None
                    if self.runtime.container_storage_root is None
                    else self.runtime.container_storage_root.as_posix()
                ),
                "image_prefix": self.runtime.image_prefix,
                "source_root": self.runtime.source_root.as_posix(),
                "state_root": self.runtime.state_root.as_posix(),
                "task": self.task.to_value(),
            },
        )

    def request(self, operation: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if self._closed:
            raise RuntimeError("external worker is closed")
        self._request_id += 1
        wire = json.dumps(
            {
                "operation": operation,
                "payload": normalize_json(payload),
                "protocol": _PROTOCOL,
                "request_id": self._request_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        if len(wire) > _MAX_MESSAGE_BYTES:
            raise ValueError("external worker request is too large")
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("external worker request channel is unavailable")
        stdin.write(wire + b"\n")
        stdin.flush()
        ready, _, _ = select.select(
            (self._response_fd,),
            (),
            (),
            self.runtime.request_timeout_seconds,
        )
        if not ready:
            raise RuntimeError("external worker request timed out")
        response = b""
        while not response.endswith(b"\n"):
            chunk = os.read(self._response_fd, 65536)
            if not chunk:
                raise RuntimeError("external worker exited without a response")
            response += chunk
            if len(response) > _MAX_MESSAGE_BYTES:
                raise RuntimeError("external worker response is too large")
        value = json.loads(response)
        if (
            not isinstance(value, dict)
            or set(value) != {"protocol", "request_id", "result"}
            or value["protocol"] != _PROTOCOL
            or value["request_id"] != self._request_id
            or not isinstance(value["result"], dict)
        ):
            raise RuntimeError("external worker response is incompatible")
        result = normalize_json(value["result"])
        if not isinstance(result, dict):
            raise RuntimeError("external worker result must be an object")
        return result

    async def request_async(
        self,
        operation: str,
        payload: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        future = self._executor.submit(self.request, operation, payload)
        while not future.done():
            await sleep(0.001)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.request("close", {})
        finally:
            self._closed = True
            if self._process.stdin is not None:
                self._process.stdin.close()
            os.close(self._response_fd)
            self._process.wait(timeout=self.runtime.request_timeout_seconds)

    async def close_async(self) -> None:
        if not self._closed:
            future: Future[None] = self._executor.submit(self.close)
            while not future.done():
                await sleep(0.001)
            future.result()
        self._executor.shutdown(wait=True, cancel_futures=False)


@dataclass(slots=True)
class ExternalProcessEnvironment:
    task: RolloutTask
    benchmark: Benchmark
    worker: _ExternalWorkerClient
    _steps: int = field(default=0, init=False, repr=False)

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
        if step_index <= self._steps:
            raise ValueError("external process steps must be strictly increasing")
        self._steps = step_index
        if action.kind is ActionKind.SKILL and action.skill_id is not None:
            return EnvironmentObservation(
                public_value={"status": "skill-invoked"},
                observation_status="success",
                invoked_skill_ids=(action.skill_id,),
                budget_usage=BudgetVector(tool_calls=1),
            )
        if action.kind is ActionKind.COMPLETE and (
            self.benchmark is not Benchmark.APPWORLD or self.task.action_surface is None
        ):
            return EnvironmentObservation(
                public_value={"status": "submitted"},
                observation_status="success",
                terminal_submission={
                    "benchmark_id": self.benchmark.value,
                    "task_id": self.task.task_id,
                },
                terminal=True,
            )
        arguments = self._tool_arguments(action)
        if arguments is None:
            return EnvironmentObservation(
                public_value={"error": "unsupported_external_process_action"},
                observation_status="tool_error",
                budget_usage=BudgetVector(tool_calls=1),
            )
        result = await self.worker.request_async("execute", arguments)
        return EnvironmentObservation(
            public_value=result,
            observation_status="success",
            budget_usage=BudgetVector(tool_calls=1),
        )

    def _tool_arguments(self, action: StructuredAction) -> dict[str, JsonValue] | None:
        if action.kind is not ActionKind.TOOL or not isinstance(action.arguments, dict):
            return None
        if self.benchmark is Benchmark.APPWORLD:
            if (
                action.resource_id != "appworld"
                or action.name != "execute"
                or set(action.arguments) != {"code"}
                or type(action.arguments["code"]) is not str
            ):
                return None
            return {"code": action.arguments["code"]}
        if (
            action.resource_id != "skillflow-bench"
            or action.name != "execute"
            or set(action.arguments) != {"command"}
            or type(action.arguments["command"]) is not str
        ):
            return None
        return {"command": action.arguments["command"]}

    def validate_completion(self, submission: JsonValue) -> bool:
        if self.benchmark is Benchmark.APPWORLD:
            if self.task.action_surface is not None:
                return submission == {"submit": True}
            return submission == {
                "benchmark_id": self.benchmark.value,
                "task_id": self.task.task_id,
            }
        return (
            isinstance(submission, dict)
            and submission.get("benchmark_id") == self.benchmark.value
            and submission.get("task_id") == self.task.task_id
        )


@dataclass(slots=True)
class ExternalProcessTerminalEvaluator:
    task: RolloutTask
    benchmark: Benchmark
    worker: _ExternalWorkerClient
    _evaluated: bool = field(default=False, init=False, repr=False)

    async def evaluate(self, request: TerminalEvaluationRequest) -> TerminalReward:
        if request.task_id != self.task.task_id or self._evaluated:
            raise TerminalEvaluatorError("external evaluator request is incompatible")
        self._evaluated = True
        try:
            if isinstance(request.evaluation_input, NoTerminalSubmission):
                return no_submission_reward(
                    request,
                    native_metric_name=(
                        "appworld-test-pass-rate"
                        if self.benchmark is Benchmark.APPWORLD
                        else "skillflow-bench-verifier-reward"
                    ),
                    native_payload={"benchmark_id": self.benchmark.value},
                    environment_id=self.task.environment_id,
                    verifier_version=f"{self.benchmark.value}-official-worker@1",
                )
            submission = normalize_json(submitted_value(request))
            expected_submission: JsonValue = (
                {"submit": True}
                if self.benchmark is Benchmark.APPWORLD and self.task.action_surface is not None
                else {
                    "benchmark_id": self.benchmark.value,
                    "task_id": self.task.task_id,
                }
            )
            if submission != expected_submission:
                raise RuntimeError("external evaluator submitted value is incompatible")
            result = await self.worker.request_async("evaluate", {})
            success = result.get("success")
            value = result.get("value")
            metric = result.get("metric")
            if (
                type(success) is not bool
                or isinstance(value, bool)
                or not isinstance(value, int | float)
                or type(metric) is not str
            ):
                raise RuntimeError("external evaluator result is incompatible")
            reward_value = float(value)
            if not math.isfinite(reward_value) or not 0.0 <= reward_value <= 1.0:
                raise RuntimeError("external evaluator reward is outside [0,1]")
            return TerminalReward(
                value=reward_value,
                success=success,
                success_rule=SuccessRule.R_EQUALS_ONE,
                success_threshold=None,
                native_metric_name=metric,
                native_payload={"benchmark_id": self.benchmark.value},
                environment_id=self.task.environment_id,
                verifier_version=f"{self.benchmark.value}-official-worker@1",
            )
        except Exception as error:
            raise TerminalEvaluatorError("external official evaluator failed") from error
        finally:
            await self.worker.close_async()


@dataclass(frozen=True, slots=True)
class ExternalProcessSessionFactory(PrivateSessionFactory):
    runtime: ExternalProcessRuntime

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        context = task.public_context
        if (
            not isinstance(context, dict)
            or context.get("benchmark_id") != self.runtime.benchmark.value
        ):
            raise ValueError("external process task has another runtime route")
        worker = _ExternalWorkerClient(self.runtime, task)
        return RolloutSessionBundle(
            environment=ExternalProcessEnvironment(task, self.runtime.benchmark, worker),
            evaluator=ExternalProcessTerminalEvaluator(task, self.runtime.benchmark, worker),
            retrieved_skills=(),
            cleanup=worker.close_async,
        )


__all__ = [
    "ExternalProcessEnvironment",
    "ExternalProcessRuntime",
    "ExternalProcessSessionFactory",
    "ExternalProcessTerminalEvaluator",
]
