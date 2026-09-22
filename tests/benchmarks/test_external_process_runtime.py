from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from skillev_private.benchmarks.external_process_runtime import (
    ExternalProcessRuntime,
    ExternalProcessSessionFactory,
    ExternalProcessTerminalEvaluator,
)

from skillev.contracts import stable_hash
from skillev.experiments import Benchmark
from skillev.rollout import (
    NoSubmissionReason,
    NoTerminalSubmission,
    RolloutTask,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
    TerminalEvaluatorError,
)
from skillev.runtime import ActionKind, AttemptFailureCode, AttemptFailureStage, StructuredAction


class _NoSubmissionWorker:
    def __init__(self) -> None:
        self.request_count = 0
        self.close_count = 0

    async def request_async(self, operation: str, payload: object) -> object:
        self.request_count += 1
        raise AssertionError(f"no-submission must not call {operation}: {payload}")

    async def close_async(self) -> None:
        self.close_count += 1


def test_external_process_no_submission_scores_zero_without_worker_evaluation() -> None:
    worker = _NoSubmissionWorker()
    evaluator = ExternalProcessTerminalEvaluator(
        _task(),
        Benchmark.APPWORLD,
        worker,  # type: ignore[arg-type]
    )
    request = TerminalEvaluationRequest(
        trajectory_id="appworld-no-submission",
        task_id=_task().task_id,
        termination=RolloutTermination.HORIZON_EXHAUSTED,
        evaluation_input=NoTerminalSubmission(NoSubmissionReason.HORIZON_EXHAUSTED),
        public_transcript_hash=stable_hash({"public": "no submission"}),
    )

    reward = asyncio.run(evaluator.evaluate(request))

    assert reward.value == 0.0
    assert reward.success is False
    assert worker.request_count == 0
    assert worker.close_count == 1


class _BrokenWorker(_NoSubmissionWorker):
    async def request_async(self, operation: str, payload: object) -> object:
        self.request_count += 1
        raise RuntimeError(f"broken {operation}: {payload}")


def test_external_process_worker_failure_is_typed_terminal_infrastructure() -> None:
    worker = _BrokenWorker()
    evaluator = ExternalProcessTerminalEvaluator(
        _task(),
        Benchmark.APPWORLD,
        worker,  # type: ignore[arg-type]
    )
    request = TerminalEvaluationRequest(
        trajectory_id="appworld-submitted",
        task_id=_task().task_id,
        termination=RolloutTermination.COMPLETED,
        evaluation_input=SubmittedTerminalValue(
            {"benchmark_id": Benchmark.APPWORLD.value, "task_id": _task().task_id}
        ),
        public_transcript_hash=stable_hash({"public": "submitted"}),
    )

    with pytest.raises(TerminalEvaluatorError) as captured:
        asyncio.run(evaluator.evaluate(request))

    assert captured.value.code is AttemptFailureCode.TERMINAL_EVALUATOR_FAILED
    assert captured.value.stage is AttemptFailureStage.TERMINAL_EVALUATION
    assert worker.close_count == 1


def _write_fake_appworld(source_root: Path) -> None:
    package = source_root / "src" / "appworld"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "environment.py").write_text(
        """
class _Tracker:
    pass_count = 2
    num_tests = 2
    success = True


class AppWorld:
    def __init__(self, **kwargs):
        assert kwargs["load_ground_truth"] is True
        assert kwargs["raise_on_failure"] is True

    def execute(self, code):
        return f"public:{code}"

    def evaluate(self, *, suppress_errors):
        assert suppress_errors is True
        return _Tracker()

    def close(self):
        return None
""".lstrip(),
        encoding="utf-8",
    )


def _task() -> RolloutTask:
    return RolloutTask(
        task_id="appworld-fixture-task",
        environment_id="appworld-fixture-environment",
        task_family="appworld/fixture",
        context_id="appworld/fixture",
        query="Complete the public fixture task.",
        available_tools=("appworld.execute",),
        public_context={
            "benchmark_id": Benchmark.APPWORLD.value,
            "source_id": "fixture-source-id",
        },
    )


def _skillflow_task() -> RolloutTask:
    return RolloutTask(
        task_id="skillflow-fixture-task",
        environment_id="skillflow-fixture-environment",
        task_family="skillflow-bench/fixture",
        context_id="skillflow-bench/fixture",
        query="Complete the public fixture workspace task.",
        available_tools=("skillflow-bench.execute",),
        public_context={
            "benchmark_id": Benchmark.SKILLFLOW_BENCH.value,
            "source_id": "family/task",
        },
    )


def test_appworld_worker_executes_and_evaluates_in_child_interpreter(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    state_root = tmp_path / "state"
    state_root.mkdir()
    _write_fake_appworld(source_root)
    session = ExternalProcessSessionFactory(
        ExternalProcessRuntime(
            benchmark=Benchmark.APPWORLD,
            interpreter_path=Path(sys.executable),
            source_root=source_root,
            state_root=state_root,
            request_timeout_seconds=5.0,
        )
    ).create(_task())

    observation = asyncio.run(
        session.environment.execute(
            StructuredAction(
                kind=ActionKind.TOOL,
                name="execute",
                arguments={"code": "print('public')"},
                resource_id="appworld",
            ),
            step_index=1,
        )
    )
    assert observation.public_value == {"output": "public:print('public')"}

    terminal = asyncio.run(
        session.environment.execute(
            StructuredAction(kind=ActionKind.COMPLETE, name="complete", arguments={}),
            step_index=2,
        )
    )
    assert terminal.terminal is True
    reward = asyncio.run(
        session.evaluator.evaluate(
            TerminalEvaluationRequest(
                trajectory_id="appworld-fixture-trajectory",
                task_id=_task().task_id,
                termination=RolloutTermination.COMPLETED,
                evaluation_input=SubmittedTerminalValue(terminal.terminal_submission),
                public_transcript_hash=stable_hash({"public": "transcript"}),
            )
        )
    )
    assert reward.value == 1.0
    assert reward.success is True
    assert (state_root / ".cache").is_dir()


def test_skillflow_worker_uses_prebuilt_image_and_delays_tests_until_evaluation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    state_root = tmp_path / "state"
    storage_root = tmp_path / "storage"
    task_root = source_root / "family" / "task"
    (task_root / "tests").mkdir(parents=True)
    (task_root / "tests" / "test.sh").write_text("exit 0\n", encoding="utf-8")
    image_root = state_root / "images" / "family" / "task"
    image_root.mkdir(parents=True)
    (image_root / "image-id.txt").write_text("skillev-skillflow:fixture\n", encoding="utf-8")
    (storage_root / "run").mkdir(parents=True)
    (storage_root / "storage.conf").write_text("fixture\n", encoding="utf-8")
    runtime_root = tmp_path / "runtime"
    runtime_path = runtime_root / "usr" / "bin" / "podman"
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_text(
        """#!/bin/sh
set -eu
case "$1" in
  run|cp|rm) exit 0 ;;
  exec)
    case "$*" in
      *"test -f /logs/verifier/reward.txt"*) exit 1 ;;
      *) printf 'public-command-output' ; exit 0 ;;
    esac
    ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    runtime_path.chmod(0o755)
    session = ExternalProcessSessionFactory(
        ExternalProcessRuntime(
            benchmark=Benchmark.SKILLFLOW_BENCH,
            interpreter_path=Path(sys.executable),
            source_root=source_root,
            state_root=state_root,
            request_timeout_seconds=5.0,
            container_runtime_path=runtime_path,
            container_storage_root=storage_root,
            image_prefix="skillev-skillflow:",
        )
    ).create(_skillflow_task())

    observation = asyncio.run(
        session.environment.execute(
            StructuredAction(
                kind=ActionKind.TOOL,
                name="execute",
                arguments={"command": "python3 work.py"},
                resource_id="skillflow-bench",
            ),
            step_index=1,
        )
    )
    assert observation.public_value == {
        "exit_code": 0,
        "stderr": "",
        "stdout": "public-command-output",
    }
    terminal = asyncio.run(
        session.environment.execute(
            StructuredAction(kind=ActionKind.COMPLETE, name="complete", arguments={}),
            step_index=2,
        )
    )
    reward = asyncio.run(
        session.evaluator.evaluate(
            TerminalEvaluationRequest(
                trajectory_id="skillflow-fixture-trajectory",
                task_id=_skillflow_task().task_id,
                termination=RolloutTermination.COMPLETED,
                evaluation_input=SubmittedTerminalValue(terminal.terminal_submission),
                public_transcript_hash=stable_hash({"public": "transcript"}),
            )
        )
    )
    assert reward.value == 1.0
    assert reward.success is True
