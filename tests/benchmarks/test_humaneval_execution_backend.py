from __future__ import annotations

import ast
import asyncio
import inspect
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.benchmarks import (
    HUMANEVAL_RESOURCE_LIMITS,
    CodeExecutionInfrastructureError,
    CodeExecutionRequest,
    CodeExecutionStatus,
    IsolatedHumanEvalExecutionBackend,
    humaneval_official,
)


def _request(
    completion: str,
    *,
    test_source: str = "def check(candidate):\n    assert candidate(2) == 3\n",
) -> CodeExecutionRequest:
    return CodeExecutionRequest(
        task_id="synthetic-code-task",
        prompt="def synthetic_transform(value):\n",
        completion=completion,
        test_source=test_source,
        entry_point="synthetic_transform",
    )


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("    return value + 1\n", CodeExecutionStatus.PASSED),
        ("    return (value + 1\n", CodeExecutionStatus.SYNTAX_ERROR),
        ("    raise RuntimeError('candidate failure')\n", CodeExecutionStatus.RUNTIME_ERROR),
        ("    return value - 1\n", CodeExecutionStatus.TEST_FAILURE),
    ],
)
def test_isolated_backend_distinguishes_candidate_outcomes(
    completion: str,
    expected: CodeExecutionStatus,
) -> None:
    result = asyncio.run(IsolatedHumanEvalExecutionBackend().run(_request(completion)))

    assert result.status is expected


def test_complete_module_is_evaluated_without_an_unfinished_prefix() -> None:
    request = replace(
        _request("    return value + 1\n"),
        prompt="",
        completion="def synthetic_transform(value):\n    return value + 1\n",
    )
    result = asyncio.run(IsolatedHumanEvalExecutionBackend().run(request))
    assert result.status is CodeExecutionStatus.PASSED


def test_namespace_start_failure_is_infrastructure_not_a_native_zero(tmp_path: Path) -> None:
    from skillev.evaluation.actor_sandbox import ActorSandbox

    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("actual namespace failure regression requires bubblewrap")
    sandbox = replace(
        ActorSandbox.current(tmp_path / "missing-public-source"), bubblewrap=Path(bwrap)
    )
    backend = IsolatedHumanEvalExecutionBackend(command_prefix=sandbox.command()[:-1])
    with pytest.raises(CodeExecutionInfrastructureError):
        asyncio.run(backend.run(_request("    return value + 1\n")))


def test_candidate_executes_in_another_process_under_fixed_limits() -> None:
    parent_pid = os.getpid()
    limits = HUMANEVAL_RESOURCE_LIMITS
    completion = (
        "    import os, resource\n"
        "    return (os.getpid(), resource.getrlimit(resource.RLIMIT_CPU), "
        "resource.getrlimit(resource.RLIMIT_AS), "
        "resource.getrlimit(resource.RLIMIT_NPROC), "
        "resource.getrlimit(resource.RLIMIT_FSIZE))\n"
    )
    test_source = (
        "def check(candidate):\n"
        "    pid, cpu, memory, processes, file_size = candidate(None)\n"
        f"    assert pid != {parent_pid}\n"
        f"    assert cpu == ({limits.cpu_seconds}, {limits.cpu_seconds})\n"
        f"    assert memory == ({limits.address_space_bytes}, {limits.address_space_bytes})\n"
        f"    assert processes == ({limits.process_count}, {limits.process_count})\n"
        f"    assert file_size == ({limits.file_size_bytes}, {limits.file_size_bytes})\n"
    )

    result = asyncio.run(
        IsolatedHumanEvalExecutionBackend().run(_request(completion, test_source=test_source))
    )

    assert result.status is CodeExecutionStatus.PASSED
    assert limits.open_file_count == 32
    assert limits.wall_timeout_seconds == 3.0


def test_sleeping_candidate_hits_fixed_wall_timeout() -> None:
    result = asyncio.run(
        IsolatedHumanEvalExecutionBackend().run(
            _request("    import time\n    time.sleep(10)\n    return value + 1\n")
        )
    )

    assert result.status is CodeExecutionStatus.TIMEOUT


def test_candidate_process_exit_is_runtime_failure_not_infrastructure() -> None:
    result = asyncio.run(
        IsolatedHumanEvalExecutionBackend().run(_request("    import os\n    os._exit(9)\n"))
    )

    assert result.status is CodeExecutionStatus.RUNTIME_ERROR


def test_invalid_private_test_is_infrastructure_failure_without_private_text() -> None:
    private_marker = "SYNTHETIC-PRIVATE-TEST-MARKER"
    request = _request(
        "    return value + 1\n",
        test_source=f"def check(  # {private_marker}",
    )

    with pytest.raises(CodeExecutionInfrastructureError) as caught:
        asyncio.run(IsolatedHumanEvalExecutionBackend().run(request))

    assert private_marker not in str(caught.value)


def test_process_start_failure_is_infrastructure_not_candidate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_start(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(subprocess, "run", fail_start)

    with pytest.raises(CodeExecutionInfrastructureError):
        asyncio.run(IsolatedHumanEvalExecutionBackend().run(_request("    return value + 1\n")))


def test_backend_has_no_parent_process_exec_or_eval_call() -> None:
    tree = ast.parse(inspect.getsource(humaneval_official))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not {"eval", "exec"} & calls
    assert replace(HUMANEVAL_RESOURCE_LIMITS) == HUMANEVAL_RESOURCE_LIMITS
