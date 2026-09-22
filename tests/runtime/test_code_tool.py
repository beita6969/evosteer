from __future__ import annotations

import sys
from pathlib import Path

import pytest

from skillev.runtime.code_tool import (
    OUTER_NETWORK_POLICY_ENV,
    SANDBOXED_PYTHON_ACTION,
    PythonSandboxLimits,
    PythonSandboxStrategy,
    SandboxedPythonTool,
)
from skillev.runtime.sandbox import SandboxViolationError, SandboxWorkspace
from skillev.runtime.tools import ToolRequest


def _tool(
    tmp_path: Path,
    *,
    wall_time_seconds: float = 2.0,
    max_source_bytes: int = 4096,
    max_input_bytes: int = 4096,
    max_output_bytes: int = 8192,
) -> SandboxedPythonTool:
    return SandboxedPythonTool(
        strategy=PythonSandboxStrategy.OUTER_OFFLINE_CONTAINER,
        limits=PythonSandboxLimits(
            wall_time_seconds=wall_time_seconds,
            cpu_time_seconds=2,
            memory_bytes=512 * 1024 * 1024,
            max_output_bytes=max_output_bytes,
            max_source_bytes=max_source_bytes,
            max_input_bytes=max_input_bytes,
            max_file_bytes=1024 * 1024,
            max_open_files=32,
            scratch_bytes=2 * 1024 * 1024,
        ),
        python_executable=sys.executable,
        read_only_runtime_roots=(
            str(Path(sys.base_prefix).resolve()),
            "/usr",
            "/usr/local",
            "/lib",
            "/lib64",
        ),
        scratch_root=tmp_path,
    )


def _invoke(tool: SandboxedPythonTool, source: str, payload: object = None):
    return tool.invoke(
        ToolRequest(
            action=SANDBOXED_PYTHON_ACTION,
            arguments={"input": payload, "source": source},
        )
    )


def test_workspace_confines_paths_and_enforces_byte_limits(tmp_path: Path) -> None:
    workspace = SandboxWorkspace(tmp_path / "workspace", max_file_bytes=4)

    workspace.write_bytes("nested/value.bin", b"1234")

    assert workspace.read_bytes("nested/value.bin") == b"1234"
    assert workspace.exists("nested/value.bin")
    with pytest.raises(SandboxViolationError):
        workspace.read_bytes("../outside")
    with pytest.raises(ValueError):
        workspace.write_bytes("too-large.bin", b"12345")


def test_outer_offline_tool_runs_json_function_with_sanitized_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OUTER_NETWORK_POLICY_ENV, "disabled")
    monkeypatch.setenv("LOCAL_TEST_API_KEY", "must-not-enter-child")
    source = """
import os

def main(payload):
    print("local execution")
    return {
        "environment_names": sorted(os.environ),
        "payload": payload,
    }
"""

    outcome = _invoke(_tool(tmp_path), source, {"number": 3})

    assert outcome.completed
    assert isinstance(outcome.value, dict)
    assert outcome.value["status"] == "completed"
    assert str(outcome.value["schema_id"]).startswith("skillev.")
    assert outcome.value["stdout"] == "local execution\n"
    result = outcome.value["result"]
    assert isinstance(result, dict)
    assert result["payload"] == {"number": 3}
    assert "LOCAL_TEST_API_KEY" not in result["environment_names"]


def test_runner_blocks_network_even_in_outer_offline_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OUTER_NETWORK_POLICY_ENV, "disabled")
    source = """
import socket

def main(payload):
    del payload
    socket.socket()
    return "unreachable"
"""

    outcome = _invoke(_tool(tmp_path), source)

    assert not outcome.completed
    assert isinstance(outcome.value, dict)
    assert outcome.value["status"] == "failed"
    assert outcome.value["failure_kind"] in {"program_error", "resource_limit"}


def test_source_and_input_byte_limits_are_checked_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OUTER_NETWORK_POLICY_ENV, "disabled")
    source_limited = _tool(tmp_path, max_source_bytes=24)
    input_limited = _tool(tmp_path, max_input_bytes=8)

    with pytest.raises(ValueError):
        _invoke(source_limited, "def main(payload):\n    return payload\n")
    with pytest.raises(ValueError):
        _invoke(
            input_limited,
            "def main(payload):\n    return payload\n",
            {"value": "long"},
        )


def test_wall_timeout_stops_the_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OUTER_NETWORK_POLICY_ENV, "disabled")
    source = """
def main(payload):
    del payload
    while True:
        pass
"""

    outcome = _invoke(_tool(tmp_path, wall_time_seconds=0.5), source)

    assert not outcome.completed
    assert isinstance(outcome.value, dict)
    assert outcome.value["failure_kind"] == "timeout"
