from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from skillev_private.benchmarks.swebench_official import SWE_HARNESS_PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[2]
WORKER = (
    ROOT / "packages/private-evaluation/src/skillev_private/benchmarks/swebench_official_worker.py"
)
_RUNNER = r"""
import importlib.util
import pathlib
import sys

spec = importlib.util.spec_from_file_location("synthetic_swe_worker", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.TESTBED_ROOT = pathlib.Path(sys.argv[2])
module.WORK_ROOT = pathlib.Path(sys.argv[3])
raise SystemExit(module.main())
"""


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - test arguments are locally constructed.
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )


def _testbed(root: Path) -> tuple[Path, str]:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "worker-test@example.invalid")
    _git(root, "config", "user.name", "Worker Test")
    (root / "calc.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("*.cache\n", encoding="utf-8")
    (root / "binary.dat").write_bytes(b"\x00base-binary-content\xff\n")
    (root / "unchanged-large.bin").write_bytes(b"UNCHANGED-SENTINEL\x00" * 16384)
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_public.py").write_text(
        "from calc import answer\n\ndef test_public():\n    assert answer() == 1\n",
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "synthetic base")
    return root, _git(root, "rev-parse", "HEAD").stdout.strip()


def _add_image_build_commit(root: Path) -> str:
    """Match official images, whose initial HEAD is above the case base."""

    (root / "image-build.txt").write_text("official image build\n", encoding="utf-8")
    _git(root, "add", "image-build.txt")
    _git(root, "commit", "-q", "-m", "official image build")
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def _request(operation: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "deployment_id": "sha256:synthetic-deployment",
        "environment_image_id": "synthetic-official-image",
        "harness_revision": "1" * 40,
        "operation": operation,
        "payload": payload,
        "protocol_version": SWE_HARNESS_PROTOCOL_VERSION,
        "request_id": f"request-{operation}",
    }


def _worker(
    testbed: Path,
    work: Path,
    operation: str,
    payload: dict[str, object],
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(  # noqa: S603 - invokes the fixed worker test runner.
        (sys.executable, "-c", _RUNNER, str(WORKER), str(testbed), str(work)),
        input=json.dumps(_request(operation, payload)).encode(),
        capture_output=True,
        check=False,
    )


def _ok(
    testbed: Path,
    work: Path,
    operation: str,
    payload: dict[str, object],
) -> dict[str, object]:
    completed = _worker(testbed, work, operation, payload)
    assert completed.returncode == 0, completed.stderr.decode()
    response = cast(dict[str, object], json.loads(completed.stdout))
    request = _request(operation, payload)
    assert set(response) == {
        "deployment_id",
        "environment_image_id",
        "harness_revision",
        "protocol_version",
        "request_id",
        "result",
    }
    assert response["deployment_id"] == request["deployment_id"]
    assert response["environment_image_id"] == request["environment_image_id"]
    assert response["harness_revision"] == request["harness_revision"]
    assert response["protocol_version"] == SWE_HARNESS_PROTOCOL_VERSION
    assert response["request_id"] == request["request_id"]
    return cast(dict[str, object], response["result"])


def _initialize(testbed: Path, work: Path, base_commit: str) -> dict[str, object]:
    return _ok(
        testbed,
        work,
        "workspace-initialize",
        {
            "base_commit": base_commit,
            "instance_id": "example__project-1",
            "max_steps": 12,
            "repo": "example/project",
            "test_command": f"{shlex.quote(sys.executable)} -m pytest -q -s",
            "workspace_id": "workspace-1",
        },
    )


def _execute(
    testbed: Path,
    work: Path,
    *,
    kind: str,
    arguments: dict[str, object],
    step_index: int,
) -> dict[str, object]:
    return _ok(
        testbed,
        work,
        "workspace-execute",
        {
            "command": {"arguments": arguments, "kind": kind},
            "step_index": step_index,
            "workspace_id": "workspace-1",
        },
    )


def _patch_for(root: Path, relative: str, content: str) -> str:
    path = root / relative
    existed = path.exists()
    original = path.read_bytes() if existed else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if not existed:
        _git(root, "add", "-N", relative)
    patch = _git(root, "diff", "--binary", "--", relative).stdout
    _git(root, "reset", "-q", "--", relative)
    if original is None:
        path.unlink()
    else:
        path.write_bytes(original)
    return patch


def _verify_payload(
    *,
    base_commit: str,
    candidate_patch: str | None,
    test_patch: str,
    eval_script: str = (
        "set -e\n"
        "cd /testbed\n"
        "git apply /root/test.patch\n"
        f"{shlex.quote(sys.executable)} -m pytest -q -s private_test.py\n"
        "printf 'UNTOUCHED:%s:%s\\n' /testbed_extra /root/test.patch.bak\n"
        "printf 'OFFICIAL-EVAL-END\\n'\n"
    ),
) -> dict[str, object]:
    return {
        "base_commit": base_commit,
        "candidate_patch": candidate_patch,
        "eval_script": eval_script,
        "instance_id": "example__project-1",
        "repo": "example/project",
        "test_patch": test_patch,
        "version": "synthetic-version",
    }


def test_worker_executes_safe_public_tools_with_measured_one_call_usage(
    tmp_path: Path,
) -> None:
    testbed, base_commit = _testbed(tmp_path / "testbed")
    work = tmp_path / "work"
    initialized = _initialize(testbed, work, base_commit)

    assert initialized == {
        "base_commit": base_commit,
        "initialized": True,
        "instance_id": "example__project-1",
        "workspace_id": "workspace-1",
    }
    read = _execute(
        testbed,
        work,
        kind="read",
        arguments={"path": "calc.py"},
        step_index=1,
    )
    written = _execute(
        testbed,
        work,
        kind="write",
        arguments={"content": "public needle\n", "path": "notes/public.txt"},
        step_index=2,
    )
    searched = _execute(
        testbed,
        work,
        kind="search",
        arguments={"path": ".", "query": "public needle"},
        step_index=3,
    )
    tested = _execute(
        testbed,
        work,
        kind="test",
        arguments={"target": "tests/test_public.py"},
        step_index=4,
    )
    submitted = _execute(
        testbed,
        work,
        kind="submit_patch",
        arguments={"patch": "synthetic patch"},
        step_index=5,
    )
    escaped = _execute(
        testbed,
        work,
        kind="read",
        arguments={"path": "../private.txt"},
        step_index=6,
    )

    assert "return 1" in cast(dict[str, object], read["public_observation"])["content"]
    assert cast(dict[str, object], written["public_observation"])["status"] == "completed"
    matches = cast(dict[str, object], searched["public_observation"])["matches"]
    assert matches == [{"line": "public needle", "line_number": 1, "path": "notes/public.txt"}]
    assert cast(dict[str, object], tested["public_observation"])["returncode"] == 0
    assert submitted["terminal"] is True
    assert cast(dict[str, object], escaped["public_observation"])["status"] == "error"
    for result in (read, written, searched, tested, submitted, escaped):
        budget = cast(dict[str, object], result["budget_usage"])
        assert budget["tool_calls"] == 1
        assert cast(int, budget["wall_time_milliseconds"]) >= 1
        assert set(budget) == {
            "agent_turns",
            "input_tokens",
            "model_calls",
            "output_tokens",
            "tool_calls",
            "wall_time_milliseconds",
        }


def test_worker_initialization_resets_official_image_build_commit_to_case_base(
    tmp_path: Path,
) -> None:
    testbed, base_commit = _testbed(tmp_path / "testbed")
    image_head = _add_image_build_commit(testbed)
    assert image_head != base_commit

    work = tmp_path / "work"
    initialized = _initialize(testbed, work, base_commit)

    assert initialized["base_commit"] == base_commit
    assert _git(work, "rev-parse", "HEAD").stdout.strip() == base_commit
    assert not (work / "image-build.txt").exists()
    assert _git(work, "status", "--porcelain").stdout == ""


def test_verify_applies_patches_runs_exact_script_and_removes_private_truth(
    tmp_path: Path,
) -> None:
    testbed, base_commit = _testbed(tmp_path / "testbed")
    work = tmp_path / "work"
    _initialize(testbed, work, base_commit)
    candidate_patch = _patch_for(
        testbed,
        "calc.py",
        "def answer():\n    return 2\n",
    )
    private_canary = "PRIVATE-SWE-TRUTH-CANARY"
    test_patch = _patch_for(
        testbed,
        "private_test.py",
        (
            "from calc import answer\n\n"
            "def test_private():\n"
            f"    # {private_canary}\n"
            "    assert answer() == 2\n"
        ),
    )

    result = _ok(
        testbed,
        work,
        "verify",
        _verify_payload(
            base_commit=base_commit,
            candidate_patch=candidate_patch,
            test_patch=test_patch,
        ),
    )

    assert set(result) == {"instance_id", "candidate_patch_applied", "test_output"}
    assert result["instance_id"] == "example__project-1"
    assert result["candidate_patch_applied"] is True
    assert "1 passed" in cast(str, result["test_output"])
    assert "OFFICIAL-EVAL-END" in cast(str, result["test_output"])
    assert "UNTOUCHED:/testbed_extra:/root/test.patch.bak" in cast(str, result["test_output"])
    assert (work / "calc.py").read_text() == "def answer():\n    return 1\n"
    assert not (work / "private_test.py").exists()
    searched = _execute(
        testbed,
        work,
        kind="search",
        arguments={"path": ".", "query": private_canary},
        step_index=1,
    )
    assert cast(dict[str, object], searched["public_observation"])["matches"] == []


@pytest.mark.parametrize("candidate_patch", [None, "", "not a patch"])
def test_verify_candidate_absence_or_apply_failure_is_an_unresolved_raw_result(
    tmp_path: Path,
    candidate_patch: str | None,
) -> None:
    testbed, base_commit = _testbed(tmp_path / "testbed")
    work = tmp_path / "work"
    _initialize(testbed, work, base_commit)
    test_patch = _patch_for(
        testbed,
        "private_test.py",
        "def test_private():\n    assert True\n",
    )

    result = _ok(
        testbed,
        work,
        "verify",
        _verify_payload(
            base_commit=base_commit,
            candidate_patch=candidate_patch,
            test_patch=test_patch,
            eval_script="printf 'must-not-run\\n'\n",
        ),
    )

    assert result == {
        "candidate_patch_applied": False,
        "instance_id": "example__project-1",
        "test_output": "",
    }


def test_verify_returns_raw_test_patch_failure_but_rejects_identity_mismatch(
    tmp_path: Path,
) -> None:
    testbed, base_commit = _testbed(tmp_path / "testbed")
    work = tmp_path / "work"
    _initialize(testbed, work, base_commit)
    candidate_patch = _patch_for(
        testbed,
        "calc.py",
        "def answer():\n    return 2\n",
    )

    invalid_test = _ok(
        testbed,
        work,
        "verify",
        _verify_payload(
            base_commit=base_commit,
            candidate_patch=candidate_patch,
            test_patch="not a patch",
        ),
    )
    mismatched = _verify_payload(
        base_commit=base_commit,
        candidate_patch=candidate_patch,
        test_patch="not a patch",
    )
    mismatched["instance_id"] = "another-instance"
    wrong_identity = _worker(testbed, work, "verify", mismatched)
    extra_field = _verify_payload(
        base_commit=base_commit,
        candidate_patch=candidate_patch,
        test_patch="not a patch",
    )
    extra_field["fail_to_pass"] = ["private_test.py::test_private"]
    extra_wire = _worker(testbed, work, "verify", extra_field)

    assert invalid_test["candidate_patch_applied"] is True
    assert "error:" in cast(str, invalid_test["test_output"]).lower()
    assert (work / "calc.py").read_text() == "def answer():\n    return 1\n"
    assert wrong_identity.returncode != 0
    assert wrong_identity.stdout == b""
    assert wrong_identity.stderr == b"official SWE harness worker infrastructure failure\n"
    assert extra_wire.returncode != 0
    assert extra_wire.stdout == b""
    assert extra_wire.stderr == b"official SWE harness worker infrastructure failure\n"


def test_worker_is_executable_standalone_stdlib_and_protocol_pinned() -> None:
    source = WORKER.read_text(encoding="utf-8")

    assert source.startswith("#!/usr/bin/env python3\n")
    assert os.access(WORKER, os.X_OK)
    assert f'SWE_HARNESS_PROTOCOL_VERSION = "{SWE_HARNESS_PROTOCOL_VERSION}"' in source
    assert "skillev" not in "\n".join(
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    )
