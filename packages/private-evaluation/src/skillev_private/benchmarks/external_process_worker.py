"""Child entrypoint for the official AppWorld and SkillFlow-Bench runtimes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

_PROTOCOL = "skillev-external-process-worker@1"


def _object(value: object, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} has an incompatible shape")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _absolute_path(value: object, *, label: str, directory: bool) -> Path:
    path = Path(_text(value, label=label))
    if not path.is_absolute() or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{label} has an invalid path")
    return path


def _source_id(task: dict[str, Any]) -> str:
    context = task.get("public_context")
    if not isinstance(context, dict):
        raise ValueError("external task context is incompatible")
    source_id = _text(context.get("source_id"), label="external task source_id")
    path = PurePosixPath(source_id)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != source_id:
        raise ValueError("external task source_id is not normalized")
    return source_id


class _AppWorldSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        source_root = _absolute_path(payload["source_root"], label="source_root", directory=True)
        state_root = _absolute_path(payload["state_root"], label="state_root", directory=True)
        task = _object(
            payload["task"],
            fields={
                "action_surface",
                "available_tools",
                "budget_profile",
                "context_id",
                "environment_id",
                "model_visible_messages",
                "public_context",
                "query",
                "task_family",
                "task_id",
            },
            label="AppWorld task",
        )
        sys.path.insert(0, str(source_root / "src"))
        os.environ["APPWORLD_ROOT"] = str(state_root)
        cache_root = state_root / ".cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["APPWORLD_CACHE"] = str(cache_root)
        from appworld.environment import AppWorld  # type: ignore[import-not-found]

        self._world = AppWorld(
            task_id=_source_id(task),
            experiment_name=f"skillev-{uuid.uuid4().hex}",
            load_ground_truth=True,
            raise_on_failure=True,
        )

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        code = _text(payload.get("code"), label="AppWorld code")
        return {"output": self._world.execute(code)}

    def evaluate(self) -> dict[str, Any]:
        tracker = self._world.evaluate(suppress_errors=True)
        value = tracker.pass_count / tracker.num_tests if tracker.num_tests else 0.0
        return {
            "metric": "appworld-test-pass-rate",
            "success": tracker.success,
            "value": float(value),
        }

    def close(self) -> None:
        self._world.close()


class _SkillFlowSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        source_root = _absolute_path(payload["source_root"], label="source_root", directory=True)
        state_root = _absolute_path(payload["state_root"], label="state_root", directory=True)
        runtime = _absolute_path(
            payload["container_runtime_path"],
            label="container runtime",
            directory=False,
        )
        storage = _absolute_path(
            payload["container_storage_root"],
            label="container storage",
            directory=True,
        )
        prefix = _text(payload["image_prefix"], label="SkillFlow image prefix")
        task = _object(
            payload["task"],
            fields={
                "action_surface",
                "available_tools",
                "budget_profile",
                "context_id",
                "environment_id",
                "model_visible_messages",
                "public_context",
                "query",
                "task_family",
                "task_id",
            },
            label="SkillFlow task",
        )
        source_id = _source_id(task)
        self._task_root = source_root / PurePosixPath(source_id)
        if not self._task_root.is_dir():
            raise ValueError("SkillFlow task root is absent")
        image_id_file = state_root / "images" / PurePosixPath(source_id) / "image-id.txt"
        if not image_id_file.is_file():
            raise ValueError("SkillFlow task image identity is absent")
        image = _text(image_id_file.read_text(encoding="utf-8"), label="SkillFlow image")
        if not image.startswith(prefix):
            raise ValueError("SkillFlow task image has another frozen prefix")
        self._runtime = runtime
        self._storage = storage
        self._runtime_root = runtime.parent.parent.parent
        self._container = f"skillev-{uuid.uuid4().hex}"
        self._run(
            "run",
            "--detach",
            "--name",
            self._container,
            "--network",
            "none",
            image,
            "sleep",
            "infinity",
        )

    def _run(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["CONTAINERS_STORAGE_CONF"] = str(self._storage / "storage.conf")
        environment["XDG_RUNTIME_DIR"] = str(self._storage / "run")
        environment["PATH"] = f"{self._runtime.parent}:{environment.get('PATH', '')}"
        library_path = self._runtime_root / "usr" / "lib" / "x86_64-linux-gnu"
        environment["LD_LIBRARY_PATH"] = ":".join(
            part
            for part in (
                library_path.as_posix(),
                environment.get("LD_LIBRARY_PATH", ""),
            )
            if part
        )
        environment["CONTAINERS_CONF"] = str(
            self._runtime_root / "usr" / "share" / "containers" / "containers.conf"
        )
        environment["CONTAINERS_REGISTRIES_CONF"] = str(
            self._runtime_root / "etc" / "containers" / "registries.conf"
        )
        completed = subprocess.run(  # noqa: S603 - deployment-pinned Podman CLI
            (str(self._runtime), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env=environment,
        )
        if check and completed.returncode != 0:
            raise RuntimeError("SkillFlow container command failed")
        return completed

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = _text(payload.get("command"), label="SkillFlow command")
        result = self._run("exec", "--workdir", "/root", self._container, "bash", "-lc", command)
        return {
            "exit_code": result.returncode,
            "stderr": result.stderr[-32768:],
            "stdout": result.stdout[-32768:],
        }

    def evaluate(self) -> dict[str, Any]:
        tests = self._task_root / "tests"
        if not tests.is_dir():
            raise ValueError("SkillFlow private tests are absent")
        self._run("exec", self._container, "mkdir", "-p", "/tests", "/logs/verifier")
        self._run("cp", f"{tests.as_posix()}/.", f"{self._container}:/tests")
        completed = self._run(
            "exec",
            "--workdir",
            "/root",
            self._container,
            "bash",
            "/tests/test.sh",
            check=False,
        )
        reward = 1.0 if completed.returncode == 0 else 0.0
        reward_read = self._run(
            "exec",
            self._container,
            "bash",
            "-lc",
            "test -f /logs/verifier/reward.txt && cat /logs/verifier/reward.txt",
            check=False,
        )
        if reward_read.returncode == 0 and reward_read.stdout.strip():
            reward = float(reward_read.stdout.strip())
        return {
            "metric": "skillflow-bench-verifier-reward",
            "success": reward == 1.0,
            "value": reward,
        }

    def close(self) -> None:
        self._run("rm", "--force", self._container, check=False)


def _response(fd: int, request_id: int, result: dict[str, Any]) -> None:
    wire = json.dumps(
        {"protocol": _PROTOCOL, "request_id": request_id, "result": result},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    os.write(fd, wire + b"\n")


def main(response_fd: int) -> None:
    session: _AppWorldSession | _SkillFlowSession | None = None
    for line in sys.stdin.buffer:
        request = _object(
            json.loads(line),
            fields={"operation", "payload", "protocol", "request_id"},
            label="external worker request",
        )
        if request["protocol"] != _PROTOCOL or type(request["request_id"]) is not int:
            raise ValueError("external worker protocol differs")
        operation = _text(request["operation"], label="external worker operation")
        payload = request["payload"]
        if not isinstance(payload, dict):
            raise ValueError("external worker payload must be an object")
        if operation == "initialize":
            if session is not None:
                raise RuntimeError("external worker is already initialized")
            fields = {
                "benchmark",
                "container_runtime_path",
                "container_storage_root",
                "image_prefix",
                "source_root",
                "state_root",
                "task",
            }
            payload = _object(payload, fields=fields, label="external initialization")
            if payload["benchmark"] == "appworld":
                session = _AppWorldSession(payload)
            elif payload["benchmark"] == "skillflow-bench":
                session = _SkillFlowSession(payload)
            else:
                raise ValueError("external worker benchmark is unsupported")
            result = {"ready": True}
        elif session is None:
            raise RuntimeError("external worker is not initialized")
        elif operation == "execute":
            result = session.execute(payload)
        elif operation == "evaluate":
            result = session.evaluate()
        elif operation == "close":
            session.close()
            result = {"closed": True}
            _response(response_fd, request["request_id"], result)
            return
        else:
            raise ValueError("external worker operation is unsupported")
        _response(response_fd, request["request_id"], result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-fd", type=int, required=True)
    arguments = parser.parse_args()
    main(arguments.response_fd)
