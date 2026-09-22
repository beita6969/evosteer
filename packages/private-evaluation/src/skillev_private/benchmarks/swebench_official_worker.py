#!/usr/bin/env python3
"""Standalone one-request worker for an official SWE-bench instance image.

The worker is mounted read-only into the instance image.  It intentionally
uses only the Python standard library so it can run under the Python shipped by
official SWE-bench images.  The model-visible operations never read the
worker's private metadata directory under ``.git``.
"""

import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import typing

SWE_HARNESS_PROTOCOL_VERSION = "skillev-swebench-official-harness@1"
TESTBED_ROOT = pathlib.Path("/testbed")
WORK_ROOT = pathlib.Path("/skillev/work")

_REQUEST_FIELDS = {
    "deployment_id",
    "environment_image_id",
    "harness_revision",
    "operation",
    "payload",
    "protocol_version",
    "request_id",
}
_STATE_FORMAT = "skillev-swebench-official-worker-state@1"
_MAX_SEARCH_MATCHES = 200

JsonObject = typing.Dict[str, typing.Any]  # noqa: UP006 - Python 3.6 worker syntax.


class InfrastructureError(RuntimeError):
    """The official image, repository, or worker state is unusable."""


class WorkspaceCommandError(ValueError):
    """A model-issued workspace operation cannot be completed safely."""


def _object(
    value: typing.Any,
    fields: typing.Optional[typing.Set[str]] = None,  # noqa: UP006, UP045
    label: str = "value",
) -> JsonObject:
    if type(value) is not dict:
        raise InfrastructureError(f"{label} must be a JSON object")
    if fields is not None and set(value) != fields:
        raise InfrastructureError(f"{label} has an incompatible field set")
    return typing.cast(JsonObject, value)


def _text(value: typing.Any, label: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value:
        raise InfrastructureError(f"{label} must be text")
    if not allow_empty and not value.strip():
        raise InfrastructureError(f"{label} cannot be empty")
    return value


def _command_text(value: typing.Any, label: str, allow_empty: bool = False) -> str:
    if type(value) is not str or "\x00" in value:
        raise WorkspaceCommandError(f"{label} must be text")
    if not allow_empty and not value.strip():
        raise WorkspaceCommandError(f"{label} cannot be empty")
    return value


def _canonical_bytes(value: typing.Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_json(path: pathlib.Path, value: typing.Any) -> None:
    payload = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _run(
    argv: typing.Sequence[str],
    cwd: typing.Optional[pathlib.Path] = None,  # noqa: UP045
    stdin: typing.Optional[bytes] = None,  # noqa: UP045
    combine_output: bool = False,
) -> typing.Any:
    return subprocess.run(  # noqa: S603 - argv is fixed or an official trusted command.
        argv,
        cwd=None if cwd is None else str(cwd),
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if combine_output else subprocess.PIPE,
        check=False,
    )


def _git(
    arguments: typing.Sequence[str],
    stdin: typing.Optional[bytes] = None,  # noqa: UP045
    combine_output: bool = False,
) -> typing.Any:
    return _run(
        ["git", "-C", str(WORK_ROOT), *list(arguments)],
        stdin=stdin,
        combine_output=combine_output,
    )


def _metadata_root() -> pathlib.Path:
    return WORK_ROOT / ".git" / "skillev-worker"


def _state_path() -> pathlib.Path:
    return _metadata_root() / "state.json"


def _remove_path(path: pathlib.Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(str(path))


def _copy_entry(source: pathlib.Path, destination: pathlib.Path) -> None:
    if source.is_symlink():
        destination.symlink_to(os.readlink(str(source)), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(str(source), str(destination), symlinks=True)
    elif source.is_file():
        shutil.copy2(str(source), str(destination), follow_symlinks=False)
    else:
        raise InfrastructureError("official repository contains an unsupported file type")


def _clear_workspace(keep_git: bool = False) -> None:
    if not WORK_ROOT.is_dir():
        WORK_ROOT.mkdir(parents=True)
        return
    for child in WORK_ROOT.iterdir():
        if keep_git and child.name == ".git":
            continue
        _remove_path(child)


def _copy_testbed() -> None:
    if not TESTBED_ROOT.is_dir() or not (TESTBED_ROOT / ".git").is_dir():
        raise InfrastructureError("official /testbed Git repository is unavailable")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    _clear_workspace()
    for child in TESTBED_ROOT.iterdir():
        _copy_entry(child, WORK_ROOT / child.name)
    if not (WORK_ROOT / ".git").is_dir():
        raise InfrastructureError("fresh workspace has no Git metadata")


def _load_state(
    workspace_id: typing.Optional[str] = None,  # noqa: UP045
) -> JsonObject:
    try:
        value = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise InfrastructureError("official workspace state is unavailable") from error
    state = _object(
        value,
        {
            "base_commit",
            "format",
            "instance_id",
            "repo",
            "test_command",
            "workspace_id",
        },
        "workspace state",
    )
    if state["format"] != _STATE_FORMAT:
        raise InfrastructureError("official workspace state format is unsupported")
    for name in ("base_commit", "instance_id", "repo", "test_command", "workspace_id"):
        _text(state[name], f"workspace {name}")
    if workspace_id is not None and state["workspace_id"] != workspace_id:
        raise InfrastructureError("official workspace identity differs")
    return state


def _initialize(payload: typing.Any) -> JsonObject:
    payload = _object(
        payload,
        {
            "base_commit",
            "instance_id",
            "max_steps",
            "repo",
            "test_command",
            "workspace_id",
        },
        "workspace initialize payload",
    )
    base_commit = _text(payload["base_commit"], "base_commit")
    instance_id = _text(payload["instance_id"], "instance_id")
    repo = _text(payload["repo"], "repo")
    test_command = _text(payload["test_command"], "test_command")
    workspace_id = _text(payload["workspace_id"], "workspace_id")
    if type(payload["max_steps"]) is not int or payload["max_steps"] < 1:
        raise InfrastructureError("max_steps must be a positive integer")

    _copy_testbed()
    # Official SWE-bench images add a deterministic image-build commit on top
    # of the case's base commit.  The official evaluation scripts reset that
    # repository to ``base_commit`` before applying a candidate patch; requiring
    # the image's initial HEAD to equal the case base rejects valid official
    # images.  Verify that the requested commit is present, then establish the
    # same clean base state in the host workspace.
    commit = _git(["cat-file", "-e", base_commit + "^{commit}"])
    if commit.returncode != 0:
        raise InfrastructureError("official /testbed lacks the requested base commit")
    reset = _git(["reset", "--hard", base_commit])
    clean = _git(["clean", "-fdx"])
    head = _git(["rev-parse", "HEAD"])
    if (
        reset.returncode != 0
        or clean.returncode != 0
        or head.returncode != 0
        or head.stdout.decode("ascii", "replace").strip() != base_commit
    ):
        raise InfrastructureError("official workspace could not establish its base commit")
    _write_json(
        _state_path(),
        {
            "base_commit": base_commit,
            "format": _STATE_FORMAT,
            "instance_id": instance_id,
            "repo": repo,
            "test_command": test_command,
            "workspace_id": workspace_id,
        },
    )
    return {
        "base_commit": base_commit,
        "initialized": True,
        "instance_id": instance_id,
        "workspace_id": workspace_id,
    }


def _relative_user_path(value: typing.Any, allow_root: bool = False) -> pathlib.PurePosixPath:
    text = _command_text(value, "workspace path")
    path = pathlib.PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise WorkspaceCommandError("workspace path is outside the public repository")
    normalized = path.as_posix()
    if normalized in ("", "."):
        if allow_root:
            return pathlib.PurePosixPath(".")
        raise WorkspaceCommandError("workspace path cannot be the repository root")
    if normalized != text:
        raise WorkspaceCommandError("workspace path must be normalized POSIX text")
    return path


def _within_workspace(path: pathlib.Path) -> pathlib.Path:
    workspace = WORK_ROOT.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise WorkspaceCommandError(
            "workspace path follows a link outside the repository"
        ) from error
    return resolved


def _existing_public_path(
    value: typing.Any, allow_root: bool = False
) -> typing.Tuple[pathlib.PurePosixPath, pathlib.Path]:  # noqa: UP006
    relative = _relative_user_path(value, allow_root=allow_root)
    candidate = WORK_ROOT / relative
    if not candidate.exists():
        raise WorkspaceCommandError("workspace path does not exist")
    return relative, _within_workspace(candidate)


def _write_public_file(path_text: typing.Any, content: typing.Any) -> JsonObject:
    relative = _relative_user_path(path_text)
    content = _command_text(content, "content", allow_empty=True)
    candidate = WORK_ROOT / relative
    parent = _within_workspace(candidate.parent)
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / candidate.name
    if candidate.is_symlink() or candidate.is_dir():
        raise WorkspaceCommandError("write target must be a regular file")
    payload = content.encode("utf-8")
    temporary = candidate.with_name(candidate.name + ".skillev-tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(candidate))
    return {
        "byte_count": len(payload),
        "kind": "write",
        "path": relative.as_posix(),
        "status": "completed",
    }


def _read_public_file(path_text: typing.Any) -> JsonObject:
    relative, path = _existing_public_path(path_text)
    if not path.is_file() or path.is_symlink():
        raise WorkspaceCommandError("read target must be a regular text file")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise WorkspaceCommandError("read target is not UTF-8 text") from error
    return {
        "content": content,
        "kind": "read",
        "path": relative.as_posix(),
        "status": "completed",
    }


def _search_public_text(path_text: typing.Any, query: typing.Any) -> JsonObject:
    relative, root = _existing_public_path(path_text, allow_root=True)
    query = _command_text(query, "query")
    candidates = []  # type: typing.List[pathlib.Path]
    if root.is_file() and not root.is_symlink():
        candidates.append(root)
    elif root.is_dir():
        for directory, names, filenames in os.walk(str(root), followlinks=False):
            names[:] = sorted(name for name in names if name != ".git")
            for filename in sorted(filenames):
                candidate = pathlib.Path(directory) / filename
                if not candidate.is_symlink():
                    candidates.append(candidate)
    else:
        raise WorkspaceCommandError("search target must be a file or directory")

    matches = []  # type: typing.List[JsonObject]
    for candidate in candidates:
        _within_workspace(candidate)
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        display = candidate.resolve().relative_to(WORK_ROOT.resolve()).as_posix()
        for line_number, line in enumerate(lines, 1):
            if query in line:
                matches.append({"line": line, "line_number": line_number, "path": display})
                if len(matches) >= _MAX_SEARCH_MATCHES:
                    return {
                        "kind": "search",
                        "matches": matches,
                        "path": relative.as_posix(),
                        "query": query,
                        "status": "completed",
                        "truncated": True,
                    }
    return {
        "kind": "search",
        "matches": matches,
        "path": relative.as_posix(),
        "query": query,
        "status": "completed",
        "truncated": False,
    }


def _test_target(target: typing.Any, state: typing.Mapping[str, typing.Any]) -> JsonObject:
    target = _command_text(target, "test target")
    if target.startswith("-"):
        raise WorkspaceCommandError("test target cannot be an option")
    path_text = target.split("::", 1)[0]
    _existing_public_path(path_text, allow_root=True)
    command = state["test_command"] + " " + shlex.quote(target)
    completed = _run(
        ["/bin/bash", "-c", command],
        cwd=WORK_ROOT,
        combine_output=True,
    )
    return {
        "kind": "test",
        "output": completed.stdout.decode("utf-8", "replace"),
        "returncode": completed.returncode,
        "status": "completed",
        "target": target,
    }


def _budget(elapsed_seconds: float) -> typing.Dict[str, int]:  # noqa: UP006
    milliseconds = max(1, round(elapsed_seconds * 1000.0))
    return {
        "agent_turns": 0,
        "input_tokens": 0,
        "model_calls": 0,
        "output_tokens": 0,
        "tool_calls": 1,
        "wall_time_milliseconds": milliseconds,
    }


def _execute(payload: typing.Any) -> JsonObject:
    payload = _object(
        payload,
        {"command", "step_index", "workspace_id"},
        "workspace execute payload",
    )
    workspace_id = _text(payload["workspace_id"], "workspace_id")
    state = _load_state(workspace_id)
    step_index = payload["step_index"]
    if type(step_index) is not int or step_index < 1:
        raise InfrastructureError("step_index must be a positive integer")
    command = _object(payload["command"], {"arguments", "kind"}, "workspace command")
    kind = _text(command["kind"], "command kind")
    arguments = _object(command["arguments"], label="command arguments")

    started = time.monotonic()
    terminal = False
    try:
        if kind == "read":
            if set(arguments) != {"path"}:
                raise WorkspaceCommandError("read arguments are invalid")
            observation = _read_public_file(arguments["path"])
        elif kind == "write":
            if set(arguments) != {"content", "path"}:
                raise WorkspaceCommandError("write arguments are invalid")
            observation = _write_public_file(arguments["path"], arguments["content"])
        elif kind == "search":
            if set(arguments) != {"path", "query"}:
                raise WorkspaceCommandError("search arguments are invalid")
            observation = _search_public_text(arguments["path"], arguments["query"])
        elif kind == "test":
            if set(arguments) != {"target"}:
                raise WorkspaceCommandError("test arguments are invalid")
            observation = _test_target(arguments["target"], state)
        elif kind == "submit_patch":
            if set(arguments) != {"patch"}:
                raise WorkspaceCommandError("submit_patch arguments are invalid")
            patch = _command_text(arguments["patch"], "patch", allow_empty=True)
            observation = {
                "kind": "submit_patch",
                "patch_byte_count": len(patch.encode("utf-8")),
                "status": "submitted",
            }
            terminal = True
        else:
            raise WorkspaceCommandError("workspace command kind is unsupported")
    except WorkspaceCommandError as error:
        observation = {"error": str(error), "kind": kind, "status": "error"}
    elapsed = time.monotonic() - started
    return {
        "budget_usage": _budget(elapsed),
        "public_observation": observation,
        "step_index": step_index,
        "terminal": terminal,
        "workspace_id": workspace_id,
    }


def _reset_to_base(state: typing.Mapping[str, typing.Any]) -> None:
    reset = _git(["reset", "--hard", state["base_commit"]])
    if reset.returncode != 0:
        raise InfrastructureError("official workspace could not reset to its base commit")
    clean = _git(["clean", "-fdx"])
    if clean.returncode != 0:
        raise InfrastructureError("official workspace could not remove prior public edits")


def _apply_patch(patch: typing.Union[str, bytes]) -> bool:  # noqa: UP007
    encoded = patch if isinstance(patch, bytes) else patch.encode("utf-8")
    checked = _git(["apply", "--check", "--whitespace=nowarn", "-"], stdin=encoded)
    if checked.returncode != 0:
        return False
    applied = _git(["apply", "--whitespace=nowarn", "-"], stdin=encoded)
    return int(applied.returncode) == 0


def _translate_official_eval_script(eval_script: str, private_test_patch_path: pathlib.Path) -> str:
    """Translate only the two deployment paths used by official eval scripts."""

    terminator = r"(?=$|[\s/\"'`;|&()<>{}\[\]])"

    def replace_path(script: str, source: str, destination: str) -> str:
        pattern = r"(?<![A-Za-z0-9_.-])" + re.escape(source) + terminator
        return re.sub(pattern, lambda _match: destination, script)

    translated = replace_path(
        eval_script,
        "/root/test.patch",
        str(private_test_patch_path),
    )
    return replace_path(translated, "/testbed", str(WORK_ROOT))


def _verify(payload: typing.Any) -> JsonObject:
    required = {
        "base_commit",
        "candidate_patch",
        "eval_script",
        "instance_id",
        "repo",
        "test_patch",
        "version",
    }
    payload = _object(payload, required, "verify payload")
    base_commit = _text(payload["base_commit"], "base_commit")
    instance_id = _text(payload["instance_id"], "instance_id")
    repo = _text(payload["repo"], "repo")
    _text(payload["version"], "version")
    eval_script = _text(payload["eval_script"], "eval_script")
    test_patch = _text(payload["test_patch"], "test_patch")
    candidate_patch = payload["candidate_patch"]
    if candidate_patch is not None and type(candidate_patch) is not str:
        raise InfrastructureError("candidate_patch must be text or null")

    state = _load_state()
    if (
        state["base_commit"] != base_commit
        or state["instance_id"] != instance_id
        or state["repo"] != repo
    ):
        raise InfrastructureError("verify identity differs from initialized workspace")
    _reset_to_base(state)
    if candidate_patch is None or not candidate_patch.strip():
        return {
            "candidate_patch_applied": False,
            "instance_id": instance_id,
            "test_output": "",
        }
    if not _apply_patch(candidate_patch):
        return {
            "candidate_patch_applied": False,
            "instance_id": instance_id,
            "test_output": "",
        }
    _metadata_root().mkdir(parents=True, exist_ok=True)
    patch_descriptor, patch_name = tempfile.mkstemp(
        prefix="test-patch-", suffix=".diff", dir=str(_metadata_root())
    )
    script_descriptor, script_name = tempfile.mkstemp(
        prefix="eval-", suffix=".sh", dir=str(_metadata_root())
    )
    completed = None  # type: typing.Any
    try:
        with os.fdopen(patch_descriptor, "wb") as stream:
            stream.write(test_patch.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(patch_name, 0o600)
        translated_script = _translate_official_eval_script(
            eval_script,
            pathlib.Path(patch_name),
        )
        with os.fdopen(script_descriptor, "wb") as stream:
            stream.write(translated_script.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(script_name, 0o700)
        completed = _run(
            ["/bin/bash", script_name],
            cwd=WORK_ROOT,
            combine_output=True,
        )
    finally:
        for private_path in (script_name, patch_name):
            try:
                os.unlink(private_path)
            except FileNotFoundError:
                pass
        _reset_to_base(state)
    if completed is None:
        raise InfrastructureError("official eval script did not execute")
    return {
        "candidate_patch_applied": True,
        "instance_id": instance_id,
        "test_output": completed.stdout.decode("utf-8", "replace"),
    }


def _dispatch(operation: str, payload: JsonObject) -> JsonObject:
    if operation == "workspace-initialize":
        return _initialize(payload)
    if operation == "workspace-execute":
        return _execute(payload)
    if operation == "verify":
        return _verify(payload)
    raise InfrastructureError("official SWE harness operation is unsupported")


def _handle(request: typing.Any) -> JsonObject:
    request = _object(request, _REQUEST_FIELDS, "official SWE harness request")
    if request["protocol_version"] != SWE_HARNESS_PROTOCOL_VERSION:
        raise InfrastructureError("official SWE harness protocol version differs")
    for name in (
        "deployment_id",
        "environment_image_id",
        "harness_revision",
        "operation",
        "request_id",
    ):
        _text(request[name], name)
    payload = _object(request["payload"], label="request payload")
    result = _dispatch(request["operation"], payload)
    return {
        "deployment_id": request["deployment_id"],
        "environment_image_id": request["environment_image_id"],
        "harness_revision": request["harness_revision"],
        "protocol_version": SWE_HARNESS_PROTOCOL_VERSION,
        "request_id": request["request_id"],
        "result": result,
    }


def main() -> int:
    try:
        raw = sys.stdin.buffer.read()
        if not raw:
            raise InfrastructureError("official SWE harness request is empty")
        request = json.loads(raw.decode("utf-8"))
        response = _handle(request)
        sys.stdout.buffer.write(_canonical_bytes(response) + b"\n")
        sys.stdout.buffer.flush()
        return 0
    except Exception:
        sys.stderr.write("official SWE harness worker infrastructure failure\n")
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
