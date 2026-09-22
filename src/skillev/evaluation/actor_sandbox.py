"""Filesystem/process/network isolation for evaluated actors and their peers.

Only public Python source and interpreter dependencies are mounted. All model
and environment access uses the parent-owned, task-scoped standard-I/O broker.
Evaluator data, host processes, credentials and host network are not exposed.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PRIVATE_TMP = "/tmp"  # noqa: S108 -- a fresh private tmpfs inside each namespace


@dataclass(frozen=True, slots=True)
class ActorSandbox:
    public_source: Path
    interpreter: Path
    python_prefix: Path
    base_prefix: Path
    bubblewrap: Path = Path("/usr/bin/bwrap")
    dependency_prefixes: tuple[Path, ...] = ()
    python_path: tuple[Path, ...] = ()

    @classmethod
    def current(cls, public_source: Path) -> ActorSandbox:
        return cls(
            public_source.resolve(),
            Path(sys.executable),
            Path(sys.prefix),
            Path(sys.base_prefix),
            Path(shutil.which("bwrap") or "/usr/bin/bwrap"),
        )

    def command(self, *python_arguments: str) -> tuple[str, ...]:
        if not self.bubblewrap.is_file():
            raise RuntimeError(
                "bubblewrap is required; actor isolation cannot silently be disabled"
            )
        command = [
            str(self.bubblewrap),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            _PRIVATE_TMP,
        ]
        mounts = [
            Path("/usr"),
            Path("/lib"),
            Path("/lib64"),
            self.base_prefix,
            self.python_prefix,
            *self.dependency_prefixes,
            *self.python_path,
        ]
        interpreter_link = self.interpreter
        while interpreter_link.is_symlink():
            target = interpreter_link.parent / os.readlink(interpreter_link)
            # The deployed venv points through an older, unversioned CPython
            # directory alias; sys.base_prefix contains only its resolved name.
            if target.parent.name == "bin":
                mounts.append(Path(os.path.abspath(target.parent.parent)))
            interpreter_link = target
        seen: set[Path] = set()
        # Copied virtualenvs can retain absolute links into the resolved base
        # interpreter prefix. Mount that same dependency tree under both names;
        # do not mount a broad home/data parent just to make a link resolve.
        for path in (*mounts, *(prefix.resolve() for prefix in mounts)):
            if path.exists() and path not in seen:
                command.extend(("--ro-bind", str(path), str(path)))
                seen.add(path)
        command.extend(
            (
                "--ro-bind",
                str(self.public_source),
                "/app/src",
                "--setenv",
                "PYTHONPATH",
                os.pathsep.join(("/app/src", *(str(path) for path in self.python_path))),
                "--setenv",
                "CUDA_VISIBLE_DEVICES",
                "",
                "--setenv",
                "HOME",
                _PRIVATE_TMP,
                "--setenv",
                "TMPDIR",
                _PRIVATE_TMP,
                "--setenv",
                "OMP_NUM_THREADS",
                "1",
                "--setenv",
                "OPENBLAS_NUM_THREADS",
                "1",
                "--setenv",
                "TOKENIZERS_PARALLELISM",
                "false",
                "--chdir",
                _PRIVATE_TMP,
                str(self.interpreter),
                *python_arguments,
            )
        )
        return tuple(command)

    async def run(
        self,
        initial: dict[str, Any],
        handle: Callable[[dict[str, Any]], Awaitable[object]],
        *,
        stderr_path: Path,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        stderr_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with stderr_path.open("ab") as stderr:
            process = await asyncio.create_subprocess_exec(
                *self.command("-m", "skillev.evaluation.integrity_actor"),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                env={"CUDA_VISIBLE_DEVICES": "", "PATH": os.defpath},
                limit=16 * 1024 * 1024,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                async with asyncio.timeout(timeout_seconds):
                    process.stdin.write((json.dumps(initial, ensure_ascii=False) + "\n").encode())
                    await process.stdin.drain()
                    while raw := await process.stdout.readline():
                        message = json.loads(raw)
                        if message.get("operation") == "final":
                            process.stdin.close()
                            if await process.wait() != 0:
                                raise RuntimeError(
                                    "isolated actor failed after submitting its final"
                                )
                            return dict(message["value"])
                        response = await handle(message)
                        process.stdin.write(
                            (json.dumps(response, ensure_ascii=False) + "\n").encode()
                        )
                        await process.stdin.drain()
                    raise RuntimeError(
                        "isolated actor exited without a final; inspect private stderr"
                    )
            finally:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 10)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
