"""Inspect build tools and actual serving CLI in the same child environment."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def inspect_build_tools(*, check_cli: bool = True) -> dict[str, object]:
    environment = dict(os.environ)
    cuda_home = environment.get("CUDA_HOME")
    search_path = environment.get("PATH", "")
    if cuda_home:
        search_path = str(Path(cuda_home) / "bin") + os.pathsep + search_path
    tools = {}
    for name in ("ninja", "nvcc"):
        path = shutil.which(name, path=search_path)
        if path is None:
            raise RuntimeError(f"serving child environment lacks {name}")
        result = subprocess.run(  # noqa: S603 - fixed build-tool argv, no shell
            [path, "--version"], capture_output=True, text=True, check=True, timeout=15
        )
        tools[name] = {"path": path, "version": result.stdout.strip()}
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "sglang", "flashinfer-python", "flash-linear-attention")
    }
    if check_cli:
        result = subprocess.run(  # noqa: S603 - fixed build-tool argv, no shell
            [sys.executable, "-m", "sglang.launch_server", "--help"],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        if any(
            flag not in result.stdout
            for flag in ("--max-running-requests", "--max-loras-per-batch")
        ):
            raise RuntimeError("installed serving CLI lacks mixed-capacity support")
    return {"python": sys.executable, "tools": tools, "packages": packages}


if __name__ == "__main__":
    print(json.dumps(inspect_build_tools(), sort_keys=True))
