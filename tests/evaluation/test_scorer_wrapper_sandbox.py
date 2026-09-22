"""A deployed scorer wrapper must run without exposing its private siblings."""

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.evaluation.integrity_runtime import _interpreter_sandbox

from skillev.evaluation.actor_sandbox import ActorSandbox


def test_trusted_wrapper_and_declared_dependencies_survive_isolation(tmp_path, monkeypatch):
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("requires Linux bubblewrap")
    dependency = tmp_path / "packages"
    dependency.mkdir()
    (dependency / "scorer_dependency.py").write_text("value = 42\n")
    secret = tmp_path / "not-an-interpreter-dependency.txt"
    secret.write_text("must stay outside scorer namespace")
    inherited = tmp_path / "coordinator-only"
    inherited.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(inherited))
    wrapper = tmp_path / "scorer-python"
    wrapper.write_text(
        f'#!/usr/bin/bash\nexport PYTHONPATH="{dependency}"\nexec "{sys.executable}" "$@"\n'
    )
    wrapper.chmod(0o700)
    sandbox = _interpreter_sandbox(
        wrapper, Path(__file__).resolve().parents[2] / "src", Path(bwrap)
    )
    result = subprocess.run(  # noqa: S603 -- synthetic wrapper, CPU only
        sandbox.command(
            "-c",
            "import scorer_dependency,json,pathlib; "
            "print(json.dumps([scorer_dependency.value, "
            f"pathlib.Path({str(secret)!r}).exists(), "
            f"pathlib.Path({str(inherited)!r}).exists()]))",
        ),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env={"CUDA_VISIBLE_DEVICES": ""},
    )
    assert json.loads(result.stdout) == [42, False, False]


def test_current_sandbox_uses_installed_binary_without_disabling_isolation(tmp_path, monkeypatch):
    from pathlib import Path

    from skillev.evaluation import actor_sandbox

    executable = tmp_path / "bin" / "bwrap"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setattr(actor_sandbox.shutil, "which", lambda _name: str(executable))
    sandbox = ActorSandbox.current(tmp_path)
    command = sandbox.command("-c", "pass")
    assert command[0] == str(executable)
    assert "--unshare-all" in command
    assert "--clearenv" in command
    missing = replace(sandbox, bubblewrap=Path(tmp_path / "absent"))
    with pytest.raises(RuntimeError):
        missing.command("-c", "pass")
