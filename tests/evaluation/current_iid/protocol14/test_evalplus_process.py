from __future__ import annotations

import subprocess
from pathlib import Path

from skillev_private.benchmarks import evalplus_process
from skillev_private.benchmarks.evalplus_process import (
    EvalPlusProcessConfig,
    run_evalplus,
)


def test_evalplus_process_creates_package_cache_directory(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "missing-cache-root"
    samples = tmp_path / "samples.jsonl"
    samples.write_text("{}\n", encoding="utf-8")
    derived = tmp_path / "samples_eval_results.json"

    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))
    monkeypatch.setattr(evalplus_process, "version", lambda _name: "0.2.0")

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert (cache_root / "evalplus").is_dir()
        derived.write_text('{"eval": {}}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(evalplus_process.subprocess, "run", fake_run)
    result = run_evalplus(
        config=EvalPlusProcessConfig(
            python_executable=Path("/usr/bin/python3"),
            dataset="mbpp",
            version="v0.2.0",
            workers=1,
            timeout_seconds=10,
        ),
        samples=samples,
        output=tmp_path / "results.json",
        stdout=tmp_path / "stdout.log",
        stderr=tmp_path / "stderr.log",
    )

    assert result == {"eval": {}}
