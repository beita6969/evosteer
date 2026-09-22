from __future__ import annotations

import argparse
from pathlib import Path

from run_training import _run_torchrun, _torchrun_command, _training_visible_devices
from skillev.runtime.formal_preflight import FormalBaselineConfig


def _arguments() -> argparse.Namespace:
    return argparse.Namespace(
        config="configs/baseline/paper_v1_250step.yaml",
        allow_resume_git_mismatch=False,
        fresh=True,
        max_steps=3,
        resume=None,
    )


def test_gpu4_restart_uses_committed_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "checkpoint_step_000002"
    command = _torchrun_command(
        _arguments(),
        run_directory=tmp_path,
        processes=3,
        resume_override=str(checkpoint),
    )

    assert "--nproc_per_node=3" in command
    assert "--fresh" not in command
    assert command[command.index("--resume") + 1] == str(checkpoint)


def test_resume_git_override_is_forwarded_to_workers(tmp_path: Path) -> None:
    arguments = _arguments()
    arguments.fresh = False
    arguments.resume = str(tmp_path / "checkpoint_step_000001")
    arguments.allow_resume_git_mismatch = True

    command = _torchrun_command(arguments, run_directory=tmp_path, processes=2)

    assert "--allow-resume-git-mismatch" in command


def test_host_local_gpu_map_controls_steady_and_expanded_visibility(monkeypatch) -> None:
    config = FormalBaselineConfig.read("configs/baseline/paper_v1_250step.yaml")
    monkeypatch.setenv("SKILLEV_TRAINING_GPU_MAP", "3,4,7")

    assert _training_visible_devices(config, expanded=False) == "3,4"
    assert _training_visible_devices(config, expanded=True) == "3,4,7"


def test_host_local_gpu_uuids_control_steady_and_expanded_visibility(monkeypatch) -> None:
    config = FormalBaselineConfig.read("configs/baseline/paper_v1_250step.yaml")
    monkeypatch.setenv("SKILLEV_TRAINING_GPU_MAP", "3,4,7")
    monkeypatch.setenv(
        "SKILLEV_TRAINING_CUDA_VISIBLE_DEVICES",
        "GPU-coordinator,GPU-primary,GPU-standby",
    )

    assert _training_visible_devices(config, expanded=False) == "GPU-coordinator,GPU-primary"
    assert _training_visible_devices(config, expanded=True) == (
        "GPU-coordinator,GPU-primary,GPU-standby"
    )


def test_torchrun_isolated_from_terminal_signals(tmp_path: Path, monkeypatch) -> None:
    observed = {}

    class Process:
        def wait(self) -> int:
            return 0

    def popen(command, **kwargs):
        observed.update(kwargs)
        observed["command"] = command
        return Process()

    monkeypatch.setattr("run_training.subprocess.Popen", popen)

    assert _run_torchrun(["python", "worker.py"], environment={}, run_directory=tmp_path) == 0
    assert observed["start_new_session"] is True
