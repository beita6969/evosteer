from __future__ import annotations

import sys

import pytest

from scripts.run_gate_4c_once import _args


def _argv(physical_gpu: int) -> list[str]:
    return [
        "run_gate_4c_once.py",
        "--python-executable",
        "python",
        "--gate-script",
        "gate.py",
        "--spec",
        "spec.json",
        "--local-model-path",
        "model",
        "--run-directory",
        "run",
        "--physical-gpu",
        str(physical_gpu),
    ]


@pytest.mark.parametrize("physical_gpu", range(8))
def test_all_physical_gpu_indices_are_selectable(monkeypatch, physical_gpu: int) -> None:
    monkeypatch.setattr(sys, "argv", _argv(physical_gpu))

    assert _args().physical_gpu == physical_gpu


@pytest.mark.parametrize("physical_gpu", [-1, 8])
def test_nonexistent_physical_gpu_indices_are_rejected(monkeypatch, physical_gpu: int) -> None:
    monkeypatch.setattr(sys, "argv", _argv(physical_gpu))

    with pytest.raises(SystemExit) as error:
        _args()

    assert error.value.code != 0
