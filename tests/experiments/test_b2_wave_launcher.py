from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from skillev_private.experiments.b2_wave_launcher import (
    B2WaveSlot,
    SubprocessArmLauncher,
    execute_b2_waves,
    load_b2_waves,
)


class _Process:
    def __init__(self, *, returncode: int, events: list[str], label: str) -> None:
        self._returncode = returncode
        self._events = events
        self._label = label

    def wait(self) -> int:
        self._events.append(f"wait:{self._label}")
        return self._returncode


def _slot(kind: str, *, order: int, wave: int, gpu: int) -> B2WaveSlot:
    return B2WaveSlot(
        builder_kind=kind,
        attempt_id=f"attempt-{kind}",
        exact_input_sha256=f"sha256:{order:064x}",
        launch_order=order,
        physical_gpu=gpu,
        wave=wave,
    )


def test_execute_waves_launches_every_frozen_slot_once_after_failure() -> None:
    waves = (
        (_slot("a", order=1, wave=1, gpu=1), _slot("b", order=2, wave=1, gpu=2)),
        (_slot("c", order=3, wave=2, gpu=1), _slot("d", order=4, wave=2, gpu=2)),
        (_slot("e", order=5, wave=3, gpu=1), _slot("f", order=6, wave=3, gpu=2)),
        (_slot("g", order=7, wave=4, gpu=1),),
    )
    events: list[str] = []

    def launch(slot: B2WaveSlot) -> _Process:
        events.append(f"launch:{slot.builder_kind}")
        return _Process(
            returncode=1 if slot.builder_kind == "a" else 0,
            events=events,
            label=slot.builder_kind,
        )

    exits = execute_b2_waves(waves, launch=launch, emit=events.append)

    assert [item.slot.builder_kind for item in exits] == list("abcdefg")
    assert [item.returncode for item in exits] == [1, 0, 0, 0, 0, 0, 0]
    assert events.index("launch:b") < events.index("wait:a")
    assert events.index("launch:c") > events.index("wait:b")
    assert events.count("launch:g") == 1


def test_load_waves_preserves_frozen_order_and_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "format": "skillev-b2-execution-plan@2",
                "slots": [
                    {
                        "attempt_id": "attempt-b",
                        "builder_kind": "b",
                        "exact_input_sha256": "sha256:b",
                        "launch_order": 2,
                        "physical_gpu": 2,
                        "wave": 1,
                    },
                    {
                        "attempt_id": "attempt-a",
                        "builder_kind": "a",
                        "exact_input_sha256": "sha256:a",
                        "launch_order": 1,
                        "physical_gpu": 1,
                        "wave": 1,
                    },
                    {
                        "attempt_id": "attempt-c",
                        "builder_kind": "c",
                        "exact_input_sha256": "sha256:c",
                        "launch_order": 3,
                        "physical_gpu": 1,
                        "wave": 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    waves = load_b2_waves(path)

    assert [[slot.builder_kind for slot in wave] for wave in waves] == [["a", "b"], ["c"]]
    assert [[slot.physical_gpu for slot in wave] for wave in waves] == [[1, 2], [1]]


def test_subprocess_launcher_passes_exact_kind_and_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    class _Popen:
        def __init__(self, args: object, **kwargs: Any) -> None:
            calls.append({"args": args, **kwargs})

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", _Popen)
    launcher = SubprocessArmLauncher(
        python_executable=Path("/runtime/python"),
        operator_path=Path("/private/operator.py"),
        log_directory=tmp_path,
    )
    slot = _slot("full", order=1, wave=1, gpu=2)

    process = launcher(slot)
    launcher.close()

    assert process.wait() == 0
    assert calls[0]["args"] == (
        "/runtime/python",
        "/private/operator.py",
        "run-arm",
        "full",
        "--physical-gpu",
        "2",
    )
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
