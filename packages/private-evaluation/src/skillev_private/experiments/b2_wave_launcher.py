"""Execute a frozen B2 arm plan without result-dependent rescheduling.

The launcher is deliberately small: it reads the already-public execution
plan, starts every arm in its frozen wave and GPU coordinate, waits for the
whole wave, and then advances regardless of arm exit status.  It never retries
or substitutes a slot.  Keeping this logic in tested Python avoids shell
dynamic-scope mistakes silently skipping formal slots.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

B2_EXECUTION_PLAN_FORMAT = "skillev-b2-execution-plan@2"


@dataclass(frozen=True, slots=True)
class B2WaveSlot:
    """One immutable process coordinate from the frozen execution plan."""

    builder_kind: str
    attempt_id: str
    exact_input_sha256: str
    launch_order: int
    physical_gpu: int
    wave: int


@dataclass(frozen=True, slots=True)
class B2ArmExit:
    """Observed process exit for one frozen slot."""

    slot: B2WaveSlot
    returncode: int


class ArmProcess(Protocol):
    """The only process operation needed by the wave scheduler."""

    def wait(self) -> int: ...


ArmLauncher = Callable[[B2WaveSlot], ArmProcess]
EventEmitter = Callable[[str], None]


def _required_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _required_positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _required_non_negative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def load_b2_waves(path: Path) -> tuple[tuple[B2WaveSlot, ...], ...]:
    """Load slots in the exact launch order and wave grouping frozen by B1."""

    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != B2_EXECUTION_PLAN_FORMAT:
        raise ValueError("unsupported B2 execution plan")
    raw_slots = value.get("slots")
    if not isinstance(raw_slots, list) or not raw_slots:
        raise ValueError("B2 execution plan requires slots")

    slots: list[B2WaveSlot] = []
    for raw in raw_slots:
        if not isinstance(raw, dict):
            raise ValueError("B2 execution slot must be an object")
        slots.append(
            B2WaveSlot(
                builder_kind=_required_text(raw.get("builder_kind"), field="builder_kind"),
                attempt_id=_required_text(raw.get("attempt_id"), field="attempt_id"),
                exact_input_sha256=_required_text(
                    raw.get("exact_input_sha256"), field="exact_input_sha256"
                ),
                launch_order=_required_positive_int(raw.get("launch_order"), field="launch_order"),
                physical_gpu=_required_non_negative_int(
                    raw.get("physical_gpu"), field="physical_gpu"
                ),
                wave=_required_positive_int(raw.get("wave"), field="wave"),
            )
        )

    ordered = sorted(slots, key=lambda slot: slot.launch_order)
    if [slot.launch_order for slot in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("B2 launch_order must be contiguous and unique")
    waves = sorted({slot.wave for slot in ordered})
    if waves != list(range(1, len(waves) + 1)):
        raise ValueError("B2 wave ordinals must be contiguous")

    grouped: list[tuple[B2WaveSlot, ...]] = []
    for wave in waves:
        members = tuple(slot for slot in ordered if slot.wave == wave)
        if len({slot.physical_gpu for slot in members}) != len(members):
            raise ValueError("a B2 wave cannot schedule two arms on one physical GPU")
        grouped.append(members)
    return tuple(grouped)


def execute_b2_waves(
    waves: Sequence[Sequence[B2WaveSlot]],
    *,
    launch: ArmLauncher,
    emit: EventEmitter,
) -> tuple[B2ArmExit, ...]:
    """Run all frozen waves once, continuing after non-zero arm exits."""

    exits: list[B2ArmExit] = []
    for ordinal, wave in enumerate(waves, start=1):
        emit(f"wave_start wave={ordinal}")
        running = [(slot, launch(slot)) for slot in wave]
        wave_exits: list[B2ArmExit] = []
        for slot, process in running:
            observed = B2ArmExit(slot=slot, returncode=process.wait())
            exits.append(observed)
            wave_exits.append(observed)
            emit(
                "arm_terminal "
                f"kind={slot.builder_kind} gpu={slot.physical_gpu} "
                f"rc={observed.returncode}"
            )
        emit(
            f"wave_terminal wave={ordinal} "
            + " ".join(f"rc{index}={item.returncode}" for index, item in enumerate(wave_exits, 1))
        )
    return tuple(exits)


class SubprocessArmLauncher:
    """Launch the private operator with the slot's frozen physical GPU."""

    def __init__(
        self,
        *,
        python_executable: Path,
        operator_path: Path,
        log_directory: Path,
    ) -> None:
        self._python_executable = python_executable
        self._operator_path = operator_path
        self._log_directory = log_directory
        self._streams: list[Any] = []

    def __call__(self, slot: B2WaveSlot) -> subprocess.Popen[bytes]:
        stream = (self._log_directory / f"{slot.builder_kind}.log").open("xb")
        self._streams.append(stream)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(slot.physical_gpu)
        return subprocess.Popen(  # noqa: S603 - interpreter and operator are frozen inputs
            (
                str(self._python_executable),
                str(self._operator_path),
                "run-arm",
                slot.builder_kind,
                "--physical-gpu",
                str(slot.physical_gpu),
            ),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
        )

    def close(self) -> None:
        for stream in self._streams:
            stream.close()


def run_b2_execution_plan(
    *,
    plan_path: Path,
    python_executable: Path,
    operator_path: Path,
    control_directory: Path,
    log_directory: Path,
    emit: EventEmitter = print,
) -> int:
    """Execute one fresh plan and persist each observed arm exit code."""

    marker = control_directory / "launcher-started.marker"
    with marker.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())

    waves = load_b2_waves(plan_path)
    launcher = SubprocessArmLauncher(
        python_executable=python_executable,
        operator_path=operator_path,
        log_directory=log_directory,
    )
    try:
        exits = execute_b2_waves(waves, launch=launcher, emit=emit)
    finally:
        launcher.close()

    for observed in exits:
        path = control_directory / f"{observed.slot.builder_kind}.exit-code"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(f"{observed.returncode}\n")
            stream.flush()
            os.fsync(stream.fileno())
    return int(any(observed.returncode != 0 for observed in exits))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--control-directory", type=Path, required=True)
    parser.add_argument("--log-directory", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(
        run_b2_execution_plan(
            plan_path=args.plan,
            python_executable=args.python_executable,
            operator_path=args.operator,
            control_directory=args.control_directory,
            log_directory=args.log_directory,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
