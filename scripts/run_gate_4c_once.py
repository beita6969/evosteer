#!/usr/bin/env python3
"""One-shot Gate 4c operator owning logs, GPU telemetry, and final process status."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from scripts.gate4c_durable_io import write_text_once_atomic
from scripts.gate4c_runtime_support import CUBLAS_WORKSPACE_CONFIG
from skillev.contracts import canonical_json
from skillev.experiments import (
    Gate4cChildStatus,
    Gate4cChildTerminal,
    Gate4cFailureCode,
    Gate4cOperatorStatus,
    Gate4cOperatorTerminal,
    Gate4cSpec,
)
from skillev.experiments.build_identity import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--gate-script", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--local-model-path", required=True)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--physical-gpu", type=int, choices=range(8), required=True)
    return parser.parse_args()


def _read_spec(path: Path) -> Gate4cSpec:
    return Gate4cSpec.from_value(json.loads(path.read_text(encoding="utf-8")))


def _physical_gpu_uuid(nvidia_smi: str, physical_gpu: int) -> str:
    completed = subprocess.run(  # noqa: S603 -- operator executes its pinned binary directly
        [
            nvidia_smi,
            f"--id={physical_gpu}",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    values = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if len(values) != 1:
        raise RuntimeError("Gate 4c operator did not resolve exactly one physical GPU UUID")
    return values[0]


def _stop_sampler(sampler: subprocess.Popen[bytes]) -> int:
    if sampler.poll() is None:
        sampler.send_signal(signal.SIGINT)
    try:
        return sampler.wait(timeout=15)
    except subprocess.TimeoutExpired:
        sampler.terminate()
        try:
            return sampler.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sampler.kill()
            return sampler.wait(timeout=5)


def _require_sampler_started(
    sampler: subprocess.Popen[bytes], samples_path: Path, interval_ms: int
) -> None:
    deadline = time.monotonic() + max(2.0, interval_ms * 4 / 1_000)
    while time.monotonic() < deadline:
        if samples_path.stat().st_size > 0:
            return
        if sampler.poll() is not None:
            raise RuntimeError("Gate 4c GPU sampler exited before its first record")
        time.sleep(0.01)
    raise TimeoutError("Gate 4c GPU sampler produced no initial record")


def run_gate_4c_once(
    *,
    python_executable: Path,
    gate_script: Path,
    spec_path: Path,
    local_model_path: Path,
    run_directory: Path,
    physical_gpu: int,
    nvidia_smi: str = "nvidia-smi",
) -> Gate4cOperatorTerminal:
    """Launch exactly one child; no code path launches a replacement."""

    spec = _read_spec(spec_path)
    run_directory.mkdir(parents=True, exist_ok=False)
    control = run_directory / "control"
    public = run_directory / "public"
    private = run_directory / "private"
    scratch = run_directory / "scratch"
    for directory in (control, public, private, scratch):
        directory.mkdir()
    write_text_once_atomic(
        control / "claim.json",
        canonical_json(
            {
                "format": "skillev-gate-4c-operator-claim@1",
                "physical_gpu": physical_gpu,
                "spec_content_hash": spec.content_hash,
            }
        )
        + "\n",
    )

    stdout_path = private / "stdout.log"
    stderr_path = private / "stderr.log"
    samples_path = private / "gpu-samples.csv"
    child_terminal_path = public / "child-terminal.json"
    started_ns = time.monotonic_ns()
    child: subprocess.Popen[bytes] | None = None
    sampler: subprocess.Popen[bytes] | None = None
    sampler_exit: int | None = None
    sampler_ready = False
    prelaunch_error = False
    with (
        stdout_path.open("xb") as stdout,
        stderr_path.open("xb") as stderr,
        samples_path.open("xb") as samples,
    ):
        try:
            measured_uuid = _physical_gpu_uuid(nvidia_smi, physical_gpu)
            if measured_uuid != spec.expected_physical_gpu_uuid:
                raise ValueError("allocated physical GPU UUID differs from Gate 4c spec")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
            environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
            sampler = subprocess.Popen(  # noqa: S603 -- exact nvidia-smi argument vector
                [
                    nvidia_smi,
                    f"--id={physical_gpu}",
                    "--query-gpu=timestamp,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={spec.gpu_sampling_interval_ms}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=samples,
                stderr=stderr,
                start_new_session=True,
            )
            _require_sampler_started(sampler, samples_path, spec.gpu_sampling_interval_ms)
            sampler_ready = True
            command = [
                str(python_executable),
                str(gate_script),
                "--spec",
                str(spec_path),
                "--local-model-path",
                str(local_model_path),
                "--work-directory",
                str(run_directory),
            ]
            write_text_once_atomic(
                control / "launch.json",
                canonical_json(
                    {
                        "format": "skillev-gate-4c-operator-launch@1",
                        "physical_gpu": physical_gpu,
                        "spec_content_hash": spec.content_hash,
                    }
                )
                + "\n",
            )
            child = subprocess.Popen(  # noqa: S603 -- exact Python child argument vector
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=True,
            )
            write_text_once_atomic(control / "child.pid", f"{child.pid}\n")
            child.wait()
        except BaseException:
            prelaunch_error = child is None
            stderr.write(traceback.format_exc().encode("utf-8"))
            stderr.flush()
            if child is not None and child.poll() is None:
                child.wait()
        finally:
            if sampler is not None:
                sampler_exit = _stop_sampler(sampler)

    child_return = child.returncode if child is not None else None
    child_exit_code = child_return if child_return is not None and child_return >= 0 else None
    child_signal = -child_return if child_return is not None and child_return < 0 else None
    child_terminal: Gate4cChildTerminal | None = None
    if child_terminal_path.is_file():
        try:
            child_terminal = Gate4cChildTerminal.from_value(
                json.loads(child_terminal_path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            child_terminal = None
    sampler_ok = sampler_ready and sampler_exit in {0, 130, -signal.SIGINT}
    passed = (
        child_exit_code == 0
        and sampler_ok
        and child_terminal is not None
        and child_terminal.status is Gate4cChildStatus.PASSED
        and child_terminal.spec_content_hash == spec.content_hash
    )
    if passed:
        status = Gate4cOperatorStatus.PASSED
        failure_code = None
    else:
        status = Gate4cOperatorStatus.FAILED_CONSUMED_NO_RETRY
        if not sampler_ok:
            failure_code = Gate4cFailureCode.GPU_SAMPLER_FAILED
        elif child_signal is not None:
            failure_code = Gate4cFailureCode.PROCESS_SIGNALLED
        elif child_terminal is None:
            failure_code = Gate4cFailureCode.PROCESS_EXITED_WITHOUT_CHILD_TERMINAL
        else:
            failure_code = Gate4cFailureCode.PYTHON_EXCEPTION
    terminal = Gate4cOperatorTerminal(
        status=status,
        failure_code=failure_code,
        spec_content_hash=spec.content_hash,
        physical_gpu=physical_gpu,
        expected_gpu_uuid=spec.expected_physical_gpu_uuid,
        child_pid=child.pid if child is not None else None,
        child_exit_code=child_exit_code,
        child_signal=child_signal,
        stdout_sha256=sha256_file(stdout_path),
        stderr_sha256=sha256_file(stderr_path),
        gpu_samples_sha256=sha256_file(samples_path),
        child_terminal_present=child_terminal is not None,
        child_terminal_sha256=(sha256_file(child_terminal_path) if child_terminal else None),
        elapsed_ns=time.monotonic_ns() - started_ns,
    )
    write_text_once_atomic(
        control / "operator-terminal.json",
        canonical_json(terminal.to_value()) + "\n",
    )
    recorded_exit = child_exit_code
    if recorded_exit is None:
        recorded_exit = -int(child_signal or prelaunch_error)
    write_text_once_atomic(control / "exit-code.txt", f"{recorded_exit}\n")
    return terminal


def main() -> None:
    args = _args()
    terminal = run_gate_4c_once(
        python_executable=Path(args.python_executable),
        gate_script=Path(args.gate_script),
        spec_path=Path(args.spec),
        local_model_path=Path(args.local_model_path),
        run_directory=Path(args.run_directory),
        physical_gpu=args.physical_gpu,
    )
    raise SystemExit(0 if terminal.status is Gate4cOperatorStatus.PASSED else 1)


if __name__ == "__main__":
    main()
