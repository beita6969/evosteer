from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from scripts import gate4c_durable_io
from scripts.gate4c_durable_io import write_text_once_atomic
from scripts.gate4c_journal import Gate4cJournal, read_gate4c_journal
from scripts.real_9b_gate_4c import run_child
from scripts.run_gate_4c_once import run_gate_4c_once
from skillev.contracts import canonical_json, stable_hash
from skillev.experiments import (
    Gate4cChildStatus,
    Gate4cChildTerminal,
    Gate4cFailureCode,
    Gate4cOperatorStatus,
    Gate4cOperatorTerminal,
    Gate4cStage,
    Gate4cStageState,
)
from tests.experiments.test_real_9b_gate_4c import _spec


class InjectedStageError(RuntimeError):
    pass


def _operator_directories(root: Path) -> None:
    for name in ("public", "private", "scratch"):
        (root / name).mkdir(parents=True)


def _failing_execute(stage: Gate4cStage):
    def execute(*, spec_path: Path, model_path: Path, work: Path, journal: Gate4cJournal):
        del spec_path, model_path, work
        with journal.stage(stage):
            raise InjectedStageError(stage.value)

    return execute


@pytest.mark.parametrize("stage", tuple(Gate4cStage))
def test_every_gate_stage_failure_writes_one_child_terminal(
    tmp_path: Path, stage: Gate4cStage
) -> None:
    run = tmp_path / stage.value
    _operator_directories(run)
    hook = None
    execute = _failing_execute(stage)
    if stage is Gate4cStage.PROCESS_START:

        def fail_process_start() -> None:
            raise InjectedStageError(stage.value)

        hook = fail_process_start

    with pytest.raises(InjectedStageError):
        run_child(
            spec_path=tmp_path / "spec.json",
            model_path=tmp_path / "model",
            work=run,
            execute=execute,
            process_start_hook=hook,
        )

    terminal_path = run / "public" / "child-terminal.json"
    terminal = Gate4cChildTerminal.from_value(json.loads(terminal_path.read_text()))
    events = read_gate4c_journal(run / "private" / "stage-journal.jsonl")
    assert terminal.status is Gate4cChildStatus.FAILED
    assert terminal.final_stage is stage
    assert terminal.exception_type.endswith("InjectedStageError")
    assert terminal.result_sha256 is None
    assert events[-1].stage is stage
    assert events[-1].state is Gate4cStageState.FAILED
    assert len(tuple((run / "public").glob("child-terminal.json"))) == 1
    assert f"InjectedStageError: {stage.value}" not in terminal_path.read_text()
    assert str(tmp_path) not in terminal_path.read_text()


def test_gate_stage_journal_is_chained_and_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = Gate4cJournal(path)
    with journal.stage(Gate4cStage.PROCESS_START):
        pass
    with journal.stage(Gate4cStage.SPEC_READ):
        pass

    events = read_gate4c_journal(path)
    assert [event.ordinal for event in events] == [1, 2, 3, 4]
    assert events[1].previous_event_hash == events[0].content_hash
    lines = path.read_text().splitlines()
    value = json.loads(lines[2])
    value["previous_event_hash"] = stable_hash({"tampered": True})
    lines[2] = canonical_json(value)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        read_gate4c_journal(path)


def test_gate_stage_journal_fsyncs_every_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_fsync = gate4c_durable_io.os.fsync

    def record_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(gate4c_durable_io.os, "fsync", record_fsync)
    journal = Gate4cJournal(tmp_path / "journal.jsonl")
    with journal.stage(Gate4cStage.PROCESS_START):
        pass

    assert len(calls) == 2


def test_gate_write_once_does_not_replace_existing_evidence(tmp_path: Path) -> None:
    path = tmp_path / "terminal.json"
    write_text_once_atomic(path, "first\n")
    with pytest.raises(FileExistsError):
        write_text_once_atomic(path, "second\n")
    assert path.read_text() == "first\n"


def test_python_child_success_writes_one_terminal_and_result(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _operator_directories(run)
    spec = _spec()

    def execute(*, spec_path: Path, model_path: Path, work: Path, journal: Gate4cJournal):
        del spec_path, model_path, work
        with journal.stage(Gate4cStage.SPEC_READ):
            pass
        return spec, {"format": "gate-fixture", "gate_passed": True}

    terminal = run_child(
        spec_path=tmp_path / "spec.json",
        model_path=tmp_path / "model",
        work=run,
        execute=execute,
    )

    assert terminal.status is Gate4cChildStatus.PASSED
    assert terminal.final_stage is Gate4cStage.PROCESS_SUCCESS
    assert (run / "public" / "gate-4c-result.json").is_file()
    assert len(tuple((run / "public").glob("child-terminal.json"))) == 1
    events = read_gate4c_journal(run / "private" / "stage-journal.jsonl")
    assert events[-1].stage is Gate4cStage.PROCESS_SUCCESS
    assert events[-1].state is Gate4cStageState.COMPLETED


def _write_fake_nvidia_smi(path: Path, uuid: str) -> None:
    path.write_text(
        f"""#!/usr/bin/env python3
import signal
import sys
import time

if any(value == '--query-gpu=uuid' for value in sys.argv):
    print({uuid!r})
    raise SystemExit(0)

stopping = False
def stop(signum, frame):
    global stopping
    stopping = True
signal.signal(signal.SIGINT, stop)
while not stopping:
    print('2026/08/11, {uuid}, 0, 81559, 0, 30, 70', flush=True)
    time.sleep(0.02)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_native_exit_child(path: Path) -> None:
    path.write_text(
        """import os
import sys
print('owned stdout', flush=True)
print('owned stderr', file=sys.stderr, flush=True)
os._exit(73)
""",
        encoding="utf-8",
    )


def _write_failing_nvidia_smi(path: Path, uuid: str) -> None:
    path.write_text(
        f"""#!/usr/bin/env python3
import sys
if any(value == '--query-gpu=uuid' for value in sys.argv):
    print({uuid!r})
    raise SystemExit(0)
raise SystemExit(9)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_success_child(path: Path) -> None:
    path.write_text(
        """import argparse
import json
from pathlib import Path
from skillev.contracts import canonical_json, stable_hash
from skillev.experiments import Gate4cChildTerminal, Gate4cSpec, Gate4cStage

parser = argparse.ArgumentParser()
parser.add_argument('--spec')
parser.add_argument('--local-model-path')
parser.add_argument('--work-directory')
args = parser.parse_args()
spec = Gate4cSpec.from_value(json.loads(Path(args.spec).read_text()))
identity = stable_hash({'fixture': 'operator-success'})
terminal = Gate4cChildTerminal.passed(
    spec_content_hash=spec.content_hash,
    final_stage=Gate4cStage.PROCESS_SUCCESS,
    result_sha256=identity,
    stage_journal_sha256=identity,
    elapsed_ns=1,
)
Path(args.work_directory, 'public', 'child-terminal.json').write_text(
    canonical_json(terminal.to_value()) + '\\n'
)
print('success child stdout', flush=True)
""",
        encoding="utf-8",
    )


def test_operator_records_native_child_exit_logs_and_sampler_lifecycle(tmp_path: Path) -> None:
    spec = _spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(canonical_json(spec.to_value()))
    nvidia_smi = tmp_path / "nvidia-smi"
    _write_fake_nvidia_smi(nvidia_smi, spec.expected_physical_gpu_uuid)
    child = tmp_path / "native-exit.py"
    _write_native_exit_child(child)
    run = tmp_path / "run"

    terminal = run_gate_4c_once(
        python_executable=Path(sys.executable),
        gate_script=child,
        spec_path=spec_path,
        local_model_path=tmp_path / "model",
        run_directory=run,
        physical_gpu=1,
        nvidia_smi=str(nvidia_smi),
    )

    restored = Gate4cOperatorTerminal.from_value(
        json.loads((run / "control" / "operator-terminal.json").read_text())
    )
    assert restored == terminal
    assert terminal.status is Gate4cOperatorStatus.FAILED_CONSUMED_NO_RETRY
    assert terminal.failure_code is Gate4cFailureCode.PROCESS_EXITED_WITHOUT_CHILD_TERMINAL
    assert terminal.child_exit_code == 73
    assert not terminal.child_terminal_present
    assert "owned stdout" in (run / "private" / "stdout.log").read_text()
    assert "owned stderr" in (run / "private" / "stderr.log").read_text()
    assert spec.expected_physical_gpu_uuid in (run / "private" / "gpu-samples.csv").read_text()
    assert not os.path.exists(f"/proc/{terminal.child_pid}")
    with pytest.raises(FileExistsError):
        run_gate_4c_once(
            python_executable=Path(sys.executable),
            gate_script=child,
            spec_path=spec_path,
            local_model_path=tmp_path / "model",
            run_directory=run,
            physical_gpu=1,
            nvidia_smi=str(nvidia_smi),
        )


def test_operator_records_sampler_start_failure_without_launching_child(tmp_path: Path) -> None:
    spec = _spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(canonical_json(spec.to_value()))
    nvidia_smi = tmp_path / "nvidia-smi"
    _write_failing_nvidia_smi(nvidia_smi, spec.expected_physical_gpu_uuid)
    child = tmp_path / "must-not-run.py"
    child.write_text("raise RuntimeError('child launched')\n")
    run = tmp_path / "run"

    terminal = run_gate_4c_once(
        python_executable=Path(sys.executable),
        gate_script=child,
        spec_path=spec_path,
        local_model_path=tmp_path / "model",
        run_directory=run,
        physical_gpu=1,
        nvidia_smi=str(nvidia_smi),
    )

    assert terminal.status is Gate4cOperatorStatus.FAILED_CONSUMED_NO_RETRY
    assert terminal.failure_code is Gate4cFailureCode.GPU_SAMPLER_FAILED
    assert terminal.child_pid is None
    assert not (run / "control" / "child.pid").exists()


def test_operator_passes_only_with_matching_success_child_terminal(tmp_path: Path) -> None:
    spec = _spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(canonical_json(spec.to_value()))
    nvidia_smi = tmp_path / "nvidia-smi"
    _write_fake_nvidia_smi(nvidia_smi, spec.expected_physical_gpu_uuid)
    child = tmp_path / "success.py"
    _write_success_child(child)

    terminal = run_gate_4c_once(
        python_executable=Path(sys.executable),
        gate_script=child,
        spec_path=spec_path,
        local_model_path=tmp_path / "model",
        run_directory=tmp_path / "run",
        physical_gpu=1,
        nvidia_smi=str(nvidia_smi),
    )

    assert terminal.status is Gate4cOperatorStatus.PASSED
    assert terminal.failure_code is None
    assert terminal.child_exit_code == 0
    assert terminal.child_terminal_present


def test_public_terminals_exclude_private_failure_details(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _operator_directories(run)
    with pytest.raises(InjectedStageError):
        run_child(
            spec_path=tmp_path / "private-spec.json",
            model_path=tmp_path / "private-model",
            work=run,
            execute=_failing_execute(Gate4cStage.BUNDLE_VERIFIED),
        )
    public = (run / "public" / "child-terminal.json").read_text()
    assert str(tmp_path) not in public
    assert "private-spec" not in public
    assert "private-model" not in public
    assert "InjectedStageError(" not in public
    assert (run / "private" / "traceback.txt").is_file()
