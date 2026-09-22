"""Resume, exact-denominator and actual-process isolation regressions."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from skillev.evaluation.actor_sandbox import ActorSandbox
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.integrity_metric_schema import NATIVE_VERIFIER_VERSIONS
from skillev.evaluation.integrity_pipeline import FrozenPanel, run_paired
from skillev.evaluation.integrity_results import NativeScore
from skillev.evaluation.integrity_resume import EvaluationRunMode
from skillev.evaluation.sealed_candidates import (
    CandidateJournal,
    CandidateReader,
    EventOrigin,
    FinalCandidate,
)
from skillev.evaluation.step0_integrity import InferenceArm, InterventionCounts
from tests.evaluation.test_step0_integrity_results import controls


class Runtime:
    def __init__(self, *, fail_score: bool = False, budget: int = 10) -> None:
        self.generated = self.graded = self.closed = 0
        self.fail_score, self.budget = fail_score, budget

    def controls(self, arm, entries):
        return replace(
            controls(),
            public_inputs=tuple((entry.task_id, entry.render()) for entry in entries),
            budgets={"output_tokens": self.budget},
        )

    def interventions(self, arm):
        return InterventionCounts(model_calls=self.generated)

    def validate_candidate(self, reader, entry, arm, run_id):
        # Pipeline scheduling fixture only; real provenance is exercised by the
        # isolated broker tests with persisted actual fake-transport outputs.
        final = reader.get(run_id, arm.arm_id, entry.task_id)
        if final.policy_id != "fixture-policy":
            raise ValueError("synthetic candidate belongs to a different policy")

    async def generate(self, entry, arm, run_id):
        self.generated += 1
        return FinalCandidate(
            run_id,
            arm.arm_id,
            entry.task_id,
            "fixture-policy",
            "owner-final",
            "42",
            "integer",
            10,
            2,
            {"model_calls": 1},
        )

    async def score(self, reader, scope, benchmark):
        assert reader.get(*scope).text == "42"
        self.graded += 1
        if self.fail_score and self.graded == 2:
            raise RuntimeError("synthetic grader outage")
        return NativeScore(
            scope[2],
            benchmark,
            "accuracy",
            1.0,
            verifier_version=NATIVE_VERIFIER_VERSIONS[benchmark],
        )

    async def close(self):
        self.closed += 1

    async def refresh(self):
        pass


def panel():
    return FrozenPanel(
        (PublicTaskView.from_record("one", "aime-2026", {"problem": "Synthetic task"}),),
        "nonfinal-fixture",
        "synthetic, no benchmark examples",
    )


def run(runtime, directory):
    return asyncio.run(
        run_paired(
            runtime,
            panel(),
            (InferenceArm("A1"), InferenceArm("A2")),
            run_id="fixture",
            directory=directory,
            canary=True,
            run_mode=EvaluationRunMode.SAME_RUN_RESUME
            if (directory / "frozen-plan-private.json").exists()
            else EvaluationRunMode.FORMAL_FRESH,
        )
    )


def test_resume_does_not_regenerate_regrade_or_lose_recorded_cost(tmp_path: Path) -> None:
    first = Runtime(fail_score=True)
    with pytest.raises(RuntimeError):
        run(first, tmp_path)
    assert first.generated == 2
    assert first.closed == 1
    resumed = Runtime()
    report = run(resumed, tmp_path)
    assert resumed.generated == 0
    assert resumed.graded == 1
    assert report["interventions"]["A1"]["model_calls"] == 1
    assert report["interventions"]["A2"]["model_calls"] == 1
    complete = Runtime()
    assert run(complete, tmp_path) == report
    assert complete.generated == 0
    assert complete.graded == 0


def test_resume_rejects_changed_actual_runtime_budget_before_any_model_call(tmp_path: Path) -> None:
    run(Runtime(), tmp_path)
    changed = Runtime(budget=20)
    with pytest.raises(ValueError):
        run(changed, tmp_path)
    assert changed.generated == 0
    assert changed.graded == 0
    assert changed.closed == 1


def test_real_runtime_change_detected_after_generation_prevents_scoring(tmp_path: Path) -> None:
    class ChangedRuntime(Runtime):
        async def refresh(self):
            self.budget += 1

    runtime = ChangedRuntime()
    with pytest.raises(RuntimeError):
        run(runtime, tmp_path)
    assert runtime.generated == 2
    assert runtime.graded == 0
    assert runtime.closed == 1


def test_coordinator_concurrency_is_an_observed_control(tmp_path: Path) -> None:
    class MismatchedRuntime(Runtime):
        def controls(self, arm, entries):
            value = super().controls(arm, entries)
            return replace(value, service={**value.service, "coordinator_concurrency": 3})

    runtime = MismatchedRuntime()
    with pytest.raises(ValueError):
        run(runtime, tmp_path)
    assert runtime.generated == runtime.graded == 0
    assert runtime.closed == 1


def test_bounded_dispatch_replenishes_work_without_exceeding_concurrency(tmp_path: Path) -> None:
    class ConcurrentRuntime(Runtime):
        active = peak = 0

        async def generate(self, entry, arm, run_id):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0)
                return await super().generate(entry, arm, run_id)
            finally:
                self.active -= 1

    runtime = ConcurrentRuntime()
    entries = tuple(
        PublicTaskView.from_record(str(index), "aime-2026", {"problem": "Synthetic task"})
        for index in range(5)
    )
    asyncio.run(
        run_paired(
            runtime,
            replace(panel(), entries=entries),
            (InferenceArm("A1"), InferenceArm("A2")),
            run_id="fixture",
            directory=tmp_path,
            concurrency=3,
            canary=True,
        )
    )
    assert runtime.generated == runtime.graded == 10
    assert runtime.peak == 3
    assert runtime.active == 0
    assert runtime.closed == 1


def test_external_cancellation_closes_all_active_actors(tmp_path: Path) -> None:
    class WaitingRuntime(Runtime):
        active = 0

        async def generate(self, entry, arm, run_id):
            self.active += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.active -= 1

        async def close(self):
            assert self.active == 0
            await super().close()

    runtime = WaitingRuntime()

    async def cancel():
        job = asyncio.create_task(
            run_paired(
                runtime,
                panel(),
                (InferenceArm("A1"), InferenceArm("A2")),
                run_id="fixture",
                directory=tmp_path,
                canary=True,
            )
        )
        async with asyncio.timeout(5):
            while runtime.active < 2:
                await asyncio.sleep(0)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job

    asyncio.run(cancel())
    assert runtime.graded == 0
    assert runtime.closed == 1


def test_generation_failure_preserves_active_answers_and_stops_admission(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FailedRuntime(Runtime):
        def __init__(self):
            super().__init__()
            self.peer_started = asyncio.Event()
            self.failed = asyncio.Event()
            self.peer_closed = False
            self.admitted = []

        async def generate(self, entry, arm, run_id):
            self.admitted.append((entry.task_id, arm.arm_id))
            if arm.arm_id == "A1":
                await self.peer_started.wait()
                self.failed.set()
                raise RuntimeError("synthetic serving outage")
            self.peer_started.set()
            try:
                await self.failed.wait()
                await asyncio.sleep(0)
                return await super().generate(entry, arm, run_id)
            finally:
                self.peer_closed = True

        async def close(self):
            assert self.peer_closed
            await super().close()

    runtime = FailedRuntime()
    with pytest.raises(RuntimeError):
        asyncio.run(
            run_paired(
                runtime,
                replace(
                    panel(),
                    entries=(
                        *panel().entries,
                        PublicTaskView.from_record("two", "aime-2026", {"problem": "Another"}),
                    ),
                ),
                (InferenceArm("A1"), InferenceArm("A2")),
                run_id="fixture",
                directory=tmp_path,
                concurrency=2,
                canary=True,
            )
        )
    assert runtime.admitted == [("one", "A1"), ("one", "A2")]
    assert runtime.closed == 1
    assert runtime.graded == 0
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert "failure_type" in progress[0]
    assert progress[0]["completed"] == 0
    assert progress[-1]["completed"] == 1
    assert all(row["eta_seconds"] is None for row in progress)
    reader = CandidateReader(tmp_path / "candidates-private.sqlite")
    try:
        assert reader.get("fixture", "A2", "one").text == "42"
    finally:
        reader.close()


@pytest.mark.parametrize("stage", ["environment-reset", "rendered-request"])
def test_interrupted_real_episode_cannot_receive_a_fresh_budget(tmp_path: Path, stage: str) -> None:
    from skillev_private.evaluation.integrity_runtime import PrivateIntegrityRuntime

    runtime = object.__new__(PrivateIntegrityRuntime)
    runtime.thinking_policy = None
    runtime.journal = CandidateJournal(tmp_path / "partial.sqlite")
    entry = panel().entries[0]
    runtime.journal.record(("run", "A1", entry.task_id), stage, {})
    try:
        with pytest.raises(RuntimeError):
            asyncio.run(runtime.generate(entry, InferenceArm("A1"), "run"))
    finally:
        runtime.journal.close()


def test_acknowledged_semantic_unknown_is_not_a_missing_transport_ack(tmp_path: Path) -> None:
    journal = CandidateJournal(tmp_path / "state.sqlite")
    scope = ("run", "arm", "case")
    journal.intent(scope, "first", "look", 1)
    journal.acknowledge(scope, "first", status="unknown", observation="Public observation returned")
    journal.intent(scope, "second", "look", 2)
    with pytest.raises(RuntimeError):
        journal.intent(scope, "third", "look", 3)
    journal.close()


def test_scorer_handle_cannot_replace_a_final_or_write_the_database(tmp_path: Path) -> None:
    journal = CandidateJournal(tmp_path / "state.sqlite")
    journal.seal(FinalCandidate("run", "arm", "case", "qwen", "owner-final", "42", "integer", 1, 1))
    reader = CandidateReader(tmp_path / "state.sqlite")
    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        reader.connection.execute("DELETE FROM candidates")
    reader.close()
    journal.close()


def test_communication_summary_detects_dropped_feedback_and_unmatched_execution(
    tmp_path: Path,
) -> None:
    from skillev.evaluation.integrity_communication_report import communication_summary
    from skillev.evaluation.step0_completion import native_action_constraint

    journal = CandidateJournal(tmp_path / "trace.sqlite")
    scope = ("run", "arm", "case")
    actions = ("search", "click[< Prev]", "click[Back to Search]")
    journal.record(
        scope,
        "action-surface",
        {
            "mode": "webshop",
            "native_actions": actions,
            "advertised_actions": actions,
            "tool_schema": native_action_constraint(
                "webshop", available_actions=actions
            ).json_schema,
        },
        origin=EventOrigin.ACTOR_DIAGNOSTIC,
    )
    journal.record(scope, "rendered-request", {}, origin=EventOrigin.ACTOR_DIAGNOSTIC)
    journal.record(scope, "model-transport-start", {}, origin=EventOrigin.MODEL_TRANSPORT)
    journal.record(scope, "environment-reset", {}, origin=EventOrigin.ENVIRONMENT)
    journal.record(scope, "model-response", {}, origin=EventOrigin.ACTOR_DIAGNOSTIC)
    journal.record(scope, "model-transport-complete", {}, origin=EventOrigin.MODEL_TRANSPORT)
    journal.intent(scope, "request", "click[< Prev]", 1)
    journal.acknowledge(scope, "request", status="unknown", observation="Public page")
    journal.record(scope, "environment-result", {}, origin=EventOrigin.ENVIRONMENT)
    failed = communication_summary(journal, (scope,))
    assert failed["status"] == "transport-defect"
    assert failed["unmatched_executions"] == 1
    assert failed["feedback_mismatches"] == 1
    assert failed["surface_mismatches"] == 0
    journal.record(
        scope,
        "decision-transport",
        {
            "result": {"action": "click[< Prev]"},
            "decision": {"state_revision": 1},
        },
        origin=EventOrigin.ACTOR_DIAGNOSTIC,
    )
    journal.record(
        scope,
        "public-transition",
        {"observation": "Public page"},
        origin=EventOrigin.ACTOR_DIAGNOSTIC,
    )
    assert communication_summary(journal, (scope,))["status"] == "complete"
    journal.close()


def test_actual_isolated_actor_sends_only_public_messages_through_fake_broker(
    tmp_path: Path,
) -> None:
    from skillev.rollout import RolloutGenerationResult
    from skillev.runtime import BudgetVector
    from tests.evaluation.test_step0_architecture import _profile

    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("real actor process test requires bubblewrap on the Linux CPU host")
    sandbox = replace(
        ActorSandbox.current(Path(__file__).resolve().parents[2] / "src"), bubblewrap=Path(bwrap)
    )
    initial = {
        "public_task": asdict(panel().entries[0]),
        "run_id": "fixture",
        "arm": InferenceArm("A1").to_value(),
        "decoding": asdict(_profile()),
        "policy": {
            "snapshot_id": "fixture-policy",
            "backbone_id": "Qwen3.5-9B",
            "tokenizer_id": "fixture-tokenizer",
        },
        "population_id": "synthetic",
        "panel_position": 0,
        "budgets": {
            "calls_per_turn": 4,
            "total_model_calls": 4,
            "total_output_tokens": 256,
            "context_length": 8192,
            "history_input_tokens": 4096,
            "skill_instruction_tokens": 0,
        },
    }
    operations = []

    async def handle(message):
        operations.append(message["operation"])
        assert "private-rubric" not in json.dumps(message)
        if message["operation"] == "trace":
            return None
        if message["operation"] == "encode":
            return [ord(c) for c in message["text"]]
        if message["operation"] == "encode-messages":
            return [ord(c) for c in json.dumps(message["messages"])]
        if message["operation"] == "decode":
            return "".join(chr(token) for token in message["token_ids"])
        assert message["operation"] == "generate"
        return RolloutGenerationResult(
            content_token_ids=(52, 50),
            stop_token_ids=(),
            finish_reason="stop",
            policy_snapshot_id="fixture-policy",
            backend_id="fake-public-broker",
            usage=BudgetVector(
                input_tokens=len(message["request"]["input_ids"]), output_tokens=2, model_calls=1
            ),
        ).to_value()

    raw = asyncio.run(
        sandbox.run(initial, handle, stderr_path=tmp_path / "actor.log", timeout_seconds=30)
    )
    assert raw["text"] == r"\boxed{42}"
    assert raw["intervention_counts"]["model_calls"] == 1
    assert operations.count("generate") == 1


def test_real_actor_namespace_cannot_read_hidden_files_host_environment_or_network(
    tmp_path: Path,
) -> None:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip(
            "actual namespace validation runs on the approved Linux CPU host with bubblewrap"
        )
    hidden = tmp_path / "private-rubrics.json"
    hidden.write_text("synthetic evaluator secret")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    script = f"""
import json, os, socket
from pathlib import Path
result = {{
    'file_visible': Path({str(hidden)!r}).exists(),
    'env': os.environ.get('PRIVATE_EVALUATOR_SECRET'),
    'host_process': Path('/proc/{os.getpid()}/environ').exists(),
}}
try:
    socket.create_connection(('127.0.0.1', {port}), timeout=1).close()
    result['host_network'] = True
except OSError:
    result['host_network'] = False
print(json.dumps(result))
"""
    sandbox = replace(
        ActorSandbox.current(Path(__file__).resolve().parents[2] / "src"), bubblewrap=Path(bwrap)
    )
    try:
        completed = subprocess.run(  # noqa: S603 -- exercise the actual isolation boundary
            sandbox.command("-c", script),
            env={
                **os.environ,
                "PRIVATE_EVALUATOR_SECRET": "synthetic evaluator secret",
                "CUDA_VISIBLE_DEVICES": "",
            },
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    finally:
        listener.close()
    observed = json.loads(completed.stdout)
    assert observed["file_visible"] is False
    assert observed["env"] is None
    assert observed["host_process"] is False
    assert observed["host_network"] is False
    assert "synthetic evaluator secret" not in completed.stdout


def test_interpreter_alias_chain_survives_private_tmp_namespace(tmp_path: Path) -> None:
    import sys

    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("interpreter namespace test requires bubblewrap")
    alias = tmp_path / "cpython-alias"
    alias.symlink_to(Path(sys.base_prefix), target_is_directory=True)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    executable = Path(sys.executable).resolve().name
    (venv / "bin/python").symlink_to("python3")
    (venv / "bin/python3").symlink_to(alias / "bin" / executable)
    (venv / "pyvenv.cfg").write_text(
        f"home = {Path(sys.base_prefix) / 'bin'}\ninclude-system-site-packages = false\n"
    )
    sandbox = ActorSandbox(
        Path(__file__).resolve().parents[2] / "src",
        venv / "bin/python",
        venv,
        Path(sys.base_prefix),
        Path(bwrap),
    )
    completed = subprocess.run(  # noqa: S603 -- synthetic interpreter alias, no model execution
        sandbox.command("-c", "import sys; print(sys.prefix)"),
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
        env={"CUDA_VISIBLE_DEVICES": ""},
    )
    assert completed.stdout.strip() == str(venv)


def test_companion_dependencies_are_mounted_only_when_explicitly_declared(tmp_path: Path) -> None:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("dependency namespace test requires bubblewrap")
    dependency = tmp_path / "python-dependencies"
    dependency.mkdir()
    (dependency / "companion_dependency.py").write_text("value = 'public dependency'\n")
    sandbox = replace(
        ActorSandbox.current(Path(__file__).resolve().parents[2] / "src"),
        bubblewrap=Path(bwrap),
    )
    script = (
        f"import sys; sys.path.insert(0, {str(dependency)!r}); "
        "import companion_dependency; print(companion_dependency.value)"
    )
    for visible in (False, True):
        selected = replace(sandbox, dependency_prefixes=(dependency,) if visible else ())
        completed = subprocess.run(  # noqa: S603 -- synthetic dependency, no benchmark content
            selected.command("-c", script),
            capture_output=True,
            text=True,
            timeout=15,
            env={"CUDA_VISIBLE_DEVICES": ""},
        )
        assert (completed.returncode == 0) is visible
        if visible:
            assert completed.stdout.strip() == "public dependency"
