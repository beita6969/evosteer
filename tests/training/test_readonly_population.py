"""Read-only population size/chunks never change optimizer B or sampling coordinates."""

import asyncio
import json

import pytest

from skillev.rollout.readonly_collection import collect_readonly_panel
from skillev.runtime import BudgetLedger, LiveAttemptEventLog, RuntimeEventEmitter
from skillev.training.rollout_workflow import RolloutWorkflowBinding, RolloutWorkflowResources
from tests.training.fakes import FakeISOClock, OrderedSessionFactory, make_public_tasks


@pytest.mark.parametrize(("count", "chunk_size"), [(35, 8), (224, 29)])
def test_population_is_not_a_training_batch(make_training_harness, tmp_path, count, chunk_size):
    tasks = make_public_tasks(count)
    harness = make_training_harness(tasks=tasks)
    before = harness.projections.runtime_state()
    sessions = OrderedSessionFactory((1.0,) * count)
    root = tmp_path / "iid-evaluation"
    events = LiveAttemptEventLog(tmp_path / "eval-events.jsonl", run_id="eval", attempt_id="eval")
    result = asyncio.run(
        collect_readonly_panel(
            root=root,
            tasks=tasks,
            generator=harness.generator,
            sessions=sessions,
            library=harness.library.state,
            rollout=harness.config.rollout,
            assembler=harness.config.rollout.context_assembler(maximum_h0_tokens=2048),
            epsilon_min=harness.config.method.epsilon_min,
            condition_id="same-architecture",
            sampling_schedule_id="fixed-evaluation-seed0",
            ordered_sequence_id="ordered-sources",
            resources=RolloutWorkflowResources(RolloutWorkflowBinding()),
            ledger=BudgetLedger(
                run_id="eval",
                attempt_id="eval",
                cap=harness.config.rollout.per_rollout_maximum.scale(count),
            ),
            emitter=RuntimeEventEmitter(events, "eval"),
            clock=FakeISOClock(),
            chunk_size=chunk_size,
        )
    )
    assert [(row.position, row.task_id) for row in result.outcomes] == list(
        enumerate(task.task_id for task in tasks)
    )
    assert len(result.artifacts) == count
    assert sessions.cleanup_count == count
    assert harness.config.execution.batch_size == 2  # no fake panel-sized training config
    assert harness.task_provider.cursor == harness.loop.optimizer_step == 0
    assert harness.projections.runtime_state() == before
    assert not (root / "checkpoints").exists()
    isolation = json.loads((root / "isolation.json").read_text())
    assert all(
        isolation[key] == 0
        for key in (
            "training_updates",
            "posterior_updates",
            "skill_evolution",
            "training_evidence_writes",
        )
    )
    coordinates = [
        artifact.manifest.sampling_coordinate.sequence_position for artifact in result.artifacts
    ]
    assert coordinates == list(range(count))


def test_preparation_failure_keeps_panel_denominator_without_false_rewards(
    make_training_harness, tmp_path
):
    harness = make_training_harness()
    tasks = harness.task_provider.tasks[:5]

    class BrokenSessions:
        async def prepare_tasks(self, _):
            raise TimeoutError("native environment unavailable")

        def create(self, _):
            raise AssertionError("must not attempt an unprepared environment")

    log = LiveAttemptEventLog(tmp_path / "eval-events.jsonl", run_id="eval", attempt_id="eval")
    result = asyncio.run(
        collect_readonly_panel(
            root=tmp_path / "eval",
            tasks=tasks,
            generator=harness.generator,
            sessions=BrokenSessions(),
            library=harness.library.state,
            rollout=harness.config.rollout,
            assembler=harness.config.rollout.context_assembler(maximum_h0_tokens=2048),
            epsilon_min=0.1,
            condition_id="test",
            sampling_schedule_id="frozen",
            ordered_sequence_id="frozen",
            resources=RolloutWorkflowResources(RolloutWorkflowBinding()),
            ledger=BudgetLedger(
                run_id="eval",
                attempt_id="eval",
                cap=harness.config.rollout.per_rollout_maximum.scale(len(tasks)),
            ),
            emitter=RuntimeEventEmitter(log, "eval"),
            clock=FakeISOClock(),
            chunk_size=2,
        )
    )
    assert len(result.outcomes) == len(tasks)
    assert all(row.artifact is None and row.infrastructure_error for row in result.outcomes)
    assert result.consumed_budget.model_calls == 0
    with pytest.raises(ValueError):
        _ = result.artifacts


def _failure_collection(harness, tmp_path, sessions):
    tasks = harness.task_provider.tasks[:5]
    log = LiveAttemptEventLog(tmp_path / "eval-events.jsonl", run_id="eval", attempt_id="eval")
    return collect_readonly_panel(
        root=tmp_path / "eval",
        tasks=tasks,
        generator=harness.generator,
        sessions=sessions,
        library=harness.library.state,
        rollout=harness.config.rollout,
        assembler=harness.config.rollout.context_assembler(maximum_h0_tokens=2048),
        epsilon_min=0.1,
        condition_id="test",
        sampling_schedule_id="frozen",
        ordered_sequence_id="frozen",
        resources=RolloutWorkflowResources(RolloutWorkflowBinding()),
        ledger=BudgetLedger(
            run_id="eval",
            attempt_id="eval",
            cap=harness.config.rollout.per_rollout_maximum.scale(len(tasks)),
        ),
        emitter=RuntimeEventEmitter(log, "eval"),
        clock=FakeISOClock(),
        chunk_size=1,
    )


@pytest.mark.parametrize("stage", ["preparation", "episode"])
@pytest.mark.parametrize("kind", ["infrastructure", "bug", "configuration", "cancelled"])
def test_only_explicit_infrastructure_becomes_unknown(
    make_training_harness, tmp_path, monkeypatch, stage, kind
):
    from skillev_private.benchmarks.official_process import OfficialEnvironmentInfrastructureError

    from skillev.rollout import readonly_collection

    error = {
        "infrastructure": OfficialEnvironmentInfrastructureError("worker unavailable"),
        "bug": AssertionError("implementation bug"),
        "configuration": ValueError("condition mismatch"),
        "cancelled": asyncio.CancelledError(),
    }[kind]
    calls = []

    async def fail(*args, **kwargs):
        calls.append(stage)
        raise error

    class Sessions:
        async def prepare_tasks(self, tasks):
            if stage == "preparation":
                await fail()
            return tasks

        def create(self, task):
            raise AssertionError("the mocked episode must not create a session")

    monkeypatch.setattr(readonly_collection, "execute_episode", fail)
    harness = make_training_harness()
    collect = _failure_collection(harness, tmp_path, Sessions())
    root = tmp_path / "eval"
    if kind == "infrastructure":
        result = asyncio.run(collect)
        assert len(result.outcomes) == len(harness.task_provider.tasks[:5])
        assert all(row.artifact is None for row in result.outcomes)
        assert result.consumed_budget.model_calls == 0
        persisted = [json.loads(p.read_text()) for p in sorted(root.glob("episode-*.json"))]
        assert len(persisted) == len(result.outcomes)
        assert all(row["artifact"] is None for row in persisted)
        assert all("reward" not in row and "success" not in row for row in persisted)
        failure = json.loads(next(root.glob("failure-*.json")).read_text())
        assert failure["message"] == "worker unavailable"
        assert "OfficialEnvironmentInfrastructureError" in failure["traceback"]
        assert failure["status"] == "infrastructure-error"
        if stage == "preparation":
            assert all(row["execution_status"] == "not-started" for row in persisted)
            assert failure["chunk_start"] == 0
            assert failure["originating_task_id"] is None
            assert "task_id" not in failure
    else:
        with pytest.raises(type(error)):
            asyncio.run(collect)
        failure = json.loads(next(root.glob("failure-*.json")).read_text())
        assert failure["stage"] == stage
        assert failure["error_type"] == type(error).__name__
        assert failure["native_label"] is None
        assert failure["status"] == ("cancelled" if kind == "cancelled" else "unexpected-error")
        assert not list(root.glob("episode-*.json"))
    assert calls == [stage]  # No replacement and no later chunk after failure.
    assert json.loads((root / "rollout-progress.json").read_text())["status"] == "failed"
    assert len(json.loads((root / "plan-private.json").read_text())["tasks"]) == len(
        harness.task_provider.tasks[:5]
    )


def test_hydration_population_change_propagates(make_training_harness, tmp_path):
    class WrongPopulation:
        async def prepare_tasks(self, tasks):
            return ()

        def create(self, task):
            raise AssertionError("must reject population mismatch before execution")

    with pytest.raises(ValueError):
        asyncio.run(_failure_collection(make_training_harness(), tmp_path, WrongPopulation()))
    failure = json.loads(next((tmp_path / "eval").glob("failure-*.json")).read_text())
    assert failure["stage"] == "preparation"
    assert failure["status"] == "unexpected-error"
