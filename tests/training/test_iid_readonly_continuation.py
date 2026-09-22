"""Synthetic CPU episodes; safe preparation-only continuation never resamples a prefix."""

import asyncio
import json
import sqlite3

import pytest
from skillev_private.evaluation.iid_episode_runtime import (
    continuation_prefix,
    copy_continuation_journal,
    load_iid_continuation,
)

from skillev.rollout.readonly_collection import collect_readonly_panel
from skillev.runtime import BudgetLedger, LiveAttemptEventLog, RuntimeEventEmitter
from skillev.runtime.request_journal import DurableRequestJournal
from skillev.training.inflight import durable_json
from skillev.training.rollout_workflow import RolloutWorkflowBinding, RolloutWorkflowResources
from tests.training.fakes import FakeISOClock, OrderedSessionFactory


def collect(harness, root, sessions, continuation=None, chunk_size=2):
    root.mkdir()
    tasks = harness.task_provider.tasks[:5]
    ledger = BudgetLedger(
        run_id="original-arm",
        attempt_id="evaluation-only",
        cap=harness.config.rollout.per_rollout_maximum.scale(len(tasks)),
    )
    events = LiveAttemptEventLog(
        root / "events-private.jsonl", run_id=ledger.run_id, attempt_id=ledger.attempt_id
    )
    result = asyncio.run(
        collect_readonly_panel(
            root=root / "episodes",
            tasks=tasks,
            generator=harness.generator,
            sessions=sessions,
            library=harness.library.state,
            rollout=harness.config.rollout,
            assembler=harness.config.rollout.context_assembler(maximum_h0_tokens=2048),
            epsilon_min=0.1,
            condition_id="fixed",
            sampling_schedule_id="fixed-seed0",
            ordered_sequence_id="same-order",
            resources=RolloutWorkflowResources(RolloutWorkflowBinding()),
            ledger=ledger,
            emitter=RuntimeEventEmitter(events, "iid-evaluation"),
            clock=FakeISOClock(),
            chunk_size=chunk_size,
            continuation=continuation,
        )
    )
    durable_json(
        root / "expanded-controls-private.json", {"arm": "skills-off", "controls": {"fixed": True}}
    )
    durable_json(
        root / "summary.json",
        {
            "consumed_budget": result.consumed_budget.to_value(),
            "elapsed_seconds": result.elapsed_seconds,
        },
    )
    return result, ledger


class PreparationFailure(OrderedSessionFactory):
    def __init__(self):
        super().__init__((1.0,) * 5)
        self.prepared = 0

    async def prepare_tasks(self, tasks):
        self.prepared += 1
        if self.prepared == 2:
            from skillev_private.benchmarks.official_process import (
                OfficialEnvironmentInfrastructureError,
            )

            raise OfficialEnvironmentInfrastructureError("synthetic hydration: original diagnostic")
        return tasks


def prepared(harness, source):
    result, ledger = collect(harness, source, PreparationFailure())
    DurableRequestJournal(source.parent / f"{source.name}-evaluation-requests.sqlite3")
    expanded = json.loads((source / "expanded-controls-private.json").read_text())
    return result, ledger, expanded


def test_continuation_preserves_prefix_bytes_budget_and_original_coordinates(
    make_training_harness, tmp_path
):
    harness = make_training_harness()
    source = tmp_path / "original-arm"
    original, old_ledger, expanded = prepared(harness, source)
    saved = [(source / "episodes" / f"episode-{i:06d}-private.json").read_bytes() for i in range(2)]
    continuation, run_id, attempt = load_iid_continuation(
        source, expanded=expanded, chunk_size=2, tokenizer=harness.generator.tokenizer
    )
    assert run_id == "original-arm"
    assert attempt == "evaluation-only"
    assert len(continuation.outcomes) == 2
    target_journal = tmp_path / "new-evaluation-requests.sqlite3"
    copy_continuation_journal(source, target_journal)
    sessions = OrderedSessionFactory((1.0,) * 3)
    result, new_ledger = collect(harness, tmp_path / "new", sessions, continuation)
    assert sessions.cleanup_count == 3
    assert len(result.artifacts) == 5
    old_ids = {e.reservation.reservation_id for e in old_ledger.entries}
    assert len([e for e in new_ledger.entries if e.reservation.reservation_id in old_ids]) == len(
        old_ids
    )
    for i in range(2):
        assert (
            tmp_path / "new" / "episodes" / f"episode-{i:06d}-private.json"
        ).read_bytes() == saved[i]
        assert (source / "episodes" / f"episode-{i:06d}-private.json").read_bytes() == saved[i]
    assert result.consumed_budget.model_calls > original.consumed_budget.model_calls
    assert [a.manifest.sampling_coordinate.sequence_position for a in result.artifacts] == list(
        range(5)
    )
    assert all(
        a.record.trajectory_id == f"episodes-eval-{i:06d}" for i, a in enumerate(result.artifacts)
    )
    assert json.loads((source / "episodes/rollout-progress.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("change", ["controls", "chunk", "started", "pending", "suffix-request"])
def test_unsafe_or_changed_continuation_refused(make_training_harness, tmp_path, change):
    harness = make_training_harness()
    source = tmp_path / "original-arm"
    _, _, expanded = prepared(harness, source)
    chunk = 2
    if change == "controls":
        expanded["controls"]["fixed"] = False
    elif change == "chunk":
        chunk = 1
    elif change == "started":
        durable_json(source / "episodes/started-episode-000002.json", {"position": 2})
    elif change in {"pending", "suffix-request"}:
        path = source.parent / f"{source.name}-evaluation-requests.sqlite3"
        identity = ["episodes-eval-000000" if change == "pending" else "episodes-eval-000002", "1"]
        with sqlite3.connect(path) as db:
            db.execute(
                "INSERT INTO requests VALUES (?,?,?,?,?,?)",
                (
                    json.dumps(identity),
                    "synthetic",
                    b"{}",
                    "DISPATCHED" if change == "pending" else "COMPLETED",
                    200,
                    b"{}",
                ),
            )
        with pytest.raises(ValueError):
            copy_continuation_journal(source, tmp_path / "copy.sqlite3")
        assert not (tmp_path / "copy.sqlite3").exists()
        return
    with pytest.raises(ValueError):
        load_iid_continuation(
            source, expanded=expanded, chunk_size=chunk, tokenizer=harness.generator.tokenizer
        )


def test_legacy_budget_restores_original_events_not_summary_charges(
    make_training_harness, tmp_path
):
    harness = make_training_harness()
    source = tmp_path / "original-arm"
    original, old_ledger, expanded = prepared(harness, source)
    (source / "episodes/ledger-private.json").unlink()
    plan_path = source / "episodes/plan-private.json"
    plan = json.loads(plan_path.read_text())
    plan.pop("chunk_size")
    durable_json(plan_path, plan)
    request = tmp_path / "original-request.json"
    durable_json(request, {"arm": "skills-off", "chunk_size": 2})
    restored, _, _ = load_iid_continuation(
        source,
        expanded=expanded,
        chunk_size=2,
        tokenizer=harness.generator.tokenizer,
        original_request=request,
    )
    assert {e.reservation.reservation_id for e in restored.entries} == {
        e.reservation.reservation_id for e in old_ledger.entries
    }
    assert (
        sum(e.settlement.actual.model_calls for e in restored.entries)
        == original.consumed_budget.model_calls
    )
    with pytest.raises(ValueError):
        load_iid_continuation(
            source, expanded=expanded, chunk_size=2, tokenizer=harness.generator.tokenizer
        )


def test_originating_preparation_message_is_not_assigned_to_unstarted_rows(
    make_training_harness, tmp_path
):
    source = tmp_path / "original-arm"
    prepared(make_training_harness(), source)
    rows, start = continuation_prefix(source)
    assert start == 2
    assert all(row["execution_status"] == "not-started" for row in rows[start:])
    assert all(row["preparation_chunk_start"] == start for row in rows[start:])
    assert all(row["originating_failure_position"] is None for row in rows[start:])
    error = json.loads((source / "episodes/failure-preparation-000002-private.json").read_text())
    assert error["originating_task_id"] is None
    assert error["chunk_start"] == start
    assert "task_id" not in error
    assert "synthetic hydration" in error["message"]
    assert "OfficialEnvironmentInfrastructureError" in error["traceback"]
