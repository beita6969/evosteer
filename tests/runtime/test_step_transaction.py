from __future__ import annotations

from pathlib import Path

import pytest

from skillev.runtime import (
    EventEnvelope,
    EventType,
    LiveAttemptEventLog,
    RuntimeEventEmitter,
    StepTransactionJournal,
    StepTransactionReconciler,
    StepTransactionState,
)
from skillev.runtime.event_log_reader import read_event_history


def _event(sequence: int) -> EventEnvelope:
    return EventEnvelope.create(
        event_type=EventType.TRAINING_STEP_COMMITTED,
        run_id="run",
        attempt_id="attempt",
        producer_id="training",
        producer_seq=sequence,
        occurred_at="2026-08-18T00:00:00Z",
        payload={"optimizer_step": sequence},
    )


def test_step_transaction_advances_exactly_and_survives_reopen(tmp_path: Path) -> None:
    journal = StepTransactionJournal((tmp_path / "transactions").resolve())
    record = journal.begin(
        optimizer_step=1,
        batch_id="batch-1",
        policy_snapshot_before="policy-0",
    )
    record = journal.advance(
        record,
        StepTransactionState.OPTIMIZER_APPLIED,
        policy_snapshot_after="policy-1",
    )
    record = journal.advance(record, StepTransactionState.PROJECTION_INSTALLED)
    record = journal.advance(record, StepTransactionState.EVOLUTION_RESOLVED)
    record = journal.advance(
        record,
        StepTransactionState.CHECKPOINT_PUBLISHED,
        checkpoint_name="step-00000001",
        source_events=(_event(1),),
    )
    record = journal.advance(
        record,
        StepTransactionState.ADAPTER_COMMITTED,
        adapter_revision="formal-step-1",
    )

    reopened = StepTransactionJournal(journal.directory)
    assert reopened.pending() == (record,)
    record = reopened.advance(record, StepTransactionState.SOURCE_EVENTS_PUBLISHED)
    record = reopened.advance(record, StepTransactionState.COMMITTED)
    assert reopened.pending() == ()
    assert reopened.load(1) == record


def test_step_transaction_rejects_skip_and_duplicate_begin(tmp_path: Path) -> None:
    journal = StepTransactionJournal((tmp_path / "transactions").resolve())
    record = journal.begin(
        optimizer_step=4,
        batch_id="batch-4",
        policy_snapshot_before="policy-3",
    )
    with pytest.raises(ValueError):
        journal.advance(record, StepTransactionState.PROJECTION_INSTALLED)
    with pytest.raises(FileExistsError):
        journal.begin(
            optimizer_step=4,
            batch_id="batch-4",
            policy_snapshot_before="policy-3",
        )


@pytest.mark.parametrize(
    "state",
    [
        StepTransactionState.PREPARED,
        StepTransactionState.OPTIMIZER_APPLIED,
        StepTransactionState.PROJECTION_INSTALLED,
        StepTransactionState.EVOLUTION_RESOLVED,
    ],
)
def test_pre_checkpoint_rollback_preserves_evidence_and_allows_replay(
    tmp_path: Path,
    state: StepTransactionState,
) -> None:
    journal = StepTransactionJournal((tmp_path / "transactions").resolve())
    record = journal.begin(
        optimizer_step=1,
        batch_id="batch-1",
        policy_snapshot_before="policy-0",
    )
    if state is not StepTransactionState.PREPARED:
        record = journal.advance(
            record,
            StepTransactionState.OPTIMIZER_APPLIED,
            policy_snapshot_after="policy-1",
        )
    if state in {
        StepTransactionState.PROJECTION_INSTALLED,
        StepTransactionState.EVOLUTION_RESOLVED,
    }:
        record = journal.advance(record, StepTransactionState.PROJECTION_INSTALLED)
    if state is StepTransactionState.EVOLUTION_RESOLVED:
        record = journal.advance(record, StepTransactionState.EVOLUTION_RESOLVED)

    rolled_back = journal.rollback_before_checkpoint(record)

    assert rolled_back.state is StepTransactionState.ROLLED_BACK
    assert journal.pending() == ()
    archives = tuple(journal.directory.glob("step-00000001.rolled-back-*.json"))
    assert len(archives) == 1
    replay = journal.begin(
        optimizer_step=1,
        batch_id="batch-1-replay",
        policy_snapshot_before="policy-0",
    )
    assert replay.state is StepTransactionState.PREPARED


def test_post_checkpoint_transaction_cannot_be_rolled_back(tmp_path: Path) -> None:
    journal = StepTransactionJournal((tmp_path / "transactions").resolve())
    record = journal.begin(
        optimizer_step=1,
        batch_id="batch-1",
        policy_snapshot_before="policy-0",
    )
    record = journal.advance(
        record,
        StepTransactionState.OPTIMIZER_APPLIED,
        policy_snapshot_after="policy-1",
    )
    record = journal.advance(record, StepTransactionState.PROJECTION_INSTALLED)
    record = journal.advance(record, StepTransactionState.EVOLUTION_RESOLVED)
    record = journal.advance(
        record,
        StepTransactionState.CHECKPOINT_PUBLISHED,
        checkpoint_name="step-00000001",
        source_events=(_event(1),),
    )

    with pytest.raises(ValueError):
        journal.rollback_before_checkpoint(record)


def test_step_transaction_does_not_delete_incomplete_staging(tmp_path: Path) -> None:
    directory = (tmp_path / "transactions").resolve()
    journal = StepTransactionJournal(directory)
    staging = directory / "step-00000001.json.staging"
    staging.write_text("incomplete", encoding="utf-8")
    with pytest.raises(FileExistsError):
        journal.begin(
            optimizer_step=1,
            batch_id="batch-1",
            policy_snapshot_before="policy-0",
        )
    assert staging.read_text(encoding="utf-8") == "incomplete"


def test_reconciler_finishes_checkpoint_adapter_and_source_events(tmp_path: Path) -> None:
    journal = StepTransactionJournal((tmp_path / "transactions").resolve())
    record = journal.begin(
        optimizer_step=1,
        batch_id="batch-1",
        policy_snapshot_before="policy-0",
    )
    record = journal.advance(
        record,
        StepTransactionState.OPTIMIZER_APPLIED,
        policy_snapshot_after="policy-1",
    )
    record = journal.advance(record, StepTransactionState.PROJECTION_INSTALLED)
    record = journal.advance(record, StepTransactionState.EVOLUTION_RESOLVED)
    event = _event(1)
    record = journal.advance(
        record,
        StepTransactionState.CHECKPOINT_PUBLISHED,
        checkpoint_name="step-00000001",
        source_events=(event,),
    )
    checkpoint_calls: list[str] = []
    adapter_calls: list[int] = []
    log = LiveAttemptEventLog(
        tmp_path / "events.jsonl",
        run_id="run",
        attempt_id="attempt",
    )
    reconciler = StepTransactionReconciler(
        journal=journal,
        emitter=RuntimeEventEmitter(log=log, producer_id="training"),
        require_checkpoint=lambda item: checkpoint_calls.append(item.checkpoint_name or ""),
        ensure_adapter=lambda item: adapter_calls.append(item.optimizer_step) or "formal-step-1",
    )

    committed = reconciler.reconcile(record)

    assert committed.state is StepTransactionState.COMMITTED
    assert checkpoint_calls == ["step-00000001"]
    assert adapter_calls == [1]
    assert read_event_history(log.path) == (event,)


def test_reconciler_refuses_step_without_checkpoint(tmp_path: Path) -> None:
    journal = StepTransactionJournal((tmp_path / "transactions").resolve())
    record = journal.begin(
        optimizer_step=1,
        batch_id="batch-1",
        policy_snapshot_before="policy-0",
    )
    log = LiveAttemptEventLog(
        tmp_path / "events.jsonl",
        run_id="run",
        attempt_id="attempt",
    )
    reconciler = StepTransactionReconciler(
        journal=journal,
        emitter=RuntimeEventEmitter(log=log, producer_id="training"),
        require_checkpoint=lambda _: None,
        ensure_adapter=lambda _: "unexpected",
    )

    with pytest.raises(RuntimeError):
        reconciler.reconcile(record)
