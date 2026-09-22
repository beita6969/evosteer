import asyncio
import json
import shutil
from pathlib import Path

import pytest
from skillev_private.experiments.bayesian_branch import branch_checkpoint

from skillev.runtime import StepTransactionJournal, StepTransactionState
from skillev.training.checkpoint import FilesystemTrainingCheckpointStore
from tests.application.test_full_vertical_loop import build_application_fixture


@pytest.fixture
def source(tmp_path, training_backbone_config):
    root = tmp_path / "original/v3-application"
    fixture = build_application_fixture(root, training_backbone_config, cycles=2)
    asyncio.run(
        fixture.application.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=3)
    )
    shutil.copytree(root / "snapshots", root / "checkpoints")
    declarations = {
        "formal-config.json": {"condition": "synthetic-storage-fixture"},
        "run-clock.json": {"elapsed_seconds": 1},
        "resolved-run-plan.json": fixture.run_plan.to_value(),
        "effective-condition-process-fixture.json": {"scientific": {"data": "synthetic"}},
    }
    for name, value in declarations.items():
        (root / name).write_text(json.dumps(value, indent=2) + "\n")
    for name in ("quality", "inflight"):
        (root / name).mkdir()
        (root / name / "old-state.json").write_text('{"fixture":"not-branch-input"}\n')
    return root


def file_bytes(directory):
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_step_two_branch_copies_only_committed_prefix_without_mutating_step_three(source, tmp_path):
    snapshot = source / "checkpoints/step-00000002"
    original_events = (source / "events.jsonl").read_bytes()
    original_step_three = file_bytes(source / "checkpoints/step-00000003")
    original_journals = file_bytes(source / "checkpoints/step-transactions")
    target = tmp_path / "new-condition/v3-application"
    copied = branch_checkpoint(source_root=source, snapshot=snapshot, target_root=target)
    assert copied == target / "checkpoints/step-00000002"
    assert file_bytes(copied) == file_bytes(snapshot)
    store = FilesystemTrainingCheckpointStore(root=copied.parent)
    before = FilesystemTrainingCheckpointStore(root=snapshot.parent).load_metadata(snapshot)
    after = store.load_metadata(copied)
    assert after.experiment_id == before.experiment_id == "v3-application"
    assert after.execution_state.task_cursor == before.execution_state.task_cursor
    assert after.execution_state.task_cursor.curriculum_id == "test-public-tasks@3"
    assert after.identity == before.identity
    assert after.optimizer_step == 2
    journal = StepTransactionJournal(target / "checkpoints/step-transactions")
    assert sorted(path.name for path in journal.directory.iterdir()) == [
        "step-00000001.json",
        "step-00000002.json",
    ]
    for step in (1, 2):
        name = f"step-{step:08d}.json"
        assert (journal.directory / name).read_bytes() == original_journals[Path(name)]
        assert journal.load(step).state is StepTransactionState.COMMITTED
    prefix = (target / "events.jsonl").read_bytes()
    assert original_events.startswith(prefix)
    assert len(prefix) < len(original_events)
    events = [json.loads(line) for line in prefix.splitlines()]
    assert [
        event["payload"]["optimizer_step"]
        for event in events
        if event["event_type"] == "training_step_committed"
    ] == [1, 2]
    event_ids = {event["event_id"] for event in events}
    source_journal = StepTransactionJournal(source / "checkpoints/step-transactions")
    assert not event_ids.intersection(
        event.event_id for event in source_journal.load(3).source_events
    )
    assert {
        event.event_id for step in (1, 2) for event in journal.load(step).source_events
    } <= event_ids
    for name in ("quality", "inflight"):
        assert not (target / name).exists()
        assert (source / name / "old-state.json").is_file()
    for name in (
        "formal-config.json",
        "run-clock.json",
        "resolved-run-plan.json",
        "effective-condition-process-fixture.json",
    ):
        assert (target / name).read_bytes() == (source / name).read_bytes()
    declaration = json.loads((target / "branch-source.json").read_text())
    assert declaration["optimizer_step"] == 2
    assert declaration["metrics_start_step"] == 3
    assert declaration["source_checkpoint"] == str(snapshot)
    assert declaration["sampling_condition"] == before.identity.sampling_schedule_algorithm
    assert not (target / "checkpoints/step-00000003").exists()
    assert (source / "events.jsonl").read_bytes() == original_events
    assert file_bytes(source / "checkpoints/step-00000003") == original_step_three
    assert file_bytes(source / "checkpoints/step-transactions") == original_journals


@pytest.mark.parametrize("missing", ["complete", "journal", "uncommitted"])
def test_branch_rejects_incomplete_source_before_creating_target(source, tmp_path, missing):
    snapshot = source / "checkpoints/step-00000002"
    journal_path = source / "checkpoints/step-transactions/step-00000002.json"
    if missing == "complete":
        (snapshot / "COMPLETE").unlink()
    elif missing == "journal":
        journal_path.unlink()
    else:
        value = json.loads(journal_path.read_text())
        value["state"] = StepTransactionState.SOURCE_EVENTS_PUBLISHED.value
        journal_path.write_text(json.dumps(value))
    original_events = (source / "events.jsonl").read_bytes()
    target = tmp_path / "new-condition/v3-application"
    with pytest.raises((ValueError, FileNotFoundError)):
        branch_checkpoint(source_root=source, snapshot=snapshot, target_root=target)
    assert not target.exists()
    assert (source / "events.jsonl").read_bytes() == original_events
