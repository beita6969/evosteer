from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from skillev.application import SKILLEVApplication, TerminalComponents
from skillev.application_recovery import require_pending_step_matches
from skillev.application_snapshot import require_full_execution_coherence
from skillev.contracts import TrainingStepCommit
from skillev.evolution import PhiBudgetAuthority
from skillev.evolution.authoring_journal import UnresolvedAuthoringCallError
from skillev.policy import PrivateInitialCheckpointBinding
from skillev.runtime import (
    BudgetLedger,
    EventType,
    StepTransactionState,
)
from skillev.runtime.event_log_reader import read_event_history
from skillev.training import PrivateCheckpointStorageBinding
from skillev.training.checkpoint import FilesystemTrainingCheckpointStore
from tests.application.test_full_vertical_loop import (
    _BaseSessionFactory,
    _optimizer_steps,
    _ScriptedQwenBackbone,
    _StepPublisher,
    build_application_fixture,
)
from tests.training.fakes import FakeISOClock, OrderedTaskProvider


class InjectedInterruptionError(RuntimeError):
    pass


def restore_fixture(
    fixture,
    directory: Path,
    backbone_config,
    *,
    backbone=None,
    public_identity=None,
    continuation=None,
    reasoning_continuation=None,
    skill_exposure_continuation=None,
    domain_subset_continuation=None,
    healthbench_judge_continuation=None,
    format_review_continuation=None,
):
    store = FilesystemTrainingCheckpointStore(root=directory.parent)
    snapshot = store.load_metadata(directory)
    state = snapshot.execution_state
    require_full_execution_coherence(state, snapshot.optimizer_step)
    if backbone is None:
        backbone = _ScriptedQwenBackbone(backbone_config)
    publisher = _StepPublisher([], [], [])

    class TaskFactory:
        def from_exact_state(self, cursor):
            provider = OrderedTaskProvider(fixture.tasks, cursor=cursor.cursor)
            assert provider.runtime_state == cursor
            return provider

    # Exercise the public resume entrypoint, including automatic transaction
    # reconciliation, rather than duplicating its hydration in the test.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("skillev.application.build_qwen_policy_backbone", lambda _config: backbone)
        application = SKILLEVApplication.resume(
            snapshot_directory=directory,
            backbone_config=backbone_config,
            initial_checkpoint=PrivateInitialCheckpointBinding(
                directory=str(fixture.initial_policy_directory),
                trainable_state=fixture.initial_trainable_state,
            ),
            task_provider_factory=TaskFactory(),
            base_session_factory=_BaseSessionFactory(
                fixture.rewards, cursor=state.task_cursor.cursor
            ),
            terminal_components=TerminalComponents(
                ledger=BudgetLedger(
                    run_id="v3-application", attempt_id="attempt-1", cap=fixture.attempt_budget
                ),
                authoring_authority=fixture.authority,
                phi_budget=PhiBudgetAuthority(cap=fixture.phi_maximum),
            ),
            checkpoint_storage=PrivateCheckpointStorageBinding(directory=str(directory.parent)),
            public_identity=fixture.public_identity if public_identity is None else public_identity,
            horizon_continuation=continuation,
            reasoning_continuation=reasoning_continuation,
            skill_exposure_continuation=skill_exposure_continuation,
            domain_subset_continuation=domain_subset_continuation,
            healthbench_judge_continuation=healthbench_judge_continuation,
            format_review_continuation=format_review_continuation,
            step_transaction_journal=fixture.journal,
            event_log=fixture.event_log,
            clock=FakeISOClock(),
            step_adapter_publisher_factory=lambda _backbone: publisher,
        )
    actual_optimizer = application.training_loop.optimizer.state_dict()
    saved_optimizer = torch.load(directory / snapshot.optimizer_file, weights_only=True)
    assert actual_optimizer["param_groups"] == saved_optimizer["param_groups"]
    torch.testing.assert_close(actual_optimizer["state"], saved_optimizer["state"], rtol=0, atol=0)
    assert application.projections.runtime_state() == state.projections
    assert application.library.state == state.library
    assert application.training_loop.task_provider.runtime_state == state.task_cursor
    assert not fixture.journal.pending()
    assert (
        application.evolution_loop.step_transaction_journal.directory == fixture.journal.directory
    )
    return application, publisher


@pytest.mark.parametrize(
    "stage",
    [
        "prepared",
        "optimizer-applied",
        "projection-installed",
        "evolution-resolved",
        "checkpoint-before-journal",
        "checkpoint-published",
        "adapter-committed",
        "source-events-published",
        "committed",
        "library-before-projection",
        "authoring",
    ],
)
def test_each_interruption_recovers_one_model_posterior_library_and_adapter_commit(
    tmp_path,
    training_backbone_config,
    monkeypatch,
    stage,
) -> None:
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    app = fixture.application
    asyncio.run(app.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1))
    initial_library = app.library.current_version
    journal = fixture.journal
    original_advance = journal.advance
    original_begin = journal.begin

    def begin(**kwargs):
        record = original_begin(**kwargs)
        if stage == "prepared" and record.optimizer_step == 2:
            raise InjectedInterruptionError(stage)
        return record

    def advance(record, state, **kwargs):
        if (
            stage == "checkpoint-before-journal"
            and state is StepTransactionState.CHECKPOINT_PUBLISHED
        ):
            raise InjectedInterruptionError(stage)
        updated = original_advance(record, state, **kwargs)
        if updated.optimizer_step == 2 and state.value == stage:
            raise InjectedInterruptionError(stage)
        return updated

    def fail(*args, **kwargs):
        raise InjectedInterruptionError(stage)

    with monkeypatch.context() as patch:
        patch.setattr(journal, "begin", begin)
        patch.setattr(journal, "advance", advance)
        if stage == "library-before-projection":
            patch.setattr(app.projections, "commit_reset", fail)
        if stage == "authoring":
            patch.setattr(type(app.evolution_loop.author), "author", fail)
        with pytest.raises(InjectedInterruptionError):
            asyncio.run(app.evolution_loop.run(fixture.run_plan))
        with pytest.raises(RuntimeError):
            asyncio.run(app.evolution_loop.run(fixture.run_plan))

    record = journal.load(2)
    durable = record.state in {
        StepTransactionState.CHECKPOINT_PUBLISHED,
        StepTransactionState.ADAPTER_COMMITTED,
        StepTransactionState.SOURCE_EVENTS_PUBLISHED,
        StepTransactionState.COMMITTED,
    }
    step = 2 if durable else 1
    directory = tmp_path / "snapshots" / f"step-{step:08d}"
    recovered, publisher = restore_fixture(fixture, directory, training_backbone_config)
    assert recovered.training_loop.optimizer_step == step
    if durable:
        require_pending_step_matches(recovered, record)
        assert recovered.library.current_version != initial_library
        z_parameters = recovered.training_loop.backbone.parameter_groups().z_head
        assert all(
            parameter not in recovered.training_loop.optimizer.state for parameter in z_parameters
        )
    else:
        assert recovered.library.current_version == initial_library
    if stage == "checkpoint-before-journal":
        assert not (tmp_path / "snapshots" / "step-00000002").exists()
        assert tuple((tmp_path / "snapshots").glob("uncommitted-step-00000002-*"))

    if stage == "authoring":
        # A lost/failed external response is not permission to request a new
        # random draft under the original phase identity.
        with pytest.raises(UnresolvedAuthoringCallError):
            asyncio.run(recovered.evolution_loop.run(fixture.run_plan))
        assert recovered.training_loop.backbone.authoring_prompts == []
        return
    original_draft_calls = len(app.training_loop.backbone.authoring_prompts)
    summary = asyncio.run(recovered.evolution_loop.run(fixture.run_plan))
    assert original_draft_calls + len(recovered.training_loop.backbone.authoring_prompts) == 1
    assert summary.final_optimizer_step == 3
    assert summary.cycles_committed_in_run == 1
    assert publisher.committed[-1][0] == 3
    batches = recovered.projections.posterior_provenance.batches
    assert [batch.optimizer_step for batch in batches] == [1, 2, 3]
    assert batches[1].library_version == initial_library
    assert batches[2].library_version == recovered.library.current_version
    recovered.projections.posterior_provenance.require_reconstructed_cells(
        recovered.projections.calibration_cells
    )
    groups = recovered.training_loop.backbone.parameter_groups()
    assert set(
        _optimizer_steps(recovered.training_loop.optimizer, (*groups.forward, *groups.backward))
    ) == {3}
    assert set(_optimizer_steps(recovered.training_loop.optimizer, groups.z_head)) == {1}
    commits = tuple(
        TrainingStepCommit.from_value(event.payload)
        for event in read_event_history(fixture.event_log.path)
        if event.event_type is EventType.TRAINING_STEP_COMMITTED
    )
    assert [commit.optimizer_step for commit in commits] == [1, 2, 3]
    assert sum(len(commit.posterior_batch.updates) for commit in commits) == len(
        recovered.projections.posterior_provenance.event_ids
    )


def test_orphan_archive_is_step_scoped_and_preserves_earlier_commits(tmp_path):
    store = FilesystemTrainingCheckpointStore(root=tmp_path)
    preserved = ("step-00000001", "step-00000003", "unrelated")
    orphaned = (
        "step-00000002",
        "cadence-step-00000002",
        "phase-00000001-step-00000002",
        ".step-00000002.staging-incomplete",
        ".phase-00000001-step-00000002.staging-incomplete",
        ".phase-00000002-step-00000002.staging-incomplete",
    )
    for name in (*preserved, *orphaned):
        (tmp_path / name).mkdir()
        (tmp_path / name / "marker").write_text(name)
    archived = store.archive_uncommitted_step(2)
    assert len(archived) == len(orphaned)
    assert all(path.joinpath("marker").read_text() == path.name for path in archived)
    assert all((tmp_path / name).exists() for name in preserved)
    assert all(not (tmp_path / name).exists() for name in orphaned)
    assert store.archive_uncommitted_step(2) == ()


def test_optimizer_layout_is_checked_before_installing_checkpoint_model(
    tmp_path,
    training_backbone_config,
    monkeypatch,
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    app = fixture.application
    asyncio.run(app.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1))
    directory = tmp_path / "snapshots" / "step-00000001"
    state = torch.load(directory / "optimizer.pt", weights_only=True)
    state["param_groups"][0]["params"].reverse()
    torch.save(state, directory / "optimizer.pt")
    installed = []
    monkeypatch.setattr(app.backbone, "load_checkpoint", lambda path: installed.append(path))
    store = FilesystemTrainingCheckpointStore(root=directory.parent)
    with pytest.raises(ValueError):
        store.restore(
            directory,
            backbone=app.backbone,
            optimizer=app.training_loop.optimizer,
            expected_experiment_id=store.load_metadata(directory).experiment_id,
            expected_identity=fixture.public_identity.runtime_snapshot_identity(),
        )
    assert installed == []


@pytest.mark.parametrize("stage", ["prepared", "optimizer-applied", "projection-installed"])
def test_first_step_failure_restores_complete_initial_state(
    tmp_path,
    training_backbone_config,
    monkeypatch,
    stage,
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    app = fixture.application
    original_begin = fixture.journal.begin
    original_advance = fixture.journal.advance

    def begin(**kwargs):
        record = original_begin(**kwargs)
        if stage == "prepared":
            raise InjectedInterruptionError(stage)
        return record

    def advance(record, state, **kwargs):
        updated = original_advance(record, state, **kwargs)
        if state.value == stage:
            raise InjectedInterruptionError(stage)
        return updated

    with monkeypatch.context() as patch:
        patch.setattr(fixture.journal, "begin", begin)
        patch.setattr(fixture.journal, "advance", advance)
        with pytest.raises(InjectedInterruptionError):
            asyncio.run(app.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1))
    restored, _ = restore_fixture(
        fixture,
        tmp_path / "snapshots" / "initial-step-00000000",
        training_backbone_config,
    )
    assert restored.training_loop.optimizer_step == 0
    assert restored.projections.calibration_cells == ()
    summary = asyncio.run(
        restored.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1)
    )
    assert summary.final_optimizer_step == 1
    commits = [
        e
        for e in read_event_history(fixture.event_log.path)
        if e.event_type is EventType.TRAINING_STEP_COMMITTED
    ]
    assert len(commits) == 1


def test_recovery_cannot_silently_skip_a_saved_evolution_phase(
    tmp_path,
    training_backbone_config,
    monkeypatch,
):
    from skillev.evolution import NoPhaseTransition
    from skillev.evolution.detector import NoPhaseTransitionReason

    fixture = build_application_fixture(tmp_path, training_backbone_config)
    app = fixture.application
    asyncio.run(app.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1))

    def fail(_state):
        raise InjectedInterruptionError("library-before-projection")

    with monkeypatch.context() as patch:
        patch.setattr(app.projections, "commit_reset", fail)
        with pytest.raises(InjectedInterruptionError):
            asyncio.run(app.evolution_loop.run(fixture.run_plan))
    restored, _ = restore_fixture(
        fixture,
        tmp_path / "snapshots" / "step-00000001",
        training_backbone_config,
    )
    original = restored.detector.preview_observation

    def no_phase(diagnostic):
        observation = original(diagnostic)
        return NoPhaseTransition(
            replace(observation.next_state, phase_already_triggered=False),
            NoPhaseTransitionReason.RESIDUAL_NOT_STAGNANT,
        )

    monkeypatch.setattr(restored.detector, "preview_observation", no_phase)
    with pytest.raises(UnresolvedAuthoringCallError):
        asyncio.run(restored.evolution_loop.run(fixture.run_plan))
    assert restored.backbone.authoring_prompts == []
    assert not (tmp_path / "snapshots" / "step-00000002").exists()


def test_resume_does_not_treat_a_missing_step_transaction_as_a_new_run(
    tmp_path,
    training_backbone_config,
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    asyncio.run(
        fixture.application.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1)
    )
    # Simulate copying a model snapshot without its associated commit records.
    for path in fixture.journal.directory.glob("*.json"):
        path.unlink()
    with pytest.raises(RuntimeError):
        restore_fixture(fixture, tmp_path / "snapshots" / "step-00000001", training_backbone_config)
