"""Real tiny CPU Adam/projection resume; no Qwen service or production evidence."""

import asyncio
from dataclasses import replace

import pytest
import torch
from skillev_private.experiments.protocol_v10_application_input import (
    ProtocolV10ApplicationIdentity,
)

from skillev.application import SKILLEVApplication, TerminalComponents
from skillev.application_continuation import PlanContinuation
from skillev.evolution import PhiBudgetAuthority
from skillev.experiments import FormalMethodV10
from skillev.policy import PrivateInitialCheckpointBinding
from skillev.runtime import BudgetLedger, LiveAttemptEventLog
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan, RunSlotKind
from skillev.training import PrivateCheckpointStorageBinding
from tests.application.test_full_vertical_loop import (
    _BaseSessionFactory,
    _ScriptedQwenBackbone,
    _StepPublisher,
    build_application_fixture,
)
from tests.training.fakes import FakeISOClock, OrderedTaskProvider


def test_append_preserves_closure_history_and_old_serialization():
    old = ExactAttemptRunPlan(2, 1, 2)
    old_value = old.to_value()
    new = old.append(phase_search_steps=6, closure_steps=1)
    assert ExactAttemptRunPlan.from_value(old_value) == old
    assert "segments" not in old_value
    assert ExactAttemptRunPlan.from_value(new.to_value()) == new
    assert [new.slot_kind(i) for i in range(1, 11)] == [
        RunSlotKind.PHASE_SEARCH,
        RunSlotKind.PHASE_SEARCH,
        RunSlotKind.CLOSURE,
        *([RunSlotKind.PHASE_SEARCH] * 6),
        RunSlotKind.CLOSURE,
    ]
    assert new.maximum_cycles == 2
    assert old.to_value() == old_value


def target_identity(source, plan, cursor):
    return ProtocolV10ApplicationIdentity(
        method=FormalMethodV10.BAYESIAN_IMPROVE_FULL,
        application_config=source.application_config,
        run_plan=plan,
        initial_run_cursor=replace(cursor, run_plan_hash=plan.content_hash),
        phase_checkpoint_cycle_ordinals=source.phase_checkpoint_cycle_ordinals,
        snapshot_identity=replace(
            source.runtime_snapshot_identity(), run_plan_hash=plan.content_hash
        ),
    )


def test_complete_three_append_runs_only_seven_new_adam_transactions(
    tmp_path, training_backbone_config, monkeypatch
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    old = fixture.application
    asyncio.run(old.evolution_loop.run(fixture.run_plan))
    directory = old.evolution_loop.final_training_snapshot_directory
    saved = old.snapshot_store.load_exact(directory)
    event_bytes = fixture.event_log.path.read_bytes()
    old_files = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    old_journals = {p: p.read_bytes() for p in fixture.journal.directory.rglob("*") if p.is_file()}
    plan = fixture.run_plan.append(phase_search_steps=6, closure_steps=1)
    target = target_identity(fixture.public_identity, plan, saved.execution_state.run_cursor)
    tasks = fixture.tasks + tuple(
        replace(fixture.tasks[-1], task_id=f"appended-{i}") for i in range(14)
    )
    transition = PlanContinuation(
        fixture.public_identity,
        3,
        tuple(t.task_id for t in fixture.tasks),
        tuple(t.task_id for t in tasks),
    )
    transition.require_target(target)
    ledger = BudgetLedger(
        run_id="v3-application", attempt_id="attempt-1", cap=fixture.attempt_budget.scale(4)
    )
    ledger.restore_completed(old.training_loop.ledger.entries)
    original_spend = ledger.settled

    class TaskFactory:
        def from_exact_state(self, state):
            provider = OrderedTaskProvider(tasks, cursor=state.cursor)
            assert provider.runtime_state == state
            return provider

    backbone = _ScriptedQwenBackbone(training_backbone_config)
    monkeypatch.setattr("skillev.application.build_qwen_policy_backbone", lambda _: backbone)
    publisher = _StepPublisher([], [], [])
    app = SKILLEVApplication.resume(
        snapshot_directory=directory,
        backbone_config=training_backbone_config,
        initial_checkpoint=PrivateInitialCheckpointBinding(
            directory=str(fixture.initial_policy_directory),
            trainable_state=fixture.initial_trainable_state,
        ),
        task_provider_factory=TaskFactory(),
        base_session_factory=_BaseSessionFactory((1.0,) * 20, cursor=6),
        terminal_components=TerminalComponents(
            ledger, fixture.authority, PhiBudgetAuthority(fixture.phi_maximum)
        ),
        checkpoint_storage=PrivateCheckpointStorageBinding(directory=str(directory.parent)),
        public_identity=target,
        plan_continuation=transition,
        step_transaction_journal=fixture.journal,
        event_log=LiveAttemptEventLog.resume(
            fixture.event_log.path, run_id="v3-application", attempt_id="attempt-1"
        ),
        clock=FakeISOClock(),
        step_adapter_publisher_factory=lambda _: publisher,
    )
    assert app.training_loop.optimizer_step == 3
    assert app.projections.runtime_state() == saved.execution_state.projections
    assert app.library.state == saved.execution_state.library
    assert app.training_loop.task_provider.runtime_state == saved.execution_state.task_cursor
    assert (
        app.evolution_loop._run_progress.state.committed_cycles
        == saved.execution_state.run_cursor.committed_cycles
    )
    assert (
        app.training_loop.optimizer.state_dict()["param_groups"]
        == old.training_loop.optimizer.state_dict()["param_groups"]
    )
    torch.testing.assert_close(
        app.training_loop.optimizer.state_dict()["state"],
        old.training_loop.optimizer.state_dict()["state"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        backbone.named_trainable_parameters(),
        old.training_loop.backbone.named_trainable_parameters(),
        rtol=0,
        atol=0,
    )
    assert ledger.settled == original_spend
    checked = []
    detector = app.evolution_loop._detector
    original = detector.commit_observation

    def check(*args, **kwargs):
        checked.append(app.training_loop.optimizer_step)
        return original(*args, **kwargs)

    monkeypatch.setattr(detector, "commit_observation", check)
    summary = asyncio.run(app.evolution_loop.run(plan))
    assert summary.final_optimizer_step == app.training_loop.optimizer_step == 10
    assert len(app.projections.posterior_provenance.batches) == 10
    assert (
        app.projections.posterior_provenance.batches[:3]
        == saved.execution_state.projections.posterior_provenance.batches
    )
    assert app.training_loop.task_provider.runtime_state.cursor == 20
    assert checked == list(range(4, 10))
    assert all(p.read_bytes() == data for p, data in old_files.items())
    assert all(p.read_bytes() == data for p, data in old_journals.items())
    assert fixture.event_log.path.read_bytes().startswith(event_bytes)
    assert [fixture.journal.load(i).optimizer_step for i in range(1, 11)] == list(range(1, 11))
    assert app.evolution_loop._run_progress.state.committed_cycles <= plan.maximum_cycles
    # The explicit append permission is narrow: no new method, old task prefix or cycle allowance.
    with pytest.raises(ValueError):
        replace(
            transition, target_task_ids=("changed", *transition.target_task_ids[1:])
        ).require_target(target)
    with pytest.raises(ValueError):
        replace(transition, optimizer_step=2).require_target(target)
    with pytest.raises(ValueError):
        transition.require_target(
            target_identity(
                fixture.public_identity,
                replace(plan, maximum_cycles=2),
                saved.execution_state.run_cursor,
            )
        )
