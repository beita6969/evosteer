import asyncio
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from skillev_private.experiments.protocol_v10_application_input import (
    ProtocolV10ApplicationIdentity,
)

from skillev.application import SKILLEVApplication
from skillev.application_continuation import HorizonContinuation, ReasoningContinuation
from skillev.experiments import FormalMethodV10
from skillev.runtime import LiveAttemptEventLog, StepTransactionJournal
from skillev.training import PrivateCheckpointStorageBinding
from skillev.training.config import INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT
from tests.application.test_bayesian_recovery import restore_fixture
from tests.application.test_full_vertical_loop import (
    _ScriptedQwenBackbone,
    build_application_fixture,
)


def with_config(identity, config):
    return ProtocolV10ApplicationIdentity(
        method=FormalMethodV10.BAYESIAN_IMPROVE_FULL,
        application_config=config,
        run_plan=identity.run_plan,
        initial_run_cursor=identity.initial_run_cursor,
        phase_checkpoint_cycle_ordinals=identity.phase_checkpoint_cycle_ordinals,
        snapshot_identity=replace(
            identity.runtime_snapshot_identity(),
            application_config_hash=config.content_hash,
            public_identity_content_hash=config.content_hash,
        ),
    )


def with_rollout(identity, **changes):
    config = identity.application_config
    return with_config(
        identity,
        replace(
            config,
            trainer=replace(config.trainer, rollout=replace(config.trainer.rollout, **changes)),
        ),
    )


def larger_reasoning(identity, **changes):
    rollout = identity.application_config.trainer.rollout
    return with_rollout(
        identity,
        max_reasoning_tokens=rollout.max_reasoning_tokens * 2,
        per_rollout_maximum=replace(
            rollout.per_rollout_maximum,
            output_tokens=rollout.max_turns
            * (rollout.max_reasoning_tokens * 2 + rollout.max_action_tokens),
        ),
        **changes,
    )


def test_native_catalog_and_reasoning_envelope_are_explicit(tmp_path, training_backbone_config):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    source = with_rollout(
        fixture.public_identity,
        format=INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT,
        phase_context=True,
        action_wire="native-single-tool-call@1",
    )
    target = larger_reasoning(source, reasoning_tool_catalog=True)
    assert ReasoningContinuation(source, 2).require_target(target) == source.snapshot_identity
    assert (
        ReasoningContinuation(source, 2).require_target(
            with_rollout(source, reasoning_tool_catalog=True)
        )
        == source.snapshot_identity
    )
    with pytest.raises(ValueError):
        HorizonContinuation(source, 2).require_target(target)


@pytest.mark.parametrize(
    "change",
    [
        {"base_seed": 3},
        {"reasoning_native_thinking": True},
        {"skill_exposure": "catalog-then-read@1"},
        {"reasoning_by_domain": (("healthbench", True),)},
    ],
)
def test_reasoning_transition_does_not_change_sampling_or_other_prompt_axes(
    tmp_path, training_backbone_config, change
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    source = with_rollout(
        fixture.public_identity,
        format=INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT,
        phase_context=True,
        action_wire="native-single-tool-call@1",
    )
    with pytest.raises(ValueError):
        ReasoningContinuation(source, 0).require_target(larger_reasoning(source, **change))


@pytest.mark.parametrize(
    "field",
    [
        "method_identity_hash",
        "initial_trainable_state_hash",
        "initial_library_version",
        "ordered_task_sequence_hash",
        "sampling_schedule_hash",
        "protocol_hash",
        "base_model_artifact_hash",
        "tokenizer_artifact_hash",
        "implementation_build_hash",
        "formal_execution_hash",
        "run_plan_hash",
    ],
)
def test_reasoning_transition_preserves_existing_runtime_identities(
    tmp_path, training_backbone_config, field
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    source = fixture.public_identity
    target = larger_reasoning(source)
    other = source.runtime_snapshot_identity().public_identity_content_hash
    assert getattr(target.snapshot_identity, field) != other
    with pytest.raises(ValueError):
        ReasoningContinuation(source, 0).require_target(
            replace(target, snapshot_identity=replace(target.snapshot_identity, **{field: other}))
        )


@pytest.mark.parametrize(
    "axis", ["horizon", "input", "optimizer", "reward", "experiment", "scorer", "sampling"]
)
def test_reasoning_transition_rejects_other_scientific_changes(
    tmp_path, training_backbone_config, axis
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    source = fixture.public_identity
    target = larger_reasoning(source)
    config = target.application_config
    rollout = config.trainer.rollout
    if axis == "horizon":
        target = with_rollout(
            target,
            max_turns=rollout.max_turns * 2,
            per_rollout_maximum=rollout.per_rollout_maximum.scale(2),
        )
    elif axis == "input":
        target = with_rollout(
            target,
            per_rollout_maximum=replace(
                rollout.per_rollout_maximum,
                input_tokens=rollout.per_rollout_maximum.input_tokens * 2,
            ),
        )
    elif axis in {"optimizer", "reward", "experiment"}:
        trainer = config.trainer
        if axis == "optimizer":
            trainer = replace(
                trainer, optimizer=replace(trainer.optimizer, adapter_learning_rate=0.002)
            )
        elif axis == "reward":
            trainer = replace(trainer, method=replace(trainer.method, epsilon_min=0.02))
        else:
            trainer = replace(trainer, execution=replace(trainer.execution, experiment_id="branch"))
        target = with_config(target, replace(config, trainer=trainer))
    else:
        change = (
            {"terminal_evaluation_conditions_json": '{"scorer":"different"}'}
            if axis == "scorer"
            else {"sampling_schedule_algorithm": "other-order"}
        )
        target = replace(target, snapshot_identity=replace(target.snapshot_identity, **change))
    with pytest.raises(ValueError):
        ReasoningContinuation(source, 0).require_target(target)


def test_branch_from_step_two_preserves_full_state_and_keeps_later_source_history(
    tmp_path, training_backbone_config
):
    fixture = build_application_fixture(tmp_path / "source", training_backbone_config, cycles=2)
    asyncio.run(
        fixture.application.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=3)
    )
    original = tmp_path / "source/snapshots/step-00000002"
    source_metadata = fixture.application.snapshot_store.load_exact(original)
    original_later = (fixture.journal.directory / "step-00000003.json").read_bytes()
    branch = tmp_path / "branch/snapshots/step-00000002"
    shutil.copytree(original, branch)
    journal = StepTransactionJournal((branch.parent / "step-transactions").resolve())
    for step in (1, 2):
        name = f"step-{step:08d}.json"
        shutil.copyfile(fixture.journal.directory / name, journal.directory / name)
    branch_fixture = SimpleNamespace(
        **{
            **vars(fixture),
            "journal": journal,
            "event_log": LiveAttemptEventLog(
                tmp_path / "branch/events.jsonl", run_id="branch", attempt_id="resume"
            ),
        }
    )
    target = larger_reasoning(fixture.public_identity)
    restored, _ = restore_fixture(
        branch_fixture,
        branch,
        training_backbone_config,
        public_identity=target,
        reasoning_continuation=ReasoningContinuation(fixture.public_identity, 2),
    )
    assert restored.training_loop.optimizer_step == 2
    assert restored.training_loop.policy_snapshot_id == journal.load(2).policy_snapshot_after
    boundary = restored.evolution_loop.save_condition_boundary("reasoning-from-step-two")
    saved = restored.snapshot_store.load_exact(boundary)
    assert saved.execution_state == source_metadata.execution_state
    assert saved.experiment_id == source_metadata.experiment_id
    assert saved.identity == target.runtime_snapshot_identity()
    # Same complete named Adam state, all F/B/Z tensors and versions, not just a policy reload.
    before_optimizer = torch.load(original / source_metadata.optimizer_file, weights_only=True)
    after_optimizer = torch.load(boundary / saved.optimizer_file, weights_only=True)
    torch.testing.assert_close(
        after_optimizer.pop("state"), before_optimizer.pop("state"), rtol=0, atol=0
    )
    assert after_optimizer == before_optimizer
    policy = original / source_metadata.policy_directory
    for file in policy.rglob("*"):
        if not file.is_file():
            continue
        other = boundary / saved.policy_directory / file.relative_to(policy)
        if file.suffix == ".pt":
            torch.testing.assert_close(
                torch.load(file, weights_only=True),
                torch.load(other, weights_only=True),
                rtol=0,
                atol=0,
            )
        elif file.suffix == ".safetensors":
            torch.testing.assert_close(load_file(file), load_file(other), rtol=0, atol=0)
        else:
            assert other.read_bytes() == file.read_bytes()
    # A new process can exactly resume the new boundary; this does not reset the cursor to zero.
    again, _ = restore_fixture(
        branch_fixture, boundary, training_backbone_config, public_identity=target
    )
    assert again.training_loop.optimizer_step == 2
    assert again.training_loop.task_provider.runtime_state == saved.execution_state.task_cursor
    # Multi-cycle scripted generation reads the restored library, not an external model.
    again.training_loop.backbone.fixture_library = again.library
    asyncio.run(again.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1))
    assert again.training_loop.optimizer_step == 3
    assert again.projections.posterior_provenance.batches[:2] == (
        saved.execution_state.projections.posterior_provenance.batches
    )
    assert fixture.application.training_loop.optimizer_step == 3
    assert (fixture.journal.directory / "step-00000003.json").read_bytes() == original_later


@pytest.mark.parametrize("bad_boundary", ["step", "pending", "both"])
def test_transition_rejects_noncommitted_or_ambiguous_boundary_before_model_load(
    tmp_path, training_backbone_config, monkeypatch, bad_boundary
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    asyncio.run(
        fixture.application.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1)
    )
    target = larger_reasoning(fixture.public_identity)
    if bad_boundary == "pending":
        fixture.journal.begin(optimizer_step=2, batch_id="pending", policy_snapshot_before="before")
    monkeypatch.setattr(
        _ScriptedQwenBackbone,
        "load_checkpoint",
        lambda *args: pytest.fail("invalid transition must fail before loading model state"),
    )
    with pytest.raises(ValueError):
        restore_fixture(
            fixture,
            tmp_path / "snapshots/step-00000001",
            training_backbone_config,
            public_identity=target,
            reasoning_continuation=ReasoningContinuation(
                fixture.public_identity, 0 if bad_boundary == "step" else 1
            ),
            continuation=HorizonContinuation(fixture.public_identity, 1)
            if bad_boundary == "both"
            else None,
        )


def test_formal_wrapper_forwards_explicit_continuation_and_branch_journal(tmp_path, monkeypatch):
    captured = {}
    application = object()

    def resume(**kwargs):
        captured.update(kwargs)
        return application

    monkeypatch.setattr(SKILLEVApplication, "resume", resume)
    runtime = SimpleNamespace(
        rollout_generator_factory=object(),
        gradient_preparer=object(),
        workflow_resources=SimpleNamespace(binding=object()),
        step_adapter_publisher_factory=object(),
        skill_author_factory=object(),
        require_bound=lambda app: captured.update(bound=app),
    )
    declaration = ReasoningContinuation(SimpleNamespace(), 2)
    storage = PrivateCheckpointStorageBinding(directory=str(tmp_path / "branch-checkpoints"))
    result = SKILLEVApplication.resume_formal(
        snapshot_directory=tmp_path / "source-step-two",
        backbone_config=None,
        task_provider_factory=None,
        base_session_factory=None,
        terminal_components=None,
        checkpoint_storage=storage,
        initial_checkpoint=None,
        public_identity=None,
        event_log=None,
        clock=lambda: "fixture",
        runtime=runtime,
        reasoning_continuation=declaration,
    )
    assert result is application
    assert captured["bound"] is application
    assert captured["reasoning_continuation"] is declaration
    assert captured["horizon_continuation"] is None
    assert (
        captured["step_transaction_journal"].directory
        == (tmp_path / "branch-checkpoints/step-transactions").resolve()
    )
