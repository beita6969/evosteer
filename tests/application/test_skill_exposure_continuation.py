import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from skillev_private.experiments.bayesian_condition_transition import (
    observer_condition_starts,
    save_condition_transition,
)
from skillev_private.experiments.bayesian_improve_training import _bind_run_directory
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from skillev.application import SKILLEVApplication
from skillev.application_continuation import SkillExposureContinuation
from skillev.contracts import canonical_json
from skillev.runtime import ActionKind, StructuredAction
from skillev.training import PrivateCheckpointStorageBinding
from skillev.training.config import INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT, PolicyRolloutConfig
from tests.application.test_bayesian_recovery import restore_fixture
from tests.application.test_full_vertical_loop import build_application_fixture
from tests.application.test_reasoning_continuation import with_rollout

TRANSITIONS = [
    ("catalog-then-read@1", "catalog-then-read@3"),
    ("catalog-then-read@3", "catalog-then-read@1"),
    ("catalog-then-read@2", "catalog-then-read@1"),
    ("full-inline", "catalog-then-read@1"),
    ("full-inline", "catalog-then-read@2"),
    ("catalog-then-read@1", "catalog-then-read@2"),
]


@pytest.mark.parametrize(("before", "exposure"), TRANSITIONS)
def test_catalog_transition_changes_only_exposure_and_requires_saved_source(
    tmp_path, before, exposure
):
    old = BayesianFormalConfig(
        format="skillev-bayesian-formal-training@6",
        phase_context=True,
        action_wire="native-single-tool-call@3",
        token_budget_notice=True,
        skill_exposure=before,
    )
    new = replace(old, skill_exposure=exposure)
    root = tmp_path / "run"
    _bind_run_directory(root, old, None)
    assert observer_condition_starts(root, old) == {1: old.condition}
    snapshot = root / "checkpoints/step-00000046"
    assert old.condition != new.condition
    assert _bind_run_directory(root, new, snapshot, allow_catalog_read=True) == old
    with pytest.raises(ValueError):
        _bind_run_directory(tmp_path / "fresh", new, None, allow_catalog_read=True)
    with pytest.raises(ValueError):
        _bind_run_directory(root, new, snapshot)
    for target in (replace(new, max_turns=25), replace(new, adapter_learning_rate=0.002)):
        with pytest.raises(ValueError):
            _bind_run_directory(root, target, snapshot, allow_catalog_read=True)
    with pytest.raises(ValueError):
        _bind_run_directory(
            root, new, snapshot, allow_catalog_read=True, allow_token_budget_notice=True
        )
    assert json.loads((root / "formal-config.json").read_text()) == old.to_value()
    saved_names = []
    app = SimpleNamespace(
        training_loop=SimpleNamespace(optimizer_step=46),
        snapshot_identity=SimpleNamespace(sampling_schedule_algorithm="unchanged-curriculum"),
        evolution_loop=SimpleNamespace(save_condition_boundary=saved_names.append),
    )
    boundary = save_condition_transition(
        app, root=root, source_snapshot=snapshot, source=old, target=new, catalog_read=True
    )
    declaration = json.loads((root / "condition-current.json").read_text())
    assert declaration["effective_from_optimizer_step"] == 47
    assert declaration["source_config"] == old.to_value()
    assert declaration["config"] == new.to_value()
    assert declaration["sampling_condition"] == "unchanged-curriculum"
    assert saved_names == [boundary.name]
    assert observer_condition_starts(root, new) == {1: old.condition, 47: new.condition}
    with pytest.raises(ValueError):
        observer_condition_starts(root, old)


@pytest.mark.parametrize(("before", "exposure"), TRANSITIONS)
def test_catalog_restore_preserves_method_state_and_resumes_new_boundary(
    tmp_path, training_backbone_config, monkeypatch, before, exposure
):
    # The existing scripted fixture uses the same action wire, but opts into a
    # rollout version that can represent either exposure condition.
    monkeypatch.setattr(
        "tests.application.test_full_vertical_loop.PolicyRolloutConfig",
        lambda **kwargs: PolicyRolloutConfig(
            format=INTERFACE_POLICY_ROLLOUT_CONFIG_FORMAT, skill_exposure=before, **kwargs
        ),
    )
    monkeypatch.setattr(
        "tests.application.test_full_vertical_loop._skill_action_text",
        lambda skill_id: canonical_json(
            StructuredAction(ActionKind.SKILL, "invoke", {}, "skill-runtime", skill_id).to_value()
        ),
    )
    fixture = build_application_fixture(tmp_path, training_backbone_config, cycles=2)
    asyncio.run(
        fixture.application.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1)
    )
    source = fixture.public_identity
    target = with_rollout(source, skill_exposure=exposure)
    declaration = SkillExposureContinuation(source, 1)
    assert declaration.require_target(target) == source.runtime_snapshot_identity()
    changed_horizon = with_rollout(
        target,
        max_turns=2,
        per_rollout_maximum=target.application_config.trainer.rollout.per_rollout_maximum.scale(2),
    )
    for bad in (source, with_rollout(target, base_seed=1), changed_horizon):
        with pytest.raises(ValueError):
            declaration.require_target(bad)
    if before == "full-inline":
        with pytest.raises(ValueError):
            SkillExposureContinuation(target, 1).require_target(source)
    else:
        assert (
            SkillExposureContinuation(target, 1).require_target(source)
            == target.runtime_snapshot_identity()
        )
    original = tmp_path / "snapshots/step-00000001"
    restored, _ = restore_fixture(
        fixture,
        original,
        training_backbone_config,
        public_identity=target,
        skill_exposure_continuation=declaration,
    )
    # restore_fixture compares complete optimizer state, posterior, library and cursor.
    assert restored.training_loop.optimizer_step == 1
    boundary = restored.evolution_loop.save_condition_boundary("catalog-from-step-one")
    saved = restored.snapshot_store.load_exact(boundary)
    before = fixture.application.snapshot_store.load_exact(original)
    assert saved.execution_state == before.execution_state
    again, _ = restore_fixture(fixture, boundary, training_backbone_config, public_identity=target)
    assert again.training_loop.optimizer_step == 1
    rollout = target.application_config.trainer.rollout
    assert rollout.context_assembler(maximum_h0_tokens=10000).skill_exposure == exposure
    again.training_loop.backbone.fixture_library = again.library
    asyncio.run(again.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1))
    assert again.training_loop.optimizer_step == 2
    assert again.projections.posterior_provenance.batches[:1] == (
        saved.execution_state.projections.posterior_provenance.batches
    )
    assert again.projections.posterior_provenance.batches[-1].posterior.updates
    prompts = again.training_loop.backbone.policy_prompts
    assert prompts
    assert all("Use the first complete seed procedure" not in prompt for prompt in prompts)


def test_formal_wrapper_forwards_catalog_continuation(tmp_path, monkeypatch):
    captured = {}
    sentinel = object()

    def resume(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(SKILLEVApplication, "resume", resume)
    runtime = SimpleNamespace(
        rollout_generator_factory=None,
        gradient_preparer=None,
        workflow_resources=SimpleNamespace(binding=None),
        step_adapter_publisher_factory=None,
        skill_author_factory=None,
        require_bound=lambda value: None,
    )
    declaration = SkillExposureContinuation(SimpleNamespace(), 46)
    assert (
        SKILLEVApplication.resume_formal(
            snapshot_directory=tmp_path / "step46",
            backbone_config=None,
            task_provider_factory=None,
            base_session_factory=None,
            terminal_components=None,
            checkpoint_storage=PrivateCheckpointStorageBinding(directory=str(tmp_path)),
            initial_checkpoint=None,
            public_identity=None,
            event_log=None,
            clock=lambda: "now",
            runtime=runtime,
            skill_exposure_continuation=declaration,
        )
        is sentinel
    )
    assert captured["skill_exposure_continuation"] is declaration
