import asyncio
from dataclasses import replace

import pytest
from skillev_private.experiments.bayesian_improve_training import _bind_run_directory
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig
from skillev_private.experiments.protocol_v10_application_input import (
    ProtocolV10ApplicationIdentity,
)

from skillev.application_continuation import HorizonContinuation
from skillev.experiments import FormalMethodV10
from tests.application.test_bayesian_recovery import restore_fixture
from tests.application.test_full_vertical_loop import build_application_fixture


def target_identity(fixture, *, multiplier=2):
    source = fixture.public_identity
    old = source.application_config
    rollout = old.trainer.rollout
    new = replace(
        old,
        trainer=replace(
            old.trainer,
            rollout=replace(
                rollout,
                max_turns=rollout.max_turns * multiplier,
                per_rollout_maximum=rollout.per_rollout_maximum.scale(multiplier),
            ),
        ),
    )
    return ProtocolV10ApplicationIdentity(
        FormalMethodV10.BAYESIAN_IMPROVE_FULL,
        new,
        source.run_plan,
        source.initial_run_cursor,
        replace(
            source.runtime_snapshot_identity(),
            application_config_hash=new.content_hash,
            public_identity_content_hash=new.content_hash,
        ),
        source.phase_checkpoint_cycle_ordinals,
    )


def test_declared_horizon_transition_preserves_full_state_and_named_optimizer(
    tmp_path, training_backbone_config
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    asyncio.run(
        fixture.application.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1)
    )
    directory = tmp_path / "snapshots/step-00000001"
    target = target_identity(fixture)
    transition = HorizonContinuation(fixture.public_identity, 1)
    restored, _ = restore_fixture(
        fixture,
        directory,
        training_backbone_config,
        public_identity=target,
        continuation=transition,
    )
    assert restored.training_loop.config == target.application_config.trainer
    assert restored.snapshot_identity == target.runtime_snapshot_identity()
    before = restored.projections.runtime_state()
    saved = restored.evolution_loop.save_condition_boundary("condition-after-step-1")
    metadata = restored.snapshot_store.load_exact(saved)
    assert metadata.execution_state.projections == before
    assert metadata.identity == target.runtime_snapshot_identity()
    assert fixture.application.snapshot_store.load_exact(directory).identity != metadata.identity
    assert metadata.optimizer_step == 1


def test_transition_rejects_unrelated_method_change(tmp_path, training_backbone_config):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    target = target_identity(fixture)
    bad_config = replace(
        target.application_config,
        diagnostics=replace(target.application_config.diagnostics, window_size=3),
    )
    changed = replace(
        target,
        application_config=bad_config,
        snapshot_identity=replace(
            target.snapshot_identity, application_config_hash=bad_config.content_hash
        ),
    )
    with pytest.raises(ValueError):
        HorizonContinuation(fixture.public_identity, 0).require_target(changed)


def test_formal_horizon_change_is_explicit_and_never_overwrites_original(tmp_path):
    source = BayesianFormalConfig(max_turns=50, static_max_turns=2)
    target = replace(source, max_turns=25, static_max_turns=5)
    root = tmp_path / "formal"
    _bind_run_directory(root, source, None)
    original = (root / "formal-config.json").read_text()
    saved = root / "checkpoints/step-00000002"
    with pytest.raises(ValueError):
        _bind_run_directory(root, target, saved)
    assert _bind_run_directory(root, target, saved, allow_new_horizons=True) == source
    assert (root / "formal-config.json").read_text() == original
    with pytest.raises(ValueError):
        _bind_run_directory(
            root, replace(target, adapter_learning_rate=2e-4), saved, allow_new_horizons=True
        )
