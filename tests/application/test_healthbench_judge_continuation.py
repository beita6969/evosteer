import asyncio
from dataclasses import replace

import pytest

from skillev.contracts import canonical_json
from skillev.evaluation.external_judge_policy import (
    DIRECT_HEALTHBENCH_PROFILE,
    GATEWAY_HEALTHBENCH_PROFILE,
)
from skillev.evaluation.healthbench_luna_profile import PROFILE_ID, healthbench_condition
from skillev.healthbench_judge_continuation import HealthBenchJudgeContinuation
from tests.application import test_full_vertical_loop as vertical
from tests.application.test_bayesian_recovery import restore_fixture
from tests.application.test_reasoning_continuation import with_config, with_rollout


@pytest.mark.parametrize("previous", [None, DIRECT_HEALTHBENCH_PROFILE])
@pytest.mark.parametrize("target_profile", sorted({PROFILE_ID, GATEWAY_HEALTHBENCH_PROFILE}))
def test_complete_judge_boundary_preserves_optimizer_evidence_and_cursor(
    tmp_path, training_backbone_config, monkeypatch, previous, target_profile
):
    legacy = canonical_json(
        {
            "mbpp-plus": {"fixture": "unchanged"},
            **({"healthbench": healthbench_condition(previous)} if previous else {}),
        }
    )
    original = vertical.make_public_identity

    def identity(**kwargs):
        source = original(**kwargs)
        source = with_config(source, source.application_config)
        return replace(
            source,
            snapshot_identity=replace(
                source.snapshot_identity, terminal_evaluation_conditions_json=legacy
            ),
        )

    monkeypatch.setattr(vertical, "make_public_identity", identity)
    monkeypatch.setattr(
        vertical._BaseSessionFactory, "terminal_evaluation_conditions_json", legacy, raising=False
    )
    fixture = vertical.build_application_fixture(tmp_path, training_backbone_config)
    asyncio.run(
        fixture.application.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1)
    )
    checkpoint = tmp_path / "snapshots/step-00000001"
    before = fixture.application.snapshot_store.load_exact(checkpoint)
    source = fixture.public_identity
    conditions = canonical_json(
        {
            "mbpp-plus": {"fixture": "unchanged"},
            "healthbench": healthbench_condition(target_profile),
        }
    )
    target = replace(
        source,
        snapshot_identity=replace(
            source.snapshot_identity, terminal_evaluation_conditions_json=conditions
        ),
    )
    continuation = HealthBenchJudgeContinuation(source, 1)
    assert continuation.require_target(target) == source.snapshot_identity
    with pytest.raises(ValueError):
        continuation.require_target(with_rollout(target, base_seed=3))
    wrong = replace(
        target,
        snapshot_identity=replace(
            target.snapshot_identity,
            terminal_evaluation_conditions_json=canonical_json(
                {"healthbench": healthbench_condition(target_profile)}
            ),
        ),
    )
    with pytest.raises(ValueError):
        continuation.require_target(wrong)
    monkeypatch.setattr(
        vertical._BaseSessionFactory, "terminal_evaluation_conditions_json", conditions
    )
    restored, _ = restore_fixture(
        fixture,
        checkpoint,
        training_backbone_config,
        public_identity=target,
        healthbench_judge_continuation=continuation,
    )
    assert restored.training_loop.optimizer_step == 1
    boundary = restored.evolution_loop.save_condition_boundary("healthbench-luna-step-one")
    after = restored.snapshot_store.load_exact(boundary)
    assert before.execution_state == after.execution_state
    assert after.identity.terminal_evaluation_conditions_json == conditions
    assert fixture.application.snapshot_store.load_exact(checkpoint) == before
