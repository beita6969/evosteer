import asyncio
import json
from dataclasses import replace

import pytest
from skillev_private.experiments.bayesian_condition_transition import resume_condition
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from skillev.contracts import canonical_json
from skillev.evaluation.format_content_review import format_review_condition
from skillev.format_review_continuation import FormatReviewContinuation
from tests.application import test_full_vertical_loop as vertical
from tests.application.test_bayesian_recovery import restore_fixture
from tests.application.test_reasoning_continuation import with_config, with_rollout


def test_format_review_boundary_keeps_full_optimizer_posterior_library_and_cursor(
    tmp_path, training_backbone_config, monkeypatch
):
    old_conditions = {"mbpp-plus": {"fixture": "unchanged"}}
    old = canonical_json(old_conditions)
    original = vertical.make_public_identity

    def identity(**kwargs):
        source = original(**kwargs)
        source = with_config(source, source.application_config)
        return replace(
            source,
            snapshot_identity=replace(
                source.snapshot_identity,
                terminal_evaluation_conditions_json=old,
            ),
        )

    monkeypatch.setattr(vertical, "make_public_identity", identity)
    monkeypatch.setattr(
        vertical._BaseSessionFactory, "terminal_evaluation_conditions_json", old, raising=False
    )
    fixture = vertical.build_application_fixture(tmp_path, training_backbone_config)
    asyncio.run(
        fixture.application.evolution_loop.run(fixture.run_plan, maximum_steps_this_attempt=1)
    )
    checkpoint = tmp_path / "snapshots/step-00000001"
    before = fixture.application.snapshot_store.load_exact(checkpoint)
    source = fixture.public_identity
    new = canonical_json({**old_conditions, "format_content_review": format_review_condition(2)})
    target = replace(
        source,
        snapshot_identity=replace(
            source.snapshot_identity, terminal_evaluation_conditions_json=new
        ),
    )
    continuation = FormatReviewContinuation(source, 1)
    assert continuation.require_target(target) == source.snapshot_identity
    with pytest.raises(ValueError):
        continuation.require_target(with_rollout(target, base_seed=3))
    with pytest.raises(ValueError):
        FormatReviewContinuation(source, 2).require_target(target)
    monkeypatch.setattr(vertical._BaseSessionFactory, "terminal_evaluation_conditions_json", new)
    restored, _ = restore_fixture(
        fixture,
        checkpoint,
        training_backbone_config,
        public_identity=target,
        format_review_continuation=continuation,
    )
    assert restored.training_loop.optimizer_step == 1
    after_path = restored.evolution_loop.save_condition_boundary("format-review-step-one")
    after = restored.snapshot_store.load_exact(after_path)
    assert after.execution_state == before.execution_state
    assert after.identity.terminal_evaluation_conditions_json == new
    assert fixture.application.snapshot_store.load_exact(checkpoint) == before


def test_condition_change_is_opt_in_and_cannot_change_any_other_field(tmp_path):
    source = BayesianFormalConfig()
    (tmp_path / "formal-config.json").write_text(json.dumps(source.to_value()))
    target = replace(source, format_review_from_step=81)
    assert BayesianFormalConfig.load(tmp_path / "formal-config.json") == source
    (tmp_path / "review-config.json").write_text(json.dumps(target.to_value()))
    assert BayesianFormalConfig.load(tmp_path / "review-config.json") == target
    assert "format_review_from_step" not in source.to_value()
    assert "from81" in target.condition
    with pytest.raises(ValueError):
        resume_condition(tmp_path, target, allow_new_horizons=False)
    assert (
        resume_condition(tmp_path, target, allow_new_horizons=False, allow_format_review=True)
        == source
    )
    with pytest.raises(ValueError):
        resume_condition(
            tmp_path,
            replace(target, max_turns=25),
            allow_new_horizons=False,
            allow_format_review=True,
        )
