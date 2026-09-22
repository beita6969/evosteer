import json
from dataclasses import replace

import pytest
from skillev_private.experiments.bayesian_improve_training import _bind_run_directory
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig

from skillev.application_continuation import ReasoningContinuation, TokenBudgetNoticeContinuation
from skillev.training.config import PolicyRolloutConfig
from tests.application.test_full_vertical_loop import build_application_fixture
from tests.application.test_reasoning_continuation import larger_reasoning, with_rollout


def test_budget_notice_is_an_opt_in_roundtrippable_formal_condition(tmp_path):
    old = BayesianFormalConfig(
        format="skillev-bayesian-formal-training@6",
        phase_context=True,
        action_wire="native-single-tool-call@3",
    )
    new = replace(old, token_budget_notice=True)
    assert "token_budget_notice" not in old.to_value()
    assert old.condition != new.condition
    path = tmp_path / "config.json"
    for config in (old, new):
        path.write_text(json.dumps(config.to_value()))
        assert BayesianFormalConfig.load(path) == config
        rollout = config.application_config("fixture").trainer.rollout
        assert PolicyRolloutConfig.from_value(rollout.to_value()) == rollout
        assert rollout.token_budget_notice == config.token_budget_notice
    root = tmp_path / "run"
    _bind_run_directory(root, old, None)
    snapshot = root / "checkpoints/step-00000022"
    with pytest.raises(ValueError):
        _bind_run_directory(root, new, snapshot, allow_new_reasoning=True)
    assert _bind_run_directory(root, new, snapshot, allow_token_budget_notice=True) == old
    with pytest.raises(ValueError):
        _bind_run_directory(
            root, replace(new, max_reasoning_tokens=2048), snapshot, allow_token_budget_notice=True
        )


def test_notice_continuation_preserves_actual_budget_and_all_existing_conditions(
    tmp_path, training_backbone_config
):
    fixture = build_application_fixture(tmp_path, training_backbone_config)
    old = with_rollout(
        fixture.public_identity,
        format="skillev-policy-rollout@7",
        phase_context=True,
        action_wire="native-single-tool-call@3",
    )
    new = with_rollout(old, token_budget_notice=True)
    assert TokenBudgetNoticeContinuation(old, 22).require_target(new) == old.snapshot_identity
    with pytest.raises(ValueError):
        ReasoningContinuation(old, 22).require_target(new)
    with pytest.raises(ValueError):
        TokenBudgetNoticeContinuation(old, 22).require_target(larger_reasoning(new))
    with pytest.raises(ValueError):
        TokenBudgetNoticeContinuation(new, 22).require_target(old)
