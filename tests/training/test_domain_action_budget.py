from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig
from skillev_private.experiments.fresh_restart import load_fresh_config


def test_domain_action_cap_is_an_explicit_condition_and_global_reservation_not_other_tasks():
    base = BayesianFormalConfig.load(
        Path("configs/training/bayesianimprove_development_handoff.yaml")
    )
    candidate = replace(base, action_tokens_by_domain=(("healthbench", 4096),))
    assert base.max_action_tokens == 2048
    assert candidate.sampling_config.max_action_tokens == 4096
    assert candidate.domain_task_budgets["healthbench"].max_action_tokens == 4096
    assert candidate.domain_task_budgets["alfworld"].max_action_tokens == 2048
    assert candidate.domain_task_budgets["aime-2026"].max_reasoning_tokens == 16384
    assert candidate.condition != base.condition
    assert (
        candidate.sampling_config.per_rollout_maximum.output_tokens
        > base.sampling_config.per_rollout_maximum.output_tokens
    )
    assert "action_tokens_by_domain" not in base.to_value()


@pytest.mark.parametrize(
    "overrides", [(("other", 1),), (("healthbench", 0),), (("healthbench", True),)]
)
def test_invalid_domain_budget_is_rejected(overrides):
    with pytest.raises(ValueError):
        BayesianFormalConfig(
            format="skillev-bayesian-formal-training@6", action_tokens_by_domain=overrides
        )


def test_skill_candidate_is_independent_and_no_quota():
    candidate = load_fresh_config(
        Path("configs/training/bayesianimprove_skill_value_candidate.yaml")
    )
    assert candidate.initial_skill_profile == "public-procedure-advice@3"
    assert candidate.skill_exposure == "catalog-then-read@1"
    assert candidate.action_tokens_by_domain == ()  # no budget gain claimed without a comparison
    assert candidate.max_turns == 25


def test_new_skill_candidate_requires_explicit_library_but_legacy_controls_keep_defaults(tmp_path):
    legacy = load_fresh_config(Path("configs/training/bayesianimprove_development_handoff.yaml"))
    assert legacy.initial_skill_profile == "public-advisory@2"
    assert legacy.action_tokens_by_domain == ()
    raw = yaml.safe_load(
        Path("configs/training/bayesianimprove_skill_value_candidate.yaml").read_text()
    )
    raw.pop("initial_skill_profile")
    path = tmp_path / "undeclared-library.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_fresh_config(path)


def test_skill_learning_budget_candidate_keeps_other_domains_and_method_unchanged():
    from skillev.training.performance_config import TrainingPerformanceConfig

    base = load_fresh_config(
        Path("configs/training/bayesianimprove_skill_mechanics_candidate.yaml")
    )
    candidate = load_fresh_config(
        Path("configs/training/bayesianimprove_skill_learning_candidate.yaml")
    )
    expected = replace(
        base,
        action_tokens_by_domain=(("healthbench", 4096),),
        performance_profile=candidate.performance_profile,
    )
    assert candidate == expected
    assert candidate.domain_task_budgets["healthbench"].max_action_tokens == 4096
    assert candidate.domain_task_budgets["alfworld"].max_action_tokens == 2048
    assert candidate.max_turns == 25
    assert candidate.static_max_turns == 8
    assert candidate.skill_exposure == "catalog-then-read@1"
    assert candidate.condition != base.condition
    execution = TrainingPerformanceConfig.load(candidate.performance_profile)
    assert execution.actor_requests == execution.max_running_requests == 32
    assert execution.resident_trajectories == 32
    assert TrainingPerformanceConfig.load(base.performance_profile).actor_requests == 8
