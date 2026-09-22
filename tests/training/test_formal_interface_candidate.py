"""Versioned candidate configuration reaches the formal collection factory."""

from dataclasses import replace

import pytest
import yaml
from skillev_private.experiments.bayesian_training_config import BayesianFormalConfig


@pytest.mark.parametrize(
    "wire", ["native-single-tool-call@1", "native-single-tool-call@2", "native-single-tool-call@3"]
)
@pytest.mark.parametrize("exposure", ["catalog-then-read@1", "catalog-then-read@2"])
def test_formal_native_catalog_candidate_roundtrip_and_factory(tmp_path, wire, exposure):
    legacy = BayesianFormalConfig()
    candidate = replace(
        legacy,
        format="skillev-bayesian-formal-training@4",
        phase_context=True,
        action_wire=wire,
        skill_exposure=exposure,
    )
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(candidate.to_value()))
    restored = BayesianFormalConfig.load(path)
    assert restored == candidate
    rollout = restored.application_config("candidate-step-zero").trainer.rollout
    assembler = rollout.context_assembler(maximum_h0_tokens=restored.max_input_tokens)
    assert assembler.action_wire == candidate.action_wire
    assert assembler.skill_exposure == candidate.skill_exposure
    assert candidate.condition != legacy.condition
    assert rollout.condition_id != legacy.application_config("legacy").trainer.rollout.condition_id
    assert "phase_context" not in legacy.to_value()
    with pytest.raises(ValueError):
        replace(legacy, phase_context=True)


def test_domain_reasoning_budget_changes_identity_and_envelope_not_other_tasks(tmp_path):
    base = BayesianFormalConfig(format="skillev-bayesian-formal-training@6")
    candidate = replace(
        base, max_turns=50, reasoning_tokens_by_domain=(("aime-2026", 16384), ("triviaqa", 4096))
    )
    assert candidate.condition != base.condition
    assert candidate.maximum_reasoning_tokens == 16384
    assert (
        candidate.application_config("larger-reasoning").trainer.rollout.max_reasoning_tokens
        == 16384
    )
    assert candidate.static_task_budget.max_reasoning_tokens == 1024
    assert candidate.domain_task_budgets["aime-2026"].max_reasoning_tokens == 16384
    assert candidate.domain_task_budgets["triviaqa"].max_reasoning_tokens == 4096
    assert candidate.task_budget.max_turns == 50
    assert candidate.static_task_budget.max_turns == 8
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(candidate.to_value()))
    assert BayesianFormalConfig.load(path) == candidate
    path.write_text(yaml.safe_dump(base.to_value()))
    assert BayesianFormalConfig.load(path) == base
    assert "reasoning_tokens_by_domain" not in base.to_value()


def test_reasoning_tool_catalog_is_a_persisted_opt_in_condition(tmp_path):
    from skillev.training.config import PolicyRolloutConfig

    base = BayesianFormalConfig(
        format="skillev-bayesian-formal-training@6",
        phase_context=True,
        action_wire="native-single-tool-call@2",
    )
    candidate = replace(base, reasoning_tool_catalog=True)
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(candidate.to_value()))
    assert BayesianFormalConfig.load(path) == candidate
    rollout = candidate.application_config("candidate").trainer.rollout
    assert PolicyRolloutConfig.from_value(rollout.to_value()).reasoning_tool_catalog
    assert candidate.condition != base.condition
    old_rollout = base.application_config("base").trainer.rollout
    assert rollout.condition_id != old_rollout.condition_id
    assert not PolicyRolloutConfig.from_value(old_rollout.to_value()).reasoning_tool_catalog
    assert rollout.context_assembler(maximum_h0_tokens=10000).assembler_version != (
        old_rollout.context_assembler(maximum_h0_tokens=10000).assembler_version
    )
    path.write_text(yaml.safe_dump(base.to_value()))
    assert BayesianFormalConfig.load(path) == base
    with pytest.raises(ValueError):
        BayesianFormalConfig(reasoning_tool_catalog=True)


@pytest.mark.parametrize(
    "budgets",
    [
        (("webshop", 1024),),
        (("aime-2026", 0),),
        (("aime-2026", True),),
        (("aime-2026", 1024), ("aime-2026", 2048)),
    ],
)
def test_domain_reasoning_budgets_are_positive_unique_seven_domain_controls(budgets):
    with pytest.raises(ValueError):
        BayesianFormalConfig(
            format="skillev-bayesian-formal-training@6", reasoning_tokens_by_domain=budgets
        )
