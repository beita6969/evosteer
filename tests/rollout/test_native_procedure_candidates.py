from dataclasses import replace
from pathlib import Path

import pytest
from skillev_private.experiments.fresh_restart import load_fresh_config

from skillev.evolution.retriever import TaskRetrievalFeatures, applicability_matches
from skillev.experiments._evolution_preflight_seed import planned_seed_documents
from skillev.experiments.native_procedure_candidates import PROFILE


def test_mechanics_library_keeps_old_conditions_and_does_not_claim_execution():
    old = planned_seed_documents("public-procedure-advice@3")
    new = planned_seed_documents(PROFILE)
    old_by_id = {d.manifest.skill_id: d for d in old}
    inherited = [d for d in new if d.manifest.skill_id in old_by_id]
    assert inherited
    assert all(d == old_by_id[d.manifest.skill_id] for d in inherited)
    revised = [d for d in new if d.manifest.skill_id not in old_by_id]
    assert len(revised) == 2
    assert all(d.applicability.required_tools == ("act",) for d in revised)
    assert all(d.applicability.contexts == ("alfworld:task",) for d in revised)
    assert all(
        any(r.requirement_id == "advice-is-not-execution" for r in d.requirements) for d in revised
    )
    assert planned_seed_documents("public-procedure-advice@3") == old


def test_mechanics_are_offered_only_on_the_actual_interactive_interface():
    features = TaskRetrievalFeatures(
        "public", "alfworld/task", "alfworld:task", ("act",), "alfworld", "public goal"
    )
    documents = planned_seed_documents(PROFILE)
    offered = [
        d
        for d in documents
        if applicability_matches(d.applicability, features, skill_id=d.manifest.skill_id)
    ]
    assert len(offered) == 2
    for alternate in (
        replace(features, available_tools=()),
        replace(features, context="humaneval:task"),
    ):
        assert not any(
            applicability_matches(d.applicability, alternate, skill_id=d.manifest.skill_id)
            for d in offered
        )


def test_explicit_native_procedure_condition_preserves_training_math_and_budgets():
    base = load_fresh_config(Path("configs/training/bayesianimprove_skill_value_candidate.yaml"))
    candidate = replace(base, initial_skill_profile=PROFILE)
    assert candidate.condition != base.condition
    assert candidate.domain_task_budgets == base.domain_task_budgets
    assert candidate.sampling_config == base.sampling_config
    assert candidate.ttb_beta == base.ttb_beta
    assert candidate.epsilon == base.epsilon
    assert (
        load_fresh_config(Path("configs/training/bayesianimprove_skill_mechanics_candidate.yaml"))
        == candidate
    )
    with pytest.raises(ValueError):
        replace(candidate, skill_exposure="full-inline@1")
