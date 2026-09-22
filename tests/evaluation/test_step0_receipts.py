from dataclasses import replace

import pytest

from skillev.evaluation.step0_receipts import (
    ArchitectureIdentityReceipt,
    PairedArchitectureComparison,
    paired_outcome_counts,
)


def _identity() -> ArchitectureIdentityReceipt:
    return ArchitectureIdentityReceipt(
        method_id="skillev-bayesian-improve-step-zero@1",
        architecture_family="skillev-bayesian-improve",
        architecture_lineage="skillflow-plus-idea-tex",
        architecture_components=(
            "seeded-skill-controller",
            "trajectory-balance-gflownet",
            "beta-bernoulli-lcb-calibration",
            "operator-driven-skill-evolution",
        ),
        controller_id="controller@2",
        optimizer_steps=0,
        forward_adapter_active=False,
        backward_adapter_active=False,
        posterior_active=False,
        calibration_active=False,
        operator_active=False,
        seed_library_id="answer-free-seeds@1",
        seed_skill_ids=("skill-a",),
        retrieval_policy_id="top1@1",
        initial_context_profile="seeded-step-zero@1",
        reasoning_authority="frozen-deterministic",
        reasoning_contract_resolved=True,
        action_policy_authority="adapter-free-evaluation-matched-base",
    )


def test_step_zero_identity_cannot_claim_trained_components() -> None:
    assert _identity().to_value()["optimizer_steps"] == 0
    with pytest.raises(ValueError):
        replace(_identity(), posterior_active=True)
    with pytest.raises(ValueError):
        replace(_identity(), reasoning_contract_resolved=False)


def test_step_zero_identity_accepts_only_frozen_reasoning_authorities() -> None:
    for authority in (
        "frozen-benchmark-matched",
        "frozen-benchmark-conditioned-mixed",
    ):
        assert replace(_identity(), reasoning_authority=authority).reasoning_authority == authority
    with pytest.raises(ValueError):
        replace(_identity(), reasoning_authority="trainable-sampler")


def test_trained_identity_requires_and_records_forward_policy() -> None:
    trained = replace(
        _identity(),
        method_id="skillev-bayesian-improve-trained-step-16@1",
        optimizer_steps=16,
        forward_adapter_active=True,
        posterior_active=True,
        calibration_active=True,
        reasoning_authority="trained-forward-policy-benchmark-conditioned-mixed",
        action_policy_authority="retrieved-skill-orchestration-plus-trained-forward-adapter",
    )

    assert trained.to_value()["optimizer_steps"] == 16
    with pytest.raises(ValueError):
        replace(trained, forward_adapter_active=False)


def test_pair_requires_identical_panel_and_scorer() -> None:
    fields = {
        "pair_id": "pair-1",
        "benchmark": "hotpotqa",
        "task_ids": ("one", "two"),
        "model_service_receipt_id": "service-1",
        "environment_receipt_id": None,
        "evaluator_receipt_id": "scorer-1",
        "backbone_condition_id": "backbone@1",
        "architecture_condition_id": "step0@1",
        "scorer_condition_equal": True,
        "panel_equal": True,
    }
    PairedArchitectureComparison(**fields)
    with pytest.raises(ValueError):
        PairedArchitectureComparison(**{**fields, "panel_equal": False})


def test_paired_outcomes_preserve_all_four_cells() -> None:
    counts = paired_outcome_counts((True, True, False, False), (True, False, True, False))
    assert counts.total == 4
    assert counts.both_pass == 1
    assert counts.backbone_only_pass == 1
    assert counts.step0_only_pass == 1
    assert counts.both_fail == 1
