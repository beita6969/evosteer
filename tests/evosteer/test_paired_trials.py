"""Reference intervention identity and complete-pair evidence boundaries."""

from dataclasses import replace

import pytest

from skillev.evolution.paired_trials import (
    ComparisonContext,
    PairedTrialOutcome,
    PairedTrialSpec,
    PairScheduler,
    TrialInitialCondition,
    TrialOutcome,
)


def context():
    return ComparisonContext(
        "math", "reference-1", "executor-1", "background-1", "value-1", "env-1"
    )


def specification():
    return PairScheduler.plan(
        "comparison-1", "skill-1", TrialInitialCondition("q-1", "reset-1", "solver", context())
    )


def test_first_binding_and_lifetime_control_exclusion_use_same_reference():
    pair = specification()
    assert pair.positive.bound_skill_ids == ("skill-1",)
    assert pair.negative.bound_skill_ids == ()
    assert pair.negative.excluded_skill_ids == ("skill-1",)
    assert pair.positive.excluded_skill_ids == ()
    assert (
        pair.positive.continuation_reference_id
        == pair.negative.continuation_reference_id
        == "reference-1"
    )
    assert pair.positive.first_role == pair.negative.first_role == "solver"
    assert specification() == pair


@pytest.mark.parametrize("field", ["task_id", "reset_id", "first_role"])
def test_mismatched_pair_initial_condition_is_rejected(field):
    pair = specification()
    initial = replace(pair.negative.initial_condition, **{field: "different"})
    with pytest.raises(ValueError, match="initial_condition"):
        PairedTrialSpec(pair.positive, replace(pair.negative, initial_condition=initial))


@pytest.mark.parametrize(
    "field",
    [
        "task_family",
        "reference_snapshot_id",
        "executor_snapshot_id",
        "background_menu_id",
        "value_snapshot_id",
        "environment_config_id",
    ],
)
def test_changed_reference_executor_value_or_background_is_rejected(field):
    pair = specification()
    changed = replace(context(), **{field: "different"})
    observed = replace(pair.positive.initial_condition, comparison_context=changed)
    with pytest.raises(ValueError, match="observed execution identity"):
        TrialOutcome(pair.positive, 1.0, observed_initial_condition=observed)


def test_pair_requires_two_opposite_arms_and_exact_pair_identity():
    pair = specification()
    with pytest.raises(ValueError):
        PairedTrialSpec(pair.positive, pair.positive)
    with pytest.raises(ValueError, match="pair_id"):
        PairedTrialSpec(pair.positive, replace(pair.negative, pair_id="different"))
    with pytest.raises(ValueError, match="exclusion"):
        TrialOutcome(pair.negative, 0.0, control_exclusion_honored=False)


def test_incomplete_or_side_effecting_outcomes_are_not_eligible():
    pair = specification()
    incomplete = PairedTrialOutcome(
        TrialOutcome(pair.positive, 1, side_effect_free=True),
        TrialOutcome(pair.negative, None, completed=False, side_effect_free=True),
    )
    assert not incomplete.eligible
    side_effect = PairedTrialOutcome(
        TrialOutcome(pair.positive, 1, side_effect_free=False),
        TrialOutcome(pair.negative, 0, side_effect_free=True),
    )
    assert not side_effect.eligible
    complete = PairedTrialOutcome(
        TrialOutcome(pair.positive, 1, side_effect_free=True),
        TrialOutcome(pair.negative, 0, side_effect_free=True),
    )
    assert complete.eligible
    assert PairedTrialOutcome.from_value(complete.to_value()) == complete
    assert PairedTrialOutcome.from_value(incomplete.to_value()) == incomplete


def test_missing_explicit_risk_confirmation_is_not_eligible_pair_evidence():
    pair = specification()
    unassessed = PairedTrialOutcome(TrialOutcome(pair.positive, 1), TrialOutcome(pair.negative, 0))
    assert not unassessed.eligible
    assert not unassessed.positive.side_effect_free


@pytest.mark.parametrize("reward", [-0.1, 1.1, float("nan"), float("inf"), True, None])
def test_completed_trial_reward_is_bounded_and_required(reward):
    with pytest.raises(ValueError):
        TrialOutcome(specification().positive, reward)
