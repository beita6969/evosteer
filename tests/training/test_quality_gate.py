from dataclasses import replace

import pytest

from skillev.training.quality_gate import (
    ProtocolProbe,
    QualityGatePolicy,
    QualityRule,
    evaluate_quality,
)


def policy():
    return QualityGatePolicy(
        "test@1", "fixed-panel", "raw", 2, 10, 2, (QualityRule("structure", 0.8, 0.1),)
    )


def probe(step, score):
    return ProtocolProbe(
        f"probe-{step}", "fixed-panel", "raw", f"policy-{step}", step, 12, {"structure": score}
    )


def decide(step, probes=(), baseline=None):
    return evaluate_quality(
        policy(),
        baseline=baseline or probe(0, 1.0),
        probes=probes,
        policy_step=step,
        policy_snapshot_id=f"policy-{step}",
    )


def test_sustained_fixed_panel_regression_survives_restart_and_duplicate_probe():
    first, second = probe(2, 0.7), probe(4, 0.6)
    assert decide(2, (first,)).status == "warning"
    result = decide(4, (first, second))
    assert result.action == "pause-after-commit"
    assert result.status == "regressed"
    assert decide(4, (second, first, first)) == result
    with pytest.raises(ValueError):
        decide(4, (second, replace(second, metrics={"structure": 1.0})))


def test_quality_architecture_and_library_axes_are_not_silent_substitutions():
    rules = replace(policy(), architecture_id="X", library_axis="fixed-initial-library")
    baseline = replace(
        probe(0, 1.0),
        architecture_id="X",
        library_snapshot_id="initial",
        execution_controls={"thinking": False, "scorer": "native"},
    )
    current = replace(baseline, evidence_id="probe-2", policy_step=2, policy_snapshot_id="policy-2")

    def compare(value, selected=rules, library=None):
        return evaluate_quality(
            selected,
            baseline=baseline,
            probes=(value,),
            policy_step=2,
            policy_snapshot_id="policy-2",
            library_snapshot_id=library,
        )

    assert compare(current).status == "verified"
    for changed in (
        replace(current, library_snapshot_id=None),
        replace(current, library_snapshot_id="evolved"),
        replace(current, execution_controls={"thinking": True, "scorer": "native"}),
        replace(current, execution_controls=None),
    ):
        assert compare(changed).status == "metrics-missing"
    combined = replace(rules, library_axis="combined-policy-library")
    evolved = replace(current, library_snapshot_id="evolved")
    assert compare(evolved, combined, "evolved").status == "verified"
    assert compare(evolved, combined, "newer").status == "metrics-missing"


def test_missing_or_stale_or_small_probe_never_passes():
    for current in (
        (),
        (probe(2, None),),
        (replace(probe(2, 1.0), source_question_count=4),),
        (replace(probe(2, 1.0), policy_snapshot_id="old"),),
    ):
        assert decide(2, current).status == "metrics-missing"
        assert decide(2, current).action == "pause-after-commit"


def test_difficult_training_batch_is_not_used_as_fixed_probe():
    assert decide(2, (probe(2, 0.95),)).status == "verified"
    assert decide(3).status == "not-due"
    assert decide(0, baseline=probe(0, 0.5)).action == "pause-after-commit"


def test_initial_baseline_requires_matching_policy_and_actual_source_count():
    assert decide(0).status == "verified"
    assert (
        decide(0, baseline=replace(probe(0, 1.0), panel_id="different")).status == "metrics-missing"
    )
    result = evaluate_quality(
        policy(), baseline=None, probes=(), policy_step=0, policy_snapshot_id="policy-0"
    )
    assert result.action == "pause-after-commit"


@pytest.mark.parametrize(
    ("domain", "required"),
    [
        ("hotpotqa", 2),
        ("triviaqa", 2),
        ("aime-2026", 2),
        ("healthbench", 1),
        ("alfworld", 1),
        ("mbpp-plus", 2),
        ("humaneval", 2),
    ],
)
def test_initial_native_success_floor_is_not_a_later_retention_floor(domain, required):
    metric = f"{domain}/success_fraction"
    rules = replace(
        policy(),
        rules=(QualityRule(metric, 0.0, 1.0, baseline_minimum=required / 4),),
    )
    below = replace(probe(0, 1.0), metrics={metric: (required - 1) / 4, "reward_mean": 1.0})
    accepted = replace(below, metrics={metric: required / 4})

    def evaluate(baseline, step=0, probes=()):
        return evaluate_quality(
            rules,
            baseline=baseline,
            probes=probes,
            policy_step=step,
            policy_snapshot_id=f"policy-{step}",
        )

    assert evaluate(below).action == "pause-after-commit"
    assert evaluate(accepted).status == "verified"
    # A passing current score cannot cure an unqualified initial policy on resume.
    current = replace(probe(2, 1.0), metrics={metric: 1.0})
    assert evaluate(below, 2, (current,)).action == "pause-after-commit"
    # Later regression is governed by the separately declared retention rules.
    current = replace(current, metrics={metric: 0.0})
    assert evaluate(accepted, 2, (current,)).status == "verified"
    missing = replace(below, metrics={"reward_mean": 1.0})
    assert evaluate(missing).status == "metrics-missing"


@pytest.mark.parametrize("floor", [float("nan"), float("inf"), -1.0])
def test_invalid_initial_floor_is_rejected(floor):
    with pytest.raises(ValueError):
        QualityRule("success_fraction", 0.0, 1.0, baseline_minimum=floor)


def test_owner_t0_override_preserves_later_retention_and_missing_evidence():
    rules = replace(
        policy(), initial_admission_override="Owner approved fresh start below T0 floor"
    )
    baseline = probe(0, 0.5)

    def check(step, history=(), initial=baseline):
        return evaluate_quality(
            rules,
            baseline=initial,
            probes=history,
            policy_step=step,
            policy_snapshot_id=f"policy-{step}",
        )

    assert check(0).action == "continue"
    assert check(0, initial=probe(0, None)).status == "metrics-missing"
    assert (
        check(0, initial=replace(baseline, policy_snapshot_id="wrong")).status == "metrics-missing"
    )
    assert check(2, (probe(2, 0.5),)).status == "warning"
    assert check(4, (probe(2, 0.5), probe(4, 0.5))).action == "pause-after-commit"
    assert check(4, (probe(2, 0.5), probe(4, 0.9))).status == "verified"
    assert baseline.metrics == {"structure": 0.5}


def test_owner_t0_override_requires_nonempty_recorded_reason():
    with pytest.raises(ValueError):
        replace(policy(), initial_admission_override=" ")
