"""Admission decisions from complete pairs, persistent budgets and frozen menus."""

import copy
import json
import math
from dataclasses import replace

import pytest

from skillev.contracts import stable_hash
from skillev.evolution.paired_trials import (
    ComparisonContext,
    PairedTrialOutcome,
    PairScheduler,
    TrialInitialCondition,
    TrialOutcome,
)
from skillev.evolution.validated_admission import (
    AdmissionConfig,
    AdmissionLedger,
    SkillEntry,
    SkillStatus,
    exact_binomial_sign_p,
)
from skillev.runtime.contracts import SkillManifest
from skillev.runtime.skills import SkillApplicability, SkillDocument, SkillRequirement


def context():
    return ComparisonContext("math", "reference", "executor", "background", "value", "environment")


def skill(name="candidate", family="math"):
    body = "Verify the calculation for " + name
    return SkillEntry(name, family, stable_hash(body), body=body)


def candidate_ledger(config=None):
    ledger = AdmissionLedger(config)
    ledger.propose(skill(), "author-1")
    comparison = ledger.register_comparison("candidate", context())
    return ledger, comparison


def outcome(ledger, comparison, index, rewards=(1.0, 0.0), task_id=None):
    registered = ledger.comparison(comparison)
    initial = TrialInitialCondition(
        task_id or f"q-{index}", f"reset-{index}", "solver", registered.context
    )
    pair = PairScheduler.plan(
        comparison, registered.skill_id, initial, f"{comparison}/pair-{index}"
    )
    return PairedTrialOutcome(
        TrialOutcome(pair.positive, rewards[0], side_effect_free=True),
        TrialOutcome(pair.negative, rewards[1], side_effect_free=True),
    )


def test_candidate_can_be_tried_before_validation_and_batch_menu_never_changes():
    ledger, comparison = candidate_ledger()
    before = ledger.freeze_menu("batch-1", "math")
    assert before.visible_skill_ids == ("candidate",)
    assert before.bodies["candidate"] == skill().body
    assert before.entries[0].status is SkillStatus.CANDIDATE
    for index in range(1, 16):
        decision = ledger.record_pair(outcome(ledger, comparison, index))
    assert decision.action == "promote"
    assert ledger.status("candidate") is SkillStatus.VALIDATED
    assert ledger.freeze_menu("batch-1", "math") == before
    after = ledger.freeze_menu("batch-2", "math")
    assert after.entries[0].status is SkillStatus.VALIDATED
    assert after.entries[0].admission_basis == "paired-sign-test"
    assert after.menu_id != before.menu_id
    assert after.background_id(("candidate",)) == before.background_id(("candidate",))


def test_one_proposal_per_author_window_and_capacity_checks_are_atomic():
    ledger, _ = candidate_ledger()
    frozen = ledger.to_value()
    with pytest.raises(ValueError, match="one proposal"):
        ledger.propose(skill("another", "code"), "author-1")
    with pytest.raises(ValueError, match="capacity"):
        ledger.propose(skill("another"), "author-2")
    assert ledger.to_value() == frozen
    ledger.propose(skill("another", "code"), "author-2")
    assert ledger.freeze_menu("batch").visible_skill_ids == ("another", "candidate")


def test_run_budget_configuration_cannot_be_replaced_midstream():
    ledger, comparison = candidate_ledger()
    ledger.record_pair(outcome(ledger, comparison, 1))
    with pytest.raises(AttributeError):
        ledger.config = AdmissionConfig(alpha=0.9)


def test_ties_change_effect_denominator_without_spending_or_advancing_look():
    ledger, comparison = candidate_ledger()
    first = ledger.record_pair(outcome(ledger, comparison, 1))
    assert first.alpha_each_direction == pytest.approx(0.05 / 8)
    assert first.observation_index == 1
    spent = ledger.alpha_spent
    tied = ledger.record_pair(outcome(ledger, comparison, 2, (0.5, 0.8)))
    assert (tied.wins, tied.losses, tied.ties) == (1, 0, 1)
    assert tied.effect_size == 0.5
    assert tied.observation_index == 1
    assert tied.p_positive is None
    assert tied.alpha_each_direction == 0
    assert ledger.alpha_spent == spent


def test_validation_round_spends_once_for_multiple_new_pairs_and_never_for_ties_or_replay():
    ledger, comparison = candidate_ledger()
    for index in range(1, 4):
        decision = ledger.record_pair(outcome(ledger, comparison, index), validate=False)
        assert decision.action == "defer"
    assert ledger.alpha_spent == 0
    assert ledger.comparison(comparison).last_tested_discordants == 0
    assert ledger.comparison(comparison).observation_index == 0
    decision = ledger.validate_comparison(comparison)
    assert decision.p_positive == 2**-3
    assert decision.alpha_each_direction == pytest.approx(0.05 / 8)
    assert decision.observation_index == 1
    assert ledger.comparison(comparison).last_tested_discordants == 3
    ledger.record_pair(outcome(ledger, comparison, 4, (1, 1)), validate=False)
    saved = ledger.to_value()
    assert ledger.validate_comparison(comparison) is None
    assert ledger.to_value() == saved
    restored = AdmissionLedger.from_value(json.loads(json.dumps(saved)))
    assert restored.validate_comparison(comparison) is None
    restored.record_pair(outcome(restored, comparison, 5, (0, 1)), validate=False)
    result = restored.validate_comparison(comparison)
    assert result.observation_index == 2
    assert result.alpha_each_direction == pytest.approx(0.05 / (2 * 6 * 2))
    assert result.effect_size == (3 - 1) / 5
    assert restored.comparison(comparison).last_tested_discordants == 4


def test_full_validated_slots_and_stalled_candidate_do_not_block_evidence_based_retirement():
    ledger = AdmissionLedger()
    for index in range(3):
        ledger.add_seed(skill(f"seed-{index}"))
    ledger.propose(skill(), "window")
    candidate = ledger.register_comparison("candidate", context())
    for index in range(15):
        ledger.record_pair(outcome(ledger, candidate, index), validate=False)
    assert ledger.validate_comparison(candidate).action == "defer_capacity"
    assert ledger.trial_priority("math")[0].skill_id == "candidate"
    priority = ledger.trial_priority("math", reevaluation_due=True)
    assert [entry.skill_id for entry in priority] == ["seed-0", "seed-1", "seed-2"]
    reevaluation = ledger.register_comparison("seed-0", context(), reevaluation=True)
    for index in range(15):
        ledger.record_pair(outcome(ledger, reevaluation, index, (0, 1)), validate=False)
    assert ledger.validate_comparison(reevaluation).action == "retire"
    assert ledger.status("candidate") is SkillStatus.CANDIDATE
    assert ledger.paired_observation_count("candidate") == 15
    assert all(
        item.skill_id != "seed-0" for item in ledger.trial_priority("math", reevaluation_due=True)
    )
    # Opening a slot alone does not manufacture another statistical observation.
    assert ledger.validate_comparison(candidate) is None
    ledger.record_pair(outcome(ledger, candidate, 16), validate=False)
    assert ledger.validate_comparison(candidate).action == "promote"
    assert AdmissionLedger.from_value(ledger.to_value()).to_value() == ledger.to_value()


def test_candidate_priority_counts_whole_pairs_including_ties_across_contexts():
    ledger = AdmissionLedger(AdmissionConfig(max_candidates_per_family=2))
    ledger.propose(skill("a"), "window-a")
    ledger.propose(skill("b"), "window-b")
    comparison = ledger.register_comparison("a", context())
    ledger.record_pair(outcome(ledger, comparison, 1, (0, 0)), validate=False)
    assert [item.skill_id for item in ledger.trial_priority("math")] == ["b", "a"]
    assert [item.skill_id for item in ledger.trial_priority("math", reevaluation_due=True)] == [
        "b",
        "a",
    ]
    assert ledger.alpha_spent == 0
    assert ledger.trial_priority("unknown") == ()


def test_incomplete_and_duplicate_whole_pairs_do_not_update_counts():
    ledger, comparison = candidate_ledger()
    pair = outcome(ledger, comparison, 1)
    incomplete = PairedTrialOutcome(
        pair.positive, replace(pair.negative, reward=None, completed=False)
    )
    frozen = ledger.to_value()
    assert ledger.record_pair(incomplete) is None
    assert not ledger.has_task(comparison, "q-1")
    assert ledger.to_value() == frozen
    ledger.record_pair(pair)
    assert ledger.has_task(comparison, "q-1")
    after = ledger.to_value()
    assert ledger.record_pair(pair) is None
    assert ledger.to_value() == after
    with pytest.raises(ValueError, match="reused"):
        ledger.record_pair(PairedTrialOutcome(replace(pair.positive, reward=0), pair.negative))
    assert ledger.to_value() == after


def test_same_task_cannot_be_renamed_by_pair_id_to_create_independent_evidence():
    ledger, comparison = candidate_ledger()
    ledger.record_pair(outcome(ledger, comparison, 1, task_id="source-1"))
    with pytest.raises(ValueError, match="repeated task"):
        ledger.record_pair(outcome(ledger, comparison, 2, task_id="source-1"))
    assert ledger.comparison(comparison).wins == 1


def test_changed_comparison_background_is_rejected_even_if_pair_arms_match():
    ledger, comparison = candidate_ledger()
    pair = outcome(ledger, comparison, 1)
    initial = replace(
        pair.positive.spec.initial_condition,
        comparison_context=replace(context(), value_snapshot_id="new-value"),
    )
    wrong = PairScheduler.plan(comparison, "candidate", initial)
    with pytest.raises(ValueError, match="frozen comparison context"):
        ledger.record_pair(
            PairedTrialOutcome(TrialOutcome(wrong.positive, 1), TrialOutcome(wrong.negative, 0))
        )


def test_frozen_condition_change_requires_new_index_without_refunding_alpha():
    ledger, comparison = candidate_ledger()
    ledger.record_pair(outcome(ledger, comparison, 1))
    spent = ledger.alpha_spent
    changed = replace(context(), value_snapshot_id="value-next")
    with pytest.raises(ValueError, match="open comparison"):
        ledger.register_comparison("candidate", context())
    ledger.supersede_comparison(comparison, "value-head snapshot changed")
    assert ledger.to_value()["events"][-1]["kind"] == "supersede"
    new = ledger.register_comparison("candidate", changed)
    assert ledger.comparison(new).comparison_index == 2
    assert ledger.alpha_spent == spent
    assert ledger.status("candidate") is SkillStatus.CANDIDATE
    assert AdmissionLedger.from_value(ledger.to_value()).to_value() == ledger.to_value()


@pytest.mark.parametrize("reevaluation", [False, True])
def test_retirement_is_absorbing_across_concurrent_comparison_contexts(reevaluation):
    ledger = AdmissionLedger()
    if reevaluation:
        ledger.add_seed(skill())
    else:
        ledger.propose(skill(), "window")
    negative = ledger.register_comparison("candidate", context(), reevaluation=reevaluation)
    positive = ledger.register_comparison(
        "candidate",
        replace(context(), environment_config_id="other-environment"),
        reevaluation=reevaluation,
    )
    for index in range(15):
        ledger.record_pair(outcome(ledger, negative, index, (0, 1)), validate=False)
        ledger.record_pair(outcome(ledger, positive, index, (1, 0)), validate=False)
    assert ledger.alpha_spent == 0
    assert ledger.validate_comparison(negative).action == "retire"
    spent = ledger.alpha_spent
    assert spent == pytest.approx(ledger.config.alpha / 4)
    decision = ledger.validate_comparison(positive)
    assert decision.action == "comparison_closed"
    assert decision.status_before is decision.status_after is SkillStatus.RETIRED
    assert decision.p_positive is decision.p_negative is None
    assert decision.alpha_each_direction == 0
    assert decision.observation_index == 0
    assert ledger.status("candidate") is SkillStatus.RETIRED
    assert ledger.comparison(positive).closed
    assert ledger.comparison(positive).wins == 15
    assert ledger.comparison(positive).last_tested_discordants == 0
    assert ledger.alpha_spent == spent
    assert ledger.freeze_menu("after-retirement").visible_skill_ids == ()
    # Late complete evidence remains auditable without reopening the decision.
    late = ledger.record_pair(outcome(ledger, positive, 16))
    assert late.action == "comparison_closed"
    assert late.wins == 16
    assert ledger.status("candidate") is SkillStatus.RETIRED
    assert ledger.alpha_spent == spent
    assert ledger.validate_comparison(positive) is None
    saved = json.loads(json.dumps(ledger.to_value()))
    assert AdmissionLedger.from_value(saved).to_value() == saved


def test_retired_skill_closes_remaining_tie_only_context_without_a_statistical_look():
    ledger, negative = candidate_ledger()
    tied = ledger.register_comparison(
        "candidate", replace(context(), executor_snapshot_id="other-executor")
    )
    ledger.record_pair(outcome(ledger, tied, 0, (1, 1)), validate=False)
    for index in range(15):
        ledger.record_pair(outcome(ledger, negative, index, (0, 1)), validate=False)
    ledger.validate_comparison(negative)
    spent = ledger.alpha_spent
    # The other context can receive already scheduled pairs before it is closed.
    pending = ledger.record_pair(outcome(ledger, tied, 1, (0, 0)), validate=False)
    assert pending.ties == 2
    assert not ledger.comparison(tied).closed
    decision = ledger.validate_comparison(tied)
    assert decision.action == "comparison_closed"
    assert decision.ties == 2
    assert decision.observation_index == 0
    assert ledger.comparison(tied).closed
    assert ledger.alpha_spent == spent
    saved = json.loads(json.dumps(ledger.to_value()))
    assert AdmissionLedger.from_value(saved).to_value() == saved


def test_validated_skill_gets_new_registered_comparison_and_can_retire():
    ledger, comparison = candidate_ledger()
    for index in range(1, 16):
        ledger.record_pair(outcome(ledger, comparison, index))
    initial_spend = ledger.alpha_spent
    first_menu = ledger.freeze_menu("validated")
    new = ledger.register_comparison(
        "candidate", replace(context(), background_menu_id="new-background"), reevaluation=True
    )
    assert ledger.comparison(new).comparison_index == 2
    assert ledger.comparison(new).observation_index == 0
    assert ledger.alpha_spent == initial_spend
    for index in range(1, 30):
        decision = ledger.record_pair(outcome(ledger, new, index, (0, 1)))
        if decision.action == "retire":
            break
    assert ledger.status("candidate") is SkillStatus.RETIRED
    assert initial_spend < ledger.alpha_spent <= ledger.config.alpha
    assert ledger.freeze_menu("validated") == first_menu
    assert ledger.freeze_menu("retired").visible_skill_ids == ()
    with pytest.raises(ValueError):
        ledger.register_comparison("candidate", context(), reevaluation=True)


def test_full_validated_capacity_defers_candidate_without_overfilling():
    ledger = AdmissionLedger()
    for index in range(3):
        ledger.add_seed(skill(f"seed-{index}"))
    assert all(item.admission_basis == "configured-seed" for item in ledger.skills)
    ledger.propose(skill(), "author")
    comparison = ledger.register_comparison("candidate", context())
    for index in range(1, 16):
        result = ledger.record_pair(outcome(ledger, comparison, index))
    assert result.action == "defer_capacity"
    assert ledger.status("candidate") is SkillStatus.CANDIDATE
    assert not ledger.comparison(comparison).closed
    assert len(ledger.freeze_menu("all").entries) == 4
    with pytest.raises(ValueError, match="capacity"):
        ledger.add_seed(skill("seed-4"))


def test_library_capacity_defaults_and_global_limit_cover_seeds_and_proposals():
    assert AdmissionConfig().max_library_size == 60
    assert AdmissionConfig().max_library_per_family == 12
    ledger = AdmissionLedger(AdmissionConfig(max_library_size=2))
    ledger.add_seed(skill("seed", "code"))
    assert ledger.can_propose("math")
    ledger.propose(skill(), "window")
    saved = ledger.to_value()
    assert not ledger.can_propose("new-family")
    with pytest.raises(ValueError, match="total library capacity"):
        ledger.propose(skill("new", "new-family"), "unused-window")
    with pytest.raises(ValueError, match="total library capacity"):
        ledger.add_seed(skill("new", "new-family"))
    assert ledger.to_value() == saved
    restored = AdmissionLedger.from_value(json.loads(json.dumps(saved)))
    assert restored.config.max_library_size == 2
    assert not restored.can_propose("new-family")


def test_retired_history_still_consumes_per_family_library_capacity():
    ledger, comparison = candidate_ledger(AdmissionConfig(max_library_per_family=2))
    assert not ledger.can_propose("math")  # Candidate slot is occupied.
    for index in range(1, 16):
        ledger.record_pair(outcome(ledger, comparison, index, (0, 1)))
    assert ledger.status("candidate") is SkillStatus.RETIRED
    assert ledger.can_propose("math")  # One history slot remains.
    ledger.add_seed(skill("seed"))
    assert not ledger.can_propose("math")
    assert ledger.can_propose("code")
    assert ledger.freeze_menu("remaining").visible_skill_ids == ("seed",)
    saved = ledger.to_value()
    with pytest.raises(ValueError, match="total library capacity for family"):
        ledger.propose(skill("third"), "new-window")
    assert ledger.to_value() == saved
    restored = AdmissionLedger.from_value(json.loads(json.dumps(saved)))
    assert restored.config.max_library_per_family == 2
    assert not restored.can_propose("math")
    assert len(restored.skills) == 2


def test_capacity_preflight_is_read_only_and_full_validated_slots_allow_a_candidate_trial():
    ledger = AdmissionLedger()
    for index in range(3):
        ledger.add_seed(skill(f"seed-{index}"))
    saved = ledger.to_value()
    assert ledger.can_propose("math")
    assert ledger.to_value() == saved
    with pytest.raises(ValueError):
        ledger.can_propose("")
    ledger.propose(skill(), "trial")
    assert not ledger.can_propose("math")


def test_exact_binomial_probabilities_and_large_tail_are_stable():
    assert exact_binomial_sign_p(0, 0) == 1
    assert exact_binomial_sign_p(2, 1) == 0.75
    assert exact_binomial_sign_p(15, 15) == 2**-15
    assert exact_binomial_sign_p(10001, 5001) == pytest.approx(0.5, abs=1e-10)
    assert math.isfinite(exact_binomial_sign_p(10000, 8000))
    assert 0 <= exact_binomial_sign_p(10000, 8000) <= 1
    with pytest.raises(ValueError):
        exact_binomial_sign_p(2, 3)


def test_json_roundtrip_preserves_ids_budget_dedup_snapshots_and_rejects_tampering():
    ledger, comparison = candidate_ledger()
    old_menu = ledger.freeze_menu("old")
    for index in range(1, 17):
        ledger.record_pair(outcome(ledger, comparison, index))
    assert ledger.comparison(comparison).closed
    assert ledger.comparison(comparison).observation_index == 15
    saved = json.loads(json.dumps(ledger.to_value()))
    restored = AdmissionLedger.from_value(saved)
    assert restored.to_value() == saved
    assert restored.freeze_menu("old") == old_menu
    assert restored.record_pair(outcome(restored, comparison, 1)) is None
    tampered = copy.deepcopy(saved)
    tampered["state"]["comparisons"][0]["wins"] += 1
    with pytest.raises(ValueError, match="source events"):
        AdmissionLedger.from_value(tampered)


def test_existing_skill_document_mapping_preserves_body_and_identity():
    applicability = SkillApplicability(("math",), ("*",), (), ())
    requirements = (SkillRequirement("check", "Check the arithmetic."),)
    content = {
        "title": "Arithmetic",
        "summary": "Check calculations.",
        "instructions": "Recompute.",
        "applicability": applicability.to_value(),
        "requirements": [item.to_value() for item in requirements],
    }
    document = SkillDocument(
        SkillManifest(
            "arithmetic", "1", stable_hash(content), "input", "output", "test", stable_hash("test")
        ),
        "Arithmetic",
        "Check calculations.",
        "Recompute.",
        applicability,
        requirements,
    )
    entry = SkillEntry.from_document(document, "math")
    assert json.loads(entry.body) == content
    ledger = AdmissionLedger()
    ledger.add_seed(entry)
    assert (
        AdmissionLedger.from_value(json.loads(json.dumps(ledger.to_value()))).to_value()
        == ledger.to_value()
    )


@pytest.mark.parametrize(
    "configuration",
    [
        {"alpha": 0},
        {"alpha": 1},
        {"max_validated_per_family": 0},
        {"success_threshold": float("nan")},
        {"max_library_size": 0},
        {"max_library_per_family": -1},
        {"max_library_size": True},
    ],
)
def test_invalid_statistical_configuration_is_rejected(configuration):
    with pytest.raises(ValueError):
        AdmissionConfig(**configuration)
