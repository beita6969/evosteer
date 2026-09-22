"""Injected frozen author, public evidence and measured one-call window behavior."""

import json
from dataclasses import replace

import pytest

from skillev.contracts.canonical import canonical_json, stable_hash
from skillev.contracts.evosteer import DecisionRecord, EvoTask, EvoTrajectory
from skillev.contracts.evosteer_risk import TrajectoryRiskAssessment
from skillev.evolution.evosteer_author import (
    FrozenSkillAuthor,
    SkillAuthorCallError,
    SkillAuthorConfig,
    SkillAuthorOutputError,
    SkillAuthorWindowError,
)
from skillev.runtime.budget_ledger import (
    BudgetExceededError,
    BudgetLedger,
    DuplicateBudgetReservationError,
)
from skillev.runtime.contracts import BudgetReservation, BudgetSettlement, BudgetVector


def procedure():
    return {
        "name": "Independent calculation check",
        "description": "Verify intermediate numerical steps before finalizing.",
        "trigger": "A multi-step calculation has produced a tentative answer.",
        "plan": ["Recompute the critical step independently.", "Resolve conflicting results."],
        "pitfall": "Repeating the same mistaken derivation.",
        "constraint": "Use only the supplied problem and available tools.",
    }


class Model:
    reference_id = "separate-frozen-author@1"

    def __init__(self, response=None, error=None):
        self.response = canonical_json(procedure()) if response is None else response
        self.error = error
        self.calls = []

    def frozen_text(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if self.error:
            raise self.error
        return self.response, 101, 23


def budget(calls=10):
    return BudgetLedger(
        run_id="run",
        attempt_id="attempt",
        cap=BudgetVector(input_tokens=100_000, output_tokens=20_000, model_calls=calls),
    )


def assessed(source, *, accepted=True, side_effect_free=True):
    return replace(
        source,
        risk=TrajectoryRiskAssessment(
            "test-risk-assessor",
            source.risk_evidence_id,
            accepted,
            side_effect_free,
            () if accepted else ("unsafe execution",),
        ),
    )


def trajectory(sample="success", task_id="q", reward=1.0, family="math"):
    task = EvoTask(
        task_id, family, f"Compute and independently verify the numerical result of task {task_id}."
    )
    action = canonical_json({"kind": "STOP"})
    state = {"public": True, "sample": sample}
    decision = DecisionRecord(
        stable_hash(state),
        canonical_json(state),
        action,
        (1,),
        (2,),
        ((2,),),
        (0.0,) * 30,
        (0.0,),
        0.5,
    )
    terminal = {
        "stopped": True,
        "poisoned": False,
        "runtime_id": "runtime-secret",
        "history": [
            {
                "action": {"kind": "ADD_AGENT", "role_id": "solver"},
                "observation": {
                    "node_result": {"output": "Public calculation attempt", "gold": "GOLD-SECRET"},
                    "node_request": {"skill_body": "NODE-REQUEST-SECRET"},
                    "private_payload": "PRIVATE-SECRET",
                    "reservation_id": "BUDGET-SECRET",
                },
            }
        ],
    }
    return assessed(
        EvoTrajectory(
            sample,
            "batch",
            task,
            "current",
            "actor",
            "reference",
            "executor",
            "menu",
            "value",
            "statistics",
            (decision,),
            canonical_json(terminal),
            reward,
            "Actual generated output",
            canonical_json({}),
        )
    )


def test_independent_injected_author_produces_one_canonical_procedure_and_charges_usage():
    model, ledger = Model(), budget()
    author = FrozenSkillAuthor(model, budget=ledger)
    entry = author.propose((trajectory(),), family="math", window_id="window", seed=17)
    assert json.loads(entry.body) == procedure()
    assert entry.content_hash == stable_hash(entry.body)
    assert entry.family == "math"
    assert len(model.calls) == 1
    assert model.calls[0][1] == {
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "seed": 17,
        "input_limit": 8192,
    }
    assert ledger.settled == BudgetVector(input_tokens=101, output_tokens=23, model_calls=1)
    assert ledger.reserved == BudgetVector()
    assert author.reports[0].author_identity == "separate-frozen-author@1"
    assert author.reports[0].skill_id == entry.skill_id
    with pytest.raises(SkillAuthorWindowError):
        author.propose((trajectory(),), family="math", window_id="window", seed=18)
    assert len(model.calls) == 1


def test_success_failure_same_task_contrast_precedes_unrelated_successes_and_is_bounded():
    model = Model()
    author = FrozenSkillAuthor(model, budget=budget(), config=SkillAuthorConfig(max_trajectories=2))
    author.propose(
        (trajectory("unrelated", "a"), trajectory("weak", "z", 0), trajectory("strong", "z")),
        family="math",
        window_id="contrast",
        seed=1,
    )
    assert author.reports[0].selected_sample_ids == ("strong", "weak")
    evidence = json.loads(model.calls[0][0].split("\n", 1)[1])["scored_public_trajectories"]
    assert len(evidence) == 2
    assert [item["reward"] for item in evidence] == [1, 0]
    assert [item["task_id"] for item in evidence] == ["z", "z"]


def test_author_sees_public_execution_output_and_scores_but_not_private_runtime_payloads():
    model = Model()
    author = FrozenSkillAuthor(model, budget=budget())
    author.propose((trajectory(),), family="math", window_id="public", seed=1)
    text = model.calls[0][0]
    assert "Public calculation attempt" in text
    assert "Actual generated output" in text
    assert '"reward":1.0' in text
    for secret in (
        "GOLD-SECRET",
        "NODE-REQUEST-SECRET",
        "PRIVATE-SECRET",
        "BUDGET-SECRET",
        "runtime-secret",
    ):
        assert secret not in text


def test_long_public_histories_are_excerpted_without_dropping_the_selected_contrast():
    model = Model()
    config = SkillAuthorConfig(input_limit=4000, max_trajectories=2)
    author = FrozenSkillAuthor(model, budget=budget(), config=config)
    examples = []
    for sample, reward in (("success", 1), ("failure", 0)):
        source = trajectory(sample, reward=reward)
        terminal = json.loads(source.terminal_state_json)
        terminal["history"] *= 30
        for event in terminal["history"]:
            event["observation"]["node_result"]["output"] = "A long public observation. " * 500
        examples.append(
            assessed(replace(source, terminal_state_json=canonical_json(terminal), risk=None))
        )
    author.propose(tuple(examples), family="math", window_id="long", seed=0)
    prompt = model.calls[0][0]
    assert len(prompt.encode("utf-8")) <= config.input_limit
    material = json.loads(prompt.split("\n", 1)[1])
    assert len(material["scored_public_trajectories"]) == 2
    assert material["excerpt_limits"]["history_events"] < config.max_history_events


def test_empty_or_unhealthy_family_evidence_returns_none_without_a_call():
    model, ledger = Model(), budget()
    author = FrozenSkillAuthor(model, budget=ledger)
    poisoned = replace(
        trajectory(),
        terminal_state_json=canonical_json({"stopped": True, "poisoned": True}),
        risk=None,
    )
    assert (
        author.propose(
            (poisoned, trajectory("code", family="code")), family="math", window_id="empty", seed=0
        )
        is None
    )
    assert not model.calls
    assert ledger.settled == BudgetVector()
    assert author.reports[0].status == "no-eligible-evidence"


def test_explicit_null_is_a_charged_no_proposal_not_a_fabricated_skill():
    model, ledger = Model("null"), budget()
    author = FrozenSkillAuthor(model, budget=ledger)
    assert author.propose((trajectory(),), family="math", window_id="none", seed=0) is None
    assert author.reports[0].status == "no-proposal"
    assert ledger.settled.model_calls == 1


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "```json\n{}\n```",
        "{}",
        "[]",
        canonical_json({**procedure(), "answer": "a source-specific answer"}),
        canonical_json({**procedure(), "plan": []}),
        canonical_json({**procedure(), "constraint": "  "}),
        canonical_json({**procedure(), "name": ["invalid"]}),
        '{"name":"a","name":"b"}',
    ],
)
def test_invalid_output_is_explicit_and_real_usage_remains_charged(response):
    model, ledger = Model(response), budget()
    author = FrozenSkillAuthor(model, budget=ledger)
    with pytest.raises(SkillAuthorOutputError) as captured:
        author.propose((trajectory(),), family="math", window_id="bad", seed=0)
    assert captured.value.raw_output == response
    assert captured.value.reason
    assert author.reports[0].status == "invalid-output"
    assert ledger.settled == BudgetVector(input_tokens=101, output_tokens=23, model_calls=1)
    with pytest.raises(SkillAuthorWindowError):
        author.propose((trajectory(),), family="math", window_id="bad", seed=1)


def test_obvious_verbatim_source_task_copy_is_rejected_without_claiming_full_leak_detection():
    source = trajectory()
    model = Model(canonical_json({**procedure(), "plan": source.task.prompt}))
    author = FrozenSkillAuthor(model, budget=budget())
    with pytest.raises(SkillAuthorOutputError, match="copies a source task"):
        author.propose((source,), family="math", window_id="copy", seed=1)


def test_budget_envelope_rejection_occurs_before_any_model_call():
    model = Model()
    author = FrozenSkillAuthor(model, budget=budget(calls=0))
    with pytest.raises(BudgetExceededError):
        author.propose((trajectory(),), family="math", window_id="budget", seed=1)
    assert not model.calls


def test_backend_failure_leaves_unknown_usage_reserved_and_does_not_retry():
    model, ledger = Model(error=RuntimeError("backend failed")), budget()
    author = FrozenSkillAuthor(model, budget=ledger)
    with pytest.raises(SkillAuthorCallError):
        author.propose((trajectory(),), family="math", window_id="fail", seed=1)
    assert ledger.reserved.model_calls == 1
    assert ledger.settled.model_calls == 0
    assert author.reports[0].status == "call-failed-usage-unknown"
    with pytest.raises(SkillAuthorWindowError):
        author.propose((trajectory(),), family="math", window_id="fail", seed=2)
    restarted = FrozenSkillAuthor(model, budget=ledger)
    with pytest.raises(DuplicateBudgetReservationError):
        restarted.propose((trajectory(),), family="math", window_id="fail", seed=3)
    assert len(model.calls) == 1


def test_author_requires_explicit_frozen_identity_and_rejects_midrun_identity_change():
    model = Model()
    model.reference_id = ""
    with pytest.raises(ValueError, match="reference_id or frozen_identity"):
        FrozenSkillAuthor(model, budget=budget())
    model.frozen_identity = "independent-author"
    author = FrozenSkillAuthor(model, budget=budget())
    model.frozen_identity = "changed-author"
    with pytest.raises(ValueError, match="identity changed"):
        author.propose((trajectory(),), family="math", window_id="changed", seed=1)
    assert not model.calls


def test_duplicate_source_evidence_is_not_silently_used_twice():
    model = Model()
    author = FrozenSkillAuthor(model, budget=budget())
    source = trajectory()
    with pytest.raises(ValueError, match="repeats a trajectory"):
        author.propose((source, source), family="math", window_id="duplicate", seed=1)
    assert not model.calls


@pytest.mark.parametrize("verdict", ["missing", "rejected", "side-effects"])
def test_author_requires_explicit_evidence_bound_safe_risk_assessment(verdict):
    source = trajectory()
    if verdict == "missing":
        terminal = json.loads(source.terminal_state_json)
        terminal["side_effect_free"] = True  # A bare outcome flag is not an assessment.
        source = replace(source, risk=None, terminal_state_json=canonical_json(terminal))
    else:
        source = assessed(
            source, accepted=verdict != "rejected", side_effect_free=verdict != "side-effects"
        )
    model, ledger = Model(), budget()
    author = FrozenSkillAuthor(model, budget=ledger)
    assert author.propose((source,), family="math", window_id="unsafe", seed=0) is None
    assert not model.calls
    assert not ledger.entries


def test_checkpoint_restores_measured_budget_and_all_consumed_windows_without_refund():
    model, original_budget = Model(), budget(calls=2)
    author = FrozenSkillAuthor(model, budget=original_budget)
    author.propose((trajectory(),), family="math", window_id="candidate", seed=1)
    model.response = "null"
    author.propose((trajectory(),), family="math", window_id="null", seed=2)
    author.propose((), family="math", window_id="empty", seed=3)
    saved = json.loads(json.dumps(author.state_dict()))

    restored_model, restored_budget = Model(), budget(calls=2)
    restored = FrozenSkillAuthor(restored_model, budget=restored_budget)
    restored.load_state_dict(saved)
    restored.load_state_dict(saved)
    assert restored.budget is restored_budget
    assert restored.state_dict() == saved
    assert restored.reports == author.reports
    assert restored_budget.settled == original_budget.settled
    assert restored_budget.available == original_budget.available
    for window in ("candidate", "null", "empty"):
        with pytest.raises(SkillAuthorWindowError):
            restored.propose((trajectory(),), family="math", window_id=window, seed=4)
    with pytest.raises(BudgetExceededError):
        restored.propose((trajectory(),), family="math", window_id="new", seed=4)
    assert not restored_model.calls


def test_new_window_after_restore_charges_on_top_of_saved_usage():
    author = FrozenSkillAuthor(Model(), budget=budget())
    author.propose((trajectory(),), family="math", window_id="first", seed=1)
    restored = FrozenSkillAuthor(Model(), budget=budget())
    restored.load_state_dict(author.state_dict())
    restored.propose((trajectory(),), family="math", window_id="second", seed=2)
    assert restored.budget.settled == BudgetVector(
        input_tokens=202, output_tokens=46, model_calls=2
    )


def test_unknown_usage_checkpoint_keeps_reservation_and_failed_window_consumed():
    author = FrozenSkillAuthor(Model(error=RuntimeError("failed")), budget=budget())
    with pytest.raises(SkillAuthorCallError):
        author.propose((trajectory(),), family="math", window_id="unknown", seed=0)
    restored = FrozenSkillAuthor(Model(), budget=budget())
    restored.load_state_dict(author.state_dict())
    assert restored.budget.reserved == SkillAuthorConfig().maximum
    assert restored.budget.settled == BudgetVector()
    assert restored.state_dict() == author.state_dict()
    with pytest.raises(SkillAuthorWindowError):
        restored.propose((trajectory(),), family="math", window_id="unknown", seed=1)
    assert not restored.model.calls


def test_invalid_output_checkpoint_preserves_charge_and_explicit_failure_report():
    author = FrozenSkillAuthor(Model("not JSON"), budget=budget())
    with pytest.raises(SkillAuthorOutputError):
        author.propose((trajectory(),), family="math", window_id="invalid", seed=0)
    restored = FrozenSkillAuthor(Model(), budget=budget())
    restored.load_state_dict(author.state_dict())
    assert restored.reports == author.reports
    assert restored.budget.settled.model_calls == 1


@pytest.mark.parametrize(
    "tamper",
    [
        "identity",
        "config",
        "run",
        "attempt",
        "cap",
        "totals",
        "usage",
        "reservation",
        "repeated-window",
        "lost-report",
        "lost-window",
        "unknown-format",
        "unknown-state-field",
    ],
)
def test_checkpoint_rejects_incompatible_or_inconsistent_state_before_mutation(tamper):
    author = FrozenSkillAuthor(Model(), budget=budget())
    author.propose((trajectory(),), family="math", window_id="saved", seed=0)
    saved = author.state_dict()
    if tamper == "identity":
        saved["author_identity"] = "different-author"
    elif tamper == "config":
        saved["config"]["temperature"] = 0.8
    elif tamper in {"run", "attempt"}:
        field = f"{tamper}_id"
        saved["budget"][field] = "different"
        saved["budget"]["entries"][0]["reservation"][field] = "different"
    elif tamper == "cap":
        saved["budget"]["cap"]["model_calls"] += 1
    elif tamper == "totals":
        saved["budget"]["settled"]["model_calls"] = 0
    elif tamper == "usage":
        saved["reports"][0]["input_tokens"] += 1
    elif tamper == "reservation":
        saved["reports"][0]["reservation_id"] = "missing"
    elif tamper == "repeated-window":
        saved["reports"].append(saved["reports"][0])
    elif tamper == "lost-report":
        saved["reports"] = []
        saved["used_window_ids"] = []
    elif tamper == "lost-window":
        saved["used_window_ids"] = []
    elif tamper == "unknown-format":
        saved["format"] = "future-format"
    else:
        saved["extra"] = True
    restored = FrozenSkillAuthor(Model(), budget=budget())
    before = restored.state_dict()
    with pytest.raises(ValueError):
        restored.load_state_dict(saved)
    assert restored.state_dict() == before


def test_checkpoint_cannot_roll_back_a_nonempty_budget_or_forget_a_no_call_window():
    author = FrozenSkillAuthor(Model(), budget=budget())
    saved = author.state_dict()
    author.propose((trajectory(),), family="math", window_id="spent", seed=1)
    before = author.state_dict()
    with pytest.raises(ValueError, match="cannot overwrite"):
        author.load_state_dict(saved)
    assert author.state_dict() == before
    # The caller can share a ledger that already contains unrelated charges.
    injected = FrozenSkillAuthor(Model(), budget=author.budget)
    with pytest.raises(ValueError, match="cannot overwrite.*budget"):
        injected.load_state_dict(saved)
    empty = FrozenSkillAuthor(Model(), budget=budget())
    empty.propose((), family="math", window_id="no-call", seed=0)
    with pytest.raises(ValueError, match="cannot overwrite.*window"):
        empty.load_state_dict(saved)


def test_checkpoint_restores_historical_actuals_without_replaying_maxima_in_sorted_order():
    ledger = BudgetLedger(run_id="r", attempt_id="a", cap=BudgetVector(model_calls=100))
    for identity, maximum, actual in (("z", 100, 1), ("a", 99, 99)):
        ledger.reserve(
            BudgetReservation(
                identity, "r", "a", "other-component", BudgetVector(model_calls=maximum)
            )
        )
        ledger.settle(BudgetSettlement(identity, BudgetVector(model_calls=actual)))
    author = FrozenSkillAuthor(Model(), budget=ledger)
    restored_ledger = BudgetLedger(run_id="r", attempt_id="a", cap=ledger.cap)
    restored = FrozenSkillAuthor(Model(), budget=restored_ledger)
    restored.load_state_dict(author.state_dict())
    assert restored_ledger.settled.model_calls == 100
    assert restored.state_dict() == author.state_dict()
