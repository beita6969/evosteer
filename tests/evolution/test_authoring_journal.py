from __future__ import annotations

import json
from dataclasses import replace

import pytest

from skillev.evolution import AuthoringFailedError, AuthoringResult, RetainAuthoringRequest
from skillev.evolution.authoring import authoring_reservation_id, minimum_legal_retain_draft
from skillev.evolution.authoring_journal import (
    EvolutionAuthoringJournal,
    UnresolvedAuthoringCallError,
)
from skillev.runtime import BudgetLedger, BudgetReservation, BudgetSettlement, BudgetVector
from tests.fakes.skill_author import ScriptedSkillAuthor
from tests.v3_helpers import CharacterTokenizer, make_skill_document


def inputs():
    source = make_skill_document("journal")
    request = RetainAuthoringRequest(source, (), "frozen training evidence", seed=17)
    result = AuthoringResult((minimum_legal_retain_draft(source),))
    return request, result


def ledger(attempt="first"):
    return BudgetLedger(run_id="run", attempt_id=attempt, cap=BudgetVector(model_calls=2))


class ChargedAuthor(ScriptedSkillAuthor):
    def __init__(self, budget, result):
        super().__init__(CharacterTokenizer(), (result,))
        self.budget = budget

    def author(self, request):
        reservation_id = authoring_reservation_id(request)
        self.budget.reserve(
            BudgetReservation(
                reservation_id,
                self.budget.run_id,
                self.budget.attempt_id,
                "author-call",
                BudgetVector(model_calls=1),
            )
        )
        result = super().author(request)
        self.budget.settle(BudgetSettlement(reservation_id, BudgetVector(model_calls=1)))
        return result


def test_recovery_reuses_original_draft_and_charges_usage_once(tmp_path):
    request, result = inputs()
    first = ledger()
    delegate = ChargedAuthor(first, result)
    journal = EvolutionAuthoringJournal(tmp_path, decision={"phase": "phase-1"})
    assert journal.bind(delegate, first).author(request) == result
    assert len(delegate.requests) == 1
    for budget in (first, ledger("recovery")):
        unused = ScriptedSkillAuthor(CharacterTokenizer(), ())
        for _ in range(2):
            restored = EvolutionAuthoringJournal(tmp_path, decision={"phase": "phase-1"})
            assert restored.bind(unused, budget).author(request) == result
        assert unused.requests == ()
        assert budget.settled.model_calls == 1
        assert len(budget.entries) == 1


@pytest.mark.parametrize("failure_type", [AuthoringFailedError, RuntimeError])
def test_unknown_or_failed_author_response_never_becomes_no_op_or_new_draft(tmp_path, failure_type):
    request, _ = inputs()
    failed = ScriptedSkillAuthor(CharacterTokenizer(), (failure_type("lost response"),))
    journal = EvolutionAuthoringJournal(tmp_path, decision={"phase": "phase-1"})
    with pytest.raises(failure_type):
        journal.bind(failed, ledger()).author(request)
    unused = ScriptedSkillAuthor(CharacterTokenizer(), ())
    with pytest.raises(UnresolvedAuthoringCallError):
        journal.bind(unused, ledger()).author(request)
    assert unused.requests == ()
    for state in ("failed", "pending"):
        path = tmp_path / "call-0000.json"
        value = json.loads(path.read_text())
        value["state"] = state
        path.write_text(json.dumps(value))
        unused = ScriptedSkillAuthor(CharacterTokenizer(), ())
        with pytest.raises(UnresolvedAuthoringCallError):
            journal.bind(unused, ledger()).author(request)
        assert unused.requests == ()


def test_recovered_proposal_and_request_must_keep_original_evidence(tmp_path):
    request, result = inputs()
    journal = EvolutionAuthoringJournal(tmp_path, decision={"phase": "phase-1"})
    author = ScriptedSkillAuthor(CharacterTokenizer(), (result,))
    journal.bind(author, ledger()).author(request)
    with pytest.raises(ValueError):
        EvolutionAuthoringJournal(tmp_path, decision={"phase": "different-phase"})
    with pytest.raises(ValueError):
        journal.bind(author, ledger()).author(replace(request, evidence_summary="different cells"))
    assert len(author.requests) == 1
