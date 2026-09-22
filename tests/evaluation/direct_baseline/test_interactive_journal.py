from pathlib import Path

import pytest
from skillev_private.direct_reference.interactive_journal import (
    InteractiveJournal,
    InteractiveJournalRecord,
    can_resume,
)


def _record(task_id: str, attempt: int, error: str | None) -> InteractiveJournalRecord:
    return InteractiveJournalRecord(
        task_id=task_id,
        request_attempt=attempt,
        execution_contract_id="protocol13-contract",
        outcome={"infrastructure_error": error, "reward": None},
    )


def test_only_typed_infrastructure_outcome_can_be_replaced(tmp_path: Path) -> None:
    journal = InteractiveJournal(
        tmp_path / "interactive.jsonl", execution_contract_id="protocol13-contract"
    )
    failed = _record("task-1", 1, "EnvironmentCreationError")
    journal.append(failed)
    assert can_resume(failed)
    final = _record("task-1", 2, None)
    journal.append(final)
    assert journal.load() == {"task-1": final}
    with pytest.raises(ValueError, match="definitive"):
        journal.append(_record("task-1", 3, "DirectGenerationError"))


def test_journal_rejects_contract_and_attempt_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "interactive.jsonl"
    journal = InteractiveJournal(path, execution_contract_id="protocol13-contract")
    with pytest.raises(ValueError, match="next attempt"):
        journal.append(_record("task-1", 2, "DirectGenerationError"))
    other = InteractiveJournal(path, execution_contract_id="different-contract")
    journal.append(_record("task-1", 1, None))
    with pytest.raises(ValueError, match="another execution contract"):
        other.load()
