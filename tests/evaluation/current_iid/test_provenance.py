import pytest

from skillev.evaluation.current_iid.provenance import (
    validate_hotpot_full_context,
    validate_humaneval_request_receipt,
)


def test_hotpot_requires_ten_answer_free_passages() -> None:
    records = [
        {"context": list(range(10)), "public_fields": ("question", "context")} for _ in range(128)
    ]
    validate_hotpot_full_context(records)
    records[0] = {"context": list(range(10)), "public_fields": ("answer",)}
    with pytest.raises(ValueError):
        validate_hotpot_full_context(records)


def test_humaneval_is_one_request_without_repair() -> None:
    ids = [f"HumanEval/{index}" for index in range(128)]
    validate_humaneval_request_receipt(ids, request_count=128, candidate_count=128, repair_count=0)
    with pytest.raises(ValueError):
        validate_humaneval_request_receipt(
            ids, request_count=128, candidate_count=128, repair_count=1
        )
