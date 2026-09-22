from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.validate_data import reject_cross_split_overlap, validate_records


def _row(index: int, *, source: str = "HotpotQA") -> dict[str, object]:
    return {
        "answer": f"answer-{index}",
        "context": [],
        "extra": {"metric": "token_f1", "source": source},
        "question": f"public placeholder question {index}",
        "task_type": "multi_hop_qa",
    }


def test_strict_schema_and_count_accept_metadata_only_fixture() -> None:
    report = validate_records([_row(1), _row(2)], split="train", expected_count=2)

    assert report.row_count == 2
    assert report.sources == {"HotpotQA": 2}


def test_missing_data_never_creates_a_synthetic_fallback() -> None:
    with pytest.raises(ValueError):
        validate_records([_row(1)], split="train", expected_count=2)


def test_duplicate_and_cross_split_questions_are_rejected() -> None:
    duplicate = deepcopy(_row(1))
    duplicate["answer"] = "another"
    with pytest.raises(ValueError):
        validate_records([_row(1), duplicate], split="train", expected_count=2)
    with pytest.raises(ValueError):
        reject_cross_split_overlap([_row(1)], [deepcopy(_row(1))])


def test_declared_source_oversampling_is_counted_without_exposing_rows() -> None:
    duplicate = deepcopy(_row(1))
    report = validate_records(
        [_row(1), duplicate],
        split="train",
        expected_count=2,
        allowed_duplicate_extra_rows={"HotpotQA": 1},
    )

    assert report.duplicate_extra_rows == {"HotpotQA": 1}


def test_declared_oversampling_count_must_match_observed_rows() -> None:
    with pytest.raises(ValueError):
        validate_records(
            [_row(1), _row(2)],
            split="train",
            expected_count=2,
            allowed_duplicate_extra_rows={"HotpotQA": 1},
        )


def test_aime_2026_is_rejected_from_training() -> None:
    with pytest.raises(ValueError):
        validate_records([_row(1, source="AIME 2026")], split="train", expected_count=1)


def test_swebench_requires_instance_and_base_commit_metadata() -> None:
    with pytest.raises(ValueError):
        validate_records([_row(1, source="SWE-bench")], split="validation", expected_count=1)
