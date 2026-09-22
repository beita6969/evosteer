"""Answer-free provenance guards for fixed QA and code panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def validate_hotpot_full_context(records: Sequence[Mapping[str, object]]) -> None:
    if len(records) != 128:
        raise ValueError("HotpotQA current-IID panel requires 128 records")
    for record in records:
        context = record.get("context")
        if not isinstance(context, list) or len(context) != 10:
            raise ValueError("HotpotQA generation context must contain exactly ten passages")
        forbidden = {"answer", "supporting_facts", "supporting_labels", "accepted_answers"}
        public_fields = record.get("public_fields", ())
        if not isinstance(public_fields, list | tuple | set | frozenset):
            raise ValueError("HotpotQA public_fields must be a collection")
        if forbidden.intersection(public_fields):
            raise ValueError("HotpotQA public context contains target-derived fields")


def validate_humaneval_request_receipt(
    task_ids: Sequence[str], *, request_count: int, candidate_count: int, repair_count: int
) -> None:
    if len(task_ids) != 128 or len(set(task_ids)) != 128:
        raise ValueError("HumanEval requires a fixed unique 128-ID manifest")
    if request_count != 128 or candidate_count != 128 or repair_count != 0:
        raise ValueError("HumanEval requires exactly one un-repaired candidate per task")
