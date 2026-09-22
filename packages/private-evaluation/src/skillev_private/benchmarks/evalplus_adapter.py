"""Pinned EvalPlus result and carrier contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalPlusCandidateVerdict:
    task_id: str
    base_status: str
    plus_status: str

    @property
    def base_passed(self) -> bool:
        return self.base_status.casefold() in {"success", "pass", "passed"}

    @property
    def plus_passed(self) -> bool:
        return self.plus_status.casefold() in {"success", "pass", "passed"}

    @property
    def base_plus_passed(self) -> bool:
        return self.base_passed and self.plus_passed


@dataclass(frozen=True, slots=True)
class EvalPlusCarrierReceipt:
    dataset_task_count: int
    selected_task_count: int
    placeholder_task_count: int
    selected_task_ids: tuple[str, ...]
    headline_denominator: int

    def __post_init__(self) -> None:
        if self.selected_task_count != 128 or len(self.selected_task_ids) != 128:
            raise ValueError("MBPP+ selected panel must contain 128 tasks")
        if len(set(self.selected_task_ids)) != 128:
            raise ValueError("MBPP+ selected task IDs must be unique")
        if self.headline_denominator != self.selected_task_count:
            raise ValueError("MBPP+ headline denominator differs")
        if self.selected_task_count + self.placeholder_task_count != self.dataset_task_count:
            raise ValueError("EvalPlus carrier does not cover the full dataset")


def parse_evalplus_single_candidate(task_id: str, raw: object) -> EvalPlusCandidateVerdict:
    if isinstance(raw, dict) and set(raw).issuperset({"nfiles", "base", "plus"}):
        if raw["nfiles"] != 1:
            raise ValueError(f"EvalPlus task must contain one candidate: {task_id}")
        base = _legacy_status(raw["base"])
        plus = _legacy_status(raw["plus"])
        return EvalPlusCandidateVerdict(task_id, base, plus)
    if not isinstance(raw, list) or len(raw) != 1:
        raise ValueError(f"EvalPlus task must contain one candidate: {task_id}")
    item = raw[0]
    if not isinstance(item, dict):
        raise ValueError("EvalPlus candidate result is not an object")
    if "base_status" in item and "plus_status" in item:
        base = item["base_status"]
        plus = item["plus_status"]
    elif "base" in item and "plus" in item:
        base = _legacy_status(item["base"])
        plus = _legacy_status(item["plus"])
    else:
        raise ValueError("unknown EvalPlus result schema")
    if not isinstance(base, str) or not isinstance(plus, str):
        raise ValueError("EvalPlus statuses must be strings")
    embedded_task_id = item.get("task_id")
    if embedded_task_id is not None and embedded_task_id != task_id:
        raise ValueError("EvalPlus embedded task ID differs")
    return EvalPlusCandidateVerdict(task_id, base, plus)


def _legacy_status(value: object) -> str:
    try:
        status = value[0][0]  # type: ignore[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("invalid legacy EvalPlus status") from exc
    if not isinstance(status, str):
        raise ValueError("legacy EvalPlus status is not text")
    return status


__all__ = [
    "EvalPlusCandidateVerdict",
    "EvalPlusCarrierReceipt",
    "parse_evalplus_single_candidate",
]
