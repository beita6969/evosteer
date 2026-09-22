"""Strict adapter for EvalPlus revision 26d6d00 result rows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalPlusTaskVerdict:
    task_id: str
    base_status: str
    plus_status: str

    @property
    def base_passed(self) -> bool:
        return self.base_status == "pass"

    @property
    def plus_passed(self) -> bool:
        return self.plus_status == "pass"

    @property
    def base_plus_passed(self) -> bool:
        return self.base_passed and self.plus_passed


def parse_evalplus_26d6d00_task(task_id: str, value: object) -> EvalPlusTaskVerdict:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("expected exactly one completion per EvalPlus task")
    row = value[0]
    if not isinstance(row, dict):
        raise ValueError("EvalPlus task result must be an object")
    required = {"task_id", "base_status", "plus_status"}
    if not required.issubset(row):
        raise ValueError("EvalPlus v26d6d00 result schema differs")
    if row["task_id"] != task_id:
        raise ValueError("EvalPlus task identity differs")
    base = row["base_status"]
    plus = row["plus_status"]
    if not isinstance(base, str) or not isinstance(plus, str):
        raise ValueError("EvalPlus result statuses must be text")
    if base not in {"pass", "fail"} or plus not in {"pass", "fail"}:
        raise ValueError("EvalPlus result status is outside the pinned PASS/FAIL schema")
    return EvalPlusTaskVerdict(task_id, base, plus)


__all__ = ["EvalPlusTaskVerdict", "parse_evalplus_26d6d00_task"]
