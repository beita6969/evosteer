"""Read-only evaluation of one immutable validation manifest per checkpoint."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FixedValidationCase:
    task_id: str
    benchmark: str


@dataclass(frozen=True, slots=True)
class FixedValidationOutcome:
    task_id: str
    benchmark: str
    native_metric: float
    completion: bool
    parse_invalid: bool
    horizon: bool
    infrastructure_failure: bool


@dataclass(frozen=True, slots=True)
class FixedValidationReport:
    checkpoint_id: str
    per_domain_mean: dict[str, float]
    macro_mean: float
    completion_rate: float
    parse_invalid_rate: float
    horizon_rate: float
    infrastructure_failure_count: int


async def run_fixed_validation(
    *,
    checkpoint_id: str,
    cases: tuple[FixedValidationCase, ...],
    evaluate: Callable[[FixedValidationCase], Awaitable[FixedValidationOutcome]],
    mutation_state: Callable[[], Mapping[str, object]],
) -> FixedValidationReport:
    if not checkpoint_id.strip() or not cases:
        raise ValueError("fixed validation requires a checkpoint and cases")
    if len({item.task_id for item in cases}) != len(cases):
        raise ValueError("fixed validation manifest contains duplicate task IDs")
    before = dict(mutation_state())
    outcomes = tuple([await evaluate(case) for case in cases])
    if dict(mutation_state()) != before:
        raise RuntimeError("fixed validation mutated training state")
    by_domain: dict[str, list[float]] = {}
    for outcome in outcomes:
        by_domain.setdefault(outcome.benchmark, []).append(outcome.native_metric)
    per_domain = {benchmark: sum(values) / len(values) for benchmark, values in by_domain.items()}
    count = len(outcomes)
    return FixedValidationReport(
        checkpoint_id,
        per_domain,
        sum(per_domain.values()) / len(per_domain),
        sum(item.completion for item in outcomes) / count,
        sum(item.parse_invalid for item in outcomes) / count,
        sum(item.horizon for item in outcomes) / count,
        sum(item.infrastructure_failure for item in outcomes),
    )


__all__ = [
    "FixedValidationCase",
    "FixedValidationOutcome",
    "FixedValidationReport",
    "run_fixed_validation",
]
