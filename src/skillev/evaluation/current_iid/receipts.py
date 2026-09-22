"""Public, answer-free run receipts for current-IID aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from skillev.evaluation.current_iid.catalog import CurrentIIDBenchmark
from skillev.evaluation.current_iid.conditions import ExecutionContract
from skillev.experiments.protocol_v11 import BenchmarkV11


@dataclass(frozen=True, slots=True)
class CurrentIIDRunReceipt:
    benchmark: BenchmarkV11
    condition_id: str
    attempt_id: str
    planned_count: int
    final_record_count: int
    definitive_verdict_count: int
    candidate_failure_count: int
    generation_infrastructure_failures: int
    scorer_infrastructure_failures: int
    environment_infrastructure_failures: int
    metrics: dict[str, float]
    population_id: str
    prompt_profile: str
    decoding_profile: str
    actor_route: str
    context_length: int
    adapter_active: bool
    tool_surface: tuple[str, ...]
    grader_profile: str | None
    auxiliary_models: tuple[str, ...] = ()
    scaffold_id: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.condition_id,
            self.attempt_id,
            self.population_id,
            self.prompt_profile,
            self.decoding_profile,
            self.actor_route,
        ):
            if not value.strip():
                raise ValueError("run receipt identity fields must be non-empty")
        expected = 30 if self.benchmark is BenchmarkV11.AIME_2026 else 128
        if self.planned_count != expected:
            raise ValueError(f"{self.benchmark.value} requires {expected} records")
        if self.adapter_active:
            raise ValueError("backbone-only receipt cannot have active adapter")
        if self.context_length <= 0:
            raise ValueError("receipt context length must be positive")
        counts = (
            self.final_record_count,
            self.definitive_verdict_count,
            self.candidate_failure_count,
            self.generation_infrastructure_failures,
            self.scorer_infrastructure_failures,
            self.environment_infrastructure_failures,
        )
        if any(value < 0 for value in counts):
            raise ValueError("run receipt counts must be non-negative")
        if self.final_record_count != self.planned_count:
            raise ValueError("formal current-IID result requires all records")
        if self.definitive_verdict_count != self.planned_count:
            raise ValueError("formal current-IID result requires all verdicts")
        if any(
            not name.strip() or not math.isfinite(value) for name, value in self.metrics.items()
        ):
            raise ValueError("receipt metrics must have IDs and finite values")

    @property
    def infrastructure_failure_count(self) -> int:
        return (
            self.generation_infrastructure_failures
            + self.scorer_infrastructure_failures
            + self.environment_infrastructure_failures
        )


@dataclass(frozen=True, slots=True)
class DiagnosticRunReceipt:
    """Explicitly incomplete diagnostics that cannot enter formal parity."""

    benchmark: BenchmarkV11
    condition_id: str
    attempted_count: int
    definitive_verdict_count: int
    metrics: dict[str, float]
    reason: str


class FinalOutcomeKind(StrEnum):
    SCORED_SUCCESS = "scored-success"
    SCORED_FAILURE = "scored-failure"
    CANDIDATE_INVALID = "candidate-invalid"
    GENERATION_INFRASTRUCTURE = "generation-infrastructure"
    ENVIRONMENT_INFRASTRUCTURE = "environment-infrastructure"
    SCORER_INFRASTRUCTURE = "scorer-infrastructure"


@dataclass(frozen=True, slots=True)
class OutcomeCounts:
    scored_success: int
    scored_failure: int
    candidate_invalid: int
    generation_infrastructure: int
    environment_infrastructure: int
    scorer_infrastructure: int
    detail: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (
            self.scored_success,
            self.scored_failure,
            self.candidate_invalid,
            self.generation_infrastructure,
            self.environment_infrastructure,
            self.scorer_infrastructure,
            *self.detail.values(),
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("outcome counts must be non-negative integers")
        if any(not key.strip() for key in self.detail):
            raise ValueError("outcome detail keys must be non-empty")
        if self.detail and sum(self.detail.values()) != self.definitive:
            raise ValueError("outcome detail must partition definitive outcomes")

    @property
    def definitive(self) -> int:
        return self.scored_success + self.scored_failure + self.candidate_invalid

    @property
    def infrastructure(self) -> int:
        return (
            self.generation_infrastructure
            + self.environment_infrastructure
            + self.scorer_infrastructure
        )

    @property
    def total(self) -> int:
        return self.definitive + self.infrastructure


@dataclass(frozen=True, slots=True)
class MetricObservation:
    metric_id: str
    numerator: Decimal
    denominator: int
    observed_percent: Decimal
    formula: str
    aggregation: str

    def __post_init__(self) -> None:
        if not self.metric_id.strip() or not self.formula.strip() or not self.aggregation.strip():
            raise ValueError("metric observation identity is incomplete")
        if self.denominator <= 0:
            raise ValueError("metric denominator must be positive")
        expected = self.numerator / Decimal(self.denominator) * Decimal("100")
        if abs(expected - self.observed_percent) > Decimal("0.000000001"):
            raise ValueError("metric percent differs from numerator/denominator")


@dataclass(frozen=True, slots=True)
class RunProvenance:
    attempt_id: str
    generation_attempt_id: str
    scoring_attempt_id: str
    protocol_version: str
    generation_code_revision: str
    scoring_code_revision: str
    renderer_code_revision: str
    evaluator_version: str
    started_at: str
    completed_at: str

    def __post_init__(self) -> None:
        values = (
            self.attempt_id,
            self.generation_attempt_id,
            self.scoring_attempt_id,
            self.protocol_version,
            self.generation_code_revision,
            self.scoring_code_revision,
            self.renderer_code_revision,
            self.evaluator_version,
            self.started_at,
            self.completed_at,
        )
        if any(not value.strip() for value in values):
            raise ValueError("run provenance fields must be non-empty")


@dataclass(frozen=True, slots=True)
class Protocol12RunReceipt:
    benchmark: CurrentIIDBenchmark
    execution: ExecutionContract
    planned_count: int
    outcomes: OutcomeCounts
    metrics: tuple[MetricObservation, ...]
    provenance: RunProvenance
    diagnostics: dict[str, Decimal | int | str]

    def __post_init__(self) -> None:
        if self.execution.benchmark is not self.benchmark:
            raise ValueError("receipt execution belongs to another benchmark")
        if self.planned_count != self.execution.expected_count:
            raise ValueError("receipt count differs from its execution contract")
        if self.outcomes.total != self.planned_count:
            raise ValueError("one and only one final outcome is required per planned task")
        ids = tuple(metric.metric_id for metric in self.metrics)
        if len(ids) != len(set(ids)):
            raise ValueError("receipt metric IDs must be unique")

    @property
    def is_formally_complete(self) -> bool:
        return self.outcomes.infrastructure == 0 and self.outcomes.definitive == self.planned_count


CurrentIIDRunReceiptV2 = Protocol12RunReceipt


__all__ = [
    "CurrentIIDRunReceipt",
    "CurrentIIDRunReceiptV2",
    "DiagnosticRunReceipt",
    "FinalOutcomeKind",
    "MetricObservation",
    "OutcomeCounts",
    "Protocol12RunReceipt",
    "RunProvenance",
]
