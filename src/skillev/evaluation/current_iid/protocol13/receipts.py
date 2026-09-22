"""Answer-free, conserved run receipts for Protocol 13."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .catalog import Protocol13Benchmark
from .contracts import ExecutionContractV3
from .targets import MetricProjection


@dataclass(frozen=True, slots=True)
class OutcomeCountsV4:
    """Conserved outcome counts without inventing a binary quality threshold.

    Every planned item is either definitively scored, candidate-invalid, or an
    infrastructure failure.  Benchmarks with a native binary verdict may also
    provide the optional success/failure partition.  Continuous metrics such as
    HealthBench deliberately leave that partition absent.
    """

    definitive_scored: int
    candidate_invalid: int
    generation_infrastructure: int
    environment_infrastructure: int
    scorer_infrastructure: int
    binary_success: int | None = None
    binary_failure: int | None = None
    detail: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (
            self.definitive_scored,
            self.candidate_invalid,
            self.generation_infrastructure,
            self.environment_infrastructure,
            self.scorer_infrastructure,
            *self.detail.values(),
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("outcome counts must be non-negative integers")
        if any(not name.strip() for name in self.detail):
            raise ValueError("outcome detail keys must be non-empty")
        if (self.binary_success is None) != (self.binary_failure is None):
            raise ValueError("binary outcome partition is incomplete")
        if self.binary_success is not None:
            if type(self.binary_success) is not int or self.binary_success < 0:
                raise ValueError("binary success must be a non-negative integer")
            if type(self.binary_failure) is not int or self.binary_failure < 0:
                raise ValueError("binary failure must be a non-negative integer")
            if self.binary_success + self.binary_failure != self.definitive_scored:
                raise ValueError("binary outcomes must partition definitive scores")
        if self.detail and sum(self.detail.values()) != self.definitive_scored:
            raise ValueError("outcome detail must partition definitive scores")

    @property
    def definitive(self) -> int:
        return self.definitive_scored + self.candidate_invalid

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

    @property
    def scored_success(self) -> int:
        """Compatibility projection for binary benchmark renderers."""

        return self.binary_success or 0

    @property
    def scored_failure(self) -> int:
        """Compatibility projection for binary benchmark renderers."""

        return self.binary_failure or 0


# Source compatibility for integrations that still import the old class name.
OutcomeCountsV3 = OutcomeCountsV4


@dataclass(frozen=True, slots=True)
class MetricObservationV3:
    metric_id: str
    unit: str
    numerator: Decimal
    denominator: int
    denominator_id: str
    projection: MetricProjection
    observed_percent: Decimal
    formula: str
    aggregation: str

    def __post_init__(self) -> None:
        identity = (
            self.metric_id,
            self.unit,
            self.denominator_id,
            self.formula,
            self.aggregation,
        )
        if any(not value.strip() for value in identity):
            raise ValueError("metric observation identity is incomplete")
        if type(self.denominator) is not int or self.denominator <= 0:
            raise ValueError("metric denominator must be a positive integer")
        raw_percent = self.numerator / Decimal(self.denominator) * Decimal("100")
        expected = raw_percent
        if self.projection is MetricProjection.CLIP_MEAN_TO_UNIT_INTERVAL_PERCENT:
            expected = min(Decimal("100"), max(Decimal("0"), raw_percent))
        if abs(expected - self.observed_percent) > Decimal("0.000000001"):
            raise ValueError("metric observation differs from its projection contract")


@dataclass(frozen=True, slots=True)
class RunProvenanceV3:
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
        if any(
            not value.strip()
            for value in (
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
        ):
            raise ValueError("run provenance fields must be non-empty")
        if self.protocol_version != "skillev-benchmark-protocol@13":
            raise ValueError("receipt provenance does not belong to Protocol 13")


@dataclass(frozen=True, slots=True)
class Protocol13RunReceipt:
    benchmark: Protocol13Benchmark
    execution: ExecutionContractV3
    planned_count: int
    outcomes: OutcomeCountsV4
    metrics: tuple[MetricObservationV3, ...]
    provenance: RunProvenanceV3
    diagnostics: dict[str, Decimal | int | str]
    runtime_execution_attempt_id: str = "legacy-unvalidated"
    runtime_contract_matched: bool = False
    final_panel_used_for_selection: bool = True

    def __post_init__(self) -> None:
        if self.execution.benchmark is not self.benchmark:
            raise ValueError("receipt execution belongs to another benchmark")
        if self.planned_count != self.execution.expected_count:
            raise ValueError("planned count differs from execution contract")
        if self.outcomes.total != self.planned_count:
            raise ValueError("exactly one final outcome is required per planned task")
        ids = tuple(metric.metric_id for metric in self.metrics)
        if len(ids) != len(set(ids)):
            raise ValueError("receipt metric IDs must be unique")
        for name, value in self.diagnostics.items():
            if not name.strip() or isinstance(value, float):
                raise ValueError("diagnostics must use named Decimal/int/string values")
        if not self.runtime_execution_attempt_id.strip():
            raise ValueError("runtime execution attempt ID must be non-empty")
        if type(self.runtime_contract_matched) is not bool:
            raise ValueError("runtime contract match flag must be boolean")
        if type(self.final_panel_used_for_selection) is not bool:
            raise ValueError("final-panel selection flag must be boolean")

    @property
    def is_formally_complete(self) -> bool:
        return (
            self.outcomes.infrastructure == 0
            and self.outcomes.definitive == self.planned_count
            and self.runtime_contract_matched
            and not self.final_panel_used_for_selection
        )


__all__ = [
    "MetricObservationV3",
    "OutcomeCountsV3",
    "OutcomeCountsV4",
    "Protocol13RunReceipt",
    "RunProvenanceV3",
]
