"""Scientifically typed reference targets for Protocol 13."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from .catalog import Protocol13Benchmark
from .contracts import ExecutionContractV3, ReferenceEligibility


class TargetStatus(StrEnum):
    EXACT_MATCHED = "exact-matched"
    EXTERNAL_MATCHED = "external-matched"
    OWNER_DEFINED_GOAL = "owner-defined-goal"
    PAPER_REPORTED_PROTOCOL_INCOMPLETE = "paper-reported-protocol-incomplete"
    EXTERNAL_AGGREGATE_ANCHOR = "external-aggregate-anchor"
    DIAGNOSTIC = "diagnostic"
    UNDEFINED = "undefined"


class MetricProjection(StrEnum):
    IDENTITY_PERCENT = "identity-percent"
    CLIP_MEAN_TO_UNIT_INTERVAL_PERCENT = "clip-mean-to-unit-interval-percent"


@dataclass(frozen=True, slots=True)
class EvidenceLocator:
    source_kind: str
    repository_or_paper: str
    revision_or_version: str
    path_or_section: str
    supported_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        required = (
            self.source_kind,
            self.repository_or_paper,
            self.revision_or_version,
            self.path_or_section,
        )
        if any(not value.strip() for value in required):
            raise ValueError("target evidence locator is incomplete")
        if len(self.supported_fields) != len(set(self.supported_fields)):
            raise ValueError("target evidence fields must be unique")
        if any(not item.strip() for item in self.supported_fields):
            raise ValueError("target evidence fields must be non-empty")


@dataclass(frozen=True, slots=True)
class ReferencePopulationV4:
    population_id: str
    dataset_revision: str
    selection_rule: str
    expected_count: int
    panel_manifest_id: str | None

    def __post_init__(self) -> None:
        if any(
            not item.strip()
            for item in (self.population_id, self.dataset_revision, self.selection_rule)
        ):
            raise ValueError("reference population identity is incomplete")
        if self.expected_count <= 0:
            raise ValueError("reference population count must be positive")

    def matches(self, execution: ExecutionContractV3) -> bool:
        return (
            self.population_id == execution.population_id
            and self.dataset_revision == execution.dataset_revision
            and self.selection_rule == execution.selection_rule
            and self.expected_count == execution.expected_count
            and self.panel_manifest_id == execution.panel_manifest_id
        )


@dataclass(frozen=True, slots=True)
class ReferenceAggregationV4:
    mode: str
    seeds: tuple[int, ...]
    run_count: int
    reducer: str
    dispersion: str | None

    def __post_init__(self) -> None:
        if not self.mode.strip() or not self.reducer.strip():
            raise ValueError("reference aggregation identity is incomplete")
        if self.run_count <= 0:
            raise ValueError("reference run count must be positive")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("reference seeds must be unique")
        if self.seeds and len(self.seeds) != self.run_count:
            raise ValueError("reference seed count differs from run count")


@dataclass(frozen=True, slots=True)
class ReferenceMetricV3:
    metric_id: str
    unit: str
    denominator_id: str
    projection: MetricProjection
    formula: str
    aggregation: str
    reference_percent: Decimal
    maximum_gap_pp_exclusive: Decimal = Decimal("7.0")
    required_for_formal_gate: bool = True

    def __post_init__(self) -> None:
        identity = (
            self.metric_id,
            self.unit,
            self.denominator_id,
            self.formula,
            self.aggregation,
        )
        if any(not value.strip() for value in identity):
            raise ValueError("reference metric identity is incomplete")
        if not Decimal("0") <= self.reference_percent <= Decimal("100"):
            raise ValueError("reference metric lies outside [0, 100]")
        if self.maximum_gap_pp_exclusive != Decimal("7.0"):
            raise ValueError("Protocol 13 requires a strict 7pp threshold")
        if type(self.required_for_formal_gate) is not bool:
            raise ValueError("metric gate flag must be boolean")


@dataclass(frozen=True, slots=True)
class ReferenceTargetV4:
    benchmark: Protocol13Benchmark
    status: TargetStatus
    observed_execution: ExecutionContractV3
    reference_execution: ExecutionContractV3 | None
    reference_population: ReferencePopulationV4 | None
    reference_aggregation: ReferenceAggregationV4 | None
    metrics: tuple[ReferenceMetricV3, ...]
    evidence: EvidenceLocator | None

    def __post_init__(self) -> None:
        if self.observed_execution.benchmark is not self.benchmark:
            raise ValueError("observed execution belongs to another benchmark")
        if self.status is TargetStatus.UNDEFINED:
            if (
                any(
                    value is not None
                    for value in (
                        self.reference_execution,
                        self.reference_population,
                        self.reference_aggregation,
                        self.evidence,
                    )
                )
                or self.metrics
            ):
                raise ValueError("undefined target cannot carry reference evidence")
            return
        if not self.metrics or self.evidence is None:
            raise ValueError("defined target requires metrics and evidence")
        if self.reference_execution is not None and (
            self.reference_execution.benchmark is not self.benchmark
        ):
            raise ValueError("reference execution belongs to another benchmark")
        ids = tuple(metric.metric_id for metric in self.metrics)
        if len(ids) != len(set(ids)):
            raise ValueError("target metric IDs must be unique")

    @property
    def can_enter_formal_gate(self) -> bool:
        execution = self.observed_execution
        return (
            self.status in {TargetStatus.EXACT_MATCHED, TargetStatus.EXTERNAL_MATCHED}
            and self.reference_execution == execution
            and self.reference_population is not None
            and self.reference_population.matches(execution)
            and self.reference_aggregation is not None
            and execution.reference_eligibility is ReferenceEligibility.FORMAL
            and any(metric.required_for_formal_gate for metric in self.metrics)
        )

    @property
    def evidence_status(self) -> str:
        return self.status.value


ReferenceTargetV3 = ReferenceTargetV4


__all__ = [
    "EvidenceLocator",
    "MetricProjection",
    "ReferenceAggregationV4",
    "ReferenceMetricV3",
    "ReferencePopulationV4",
    "ReferenceTargetV3",
    "ReferenceTargetV4",
    "TargetStatus",
]
