"""Matched, pre-frozen target identities for Protocol 14."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from .catalog import Protocol14Benchmark
from .contracts import ExecutionContractV4, FormalEligibility


class TargetStatus(StrEnum):
    FROZEN_MATCHED_REFERENCE = "frozen-matched-reference"
    PUBLISHED_EXTERNAL_ANCHOR = "published-external-anchor"
    DIAGNOSTIC = "diagnostic"
    PENDING = "pending"


class TargetScope(StrEnum):
    STANDALONE_BENCHMARK = "standalone-benchmark"
    COMPOSITE = "composite"


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
        if not self.supported_fields or len(self.supported_fields) != len(
            set(self.supported_fields)
        ):
            raise ValueError("target evidence fields must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class ReferenceMetricV4:
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
        if not Decimal(0) <= self.reference_percent <= Decimal(100):
            raise ValueError("reference metric lies outside [0, 100]")
        if self.maximum_gap_pp_exclusive != Decimal("7.0"):
            raise ValueError("Protocol 14 uses a strict 7 percentage-point threshold")


@dataclass(frozen=True, slots=True)
class PublishedAnchor:
    label: str
    scope: TargetScope
    metric_id: str
    value_percent: Decimal
    evidence: EvidenceLocator
    usable_as_formal_target: bool = False

    def __post_init__(self) -> None:
        if not self.label.strip() or not self.metric_id.strip():
            raise ValueError("published anchor identity is incomplete")
        if not Decimal(0) <= self.value_percent <= Decimal(100):
            raise ValueError("published anchor lies outside [0, 100]")
        if self.scope is TargetScope.COMPOSITE and self.usable_as_formal_target:
            raise ValueError("a composite anchor cannot gate a standalone benchmark")


@dataclass(frozen=True, slots=True)
class ReferenceTargetV5:
    benchmark: Protocol14Benchmark
    status: TargetStatus
    scope: TargetScope
    observed_execution: ExecutionContractV4
    reference_execution: ExecutionContractV4 | None
    metrics: tuple[ReferenceMetricV4, ...]
    evidence: EvidenceLocator | None
    reference_receipt_id: str | None
    frozen_at: str | None
    published_anchors: tuple[PublishedAnchor, ...] = ()

    def __post_init__(self) -> None:
        if self.observed_execution.benchmark is not self.benchmark:
            raise ValueError("observed execution belongs to another benchmark")
        ids = tuple(metric.metric_id for metric in self.metrics)
        if len(ids) != len(set(ids)):
            raise ValueError("target metric IDs must be unique")
        if self.status is TargetStatus.FROZEN_MATCHED_REFERENCE:
            if self.scope is not TargetScope.STANDALONE_BENCHMARK:
                raise ValueError("formal target must be standalone")
            if self.reference_execution != self.observed_execution:
                raise ValueError("formal reference execution differs from candidate execution")
            required = (self.evidence, self.reference_receipt_id, self.frozen_at)
            if any(value is None for value in required) or not self.metrics:
                raise ValueError("frozen target lacks evidence, receipt, time, or metrics")
        elif any(
            value is not None
            for value in (self.reference_execution, self.reference_receipt_id, self.frozen_at)
        ):
            raise ValueError("non-frozen target cannot carry a frozen reference identity")
        if self.reference_receipt_id is not None and not self.reference_receipt_id.strip():
            raise ValueError("reference receipt ID must be non-empty")
        if self.frozen_at is not None and not self.frozen_at.strip():
            raise ValueError("target freeze time must be non-empty")
        if self.benchmark is Protocol14Benchmark.MBPP_PLUS:
            if any(metric.metric_id == "evalplus_aggregate" for metric in self.metrics):
                raise ValueError("EvalPlus aggregate cannot be the MBPP+ target")

    @property
    def can_enter_formal_gate(self) -> bool:
        return (
            self.status is TargetStatus.FROZEN_MATCHED_REFERENCE
            and self.scope is TargetScope.STANDALONE_BENCHMARK
            and self.reference_execution == self.observed_execution
            and self.observed_execution.formal_eligibility is FormalEligibility.FORMAL
            and bool(self.metrics)
            and any(metric.required_for_formal_gate for metric in self.metrics)
            and self.evidence is not None
            and self.reference_receipt_id is not None
            and self.frozen_at is not None
        )


__all__ = [
    "EvidenceLocator",
    "MetricProjection",
    "PublishedAnchor",
    "ReferenceMetricV4",
    "ReferenceTargetV5",
    "TargetScope",
    "TargetStatus",
]
