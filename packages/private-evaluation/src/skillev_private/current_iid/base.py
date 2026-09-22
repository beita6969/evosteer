"""Shared per-task outcome conservation for Protocol 12 runners."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from skillev.evaluation.current_iid.conditions import ExecutionContract
from skillev.evaluation.current_iid.receipts import (
    FinalOutcomeKind,
    MetricObservation,
    OutcomeCounts,
    Protocol12RunReceipt,
    RunProvenance,
)

INFRA_KINDS = frozenset(
    {
        FinalOutcomeKind.GENERATION_INFRASTRUCTURE,
        FinalOutcomeKind.ENVIRONMENT_INFRASTRUCTURE,
        FinalOutcomeKind.SCORER_INFRASTRUCTURE,
    }
)


@dataclass(frozen=True, slots=True)
class PerTaskFinalOutcome:
    task_id: str
    kind: FinalOutcomeKind
    metrics: dict[str, Decimal]
    detail_code: str

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.detail_code.strip():
            raise ValueError("per-task outcome identity is incomplete")
        if self.kind in INFRA_KINDS and self.metrics:
            raise ValueError("infrastructure outcome cannot carry definitive metrics")


def outcomes_to_counts(rows: Sequence[PerTaskFinalOutcome]) -> OutcomeCounts:
    by_kind = Counter(row.kind for row in rows)
    detail = Counter(row.detail_code for row in rows if row.kind not in INFRA_KINDS)
    return OutcomeCounts(
        scored_success=by_kind[FinalOutcomeKind.SCORED_SUCCESS],
        scored_failure=by_kind[FinalOutcomeKind.SCORED_FAILURE],
        candidate_invalid=by_kind[FinalOutcomeKind.CANDIDATE_INVALID],
        generation_infrastructure=by_kind[FinalOutcomeKind.GENERATION_INFRASTRUCTURE],
        environment_infrastructure=by_kind[FinalOutcomeKind.ENVIRONMENT_INFRASTRUCTURE],
        scorer_infrastructure=by_kind[FinalOutcomeKind.SCORER_INFRASTRUCTURE],
        detail=dict(detail),
    )


@dataclass(frozen=True, slots=True)
class MetricAggregationSpec:
    metric_id: str
    value_key: str
    formula: str
    aggregation: str


def build_protocol12_receipt(
    *,
    execution: ExecutionContract,
    outcomes: Sequence[PerTaskFinalOutcome],
    metric_specs: Sequence[MetricAggregationSpec],
    provenance: RunProvenance,
    diagnostics: dict[str, Decimal | int | str] | None = None,
) -> Protocol12RunReceipt:
    """Build a receipt only after one conserved final outcome exists per task."""

    if len(outcomes) != execution.expected_count:
        raise ValueError("per-task outcomes differ from the execution population")
    if len({item.task_id for item in outcomes}) != len(outcomes):
        raise ValueError("per-task final outcomes must be task-unique")
    counts = outcomes_to_counts(outcomes)
    definitive = tuple(item for item in outcomes if item.kind not in INFRA_KINDS)
    metrics: list[MetricObservation] = []
    for spec in metric_specs:
        if not all(spec.value_key in item.metrics for item in definitive):
            raise ValueError(f"metric {spec.metric_id} is absent from a definitive outcome")
        numerator = sum((item.metrics[spec.value_key] for item in definitive), Decimal(0))
        denominator = len(definitive)
        if denominator <= 0:
            raise ValueError("a metric cannot be aggregated without definitive outcomes")
        metrics.append(
            MetricObservation(
                metric_id=spec.metric_id,
                numerator=numerator,
                denominator=denominator,
                observed_percent=numerator / Decimal(denominator) * Decimal(100),
                formula=spec.formula,
                aggregation=spec.aggregation,
            )
        )
    return Protocol12RunReceipt(
        benchmark=execution.benchmark,
        execution=execution,
        planned_count=execution.expected_count,
        outcomes=counts,
        metrics=tuple(metrics),
        provenance=provenance,
        diagnostics={} if diagnostics is None else diagnostics,
    )


__all__ = [
    "INFRA_KINDS",
    "MetricAggregationSpec",
    "PerTaskFinalOutcome",
    "build_protocol12_receipt",
    "outcomes_to_counts",
]
