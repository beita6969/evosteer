"""Answer-free, denominator-conserving Protocol 14 run receipts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from .catalog import Protocol14Benchmark
from .contracts import ExecutionContractV4
from .targets import MetricProjection


class AttemptRole(StrEnum):
    REFERENCE = "reference"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class OutcomeCountsV5:
    scored_valid: int
    model_output_invalid: int
    generation_infrastructure: int
    environment_infrastructure: int
    scorer_infrastructure: int
    binary_success: int | None = None
    binary_failure: int | None = None
    definitive_detail: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (
            self.scored_valid,
            self.model_output_invalid,
            self.generation_infrastructure,
            self.environment_infrastructure,
            self.scorer_infrastructure,
            *self.definitive_detail.values(),
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("outcome counts must be non-negative integers")
        if any(not key.strip() for key in self.definitive_detail):
            raise ValueError("outcome detail keys must be non-empty")
        if (self.binary_success is None) != (self.binary_failure is None):
            raise ValueError("binary outcome partition is incomplete")
        if self.binary_success is not None:
            if type(self.binary_success) is not int or type(self.binary_failure) is not int:
                raise ValueError("binary outcomes must be integers")
            if self.binary_success < 0 or self.binary_failure < 0:
                raise ValueError("binary outcomes must be non-negative")
            if self.binary_success + self.binary_failure != self.definitive:
                raise ValueError("binary outcomes must include model-output invalid failures")
        if self.definitive_detail and sum(self.definitive_detail.values()) != self.definitive:
            raise ValueError("definitive detail must partition all definitive outcomes")

    @property
    def definitive(self) -> int:
        return self.scored_valid + self.model_output_invalid

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

    def to_mapping(self) -> dict[str, object]:
        return {
            "scored_valid": self.scored_valid,
            "model_output_invalid": self.model_output_invalid,
            "generation_infrastructure": self.generation_infrastructure,
            "environment_infrastructure": self.environment_infrastructure,
            "scorer_infrastructure": self.scorer_infrastructure,
            "binary_success": self.binary_success,
            "binary_failure": self.binary_failure,
            "definitive_detail": self.definitive_detail,
        }


@dataclass(frozen=True, slots=True)
class MetricObservationV4:
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
            raise ValueError("metric denominator must be positive")
        raw = self.numerator / Decimal(self.denominator) * Decimal(100)
        expected = raw
        if self.projection is MetricProjection.CLIP_MEAN_TO_UNIT_INTERVAL_PERCENT:
            expected = min(Decimal(100), max(Decimal(0), raw))
        if abs(expected - self.observed_percent) > Decimal("0.000000001"):
            raise ValueError("metric observation differs from its formula")

    def to_mapping(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "unit": self.unit,
            "numerator": str(self.numerator),
            "denominator": self.denominator,
            "denominator_id": self.denominator_id,
            "projection": self.projection.value,
            "observed_percent": str(self.observed_percent),
            "formula": self.formula,
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True, slots=True)
class Protocol14RunReceipt:
    benchmark: Protocol14Benchmark
    role: AttemptRole
    execution: ExecutionContractV4
    attempt_id: str
    planned_count: int
    outcomes: OutcomeCountsV5
    metrics: tuple[MetricObservationV4, ...]
    generation_code_revision: str
    scoring_code_revision: str
    evaluator_version: str
    started_at: str
    completed_at: str
    runtime_contract_matched: bool
    final_panel_used_for_selection: bool
    diagnostics: dict[str, Decimal | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.execution.benchmark is not self.benchmark:
            raise ValueError("receipt execution belongs to another benchmark")
        if self.planned_count != self.execution.expected_count:
            raise ValueError("planned count differs from execution")
        if self.outcomes.total != self.planned_count:
            raise ValueError("exactly one terminal outcome is required per planned row")
        required = (
            self.attempt_id,
            self.generation_code_revision,
            self.scoring_code_revision,
            self.evaluator_version,
            self.started_at,
            self.completed_at,
        )
        if any(not value.strip() for value in required):
            raise ValueError("receipt provenance is incomplete")
        ids = tuple(metric.metric_id for metric in self.metrics)
        if len(ids) != len(set(ids)):
            raise ValueError("receipt metric IDs must be unique")
        if any(
            not key.strip() or isinstance(value, float) for key, value in self.diagnostics.items()
        ):
            raise ValueError("diagnostics must use named Decimal/int/string scalars")

    @property
    def is_formally_complete(self) -> bool:
        return (
            self.outcomes.infrastructure == 0
            and self.outcomes.definitive == self.planned_count
            and self.runtime_contract_matched
            and not self.final_panel_used_for_selection
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "format": "skillev-protocol14-run-receipt@1",
            "benchmark": self.benchmark.value,
            "role": self.role.value,
            "condition_id": self.execution.condition_id,
            "attempt_id": self.attempt_id,
            "planned_count": self.planned_count,
            "outcomes": self.outcomes.to_mapping(),
            "metrics": [metric.to_mapping() for metric in self.metrics],
            "generation_code_revision": self.generation_code_revision,
            "scoring_code_revision": self.scoring_code_revision,
            "evaluator_version": self.evaluator_version,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "runtime_contract_matched": self.runtime_contract_matched,
            "final_panel_used_for_selection": self.final_panel_used_for_selection,
            "diagnostics": {
                key: str(value) if isinstance(value, Decimal) else value
                for key, value in self.diagnostics.items()
            },
        }


def write_run_receipt(path: Path, receipt: Protocol14RunReceipt) -> None:
    path.write_text(
        json.dumps(receipt.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_run_receipt(path: Path, *, execution: ExecutionContractV4) -> Protocol14RunReceipt:
    root = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or set(root) != {
        "format",
        "benchmark",
        "role",
        "condition_id",
        "attempt_id",
        "planned_count",
        "outcomes",
        "metrics",
        "generation_code_revision",
        "scoring_code_revision",
        "evaluator_version",
        "started_at",
        "completed_at",
        "runtime_contract_matched",
        "final_panel_used_for_selection",
        "diagnostics",
    }:
        raise ValueError("Protocol 14 run receipt fields differ")
    if (
        root["format"] != "skillev-protocol14-run-receipt@1"
        or root["benchmark"] != execution.benchmark.value
        or root["condition_id"] != execution.condition_id
    ):
        raise ValueError("Protocol 14 run receipt identity differs")
    outcomes = _mapping(root["outcomes"], "outcomes")
    if set(outcomes) != {
        "scored_valid",
        "model_output_invalid",
        "generation_infrastructure",
        "environment_infrastructure",
        "scorer_infrastructure",
        "binary_success",
        "binary_failure",
        "definitive_detail",
    }:
        raise ValueError("Protocol 14 outcome fields differ")
    detail_raw = _mapping(outcomes["definitive_detail"], "outcome detail")
    detail = {key: _integer(value, key) for key, value in detail_raw.items()}
    metrics_raw = root["metrics"]
    if not isinstance(metrics_raw, list):
        raise ValueError("Protocol 14 receipt metrics must be a list")
    metrics: list[MetricObservationV4] = []
    for value in metrics_raw:
        row = _mapping(value, "metric")
        if set(row) != {
            "metric_id",
            "unit",
            "numerator",
            "denominator",
            "denominator_id",
            "projection",
            "observed_percent",
            "formula",
            "aggregation",
        }:
            raise ValueError("Protocol 14 receipt metric fields differ")
        metrics.append(
            MetricObservationV4(
                metric_id=_text(row["metric_id"], "metric ID"),
                unit=_text(row["unit"], "unit"),
                numerator=Decimal(_text(row["numerator"], "numerator")),
                denominator=_integer(row["denominator"], "denominator"),
                denominator_id=_text(row["denominator_id"], "denominator ID"),
                projection=MetricProjection(_text(row["projection"], "projection")),
                observed_percent=Decimal(_text(row["observed_percent"], "observed percent")),
                formula=_text(row["formula"], "formula"),
                aggregation=_text(row["aggregation"], "aggregation"),
            )
        )
    diagnostics_raw = _mapping(root["diagnostics"], "diagnostics")
    diagnostics: dict[str, Decimal | int | str] = {}
    for key, value in diagnostics_raw.items():
        if type(value) is int or type(value) is str:
            diagnostics[key] = value
        else:
            raise ValueError("Protocol 14 diagnostics must contain strings or integers")
    return Protocol14RunReceipt(
        benchmark=execution.benchmark,
        role=AttemptRole(_text(root["role"], "role")),
        execution=execution,
        attempt_id=_text(root["attempt_id"], "attempt ID"),
        planned_count=_integer(root["planned_count"], "planned count"),
        outcomes=OutcomeCountsV5(
            scored_valid=_integer(outcomes["scored_valid"], "scored valid"),
            model_output_invalid=_integer(outcomes["model_output_invalid"], "model output invalid"),
            generation_infrastructure=_integer(
                outcomes["generation_infrastructure"], "generation infrastructure"
            ),
            environment_infrastructure=_integer(
                outcomes["environment_infrastructure"], "environment infrastructure"
            ),
            scorer_infrastructure=_integer(
                outcomes["scorer_infrastructure"], "scorer infrastructure"
            ),
            binary_success=_optional_integer(outcomes["binary_success"], "binary success"),
            binary_failure=_optional_integer(outcomes["binary_failure"], "binary failure"),
            definitive_detail=detail,
        ),
        metrics=tuple(metrics),
        generation_code_revision=_text(root["generation_code_revision"], "generation revision"),
        scoring_code_revision=_text(root["scoring_code_revision"], "scoring revision"),
        evaluator_version=_text(root["evaluator_version"], "evaluator version"),
        started_at=_text(root["started_at"], "start time"),
        completed_at=_text(root["completed_at"], "completion time"),
        runtime_contract_matched=_boolean(root["runtime_contract_matched"], "runtime contract"),
        final_panel_used_for_selection=_boolean(
            root["final_panel_used_for_selection"], "panel selection"
        ),
        diagnostics=diagnostics,
    )


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


__all__ = [
    "AttemptRole",
    "MetricObservationV4",
    "OutcomeCountsV5",
    "Protocol14RunReceipt",
    "load_run_receipt",
    "write_run_receipt",
]
