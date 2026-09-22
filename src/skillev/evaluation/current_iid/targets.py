"""Condition-scoped parity targets; undefined targets never manufacture gaps."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import yaml

from skillev.evaluation.current_iid.catalog import (
    ACTIVE_CURRENT_IID_BENCHMARKS,
    CurrentIIDBenchmark,
)
from skillev.evaluation.current_iid.conditions import ExecutionContract, ReferenceEligibility
from skillev.experiments.protocol_v11 import BenchmarkV11


class TargetStatus(StrEnum):
    EXACT = "exact"
    RECONSTRUCTED = "reconstructed"
    DIAGNOSTIC = "diagnostic"
    UNDEFINED = "undefined"


@dataclass(frozen=True, slots=True)
class MetricTarget:
    metric_id: str
    reference_percent: float

    def __post_init__(self) -> None:
        if not self.metric_id.strip() or not 0.0 <= self.reference_percent <= 100.0:
            raise ValueError("metric target must have an ID and lie in [0, 100]")


class _RunIdentity(Protocol):
    @property
    def condition_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class BenchmarkTarget:
    benchmark: BenchmarkV11
    condition_id: str | None
    status: TargetStatus
    metrics: tuple[MetricTarget, ...] = ()

    def __post_init__(self) -> None:
        if self.status is TargetStatus.UNDEFINED:
            if self.condition_id is not None or self.metrics:
                raise ValueError("undefined target cannot carry a condition or metrics")
            return
        if not self.condition_id or not self.metrics:
            raise ValueError("defined target requires a condition and metric targets")
        ids = tuple(item.metric_id for item in self.metrics)
        if len(ids) != len(set(ids)):
            raise ValueError("target metric IDs must be unique")

    def permits_parity(self, run: _RunIdentity) -> bool:
        return self.status in {TargetStatus.EXACT, TargetStatus.RECONSTRUCTED} and (
            self.condition_id == run.condition_id
        )


def load_target_registry(path: Path) -> dict[BenchmarkV11, BenchmarkTarget]:
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or root.get("format") != "skillev-current-iid-targets@1":
        raise ValueError("invalid current-IID target registry")
    rows = root.get("targets")
    if not isinstance(rows, dict):
        raise ValueError("target registry requires targets")
    output: dict[BenchmarkV11, BenchmarkTarget] = {}
    for benchmark in BenchmarkV11:
        raw = rows.get(benchmark.value)
        if not isinstance(raw, dict):
            raise ValueError(f"missing target for {benchmark.value}")
        status = TargetStatus(str(raw["status"]))
        metric_rows = raw.get("metrics", [])
        if not isinstance(metric_rows, list):
            raise ValueError("target metrics must be a list")
        output[benchmark] = BenchmarkTarget(
            benchmark=benchmark,
            condition_id=(None if raw.get("condition_id") is None else str(raw["condition_id"])),
            status=status,
            metrics=tuple(
                MetricTarget(
                    metric_id=str(item["metric_id"]),
                    reference_percent=float(item["reference_percent"]),
                )
                for item in metric_rows
                if isinstance(item, dict)
            ),
        )
    if len(rows) != len(output):
        raise ValueError("target registry contains non-current benchmarks")
    return output


class MetricUnit(StrEnum):
    PERCENT = "percent"


class MetricAggregation(StrEnum):
    MEAN_OVER_RECORDS = "mean-over-records"
    SUCCESS_RATE = "success-rate"
    SPLIT_SUCCESS_RATE = "split-success-rate"
    OFFICIAL_AGGREGATE = "official-aggregate"


@dataclass(frozen=True, slots=True)
class ReferenceMetric:
    metric_id: str
    unit: MetricUnit
    formula: str
    denominator: str
    aggregation: MetricAggregation
    reference_percent: Decimal
    maximum_gap_pp_exclusive: Decimal = Decimal("7.0")
    required_for_formal_gate: bool = True

    def __post_init__(self) -> None:
        if not self.metric_id.strip() or not self.formula.strip() or not self.denominator.strip():
            raise ValueError("reference metric identity is incomplete")
        if not Decimal("0") <= self.reference_percent <= Decimal("100"):
            raise ValueError("reference metric lies outside [0, 100]")
        if self.maximum_gap_pp_exclusive != Decimal("7.0"):
            raise ValueError("current-IID parity requires a strict 7pp threshold")


@dataclass(frozen=True, slots=True)
class ReferenceTargetContract:
    benchmark: CurrentIIDBenchmark
    status: TargetStatus
    execution: ExecutionContract | None
    metrics: tuple[ReferenceMetric, ...]
    evidence_source: str | None

    def __post_init__(self) -> None:
        if self.benchmark not in ACTIVE_CURRENT_IID_BENCHMARKS:
            raise ValueError("inactive benchmark cannot enter target registry")
        if self.status is TargetStatus.UNDEFINED:
            if self.execution is not None or self.metrics or self.evidence_source is not None:
                raise ValueError("undefined target cannot carry formal evidence")
            return
        if self.execution is None or not self.metrics or not self.evidence_source:
            raise ValueError("defined target requires execution, metrics and evidence")
        if self.execution.benchmark is not self.benchmark:
            raise ValueError("target execution belongs to another benchmark")
        if self.status is TargetStatus.DIAGNOSTIC and (
            self.execution.reference_eligibility is not ReferenceEligibility.DIAGNOSTIC_ONLY
        ):
            raise ValueError("diagnostic target requires diagnostic execution")
        ids = tuple(item.metric_id for item in self.metrics)
        if len(ids) != len(set(ids)):
            raise ValueError("target metric IDs must be unique")

    @property
    def can_enter_formal_gate(self) -> bool:
        return self.status in {TargetStatus.EXACT, TargetStatus.RECONSTRUCTED}


def _exact_mapping(value: object, *, fields: frozenset[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} fields differ: expected={sorted(fields)}, got={got}")
    return value


def load_target_registry_v2(
    path: Path,
    *,
    executions: dict[CurrentIIDBenchmark, ExecutionContract],
) -> dict[CurrentIIDBenchmark, ReferenceTargetContract]:
    root = _exact_mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        fields=frozenset({"format", "targets"}),
        label="target registry",
    )
    if root["format"] != "skillev-current-iid-targets@2":
        raise ValueError("invalid target-v2 registry")
    rows = root["targets"]
    if not isinstance(rows, dict) or set(rows) != {
        item.value for item in ACTIVE_CURRENT_IID_BENCHMARKS
    }:
        raise ValueError("target-v2 registry must contain exactly the authoritative nine")
    output: dict[CurrentIIDBenchmark, ReferenceTargetContract] = {}
    for benchmark in ACTIVE_CURRENT_IID_BENCHMARKS:
        raw_value = rows[benchmark.value]
        if not isinstance(raw_value, dict):
            raise ValueError("target row must be a mapping")
        status = TargetStatus(str(raw_value.get("status")))
        if status is TargetStatus.UNDEFINED:
            _exact_mapping(raw_value, fields=frozenset({"status"}), label=benchmark.value)
            output[benchmark] = ReferenceTargetContract(benchmark, status, None, (), None)
            continue
        raw = _exact_mapping(
            raw_value,
            fields=frozenset({"status", "condition_id", "evidence_source", "metrics"}),
            label=benchmark.value,
        )
        execution = executions[benchmark]
        if raw["condition_id"] != execution.condition_id:
            raise ValueError("target condition differs from Protocol 12 execution")
        metric_rows = raw["metrics"]
        if not isinstance(metric_rows, list):
            raise ValueError("target metrics must be a list")
        metrics: list[ReferenceMetric] = []
        for index, value in enumerate(metric_rows):
            metric = _exact_mapping(
                value,
                fields=frozenset(
                    {
                        "metric_id",
                        "unit",
                        "formula",
                        "denominator",
                        "aggregation",
                        "reference_percent",
                        "maximum_gap_pp_exclusive",
                        "required_for_formal_gate",
                    }
                ),
                label=f"{benchmark.value}.metrics[{index}]",
            )
            metrics.append(
                ReferenceMetric(
                    metric_id=str(metric["metric_id"]),
                    unit=MetricUnit(str(metric["unit"])),
                    formula=str(metric["formula"]),
                    denominator=str(metric["denominator"]),
                    aggregation=MetricAggregation(str(metric["aggregation"])),
                    reference_percent=Decimal(str(metric["reference_percent"])),
                    maximum_gap_pp_exclusive=Decimal(str(metric["maximum_gap_pp_exclusive"])),
                    required_for_formal_gate=bool(metric["required_for_formal_gate"]),
                )
            )
        output[benchmark] = ReferenceTargetContract(
            benchmark=benchmark,
            status=status,
            execution=execution,
            metrics=tuple(metrics),
            evidence_source=str(raw["evidence_source"]),
        )
    return output


__all__ = [
    "BenchmarkTarget",
    "MetricAggregation",
    "MetricTarget",
    "MetricUnit",
    "ReferenceMetric",
    "ReferenceTargetContract",
    "TargetStatus",
    "load_target_registry",
    "load_target_registry_v2",
]
