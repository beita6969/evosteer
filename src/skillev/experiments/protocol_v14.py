"""Corrected exact-eight IID Protocol 14 scientific identity."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Final

import yaml

from skillev.evaluation.current_iid.protocol14.catalog import (
    ACTIVE_PROTOCOL14_BENCHMARKS,
    FINAL_RECORD_COUNT,
    Protocol14Benchmark,
    expected_final_count,
)

PROTOCOL_V14_FORMAT: Final = "skillev-benchmark-protocol@14"
PROTOCOL_V14_SOURCES_FORMAT: Final = "skillev-benchmark-sources@14"


class ProtocolV14Error(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProtocolV14Population:
    benchmark: Protocol14Benchmark
    population_id: str
    dataset_revision: str
    selection_rule: str
    expected_count: int
    panel_manifest_id: str
    visibility: str
    environment_kind: str
    evaluator_kind: str

    def __post_init__(self) -> None:
        required = (
            self.population_id,
            self.dataset_revision,
            self.selection_rule,
            self.panel_manifest_id,
            self.visibility,
            self.environment_kind,
            self.evaluator_kind,
        )
        if any(not value.strip() for value in required):
            raise ProtocolV14Error("population identity is incomplete")
        if self.expected_count != expected_final_count(self.benchmark):
            raise ProtocolV14Error("population count differs from exact-eight catalog")
        if self.visibility not in {"public", "private"}:
            raise ProtocolV14Error("population visibility is invalid")


@dataclass(frozen=True, slots=True)
class ProtocolV14Spec:
    benchmarks: tuple[Protocol14Benchmark, ...]
    gate_mode: str
    threshold_exclusive: Decimal
    planned_count: int
    adapter_policy: str
    populations: tuple[ProtocolV14Population, ...]
    executable: bool
    unblock_when: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.benchmarks != ACTIVE_PROTOCOL14_BENCHMARKS:
            raise ProtocolV14Error("Protocol 14 must contain the authoritative eight in order")
        if self.gate_mode != "absolute_percentage_points":
            raise ProtocolV14Error("Protocol 14 gate unit is not absolute percentage points")
        if self.threshold_exclusive != Decimal("7.0"):
            raise ProtocolV14Error("Protocol 14 requires a strict 7pp threshold")
        if self.planned_count != FINAL_RECORD_COUNT:
            raise ProtocolV14Error("Protocol 14 planned count must be 926")
        if self.adapter_policy != "forbidden":
            raise ProtocolV14Error("Protocol 14 backbone baseline forbids adapters")
        if not self.unblock_when:
            raise ProtocolV14Error("Protocol 14 requires explicit unblock conditions")
        if tuple(item.benchmark for item in self.populations) != self.benchmarks:
            raise ProtocolV14Error(
                "Protocol 14 requires one ordered final population per benchmark"
            )
        ids = tuple(item.population_id for item in self.populations)
        if len(ids) != len(set(ids)):
            raise ProtocolV14Error("Protocol 14 population IDs must be unique")

    def require_execution_ready(self) -> None:
        if not self.executable:
            raise ProtocolV14Error("Protocol 14 final execution gate is closed")


def load_protocol_v14(protocol_path: Path, sources_path: Path) -> ProtocolV14Spec:
    protocol = _mapping(yaml.safe_load(protocol_path.read_text(encoding="utf-8")), "protocol")
    sources = _mapping(yaml.safe_load(sources_path.read_text(encoding="utf-8")), "sources")
    if set(protocol) != {
        "format",
        "benchmarks",
        "parity_gate",
        "final_evaluation",
        "execution_gate",
    }:
        raise ProtocolV14Error("Protocol 14 top-level fields differ")
    if protocol["format"] != PROTOCOL_V14_FORMAT:
        raise ProtocolV14Error("invalid Protocol 14 format")
    if set(sources) != {"format", "final_populations"}:
        raise ProtocolV14Error("Protocol 14 sources fields differ")
    if sources["format"] != PROTOCOL_V14_SOURCES_FORMAT:
        raise ProtocolV14Error("invalid Protocol 14 sources format")
    benchmarks = tuple(
        Protocol14Benchmark(_text(item, "benchmark"))
        for item in _list(protocol["benchmarks"], "benchmarks")
    )
    gate = _mapping(protocol["parity_gate"], "parity gate")
    if set(gate) != {"mode", "threshold_exclusive"}:
        raise ProtocolV14Error("parity gate fields differ")
    final = _mapping(protocol["final_evaluation"], "final evaluation")
    if set(final) != {"planned_count", "adapter_policy"}:
        raise ProtocolV14Error("final evaluation fields differ")
    execution_gate = _mapping(protocol["execution_gate"], "execution gate")
    if set(execution_gate) != {"executable", "unblock_when"}:
        raise ProtocolV14Error("execution gate fields differ")
    executable = execution_gate["executable"]
    if type(executable) is not bool:
        raise ProtocolV14Error("execution gate flag must be boolean")
    populations: list[ProtocolV14Population] = []
    expected_fields = {
        "benchmark",
        "population_id",
        "dataset_revision",
        "selection_rule",
        "expected_count",
        "panel_manifest_id",
        "visibility",
        "environment_kind",
        "evaluator_kind",
    }
    for value in _list(sources["final_populations"], "final populations"):
        row = _mapping(value, "population")
        if set(row) != expected_fields:
            raise ProtocolV14Error("Protocol 14 population fields differ")
        populations.append(
            ProtocolV14Population(
                benchmark=Protocol14Benchmark(_text(row["benchmark"], "benchmark")),
                population_id=_text(row["population_id"], "population ID"),
                dataset_revision=_text(row["dataset_revision"], "dataset revision"),
                selection_rule=_text(row["selection_rule"], "selection rule"),
                expected_count=_integer(row["expected_count"], "expected count"),
                panel_manifest_id=_text(row["panel_manifest_id"], "panel manifest ID"),
                visibility=_text(row["visibility"], "visibility"),
                environment_kind=_text(row["environment_kind"], "environment kind"),
                evaluator_kind=_text(row["evaluator_kind"], "evaluator kind"),
            )
        )
    return ProtocolV14Spec(
        benchmarks=benchmarks,
        gate_mode=_text(gate["mode"], "gate mode"),
        threshold_exclusive=Decimal(_text(gate["threshold_exclusive"], "threshold")),
        planned_count=_integer(final["planned_count"], "planned count"),
        adapter_policy=_text(final["adapter_policy"], "adapter policy"),
        populations=tuple(populations),
        executable=executable,
        unblock_when=tuple(
            _text(item, "unblock condition")
            for item in _list(execution_gate["unblock_when"], "unblock conditions")
        ),
    )


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ProtocolV14Error(f"{label} must be a mapping")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ProtocolV14Error(f"{label} must be a list")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ProtocolV14Error(f"{label} must be non-empty text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ProtocolV14Error(f"{label} must be an integer")
    return value


__all__ = [
    "PROTOCOL_V14_FORMAT",
    "PROTOCOL_V14_SOURCES_FORMAT",
    "ProtocolV14Error",
    "ProtocolV14Population",
    "ProtocolV14Spec",
    "load_protocol_v14",
]
