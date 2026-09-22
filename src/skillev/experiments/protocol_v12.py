"""Protocol 12: the owner-authoritative nine-domain current-IID identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import yaml

from skillev.evaluation.current_iid.catalog import (
    ACTIVE_CURRENT_IID_BENCHMARKS,
    CURRENT_IID_BATCH_SIZE,
    CURRENT_IID_EPISODES_PER_DOMAIN,
    CURRENT_IID_OPTIMIZER_STEPS,
    CURRENT_IID_TRAINING_RECORD_COUNT,
    CurrentIIDBenchmark,
)

PROTOCOL_V12_FORMAT: Final = "skillev-benchmark-protocol@12"


class ProtocolV12Error(ValueError):
    pass


class PopulationRoleV12(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation-iid"
    FINAL_EVALUATION = "final-evaluation"


@dataclass(frozen=True, slots=True)
class ProtocolV12TrainingShape:
    per_domain_episodes: int
    global_shuffle_seed: int
    batch_size: int
    optimizer_steps: int

    def validate(self) -> None:
        if self.per_domain_episodes != CURRENT_IID_EPISODES_PER_DOMAIN:
            raise ProtocolV12Error("Protocol 12 requires 512 episodes per domain")
        if self.global_shuffle_seed != 0:
            raise ProtocolV12Error("Protocol 12 uses the single owner-approved seed 0")
        if self.batch_size != CURRENT_IID_BATCH_SIZE:
            raise ProtocolV12Error("Protocol 12 batch size must be 16")
        if self.optimizer_steps != CURRENT_IID_OPTIMIZER_STEPS:
            raise ProtocolV12Error("Protocol 12 requires 288 optimizer steps")
        if self.per_domain_episodes * len(ACTIVE_CURRENT_IID_BENCHMARKS) != (
            CURRENT_IID_TRAINING_RECORD_COUNT
        ):
            raise ProtocolV12Error("Protocol 12 training population does not close")


@dataclass(frozen=True, slots=True)
class ProtocolV12Population:
    benchmark: CurrentIIDBenchmark
    population_id: str
    dataset_revision: str
    selection_rule: str
    role: PopulationRoleV12
    expected_count: int | None
    visibility: str
    environment_kind: str
    evaluator_kind: str

    def __post_init__(self) -> None:
        values = (
            self.population_id,
            self.dataset_revision,
            self.selection_rule,
            self.visibility,
            self.environment_kind,
            self.evaluator_kind,
        )
        if any(not value.strip() for value in values):
            raise ProtocolV12Error("population identity fields must be non-empty")
        if self.benchmark not in ACTIVE_CURRENT_IID_BENCHMARKS:
            raise ProtocolV12Error("inactive benchmark cannot enter Protocol 12")
        if self.visibility not in {"public", "private"}:
            raise ProtocolV12Error("population visibility must be public or private")
        if self.expected_count is not None and self.expected_count <= 0:
            raise ProtocolV12Error("population count must be positive or null")


@dataclass(frozen=True, slots=True)
class ProtocolV12Spec:
    benchmarks: tuple[CurrentIIDBenchmark, ...]
    training_shape: ProtocolV12TrainingShape
    populations: tuple[ProtocolV12Population, ...]
    executable: bool
    unblock_when: tuple[str, ...]
    format: str = PROTOCOL_V12_FORMAT

    def __post_init__(self) -> None:
        if self.format != PROTOCOL_V12_FORMAT:
            raise ProtocolV12Error("invalid Protocol 12 format")
        if self.benchmarks != ACTIVE_CURRENT_IID_BENCHMARKS:
            raise ProtocolV12Error("Protocol 12 must contain exactly the authoritative nine")
        self.training_shape.validate()
        if self.executable:
            raise ProtocolV12Error("formal training remains closed until the gate passes")
        if not self.unblock_when:
            raise ProtocolV12Error("Protocol 12 must state its unblock conditions")
        identities = tuple(item.population_id for item in self.populations)
        if len(identities) != len(set(identities)):
            raise ProtocolV12Error("population IDs must be unique")
        for benchmark in self.benchmarks:
            roles = {item.role for item in self.populations if item.benchmark is benchmark}
            if roles != set(PopulationRoleV12):
                raise ProtocolV12Error(f"{benchmark.value} lacks a complete population triple")

    def require_execution_ready(self) -> None:
        raise ProtocolV12Error("Protocol 12 formal training gate is closed")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ProtocolV12Error(f"{label} must be a mapping")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ProtocolV12Error(f"{label} must be a list")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ProtocolV12Error(f"{label} must be non-empty text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ProtocolV12Error(f"{label} must be an integer")
    return value


def load_protocol_v12(protocol_path: Path, source_path: Path) -> ProtocolV12Spec:
    protocol = _mapping(yaml.safe_load(protocol_path.read_text(encoding="utf-8")), "protocol")
    sources = _mapping(yaml.safe_load(source_path.read_text(encoding="utf-8")), "sources")
    if protocol.get("format") != PROTOCOL_V12_FORMAT:
        raise ProtocolV12Error("invalid Protocol 12 format")
    benchmarks = tuple(
        CurrentIIDBenchmark(_text(item, "benchmark"))
        for item in _list(protocol.get("benchmarks"), "benchmarks")
    )
    training = _mapping(protocol.get("training"), "training")
    gate = _mapping(protocol.get("execution_gate"), "execution gate")
    populations: list[ProtocolV12Population] = []
    for value in _list(sources.get("populations"), "source populations"):
        row = _mapping(value, "population")
        count = row.get("expected_count")
        if count is not None and type(count) is not int:
            raise ProtocolV12Error("population count must be an integer or null")
        populations.append(
            ProtocolV12Population(
                benchmark=CurrentIIDBenchmark(_text(row.get("benchmark"), "benchmark")),
                population_id=_text(row.get("population_id"), "population ID"),
                dataset_revision=_text(row.get("dataset_revision"), "dataset revision"),
                selection_rule=_text(row.get("selection_rule"), "selection rule"),
                role=PopulationRoleV12(_text(row.get("role"), "population role")),
                expected_count=count,
                visibility=_text(row.get("visibility"), "visibility"),
                environment_kind=_text(row.get("environment_kind"), "environment kind"),
                evaluator_kind=_text(row.get("evaluator_kind"), "evaluator kind"),
            )
        )
    executable = gate.get("executable")
    if type(executable) is not bool:
        raise ProtocolV12Error("execution gate must be boolean")
    return ProtocolV12Spec(
        benchmarks=benchmarks,
        training_shape=ProtocolV12TrainingShape(
            per_domain_episodes=_integer(training.get("per_domain_episodes"), "episodes"),
            global_shuffle_seed=_integer(training.get("global_shuffle_seed"), "seed"),
            batch_size=_integer(training.get("batch_size"), "batch size"),
            optimizer_steps=_integer(training.get("optimizer_steps"), "optimizer steps"),
        ),
        populations=tuple(populations),
        executable=executable,
        unblock_when=tuple(
            _text(item, "unblock condition")
            for item in _list(gate.get("unblock_when"), "unblock conditions")
        ),
    )


__all__ = [
    "PROTOCOL_V12_FORMAT",
    "PopulationRoleV12",
    "ProtocolV12Error",
    "ProtocolV12Population",
    "ProtocolV12Spec",
    "ProtocolV12TrainingShape",
    "load_protocol_v12",
]
