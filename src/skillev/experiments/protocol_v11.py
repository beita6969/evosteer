"""Typed Protocol 11 identity for the owner-defined current ten-IID suite.

Protocol 10 remains an immutable historical artifact.  Protocol 11 carries the
new catalog and starts closed until all populations and evaluators have been
materialized and preflighted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import yaml

PROTOCOL_V11_FORMAT: Final = "skillev-benchmark-protocol@11"
PROTOCOL_V11_SEED: Final = 0


class ProtocolV11Error(ValueError):
    """A Protocol 11 identity or execution invariant is invalid."""


class BenchmarkV11(StrEnum):
    HOTPOT_QA = "hotpotqa"
    TRIVIA_QA = "triviaqa"
    AIME_2026 = "aime-2026"
    HEALTHBENCH = "healthbench"
    WEBSHOP = "webshop"
    ALFWORLD = "alfworld"
    SPREADSHEETBENCH = "spreadsheetbench"
    APPWORLD = "appworld"
    MBPP_PLUS_HARD = "mbpp-plus-hard"
    HUMAN_EVAL = "humaneval"


ACTIVE_BENCHMARKS_V11: Final = tuple(BenchmarkV11)


class PopulationRoleV11(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation-iid"
    FINAL_EVALUATION = "final-evaluation"


@dataclass(frozen=True, slots=True)
class ProtocolV11TrainingShape:
    per_domain_episodes: int = 512
    global_shuffle_seed: int = 0
    batch_size: int = 16
    optimizer_steps: int = 320

    @property
    def total_episodes(self) -> int:
        return len(ACTIVE_BENCHMARKS_V11) * self.per_domain_episodes

    def validate(self) -> None:
        if self.per_domain_episodes != 512 or self.global_shuffle_seed != 0:
            raise ProtocolV11Error("Protocol 11 requires 512 episodes/domain and seed 0")
        if self.total_episodes != 5_120:
            raise ProtocolV11Error("Protocol 11 must contain exactly 5,120 episodes")
        if self.total_episodes != self.batch_size * self.optimizer_steps:
            raise ProtocolV11Error("Protocol 11 training shape does not close")


@dataclass(frozen=True, slots=True)
class ProtocolV11Population:
    benchmark: BenchmarkV11
    population_id: str
    source_revision: str
    selection_rule: str
    role: PopulationRoleV11
    expected_count: int | None
    visibility: str
    environment_kind: str
    evaluator_kind: str

    def __post_init__(self) -> None:
        for value in (
            self.population_id,
            self.source_revision,
            self.selection_rule,
            self.visibility,
            self.environment_kind,
            self.evaluator_kind,
        ):
            if not value.strip():
                raise ProtocolV11Error("population identity fields must be non-empty")
        if self.expected_count is not None and self.expected_count <= 0:
            raise ProtocolV11Error("population expected_count must be positive or null")
        if self.visibility not in {"public", "private"}:
            raise ProtocolV11Error("population visibility must be public or private")


@dataclass(frozen=True, slots=True)
class ProtocolV11Spec:
    benchmarks: tuple[BenchmarkV11, ...]
    training_shape: ProtocolV11TrainingShape
    populations: tuple[ProtocolV11Population, ...]
    executable: bool
    unblock_when: tuple[str, ...]
    format: str = PROTOCOL_V11_FORMAT

    def __post_init__(self) -> None:
        if self.format != PROTOCOL_V11_FORMAT:
            raise ProtocolV11Error("Protocol 11 format is required")
        if self.benchmarks != ACTIVE_BENCHMARKS_V11:
            raise ProtocolV11Error("Protocol 11 benchmark order differs from the current IID list")
        self.training_shape.validate()
        if self.executable:
            raise ProtocolV11Error("Protocol 11 formal execution gate must remain closed")
        if not self.unblock_when:
            raise ProtocolV11Error("Protocol 11 must enumerate its unblock conditions")
        ids = tuple(item.population_id for item in self.populations)
        if len(ids) != len(set(ids)):
            raise ProtocolV11Error("Protocol 11 population IDs must be unique")
        for benchmark in self.benchmarks:
            roles = {item.role for item in self.populations if item.benchmark is benchmark}
            if roles != set(PopulationRoleV11):
                raise ProtocolV11Error(
                    f"{benchmark.value} requires training/validation/final roles"
                )

    def require_execution_ready(self) -> None:
        raise ProtocolV11Error("Protocol 11 execution gate is not open")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ProtocolV11Error(f"{label} must be a mapping")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolV11Error(f"{label} must be non-empty text")
    return value


def _items(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ProtocolV11Error(f"{label} must be a list")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ProtocolV11Error(f"{label} must be an integer")
    return value


def load_protocol_v11(protocol_path: Path, source_path: Path) -> ProtocolV11Spec:
    """Load the public V11 identity and population source registry."""

    protocol = _mapping(yaml.safe_load(protocol_path.read_text(encoding="utf-8")), "protocol")
    sources = _mapping(yaml.safe_load(source_path.read_text(encoding="utf-8")), "sources")
    if protocol.get("format") != PROTOCOL_V11_FORMAT:
        raise ProtocolV11Error("invalid Protocol 11 format")
    benchmarks = tuple(
        BenchmarkV11(_text(item, "benchmark"))
        for item in _items(protocol.get("benchmarks"), "benchmarks")
    )
    training = _mapping(protocol.get("training"), "training")
    gate = _mapping(protocol.get("execution_gate"), "execution gate")
    rows = _items(sources.get("populations"), "source populations")
    populations: list[ProtocolV11Population] = []
    for raw in rows:
        row = _mapping(raw, "population")
        count = row.get("expected_count")
        if count is not None and type(count) is not int:
            raise ProtocolV11Error("population expected_count must be an integer or null")
        populations.append(
            ProtocolV11Population(
                benchmark=BenchmarkV11(_text(row.get("benchmark"), "benchmark")),
                population_id=_text(row.get("population_id"), "population id"),
                source_revision=_text(row.get("source_revision"), "source revision"),
                selection_rule=_text(row.get("selection_rule"), "selection rule"),
                role=PopulationRoleV11(_text(row.get("role"), "population role")),
                expected_count=count,
                visibility=_text(row.get("visibility"), "visibility"),
                environment_kind=_text(row.get("environment_kind"), "environment kind"),
                evaluator_kind=_text(row.get("evaluator_kind"), "evaluator kind"),
            )
        )
    executable = gate.get("executable")
    if type(executable) is not bool:
        raise ProtocolV11Error("execution gate must be boolean")
    return ProtocolV11Spec(
        benchmarks=benchmarks,
        training_shape=ProtocolV11TrainingShape(
            per_domain_episodes=_integer(training.get("per_domain_episodes"), "episode count"),
            global_shuffle_seed=_integer(training.get("global_shuffle_seed"), "shuffle seed"),
            batch_size=_integer(training.get("batch_size"), "batch size"),
            optimizer_steps=_integer(training.get("optimizer_steps"), "optimizer steps"),
        ),
        populations=tuple(populations),
        executable=executable,
        unblock_when=tuple(
            _text(item, "unblock condition")
            for item in _items(gate.get("unblock_when"), "unblock conditions")
        ),
    )


__all__ = [
    "ACTIVE_BENCHMARKS_V11",
    "PROTOCOL_V11_FORMAT",
    "BenchmarkV11",
    "PopulationRoleV11",
    "ProtocolV11Error",
    "ProtocolV11Population",
    "ProtocolV11Spec",
    "ProtocolV11TrainingShape",
    "load_protocol_v11",
]
