"""Protocol 13 exact-eight current-IID scientific identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import yaml

from skillev.evaluation.current_iid.protocol13.catalog import (
    ACTIVE_PROTOCOL13_BENCHMARKS,
    CHECKPOINT_COUNT,
    CHECKPOINT_EVERY_STEPS,
    EFFECTIVE_BATCH_SIZE,
    QUESTIONS_PER_DOMAIN,
    QUESTIONS_PER_STEP,
    TRAINING_QUESTION_COUNT,
    TRAINING_STEPS,
    TRAINING_TRAJECTORY_COUNT,
    TRAJECTORIES_PER_QUESTION,
    Protocol13Benchmark,
)

PROTOCOL_V13_FORMAT: Final = "skillev-benchmark-protocol@13"
PROTOCOL_V13_SOURCES_FORMAT: Final = "skillev-benchmark-sources@13"


class ProtocolV13Error(ValueError):
    pass


class PopulationRoleV13(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation-iid"
    FINAL_EVALUATION = "final-evaluation"


@dataclass(frozen=True, slots=True)
class ProtocolV13TrainingShape:
    questions_per_domain: int
    selection_seed: int
    questions_per_step: int
    trajectories_per_question: int
    effective_batch_size: int
    optimizer_steps: int
    checkpoint_every_steps: int
    checkpoint_count: int
    sampling_rule: str

    def validate(self) -> None:
        observed = (
            self.questions_per_domain,
            self.selection_seed,
            self.questions_per_step,
            self.trajectories_per_question,
            self.effective_batch_size,
            self.optimizer_steps,
            self.checkpoint_every_steps,
            self.checkpoint_count,
            self.sampling_rule,
        )
        expected = (
            QUESTIONS_PER_DOMAIN,
            0,
            QUESTIONS_PER_STEP,
            TRAJECTORIES_PER_QUESTION,
            EFFECTIVE_BATCH_SIZE,
            TRAINING_STEPS,
            CHECKPOINT_EVERY_STEPS,
            CHECKPOINT_COUNT,
            "dataset-balanced-step-major",
        )
        if observed != expected:
            raise ProtocolV13Error(
                f"Protocol 13 training shape differs: expected={expected}, observed={observed}"
            )
        if self.questions_per_domain * len(ACTIVE_PROTOCOL13_BENCHMARKS) != (
            TRAINING_QUESTION_COUNT
        ):
            raise ProtocolV13Error("Protocol 13 question population does not close")
        if self.effective_batch_size * self.optimizer_steps != TRAINING_TRAJECTORY_COUNT:
            raise ProtocolV13Error("Protocol 13 trajectory population does not close")


@dataclass(frozen=True, slots=True)
class ProtocolV13Population:
    benchmark: Protocol13Benchmark
    population_id: str
    dataset_revision: str
    selection_rule: str
    role: PopulationRoleV13
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
            raise ProtocolV13Error("population identity fields must be non-empty")
        if self.visibility not in {"public", "private"}:
            raise ProtocolV13Error("population visibility must be public or private")
        if self.expected_count is not None and (
            type(self.expected_count) is not int or self.expected_count <= 0
        ):
            raise ProtocolV13Error("population count must be positive or null")


@dataclass(frozen=True, slots=True)
class ProtocolV13Spec:
    benchmarks: tuple[Protocol13Benchmark, ...]
    training_shape: ProtocolV13TrainingShape
    populations: tuple[ProtocolV13Population, ...]
    executable: bool
    unblock_when: tuple[str, ...]
    format: str = PROTOCOL_V13_FORMAT

    def __post_init__(self) -> None:
        if self.format != PROTOCOL_V13_FORMAT:
            raise ProtocolV13Error("invalid Protocol 13 format")
        if self.benchmarks != ACTIVE_PROTOCOL13_BENCHMARKS:
            raise ProtocolV13Error("Protocol 13 must contain the authoritative eight in order")
        self.training_shape.validate()
        if not self.unblock_when:
            raise ProtocolV13Error("Protocol 13 requires explicit unblock conditions")
        population_ids = tuple(item.population_id for item in self.populations)
        if len(population_ids) != len(set(population_ids)):
            raise ProtocolV13Error("population IDs must be unique")
        for benchmark in self.benchmarks:
            benchmark_populations = tuple(
                item for item in self.populations if item.benchmark is benchmark
            )
            roles = {item.role for item in benchmark_populations}
            if roles != set(PopulationRoleV13):
                raise ProtocolV13Error(f"{benchmark.value} lacks a complete population triple")
            if len(benchmark_populations) != len(PopulationRoleV13):
                raise ProtocolV13Error(f"{benchmark.value} population roles are not unique")
            final = next(
                item
                for item in benchmark_populations
                if item.role is PopulationRoleV13.FINAL_EVALUATION
            )
            expected = 30 if benchmark is Protocol13Benchmark.AIME_2026 else 128
            if final.expected_count != expected:
                raise ProtocolV13Error(
                    f"{benchmark.value} final population must contain {expected} records"
                )

    def require_execution_ready(self) -> None:
        if not self.executable:
            raise ProtocolV13Error("Protocol 13 formal execution gate is closed")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ProtocolV13Error(f"{label} must be a mapping")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ProtocolV13Error(f"{label} must be a list")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ProtocolV13Error(f"{label} must be non-empty text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ProtocolV13Error(f"{label} must be an integer")
    return value


def load_protocol_v13(protocol_path: Path, source_path: Path) -> ProtocolV13Spec:
    protocol = _mapping(yaml.safe_load(protocol_path.read_text(encoding="utf-8")), "protocol")
    sources = _mapping(yaml.safe_load(source_path.read_text(encoding="utf-8")), "sources")
    if set(protocol) != {"format", "benchmarks", "training", "execution_gate"}:
        raise ProtocolV13Error("Protocol 13 top-level fields differ")
    if protocol["format"] != PROTOCOL_V13_FORMAT:
        raise ProtocolV13Error("invalid Protocol 13 format")
    if set(sources) != {"format", "populations"}:
        raise ProtocolV13Error("Protocol 13 source fields differ")
    if sources["format"] != PROTOCOL_V13_SOURCES_FORMAT:
        raise ProtocolV13Error("invalid Protocol 13 source format")
    benchmarks = tuple(
        Protocol13Benchmark(_text(item, "benchmark"))
        for item in _list(protocol["benchmarks"], "benchmarks")
    )
    training = _mapping(protocol["training"], "training")
    if set(training) != {
        "questions_per_domain",
        "selection_seed",
        "questions_per_step",
        "trajectories_per_question",
        "effective_batch_size",
        "optimizer_steps",
        "checkpoint_every_steps",
        "checkpoint_count",
        "sampling_rule",
    }:
        raise ProtocolV13Error("Protocol 13 training fields differ")
    gate = _mapping(protocol["execution_gate"], "execution gate")
    if set(gate) != {"executable", "unblock_when"}:
        raise ProtocolV13Error("Protocol 13 execution gate fields differ")
    populations: list[ProtocolV13Population] = []
    expected_population_fields = {
        "benchmark",
        "population_id",
        "dataset_revision",
        "selection_rule",
        "role",
        "expected_count",
        "visibility",
        "environment_kind",
        "evaluator_kind",
    }
    for value in _list(sources["populations"], "source populations"):
        row = _mapping(value, "population")
        if set(row) != expected_population_fields:
            raise ProtocolV13Error("Protocol 13 population fields differ")
        count = row["expected_count"]
        if count is not None and type(count) is not int:
            raise ProtocolV13Error("population count must be an integer or null")
        populations.append(
            ProtocolV13Population(
                benchmark=Protocol13Benchmark(_text(row["benchmark"], "benchmark")),
                population_id=_text(row["population_id"], "population ID"),
                dataset_revision=_text(row["dataset_revision"], "dataset revision"),
                selection_rule=_text(row["selection_rule"], "selection rule"),
                role=PopulationRoleV13(_text(row["role"], "population role")),
                expected_count=count,
                visibility=_text(row["visibility"], "visibility"),
                environment_kind=_text(row["environment_kind"], "environment kind"),
                evaluator_kind=_text(row["evaluator_kind"], "evaluator kind"),
            )
        )
    executable = gate["executable"]
    if type(executable) is not bool:
        raise ProtocolV13Error("execution gate must be boolean")
    return ProtocolV13Spec(
        benchmarks=benchmarks,
        training_shape=ProtocolV13TrainingShape(
            questions_per_domain=_integer(training["questions_per_domain"], "questions per domain"),
            selection_seed=_integer(training["selection_seed"], "selection seed"),
            questions_per_step=_integer(training["questions_per_step"], "questions per step"),
            trajectories_per_question=_integer(
                training["trajectories_per_question"], "trajectories per question"
            ),
            effective_batch_size=_integer(training["effective_batch_size"], "effective batch size"),
            optimizer_steps=_integer(training["optimizer_steps"], "optimizer steps"),
            checkpoint_every_steps=_integer(
                training["checkpoint_every_steps"], "checkpoint cadence"
            ),
            checkpoint_count=_integer(training["checkpoint_count"], "checkpoint count"),
            sampling_rule=_text(training["sampling_rule"], "sampling rule"),
        ),
        populations=tuple(populations),
        executable=executable,
        unblock_when=tuple(
            _text(item, "unblock condition")
            for item in _list(gate["unblock_when"], "unblock conditions")
        ),
    )


__all__ = [
    "PROTOCOL_V13_FORMAT",
    "PROTOCOL_V13_SOURCES_FORMAT",
    "PopulationRoleV13",
    "ProtocolV13Error",
    "ProtocolV13Population",
    "ProtocolV13Spec",
    "ProtocolV13TrainingShape",
    "load_protocol_v13",
]
