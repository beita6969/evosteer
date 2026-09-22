"""Read-only, resumable Protocol 10 validation and final evaluation."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

from skillev.contracts import JsonValue, TerminalReward, normalize_json
from skillev.experiments import ACTIVE_BENCHMARKS_V10, BenchmarkV10, FormalMethodV10
from skillev.experiments.protocol_v10 import PopulationRole
from skillev.rollout import RolloutSessionBundle, RolloutTask
from skillev_private.benchmarks.catalog import PrivateSessionFactory
from skillev_private.benchmarks.protocol_v10_population import (
    ProtocolV10PopulationCatalog,
    ProtocolV10PopulationSessionRegistry,
)

PROTOCOL_V10_EVALUATION_INPUT_FORMAT = "skillev-protocol-v10-evaluation-input@1"
PROTOCOL_V10_EVALUATION_RECORD_FORMAT = "skillev-protocol-v10-evaluation-record@2"
PROTOCOL_V10_EVALUATION_REPORT_FORMAT = "skillev-protocol-v10-evaluation-report@2"


class ProtocolV10EvaluationMode(StrEnum):
    VALIDATION = "validation"
    FINAL = "final-evaluation"

    @property
    def population_role(self) -> PopulationRole:
        return PopulationRole(self.value)


@dataclass(frozen=True, slots=True)
class ProtocolV10EvaluationInput:
    run_id: str
    method: FormalMethodV10
    mode: ProtocolV10EvaluationMode
    checkpoint_id: str
    protocol_id: str
    population_manifest_id: str
    evaluator_bundle_id: str
    model_id: str
    tokenizer_id: str
    generation_config_id: str
    expected_record_count: int
    shard_index: int
    shard_count: int
    seed: int = 0
    format: str = PROTOCOL_V10_EVALUATION_INPUT_FORMAT

    def __post_init__(self) -> None:
        if self.format != PROTOCOL_V10_EVALUATION_INPUT_FORMAT or self.seed != 0:
            raise ValueError("Protocol 10 evaluation identity is unsupported")
        if any(
            not value.strip()
            for value in (
                self.run_id,
                self.checkpoint_id,
                self.protocol_id,
                self.population_manifest_id,
                self.evaluator_bundle_id,
                self.model_id,
                self.tokenizer_id,
                self.generation_config_id,
            )
        ):
            raise ValueError("Protocol 10 evaluation identity is incomplete")
        if not isinstance(self.method, FormalMethodV10) or not isinstance(
            self.mode, ProtocolV10EvaluationMode
        ):
            raise TypeError("Protocol 10 evaluation enums are not closed")
        if (
            type(self.expected_record_count) is not int
            or self.expected_record_count < 1
            or type(self.shard_index) is not int
            or type(self.shard_count) is not int
            or self.shard_count < 1
            or not 0 <= self.shard_index < self.shard_count
        ):
            raise ValueError("Protocol 10 evaluation shard is invalid")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "evaluator_bundle_id": self.evaluator_bundle_id,
            "expected_record_count": self.expected_record_count,
            "format": self.format,
            "generation_config_id": self.generation_config_id,
            "method": self.method.value,
            "model_id": self.model_id,
            "mode": self.mode.value,
            "population_manifest_id": self.population_manifest_id,
            "protocol_id": self.protocol_id,
            "run_id": self.run_id,
            "seed": self.seed,
            "shard_count": self.shard_count,
            "shard_index": self.shard_index,
            "tokenizer_id": self.tokenizer_id,
        }

    @classmethod
    def from_value(cls, value: object) -> ProtocolV10EvaluationInput:
        fields = {
            "checkpoint_id",
            "evaluator_bundle_id",
            "expected_record_count",
            "format",
            "generation_config_id",
            "method",
            "model_id",
            "mode",
            "population_manifest_id",
            "protocol_id",
            "run_id",
            "seed",
            "shard_count",
            "shard_index",
            "tokenizer_id",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("Protocol 10 evaluation input has incompatible fields")
        if any(
            type(value[field]) is not str
            for field in (
                "checkpoint_id",
                "evaluator_bundle_id",
                "format",
                "generation_config_id",
                "method",
                "mode",
                "model_id",
                "population_manifest_id",
                "protocol_id",
                "run_id",
                "tokenizer_id",
            )
        ) or any(
            type(value[field]) is not int
            for field in ("expected_record_count", "seed", "shard_count", "shard_index")
        ):
            raise TypeError("Protocol 10 evaluation input fields have invalid types")
        typed = cast(dict[str, str | int], value)
        return cls(
            run_id=cast(str, typed["run_id"]),
            method=FormalMethodV10(cast(str, typed["method"])),
            mode=ProtocolV10EvaluationMode(cast(str, typed["mode"])),
            checkpoint_id=cast(str, typed["checkpoint_id"]),
            protocol_id=cast(str, typed["protocol_id"]),
            population_manifest_id=cast(str, typed["population_manifest_id"]),
            evaluator_bundle_id=cast(str, typed["evaluator_bundle_id"]),
            model_id=cast(str, typed["model_id"]),
            tokenizer_id=cast(str, typed["tokenizer_id"]),
            generation_config_id=cast(str, typed["generation_config_id"]),
            expected_record_count=cast(int, typed["expected_record_count"]),
            shard_index=cast(int, typed["shard_index"]),
            shard_count=cast(int, typed["shard_count"]),
            seed=cast(int, typed["seed"]),
            format=cast(str, typed["format"]),
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10ReadOnlyState:
    policy_snapshot_id: str
    library_version: str
    optimizer_step: int
    projection_version: str
    detector_version: str

    def __post_init__(self) -> None:
        if (
            any(
                not value.strip()
                for value in (
                    self.policy_snapshot_id,
                    self.library_version,
                    self.projection_version,
                    self.detector_version,
                )
            )
            or type(self.optimizer_step) is not int
        ):
            raise ValueError("read-only evaluation state is incomplete")


@dataclass(frozen=True, slots=True)
class ProtocolV10EvaluationOutcome:
    reward: TerminalReward
    episode_steps: int
    tool_calls: int
    model_requests: int
    model_request_errors: int
    parse_errors: int
    wall_time_seconds: float
    json_syntax_errors: int = 0
    action_schema_errors: int = 0
    unavailable_skills: int = 0
    unsupported_resources: int = 0
    unsupported_tools: int = 0
    invalid_arguments: int = 0
    invalid_completions: int = 0
    valid_actions: int = 0
    environment_admitted_actions: int = 0
    explicit_completions: int = 0
    environment_terminals: int = 0
    horizon_no_submissions: int = 0
    scorer_invocations: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.reward, TerminalReward):
            raise TypeError("evaluation outcome requires TerminalReward")
        for value in (
            self.episode_steps,
            self.tool_calls,
            self.model_requests,
            self.model_request_errors,
            self.parse_errors,
            self.json_syntax_errors,
            self.action_schema_errors,
            self.unavailable_skills,
            self.unsupported_resources,
            self.unsupported_tools,
            self.invalid_arguments,
            self.invalid_completions,
            self.valid_actions,
            self.environment_admitted_actions,
            self.explicit_completions,
            self.environment_terminals,
            self.horizon_no_submissions,
            self.scorer_invocations,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("evaluation counters must be non-negative integers")
        if (
            isinstance(self.wall_time_seconds, bool)
            or not isinstance(self.wall_time_seconds, int | float)
            or not math.isfinite(float(self.wall_time_seconds))
            or self.wall_time_seconds < 0
        ):
            raise ValueError("evaluation wall time must be finite and non-negative")
        fine_failures = (
            self.json_syntax_errors
            + self.action_schema_errors
            + self.unavailable_skills
            + self.unsupported_resources
            + self.unsupported_tools
            + self.invalid_arguments
            + self.invalid_completions
        )
        if fine_failures + self.valid_actions > self.episode_steps:
            raise ValueError("evaluation action counters exceed episode steps")
        if self.scorer_invocations > 1:
            raise ValueError("one evaluation episode can invoke its scorer at most once")


class ProtocolV10EvaluationInfrastructureError(RuntimeError):
    """Answer-free failure boundary for a trusted evaluation dependency."""

    def __init__(self, benchmark: BenchmarkV10, global_position: int) -> None:
        super().__init__(
            f"Protocol 10 {benchmark.value} evaluation infrastructure failed "
            f"at planned position {global_position}"
        )
        self.benchmark = benchmark
        self.global_position = global_position


class ProtocolV10ReadOnlyEpisodeRunner(Protocol):
    async def evaluate(
        self,
        task: RolloutTask,
        session: RolloutSessionBundle,
    ) -> ProtocolV10EvaluationOutcome: ...


class ProtocolV10StateReader(Protocol):
    def __call__(self) -> ProtocolV10ReadOnlyState: ...


@dataclass(frozen=True, slots=True)
class ProtocolV10EvaluationRecord:
    global_position: int
    benchmark: BenchmarkV10
    population_id: str
    source_id: str
    reward: float
    success: bool
    native_fields: dict[str, JsonValue]
    environment_id: str
    verifier_version: str
    episode_steps: int
    tool_calls: int
    model_requests: int
    model_request_errors: int
    parse_errors: int
    wall_time_seconds: float
    json_syntax_errors: int = 0
    action_schema_errors: int = 0
    unavailable_skills: int = 0
    unsupported_resources: int = 0
    unsupported_tools: int = 0
    invalid_arguments: int = 0
    invalid_completions: int = 0
    valid_actions: int = 0
    environment_admitted_actions: int = 0
    explicit_completions: int = 0
    environment_terminals: int = 0
    horizon_no_submissions: int = 0
    scorer_invocations: int = 0
    format: str = PROTOCOL_V10_EVALUATION_RECORD_FORMAT

    def __post_init__(self) -> None:
        if self.format != PROTOCOL_V10_EVALUATION_RECORD_FORMAT:
            raise ValueError("Protocol 10 evaluation record format is unsupported")
        if type(self.global_position) is not int or self.global_position < 0:
            raise ValueError("evaluation record position is invalid")
        if any(
            not value.strip()
            for value in (
                self.population_id,
                self.source_id,
                self.environment_id,
                self.verifier_version,
            )
        ):
            raise ValueError("evaluation record identity is incomplete")
        normalized = normalize_json(self.native_fields)
        if not isinstance(normalized, dict) or not normalized:
            raise ValueError("evaluation record native fields are incomplete")
        object.__setattr__(self, "native_fields", normalized)
        if (
            isinstance(self.reward, bool)
            or not isinstance(self.reward, int | float)
            or not math.isfinite(float(self.reward))
            or not 0.0 <= float(self.reward) <= 1.0
            or type(self.success) is not bool
        ):
            raise ValueError("evaluation record reward is invalid")
        for counter in (
            self.episode_steps,
            self.tool_calls,
            self.model_requests,
            self.model_request_errors,
            self.parse_errors,
            self.json_syntax_errors,
            self.action_schema_errors,
            self.unavailable_skills,
            self.unsupported_resources,
            self.unsupported_tools,
            self.invalid_arguments,
            self.invalid_completions,
            self.valid_actions,
            self.environment_admitted_actions,
            self.explicit_completions,
            self.environment_terminals,
            self.horizon_no_submissions,
            self.scorer_invocations,
        ):
            if type(counter) is not int or counter < 0:
                raise ValueError("evaluation record counters are invalid")
        if self.scorer_invocations > 1:
            raise ValueError("one evaluation record can invoke its scorer at most once")
        if (
            isinstance(self.wall_time_seconds, bool)
            or not isinstance(self.wall_time_seconds, int | float)
            or not math.isfinite(float(self.wall_time_seconds))
            or self.wall_time_seconds < 0
        ):
            raise ValueError("evaluation record wall time is invalid")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "action_schema_errors": self.action_schema_errors,
            "environment_id": self.environment_id,
            "episode_steps": self.episode_steps,
            "format": self.format,
            "global_position": self.global_position,
            "environment_admitted_actions": self.environment_admitted_actions,
            "environment_terminals": self.environment_terminals,
            "explicit_completions": self.explicit_completions,
            "horizon_no_submissions": self.horizon_no_submissions,
            "invalid_arguments": self.invalid_arguments,
            "invalid_completions": self.invalid_completions,
            "json_syntax_errors": self.json_syntax_errors,
            "model_request_errors": self.model_request_errors,
            "model_requests": self.model_requests,
            "native_fields": self.native_fields,
            "parse_errors": self.parse_errors,
            "population_id": self.population_id,
            "reward": self.reward,
            "source_id": self.source_id,
            "scorer_invocations": self.scorer_invocations,
            "success": self.success,
            "tool_calls": self.tool_calls,
            "unavailable_skills": self.unavailable_skills,
            "unsupported_resources": self.unsupported_resources,
            "unsupported_tools": self.unsupported_tools,
            "valid_actions": self.valid_actions,
            "verifier_version": self.verifier_version,
            "wall_time_seconds": self.wall_time_seconds,
        }

    @classmethod
    def from_value(cls, value: object) -> ProtocolV10EvaluationRecord:
        legacy_fields = {
            "benchmark",
            "environment_id",
            "episode_steps",
            "format",
            "global_position",
            "model_request_errors",
            "model_requests",
            "native_fields",
            "parse_errors",
            "population_id",
            "reward",
            "source_id",
            "success",
            "tool_calls",
            "verifier_version",
            "wall_time_seconds",
        }
        telemetry_fields = {
            "action_schema_errors",
            "environment_admitted_actions",
            "environment_terminals",
            "explicit_completions",
            "horizon_no_submissions",
            "invalid_arguments",
            "invalid_completions",
            "json_syntax_errors",
            "scorer_invocations",
            "unavailable_skills",
            "unsupported_resources",
            "unsupported_tools",
            "valid_actions",
        }
        if not isinstance(value, dict) or frozenset(value) not in {
            frozenset(legacy_fields),
            frozenset(legacy_fields | telemetry_fields),
        }:
            raise ValueError("Protocol 10 evaluation record has incompatible fields")
        legacy = set(value) == legacy_fields
        native = normalize_json(value["native_fields"])
        if not isinstance(native, dict):
            raise TypeError("evaluation record native fields must be an object")
        text_fields = (
            "benchmark",
            "environment_id",
            "format",
            "population_id",
            "source_id",
            "verifier_version",
        )
        integer_fields = (
            "episode_steps",
            "global_position",
            "model_request_errors",
            "model_requests",
            "parse_errors",
            "tool_calls",
        ) + (() if legacy else tuple(sorted(telemetry_fields)))
        if any(type(value[field]) is not str for field in text_fields):
            raise TypeError("evaluation record identities must be text")
        if any(type(value[field]) is not int for field in integer_fields):
            raise TypeError("evaluation record counters must be integers")
        if type(value["success"]) is not bool:
            raise TypeError("evaluation record success must be boolean")
        if any(
            isinstance(value[field], bool) or not isinstance(value[field], int | float)
            for field in ("reward", "wall_time_seconds")
        ):
            raise TypeError("evaluation record metrics must be numeric")
        typed = cast(dict[str, object], value)
        return cls(
            global_position=cast(int, typed["global_position"]),
            benchmark=BenchmarkV10(cast(str, typed["benchmark"])),
            population_id=cast(str, typed["population_id"]),
            source_id=cast(str, typed["source_id"]),
            reward=float(value["reward"]),
            success=cast(bool, typed["success"]),
            native_fields=native,
            environment_id=cast(str, typed["environment_id"]),
            verifier_version=cast(str, typed["verifier_version"]),
            episode_steps=cast(int, typed["episode_steps"]),
            tool_calls=cast(int, typed["tool_calls"]),
            model_requests=cast(int, typed["model_requests"]),
            model_request_errors=cast(int, typed["model_request_errors"]),
            parse_errors=cast(int, typed["parse_errors"]),
            wall_time_seconds=float(value["wall_time_seconds"]),
            json_syntax_errors=(
                cast(int, typed["parse_errors"])
                if legacy
                else cast(int, typed["json_syntax_errors"])
            ),
            action_schema_errors=(0 if legacy else cast(int, typed["action_schema_errors"])),
            unavailable_skills=(0 if legacy else cast(int, typed["unavailable_skills"])),
            unsupported_resources=(0 if legacy else cast(int, typed["unsupported_resources"])),
            unsupported_tools=(0 if legacy else cast(int, typed["unsupported_tools"])),
            invalid_arguments=(0 if legacy else cast(int, typed["invalid_arguments"])),
            invalid_completions=(0 if legacy else cast(int, typed["invalid_completions"])),
            valid_actions=(0 if legacy else cast(int, typed["valid_actions"])),
            environment_admitted_actions=(
                0 if legacy else cast(int, typed["environment_admitted_actions"])
            ),
            explicit_completions=(0 if legacy else cast(int, typed["explicit_completions"])),
            environment_terminals=(0 if legacy else cast(int, typed["environment_terminals"])),
            horizon_no_submissions=(0 if legacy else cast(int, typed["horizon_no_submissions"])),
            scorer_invocations=(0 if legacy else cast(int, typed["scorer_invocations"])),
            format=PROTOCOL_V10_EVALUATION_RECORD_FORMAT,
        )


@dataclass(frozen=True, slots=True)
class _PlannedEvaluationItem:
    global_position: int
    benchmark: BenchmarkV10
    population_id: str
    source_id: str
    task: RolloutTask
    factory: PrivateSessionFactory


def plan_protocol_v10_evaluation(
    catalog: ProtocolV10PopulationCatalog,
    sessions: ProtocolV10PopulationSessionRegistry,
    mode: ProtocolV10EvaluationMode,
) -> tuple[_PlannedEvaluationItem, ...]:
    planned: list[_PlannedEvaluationItem] = []
    for benchmark in ACTIVE_BENCHMARKS_V10:
        for population in catalog.population(benchmark, mode.population_role):
            factory = sessions.factory(population.spec.population_id)
            for item in population.items:
                planned.append(
                    _PlannedEvaluationItem(
                        global_position=len(planned),
                        benchmark=benchmark,
                        population_id=population.spec.population_id,
                        source_id=item.source_id,
                        task=item.task,
                        factory=factory,
                    )
                )
    if not planned:
        raise ValueError("Protocol 10 evaluation population is empty")
    return tuple(planned)


class ProtocolV10EvaluationStore:
    """Private append-only shard store with explicit resume semantics."""

    def __init__(self, directory: Path, exact_input: ProtocolV10EvaluationInput) -> None:
        if not directory.is_absolute():
            raise ValueError("evaluation store must use an absolute private directory")
        self.directory = directory.resolve()
        self.input = exact_input
        self.input_path = self.directory / "input.json"
        self.records_path = self.directory / "records.jsonl"
        self.complete_path = self.directory / "COMPLETE"

    def open(self) -> tuple[ProtocolV10EvaluationRecord, ...]:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        if self.input_path.exists():
            existing = ProtocolV10EvaluationInput.from_value(
                json.loads(self.input_path.read_text(encoding="utf-8"))
            )
            if existing != self.input:
                raise ValueError("evaluation store belongs to another exact input")
        else:
            descriptor = os.open(self.input_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.input.to_value(), stream, separators=(",", ":"), sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        if not self.records_path.exists():
            descriptor = os.open(
                self.records_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.close(descriptor)
        records = tuple(
            ProtocolV10EvaluationRecord.from_value(json.loads(line))
            for line in self.records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        positions = tuple(record.global_position for record in records)
        if len(positions) != len(set(positions)):
            raise ValueError("evaluation store repeats a planned position")
        return records

    def append(self, record: ProtocolV10EvaluationRecord) -> None:
        if self.complete_path.exists():
            raise RuntimeError("completed evaluation store is immutable")
        with self.records_path.open("a", encoding="utf-8", newline="\n") as stream:
            json.dump(record.to_value(), stream, separators=(",", ":"), sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def load_completed(self) -> tuple[ProtocolV10EvaluationRecord, ...]:
        records = self.open()
        if not self.complete_path.is_file():
            raise RuntimeError("evaluation shard is not complete")
        return records

    def mark_complete(self) -> None:
        descriptor = os.open(
            self.complete_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write("complete\n")
            stream.flush()
            os.fsync(stream.fileno())


@dataclass(slots=True)
class ProtocolV10EvaluationWorker:
    exact_input: ProtocolV10EvaluationInput
    catalog: ProtocolV10PopulationCatalog
    sessions: ProtocolV10PopulationSessionRegistry
    runner: ProtocolV10ReadOnlyEpisodeRunner
    state_reader: ProtocolV10StateReader
    store: ProtocolV10EvaluationStore

    async def run(self) -> tuple[ProtocolV10EvaluationRecord, ...]:
        before = self.state_reader()
        plan = plan_protocol_v10_evaluation(self.catalog, self.sessions, self.exact_input.mode)
        if len(plan) != self.exact_input.expected_record_count:
            raise ValueError("evaluation exact input differs from the frozen population count")
        shard = tuple(
            item
            for item in plan
            if item.global_position % self.exact_input.shard_count == self.exact_input.shard_index
        )
        existing = self.store.open()
        expected_positions = {item.global_position for item in shard}
        if any(record.global_position not in expected_positions for record in existing):
            raise ValueError("evaluation store contains a record from another shard")
        by_position = {record.global_position: record for record in existing}
        for item in shard:
            if item.global_position in by_position:
                continue
            session = item.factory.create(item.task)
            failure: Exception | None = None
            outcome: ProtocolV10EvaluationOutcome | None = None
            try:
                outcome = await self.runner.evaluate(item.task, session)
            except Exception as error:
                failure = error
            if session.cleanup is not None:
                try:
                    await session.cleanup()
                except Exception as error:
                    if failure is None:
                        failure = error
            if failure is not None:
                raise ProtocolV10EvaluationInfrastructureError(
                    item.benchmark,
                    item.global_position,
                ) from failure
            if outcome is None:
                raise AssertionError("evaluation runner returned no outcome")
            payload = outcome.reward.native_payload
            native = payload.get("native_fields") if isinstance(payload, dict) else None
            if not isinstance(native, dict):
                raise ValueError("Protocol 10 reward omits trusted native fields")
            record = ProtocolV10EvaluationRecord(
                global_position=item.global_position,
                benchmark=item.benchmark,
                population_id=item.population_id,
                source_id=item.source_id,
                reward=outcome.reward.value,
                success=outcome.reward.success,
                native_fields=native,
                environment_id=outcome.reward.environment_id,
                verifier_version=outcome.reward.verifier_version,
                episode_steps=outcome.episode_steps,
                tool_calls=outcome.tool_calls,
                model_requests=outcome.model_requests,
                model_request_errors=outcome.model_request_errors,
                parse_errors=outcome.parse_errors,
                wall_time_seconds=outcome.wall_time_seconds,
                json_syntax_errors=outcome.json_syntax_errors,
                action_schema_errors=outcome.action_schema_errors,
                unavailable_skills=outcome.unavailable_skills,
                unsupported_resources=outcome.unsupported_resources,
                unsupported_tools=outcome.unsupported_tools,
                invalid_arguments=outcome.invalid_arguments,
                invalid_completions=outcome.invalid_completions,
                valid_actions=outcome.valid_actions,
                environment_admitted_actions=outcome.environment_admitted_actions,
                explicit_completions=outcome.explicit_completions,
                environment_terminals=outcome.environment_terminals,
                horizon_no_submissions=outcome.horizon_no_submissions,
                scorer_invocations=outcome.scorer_invocations,
            )
            self.store.append(record)
            by_position[record.global_position] = record
        if self.state_reader() != before:
            raise RuntimeError("Protocol 10 evaluation mutated frozen training state")
        ordered = tuple(by_position[item.global_position] for item in shard)
        if not self.store.complete_path.exists():
            self.store.mark_complete()
        return ordered


@dataclass(frozen=True, slots=True)
class ProtocolV10PopulationMetrics:
    benchmark: BenchmarkV10
    population_id: str
    record_count: int
    candidate_failure_count: int
    posterior_failure_count: int
    partial_reward_count: int
    native_means: dict[str, float]
    success_rate: float

    def to_value(self) -> dict[str, JsonValue]:
        native_means = normalize_json(self.native_means)
        if not isinstance(native_means, dict):
            raise AssertionError("native metric means did not normalize to an object")
        return {
            "benchmark": self.benchmark.value,
            "candidate_failure_count": self.candidate_failure_count,
            "partial_reward_count": self.partial_reward_count,
            "posterior_failure_count": self.posterior_failure_count,
            "native_means": native_means,
            "population_id": self.population_id,
            "record_count": self.record_count,
            "success_rate": self.success_rate,
        }


@dataclass(frozen=True, slots=True)
class ProtocolV10AggregateReport:
    mode: ProtocolV10EvaluationMode
    method: FormalMethodV10
    checkpoint_id: str
    protocol_id: str
    population_manifest_id: str
    evaluator_bundle_id: str
    model_id: str
    tokenizer_id: str
    generation_config_id: str
    evaluated_records: int
    candidate_failure_count: int
    posterior_failure_count: int
    partial_reward_count: int
    action_attempt_count: int
    parse_error_count: int
    parse_error_rate: float
    json_syntax_error_count: int
    json_syntax_error_rate: float
    action_schema_error_count: int
    action_schema_error_rate: float
    unavailable_skill_count: int
    unsupported_resource_count: int
    unsupported_tool_count: int
    invalid_arguments_count: int
    invalid_completion_count: int
    valid_action_count: int
    valid_action_rate: float
    environment_admitted_action_count: int
    explicit_completion_count: int
    environment_terminal_count: int
    horizon_no_submission_count: int
    scorer_invocation_count: int
    submission_rate: float
    average_episode_steps: float
    tool_call_count: int
    model_request_count: int
    model_request_error_count: int
    model_request_error_rate: float
    wall_time_seconds: float
    overall_reward: float
    overall_success_rate: float
    benchmark_macro_reward: float
    benchmark_macro_success_rate: float
    populations: tuple[ProtocolV10PopulationMetrics, ...]
    appworld_scenario_goal_completion: float | None
    infrastructure_failure_count: int = 0
    format: str = PROTOCOL_V10_EVALUATION_REPORT_FORMAT

    def __post_init__(self) -> None:
        if self.format != PROTOCOL_V10_EVALUATION_REPORT_FORMAT:
            raise ValueError("Protocol 10 evaluation report format is unsupported")
        text = (
            self.checkpoint_id,
            self.protocol_id,
            self.population_manifest_id,
            self.evaluator_bundle_id,
            self.model_id,
            self.tokenizer_id,
            self.generation_config_id,
        )
        if any(not item.strip() for item in text):
            raise ValueError("Protocol 10 evaluation report identity is incomplete")
        counters = (
            self.evaluated_records,
            self.candidate_failure_count,
            self.posterior_failure_count,
            self.partial_reward_count,
            self.action_attempt_count,
            self.parse_error_count,
            self.json_syntax_error_count,
            self.action_schema_error_count,
            self.unavailable_skill_count,
            self.unsupported_resource_count,
            self.unsupported_tool_count,
            self.invalid_arguments_count,
            self.invalid_completion_count,
            self.valid_action_count,
            self.environment_admitted_action_count,
            self.explicit_completion_count,
            self.environment_terminal_count,
            self.horizon_no_submission_count,
            self.scorer_invocation_count,
            self.tool_call_count,
            self.model_request_count,
            self.model_request_error_count,
            self.infrastructure_failure_count,
        )
        if any(type(item) is not int or item < 0 for item in counters):
            raise ValueError("Protocol 10 evaluation report counters are invalid")
        rates = (
            self.parse_error_rate,
            self.json_syntax_error_rate,
            self.action_schema_error_rate,
            self.valid_action_rate,
            self.submission_rate,
            self.model_request_error_rate,
            self.overall_reward,
            self.overall_success_rate,
            self.benchmark_macro_reward,
            self.benchmark_macro_success_rate,
        )
        if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in rates):
            raise ValueError("Protocol 10 evaluation report rate is invalid")
        if (
            not math.isfinite(self.average_episode_steps)
            or self.average_episode_steps < 0
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds < 0
        ):
            raise ValueError("Protocol 10 evaluation report timing is invalid")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "appworld_scenario_goal_completion": self.appworld_scenario_goal_completion,
            "average_episode_steps": self.average_episode_steps,
            "candidate_failure_count": self.candidate_failure_count,
            "posterior_failure_count": self.posterior_failure_count,
            "partial_reward_count": self.partial_reward_count,
            "action_attempt_count": self.action_attempt_count,
            "action_schema_error_count": self.action_schema_error_count,
            "action_schema_error_rate": self.action_schema_error_rate,
            "json_syntax_error_count": self.json_syntax_error_count,
            "json_syntax_error_rate": self.json_syntax_error_rate,
            "unavailable_skill_count": self.unavailable_skill_count,
            "unsupported_resource_count": self.unsupported_resource_count,
            "unsupported_tool_count": self.unsupported_tool_count,
            "invalid_arguments_count": self.invalid_arguments_count,
            "invalid_completion_count": self.invalid_completion_count,
            "valid_action_count": self.valid_action_count,
            "valid_action_rate": self.valid_action_rate,
            "environment_admitted_action_count": self.environment_admitted_action_count,
            "explicit_completion_count": self.explicit_completion_count,
            "environment_terminal_count": self.environment_terminal_count,
            "horizon_no_submission_count": self.horizon_no_submission_count,
            "scorer_invocation_count": self.scorer_invocation_count,
            "submission_rate": self.submission_rate,
            "checkpoint_id": self.checkpoint_id,
            "evaluator_bundle_id": self.evaluator_bundle_id,
            "evaluated_records": self.evaluated_records,
            "format": self.format,
            "generation_config_id": self.generation_config_id,
            "infrastructure_failure_count": self.infrastructure_failure_count,
            "method": self.method.value,
            "mode": self.mode.value,
            "model_id": self.model_id,
            "model_request_error_rate": self.model_request_error_rate,
            "overall_reward": self.overall_reward,
            "overall_success_rate": self.overall_success_rate,
            "benchmark_macro_reward": self.benchmark_macro_reward,
            "benchmark_macro_success_rate": self.benchmark_macro_success_rate,
            "parse_error_rate": self.parse_error_rate,
            "parse_error_count": self.parse_error_count,
            "populations": [population.to_value() for population in self.populations],
            "population_manifest_id": self.population_manifest_id,
            "protocol_id": self.protocol_id,
            "tokenizer_id": self.tokenizer_id,
            "tool_call_count": self.tool_call_count,
            "model_request_count": self.model_request_count,
            "model_request_error_count": self.model_request_error_count,
            "wall_time_seconds": self.wall_time_seconds,
        }


def aggregate_protocol_v10_evaluation(
    exact_inputs: tuple[ProtocolV10EvaluationInput, ...],
    shards: tuple[tuple[ProtocolV10EvaluationRecord, ...], ...],
) -> ProtocolV10AggregateReport:
    if not exact_inputs or len(exact_inputs) != len(shards):
        raise ValueError("evaluation aggregation requires matching shard inputs")
    first = exact_inputs[0]
    if any(
        item.run_id != first.run_id
        or item.method is not first.method
        or item.mode is not first.mode
        or item.checkpoint_id != first.checkpoint_id
        or item.protocol_id != first.protocol_id
        or item.population_manifest_id != first.population_manifest_id
        or item.evaluator_bundle_id != first.evaluator_bundle_id
        or item.model_id != first.model_id
        or item.tokenizer_id != first.tokenizer_id
        or item.generation_config_id != first.generation_config_id
        or item.expected_record_count != first.expected_record_count
        or item.shard_count != len(exact_inputs)
        for item in exact_inputs
    ) or {item.shard_index for item in exact_inputs} != set(range(len(exact_inputs))):
        raise ValueError("evaluation shard inputs do not form one complete run")
    records = tuple(
        sorted((record for shard in shards for record in shard), key=lambda x: x.global_position)
    )
    if len(records) != first.expected_record_count or tuple(
        record.global_position for record in records
    ) != tuple(range(first.expected_record_count)):
        raise ValueError("evaluation records are incomplete or duplicated")
    grouped: dict[tuple[BenchmarkV10, str], list[ProtocolV10EvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.benchmark, record.population_id)].append(record)
    populations: list[ProtocolV10PopulationMetrics] = []
    for (benchmark, population_id), members in grouped.items():
        numeric: dict[str, list[float]] = defaultdict(list)
        for record in members:
            for key, value in record.native_fields.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    numeric[key].append(float(value))
        populations.append(
            ProtocolV10PopulationMetrics(
                benchmark=benchmark,
                population_id=population_id,
                record_count=len(members),
                candidate_failure_count=sum(
                    item.reward == 0.0 and item.scorer_invocations == 1 for item in members
                ),
                posterior_failure_count=sum(
                    not item.success and item.scorer_invocations == 1 for item in members
                ),
                partial_reward_count=sum(
                    item.reward > 0.0 and not item.success and item.scorer_invocations == 1
                    for item in members
                ),
                native_means={
                    key: math.fsum(values) / len(values) for key, values in sorted(numeric.items())
                },
                success_rate=math.fsum(float(item.success) for item in members) / len(members),
            )
        )
    scenario_members: dict[str, list[ProtocolV10EvaluationRecord]] = defaultdict(list)
    for record in records:
        if record.benchmark is BenchmarkV10.APPWORLD:
            scenario = record.native_fields.get("scenario-id")
            if type(scenario) is not str or not scenario.strip():
                raise ValueError("AppWorld evaluation record omits its scenario group")
            scenario_members[scenario].append(record)
    sgc = None
    if scenario_members:
        complete_groups = True
        for members in scenario_members.values():
            expected_values = {
                item.native_fields.get("scenario-expected-task-count") for item in members
            }
            if (
                len(expected_values) != 1
                or type(next(iter(expected_values))) is not int
                or len(members) != next(iter(expected_values))
            ):
                complete_groups = False
                break
        if complete_groups:
            sgc = math.fsum(
                float(all(item.success for item in members))
                for members in scenario_members.values()
            ) / len(scenario_members)
    model_requests = sum(item.model_requests for item in records)
    model_request_errors = sum(item.model_request_errors for item in records)
    action_attempts = sum(item.episode_steps for item in records)
    json_syntax_errors = sum(item.json_syntax_errors for item in records)
    action_schema_errors = sum(item.action_schema_errors for item in records)
    parse_errors = json_syntax_errors + action_schema_errors
    if parse_errors == 0:
        parse_errors = sum(item.parse_errors for item in records)
        json_syntax_errors = parse_errors
    valid_actions = sum(item.valid_actions for item in records)
    by_benchmark: dict[BenchmarkV10, list[ProtocolV10EvaluationRecord]] = defaultdict(list)
    for record in records:
        by_benchmark[record.benchmark].append(record)
    benchmark_rewards = [
        math.fsum(item.reward for item in members) / len(members)
        for members in by_benchmark.values()
    ]
    benchmark_success = [
        math.fsum(float(item.success) for item in members) / len(members)
        for members in by_benchmark.values()
    ]
    scorer_invocations = sum(item.scorer_invocations for item in records)
    return ProtocolV10AggregateReport(
        mode=first.mode,
        method=first.method,
        checkpoint_id=first.checkpoint_id,
        protocol_id=first.protocol_id,
        population_manifest_id=first.population_manifest_id,
        evaluator_bundle_id=first.evaluator_bundle_id,
        model_id=first.model_id,
        tokenizer_id=first.tokenizer_id,
        generation_config_id=first.generation_config_id,
        evaluated_records=len(records),
        candidate_failure_count=sum(
            item.reward == 0.0 and item.scorer_invocations == 1 for item in records
        ),
        posterior_failure_count=sum(
            not item.success and item.scorer_invocations == 1 for item in records
        ),
        partial_reward_count=sum(
            item.reward > 0.0 and not item.success and item.scorer_invocations == 1
            for item in records
        ),
        action_attempt_count=action_attempts,
        parse_error_count=parse_errors,
        parse_error_rate=(parse_errors / action_attempts if action_attempts else 0.0),
        json_syntax_error_count=json_syntax_errors,
        json_syntax_error_rate=(json_syntax_errors / action_attempts if action_attempts else 0.0),
        action_schema_error_count=action_schema_errors,
        action_schema_error_rate=(
            action_schema_errors / action_attempts if action_attempts else 0.0
        ),
        unavailable_skill_count=sum(item.unavailable_skills for item in records),
        unsupported_resource_count=sum(item.unsupported_resources for item in records),
        unsupported_tool_count=sum(item.unsupported_tools for item in records),
        invalid_arguments_count=sum(item.invalid_arguments for item in records),
        invalid_completion_count=sum(item.invalid_completions for item in records),
        valid_action_count=valid_actions,
        valid_action_rate=(valid_actions / action_attempts if action_attempts else 0.0),
        environment_admitted_action_count=sum(
            item.environment_admitted_actions for item in records
        ),
        explicit_completion_count=sum(item.explicit_completions for item in records),
        environment_terminal_count=sum(item.environment_terminals for item in records),
        horizon_no_submission_count=sum(item.horizon_no_submissions for item in records),
        scorer_invocation_count=scorer_invocations,
        submission_rate=scorer_invocations / len(records),
        average_episode_steps=sum(item.episode_steps for item in records) / len(records),
        tool_call_count=sum(item.tool_calls for item in records),
        model_request_count=model_requests,
        model_request_error_count=model_request_errors,
        model_request_error_rate=(model_request_errors / model_requests if model_requests else 0.0),
        wall_time_seconds=math.fsum(item.wall_time_seconds for item in records),
        overall_reward=math.fsum(item.reward for item in records) / len(records),
        overall_success_rate=math.fsum(float(item.success) for item in records) / len(records),
        benchmark_macro_reward=math.fsum(benchmark_rewards) / len(benchmark_rewards),
        benchmark_macro_success_rate=(math.fsum(benchmark_success) / len(benchmark_success)),
        populations=tuple(populations),
        appworld_scenario_goal_completion=sgc,
    )


@dataclass(frozen=True, slots=True)
class ProtocolV10EvaluationSupervisor:
    """Merge only complete private shards into one answer-free report."""

    exact_inputs: tuple[ProtocolV10EvaluationInput, ...]
    stores: tuple[ProtocolV10EvaluationStore, ...]
    report_path: Path

    def __post_init__(self) -> None:
        if not self.exact_inputs or len(self.exact_inputs) != len(self.stores):
            raise ValueError("evaluation supervisor requires matching shards")
        if not self.report_path.is_absolute():
            raise ValueError("evaluation report path must be absolute")
        if any(
            store.input != exact
            for store, exact in zip(self.stores, self.exact_inputs, strict=True)
        ):
            raise ValueError("evaluation supervisor store inputs differ")

    def aggregate_completed(self) -> ProtocolV10AggregateReport:
        report = aggregate_protocol_v10_evaluation(
            self.exact_inputs,
            tuple(store.load_completed() for store in self.stores),
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.report_path.parent, 0o700)
        serialized = json.dumps(report.to_value(), separators=(",", ":"), sort_keys=True) + "\n"
        if self.report_path.exists():
            if self.report_path.read_text(encoding="utf-8") != serialized:
                raise RuntimeError("evaluation report already exists with different content")
            return report
        staging = self.report_path.with_suffix(self.report_path.suffix + ".staging")
        with ExitStack() as cleanup:
            cleanup.callback(staging.unlink, missing_ok=True)
            descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staging, self.report_path)
            directory_descriptor = os.open(self.report_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            cleanup.pop_all()
        return report


__all__ = [
    "PROTOCOL_V10_EVALUATION_INPUT_FORMAT",
    "PROTOCOL_V10_EVALUATION_RECORD_FORMAT",
    "PROTOCOL_V10_EVALUATION_REPORT_FORMAT",
    "ProtocolV10AggregateReport",
    "ProtocolV10EvaluationInfrastructureError",
    "ProtocolV10EvaluationInput",
    "ProtocolV10EvaluationMode",
    "ProtocolV10EvaluationOutcome",
    "ProtocolV10EvaluationRecord",
    "ProtocolV10EvaluationStore",
    "ProtocolV10EvaluationSupervisor",
    "ProtocolV10EvaluationWorker",
    "ProtocolV10PopulationMetrics",
    "ProtocolV10ReadOnlyEpisodeRunner",
    "ProtocolV10ReadOnlyState",
    "ProtocolV10StateReader",
    "aggregate_protocol_v10_evaluation",
    "plan_protocol_v10_evaluation",
]
