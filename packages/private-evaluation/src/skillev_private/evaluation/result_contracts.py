"""Private episode records and sealed, answer-free benchmark projections.

Private records retain evaluator truth only long enough to emit numerical,
answer-free aggregates.  A public aggregate never contains task identities,
submissions, native evaluator payloads, or a per-episode trace.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skillev.contracts import (
    SCIENTIFIC_SAMPLING_ALGORITHM,
    JsonValue,
    TerminalReward,
    normalize_json,
    stable_hash,
    validate_sha256,
)
from skillev.contracts.ttb_trajectory import OBSERVATION_STATUSES
from skillev.experiments import (
    Benchmark,
    EvaluationReportingProtocol,
    ExecutionHardwareIdentity,
    FrozenTaskSequenceIdentity,
    ImplementationBuildIdentity,
    TrainingEvolutionCounts,
)
from skillev.rollout import RolloutArtifact, RolloutTermination
from skillev.runtime import BudgetVector

PRIVATE_EVALUATION_EPISODE_FORMAT = "skillev-private-evaluation-episode@2"
PUBLIC_BENCHMARK_AGGREGATE_FORMAT = "skillev-public-benchmark-aggregate@6"

if TYPE_CHECKING:
    from skillev_private.experiments.formal_evaluation_manifest import (
        FormalEvaluationTerminal,
    )


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _count(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _number(value: object, *, field: str, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or (nonnegative and normalized < 0.0):
        raise ValueError(f"{field} must be finite and within its metric range")
    return normalized


def _exact_object(
    value: object,
    *,
    label: str,
    fields: frozenset[str],
) -> dict[str, Any]:
    data = normalize_json(value)
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return data


@dataclass(frozen=True, slots=True)
class NativeMetricValue:
    """One evaluator-declared scalar that is safe to aggregate publicly."""

    metric_name: str
    value: float

    def __post_init__(self) -> None:
        _text(self.metric_name, field="native metric_name")
        # HealthBench native per-record rubric scores can be negative. Only
        # the TTB projection, not publication evidence, is clipped to [0, 1].
        object.__setattr__(
            self, "value", _number(self.value, field="native metric value", nonnegative=False)
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {"metric_name": self.metric_name, "value": self.value}

    @classmethod
    def from_value(cls, value: object) -> NativeMetricValue:
        data = _exact_object(
            value,
            label="native metric value",
            fields=frozenset({"metric_name", "value"}),
        )
        return cls(
            metric_name=_text(data["metric_name"], field="native metric_name"),
            value=_number(data["value"], field="native metric value", nonnegative=False),
        )


def public_native_metric_values(reward: TerminalReward) -> tuple[NativeMetricValue, ...]:
    """Project only evaluator-designated scalar metrics from a private reward.

    Evaluators may explicitly provide a primary native score different from
    the training projection, as well as secondary metrics, under the reserved
    ``native_payload.public_metrics`` map;
    arbitrary payload fields are never guessed or exported.
    """

    if not isinstance(reward, TerminalReward):
        raise TypeError("native metric projection requires TerminalReward")
    values: dict[str, float] = {reward.native_metric_name: reward.value}
    payload = reward.native_payload
    if isinstance(payload, dict) and "public_metrics" in payload:
        raw_metrics = payload["public_metrics"]
        if not isinstance(raw_metrics, dict):
            raise ValueError("native public_metrics must be an object")
        for name, raw_value in raw_metrics.items():
            metric_name = _text(name, field="native public metric name")
            values[metric_name] = _number(
                raw_value, field=f"native public metric {metric_name}", nonnegative=False
            )
    return tuple(
        NativeMetricValue(metric_name=name, value=value) for name, value in sorted(values.items())
    )


@dataclass(frozen=True, slots=True)
class ObservationStatusCount:
    """Answer-free count of one concrete execution outcome."""

    observation_status: str
    count: int

    def __post_init__(self) -> None:
        if self.observation_status not in OBSERVATION_STATUSES:
            raise ValueError("observation status is unsupported")
        _count(self.count, field="observation status count")

    def to_value(self) -> dict[str, JsonValue]:
        return {"count": self.count, "observation_status": self.observation_status}

    @classmethod
    def from_value(cls, value: object) -> ObservationStatusCount:
        data = _exact_object(
            value,
            label="observation status count",
            fields=frozenset({"count", "observation_status"}),
        )
        return cls(
            observation_status=_text(data["observation_status"], field="observation status"),
            count=_count(data["count"], field="observation status count"),
        )


@dataclass(frozen=True, slots=True)
class RolloutTerminationCount:
    """Answer-free count of one closed rollout termination path."""

    termination: RolloutTermination
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.termination, RolloutTermination):
            raise TypeError("termination count requires RolloutTermination")
        _count(self.count, field="termination count")

    def to_value(self) -> dict[str, JsonValue]:
        return {"count": self.count, "termination": self.termination.value}

    @classmethod
    def from_value(cls, value: object) -> RolloutTerminationCount:
        data = _exact_object(
            value,
            label="rollout termination count",
            fields=frozenset({"count", "termination"}),
        )
        return cls(
            termination=RolloutTermination(_text(data["termination"], field="termination")),
            count=_count(data["count"], field="termination count"),
        )


@dataclass(frozen=True, slots=True)
class EvaluationEpisodeTelemetry:
    """One answer-free execution/cost summary derived from an admitted artifact."""

    resource_usage: BudgetVector
    action_count: int
    action_token_count: int
    reasoning_token_count: int
    elapsed_wall_time_milliseconds: int
    observation_status_counts: tuple[ObservationStatusCount, ...]
    termination: RolloutTermination

    def __post_init__(self) -> None:
        if not isinstance(self.resource_usage, BudgetVector):
            raise TypeError("episode telemetry requires BudgetVector usage")
        action_count = _count(self.action_count, field="episode action_count")
        _count(self.action_token_count, field="episode action_token_count")
        _count(self.reasoning_token_count, field="episode reasoning_token_count")
        _count(self.elapsed_wall_time_milliseconds, field="episode elapsed wall time")
        if not isinstance(self.termination, RolloutTermination):
            raise TypeError("episode telemetry requires RolloutTermination")
        if not isinstance(self.observation_status_counts, tuple) or any(
            not isinstance(item, ObservationStatusCount) for item in self.observation_status_counts
        ):
            raise TypeError("episode telemetry status counts are invalid")
        statuses = tuple(item.observation_status for item in self.observation_status_counts)
        if statuses != tuple(sorted(statuses)) or len(set(statuses)) != len(statuses):
            raise ValueError("episode telemetry status counts must be unique and sorted")
        if any(item.count == 0 for item in self.observation_status_counts):
            raise ValueError("episode telemetry must omit zero status counts")
        if sum(item.count for item in self.observation_status_counts) != action_count:
            raise ValueError("episode status counts must cover every action")
        if self.resource_usage.agent_turns != action_count:
            raise ValueError("episode agent-turn usage must equal action_count")

    @classmethod
    def from_rollout(
        cls,
        artifact: RolloutArtifact,
        *,
        resource_usage: BudgetVector,
        elapsed_wall_time_milliseconds: int,
    ) -> EvaluationEpisodeTelemetry:
        if not isinstance(artifact, RolloutArtifact):
            raise TypeError("episode telemetry requires RolloutArtifact")
        if not isinstance(resource_usage, BudgetVector):
            raise TypeError("episode telemetry requires BudgetVector usage")
        counts = Counter(step.observation_status for step in artifact.record.steps)
        return cls(
            resource_usage=resource_usage,
            action_count=artifact.record.horizon,
            action_token_count=sum(step.action_token_count for step in artifact.record.steps),
            reasoning_token_count=sum(artifact.manifest.reasoning_token_counts),
            elapsed_wall_time_milliseconds=elapsed_wall_time_milliseconds,
            observation_status_counts=tuple(
                ObservationStatusCount(observation_status=status, count=count)
                for status, count in sorted(counts.items())
            ),
            termination=artifact.manifest.termination,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "action_count": self.action_count,
            "action_token_count": self.action_token_count,
            "elapsed_wall_time_milliseconds": self.elapsed_wall_time_milliseconds,
            "observation_status_counts": [
                item.to_value() for item in self.observation_status_counts
            ],
            "reasoning_token_count": self.reasoning_token_count,
            "resource_usage": self.resource_usage.to_value(),
            "termination": self.termination.value,
        }

    @classmethod
    def from_value(cls, value: object) -> EvaluationEpisodeTelemetry:
        data = _exact_object(
            value,
            label="evaluation episode telemetry",
            fields=frozenset(
                {
                    "action_count",
                    "action_token_count",
                    "elapsed_wall_time_milliseconds",
                    "observation_status_counts",
                    "reasoning_token_count",
                    "resource_usage",
                    "termination",
                }
            ),
        )
        statuses = data["observation_status_counts"]
        if not isinstance(statuses, list):
            raise ValueError("episode status counts must be an array")
        return cls(
            resource_usage=BudgetVector.from_value(data["resource_usage"]),
            action_count=_count(data["action_count"], field="episode action_count"),
            action_token_count=_count(
                data["action_token_count"], field="episode action_token_count"
            ),
            reasoning_token_count=_count(
                data["reasoning_token_count"], field="episode reasoning_token_count"
            ),
            elapsed_wall_time_milliseconds=_count(
                data["elapsed_wall_time_milliseconds"], field="episode elapsed wall time"
            ),
            observation_status_counts=tuple(
                ObservationStatusCount.from_value(item) for item in statuses
            ),
            termination=RolloutTermination(_text(data["termination"], field="termination")),
        )


@dataclass(frozen=True, slots=True)
class PrivateEvaluationEpisodeResult:
    """One private result; it must never cross into a public identity or aggregate."""

    task_id: str
    benchmark: Benchmark
    reward: TerminalReward
    native_metrics: tuple[NativeMetricValue, ...]
    telemetry: EvaluationEpisodeTelemetry
    frozen_state_hash: str
    task_sequence_hash: str
    sequence_position: int
    policy_snapshot_id: str
    library_version: str
    format: str = PRIVATE_EVALUATION_EPISODE_FORMAT

    def __post_init__(self) -> None:
        _text(self.task_id, field="task_id")
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("private episode benchmark must be Benchmark")
        if not isinstance(self.reward, TerminalReward):
            raise TypeError("private episode reward must be TerminalReward")
        if (
            not isinstance(self.native_metrics, tuple)
            or not self.native_metrics
            or any(not isinstance(item, NativeMetricValue) for item in self.native_metrics)
        ):
            raise TypeError("private episode native metrics are invalid")
        names = tuple(item.metric_name for item in self.native_metrics)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("private episode native metrics must be unique and sorted")
        if self.native_metrics != public_native_metric_values(self.reward):
            raise ValueError("private episode native metrics differ from evaluator projection")
        EvaluationReportingProtocol().metric_protocol(self.benchmark).require_episode_metrics(
            reward_value=self.reward.value,
            reward_native_metric_name=self.reward.native_metric_name,
            metric_values={item.metric_name: item.value for item in self.native_metrics},
        )
        if not isinstance(self.telemetry, EvaluationEpisodeTelemetry):
            raise TypeError("private episode telemetry is invalid")
        for field in ("frozen_state_hash", "task_sequence_hash"):
            value = getattr(self, field)
            if type(value) is not str or not value.startswith("sha256:"):
                raise ValueError(f"private episode {field} must be a content hash")
        if type(self.sequence_position) is not int or self.sequence_position < 0:
            raise ValueError("private episode sequence_position must be non-negative")
        _text(self.policy_snapshot_id, field="policy_snapshot_id")
        _text(self.library_version, field="library_version")
        if self.format != PRIVATE_EVALUATION_EPISODE_FORMAT:
            raise ValueError("unsupported private evaluation episode format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "format": self.format,
            "frozen_state_hash": self.frozen_state_hash,
            "library_version": self.library_version,
            "native_metrics": [item.to_value() for item in self.native_metrics],
            "policy_snapshot_id": self.policy_snapshot_id,
            "reward": self.reward.to_value(),
            "sequence_position": self.sequence_position,
            "task_id": self.task_id,
            "task_sequence_hash": self.task_sequence_hash,
            "telemetry": self.telemetry.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> PrivateEvaluationEpisodeResult:
        data = _exact_object(
            value,
            label="private evaluation episode",
            fields=frozenset(
                {
                    "benchmark",
                    "format",
                    "frozen_state_hash",
                    "library_version",
                    "native_metrics",
                    "policy_snapshot_id",
                    "reward",
                    "sequence_position",
                    "task_id",
                    "task_sequence_hash",
                    "telemetry",
                }
            ),
        )
        metrics = data["native_metrics"]
        if not isinstance(metrics, list):
            raise ValueError("private episode native_metrics must be an array")
        if any(
            type(data[field]) is not str
            for field in (
                "benchmark",
                "format",
                "frozen_state_hash",
                "library_version",
                "policy_snapshot_id",
                "task_id",
                "task_sequence_hash",
            )
        ):
            raise ValueError("private episode text fields are invalid")
        return cls(
            task_id=_text(data["task_id"], field="task_id"),
            benchmark=Benchmark(data["benchmark"]),
            reward=TerminalReward.from_value(data["reward"]),
            native_metrics=tuple(NativeMetricValue.from_value(item) for item in metrics),
            telemetry=EvaluationEpisodeTelemetry.from_value(data["telemetry"]),
            frozen_state_hash=data["frozen_state_hash"],
            task_sequence_hash=data["task_sequence_hash"],
            sequence_position=data["sequence_position"],
            policy_snapshot_id=data["policy_snapshot_id"],
            library_version=data["library_version"],
            format=data["format"],
        )


@dataclass(frozen=True, slots=True)
class NativeMetricAggregate:
    """Sufficient moments for one public native benchmark metric."""

    metric_name: str
    sample_count: int
    value_sum: float
    value_squared_sum: float

    def __post_init__(self) -> None:
        _text(self.metric_name, field="native aggregate metric_name")
        sample_count = _count(self.sample_count, field="native aggregate sample_count")
        if sample_count < 1:
            raise ValueError("native aggregate sample_count must be positive")
        value_sum = _number(self.value_sum, field="native aggregate value_sum", nonnegative=False)
        squared_sum = _number(self.value_squared_sum, field="native aggregate value_squared_sum")
        if squared_sum + 1e-12 < value_sum * value_sum / sample_count:
            raise ValueError("native aggregate moments imply negative variance")
        object.__setattr__(self, "value_sum", value_sum)
        object.__setattr__(self, "value_squared_sum", squared_sum)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "metric_name": self.metric_name,
            "sample_count": self.sample_count,
            "value_squared_sum": self.value_squared_sum,
            "value_sum": self.value_sum,
        }

    @classmethod
    def from_value(cls, value: object) -> NativeMetricAggregate:
        data = _exact_object(
            value,
            label="native metric aggregate",
            fields=frozenset({"metric_name", "sample_count", "value_squared_sum", "value_sum"}),
        )
        return cls(
            metric_name=_text(data["metric_name"], field="native aggregate metric_name"),
            sample_count=_count(data["sample_count"], field="native aggregate sample_count"),
            value_sum=_number(
                data["value_sum"], field="native aggregate value_sum", nonnegative=False
            ),
            value_squared_sum=_number(
                data["value_squared_sum"], field="native aggregate value_squared_sum"
            ),
        )


@dataclass(frozen=True, slots=True)
class EvaluationCostAggregate:
    """Answer-free summed cost of an exact frozen evaluation sequence."""

    resource_usage: BudgetVector
    action_token_count: int
    reasoning_token_count: int
    elapsed_wall_time_milliseconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.resource_usage, BudgetVector):
            raise TypeError("evaluation cost requires BudgetVector usage")
        _count(self.action_token_count, field="aggregate action_token_count")
        _count(self.reasoning_token_count, field="aggregate reasoning_token_count")
        _count(self.elapsed_wall_time_milliseconds, field="aggregate elapsed wall time")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "action_token_count": self.action_token_count,
            "elapsed_wall_time_milliseconds": self.elapsed_wall_time_milliseconds,
            "reasoning_token_count": self.reasoning_token_count,
            "resource_usage": self.resource_usage.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> EvaluationCostAggregate:
        data = _exact_object(
            value,
            label="evaluation cost aggregate",
            fields=frozenset(
                {
                    "action_token_count",
                    "elapsed_wall_time_milliseconds",
                    "reasoning_token_count",
                    "resource_usage",
                }
            ),
        )
        return cls(
            resource_usage=BudgetVector.from_value(data["resource_usage"]),
            action_token_count=_count(
                data["action_token_count"], field="aggregate action_token_count"
            ),
            reasoning_token_count=_count(
                data["reasoning_token_count"], field="aggregate reasoning_token_count"
            ),
            elapsed_wall_time_milliseconds=_count(
                data["elapsed_wall_time_milliseconds"], field="aggregate elapsed wall time"
            ),
        )


@dataclass(frozen=True, slots=True)
class PublicBenchmarkAggregateResult:
    """A sealed answer-free aggregate suitable for the scientific result bundle."""

    benchmark: Benchmark
    task_sequence_identity: FrozenTaskSequenceIdentity
    reporting: EvaluationReportingProtocol
    source_training_attempt_id: str
    source_formal_run_group_id: str
    source_training_identity_hash: str
    source_evolution_counts: TrainingEvolutionCounts
    method_identity_hash: str
    frozen_state_hash: str
    formal_evaluation_id: str
    formal_evaluation_manifest_id: str
    formal_evaluation_terminal_hash: str
    sampling_schedule_algorithm: str
    sampling_schedule_hash: str
    evaluation_implementation_build: ImplementationBuildIdentity
    evaluation_execution_hardware: ExecutionHardwareIdentity
    sample_count: int
    success_count: int
    reward_sum: float
    reward_squared_sum: float
    native_metrics: tuple[NativeMetricAggregate, ...]
    cost: EvaluationCostAggregate
    observation_status_counts: tuple[ObservationStatusCount, ...]
    termination_counts: tuple[RolloutTerminationCount, ...]
    format: str = PUBLIC_BENCHMARK_AGGREGATE_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("public aggregate benchmark must be Benchmark")
        if not isinstance(self.task_sequence_identity, FrozenTaskSequenceIdentity):
            raise TypeError("public aggregate requires frozen task sequence identity")
        if self.reporting != EvaluationReportingProtocol():
            raise ValueError("public aggregate reporting protocol differs from preregistration")
        _text(self.source_training_attempt_id, field="source_training_attempt_id")
        validate_sha256(self.source_formal_run_group_id)
        if type(self.source_training_identity_hash) is not str:
            raise TypeError("public aggregate source_training_identity_hash must be text")
        validate_sha256(self.source_training_identity_hash)
        if not isinstance(self.source_evolution_counts, TrainingEvolutionCounts):
            raise TypeError("public aggregate source evolution counts are invalid")
        _text(self.method_identity_hash, field="method_identity_hash")
        _text(self.frozen_state_hash, field="frozen_state_hash")
        for field in (
            "formal_evaluation_id",
            "formal_evaluation_manifest_id",
            "formal_evaluation_terminal_hash",
            "sampling_schedule_hash",
        ):
            validate_sha256(getattr(self, field))
        if self.sampling_schedule_algorithm != SCIENTIFIC_SAMPLING_ALGORITHM:
            raise ValueError("public aggregate uses another scientific sampling algorithm")
        if not isinstance(self.evaluation_implementation_build, ImplementationBuildIdentity):
            raise TypeError("public aggregate lacks an exact evaluation build")
        if not isinstance(self.evaluation_execution_hardware, ExecutionHardwareIdentity):
            raise TypeError("public aggregate lacks an exact evaluation hardware identity")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ValueError("public aggregate sample_count must be positive")
        expected_count = next(
            (
                item.count
                for item in self.task_sequence_identity.benchmark_counts
                if item.benchmark is self.benchmark
            ),
            None,
        )
        if expected_count != self.sample_count:
            raise ValueError("public aggregate count differs from frozen benchmark count")
        if type(self.success_count) is not int or not 0 <= self.success_count <= self.sample_count:
            raise ValueError("public aggregate success_count is invalid")
        for field in ("reward_sum", "reward_squared_sum"):
            value = _number(getattr(self, field), field=field)
            if value > self.sample_count:
                raise ValueError(f"{field} lies outside reward support")
            object.__setattr__(self, field, value)
        if self.reward_squared_sum + 1e-12 < self.reward_sum * self.reward_sum / self.sample_count:
            raise ValueError("reward aggregate moments imply negative variance")
        if (
            not isinstance(self.native_metrics, tuple)
            or not self.native_metrics
            or any(not isinstance(item, NativeMetricAggregate) for item in self.native_metrics)
        ):
            raise TypeError("public aggregate native metrics are invalid")
        metric_names = tuple(item.metric_name for item in self.native_metrics)
        if metric_names != tuple(sorted(metric_names)) or len(set(metric_names)) != len(
            metric_names
        ):
            raise ValueError("public aggregate native metrics must be unique and sorted")
        self.reporting.metric_protocol(self.benchmark).require_aggregate_metrics(
            sample_count=self.sample_count,
            metric_moments={
                item.metric_name: (
                    item.sample_count,
                    item.value_sum,
                    item.value_squared_sum,
                )
                for item in self.native_metrics
            },
        )
        if not isinstance(self.cost, EvaluationCostAggregate):
            raise TypeError("public aggregate cost is invalid")
        if self.cost.resource_usage.agent_turns != sum(
            item.count for item in self.observation_status_counts
        ):
            raise ValueError("aggregate agent turns must equal observation counts")
        if not isinstance(self.observation_status_counts, tuple) or any(
            not isinstance(item, ObservationStatusCount) for item in self.observation_status_counts
        ):
            raise TypeError("public aggregate status counts are invalid")
        statuses = tuple(item.observation_status for item in self.observation_status_counts)
        if statuses != tuple(sorted(statuses)) or len(set(statuses)) != len(statuses):
            raise ValueError("public aggregate status counts must be unique and sorted")
        if any(item.count == 0 for item in self.observation_status_counts):
            raise ValueError("public aggregate must omit zero status counts")
        if not isinstance(self.termination_counts, tuple) or any(
            not isinstance(item, RolloutTerminationCount) for item in self.termination_counts
        ):
            raise TypeError("public aggregate termination counts are invalid")
        terminations = tuple(item.termination for item in self.termination_counts)
        if terminations != tuple(sorted(terminations, key=lambda item: item.value)) or len(
            set(terminations)
        ) != len(terminations):
            raise ValueError("public aggregate termination counts must be unique and sorted")
        if any(item.count == 0 for item in self.termination_counts):
            raise ValueError("public aggregate must omit zero termination counts")
        if sum(item.count for item in self.termination_counts) != self.sample_count:
            raise ValueError("termination counts must cover every frozen episode")
        if self.format != PUBLIC_BENCHMARK_AGGREGATE_FORMAT:
            raise ValueError("unsupported public benchmark aggregate format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "cost": self.cost.to_value(),
            "format": self.format,
            "frozen_state_hash": self.frozen_state_hash,
            "formal_evaluation_id": self.formal_evaluation_id,
            "formal_evaluation_manifest_id": self.formal_evaluation_manifest_id,
            "formal_evaluation_terminal_hash": self.formal_evaluation_terminal_hash,
            "sampling_schedule_algorithm": self.sampling_schedule_algorithm,
            "sampling_schedule_hash": self.sampling_schedule_hash,
            "evaluation_implementation_build": self.evaluation_implementation_build.to_value(),
            "evaluation_execution_hardware": self.evaluation_execution_hardware.to_value(),
            "method_identity_hash": self.method_identity_hash,
            "native_metrics": [item.to_value() for item in self.native_metrics],
            "observation_status_counts": [
                item.to_value() for item in self.observation_status_counts
            ],
            "reporting": self.reporting.to_value(),
            "reward_squared_sum": self.reward_squared_sum,
            "reward_sum": self.reward_sum,
            "sample_count": self.sample_count,
            "source_training_attempt_id": self.source_training_attempt_id,
            "source_formal_run_group_id": self.source_formal_run_group_id,
            "source_evolution_counts": self.source_evolution_counts.to_value(),
            "source_training_identity_hash": self.source_training_identity_hash,
            "success_count": self.success_count,
            "task_sequence_identity": self.task_sequence_identity.to_value(),
            "termination_counts": [item.to_value() for item in self.termination_counts],
        }

    @classmethod
    def from_value(cls, value: object) -> PublicBenchmarkAggregateResult:
        data = _exact_object(
            value,
            label="public benchmark aggregate",
            fields=frozenset(
                {
                    "benchmark",
                    "cost",
                    "format",
                    "frozen_state_hash",
                    "formal_evaluation_id",
                    "formal_evaluation_manifest_id",
                    "formal_evaluation_terminal_hash",
                    "sampling_schedule_algorithm",
                    "sampling_schedule_hash",
                    "evaluation_implementation_build",
                    "evaluation_execution_hardware",
                    "method_identity_hash",
                    "native_metrics",
                    "observation_status_counts",
                    "reporting",
                    "reward_squared_sum",
                    "reward_sum",
                    "sample_count",
                    "source_training_attempt_id",
                    "source_formal_run_group_id",
                    "source_evolution_counts",
                    "source_training_identity_hash",
                    "success_count",
                    "task_sequence_identity",
                    "termination_counts",
                }
            ),
        )
        metric_rows = data["native_metrics"]
        status_rows = data["observation_status_counts"]
        termination_rows = data["termination_counts"]
        if not all(isinstance(item, list) for item in (metric_rows, status_rows, termination_rows)):
            raise ValueError("public aggregate rows must be arrays")
        for field in (
            "benchmark",
            "format",
            "frozen_state_hash",
            "method_identity_hash",
            "source_training_attempt_id",
            "source_formal_run_group_id",
            "source_training_identity_hash",
            "formal_evaluation_id",
            "formal_evaluation_manifest_id",
            "formal_evaluation_terminal_hash",
            "sampling_schedule_algorithm",
            "sampling_schedule_hash",
        ):
            if type(data[field]) is not str:
                raise ValueError("public aggregate text fields are invalid")
        return cls(
            benchmark=Benchmark(data["benchmark"]),
            task_sequence_identity=FrozenTaskSequenceIdentity.from_value(
                data["task_sequence_identity"]
            ),
            reporting=EvaluationReportingProtocol.from_value(data["reporting"]),
            source_training_attempt_id=data["source_training_attempt_id"],
            source_formal_run_group_id=data["source_formal_run_group_id"],
            source_training_identity_hash=data["source_training_identity_hash"],
            source_evolution_counts=TrainingEvolutionCounts.from_value(
                data["source_evolution_counts"]
            ),
            method_identity_hash=data["method_identity_hash"],
            frozen_state_hash=data["frozen_state_hash"],
            formal_evaluation_id=data["formal_evaluation_id"],
            formal_evaluation_manifest_id=data["formal_evaluation_manifest_id"],
            formal_evaluation_terminal_hash=data["formal_evaluation_terminal_hash"],
            sampling_schedule_algorithm=data["sampling_schedule_algorithm"],
            sampling_schedule_hash=data["sampling_schedule_hash"],
            evaluation_implementation_build=ImplementationBuildIdentity.from_value(
                data["evaluation_implementation_build"]
            ),
            evaluation_execution_hardware=ExecutionHardwareIdentity.from_value(
                data["evaluation_execution_hardware"]
            ),
            sample_count=data["sample_count"],
            success_count=data["success_count"],
            reward_sum=data["reward_sum"],
            reward_squared_sum=data["reward_squared_sum"],
            native_metrics=tuple(NativeMetricAggregate.from_value(item) for item in metric_rows),
            cost=EvaluationCostAggregate.from_value(data["cost"]),
            observation_status_counts=tuple(
                ObservationStatusCount.from_value(item) for item in status_rows
            ),
            termination_counts=tuple(
                RolloutTerminationCount.from_value(item) for item in termination_rows
            ),
            format=data["format"],
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def require_comparable_wall_time(self, other: PublicBenchmarkAggregateResult) -> None:
        """Reject operational wall-time comparisons across execution identities."""

        if not isinstance(other, PublicBenchmarkAggregateResult):
            raise TypeError("wall-time comparison requires another public aggregate")
        if (
            self.evaluation_implementation_build != other.evaluation_implementation_build
            or self.evaluation_execution_hardware != other.evaluation_execution_hardware
        ):
            raise ValueError("evaluation wall time is not comparable across build or hardware")


def _sum_usage(episodes: tuple[PrivateEvaluationEpisodeResult, ...]) -> BudgetVector:
    total = BudgetVector()
    for episode in episodes:
        total = total.add(episode.telemetry.resource_usage)
    return total


def aggregate_private_episodes(
    episodes: tuple[PrivateEvaluationEpisodeResult, ...],
    *,
    benchmark: Benchmark,
    task_sequence_identity: FrozenTaskSequenceIdentity,
    expected_task_ids: tuple[str, ...],
    expected_sequence_positions: tuple[int, ...],
    state: object,
    reporting: EvaluationReportingProtocol,
    formal_evaluation_terminal: FormalEvaluationTerminal,
) -> PublicBenchmarkAggregateResult:
    """Project exact private episodes into one sealed answer-free aggregate."""

    # Keep public provenance derived from the admitted frozen state rather
    # than accepting caller-supplied identity strings.  Importing here avoids
    # a package-level cycle with private frozen-state schemas.
    from skillev_private.experiments.formal_evaluation_manifest import (
        FormalEvaluationTerminal,
        FormalEvaluationTerminalStatus,
    )
    from skillev_private.frozen_state import FrozenInferenceState
    from skillev_private.initial_baseline import InitialBaselineInferenceState
    from skillev_private.phase_anchor import PhaseAnchorInferenceState

    if not isinstance(
        state,
        FrozenInferenceState | InitialBaselineInferenceState | PhaseAnchorInferenceState,
    ):
        raise TypeError("aggregate requires an admitted frozen evaluation state")
    if reporting != EvaluationReportingProtocol():
        raise ValueError("aggregate reporting protocol differs from preregistration")
    if not isinstance(formal_evaluation_terminal, FormalEvaluationTerminal):
        raise TypeError("aggregate requires a formal evaluation terminal")
    if formal_evaluation_terminal.status is not FormalEvaluationTerminalStatus.SUCCEEDED:
        raise ValueError("aggregate requires a successful formal evaluation terminal")
    slot = formal_evaluation_terminal.slot
    if (
        slot.frozen_state_hash != state.content_hash
        or slot.policy_snapshot_id != state.policy_snapshot_id
        or slot.library_version != state.library.current_version
        or slot.task_sequence_identity != task_sequence_identity
    ):
        raise ValueError("formal evaluation terminal differs from the aggregate state")
    if (
        formal_evaluation_terminal.implementation_build is None
        or formal_evaluation_terminal.execution_hardware is None
    ):
        raise ValueError("formal evaluation terminal lacks measured provenance")
    frozen_state_hash = state.content_hash
    source_training_identity_hash = (
        state.source_training_identity_hash
        if isinstance(state, FrozenInferenceState | PhaseAnchorInferenceState)
        else state.source_identity_hash
    )
    if not episodes:
        raise ValueError("cannot aggregate an empty private evaluation")
    if any(item.benchmark is not benchmark for item in episodes):
        raise ValueError("private episodes contain another benchmark")
    expected_count = next(
        (
            item.count
            for item in task_sequence_identity.benchmark_counts
            if item.benchmark is benchmark
        ),
        None,
    )
    if len(episodes) != expected_count:
        raise ValueError("private episode count differs from frozen benchmark count")
    if (
        len(expected_task_ids) != expected_count
        or len(expected_sequence_positions) != expected_count
    ):
        raise ValueError("aggregate expected sequence does not match frozen benchmark count")
    if any(type(item) is not str or not item for item in expected_task_ids):
        raise ValueError("aggregate expected task IDs are invalid")
    if any(type(item) is not int or item < 0 for item in expected_sequence_positions):
        raise ValueError("aggregate expected sequence positions are invalid")
    if tuple(item.task_id for item in episodes) != expected_task_ids:
        raise ValueError("private episodes differ from the frozen ordered task sequence")
    if tuple(item.sequence_position for item in episodes) != expected_sequence_positions:
        raise ValueError("private episodes differ from frozen sequence positions")
    if any(
        item.frozen_state_hash != frozen_state_hash
        or item.task_sequence_hash != task_sequence_identity.ordered_task_ids_hash
        or item.policy_snapshot_id != state.policy_snapshot_id
        or item.library_version != state.library.current_version
        for item in episodes
    ):
        raise ValueError("private episodes differ from frozen state or task sequence")

    metrics: dict[str, list[float]] = {}
    statuses: Counter[str] = Counter()
    terminations: Counter[RolloutTermination] = Counter()
    for episode in episodes:
        for metric in episode.native_metrics:
            metrics.setdefault(metric.metric_name, []).append(metric.value)
        for status in episode.telemetry.observation_status_counts:
            statuses[status.observation_status] += status.count
        terminations[episode.telemetry.termination] += 1
    rewards = tuple(item.reward.value for item in episodes)
    return PublicBenchmarkAggregateResult(
        benchmark=benchmark,
        task_sequence_identity=task_sequence_identity,
        reporting=reporting,
        source_training_attempt_id=state.source_training_attempt_id,
        source_formal_run_group_id=state.source_formal_run_group_id,
        source_training_identity_hash=source_training_identity_hash,
        source_evolution_counts=state.evolution_counts,
        method_identity_hash=state.method_identity_hash,
        frozen_state_hash=frozen_state_hash,
        formal_evaluation_id=slot.evaluation_id,
        formal_evaluation_manifest_id=formal_evaluation_terminal.manifest_id,
        formal_evaluation_terminal_hash=formal_evaluation_terminal.content_hash,
        sampling_schedule_algorithm=SCIENTIFIC_SAMPLING_ALGORITHM,
        sampling_schedule_hash=slot.sampling_schedule_hash,
        evaluation_implementation_build=formal_evaluation_terminal.implementation_build,
        evaluation_execution_hardware=formal_evaluation_terminal.execution_hardware,
        sample_count=len(episodes),
        success_count=sum(item.reward.success for item in episodes),
        reward_sum=math.fsum(rewards),
        reward_squared_sum=math.fsum(item * item for item in rewards),
        native_metrics=tuple(
            NativeMetricAggregate(
                metric_name=name,
                sample_count=len(values),
                value_sum=math.fsum(values),
                value_squared_sum=math.fsum(value * value for value in values),
            )
            for name, values in sorted(metrics.items())
        ),
        cost=EvaluationCostAggregate(
            resource_usage=_sum_usage(episodes),
            action_token_count=sum(item.telemetry.action_token_count for item in episodes),
            reasoning_token_count=sum(item.telemetry.reasoning_token_count for item in episodes),
            elapsed_wall_time_milliseconds=sum(
                item.telemetry.elapsed_wall_time_milliseconds for item in episodes
            ),
        ),
        observation_status_counts=tuple(
            ObservationStatusCount(observation_status=status, count=count)
            for status, count in sorted(statuses.items())
        ),
        termination_counts=tuple(
            RolloutTerminationCount(termination=termination, count=count)
            for termination, count in sorted(terminations.items(), key=lambda item: item[0].value)
        ),
    )


__all__ = [
    "PRIVATE_EVALUATION_EPISODE_FORMAT",
    "PUBLIC_BENCHMARK_AGGREGATE_FORMAT",
    "EvaluationCostAggregate",
    "EvaluationEpisodeTelemetry",
    "NativeMetricAggregate",
    "NativeMetricValue",
    "ObservationStatusCount",
    "PrivateEvaluationEpisodeResult",
    "PublicBenchmarkAggregateResult",
    "RolloutTerminationCount",
    "aggregate_private_episodes",
    "public_native_metric_values",
]
