"""Preregistered, result-blind evaluation of committed evolution actions.

The counterfactual evidence in this module is deliberately aggregate-only.  A
private evaluator may retain the object, but its wire format contains no task
identity, task text, answer, native evaluator payload, or per-item reward.  The
classification functions consume only the protocol fixed before evaluation and
paired benchmark moments from the identical frozen task selections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from skillev.contracts import (
    EvolutionActionRecord,
    EvolutionActionType,
    JsonValue,
    normalize_json,
    stable_hash,
    validate_sha256,
)

from .comparison import BenchmarkAggregateOutcome
from .metrics import EvolutionErrorObservation
from .protocol import (
    BENCHMARK_SPECS,
    FIXED_SEED,
    Benchmark,
    BenchmarkRole,
    ProtocolFreeze,
)

EVOLUTION_ERROR_PROTOCOL_FORMAT: Final = "skillev-evolution-error-protocol@2"
EVOLUTION_COUNTERFACTUAL_FORMAT: Final = "skillev-evolution-counterfactual@2"

_IID_BENCHMARKS: Final = tuple(
    spec.benchmark for spec in BENCHMARK_SPECS if spec.role is BenchmarkRole.IID
)
_IID_ORDER: Final = {benchmark: index for index, benchmark in enumerate(_IID_BENCHMARKS)}
_FLOAT_AGGREGATE_EQUALITY_TOLERANCE: Final = Decimal("1e-12")


def _object(value: object, *, fields: frozenset[str], label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or frozenset(normalized) != fields:
        raise ValueError(f"{label} has an invalid field set")
    return normalized


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _count(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _json_object(value: dict[str, object]) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise TypeError("evolution-error value must normalize to an object")
    return normalized


def _content_hash(value: object) -> str:
    result: str = stable_hash(value)
    return result


def _outcome_value(outcome: BenchmarkAggregateOutcome) -> dict[str, JsonValue]:
    return _json_object(
        {
            "benchmark_id": outcome.benchmark_id,
            "reward_squared_sum": outcome.reward_squared_sum,
            "reward_sum": outcome.reward_sum,
            "sample_count": outcome.sample_count,
            "success_count": outcome.success_count,
        }
    )


def _outcome_from_value(value: object) -> BenchmarkAggregateOutcome:
    data = _object(
        value,
        fields=frozenset(
            {
                "benchmark_id",
                "reward_squared_sum",
                "reward_sum",
                "sample_count",
                "success_count",
            }
        ),
        label="benchmark aggregate outcome",
    )
    return BenchmarkAggregateOutcome(
        benchmark_id=_text(data["benchmark_id"], field="benchmark_id"),
        sample_count=_count(data["sample_count"], field="sample_count", positive=True),
        reward_sum=_number(data["reward_sum"], field="reward_sum"),
        reward_squared_sum=_number(data["reward_squared_sum"], field="reward_squared_sum"),
        success_count=_count(data["success_count"], field="success_count"),
    )


@dataclass(frozen=True, slots=True)
class FrozenIIDEvaluationSelection:
    """Identity and size of one admitted IID task selection, never its members."""

    benchmark: Benchmark
    selection_hash: str
    selection_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark) or self.benchmark not in _IID_ORDER:
            raise ValueError("evolution-error selections must be IID benchmarks")
        validate_sha256(self.selection_hash)
        _count(self.selection_count, field="selection_count", positive=True)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "selection_count": self.selection_count,
            "selection_hash": self.selection_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> FrozenIIDEvaluationSelection:
        data = _object(
            value,
            fields=frozenset({"benchmark", "selection_count", "selection_hash"}),
            label="frozen IID evaluation selection",
        )
        return cls(
            benchmark=Benchmark(_text(data["benchmark"], field="benchmark")),
            selection_hash=_text(data["selection_hash"], field="selection_hash"),
            selection_count=_count(data["selection_count"], field="selection_count", positive=True),
        )

    @property
    def content_hash(self) -> str:
        return _content_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class EvolutionErrorEvaluationProtocol:
    """All result-blind controls fixed before counterfactual evaluation.

    ``reward_tolerance`` is both the allowed regression margin for Retain/Prune
    and the minimum reward improvement required from Refine/Split.  Supplying it
    explicitly prevents any threshold from being inferred from observed results.
    """

    benchmark_protocol_hash: str
    protocol_freeze_id: str
    seed: int
    selections: tuple[FrozenIIDEvaluationSelection, ...]
    decoding_snapshot_id: str
    state_policy_identity: str
    post_window_optimizer_steps: int
    post_window_trajectory_count: int
    reward_tolerance: float
    minimum_product_invocations: int

    def __post_init__(self) -> None:
        validate_sha256(self.benchmark_protocol_hash)
        validate_sha256(self.protocol_freeze_id)
        ProtocolFreeze(self.benchmark_protocol_hash, self.protocol_freeze_id)
        validate_sha256(self.decoding_snapshot_id)
        validate_sha256(self.state_policy_identity)
        if self.seed != FIXED_SEED:
            raise ValueError("evolution-error evaluation must use the preregistered seed")
        if not isinstance(self.selections, tuple) or not self.selections:
            raise ValueError("evolution-error evaluation requires admitted IID selections")
        if any(not isinstance(item, FrozenIIDEvaluationSelection) for item in self.selections):
            raise TypeError("selections must contain FrozenIIDEvaluationSelection values")
        benchmarks = tuple(item.benchmark for item in self.selections)
        if len(set(benchmarks)) != len(benchmarks):
            raise ValueError("evolution-error selections must be unique")
        if benchmarks != tuple(sorted(benchmarks, key=_IID_ORDER.__getitem__)):
            raise ValueError("evolution-error selections must follow the frozen IID suite order")
        _count(
            self.post_window_optimizer_steps,
            field="post_window_optimizer_steps",
            positive=True,
        )
        _count(
            self.post_window_trajectory_count,
            field="post_window_trajectory_count",
            positive=True,
        )
        if (
            self.post_window_trajectory_count < self.post_window_optimizer_steps
            or self.post_window_trajectory_count % self.post_window_optimizer_steps
        ):
            raise ValueError(
                "post-window trajectory count must define an integer positive batch size"
            )
        tolerance = _number(self.reward_tolerance, field="reward_tolerance")
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError("reward_tolerance must lie in [0, 1]")
        object.__setattr__(self, "reward_tolerance", tolerance)
        _count(
            self.minimum_product_invocations,
            field="minimum_product_invocations",
            positive=True,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return _json_object(
            {
                "benchmark_protocol_hash": self.benchmark_protocol_hash,
                "decoding_snapshot_id": self.decoding_snapshot_id,
                "format": EVOLUTION_ERROR_PROTOCOL_FORMAT,
                "minimum_product_invocations": self.minimum_product_invocations,
                "post_window_optimizer_steps": self.post_window_optimizer_steps,
                "post_window_trajectory_count": self.post_window_trajectory_count,
                "protocol_freeze_id": self.protocol_freeze_id,
                "reward_tolerance": self.reward_tolerance,
                "seed": self.seed,
                "selections": [item.to_value() for item in self.selections],
                "state_policy_identity": self.state_policy_identity,
            }
        )

    @classmethod
    def from_value(cls, value: object) -> EvolutionErrorEvaluationProtocol:
        data = _object(
            value,
            fields=frozenset(
                {
                    "benchmark_protocol_hash",
                    "decoding_snapshot_id",
                    "format",
                    "minimum_product_invocations",
                    "post_window_optimizer_steps",
                    "post_window_trajectory_count",
                    "protocol_freeze_id",
                    "reward_tolerance",
                    "seed",
                    "selections",
                    "state_policy_identity",
                }
            ),
            label="evolution-error evaluation protocol",
        )
        if data["format"] != EVOLUTION_ERROR_PROTOCOL_FORMAT:
            raise ValueError("unsupported evolution-error protocol format")
        raw_selections = data["selections"]
        if not isinstance(raw_selections, list):
            raise ValueError("selections must be an array")
        return cls(
            benchmark_protocol_hash=_text(
                data["benchmark_protocol_hash"], field="benchmark_protocol_hash"
            ),
            protocol_freeze_id=_text(data["protocol_freeze_id"], field="protocol_freeze_id"),
            seed=_count(data["seed"], field="seed"),
            selections=tuple(
                FrozenIIDEvaluationSelection.from_value(item) for item in raw_selections
            ),
            decoding_snapshot_id=_text(data["decoding_snapshot_id"], field="decoding_snapshot_id"),
            state_policy_identity=_text(
                data["state_policy_identity"], field="state_policy_identity"
            ),
            post_window_optimizer_steps=_count(
                data["post_window_optimizer_steps"],
                field="post_window_optimizer_steps",
                positive=True,
            ),
            post_window_trajectory_count=_count(
                data["post_window_trajectory_count"],
                field="post_window_trajectory_count",
                positive=True,
            ),
            reward_tolerance=_number(data["reward_tolerance"], field="reward_tolerance"),
            minimum_product_invocations=_count(
                data["minimum_product_invocations"],
                field="minimum_product_invocations",
                positive=True,
            ),
        )

    @property
    def content_hash(self) -> str:
        return _content_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class CounterfactualBenchmarkAggregate:
    """One aggregate reward outcome bound to a frozen anonymous selection."""

    selection: FrozenIIDEvaluationSelection
    outcome: BenchmarkAggregateOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.selection, FrozenIIDEvaluationSelection):
            raise TypeError("counterfactual benchmark requires a frozen selection")
        if not isinstance(self.outcome, BenchmarkAggregateOutcome):
            raise TypeError("counterfactual benchmark requires an aggregate outcome")
        if (
            self.outcome.benchmark_id != self.selection.benchmark.value
            or self.outcome.sample_count != self.selection.selection_count
        ):
            raise ValueError("counterfactual outcome differs from its frozen selection")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "outcome": _outcome_value(self.outcome),
            "selection": self.selection.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> CounterfactualBenchmarkAggregate:
        data = _object(
            value,
            fields=frozenset({"outcome", "selection"}),
            label="counterfactual benchmark aggregate",
        )
        return cls(
            selection=FrozenIIDEvaluationSelection.from_value(data["selection"]),
            outcome=_outcome_from_value(data["outcome"]),
        )

    @property
    def content_hash(self) -> str:
        return _content_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class EvolutionCounterfactualArmAggregate:
    """Aggregate-only control or treatment evaluated under one frozen policy."""

    seed: int
    decoding_snapshot_id: str
    state_policy_identity: str
    library_version: str
    post_window_optimizer_steps: int
    benchmarks: tuple[CounterfactualBenchmarkAggregate, ...]

    def __post_init__(self) -> None:
        if self.seed != FIXED_SEED:
            raise ValueError("counterfactual arms must use the preregistered seed")
        validate_sha256(self.decoding_snapshot_id)
        validate_sha256(self.state_policy_identity)
        _text(self.library_version, field="library_version")
        _count(
            self.post_window_optimizer_steps,
            field="post_window_optimizer_steps",
            positive=True,
        )
        if not isinstance(self.benchmarks, tuple) or not self.benchmarks:
            raise ValueError("counterfactual arms require benchmark aggregates")
        if any(not isinstance(item, CounterfactualBenchmarkAggregate) for item in self.benchmarks):
            raise TypeError("benchmarks must contain CounterfactualBenchmarkAggregate values")
        benchmark_ids = tuple(item.selection.benchmark for item in self.benchmarks)
        if len(set(benchmark_ids)) != len(benchmark_ids):
            raise ValueError("counterfactual benchmark aggregates must be unique")
        if benchmark_ids != tuple(sorted(benchmark_ids, key=_IID_ORDER.__getitem__)):
            raise ValueError("counterfactual benchmarks must follow the frozen IID suite order")

    @property
    def mean_reward(self) -> float:
        count = sum(item.outcome.sample_count for item in self.benchmarks)
        result: float = math.fsum(item.outcome.reward_sum for item in self.benchmarks) / count
        return result

    def to_value(self) -> dict[str, JsonValue]:
        return _json_object(
            {
                "benchmarks": [item.to_value() for item in self.benchmarks],
                "decoding_snapshot_id": self.decoding_snapshot_id,
                "library_version": self.library_version,
                "post_window_optimizer_steps": self.post_window_optimizer_steps,
                "seed": self.seed,
                "state_policy_identity": self.state_policy_identity,
            }
        )

    @classmethod
    def from_value(cls, value: object) -> EvolutionCounterfactualArmAggregate:
        data = _object(
            value,
            fields=frozenset(
                {
                    "benchmarks",
                    "decoding_snapshot_id",
                    "library_version",
                    "post_window_optimizer_steps",
                    "seed",
                    "state_policy_identity",
                }
            ),
            label="evolution counterfactual arm",
        )
        raw_benchmarks = data["benchmarks"]
        if not isinstance(raw_benchmarks, list):
            raise ValueError("benchmarks must be an array")
        return cls(
            seed=_count(data["seed"], field="seed"),
            decoding_snapshot_id=_text(data["decoding_snapshot_id"], field="decoding_snapshot_id"),
            state_policy_identity=_text(
                data["state_policy_identity"], field="state_policy_identity"
            ),
            library_version=_text(data["library_version"], field="library_version"),
            post_window_optimizer_steps=_count(
                data["post_window_optimizer_steps"],
                field="post_window_optimizer_steps",
                positive=True,
            ),
            benchmarks=tuple(
                CounterfactualBenchmarkAggregate.from_value(item) for item in raw_benchmarks
            ),
        )

    @property
    def content_hash(self) -> str:
        return _content_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class EvolutionProductUsageAggregate:
    """Distinct selected-item inclusion and invocation counts for one product."""

    skill_id: str
    inclusion_count: int
    invocation_count: int

    def __post_init__(self) -> None:
        _text(self.skill_id, field="skill_id")
        included = _count(self.inclusion_count, field="inclusion_count")
        invoked = _count(self.invocation_count, field="invocation_count")
        if invoked > included:
            raise ValueError("product invocation_count cannot exceed inclusion_count")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "inclusion_count": self.inclusion_count,
            "invocation_count": self.invocation_count,
            "skill_id": self.skill_id,
        }

    @classmethod
    def from_value(cls, value: object) -> EvolutionProductUsageAggregate:
        data = _object(
            value,
            fields=frozenset({"inclusion_count", "invocation_count", "skill_id"}),
            label="evolution product usage aggregate",
        )
        return cls(
            skill_id=_text(data["skill_id"], field="skill_id"),
            inclusion_count=_count(data["inclusion_count"], field="inclusion_count"),
            invocation_count=_count(data["invocation_count"], field="invocation_count"),
        )

    @property
    def content_hash(self) -> str:
        return _content_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class EvolutionActionCounterfactualEvidence:
    """Private-side paired aggregates for exactly one committed action.

    The treatment is defined by the protocol as the old library plus only the
    referenced action.  ``treatment.library_version`` identifies that private
    counterfactual library and therefore need not equal the multi-action commit
    version recorded by :class:`EvolutionActionRecord`.
    """

    evaluation_protocol_hash: str
    action_id: str
    action_record_hash: str
    action_type: EvolutionActionType
    control: EvolutionCounterfactualArmAggregate
    treatment: EvolutionCounterfactualArmAggregate
    product_usage: tuple[EvolutionProductUsageAggregate, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.evaluation_protocol_hash)
        _text(self.action_id, field="action_id")
        validate_sha256(self.action_record_hash)
        if not isinstance(self.action_type, EvolutionActionType):
            raise TypeError("action_type must be EvolutionActionType")
        if not isinstance(self.control, EvolutionCounterfactualArmAggregate) or not isinstance(
            self.treatment, EvolutionCounterfactualArmAggregate
        ):
            raise TypeError("counterfactual evidence requires control and treatment arms")
        shared_fields = (
            "seed",
            "decoding_snapshot_id",
            "state_policy_identity",
            "post_window_optimizer_steps",
        )
        if any(
            getattr(self.control, field) != getattr(self.treatment, field)
            for field in shared_fields
        ):
            raise ValueError("control and treatment must share seed, decoding, policy, and window")
        if tuple(item.selection for item in self.control.benchmarks) != tuple(
            item.selection for item in self.treatment.benchmarks
        ):
            raise ValueError("control and treatment must use identical frozen task selections")
        if self.control.library_version == self.treatment.library_version:
            raise ValueError("control and treatment libraries must be distinct")
        if not isinstance(self.product_usage, tuple) or any(
            not isinstance(item, EvolutionProductUsageAggregate) for item in self.product_usage
        ):
            raise TypeError("product_usage must contain EvolutionProductUsageAggregate values")
        product_ids = tuple(item.skill_id for item in self.product_usage)
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("product usage rows must be unique")

    def to_value(self) -> dict[str, JsonValue]:
        return _json_object(
            {
                "action_id": self.action_id,
                "action_record_hash": self.action_record_hash,
                "action_type": self.action_type.value,
                "control": self.control.to_value(),
                "evaluation_protocol_hash": self.evaluation_protocol_hash,
                "format": EVOLUTION_COUNTERFACTUAL_FORMAT,
                "product_usage": [item.to_value() for item in self.product_usage],
                "treatment": self.treatment.to_value(),
            }
        )

    @classmethod
    def from_value(cls, value: object) -> EvolutionActionCounterfactualEvidence:
        data = _object(
            value,
            fields=frozenset(
                {
                    "action_id",
                    "action_record_hash",
                    "action_type",
                    "control",
                    "evaluation_protocol_hash",
                    "format",
                    "product_usage",
                    "treatment",
                }
            ),
            label="evolution action counterfactual evidence",
        )
        if data["format"] != EVOLUTION_COUNTERFACTUAL_FORMAT:
            raise ValueError("unsupported evolution counterfactual format")
        raw_product_usage = data["product_usage"]
        if not isinstance(raw_product_usage, list):
            raise ValueError("product_usage must be an array")
        return cls(
            evaluation_protocol_hash=_text(
                data["evaluation_protocol_hash"], field="evaluation_protocol_hash"
            ),
            action_id=_text(data["action_id"], field="action_id"),
            action_record_hash=_text(data["action_record_hash"], field="action_record_hash"),
            action_type=EvolutionActionType(_text(data["action_type"], field="action_type")),
            control=EvolutionCounterfactualArmAggregate.from_value(data["control"]),
            treatment=EvolutionCounterfactualArmAggregate.from_value(data["treatment"]),
            product_usage=tuple(
                EvolutionProductUsageAggregate.from_value(item) for item in raw_product_usage
            ),
        )

    @property
    def content_hash(self) -> str:
        return _content_hash(self.to_value())


def _validate_evidence_identity(
    protocol: EvolutionErrorEvaluationProtocol,
    action: EvolutionActionRecord,
    evidence: EvolutionActionCounterfactualEvidence,
) -> None:
    if not isinstance(protocol, EvolutionErrorEvaluationProtocol):
        raise TypeError("protocol must be EvolutionErrorEvaluationProtocol")
    if not isinstance(action, EvolutionActionRecord):
        raise TypeError("action must be EvolutionActionRecord")
    if not isinstance(evidence, EvolutionActionCounterfactualEvidence):
        raise TypeError("evidence must be EvolutionActionCounterfactualEvidence")
    if (
        evidence.evaluation_protocol_hash != protocol.content_hash
        or evidence.action_id != action.action_id
        or evidence.action_record_hash != action.content_hash
        or evidence.action_type is not action.action_type
    ):
        raise ValueError("counterfactual evidence identity differs from protocol or action")
    for arm in (evidence.control, evidence.treatment):
        if (
            arm.seed != protocol.seed
            or arm.decoding_snapshot_id != protocol.decoding_snapshot_id
            or arm.state_policy_identity != protocol.state_policy_identity
            or arm.post_window_optimizer_steps != protocol.post_window_optimizer_steps
            or tuple(item.selection for item in arm.benchmarks) != protocol.selections
        ):
            raise ValueError("counterfactual arm differs from the preregistered controls")
    if evidence.control.library_version != action.library_version_before:
        raise ValueError("counterfactual control is not the action's old library")
    product_ids = tuple(item.skill_id for item in evidence.product_usage)
    if product_ids != action.produced_skill_ids:
        raise ValueError("counterfactual product usage differs from the committed action")
    if any(
        item.inclusion_count > protocol.post_window_trajectory_count
        for item in evidence.product_usage
    ):
        raise ValueError("product inclusion_count exceeds the preregistered post-window population")


def classify_evolution_action_error(
    protocol: EvolutionErrorEvaluationProtocol,
    action: EvolutionActionRecord,
    evidence: EvolutionActionCounterfactualEvidence,
) -> bool:
    """Classify one action using only its preregistered paired aggregates."""

    _validate_evidence_identity(protocol, action, evidence)
    control_total = sum(
        (Decimal(str(item.outcome.reward_sum)) for item in evidence.control.benchmarks),
        start=Decimal(0),
    )
    treatment_total = sum(
        (Decimal(str(item.outcome.reward_sum)) for item in evidence.treatment.benchmarks),
        start=Decimal(0),
    )
    sample_count = sum(item.outcome.sample_count for item in evidence.control.benchmarks)
    reward_change = (treatment_total - control_total) / Decimal(sample_count)
    tolerance = Decimal(str(protocol.reward_tolerance))
    underused_product = any(
        item.invocation_count < protocol.minimum_product_invocations
        for item in evidence.product_usage
    )

    def below(value: Decimal, boundary: Decimal) -> bool:
        # Aggregate moments arrive as binary floats.  This tolerance establishes
        # numeric equality only; the scientific decision margin remains the
        # preregistered ``reward_tolerance`` above.
        return value < boundary - _FLOAT_AGGREGATE_EQUALITY_TOLERANCE

    if action.action_type is EvolutionActionType.RETAIN_COMPRESS:
        return below(reward_change, -tolerance)
    if action.action_type is EvolutionActionType.REFINE:
        return below(reward_change, tolerance)
    if action.action_type is EvolutionActionType.SPLIT:
        return underused_product or below(reward_change, tolerance)
    if action.action_type is EvolutionActionType.PRUNE:
        return below(reward_change, -tolerance)
    if action.action_type is EvolutionActionType.GENERATE:
        return underused_product
    raise ValueError("unsupported evolution action type")


def aggregate_evolution_error_observations(
    protocol: EvolutionErrorEvaluationProtocol,
    actions: tuple[EvolutionActionRecord, ...],
    evidences: tuple[EvolutionActionCounterfactualEvidence, ...],
) -> tuple[EvolutionErrorObservation, ...]:
    """Return one aggregate row for every action type, including zero-count rows."""

    if not isinstance(protocol, EvolutionErrorEvaluationProtocol):
        raise TypeError("protocol must be EvolutionErrorEvaluationProtocol")
    if not isinstance(actions, tuple) or any(
        not isinstance(item, EvolutionActionRecord) for item in actions
    ):
        raise TypeError("actions must contain EvolutionActionRecord values")
    if not isinstance(evidences, tuple) or any(
        not isinstance(item, EvolutionActionCounterfactualEvidence) for item in evidences
    ):
        raise TypeError("evidences must contain EvolutionActionCounterfactualEvidence values")
    action_ids = tuple(item.action_id for item in actions)
    evidence_ids = tuple(item.action_id for item in evidences)
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("committed evolution actions must be unique")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("counterfactual action evidence must be unique")
    if set(action_ids) != set(evidence_ids):
        raise ValueError("every committed evolution action requires exactly one evidence row")

    evidence_by_id = {item.action_id: item for item in evidences}
    evaluated = dict.fromkeys(EvolutionActionType, 0)
    erroneous = dict.fromkeys(EvolutionActionType, 0)
    for action in actions:
        evidence = evidence_by_id[action.action_id]
        evaluated[action.action_type] += 1
        erroneous[action.action_type] += int(
            classify_evolution_action_error(protocol, action, evidence)
        )
    return tuple(
        EvolutionErrorObservation(
            action_type=action_type,
            evaluated_count=evaluated[action_type],
            error_count=erroneous[action_type],
        )
        for action_type in EvolutionActionType
    )


__all__ = [
    "EVOLUTION_COUNTERFACTUAL_FORMAT",
    "EVOLUTION_ERROR_PROTOCOL_FORMAT",
    "CounterfactualBenchmarkAggregate",
    "EvolutionActionCounterfactualEvidence",
    "EvolutionCounterfactualArmAggregate",
    "EvolutionErrorEvaluationProtocol",
    "EvolutionProductUsageAggregate",
    "FrozenIIDEvaluationSelection",
    "aggregate_evolution_error_observations",
    "classify_evolution_action_error",
]
