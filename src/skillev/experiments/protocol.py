"""Historical Protocol 9 reader for the eighteen-benchmark, seven-arm suite.

Only non-sensitive identities and predeclared controls live here.  Dataset
rows, answers, verifier payloads, pilot outcomes, and per-item results are not
part of this wire format and cannot change a benchmark's frozen role.  New
formal runs use :mod:`skillev.experiments.protocol_v10`; the values in this
module remain stable only so historical Protocol 9 artifacts stay readable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, TypeAlias, assert_never

from skillev.contracts import (
    SCIENTIFIC_SAMPLING_ALGORITHM,
    JsonValue,
    normalize_json,
    scientific_sampling_schedule_hash,
    stable_hash,
)
from skillev.runtime.attempt_protocol import AttemptBuilderKind

PROTOCOL_FORMAT: Final = "skillev-benchmark-protocol@9"
FREEZE_FORMAT: Final = "skillev-benchmark-protocol-freeze@9"
METHOD_IDENTITY_FORMAT: Final = "skillev-full-method-identity@4"
FIXED_SEED: Final = 20260721
FIXED_SUBSAMPLE_SIZE: Final = 500
FIXED_SUBSAMPLE_ALGORITHM: Final = "stable-hash-ranked-item-id@1"
EVALUATION_POPULATION_THRESHOLD: Final = 1000
NATIVE_METRIC_INTERVAL: Final = "normal-moments-95@1"
BINARY_RATE_INTERVAL: Final = "wilson-95@1"

if TYPE_CHECKING:
    from .formal_execution import FormalExecutionFreeze


class Benchmark(StrEnum):
    """Recognized benchmark identities, including retired wire values."""

    HOTPOT_QA = "hotpotqa"
    TRIVIA_QA = "triviaqa"
    AIME_2026 = "aime-2026"
    MED_QA = "medqa"
    WEBSHOP = "webshop"
    ALFWORLD = "alfworld"
    BIRD_SQL = "bird-sql"
    APPWORLD = "appworld"
    MBPP_PLUS = "mbpp-plus"
    SKILLFLOW_BENCH = "skillflow-bench"
    MUSIQUE = "musique"
    NQ_OPEN = "nq-open"
    MATH_HARD = "math-hard"
    GPQA_DIAMOND = "gpqa-diamond"
    MIND2WEB = "mind2web"
    SCIENCE_WORLD = "scienceworld"
    TABLEBENCH = "tablebench"
    BFCL_V3 = "bfcl-v3"
    HUMANEVAL_PLUS = "humaneval-plus"
    TUA_BENCH = "tua-bench"


class BenchmarkRole(StrEnum):
    IID = "iid"
    OOD = "ood"


class TrainingUse(StrEnum):
    TRAINING_MIX = "training-mix"
    PROGRESS_ANCHOR = "progress-anchor"  # legacy wire value; absent from protocol v9
    EVALUATION_ONLY = "evaluation-only"


class AggregatePurpose(StrEnum):
    """Closed result-blind purposes for private aggregate accumulation."""

    IID_BASELINE = "iid-baseline"
    IID_TRAINING = "iid-training"
    IID_EVALUATION = "iid-evaluation"
    OOD_BASELINE = "ood-baseline"
    OOD_INFERENCE = "ood-inference"


class SchedulePurpose(StrEnum):
    """Schedule identities; IID_PROGRESS is retained only for old artifact reads."""

    IID_TRAINING = "iid-training"
    IID_PROGRESS = "iid-progress"
    IID_EVALUATION = "iid-evaluation"
    OOD_EVALUATION = "ood-evaluation"


class AblationArm(StrEnum):
    FULL = "full"
    SKILLFLOW_DISABLED = "skillflow-disabled"
    NO_FLOW_WEIGHTING = "no-flow-weighting"
    CAPPED_FLOW_WEIGHT = "capped-flow-weight"
    CLIPPED_IMPORTANCE = "clipped-importance"
    LCB_TO_MEAN = "lcb-to-mean"
    AND_TO_SINGLE_CONDITION = "and-to-single-condition"


def _object(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has an invalid field set")
    return normalized


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _uint64(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError(f"{field} must be an unsigned 64-bit integer")
    return value


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return result


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _sha256(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a SHA-256 content hash")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 content hash") from error
    return text


@dataclass(frozen=True, slots=True)
class BenchmarkTaskCount:
    """One answer-free benchmark count within a frozen task sequence."""

    benchmark: Benchmark
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("benchmark task count requires Benchmark")
        if type(self.count) is not int or self.count < 0:
            raise ValueError("benchmark task count must be non-negative")

    def to_value(self) -> dict[str, JsonValue]:
        return {"benchmark": self.benchmark.value, "count": self.count}

    @classmethod
    def from_value(cls, value: object) -> BenchmarkTaskCount:
        data = _object(
            value,
            fields=frozenset({"benchmark", "count"}),
            label="benchmark task count",
        )
        count = data["count"]
        if type(count) is not int:
            raise ValueError("benchmark task count must be an integer")
        return cls(
            benchmark=Benchmark(_text(data["benchmark"], field="benchmark")),
            count=count,
        )


@dataclass(frozen=True, slots=True)
class FrozenTaskSequenceIdentity:
    """Public, result-blind identity for one private ordered task schedule."""

    purpose: SchedulePurpose
    ordered_task_ids_hash: str
    task_count: int
    benchmark_counts: tuple[BenchmarkTaskCount, ...]
    schedule_algorithm: str
    format: str = "skillev-frozen-task-sequence@1"

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, SchedulePurpose):
            raise TypeError("frozen task sequence purpose must be SchedulePurpose")
        _sha256(self.ordered_task_ids_hash, field="ordered_task_ids_hash")
        if type(self.task_count) is not int or self.task_count < 1:
            raise ValueError("frozen task sequence must be non-empty")
        if not isinstance(self.benchmark_counts, tuple) or not self.benchmark_counts:
            raise ValueError("frozen task sequence requires benchmark counts")
        if any(not isinstance(item, BenchmarkTaskCount) for item in self.benchmark_counts):
            raise TypeError("frozen task sequence contains an invalid benchmark count")
        if sum(item.count for item in self.benchmark_counts) != self.task_count:
            raise ValueError("benchmark counts differ from task_count")
        benchmarks = tuple(item.benchmark for item in self.benchmark_counts)
        if benchmarks != tuple(sorted(benchmarks, key=lambda item: item.value)):
            raise ValueError("benchmark counts must use deterministic order")
        if len(set(benchmarks)) != len(benchmarks):
            raise ValueError("benchmark counts must be unique")
        _text(self.schedule_algorithm, field="schedule_algorithm")
        if self.format != "skillev-frozen-task-sequence@1":
            raise ValueError("unsupported frozen task sequence format")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark_counts": [item.to_value() for item in self.benchmark_counts],
            "format": self.format,
            "ordered_task_ids_hash": self.ordered_task_ids_hash,
            "purpose": self.purpose.value,
            "schedule_algorithm": self.schedule_algorithm,
            "task_count": self.task_count,
        }

    @classmethod
    def from_value(cls, value: object) -> FrozenTaskSequenceIdentity:
        data = _object(
            value,
            fields=frozenset(
                {
                    "benchmark_counts",
                    "format",
                    "ordered_task_ids_hash",
                    "purpose",
                    "schedule_algorithm",
                    "task_count",
                }
            ),
            label="frozen task sequence identity",
        )
        raw_counts = data["benchmark_counts"]
        if not isinstance(raw_counts, list):
            raise ValueError("frozen task sequence counts must be an array")
        task_count = data["task_count"]
        if type(task_count) is not int:
            raise ValueError("frozen task sequence task_count must be an integer")
        return cls(
            purpose=SchedulePurpose(_text(data["purpose"], field="purpose")),
            ordered_task_ids_hash=_sha256(
                data["ordered_task_ids_hash"], field="ordered_task_ids_hash"
            ),
            task_count=task_count,
            benchmark_counts=tuple(BenchmarkTaskCount.from_value(item) for item in raw_counts),
            schedule_algorithm=_text(data["schedule_algorithm"], field="schedule_algorithm"),
            format=_text(data["format"], field="format"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class FullEvaluationSampling:
    sampling_type: ClassVar[str] = "full"

    def to_value(self) -> dict[str, JsonValue]:
        return {"sampling_type": self.sampling_type}

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class FixedSubsampleEvaluation:
    SAMPLE_SIZE: ClassVar[int] = FIXED_SUBSAMPLE_SIZE
    ALGORITHM: ClassVar[str] = FIXED_SUBSAMPLE_ALGORITHM
    sampling_type: ClassVar[str] = "fixed-subsample"

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "algorithm": self.ALGORITHM,
            "sample_size": self.SAMPLE_SIZE,
            "sampling_type": self.sampling_type,
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class PopulationThresholdEvaluation:
    """Use all small evaluation populations and a frozen 500-item sample otherwise."""

    THRESHOLD: ClassVar[int] = EVALUATION_POPULATION_THRESHOLD
    SAMPLE_SIZE: ClassVar[int] = FIXED_SUBSAMPLE_SIZE
    ALGORITHM: ClassVar[str] = FIXED_SUBSAMPLE_ALGORITHM
    sampling_type: ClassVar[str] = "population-threshold"

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "algorithm": self.ALGORITHM,
            "sample_size": self.SAMPLE_SIZE,
            "sampling_type": self.sampling_type,
            "threshold": self.THRESHOLD,
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


EvaluationSampling: TypeAlias = (
    FullEvaluationSampling | FixedSubsampleEvaluation | PopulationThresholdEvaluation
)


def evaluation_sampling_from_value(value: object) -> EvaluationSampling:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError("evaluation sampling must be an object")
    sampling_type = normalized.get("sampling_type")
    if sampling_type == FullEvaluationSampling.sampling_type:
        _object(
            normalized,
            fields=frozenset({"sampling_type"}),
            label="full evaluation sampling",
        )
        return FullEvaluationSampling()
    if sampling_type == FixedSubsampleEvaluation.sampling_type:
        data = _object(
            normalized,
            fields=frozenset({"algorithm", "sample_size", "sampling_type"}),
            label="fixed-subsample evaluation",
        )
        sample_size = data["sample_size"]
        if type(sample_size) is not int:
            raise ValueError("sample_size must be an integer")
        if sample_size != FIXED_SUBSAMPLE_SIZE:
            raise ValueError("fixed subsample size differs from preregistration")
        if _text(data["algorithm"], field="algorithm") != FIXED_SUBSAMPLE_ALGORITHM:
            raise ValueError("fixed subsample algorithm differs from preregistration")
        return FixedSubsampleEvaluation()
    if sampling_type == PopulationThresholdEvaluation.sampling_type:
        data = _object(
            normalized,
            fields=frozenset({"algorithm", "sample_size", "sampling_type", "threshold"}),
            label="population-threshold evaluation",
        )
        if data["sample_size"] != FIXED_SUBSAMPLE_SIZE:
            raise ValueError("population-threshold sample size differs from preregistration")
        if data["threshold"] != EVALUATION_POPULATION_THRESHOLD:
            raise ValueError("evaluation population threshold differs from preregistration")
        if _text(data["algorithm"], field="algorithm") != FIXED_SUBSAMPLE_ALGORITHM:
            raise ValueError("population-threshold algorithm differs from preregistration")
        return PopulationThresholdEvaluation()
    raise ValueError("unsupported evaluation sampling type")


FULL_EVALUATION: Final = FullEvaluationSampling()
FIXED_500_EVALUATION: Final = FixedSubsampleEvaluation()
POPULATION_THRESHOLD_EVALUATION: Final = PopulationThresholdEvaluation()


def fixed_subsample_rank(
    benchmark_id: str,
    task_id: str,
) -> str:
    """Return the sole preregistered result-blind ordering key for one item."""

    benchmark = _text(benchmark_id, field="benchmark_id")
    identity = _text(task_id, field="task_id")
    return stable_hash(
        {
            "algorithm": FIXED_SUBSAMPLE_ALGORITHM,
            "benchmark_id": benchmark,
            "seed": FIXED_SEED,
            "task_id": identity,
        }
    )


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    benchmark: Benchmark
    role: BenchmarkRole
    evaluation_sampling: EvaluationSampling
    training_use: TrainingUse
    dataset_snapshot_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise ValueError("benchmark must be Benchmark")
        if not isinstance(self.role, BenchmarkRole):
            raise ValueError("role must be BenchmarkRole")
        if not isinstance(
            self.evaluation_sampling,
            FullEvaluationSampling | FixedSubsampleEvaluation | PopulationThresholdEvaluation,
        ):
            raise ValueError("evaluation_sampling has an unsupported variant")
        if not isinstance(self.training_use, TrainingUse):
            raise ValueError("training_use must be TrainingUse")
        _text(self.dataset_snapshot_name, field="dataset_snapshot_name")
        if self.role is BenchmarkRole.OOD:
            if self.training_use is not TrainingUse.EVALUATION_ONLY:
                raise ValueError("OOD benchmarks must be evaluation-only")
        elif self.training_use is TrainingUse.EVALUATION_ONLY:
            raise ValueError("IID benchmarks require a training role")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "dataset_snapshot_name": self.dataset_snapshot_name,
            "evaluation_sampling": self.evaluation_sampling.to_value(),
            "role": self.role.value,
            "training_use": self.training_use.value,
        }

    @classmethod
    def from_value(cls, value: object) -> BenchmarkSpec:
        data = _object(
            value,
            fields=frozenset(
                {
                    "benchmark",
                    "dataset_snapshot_name",
                    "evaluation_sampling",
                    "role",
                    "training_use",
                }
            ),
            label="benchmark spec",
        )
        return cls(
            benchmark=Benchmark(_text(data["benchmark"], field="benchmark")),
            role=BenchmarkRole(_text(data["role"], field="role")),
            evaluation_sampling=evaluation_sampling_from_value(data["evaluation_sampling"]),
            training_use=TrainingUse(_text(data["training_use"], field="training_use")),
            dataset_snapshot_name=_text(
                data["dataset_snapshot_name"], field="dataset_snapshot_name"
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


def _benchmark_spec(
    benchmark: Benchmark,
    role: BenchmarkRole,
    sampling: EvaluationSampling,
    training_use: TrainingUse,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark=benchmark,
        role=role,
        evaluation_sampling=sampling,
        training_use=training_use,
        dataset_snapshot_name=benchmark.value,
    )


BENCHMARK_SPECS: Final = (
    *(
        _benchmark_spec(
            benchmark,
            BenchmarkRole.IID,
            POPULATION_THRESHOLD_EVALUATION,
            TrainingUse.TRAINING_MIX,
        )
        for benchmark in (
            Benchmark.HOTPOT_QA,
            Benchmark.TRIVIA_QA,
            Benchmark.AIME_2026,
            Benchmark.MED_QA,
            Benchmark.WEBSHOP,
            Benchmark.ALFWORLD,
            Benchmark.BIRD_SQL,
            Benchmark.APPWORLD,
            Benchmark.MBPP_PLUS,
        )
    ),
    *(
        _benchmark_spec(
            benchmark,
            BenchmarkRole.OOD,
            (
                FULL_EVALUATION
                if benchmark in {Benchmark.GPQA_DIAMOND, Benchmark.BFCL_V3}
                else POPULATION_THRESHOLD_EVALUATION
            ),
            TrainingUse.EVALUATION_ONLY,
        )
        for benchmark in (
            Benchmark.MUSIQUE,
            Benchmark.NQ_OPEN,
            Benchmark.MATH_HARD,
            Benchmark.GPQA_DIAMOND,
            Benchmark.MIND2WEB,
            Benchmark.SCIENCE_WORLD,
            Benchmark.TABLEBENCH,
            Benchmark.BFCL_V3,
            Benchmark.HUMANEVAL_PLUS,
        )
    ),
)

ACTIVE_BENCHMARKS: Final[tuple[Benchmark, ...]] = tuple(spec.benchmark for spec in BENCHMARK_SPECS)


SCHEDULE_PURPOSES: Final[tuple[SchedulePurpose, ...]] = (
    SchedulePurpose.IID_TRAINING,
    SchedulePurpose.IID_EVALUATION,
    SchedulePurpose.OOD_EVALUATION,
)


def _benchmarks_for_schedule(purpose: SchedulePurpose) -> tuple[Benchmark, ...]:
    match purpose:
        case SchedulePurpose.IID_TRAINING:
            return tuple(
                spec.benchmark
                for spec in BENCHMARK_SPECS
                if spec.training_use is TrainingUse.TRAINING_MIX
            )
        case SchedulePurpose.IID_PROGRESS:
            return ()
        case SchedulePurpose.IID_EVALUATION:
            return tuple(
                spec.benchmark for spec in BENCHMARK_SPECS if spec.role is BenchmarkRole.IID
            )
        case SchedulePurpose.OOD_EVALUATION:
            return tuple(
                spec.benchmark for spec in BENCHMARK_SPECS if spec.role is BenchmarkRole.OOD
            )
        case _ as unreachable:
            from typing import assert_never

            assert_never(unreachable)


def _validate_schedule_identities(
    identities: tuple[FrozenTaskSequenceIdentity, ...],
) -> None:
    """Require the fixed purpose/order/domain coverage before a protocol freezes."""

    if not isinstance(identities, tuple) or any(
        not isinstance(item, FrozenTaskSequenceIdentity) for item in identities
    ):
        raise TypeError("schedule_identities must contain frozen task sequences")
    if tuple(item.purpose for item in identities) != SCHEDULE_PURPOSES:
        raise ValueError("schedule identities must contain the active fixed purposes in order")
    for identity in identities:
        expected = tuple(sorted(_benchmarks_for_schedule(identity.purpose), key=lambda x: x.value))
        actual = tuple(item.benchmark for item in identity.benchmark_counts)
        if actual != expected:
            raise ValueError("schedule benchmark counts differ from the declared benchmark role")
        if any(item.count < 1 for item in identity.benchmark_counts):
            raise ValueError("schedule benchmark counts must cover every declared benchmark")


class ArmProtocolValue(Protocol):
    arm: ClassVar[AblationArm]

    def to_value(self) -> dict[str, JsonValue]: ...


@dataclass(frozen=True, slots=True)
class FullArmProtocol:
    arm: ClassVar[AblationArm] = AblationArm.FULL

    def to_value(self) -> dict[str, JsonValue]:
        return {"arm": self.arm.value}


@dataclass(frozen=True, slots=True)
class NoBayesianCalibrationArmProtocol:
    arm: ClassVar[AblationArm] = AblationArm.SKILLFLOW_DISABLED

    def to_value(self) -> dict[str, JsonValue]:
        return {"arm": self.arm.value}


@dataclass(frozen=True, slots=True)
class NoFlowWeightingArmProtocol:
    arm: ClassVar[AblationArm] = AblationArm.NO_FLOW_WEIGHTING

    def to_value(self) -> dict[str, JsonValue]:
        return {"arm": self.arm.value}


@dataclass(frozen=True, slots=True)
class CappedFlowWeightArmProtocol:
    CAP: ClassVar[float] = 10.0
    arm: ClassVar[AblationArm] = AblationArm.CAPPED_FLOW_WEIGHT

    def to_value(self) -> dict[str, JsonValue]:
        return {"arm": self.arm.value, "cap": self.CAP}


@dataclass(frozen=True, slots=True)
class ClippedImportanceArmProtocol:
    CLIP: ClassVar[float] = 10.0
    arm: ClassVar[AblationArm] = AblationArm.CLIPPED_IMPORTANCE

    def to_value(self) -> dict[str, JsonValue]:
        return {"arm": self.arm.value, "clip": self.CLIP}


@dataclass(frozen=True, slots=True)
class PosteriorMeanArmProtocol:
    arm: ClassVar[AblationArm] = AblationArm.LCB_TO_MEAN

    def to_value(self) -> dict[str, JsonValue]:
        return {"arm": self.arm.value}


@dataclass(frozen=True, slots=True)
class ResidualOnlyPhaseArmProtocol:
    arm: ClassVar[AblationArm] = AblationArm.AND_TO_SINGLE_CONDITION

    def to_value(self) -> dict[str, JsonValue]:
        return {"arm": self.arm.value}


ArmProtocol: TypeAlias = (
    FullArmProtocol
    | NoBayesianCalibrationArmProtocol
    | NoFlowWeightingArmProtocol
    | CappedFlowWeightArmProtocol
    | ClippedImportanceArmProtocol
    | PosteriorMeanArmProtocol
    | ResidualOnlyPhaseArmProtocol
)


def arm_protocol_for_builder_kind(kind: AttemptBuilderKind) -> ArmProtocol:
    """Return the sole preregistered arm identity for one worker graph."""

    match kind:
        case AttemptBuilderKind.FULL:
            return FullArmProtocol()
        case AttemptBuilderKind.NO_BAYESIAN:
            return NoBayesianCalibrationArmProtocol()
        case AttemptBuilderKind.UNIT_FLOW:
            return NoFlowWeightingArmProtocol()
        case AttemptBuilderKind.CAPPED_FLOW:
            return CappedFlowWeightArmProtocol()
        case AttemptBuilderKind.CLIPPED_IMPORTANCE:
            return ClippedImportanceArmProtocol()
        case AttemptBuilderKind.POSTERIOR_MEAN:
            return PosteriorMeanArmProtocol()
        case AttemptBuilderKind.RESIDUAL_ONLY_PHASE:
            return ResidualOnlyPhaseArmProtocol()
        case _ as unreachable:
            from typing import assert_never

            assert_never(unreachable)


def arm_protocol_from_value(value: object) -> ArmProtocol:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError("arm protocol must be an object")
    arm = AblationArm(_text(normalized.get("arm"), field="arm"))
    if arm is AblationArm.FULL:
        _object(normalized, fields=frozenset({"arm"}), label="full arm")
        return FullArmProtocol()
    if arm is AblationArm.SKILLFLOW_DISABLED:
        _object(normalized, fields=frozenset({"arm"}), label="no-Bayesian arm")
        return NoBayesianCalibrationArmProtocol()
    if arm is AblationArm.NO_FLOW_WEIGHTING:
        _object(normalized, fields=frozenset({"arm"}), label="no-flow arm")
        return NoFlowWeightingArmProtocol()
    if arm is AblationArm.CAPPED_FLOW_WEIGHT:
        data = _object(
            normalized,
            fields=frozenset({"arm", "cap"}),
            label="capped-flow arm",
        )
        if _positive_float(data["cap"], field="cap") != CappedFlowWeightArmProtocol.CAP:
            raise ValueError("capped-flow arm cap differs from preregistration")
        return CappedFlowWeightArmProtocol()
    if arm is AblationArm.CLIPPED_IMPORTANCE:
        data = _object(
            normalized,
            fields=frozenset({"arm", "clip"}),
            label="clipped-importance arm",
        )
        if _positive_float(data["clip"], field="clip") != ClippedImportanceArmProtocol.CLIP:
            raise ValueError("clipped-importance arm clip differs from preregistration")
        return ClippedImportanceArmProtocol()
    if arm is AblationArm.LCB_TO_MEAN:
        _object(normalized, fields=frozenset({"arm"}), label="posterior-mean arm")
        return PosteriorMeanArmProtocol()
    if arm is AblationArm.AND_TO_SINGLE_CONDITION:
        _object(normalized, fields=frozenset({"arm"}), label="residual-only arm")
        return ResidualOnlyPhaseArmProtocol()
    # ``AblationArm`` is closed.  Keep the parser fail-closed if a new arm is
    # introduced without its frozen wire decoder.
    assert_never(arm)


ABLATION_PROTOCOLS: Final[tuple[ArmProtocol, ...]] = (
    FullArmProtocol(),
    NoBayesianCalibrationArmProtocol(),
    NoFlowWeightingArmProtocol(),
    CappedFlowWeightArmProtocol(),
    ClippedImportanceArmProtocol(),
    PosteriorMeanArmProtocol(),
    ResidualOnlyPhaseArmProtocol(),
)

FULL_POLICY_DISTRIBUTION: Final = "raw-categorical-softmax"
FULL_IMPORTANCE_ESTIMATOR: Final = "raw-log-importance"
FULL_FLOW_WEIGHTING: Final = "mean-normalized-uncapped"
FULL_GRADIENT_TRANSFORM: Final = "none"
FULL_OPTIMIZER: Final = "adamw-weight-decay-0"
FULL_POSTERIOR_DECISION: Final = "confidence-bound"
FULL_PHASE_RULE: Final = "residual-and-entropy"


@dataclass(frozen=True, slots=True)
class FullMethodIdentity:
    """Closed literal identity of the full method."""

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "flow_weighting": FULL_FLOW_WEIGHTING,
            "format": METHOD_IDENTITY_FORMAT,
            "gradient_transform": FULL_GRADIENT_TRANSFORM,
            "importance_estimator": FULL_IMPORTANCE_ESTIMATOR,
            "optimizer": FULL_OPTIMIZER,
            "phase_rule": FULL_PHASE_RULE,
            "policy_distribution": FULL_POLICY_DISTRIBUTION,
            "posterior_decision": FULL_POSTERIOR_DECISION,
        }

    @classmethod
    def from_value(cls, value: object) -> FullMethodIdentity:
        data = _object(
            value,
            fields=frozenset(
                {
                    "flow_weighting",
                    "format",
                    "gradient_transform",
                    "importance_estimator",
                    "optimizer",
                    "phase_rule",
                    "policy_distribution",
                    "posterior_decision",
                }
            ),
            label="full method identity",
        )
        expected = cls().to_value()
        if data != expected:
            raise ValueError("full method identity differs from the preregistered constants")
        return cls()

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class FrozenSinglePassOOD:
    protocol_type: ClassVar[str] = "frozen-library-posterior-policy-single-pass"

    def to_value(self) -> dict[str, JsonValue]:
        return {"protocol_type": self.protocol_type}

    @classmethod
    def from_value(cls, value: object) -> FrozenSinglePassOOD:
        data = _object(
            value,
            fields=frozenset({"protocol_type"}),
            label="OOD freeze protocol",
        )
        if data["protocol_type"] != cls.protocol_type:
            raise ValueError("unsupported OOD freeze protocol")
        return cls()

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


class MetricRole(StrEnum):
    """The preregistered reporting role of one native benchmark metric."""

    PRIMARY = "primary"
    SECONDARY = "secondary"


class RewardMetricMapping(StrEnum):
    """Closed maps from a reported native metric to ``TerminalReward.value``."""

    IDENTITY = "reward-equals-source-metric@1"
    DIVIDE_BY_100 = "reward-equals-source-metric-divided-by-100@1"


@dataclass(frozen=True, slots=True)
class BenchmarkMetricSpec:
    """One scalar metric that every episode of a benchmark must report."""

    metric_name: str
    minimum: float
    maximum: float
    role: MetricRole

    def __post_init__(self) -> None:
        _text(self.metric_name, field="benchmark metric name")
        minimum = _finite_float(self.minimum, field="benchmark metric minimum")
        maximum = _finite_float(self.maximum, field="benchmark metric maximum")
        if minimum < 0.0 or maximum < minimum:
            raise ValueError("benchmark metric bounds are invalid")
        if not isinstance(self.role, MetricRole):
            raise TypeError("benchmark metric role must be closed")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "maximum": self.maximum,
            "metric_name": self.metric_name,
            "minimum": self.minimum,
            "role": self.role.value,
        }

    @classmethod
    def from_value(cls, value: object) -> BenchmarkMetricSpec:
        data = _object(
            value,
            fields=frozenset({"maximum", "metric_name", "minimum", "role"}),
            label="benchmark metric specification",
        )
        return cls(
            metric_name=_text(data["metric_name"], field="benchmark metric name"),
            minimum=_finite_float(data["minimum"], field="benchmark metric minimum"),
            maximum=_finite_float(data["maximum"], field="benchmark metric maximum"),
            role=MetricRole(_text(data["role"], field="benchmark metric role")),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkMetricProtocol:
    """Exact, answer-free native reporting contract for one benchmark.

    The protocol names every metric that an episode is required to emit.  It
    closes a concrete failure mode where aggregate rows could omit a metric on
    selected episodes while remaining numerically well-formed.
    """

    benchmark: Benchmark
    metrics: tuple[BenchmarkMetricSpec, ...]
    reward_native_metric_name: str
    reward_source_metric_name: str
    reward_mapping: RewardMetricMapping
    public_report_order: tuple[str, ...]
    interval_rule: str = NATIVE_METRIC_INTERVAL
    format: str = "skillev-benchmark-metric-protocol@1"

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("benchmark metric protocol requires a closed benchmark")
        if not isinstance(self.metrics, tuple) or not self.metrics:
            raise ValueError("benchmark metric protocol requires metrics")
        if any(not isinstance(item, BenchmarkMetricSpec) for item in self.metrics):
            raise TypeError("benchmark metric protocol contains an invalid metric")
        names = tuple(item.metric_name for item in self.metrics)
        if len(set(names)) != len(names):
            raise ValueError("benchmark metric protocol repeats a metric name")
        primary = tuple(
            item.metric_name for item in self.metrics if item.role is MetricRole.PRIMARY
        )
        if len(primary) != 1:
            raise ValueError("benchmark metric protocol requires exactly one primary metric")
        _text(self.reward_native_metric_name, field="reward native metric name")
        _text(self.reward_source_metric_name, field="reward source metric name")
        if self.reward_native_metric_name not in names:
            raise ValueError("reward native metric is not a required benchmark metric")
        if self.reward_source_metric_name not in names:
            raise ValueError("reward source metric is not a required benchmark metric")
        if not isinstance(self.reward_mapping, RewardMetricMapping):
            raise TypeError("reward mapping must be closed")
        if (
            not isinstance(self.public_report_order, tuple)
            or set(self.public_report_order) != set(names)
            or len(self.public_report_order) != len(names)
            or any(type(name) is not str or not name for name in self.public_report_order)
        ):
            raise ValueError("benchmark metric report order must cover each metric exactly once")
        if self.interval_rule != NATIVE_METRIC_INTERVAL:
            raise ValueError("benchmark metric interval rule differs from preregistration")
        if self.format != "skillev-benchmark-metric-protocol@1":
            raise ValueError("unsupported benchmark metric protocol format")

    @property
    def primary_metric_name(self) -> str:
        return next(item.metric_name for item in self.metrics if item.role is MetricRole.PRIMARY)

    @property
    def required_metric_names(self) -> tuple[str, ...]:
        return tuple(sorted(item.metric_name for item in self.metrics))

    def require_episode_metrics(
        self,
        *,
        reward_value: float,
        reward_native_metric_name: str,
        metric_values: dict[str, float],
    ) -> None:
        """Validate the exact per-episode metric set and reward projection."""

        if reward_native_metric_name != self.reward_native_metric_name:
            raise ValueError("episode reward native metric differs from benchmark protocol")
        if tuple(sorted(metric_values)) != self.required_metric_names:
            raise ValueError("episode native metric set differs from benchmark protocol")
        bounds = {item.metric_name: item for item in self.metrics}
        for name, value in metric_values.items():
            normalized = _finite_float(value, field=f"native metric {name}")
            metric = bounds[name]
            if not metric.minimum <= normalized <= metric.maximum:
                raise ValueError("episode native metric lies outside its preregistered range")
        normalized_reward = _finite_float(reward_value, field="terminal reward value")
        source_value = metric_values[self.reward_source_metric_name]
        expected_reward = (
            source_value
            if self.reward_mapping is RewardMetricMapping.IDENTITY
            else source_value / 100.0
        )
        if not math.isclose(normalized_reward, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("terminal reward differs from the benchmark reward mapping")

    def require_aggregate_metrics(
        self,
        *,
        sample_count: int,
        metric_moments: dict[str, tuple[int, float, float]],
    ) -> None:
        """Require complete coverage and range-compatible aggregate moments."""

        if tuple(sorted(metric_moments)) != self.required_metric_names:
            raise ValueError("aggregate native metric set differs from benchmark protocol")
        bounds = {item.metric_name: item for item in self.metrics}
        for name, (count, value_sum, value_squared_sum) in metric_moments.items():
            if count != sample_count:
                raise ValueError("aggregate native metric does not cover every frozen episode")
            metric = bounds[name]
            if not metric.minimum * count <= value_sum <= metric.maximum * count:
                raise ValueError("aggregate native metric sum lies outside its preregistered range")
            if not 0.0 <= value_squared_sum <= metric.maximum * metric.maximum * count:
                raise ValueError("aggregate native metric squared sum lies outside its range")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "benchmark": self.benchmark.value,
            "format": self.format,
            "interval_rule": self.interval_rule,
            "metrics": [item.to_value() for item in self.metrics],
            "public_report_order": list(self.public_report_order),
            "reward_mapping": self.reward_mapping.value,
            "reward_native_metric_name": self.reward_native_metric_name,
            "reward_source_metric_name": self.reward_source_metric_name,
        }

    @classmethod
    def from_value(cls, value: object) -> BenchmarkMetricProtocol:
        data = _object(
            value,
            fields=frozenset(
                {
                    "benchmark",
                    "format",
                    "interval_rule",
                    "metrics",
                    "public_report_order",
                    "reward_mapping",
                    "reward_native_metric_name",
                    "reward_source_metric_name",
                }
            ),
            label="benchmark metric protocol",
        )
        metrics = data["metrics"]
        report_order = data["public_report_order"]
        if not isinstance(metrics, list) or not isinstance(report_order, list):
            raise ValueError("benchmark metric protocol rows must be arrays")
        text_fields = (
            "benchmark",
            "format",
            "interval_rule",
            "reward_mapping",
            "reward_native_metric_name",
            "reward_source_metric_name",
        )
        if any(type(data[field]) is not str for field in text_fields):
            raise TypeError("benchmark metric protocol text fields must be text")
        if any(type(name) is not str for name in report_order):
            raise TypeError("benchmark metric report order must contain text")
        return cls(
            benchmark=Benchmark(data["benchmark"]),
            metrics=tuple(BenchmarkMetricSpec.from_value(item) for item in metrics),
            reward_native_metric_name=data["reward_native_metric_name"],
            reward_source_metric_name=data["reward_source_metric_name"],
            reward_mapping=RewardMetricMapping(data["reward_mapping"]),
            public_report_order=tuple(report_order),
            interval_rule=data["interval_rule"],
            format=data["format"],
        )


def _metric(
    metric_name: str,
    *,
    role: MetricRole,
    maximum: float = 1.0,
) -> BenchmarkMetricSpec:
    return BenchmarkMetricSpec(
        metric_name=metric_name,
        minimum=0.0,
        maximum=maximum,
        role=role,
    )


def _metrics(
    benchmark: Benchmark,
    *,
    primary: str,
    secondary: tuple[str, ...] = (),
    reward_native_metric_name: str | None = None,
    reward_source_metric_name: str | None = None,
    reward_mapping: RewardMetricMapping = RewardMetricMapping.IDENTITY,
    maxima: dict[str, float] | None = None,
) -> BenchmarkMetricProtocol:
    """Build one literal metric protocol while keeping its report order explicit."""

    maximum_by_name = {} if maxima is None else maxima
    names = (primary, *secondary)
    return BenchmarkMetricProtocol(
        benchmark=benchmark,
        metrics=tuple(
            _metric(
                name,
                role=MetricRole.PRIMARY if name == primary else MetricRole.SECONDARY,
                maximum=maximum_by_name.get(name, 1.0),
            )
            for name in names
        ),
        reward_native_metric_name=reward_native_metric_name or primary,
        reward_source_metric_name=reward_source_metric_name or primary,
        reward_mapping=reward_mapping,
        public_report_order=names,
    )


BENCHMARK_METRIC_PROTOCOLS: Final[tuple[BenchmarkMetricProtocol, ...]] = (
    _metrics(
        Benchmark.HOTPOT_QA,
        primary="token-f1",
        secondary=("exact-match",),
    ),
    _metrics(
        Benchmark.TRIVIA_QA,
        primary="token-f1",
        secondary=("exact-match",),
    ),
    _metrics(Benchmark.AIME_2026, primary="accuracy"),
    _metrics(Benchmark.MED_QA, primary="accuracy"),
    _metrics(
        Benchmark.WEBSHOP,
        primary="webshop-score",
        secondary=("webshop-success",),
    ),
    _metrics(Benchmark.ALFWORLD, primary="alfworld-success"),
    _metrics(
        Benchmark.BIRD_SQL,
        primary="execution-accuracy",
        secondary=("valid-efficiency-score",),
    ),
    _metrics(
        Benchmark.APPWORLD,
        primary="scenario-goal-completion",
        secondary=("task-goal-completion",),
    ),
    _metrics(Benchmark.MBPP_PLUS, primary="pass@1"),
    _metrics(
        Benchmark.MUSIQUE,
        primary="token-f1",
        secondary=("exact-match",),
    ),
    _metrics(Benchmark.NQ_OPEN, primary="exact-match"),
    _metrics(Benchmark.MATH_HARD, primary="accuracy"),
    _metrics(Benchmark.GPQA_DIAMOND, primary="accuracy"),
    _metrics(
        Benchmark.MIND2WEB,
        primary="mind2web-step-success",
        secondary=(
            "mind2web-action-f1",
            "mind2web-element-accuracy",
            "mind2web-operation-accuracy",
            "mind2web-value-f1",
        ),
    ),
    _metrics(
        Benchmark.SCIENCE_WORLD,
        primary="scienceworld-native-score",
        secondary=("scienceworld-score",),
        reward_native_metric_name="scienceworld-score",
        reward_source_metric_name="scienceworld-native-score",
        reward_mapping=RewardMetricMapping.DIVIDE_BY_100,
        maxima={"scienceworld-native-score": 100.0},
    ),
    _metrics(Benchmark.TABLEBENCH, primary="accuracy"),
    _metrics(Benchmark.BFCL_V3, primary="function-call-accuracy"),
    _metrics(Benchmark.HUMANEVAL_PLUS, primary="pass@1"),
)


@dataclass(frozen=True, slots=True)
class EvaluationReportingProtocol:
    """The answer-free aggregate estimators fixed before any evaluation runs.

    The suite has one preregistered primary seed, so these are per-sample
    intervals, not an assertion of cross-seed robustness.  Native scalar
    scores use aggregate moments; binary success and status rates use Wilson
    intervals.  Keeping this object in the protocol makes a later reporting
    script unable to switch estimators after observing a result.
    """

    format: ClassVar[str] = "skillev-evaluation-reporting@2"
    native_metric_interval: ClassVar[str] = NATIVE_METRIC_INTERVAL
    binary_rate_interval: ClassVar[str] = BINARY_RATE_INTERVAL
    aggregate_projection: ClassVar[str] = "answer-free-native-scalar-moments@1"
    benchmark_metric_protocols: ClassVar[tuple[BenchmarkMetricProtocol, ...]] = (
        BENCHMARK_METRIC_PROTOCOLS
    )

    def __post_init__(self) -> None:
        benchmarks = tuple(item.benchmark for item in self.benchmark_metric_protocols)
        if benchmarks != ACTIVE_BENCHMARKS:
            raise ValueError("evaluation reporting must cover every benchmark in fixed order")

    def metric_protocol(self, benchmark: Benchmark) -> BenchmarkMetricProtocol:
        if not isinstance(benchmark, Benchmark):
            raise TypeError("metric lookup requires a closed benchmark")
        return next(item for item in self.benchmark_metric_protocols if item.benchmark is benchmark)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "aggregate_projection": self.aggregate_projection,
            "benchmark_metric_protocols": [
                item.to_value() for item in self.benchmark_metric_protocols
            ],
            "binary_rate_interval": self.binary_rate_interval,
            "format": self.format,
            "native_metric_interval": self.native_metric_interval,
        }

    @classmethod
    def from_value(cls, value: object) -> EvaluationReportingProtocol:
        data = _object(
            value,
            fields=frozenset(
                {
                    "aggregate_projection",
                    "benchmark_metric_protocols",
                    "binary_rate_interval",
                    "format",
                    "native_metric_interval",
                }
            ),
            label="evaluation reporting protocol",
        )
        metric_rows = data["benchmark_metric_protocols"]
        if not isinstance(metric_rows, list):
            raise ValueError("evaluation reporting metric protocols must be an array")
        parsed = tuple(BenchmarkMetricProtocol.from_value(item) for item in metric_rows)
        if parsed != cls().benchmark_metric_protocols:
            raise ValueError("evaluation reporting metric protocols differ from preregistration")
        if data != cls().to_value():
            raise ValueError("evaluation reporting protocol differs from preregistration")
        return cls()

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class DatasetSnapshotIdentity:
    """Non-sensitive source identity, not benchmark data."""

    name: str
    version: str
    snapshot_hash: str

    def __post_init__(self) -> None:
        _text(self.name, field="name")
        _text(self.version, field="version")
        _sha256(self.snapshot_hash, field="hash")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "hash": self.snapshot_hash,
            "name": self.name,
            "version": self.version,
        }

    @classmethod
    def from_value(cls, value: object) -> DatasetSnapshotIdentity:
        data = _object(
            value,
            fields=frozenset({"hash", "name", "version"}),
            label="dataset snapshot identity",
        )
        return cls(
            name=_text(data["name"], field="name"),
            version=_text(data["version"], field="version"),
            snapshot_hash=_sha256(data["hash"], field="hash"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class ExperimentProtocol:
    """Complete v5 protocol whose hash is frozen before result ingestion."""

    seed: int
    benchmarks: tuple[BenchmarkSpec, ...]
    ablations: tuple[ArmProtocol, ...]
    full_method_identity: FullMethodIdentity
    ood_freeze: FrozenSinglePassOOD
    dataset_snapshots: tuple[DatasetSnapshotIdentity, ...]
    schedule_identities: tuple[FrozenTaskSequenceIdentity, ...]
    formal_execution: FormalExecutionFreeze
    evaluation_reporting: EvaluationReportingProtocol = EvaluationReportingProtocol()
    format: str = PROTOCOL_FORMAT

    def __post_init__(self) -> None:
        if self.format != PROTOCOL_FORMAT:
            raise ValueError("experiment protocol has an incompatible format")
        if _uint64(self.seed, field="seed") != FIXED_SEED:
            raise ValueError("protocol must use the preregistered primary seed")
        if self.benchmarks != BENCHMARK_SPECS:
            raise ValueError("benchmark suite differs from the fixed active specs")
        if self.ablations != ABLATION_PROTOCOLS:
            raise ValueError("ablation suite differs from the fixed seven arms")
        if self.full_method_identity != FullMethodIdentity():
            raise ValueError("protocol full-method identity is not literal v3 full")
        if self.ood_freeze != FrozenSinglePassOOD():
            raise ValueError("OOD freeze contract cannot be weakened")
        if not isinstance(self.dataset_snapshots, tuple) or any(
            not isinstance(item, DatasetSnapshotIdentity) for item in self.dataset_snapshots
        ):
            raise ValueError("dataset_snapshots must contain snapshot identities")
        expected_names = tuple(spec.dataset_snapshot_name for spec in self.benchmarks)
        if tuple(item.name for item in self.dataset_snapshots) != expected_names:
            raise ValueError("dataset snapshots must align with benchmark order")
        _validate_schedule_identities(self.schedule_identities)
        from .formal_execution import FormalExecutionFreeze

        if not isinstance(self.formal_execution, FormalExecutionFreeze):
            raise TypeError("protocol formal_execution must be FormalExecutionFreeze")
        if self.evaluation_reporting != EvaluationReportingProtocol():
            raise ValueError("protocol evaluation reporting differs from preregistration")
        if self.formal_execution.seed != self.seed:
            raise ValueError("formal execution seed differs from protocol seed")
        if (
            self.formal_execution.sampling_schedule_algorithm != SCIENTIFIC_SAMPLING_ALGORITHM
            or self.formal_execution.sampling_schedule_hash
            != scientific_sampling_schedule_hash(base_seed=self.seed)
        ):
            raise ValueError("formal execution sampling schedule differs from protocol")
        training_schedule = next(
            item
            for item in self.schedule_identities
            if item.purpose is SchedulePurpose.IID_TRAINING
        )
        if self.formal_execution.training_sequence != training_schedule:
            raise ValueError("formal execution training sequence differs from protocol schedule")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "ablations": [item.to_value() for item in self.ablations],
            "benchmarks": [item.to_value() for item in self.benchmarks],
            "dataset_snapshots": [item.to_value() for item in self.dataset_snapshots],
            "evaluation_reporting": self.evaluation_reporting.to_value(),
            "format": self.format,
            "full_method_identity": self.full_method_identity.to_value(),
            "formal_execution": self.formal_execution.to_value(),
            "ood_freeze": self.ood_freeze.to_value(),
            "schedule_identities": [item.to_value() for item in self.schedule_identities],
            "sampling_schedule_algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
            "sampling_schedule_hash": scientific_sampling_schedule_hash(base_seed=self.seed),
            "seed": self.seed,
        }

    @classmethod
    def from_value(cls, value: object) -> ExperimentProtocol:
        data = _object(
            value,
            fields=frozenset(
                {
                    "ablations",
                    "benchmarks",
                    "dataset_snapshots",
                    "evaluation_reporting",
                    "formal_execution",
                    "format",
                    "full_method_identity",
                    "ood_freeze",
                    "schedule_identities",
                    "sampling_schedule_algorithm",
                    "sampling_schedule_hash",
                    "seed",
                }
            ),
            label="experiment protocol",
        )
        if data["format"] != PROTOCOL_FORMAT:
            raise ValueError("unsupported experiment protocol format")
        if data["sampling_schedule_algorithm"] != SCIENTIFIC_SAMPLING_ALGORITHM:
            raise ValueError("protocol sampling algorithm differs from preregistration")
        if data["sampling_schedule_hash"] != scientific_sampling_schedule_hash(
            base_seed=_uint64(data["seed"], field="seed")
        ):
            raise ValueError("protocol sampling schedule hash differs from its seed")
        raw_benchmarks = data["benchmarks"]
        raw_ablations = data["ablations"]
        raw_snapshots = data["dataset_snapshots"]
        raw_schedules = data["schedule_identities"]
        if not isinstance(raw_benchmarks, list):
            raise ValueError("protocol benchmarks must be an array")
        if not isinstance(raw_ablations, list):
            raise ValueError("protocol ablations must be an array")
        if not isinstance(raw_snapshots, list) or not isinstance(raw_schedules, list):
            raise ValueError("protocol collections must be arrays")
        return cls(
            seed=_uint64(data["seed"], field="seed"),
            benchmarks=tuple(BenchmarkSpec.from_value(item) for item in raw_benchmarks),
            ablations=tuple(arm_protocol_from_value(item) for item in raw_ablations),
            full_method_identity=FullMethodIdentity.from_value(data["full_method_identity"]),
            ood_freeze=FrozenSinglePassOOD.from_value(data["ood_freeze"]),
            dataset_snapshots=tuple(
                DatasetSnapshotIdentity.from_value(item) for item in raw_snapshots
            ),
            schedule_identities=tuple(
                FrozenTaskSequenceIdentity.from_value(item) for item in raw_schedules
            ),
            formal_execution=_formal_execution_from_value(data["formal_execution"]),
            evaluation_reporting=EvaluationReportingProtocol.from_value(
                data["evaluation_reporting"]
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class ProtocolFreeze:
    protocol_hash: str
    freeze_id: str
    format: str = FREEZE_FORMAT

    def __post_init__(self) -> None:
        if self.format != FREEZE_FORMAT:
            raise ValueError("protocol freeze has an incompatible format")
        _sha256(self.protocol_hash, field="protocol_hash")
        expected = stable_hash(
            {
                "format": self.format,
                "protocol_hash": self.protocol_hash,
                "result_ingestion_count": 0,
            }
        )
        if self.freeze_id != expected:
            raise ValueError("freeze_id does not match the result-blind protocol")

    @classmethod
    def create(cls, protocol: ExperimentProtocol) -> ProtocolFreeze:
        if not isinstance(protocol, ExperimentProtocol):
            raise TypeError("protocol must be ExperimentProtocol")
        identity: dict[str, JsonValue] = {
            "format": FREEZE_FORMAT,
            "protocol_hash": protocol.content_hash,
            "result_ingestion_count": 0,
        }
        return cls(
            protocol_hash=protocol.content_hash,
            freeze_id=stable_hash(identity),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "freeze_id": self.freeze_id,
            "protocol_hash": self.protocol_hash,
            "result_ingestion_count": 0,
        }

    @classmethod
    def from_value(cls, value: object) -> ProtocolFreeze:
        data = _object(
            value,
            fields=frozenset(
                {
                    "format",
                    "freeze_id",
                    "protocol_hash",
                    "result_ingestion_count",
                }
            ),
            label="protocol freeze",
        )
        if data["format"] != FREEZE_FORMAT or data["result_ingestion_count"] != 0:
            raise ValueError("invalid result-blind protocol freeze")
        return cls(
            protocol_hash=_sha256(data["protocol_hash"], field="protocol_hash"),
            freeze_id=_sha256(data["freeze_id"], field="freeze_id"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


class ProtocolNotFrozenError(RuntimeError):
    """Raised when result ingestion precedes protocol freezing."""


class ProtocolFrozenError(RuntimeError):
    """Raised when a frozen result-blind protocol is changed."""


class ResultProtocolMismatchError(RuntimeError):
    """Raised when result metadata declares another protocol."""


class ResultBlindProtocolGate:
    """Freeze lifecycle guard; it never decides benchmark membership."""

    def __init__(
        self,
        protocol: ExperimentProtocol,
        *,
        freeze: ProtocolFreeze | None = None,
    ) -> None:
        if not isinstance(protocol, ExperimentProtocol):
            raise TypeError("protocol must be ExperimentProtocol")
        if freeze is not None and (
            not isinstance(freeze, ProtocolFreeze) or freeze.protocol_hash != protocol.content_hash
        ):
            raise ValueError("freeze belongs to another protocol")
        self._protocol = protocol
        self._freeze = freeze
        self._result_ingestion_count = 0

    @property
    def protocol(self) -> ExperimentProtocol:
        return self._protocol

    @property
    def freeze_record(self) -> ProtocolFreeze | None:
        return self._freeze

    @property
    def result_ingestion_count(self) -> int:
        return self._result_ingestion_count

    def replace_protocol(self, protocol: ExperimentProtocol) -> None:
        if self._freeze is not None:
            raise ProtocolFrozenError("frozen protocol cannot be changed")
        if not isinstance(protocol, ExperimentProtocol):
            raise TypeError("protocol must be ExperimentProtocol")
        self._protocol = protocol

    def freeze(self) -> ProtocolFreeze:
        if self._freeze is None:
            self._freeze = ProtocolFreeze.create(self._protocol)
        return self._freeze

    def admit_result(self, *, protocol_hash: str) -> None:
        if self._freeze is None:
            raise ProtocolNotFrozenError("freeze the protocol before ingesting result metadata")
        if protocol_hash != self._freeze.protocol_hash:
            raise ResultProtocolMismatchError("result was produced under another protocol")
        self._result_ingestion_count += 1


def create_protocol(
    dataset_snapshots: tuple[DatasetSnapshotIdentity, ...],
    *,
    schedule_identities: tuple[FrozenTaskSequenceIdentity, ...],
    formal_execution: FormalExecutionFreeze,
) -> ExperimentProtocol:
    """Create the sole v5 protocol from result-blind identities only."""

    return ExperimentProtocol(
        seed=FIXED_SEED,
        benchmarks=BENCHMARK_SPECS,
        ablations=ABLATION_PROTOCOLS,
        full_method_identity=FullMethodIdentity(),
        ood_freeze=FrozenSinglePassOOD(),
        dataset_snapshots=dataset_snapshots,
        schedule_identities=schedule_identities,
        formal_execution=formal_execution,
        evaluation_reporting=EvaluationReportingProtocol(),
    )


def _formal_execution_from_value(value: object) -> FormalExecutionFreeze:
    """Delay the import that would otherwise make protocol/formal types cyclic."""

    from .formal_execution import FormalExecutionFreeze

    return FormalExecutionFreeze.from_value(value)


__all__ = [
    "ABLATION_PROTOCOLS",
    "BENCHMARK_METRIC_PROTOCOLS",
    "BENCHMARK_SPECS",
    "EVALUATION_POPULATION_THRESHOLD",
    "FIXED_500_EVALUATION",
    "FIXED_SEED",
    "FIXED_SUBSAMPLE_ALGORITHM",
    "FREEZE_FORMAT",
    "FULL_EVALUATION",
    "METHOD_IDENTITY_FORMAT",
    "NATIVE_METRIC_INTERVAL",
    "POPULATION_THRESHOLD_EVALUATION",
    "PROTOCOL_FORMAT",
    "SCHEDULE_PURPOSES",
    "AblationArm",
    "AggregatePurpose",
    "ArmProtocol",
    "Benchmark",
    "BenchmarkMetricProtocol",
    "BenchmarkMetricSpec",
    "BenchmarkRole",
    "BenchmarkSpec",
    "BenchmarkTaskCount",
    "CappedFlowWeightArmProtocol",
    "ClippedImportanceArmProtocol",
    "DatasetSnapshotIdentity",
    "EvaluationReportingProtocol",
    "EvaluationSampling",
    "ExperimentProtocol",
    "FixedSubsampleEvaluation",
    "FrozenSinglePassOOD",
    "FrozenTaskSequenceIdentity",
    "FullArmProtocol",
    "FullEvaluationSampling",
    "FullMethodIdentity",
    "MetricRole",
    "NoBayesianCalibrationArmProtocol",
    "NoFlowWeightingArmProtocol",
    "PopulationThresholdEvaluation",
    "PosteriorMeanArmProtocol",
    "ProtocolFreeze",
    "ProtocolFrozenError",
    "ProtocolNotFrozenError",
    "ResidualOnlyPhaseArmProtocol",
    "ResultBlindProtocolGate",
    "ResultProtocolMismatchError",
    "RewardMetricMapping",
    "SchedulePurpose",
    "TrainingUse",
    "arm_protocol_from_value",
    "create_protocol",
    "evaluation_sampling_from_value",
    "fixed_subsample_rank",
]
