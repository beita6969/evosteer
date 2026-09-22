"""Pure Protocol-v3 flow diagnostics over materialized TTB contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from skillev.contracts import (
    JsonValue,
)

from .assembly import BatchFlowInput as BatchFlowInput
from .assembly import TrajectoryFlowInput as TrajectoryFlowInput
from .assembly import assemble_batch_flow_input as assemble_batch_flow_input

DIAGNOSTICS_FORMAT = "skillev-diagnostics@4"


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """Full-method diagnostic controls.

    Importance clipping is deliberately absent from the full method.  The
    clipped estimator is an explicit experiment arm.
    """

    window_size: int = 50
    stagnation_rho: float = 0.05
    format: str = DIAGNOSTICS_FORMAT

    def __post_init__(self) -> None:
        if self.format != DIAGNOSTICS_FORMAT:
            raise ValueError(f"DiagnosticsConfig.format must be {DIAGNOSTICS_FORMAT!r}")
        if type(self.window_size) is not int or self.window_size < 1:
            raise ValueError("DiagnosticsConfig.window_size must be a positive integer")
        rho = _finite_float(
            self.stagnation_rho,
            field="DiagnosticsConfig.stagnation_rho",
        )
        if not 0.0 < rho < 1.0:
            raise ValueError("DiagnosticsConfig.stagnation_rho must lie in (0, 1)")
        object.__setattr__(self, "stagnation_rho", rho)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "stagnation_rho": self.stagnation_rho,
            "window_size": self.window_size,
        }

    @classmethod
    def from_value(cls, value: object) -> DiagnosticsConfig:
        if not isinstance(value, Mapping) or set(value) != {
            "format",
            "stagnation_rho",
            "window_size",
        }:
            raise ValueError("DiagnosticsConfig has incompatible fields")
        if value["format"] != DIAGNOSTICS_FORMAT:
            raise ValueError("DiagnosticsConfig has an incompatible format")
        window_size = value["window_size"]
        rho = value["stagnation_rho"]
        if type(window_size) is not int:
            raise ValueError("DiagnosticsConfig.window_size must be an integer")
        if isinstance(rho, bool) or not isinstance(rho, int | float):
            raise ValueError("DiagnosticsConfig.stagnation_rho must be numeric")
        return cls(window_size=window_size, stagnation_rho=float(rho))


@dataclass(frozen=True, slots=True)
class EdgeFlowDiagnostic:
    trajectory_id: str
    step_index: int
    log_importance: float
    sample_log_state_weight: float
    invoked_skill_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        location = f"edge {self.trajectory_id}:{self.step_index}"
        _finite_float(self.log_importance, field=f"{location} log_importance")
        _finite_float(self.sample_log_state_weight, field=f"{location} sample_log_state_weight")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "invoked_skill_ids": list(self.invoked_skill_ids),
            "log_importance": self.log_importance,
            "sample_log_state_weight": self.sample_log_state_weight,
            "step_index": self.step_index,
            "trajectory_id": self.trajectory_id,
        }

    @classmethod
    def from_value(cls, value: object) -> EdgeFlowDiagnostic:
        expected = {
            "invoked_skill_ids",
            "log_importance",
            "sample_log_state_weight",
            "step_index",
            "trajectory_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("EdgeFlowDiagnostic has incompatible fields")
        trajectory_id = value["trajectory_id"]
        step_index = value["step_index"]
        raw_skills = value["invoked_skill_ids"]
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("edge trajectory_id must be non-empty text")
        if type(step_index) is not int or step_index < 1:
            raise ValueError("edge step_index must be positive")
        if not isinstance(raw_skills, list) or any(
            not isinstance(item, str) or not item for item in raw_skills
        ):
            raise ValueError("edge invoked_skill_ids must be an array of text")
        return cls(
            trajectory_id=trajectory_id,
            step_index=step_index,
            log_importance=_finite_float(value["log_importance"], field="log_importance"),
            sample_log_state_weight=_finite_float(
                value["sample_log_state_weight"],
                field="sample_log_state_weight",
            ),
            invoked_skill_ids=tuple(raw_skills),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryFlowDiagnostic:
    trajectory_id: str
    horizon: int
    edges: tuple[EdgeFlowDiagnostic, ...]
    terminal_sample_log_state_weight: float

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "edges": [edge.to_value() for edge in self.edges],
            "horizon": self.horizon,
            "terminal_sample_log_state_weight": self.terminal_sample_log_state_weight,
            "trajectory_id": self.trajectory_id,
        }

    @classmethod
    def from_value(cls, value: object) -> TrajectoryFlowDiagnostic:
        expected = {
            "edges",
            "horizon",
            "terminal_sample_log_state_weight",
            "trajectory_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("TrajectoryFlowDiagnostic has incompatible fields")
        trajectory_id = value["trajectory_id"]
        horizon = value["horizon"]
        raw_edges = value["edges"]
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("trajectory_id must be non-empty text")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("trajectory horizon must be positive")
        if not isinstance(raw_edges, list):
            raise TypeError("trajectory edges must be an array")
        edges = tuple(EdgeFlowDiagnostic.from_value(item) for item in raw_edges)
        if len(edges) != horizon:
            raise ValueError("trajectory edge count differs from horizon")
        return cls(
            trajectory_id=trajectory_id,
            horizon=horizon,
            edges=edges,
            terminal_sample_log_state_weight=_finite_float(
                value["terminal_sample_log_state_weight"],
                field="terminal_sample_log_state_weight",
            ),
        )


def trajectory_flow(
    trajectory: TrajectoryFlowInput,
) -> TrajectoryFlowDiagnostic:
    """Materialize cumulative log state weight on one observed history path.

    ``sample_log_state_weight`` is a single-path prefix sample, not a
    conditionally aggregated estimate of F(H_t). Skill marginal flow pools
    these samples over invoking trajectories; it is not state conditioning.
    """

    running = 0.0
    output: list[EdgeFlowDiagnostic] = []
    for edge, invoked_skill_ids in zip(
        trajectory.edges,
        trajectory.invoked_skill_ids_by_step,
        strict=True,
    ):
        running += edge.step_importance
        output.append(
            EdgeFlowDiagnostic(
                trajectory_id=trajectory.trajectory_id,
                step_index=edge.step_index,
                log_importance=edge.step_importance,
                sample_log_state_weight=running,
                invoked_skill_ids=invoked_skill_ids,
            )
        )
    return TrajectoryFlowDiagnostic(
        trajectory_id=trajectory.trajectory_id,
        horizon=trajectory.horizon,
        edges=tuple(output),
        terminal_sample_log_state_weight=running,
    )


@dataclass(frozen=True, slots=True)
class SkillFlowStat:
    skill_id: str
    invoking_trajectory_count: int
    invoking_edge_count: int
    log_skill_flow: float

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "invoking_edge_count": self.invoking_edge_count,
            "invoking_trajectory_count": self.invoking_trajectory_count,
            "log_skill_flow": self.log_skill_flow,
            "skill_id": self.skill_id,
        }

    @classmethod
    def from_value(cls, value: object) -> SkillFlowStat:
        expected = {
            "invoking_edge_count",
            "invoking_trajectory_count",
            "log_skill_flow",
            "skill_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("SkillFlowStat has incompatible fields")
        skill_id = value["skill_id"]
        trajectory_count = value["invoking_trajectory_count"]
        edge_count = value["invoking_edge_count"]
        if not isinstance(skill_id, str) or not skill_id:
            raise ValueError("skill_id must be non-empty text")
        if type(trajectory_count) is not int or trajectory_count < 1:
            raise ValueError("invoking_trajectory_count must be positive")
        if type(edge_count) is not int or edge_count < trajectory_count:
            raise ValueError("invoking_edge_count is inconsistent")
        return cls(
            skill_id=skill_id,
            invoking_trajectory_count=trajectory_count,
            invoking_edge_count=edge_count,
            log_skill_flow=_finite_float(value["log_skill_flow"], field="log_skill_flow"),
        )


def _logsumexp(values: tuple[float, ...]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(item - maximum) for item in values))


def skill_marginal_flows(
    trajectories: tuple[TrajectoryFlowDiagnostic, ...],
) -> tuple[SkillFlowStat, ...]:
    """Aggregate state weights over invoking edges as in Equation (11)."""

    calls_by_skill: dict[str, list[tuple[str, int, float]]] = {}
    for trajectory in trajectories:
        for edge in trajectory.edges:
            for skill_id in sorted(edge.invoked_skill_ids):
                calls_by_skill.setdefault(skill_id, []).append(
                    (
                        trajectory.trajectory_id,
                        edge.step_index,
                        edge.sample_log_state_weight,
                    )
                )
    output: list[SkillFlowStat] = []
    for skill_id in sorted(calls_by_skill):
        calls = sorted(calls_by_skill[skill_id], key=lambda item: (item[0], item[1]))
        invoking_trajectories = {trajectory_id for trajectory_id, _, _ in calls}
        output.append(
            SkillFlowStat(
                skill_id=skill_id,
                invoking_trajectory_count=len(invoking_trajectories),
                invoking_edge_count=len(calls),
                log_skill_flow=(
                    _logsumexp(tuple(value for _, _, value in calls))
                    - math.log(len(invoking_trajectories))
                ),
            )
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class InsufficientResidualWindow:
    observed_batch_count: int
    required_batch_count: int
    kind: ClassVar[str] = "insufficient-window"

    def __post_init__(self) -> None:
        if type(self.observed_batch_count) is not int or self.observed_batch_count < 0:
            raise ValueError("observed_batch_count must be non-negative")
        if type(self.required_batch_count) is not int or self.required_batch_count < 2:
            raise ValueError("required_batch_count must be at least two")
        if self.observed_batch_count >= self.required_batch_count:
            raise ValueError("insufficient-window evidence already has enough batches")

    @property
    def residual_condition_met(self) -> bool:
        return False

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "observed_batch_count": self.observed_batch_count,
            "required_batch_count": self.required_batch_count,
        }

    @classmethod
    def from_value(cls, value: object) -> InsufficientResidualWindow:
        if not isinstance(value, Mapping) or set(value) != {
            "kind",
            "observed_batch_count",
            "required_batch_count",
        }:
            raise ValueError("InsufficientResidualWindow has incompatible fields")
        if value["kind"] != cls.kind:
            raise ValueError("wrong residual-window kind")
        observed = value["observed_batch_count"]
        required = value["required_batch_count"]
        if type(observed) is not int or type(required) is not int:
            raise TypeError("residual-window counts must be integers")
        return cls(observed_batch_count=observed, required_batch_count=required)


@dataclass(frozen=True, slots=True)
class ZeroPreviousResidual:
    library_version: str
    window_size: int
    previous_window_mean_delta_squared: float
    current_window_mean_delta_squared: float
    kind: ClassVar[str] = "zero-previous-residual"

    def __post_init__(self) -> None:
        if type(self.library_version) is not str or not self.library_version.strip():
            raise ValueError("library_version must be non-empty text")
        if type(self.window_size) is not int or self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.previous_window_mean_delta_squared != 0.0:
            raise ValueError("zero-previous-residual requires an exact zero previous mean")
        current = _finite_float(
            self.current_window_mean_delta_squared,
            field="current_window_mean_delta_squared",
        )
        if current < 0.0:
            raise ValueError("current window residual cannot be negative")

    @property
    def residual_condition_met(self) -> bool:
        return False

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "current_window_mean_delta_squared": self.current_window_mean_delta_squared,
            "kind": self.kind,
            "library_version": self.library_version,
            "previous_window_mean_delta_squared": self.previous_window_mean_delta_squared,
            "window_size": self.window_size,
        }

    @classmethod
    def from_value(cls, value: object) -> ZeroPreviousResidual:
        expected = {
            "current_window_mean_delta_squared",
            "kind",
            "library_version",
            "previous_window_mean_delta_squared",
            "window_size",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("ZeroPreviousResidual has incompatible fields")
        if value["kind"] != cls.kind:
            raise ValueError("wrong residual-window kind")
        library_version = value["library_version"]
        window_size = value["window_size"]
        if type(library_version) is not str or type(window_size) is not int:
            raise TypeError("zero residual identity has incompatible types")
        return cls(
            library_version=library_version,
            window_size=window_size,
            previous_window_mean_delta_squared=_finite_float(
                value["previous_window_mean_delta_squared"],
                field="previous_window_mean_delta_squared",
            ),
            current_window_mean_delta_squared=_finite_float(
                value["current_window_mean_delta_squared"],
                field="current_window_mean_delta_squared",
            ),
        )


@dataclass(frozen=True, slots=True)
class ComparableResidualWindows:
    library_version: str
    window_size: int
    previous_window_mean_delta_squared: float
    current_window_mean_delta_squared: float
    relative_improvement: float
    rho: float
    stagnant: bool
    kind: ClassVar[str] = "comparable-windows"

    def __post_init__(self) -> None:
        if type(self.library_version) is not str or not self.library_version.strip():
            raise ValueError("library_version must be non-empty text")
        if type(self.window_size) is not int or self.window_size < 1:
            raise ValueError("window_size must be positive")
        previous = _finite_float(
            self.previous_window_mean_delta_squared,
            field="previous_window_mean_delta_squared",
        )
        current = _finite_float(
            self.current_window_mean_delta_squared,
            field="current_window_mean_delta_squared",
        )
        rho = _finite_float(self.rho, field="rho")
        improvement = _finite_float(self.relative_improvement, field="relative_improvement")
        if previous <= 0.0 or current < 0.0:
            raise ValueError("comparable residual windows require previous>0 and current>=0")
        if not 0.0 < rho < 1.0:
            raise ValueError("rho must lie in (0, 1)")
        expected = (previous - current) / previous
        if not math.isclose(improvement, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("relative improvement differs from the two windows")
        if type(self.stagnant) is not bool or self.stagnant is not (expected < rho):
            raise ValueError("stagnant flag differs from the strict threshold")

    @property
    def residual_condition_met(self) -> bool:
        return self.stagnant

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "current_window_mean_delta_squared": self.current_window_mean_delta_squared,
            "kind": self.kind,
            "library_version": self.library_version,
            "previous_window_mean_delta_squared": self.previous_window_mean_delta_squared,
            "relative_improvement": self.relative_improvement,
            "rho": self.rho,
            "stagnant": self.stagnant,
            "window_size": self.window_size,
        }

    @classmethod
    def from_value(cls, value: object) -> ComparableResidualWindows:
        expected = {
            "current_window_mean_delta_squared",
            "kind",
            "library_version",
            "previous_window_mean_delta_squared",
            "relative_improvement",
            "rho",
            "stagnant",
            "window_size",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("ComparableResidualWindows has incompatible fields")
        if value["kind"] != cls.kind:
            raise ValueError("wrong residual-window kind")
        library_version = value["library_version"]
        window_size = value["window_size"]
        stagnant = value["stagnant"]
        if type(library_version) is not str or type(window_size) is not int:
            raise TypeError("comparable residual identity has incompatible types")
        if type(stagnant) is not bool:
            raise TypeError("stagnant must be boolean")
        return cls(
            library_version=library_version,
            window_size=window_size,
            previous_window_mean_delta_squared=_finite_float(
                value["previous_window_mean_delta_squared"],
                field="previous_window_mean_delta_squared",
            ),
            current_window_mean_delta_squared=_finite_float(
                value["current_window_mean_delta_squared"],
                field="current_window_mean_delta_squared",
            ),
            relative_improvement=_finite_float(
                value["relative_improvement"],
                field="relative_improvement",
            ),
            rho=_finite_float(value["rho"], field="rho"),
            stagnant=stagnant,
        )


ResidualWindowEvidence: TypeAlias = (
    InsufficientResidualWindow | ZeroPreviousResidual | ComparableResidualWindows
)


def residual_window_evidence_from_value(value: object) -> ResidualWindowEvidence:
    if not isinstance(value, Mapping):
        raise ValueError("residual window evidence must be an object")
    kind = value.get("kind")
    if kind == InsufficientResidualWindow.kind:
        return InsufficientResidualWindow.from_value(value)
    if kind == ZeroPreviousResidual.kind:
        return ZeroPreviousResidual.from_value(value)
    if kind == ComparableResidualWindows.kind:
        return ComparableResidualWindows.from_value(value)
    raise ValueError("unsupported residual-window evidence kind")


class LibrarySegmentMismatchError(RuntimeError):
    """A diagnostic source crossed a library boundary without an explicit reset."""


@dataclass(frozen=True, slots=True)
class ResidualBatchEvidence:
    delta_squared: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.delta_squared, tuple) or not self.delta_squared:
            raise ValueError("residual batch evidence must be a non-empty tuple")
        for value in self.delta_squared:
            if _finite_float(value, field="delta_squared") < 0.0:
                raise ValueError("delta_squared must be non-negative")

    def to_value(self) -> list[JsonValue]:
        return list(self.delta_squared)

    @classmethod
    def from_value(cls, value: object) -> ResidualBatchEvidence:
        if not isinstance(value, list):
            raise TypeError("residual batch evidence must be an array")
        return cls(tuple(_finite_float(item, field="delta_squared") for item in value))


@dataclass(frozen=True, slots=True)
class FreshDiagnosticsSegment:
    expected_library_version: str
    kind: str = "fresh"

    def __post_init__(self) -> None:
        if not isinstance(self.expected_library_version, str) or not self.expected_library_version:
            raise ValueError("expected_library_version must be non-empty")
        if self.kind != "fresh":
            raise ValueError("unsupported fresh diagnostics state")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "expected_library_version": self.expected_library_version,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class ActiveDiagnosticsSegment:
    library_version: str
    recent_squared_residual_batches: tuple[ResidualBatchEvidence, ...]
    kind: str = "active"

    def __post_init__(self) -> None:
        if not isinstance(self.library_version, str) or not self.library_version:
            raise ValueError("library_version must be non-empty")
        if not self.recent_squared_residual_batches:
            raise ValueError("active diagnostics state requires residual history")
        if self.kind != "active":
            raise ValueError("unsupported active diagnostics state")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "library_version": self.library_version,
            "recent_squared_residual_batches": [
                item.to_value() for item in self.recent_squared_residual_batches
            ],
        }


DiagnosticsState: TypeAlias = FreshDiagnosticsSegment | ActiveDiagnosticsSegment


def diagnostics_state_from_value(value: object) -> DiagnosticsState:
    if not isinstance(value, Mapping):
        raise TypeError("diagnostics state must be an object")
    kind = value.get("kind")
    if kind == "fresh":
        if set(value) != {"expected_library_version", "kind"}:
            raise ValueError("FreshDiagnosticsSegment has incompatible fields")
        version = value["expected_library_version"]
        if not isinstance(version, str):
            raise TypeError("expected_library_version must be text")
        return FreshDiagnosticsSegment(version)
    if kind == "active":
        if set(value) != {
            "kind",
            "library_version",
            "recent_squared_residual_batches",
        }:
            raise ValueError("ActiveDiagnosticsSegment has incompatible fields")
        version = value["library_version"]
        raw_batches = value["recent_squared_residual_batches"]
        if not isinstance(version, str) or not isinstance(raw_batches, list):
            raise TypeError("active diagnostics state has incompatible fields")
        return ActiveDiagnosticsSegment(
            library_version=version,
            recent_squared_residual_batches=tuple(
                ResidualBatchEvidence.from_value(item) for item in raw_batches
            ),
        )
    raise ValueError("unsupported diagnostics state kind")


def diagnostics_library_version(state: DiagnosticsState) -> str:
    match state:
        case FreshDiagnosticsSegment(expected_library_version=version):
            return version
        case ActiveDiagnosticsSegment(library_version=version):
            return version
    from typing import assert_never

    assert_never(state)


@dataclass(frozen=True, slots=True)
class BatchDiagnostics:
    batch_id: str
    optimizer_step: int
    library_version: str
    trajectories: tuple[TrajectoryFlowDiagnostic, ...]
    skill_flows: tuple[SkillFlowStat, ...]
    residual_window: ResidualWindowEvidence
    config: DiagnosticsConfig

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_id": self.batch_id,
            "config": self.config.to_value(),
            "format": DIAGNOSTICS_FORMAT,
            "library_version": self.library_version,
            "optimizer_step": self.optimizer_step,
            "skill_flows": [skill.to_value() for skill in self.skill_flows],
            "residual_window": self.residual_window.to_value(),
            "trajectories": [trajectory.to_value() for trajectory in self.trajectories],
        }

    @classmethod
    def from_value(cls, value: object) -> BatchDiagnostics:
        expected = {
            "batch_id",
            "config",
            "format",
            "library_version",
            "optimizer_step",
            "residual_window",
            "skill_flows",
            "trajectories",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("BatchDiagnostics has incompatible fields")
        if value["format"] != DIAGNOSTICS_FORMAT:
            raise ValueError("BatchDiagnostics has an incompatible format")
        batch_id = value["batch_id"]
        library_version = value["library_version"]
        optimizer_step = value["optimizer_step"]
        raw_trajectories = value["trajectories"]
        raw_skills = value["skill_flows"]
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("batch_id must be non-empty text")
        if not isinstance(library_version, str) or not library_version:
            raise ValueError("library_version must be non-empty text")
        if type(optimizer_step) is not int or optimizer_step < 1:
            raise ValueError("optimizer_step must be positive")
        if not isinstance(raw_trajectories, list) or not isinstance(raw_skills, list):
            raise TypeError("diagnostic trajectories and skill flows must be arrays")
        return cls(
            batch_id=batch_id,
            optimizer_step=optimizer_step,
            library_version=library_version,
            trajectories=tuple(
                TrajectoryFlowDiagnostic.from_value(item) for item in raw_trajectories
            ),
            skill_flows=tuple(SkillFlowStat.from_value(item) for item in raw_skills),
            residual_window=residual_window_evidence_from_value(value["residual_window"]),
            config=DiagnosticsConfig.from_value(value["config"]),
        )


def _pooled_window_mean(batches: tuple[tuple[float, ...], ...]) -> float:
    count = sum(len(batch) for batch in batches)
    return math.fsum(value for batch in batches for value in batch) / count


def observe_batch(
    state: DiagnosticsState,
    batch: BatchFlowInput,
    config: DiagnosticsConfig,
) -> tuple[DiagnosticsState, BatchDiagnostics]:
    """Fold one materialized training batch into Protocol-v3 diagnostics."""

    library_version = batch.stats.library_version
    match state:
        case FreshDiagnosticsSegment(expected_library_version=expected):
            if library_version != expected:
                raise LibrarySegmentMismatchError(
                    "first diagnostic batch differs from the explicitly reset library segment"
                )
            previous: tuple[ResidualBatchEvidence, ...] = ()
        case ActiveDiagnosticsSegment(
            library_version=expected,
            recent_squared_residual_batches=previous,
        ):
            if library_version != expected:
                raise LibrarySegmentMismatchError(
                    "diagnostic batch crosses a library segment without an explicit reset"
                )
        case _:
            from typing import assert_never

            assert_never(state)
    by_id = {item.trajectory_id: item for item in batch.trajectories}
    trajectories = tuple(
        trajectory_flow(by_id[residual.trajectory_id]) for residual in batch.stats.residuals
    )
    current = ResidualBatchEvidence(tuple(residual.delta**2 for residual in batch.stats.residuals))
    all_batches = (*previous, current)[-2 * config.window_size :]
    next_state: DiagnosticsState = ActiveDiagnosticsSegment(
        library_version=library_version,
        recent_squared_residual_batches=all_batches,
    )

    required = 2 * config.window_size
    if len(all_batches) < required:
        residual_window: ResidualWindowEvidence = InsufficientResidualWindow(
            observed_batch_count=len(all_batches),
            required_batch_count=required,
        )
    else:
        recent = all_batches[-required:]
        previous_mean = _pooled_window_mean(
            tuple(item.delta_squared for item in recent[: config.window_size])
        )
        current_mean = _pooled_window_mean(
            tuple(item.delta_squared for item in recent[config.window_size :])
        )
        if previous_mean == 0.0:
            residual_window = ZeroPreviousResidual(
                library_version=library_version,
                window_size=config.window_size,
                previous_window_mean_delta_squared=previous_mean,
                current_window_mean_delta_squared=current_mean,
            )
        else:
            improvement = (previous_mean - current_mean) / previous_mean
            residual_window = ComparableResidualWindows(
                library_version=library_version,
                window_size=config.window_size,
                previous_window_mean_delta_squared=previous_mean,
                current_window_mean_delta_squared=current_mean,
                relative_improvement=improvement,
                rho=config.stagnation_rho,
                stagnant=improvement < config.stagnation_rho,
            )
    return next_state, BatchDiagnostics(
        batch_id=batch.stats.batch_id,
        optimizer_step=batch.stats.optimizer_step,
        library_version=library_version,
        trajectories=trajectories,
        skill_flows=skill_marginal_flows(trajectories),
        residual_window=residual_window,
        config=config,
    )


def reset_diagnostics_segment(
    state: DiagnosticsState,
    *,
    old_library_version: str,
    new_library_version: str,
) -> FreshDiagnosticsSegment:
    if diagnostics_library_version(state) != old_library_version:
        raise LibrarySegmentMismatchError("diagnostics reset old library identity differs")
    if new_library_version == old_library_version:
        raise ValueError("diagnostics reset requires a new library version")
    return FreshDiagnosticsSegment(expected_library_version=new_library_version)
