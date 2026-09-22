"""Pure Protocol-v3 flow-weighted Bayesian calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from skillev.contracts import (
    ContextFeature,
    FailureMode,
    HorizonBucket,
    JsonValue,
    PosteriorCellState,
    PosteriorUpdateEvent,
    TokenBucket,
    TrajectoryRecord,
    TrajectoryStep,
    stable_hash,
)
from skillev.diagnostics import TrajectoryFlowDiagnostic

CALIBRATION_FORMAT: Final = "skillev-calibration@3"
EXTRACTOR_VERSION: Final = "ttb-z-extractors@3"


def _finite_number(value: object, *, field: str) -> float:
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
class CalibrationConfig:
    """Full-method Beta prior and confidence-bound controls."""

    alpha_0: float = 1.0
    beta_0: float = 1.0
    default_k: float = 1.0
    format: str = CALIBRATION_FORMAT

    def __post_init__(self) -> None:
        if self.format != CALIBRATION_FORMAT:
            raise ValueError(f"CalibrationConfig.format must be {CALIBRATION_FORMAT!r}")
        alpha_0 = _finite_number(self.alpha_0, field="CalibrationConfig.alpha_0")
        beta_0 = _finite_number(self.beta_0, field="CalibrationConfig.beta_0")
        default_k = _finite_number(
            self.default_k,
            field="CalibrationConfig.default_k",
        )
        if alpha_0 <= 0.0 or beta_0 <= 0.0:
            raise ValueError("CalibrationConfig priors must be positive")
        if default_k < 0.0:
            raise ValueError("CalibrationConfig.default_k must be non-negative")
        object.__setattr__(self, "alpha_0", alpha_0)
        object.__setattr__(self, "beta_0", beta_0)
        object.__setattr__(self, "default_k", default_k)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
            "default_k": self.default_k,
            "format": self.format,
        }

    @classmethod
    def from_value(cls, value: object) -> CalibrationConfig:
        if not isinstance(value, Mapping) or set(value) != {
            "alpha_0",
            "beta_0",
            "default_k",
            "format",
        }:
            raise ValueError("CalibrationConfig has incompatible fields")
        if value["format"] != CALIBRATION_FORMAT:
            raise ValueError("CalibrationConfig has an incompatible format")
        numeric: dict[str, float] = {}
        for field in ("alpha_0", "beta_0", "default_k"):
            item = value[field]
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise ValueError(f"CalibrationConfig.{field} must be numeric")
            numeric[field] = float(item)
        return cls(
            alpha_0=numeric["alpha_0"],
            beta_0=numeric["beta_0"],
            default_k=numeric["default_k"],
        )


@dataclass(frozen=True, slots=True)
class CellQuery:
    skill_id: str
    z: ContextFeature
    alpha: float
    beta_count: float
    mean: float
    sigma: float
    lcb: float
    ucb: float
    k: float
    observed: bool
    update_count: int
    cumulative_flow_weight: float


def query_cell(cell: PosteriorCellState, k: float) -> CellQuery:
    """One confidence convention for post-hoc queries and evolution decisions."""

    return CellQuery(
        skill_id=cell.skill_id,
        z=cell.z,
        alpha=float(cell.alpha),
        beta_count=float(cell.beta_count),
        mean=cell.mean(),
        sigma=math.sqrt(cell.variance()),
        lcb=cell.lcb(k),
        ucb=cell.ucb(k),
        k=k,
        observed=cell.update_count > 0,
        update_count=cell.update_count,
        cumulative_flow_weight=cell.cumulative_flow_weight,
    )


def extract_context(record: TrajectoryRecord) -> str:
    """Return the namespaced task-family Context axis, not episode/retrieval context."""

    if not isinstance(record, TrajectoryRecord):
        raise TypeError("context extraction requires TrajectoryRecord")
    return record.task_family


def _step_at(record: TrajectoryRecord, step_index: int) -> TrajectoryStep:
    if type(step_index) is not int or not 1 <= step_index <= record.horizon:
        raise ValueError(
            f"trajectory {record.trajectory_id!r}, step_index={step_index!r}: "
            "step index is out of range"
        )
    return record.steps[step_index - 1]


def extract_failure_mode(record: TrajectoryRecord, step_index: int) -> FailureMode:
    return FailureMode.from_observation_status(_step_at(record, step_index).observation_status)


def extract_token_bucket(record: TrajectoryRecord) -> TokenBucket:
    return TokenBucket.from_count(record.initial_context.assembled_token_count)


def extract_horizon_bucket(record: TrajectoryRecord) -> HorizonBucket:
    return HorizonBucket.from_horizon(record.horizon)


def extract_calibration_outcome(record: TrajectoryRecord, step_index: int) -> bool:
    """Return the trajectory-terminal Bernoulli label for one invocation."""

    _step_at(record, step_index)
    return record.reward.success


def extract_z(record: TrajectoryRecord, step_index: int) -> ContextFeature:
    return ContextFeature(
        context=extract_context(record),
        failure_mode=extract_failure_mode(record, step_index),
        token_bucket=extract_token_bucket(record),
        horizon_bucket=extract_horizon_bucket(record),
    )


def flow_weights(
    flows: tuple[TrajectoryFlowDiagnostic, ...],
) -> dict[tuple[str, int], float]:
    """Normalize within this batch's invoking edges to mean one.

    Non-invoking edges are excluded; scale is not shared across batches; no
    clipping, cap, or temporal smoothing is applied.
    """

    invoking = tuple(
        (
            trajectory.trajectory_id,
            edge.step_index,
            edge.sample_log_state_weight,
        )
        for trajectory in flows
        for edge in trajectory.edges
        if edge.invoked_skill_ids
    )
    if not invoking:
        return {}
    for trajectory_id, step_index, value in invoking:
        _finite_number(value, field=f"{trajectory_id}:{step_index} sample_log_state_weight")
    if len({(trajectory_id, step_index) for trajectory_id, step_index, _ in invoking}) != len(
        invoking
    ):
        raise ValueError("flow normalization repeats an invoking edge")
    maximum = max(value for _, _, value in invoking)
    shifted = tuple(math.exp(value - maximum) for _, _, value in invoking)
    mean = math.fsum(shifted) / len(shifted)
    result: dict[tuple[str, int], float] = {}
    for (trajectory_id, step_index, value), linear in zip(invoking, shifted, strict=True):
        # Keep the established arithmetic for ordinary weights. If exp(logw -
        # max) underflows, normalize in log space before the final conversion:
        # the mean-one scale can make an otherwise lost subnormal representable.
        weight = linear / mean if linear else math.exp((value - maximum) - math.log(mean))
        if weight == 0.0:
            raise ValueError(
                f"{trajectory_id}:{step_index}: normalized flow weight underflow; "
                f"sample_log_state_weight={value}, batch_maximum={maximum}"
            )
        result[trajectory_id, step_index] = weight
    return result


def prior_cell(
    *,
    skill_id: str,
    z: ContextFeature,
    config: CalibrationConfig,
) -> PosteriorCellState:
    return PosteriorCellState(
        skill_id=skill_id,
        z=z,
        alpha=config.alpha_0,
        beta_count=config.beta_0,
        alpha_0=config.alpha_0,
        beta_0=config.beta_0,
    )


def _validated_cells_before(
    cells_before: Mapping[str, PosteriorCellState],
    config: CalibrationConfig,
) -> dict[str, PosteriorCellState]:
    validated: dict[str, PosteriorCellState] = {}
    for key, cell in cells_before.items():
        if not isinstance(cell, PosteriorCellState):
            raise ValueError("cells_before values must be PosteriorCellState records")
        if key != cell.z.cell_key(cell.skill_id):
            raise ValueError("cells_before key differs from canonical cell identity")
        if float(cell.alpha_0) != config.alpha_0 or float(cell.beta_0) != config.beta_0:
            raise ValueError("cells_before prior differs from CalibrationConfig")
        validated[key] = cell
    return validated


def calibration_updates_for_batch(
    *,
    batch_id: str,
    flows: tuple[TrajectoryFlowDiagnostic, ...],
    records_by_id: Mapping[str, TrajectoryRecord],
    cells_before: Mapping[str, PosteriorCellState],
    config: CalibrationConfig,
) -> tuple[PosteriorUpdateEvent, ...]:
    """Generate the deterministic trajectory→step→skill update chain."""

    return calibration_updates_with_weights(
        batch_id=batch_id,
        flows=flows,
        records_by_id=records_by_id,
        cells_before=cells_before,
        config=config,
        weights=flow_weights(flows),
    )


def calibration_updates_with_weights(
    *,
    batch_id: str,
    flows: tuple[TrajectoryFlowDiagnostic, ...],
    records_by_id: Mapping[str, TrajectoryRecord],
    cells_before: Mapping[str, PosteriorCellState],
    config: CalibrationConfig,
    weights: Mapping[tuple[str, int], float],
) -> tuple[PosteriorUpdateEvent, ...]:
    """Apply one explicitly supplied arm kernel to the canonical update order."""

    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("batch_id must be non-empty text")
    rolling = _validated_cells_before(cells_before, config)
    expected_weight_keys = {
        (trajectory.trajectory_id, edge.step_index)
        for trajectory in flows
        for edge in trajectory.edges
        if edge.invoked_skill_ids
    }
    if set(weights) != expected_weight_keys:
        raise ValueError("calibration weights differ from invoking edges")
    normalized_weights: dict[tuple[str, int], float] = {}
    for key, value in weights.items():
        weight = _finite_number(value, field="flow_weight")
        if weight < 0.0:
            raise ValueError("flow_weight cannot be negative")
        normalized_weights[key] = weight
    updates: list[PosteriorUpdateEvent] = []
    for trajectory in flows:
        invoking_edges = tuple(edge for edge in trajectory.edges if edge.invoked_skill_ids)
        if not invoking_edges:
            continue
        try:
            record = records_by_id[trajectory.trajectory_id]
        except KeyError as error:
            raise ValueError(
                f"batch {batch_id!r}, trajectory {trajectory.trajectory_id!r}: missing record"
            ) from error
        if record.trajectory_id != trajectory.trajectory_id:
            raise ValueError("record mapping key differs from trajectory identity")
        for edge in invoking_edges:
            if edge.invoked_skill_ids != _step_at(record, edge.step_index).invoked_skill_ids:
                raise ValueError("calibration credit differs from the executed skill action")
            z = extract_z(record, edge.step_index)
            outcome = extract_calibration_outcome(record, edge.step_index)
            weight = normalized_weights[(trajectory.trajectory_id, edge.step_index)]
            for skill_id in sorted(edge.invoked_skill_ids):
                cell_key = z.cell_key(skill_id)
                prior = rolling.get(cell_key)
                if prior is None:
                    prior = prior_cell(skill_id=skill_id, z=z, config=config)
                update = PosteriorUpdateEvent(
                    alpha_before=float(prior.alpha),
                    beta_count_before=float(prior.beta_count),
                    event_id=stable_hash(
                        {
                            "batch_id": batch_id,
                            "skill_id": skill_id,
                            "step_index": edge.step_index,
                            "trajectory_id": trajectory.trajectory_id,
                        }
                    ),
                    skill_id=skill_id,
                    z=z,
                    trajectory_id=trajectory.trajectory_id,
                    step_index=edge.step_index,
                    outcome=outcome,
                    flow_weight=weight,
                    alpha_after=float(prior.alpha) + (weight if outcome else 0.0),
                    beta_count_after=float(prior.beta_count) + (0.0 if outcome else weight),
                )
                rolling[cell_key] = prior.apply(update)
                updates.append(update)
    return tuple(updates)


__all__ = [
    "CALIBRATION_FORMAT",
    "EXTRACTOR_VERSION",
    "CalibrationConfig",
    "CellQuery",
    "calibration_updates_for_batch",
    "calibration_updates_with_weights",
    "extract_calibration_outcome",
    "extract_context",
    "extract_failure_mode",
    "extract_horizon_bucket",
    "extract_token_bucket",
    "extract_z",
    "flow_weights",
    "prior_cell",
    "query_cell",
]
