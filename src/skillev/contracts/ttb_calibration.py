"""Immutable contracts for feature-conditioned Beta posterior calibration.

This module only defines canonical data, validation, and pure derived
quantities.  It deliberately has no dependency on a model or training stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .canonical import JsonValue, normalize_json, stable_hash
from .ttb_common import FLOAT_TOLERANCE


def _require_nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _require_integer(value: object, *, field: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer greater than or equal to {minimum}")
    return value


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _require_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_record(
    value: object,
    *,
    field: str,
    expected_fields: frozenset[str],
) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{field} must be a canonical JSON object")
    if set(normalized) != expected_fields:
        raise ValueError(f"{field} has an incompatible field set")
    return normalized


def _float_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)


class FailureMode(StrEnum):
    """The coarse failure-mode coordinate of ``z``.

    The six categories intentionally stay coarse so observations do not
    fragment into sparse posterior cells.
    """

    SUCCESS = "success"
    TOOL_ERROR = "tool_error"
    SCHEMA_INVALID = "schema_invalid"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"
    OTHER = "other"

    @classmethod
    def from_observation_status(cls, status: str) -> FailureMode:
        """Map a producer-emitted member of the closed observation vocabulary."""

        if type(status) is not str:
            raise ValueError("observation status must be text")
        try:
            return cls(status)
        except ValueError as error:
            raise ValueError(
                f"unknown observation_status {status!r}; producers must emit a closed status"
            ) from error


class TokenBucket(StrEnum):
    """The token-count coordinate of ``z``.

    These boundaries are part of the persisted feature semantics.  Changing
    them invalidates every accumulated posterior cell and requires rebuilding
    those cells from newly categorized evidence.
    """

    LE_1K = "le_1k"
    K1_TO_4K = "k1_to_4k"
    GT_4K = "gt_4k"

    @classmethod
    def from_count(cls, n: int) -> TokenBucket:
        count = _require_integer(n, field="token count", minimum=0)
        if count <= 1_000:
            return cls.LE_1K
        if count <= 4_000:
            return cls.K1_TO_4K
        return cls.GT_4K


class HorizonBucket(StrEnum):
    """The trajectory-horizon coordinate of ``z``.

    These boundaries are part of the persisted feature semantics.  Changing
    them invalidates every accumulated posterior cell and requires rebuilding
    those cells from newly categorized evidence.
    """

    LE_3 = "le_3"
    H4_TO_8 = "h4_to_8"
    GT_8 = "gt_8"

    @classmethod
    def from_horizon(cls, t: int) -> HorizonBucket:
        horizon = _require_integer(t, field="trajectory horizon", minimum=1)
        if horizon <= 3:
            return cls.LE_3
        if horizon <= 8:
            return cls.H4_TO_8
        return cls.GT_8


_CONTEXT_FIELDS = frozenset(
    {
        "context",
        "failure_mode",
        "horizon_bucket",
        "token_bucket",
    }
)


@dataclass(frozen=True, slots=True)
class ContextFeature:
    """The context ``z = [Context, FailureMode, TokenBucket, HorizonBucket]``."""

    context: str
    failure_mode: FailureMode
    token_bucket: TokenBucket
    horizon_bucket: HorizonBucket

    def __post_init__(self) -> None:
        _require_nonempty_text(self.context, field="context")
        if not isinstance(self.failure_mode, FailureMode):
            raise ValueError("failure_mode must be a FailureMode")
        if not isinstance(self.token_bucket, TokenBucket):
            raise ValueError("token_bucket must be a TokenBucket")
        if not isinstance(self.horizon_bucket, HorizonBucket):
            raise ValueError("horizon_bucket must be a HorizonBucket")

    def cell_key(self, skill_id: str) -> str:
        """Return the canonical content key for one ``(skill, z)`` cell."""

        skill = _require_nonempty_text(skill_id, field="skill_id")
        return stable_hash({"skill_id": skill, "z": self.to_value()})

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "context": self.context,
            "failure_mode": self.failure_mode.value,
            "horizon_bucket": self.horizon_bucket.value,
            "token_bucket": self.token_bucket.value,
        }

    @classmethod
    def from_value(cls, value: object) -> ContextFeature:
        record = _require_record(
            value,
            field="context feature",
            expected_fields=_CONTEXT_FIELDS,
        )
        return cls(
            context=_require_nonempty_text(record["context"], field="context"),
            failure_mode=FailureMode(
                _require_nonempty_text(record["failure_mode"], field="failure_mode")
            ),
            token_bucket=TokenBucket(
                _require_nonempty_text(record["token_bucket"], field="token_bucket")
            ),
            horizon_bucket=HorizonBucket(
                _require_nonempty_text(record["horizon_bucket"], field="horizon_bucket")
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


_UPDATE_FIELDS = frozenset(
    {
        "alpha_before",
        "beta_count_before",
        "alpha_after",
        "beta_count_after",
        "event_id",
        "flow_weight",
        "outcome",
        "skill_id",
        "step_index",
        "trajectory_id",
        "z",
    }
)


@dataclass(frozen=True, slots=True)
class PosteriorUpdateEvent:
    """One replayable, flow-weighted conjugate-update event.

    The before/after counts and source invocation make the committed arithmetic
    inspectable without interpreting a native continuous score as Bernoulli Y.
    """

    event_id: str
    skill_id: str
    z: ContextFeature
    trajectory_id: str
    step_index: int
    outcome: bool
    flow_weight: float
    alpha_after: float
    beta_count_after: float
    alpha_before: float
    beta_count_before: float

    def __post_init__(self) -> None:
        _require_nonempty_text(self.event_id, field="event_id")
        _require_nonempty_text(self.skill_id, field="skill_id")
        if not isinstance(self.z, ContextFeature):
            raise ValueError("z must be a ContextFeature")
        _require_nonempty_text(self.trajectory_id, field="trajectory_id")
        _require_integer(self.step_index, field="step_index", minimum=1)
        _require_bool(self.outcome, field="outcome")
        flow_weight = _require_finite_number(self.flow_weight, field="flow_weight")
        if flow_weight < 0:
            raise ValueError("flow_weight must be non-negative")
        alpha_after = _require_finite_number(self.alpha_after, field="alpha_after")
        beta_after = _require_finite_number(
            self.beta_count_after,
            field="beta_count_after",
        )
        if alpha_after <= 0 or beta_after <= 0:
            raise ValueError("posterior after-counts must be positive")
        if (
            _require_finite_number(self.alpha_before, field="alpha_before") <= 0
            or _require_finite_number(self.beta_count_before, field="beta_count_before") <= 0
        ):
            raise ValueError("posterior before-counts must be positive")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "alpha_before": float(self.alpha_before),
            "beta_count_before": float(self.beta_count_before),
            "alpha_after": float(self.alpha_after),
            "beta_count_after": float(self.beta_count_after),
            "event_id": self.event_id,
            "flow_weight": float(self.flow_weight),
            "outcome": self.outcome,
            "skill_id": self.skill_id,
            "step_index": self.step_index,
            "trajectory_id": self.trajectory_id,
            "z": self.z.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> PosteriorUpdateEvent:
        record = _require_record(
            value,
            field="posterior update event",
            expected_fields=_UPDATE_FIELDS,
        )
        return cls(
            alpha_before=_require_finite_number(record["alpha_before"], field="alpha_before"),
            beta_count_before=_require_finite_number(
                record["beta_count_before"], field="beta_count_before"
            ),
            event_id=_require_nonempty_text(record["event_id"], field="event_id"),
            skill_id=_require_nonempty_text(record["skill_id"], field="skill_id"),
            z=ContextFeature.from_value(record["z"]),
            trajectory_id=_require_nonempty_text(
                record["trajectory_id"],
                field="trajectory_id",
            ),
            step_index=_require_integer(
                record["step_index"],
                field="step_index",
                minimum=1,
            ),
            outcome=_require_bool(record["outcome"], field="outcome"),
            flow_weight=_require_finite_number(
                record["flow_weight"],
                field="flow_weight",
            ),
            alpha_after=_require_finite_number(
                record["alpha_after"],
                field="alpha_after",
            ),
            beta_count_after=_require_finite_number(
                record["beta_count_after"],
                field="beta_count_after",
            ),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


_BATCH_UPDATE_FIELDS = frozenset({"batch_id", "updates"})


@dataclass(frozen=True, slots=True)
class PosteriorBatchUpdate:
    """All ordered posterior updates projected from one committed batch."""

    batch_id: str
    updates: tuple[PosteriorUpdateEvent, ...]

    def __post_init__(self) -> None:
        _require_nonempty_text(self.batch_id, field="batch_id")
        if not isinstance(self.updates, tuple) or any(
            not isinstance(update, PosteriorUpdateEvent) for update in self.updates
        ):
            raise ValueError("updates must be a tuple of PosteriorUpdateEvent values")
        event_ids = tuple(update.event_id for update in self.updates)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("posterior batch contains duplicate update ids")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_id": self.batch_id,
            "updates": [update.to_value() for update in self.updates],
        }

    @classmethod
    def from_value(cls, value: object) -> PosteriorBatchUpdate:
        record = _require_record(
            value,
            field="posterior batch update",
            expected_fields=_BATCH_UPDATE_FIELDS,
        )
        raw_updates = record["updates"]
        if not isinstance(raw_updates, list):
            raise ValueError("posterior batch updates must be an array")
        return cls(
            batch_id=_require_nonempty_text(record["batch_id"], field="batch_id"),
            updates=tuple(PosteriorUpdateEvent.from_value(update) for update in raw_updates),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


_STATE_FIELDS = frozenset(
    {
        "alpha",
        "alpha_0",
        "beta_0",
        "beta_count",
        "last_event_id",
        "skill_id",
        "update_count",
        "z",
    }
)


@dataclass(frozen=True, slots=True)
class PosteriorCellState:
    """Cached state of one ``(skill, z)`` Beta posterior cell.

    Live queries read the committed projection snapshot. Its recorded update
    evidence reconstructs these values when that snapshot is restored.
    """

    skill_id: str
    z: ContextFeature
    alpha: float
    beta_count: float
    alpha_0: float = 1.0
    beta_0: float = 1.0
    update_count: int = 0
    last_event_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.skill_id, field="skill_id")
        if not isinstance(self.z, ContextFeature):
            raise ValueError("z must be a ContextFeature")
        alpha = _require_finite_number(self.alpha, field="alpha")
        beta_count = _require_finite_number(self.beta_count, field="beta_count")
        alpha_0 = _require_finite_number(self.alpha_0, field="alpha_0")
        beta_0 = _require_finite_number(self.beta_0, field="beta_0")
        if alpha_0 <= 0 or beta_0 <= 0:
            raise ValueError("Beta prior counts must be positive")
        if alpha < alpha_0 or beta_count < beta_0:
            raise ValueError("Posterior counts cannot be below their priors")
        _require_integer(self.update_count, field="update_count", minimum=0)
        if self.last_event_id is not None:
            _require_nonempty_text(self.last_event_id, field="last_event_id")
        if (self.update_count == 0) != (self.last_event_id is None):
            raise ValueError("update_count and last_event_id must describe the same state")
        if self.update_count == 0 and (
            not _float_close(alpha, alpha_0) or not _float_close(beta_count, beta_0)
        ):
            raise ValueError("A never-updated cell must equal its prior")

    @property
    def cumulative_flow_weight(self) -> float:
        """Weighted evidence mass, not a count of independent observations."""

        return (float(self.alpha) - float(self.alpha_0)) + (
            float(self.beta_count) - float(self.beta_0)
        )

    def mean(self) -> float:
        """Return ``alpha / (alpha + beta_count)`` without caching it."""

        alpha = float(self.alpha)
        beta_count = float(self.beta_count)
        if alpha >= beta_count:
            return 1.0 / (1.0 + beta_count / alpha)
        ratio = alpha / beta_count
        return ratio / (1.0 + ratio)

    def variance(self) -> float:
        """Return the Beta posterior variance without caching it."""

        mean = self.mean()
        alpha = float(self.alpha)
        beta_count = float(self.beta_count)
        scale = max(alpha, beta_count, 1.0)
        inverse_total_plus_one = (1.0 / scale) / (alpha / scale + beta_count / scale + 1.0 / scale)
        return mean * (1.0 - mean) * inverse_total_plus_one

    def lcb(self, k: float) -> float:
        """Return ``mean - k * sqrt(variance)`` for caller-supplied ``k``."""

        multiplier = _require_finite_number(k, field="k")
        if multiplier < 0:
            raise ValueError("k must be non-negative")
        return self.mean() - multiplier * math.sqrt(self.variance())

    def ucb(self, k: float) -> float:
        """Return ``mean + k * sqrt(variance)`` for caller-supplied ``k``."""

        multiplier = _require_finite_number(k, field="k")
        if multiplier < 0:
            raise ValueError("k must be non-negative")
        return self.mean() + multiplier * math.sqrt(self.variance())

    def apply(self, event: PosteriorUpdateEvent) -> PosteriorCellState:
        """Validate and purely apply one event, returning a new cell state."""

        if not isinstance(event, PosteriorUpdateEvent):
            raise ValueError("event must be a PosteriorUpdateEvent")
        if event.skill_id != self.skill_id or event.z != self.z:
            raise ValueError("posterior update event targets a different cell")
        if event.event_id == self.last_event_id:
            raise ValueError("posterior event has already been applied")
        if not _float_close(event.alpha_before, float(self.alpha)) or not _float_close(
            event.beta_count_before, float(self.beta_count)
        ):
            raise ValueError("posterior event before-counts differ from the current cell")

        flow_weight = float(event.flow_weight)
        expected_alpha = float(self.alpha) + (flow_weight if event.outcome else 0.0)
        expected_beta = float(self.beta_count) + (0.0 if event.outcome else flow_weight)
        if not math.isfinite(expected_alpha) or not math.isfinite(expected_beta):
            raise ValueError("posterior update overflows finite counts")
        if not _float_close(float(event.alpha_after), expected_alpha):
            raise ValueError("alpha_after does not match the recorded update")
        if not _float_close(float(event.beta_count_after), expected_beta):
            raise ValueError("beta_count_after does not match the recorded update")

        return PosteriorCellState(
            skill_id=self.skill_id,
            z=self.z,
            alpha=expected_alpha,
            beta_count=expected_beta,
            alpha_0=float(self.alpha_0),
            beta_0=float(self.beta_0),
            update_count=self.update_count + 1,
            last_event_id=event.event_id,
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "alpha": float(self.alpha),
            "alpha_0": float(self.alpha_0),
            "beta_0": float(self.beta_0),
            "beta_count": float(self.beta_count),
            "last_event_id": self.last_event_id,
            "skill_id": self.skill_id,
            "update_count": self.update_count,
            "z": self.z.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> PosteriorCellState:
        record = _require_record(
            value,
            field="posterior cell state",
            expected_fields=_STATE_FIELDS,
        )
        last_event_value = record["last_event_id"]
        if last_event_value is not None and not isinstance(last_event_value, str):
            raise ValueError("last_event_id must be text or null")
        return cls(
            skill_id=_require_nonempty_text(record["skill_id"], field="skill_id"),
            z=ContextFeature.from_value(record["z"]),
            alpha=_require_finite_number(record["alpha"], field="alpha"),
            beta_count=_require_finite_number(
                record["beta_count"],
                field="beta_count",
            ),
            alpha_0=_require_finite_number(record["alpha_0"], field="alpha_0"),
            beta_0=_require_finite_number(record["beta_0"], field="beta_0"),
            update_count=_require_integer(
                record["update_count"],
                field="update_count",
                minimum=0,
            ),
            last_event_id=last_event_value,
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


__all__ = [
    "ContextFeature",
    "FailureMode",
    "HorizonBucket",
    "PosteriorBatchUpdate",
    "PosteriorCellState",
    "PosteriorUpdateEvent",
    "TokenBucket",
]
