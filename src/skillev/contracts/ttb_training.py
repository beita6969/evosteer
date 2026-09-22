# ruff: noqa: RUF003
"""Pure data contracts for edge scoring and trajectory-balance batches.

This module deliberately contains no model, tensor, scoring, or optimization
logic.  It records the outputs of those later layers while keeping their units
and algebraic relationships explicit and independently checkable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .canonical import JsonValue, normalize_json, stable_hash
from .ttb_common import (
    FLOAT_TOLERANCE,
    require_finite_number,
    require_iso_timestamp,
    require_non_empty_text,
)

_TRAINING_STACK_ID = "training-stack"


def _require_object(value: object, *, record_name: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{record_name} must be a JSON object")
    return normalized


def _required(value: dict[str, JsonValue], key: str, *, record_name: str) -> JsonValue:
    try:
        return value[key]
    except KeyError as error:
        raise ValueError(f"{record_name}.{key} is required") from error


def _require_exact_fields(
    value: dict[str, JsonValue],
    *,
    fields: frozenset[str],
    record_name: str,
) -> None:
    if set(value) != fields:
        raise ValueError(f"{record_name} has incompatible fields")


def _string(value: JsonValue, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _integer(value: JsonValue, *, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return value


def _number(value: JsonValue, *, field_name: str) -> float:
    return require_finite_number(value, field=field_name)


def _array(value: JsonValue, *, field_name: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def _require_finite(value: object, *, field_name: str) -> None:
    require_finite_number(value, field=field_name)


def _require_non_empty(value: str, *, field_name: str) -> None:
    require_non_empty_text(value, field=field_name)


def _require_close(actual: float, expected: float, *, field_name: str) -> None:
    if not math.isfinite(expected) or not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError(f"{field_name} is inconsistent with its component values")


@dataclass(frozen=True, slots=True)
class EdgeScoreContext:
    """Scoring-time identity; the action span is the referenced trajectory step."""

    batch_id: str
    policy_snapshot_id: str
    library_version: str
    action_token_count: int
    action_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("batch_id", "policy_snapshot_id", "library_version"):
            _require_non_empty(getattr(self, name), field_name=f"edge context {name}")
        if type(self.action_token_count) is not int or self.action_token_count < 1:
            raise ValueError("edge action token count must be positive")
        if not isinstance(self.action_token_ids, tuple) or any(
            type(token) is not int or token < 0 for token in self.action_token_ids
        ):
            raise ValueError("edge action span must be an exact token tuple")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_id": self.batch_id,
            "policy_snapshot_id": self.policy_snapshot_id,
            "library_version": self.library_version,
            "action_token_count": self.action_token_count,
            "action_token_ids": list(self.action_token_ids),
        }

    @classmethod
    def from_value(cls, value: object) -> EdgeScoreContext:
        if not isinstance(value, dict):
            raise TypeError("edge scoring context must be an object")
        return cls(**{**value, "action_token_ids": tuple(value.get("action_token_ids", ()))})


@dataclass(frozen=True, slots=True)
class EdgeLogprobRecord:
    """Two policy scores and the resulting importance for one trajectory edge.

    This record only fixes the data shape and units.  It does not perform
    teacher-forced scoring.
    """

    trajectory_id: str
    step_index: int

    # P̂_F^t（式 6 前向）。单位钉死：自然对数（nats）、对 a_t 全部 K_t 个 token 的
    # logprob 求和后 ÷ K_t 的"每 token 均值"。这段注释必须原样出现在代码里——
    # 它是防单位漂移（一处用总和、一处用均值）的唯一文档点。
    forward_logprob_per_token: float
    # P̂_B^t（式 6 后向）：同一 a_t 序列、同一 K_t、hindsight 前缀（不含 r_t）。单位同上。
    backward_logprob_per_token: float
    # log I(t) = forward − backward（式 9）。冗余存储，校验自洽。
    step_importance: float

    forward_adapter_version: str
    backward_adapter_version: str
    scoring_stack_id: str
    # Standalone scoring can be unbound; training/diagnostics require this context.
    context: EdgeScoreContext | None = None

    def __post_init__(self) -> None:
        if self.context is not None and not isinstance(self.context, EdgeScoreContext):
            raise TypeError("edge context must be EdgeScoreContext")
        _require_non_empty(self.trajectory_id, field_name="EdgeLogprobRecord.trajectory_id")
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError(
                f"EdgeLogprobRecord[{self.trajectory_id}].step_index must be at least one"
            )
        for name, value in (
            ("forward_logprob_per_token", self.forward_logprob_per_token),
            ("backward_logprob_per_token", self.backward_logprob_per_token),
            ("step_importance", self.step_importance),
        ):
            _require_finite(
                value,
                field_name=f"EdgeLogprobRecord[{self.trajectory_id}:{self.step_index}].{name}",
            )
        _require_close(
            self.step_importance,
            self.forward_logprob_per_token - self.backward_logprob_per_token,
            field_name=(
                f"EdgeLogprobRecord[{self.trajectory_id}:{self.step_index}].step_importance"
            ),
        )
        _require_non_empty(
            self.forward_adapter_version,
            field_name="EdgeLogprobRecord.forward_adapter_version",
        )
        _require_non_empty(
            self.backward_adapter_version,
            field_name="EdgeLogprobRecord.backward_adapter_version",
        )
        if self.scoring_stack_id != _TRAINING_STACK_ID:
            raise ValueError(
                f"EdgeLogprobRecord[{self.trajectory_id}:{self.step_index}].scoring_stack_id "
                f"must be {_TRAINING_STACK_ID!r}"
            )

    def to_value(self) -> dict[str, JsonValue]:
        """Return the canonical-JSON-compatible record value."""

        return {
            "context": None if self.context is None else self.context.to_value(),
            "backward_adapter_version": self.backward_adapter_version,
            "backward_logprob_per_token": float(self.backward_logprob_per_token),
            "forward_adapter_version": self.forward_adapter_version,
            "forward_logprob_per_token": float(self.forward_logprob_per_token),
            "scoring_stack_id": self.scoring_stack_id,
            "step_importance": float(self.step_importance),
            "step_index": self.step_index,
            "trajectory_id": self.trajectory_id,
        }

    @classmethod
    def from_value(cls, value: object) -> EdgeLogprobRecord:
        """Rebuild a record from a JSON-compatible value and revalidate it."""

        record_name = cls.__name__
        item = _require_object(value, record_name=record_name)
        _require_exact_fields(
            item,
            fields=frozenset(
                {
                    "context",
                    "backward_adapter_version",
                    "backward_logprob_per_token",
                    "forward_adapter_version",
                    "forward_logprob_per_token",
                    "scoring_stack_id",
                    "step_importance",
                    "step_index",
                    "trajectory_id",
                }
            ),
            record_name=record_name,
        )
        return cls(
            context=None
            if item["context"] is None
            else EdgeScoreContext.from_value(item["context"]),
            trajectory_id=_string(
                _required(item, "trajectory_id", record_name=record_name),
                field_name=f"{record_name}.trajectory_id",
            ),
            step_index=_integer(
                _required(item, "step_index", record_name=record_name),
                field_name=f"{record_name}.step_index",
            ),
            forward_logprob_per_token=_number(
                _required(item, "forward_logprob_per_token", record_name=record_name),
                field_name=f"{record_name}.forward_logprob_per_token",
            ),
            backward_logprob_per_token=_number(
                _required(item, "backward_logprob_per_token", record_name=record_name),
                field_name=f"{record_name}.backward_logprob_per_token",
            ),
            step_importance=_number(
                _required(item, "step_importance", record_name=record_name),
                field_name=f"{record_name}.step_importance",
            ),
            forward_adapter_version=_string(
                _required(item, "forward_adapter_version", record_name=record_name),
                field_name=f"{record_name}.forward_adapter_version",
            ),
            backward_adapter_version=_string(
                _required(item, "backward_adapter_version", record_name=record_name),
                field_name=f"{record_name}.backward_adapter_version",
            ),
            scoring_stack_id=_string(
                _required(item, "scoring_stack_id", record_name=record_name),
                field_name=f"{record_name}.scoring_stack_id",
            ),
        )

    @property
    def content_hash(self) -> str:
        """Return the stable hash of the record's scientific content."""

        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class TrajectoryResidual:
    """The four-term residual decomposition for one batch trajectory.

    Keeping the components, rather than only :math:`Delta`, makes later
    diagnosis able to distinguish movement in the normalizer, policy scores,
    and reward term.
    """

    trajectory_id: str
    log_z: float
    # Σ_t P̂_F^t（每 token 均值之和——单位与 EdgeLogprobRecord 一致）。
    sum_forward: float
    # Σ_t P̂_B^t（每 token 均值之和——单位与 EdgeLogprobRecord 一致）。
    sum_backward: float
    log_shifted_reward: float
    raw_reward: float
    temperature_beta: float
    delta: float
    horizon: int

    def __post_init__(self) -> None:
        _require_non_empty(self.trajectory_id, field_name="TrajectoryResidual.trajectory_id")
        for name, value in (
            ("log_z", self.log_z),
            ("sum_forward", self.sum_forward),
            ("sum_backward", self.sum_backward),
            ("log_shifted_reward", self.log_shifted_reward),
            ("raw_reward", self.raw_reward),
            ("temperature_beta", self.temperature_beta),
            ("delta", self.delta),
        ):
            _require_finite(
                value,
                field_name=f"TrajectoryResidual[{self.trajectory_id}].{name}",
            )
        if not 0.0 <= self.raw_reward <= 1.0:
            raise ValueError(
                f"TrajectoryResidual[{self.trajectory_id}].raw_reward must be in [0, 1]"
            )
        if self.temperature_beta <= 0.0:
            raise ValueError(
                f"TrajectoryResidual[{self.trajectory_id}].temperature_beta must be positive"
            )
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError(
                f"TrajectoryResidual[{self.trajectory_id}].horizon must be at least one"
            )
        expected_delta = (
            self.log_z
            + self.sum_forward
            - self.temperature_beta * self.log_shifted_reward
            - self.sum_backward
        )
        _require_close(
            self.delta,
            expected_delta,
            field_name=f"TrajectoryResidual[{self.trajectory_id}].delta",
        )

    def to_value(self) -> dict[str, JsonValue]:
        """Return the canonical-JSON-compatible record value."""

        return {
            "delta": float(self.delta),
            "horizon": self.horizon,
            "log_shifted_reward": float(self.log_shifted_reward),
            "log_z": float(self.log_z),
            "raw_reward": float(self.raw_reward),
            "sum_backward": float(self.sum_backward),
            "sum_forward": float(self.sum_forward),
            "temperature_beta": float(self.temperature_beta),
            "trajectory_id": self.trajectory_id,
        }

    @classmethod
    def from_value(cls, value: object) -> TrajectoryResidual:
        """Rebuild a residual from a JSON-compatible value and revalidate it."""

        record_name = cls.__name__
        item = _require_object(value, record_name=record_name)
        _require_exact_fields(
            item,
            fields=frozenset(
                {
                    "delta",
                    "horizon",
                    "log_shifted_reward",
                    "log_z",
                    "raw_reward",
                    "sum_backward",
                    "sum_forward",
                    "temperature_beta",
                    "trajectory_id",
                }
            ),
            record_name=record_name,
        )
        return cls(
            trajectory_id=_string(
                _required(item, "trajectory_id", record_name=record_name),
                field_name=f"{record_name}.trajectory_id",
            ),
            log_z=_number(
                _required(item, "log_z", record_name=record_name),
                field_name=f"{record_name}.log_z",
            ),
            sum_forward=_number(
                _required(item, "sum_forward", record_name=record_name),
                field_name=f"{record_name}.sum_forward",
            ),
            sum_backward=_number(
                _required(item, "sum_backward", record_name=record_name),
                field_name=f"{record_name}.sum_backward",
            ),
            log_shifted_reward=_number(
                _required(item, "log_shifted_reward", record_name=record_name),
                field_name=f"{record_name}.log_shifted_reward",
            ),
            raw_reward=_number(
                _required(item, "raw_reward", record_name=record_name),
                field_name=f"{record_name}.raw_reward",
            ),
            temperature_beta=_number(
                _required(item, "temperature_beta", record_name=record_name),
                field_name=f"{record_name}.temperature_beta",
            ),
            delta=_number(
                _required(item, "delta", record_name=record_name),
                field_name=f"{record_name}.delta",
            ),
            horizon=_integer(
                _required(item, "horizon", record_name=record_name),
                field_name=f"{record_name}.horizon",
            ),
        )

    @property
    def content_hash(self) -> str:
        """Return the stable hash of the residual's scientific content."""

        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class TTBBatchStats:
    """A reconstructible health record for one TTB optimizer batch."""

    batch_id: str
    optimizer_step: int
    library_version: str
    residuals: tuple[TrajectoryResidual, ...]
    batch_loss: float
    mean_reward: float
    created_at: str

    def __post_init__(self) -> None:
        _require_non_empty(self.batch_id, field_name="TTBBatchStats.batch_id")
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError(f"TTBBatchStats[{self.batch_id}].optimizer_step must be non-negative")
        _require_non_empty(
            self.library_version,
            field_name=f"TTBBatchStats[{self.batch_id}].library_version",
        )
        require_iso_timestamp(
            self.created_at,
            field="created_at",
            location=f"TTBBatchStats[{self.batch_id}]",
        )
        if not isinstance(self.residuals, tuple) or not self.residuals:
            raise ValueError(f"TTBBatchStats[{self.batch_id}].residuals cannot be empty")
        if any(not isinstance(residual, TrajectoryResidual) for residual in self.residuals):
            raise ValueError(
                f"TTBBatchStats[{self.batch_id}].residuals must contain TrajectoryResidual records"
            )
        trajectory_ids = tuple(residual.trajectory_id for residual in self.residuals)
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError(
                f"TTBBatchStats[{self.batch_id}].residuals contain duplicate trajectory IDs"
            )
        _require_finite(
            self.batch_loss,
            field_name=f"TTBBatchStats[{self.batch_id}].batch_loss",
        )
        _require_finite(
            self.mean_reward,
            field_name=f"TTBBatchStats[{self.batch_id}].mean_reward",
        )
        try:
            expected_loss = math.fsum(
                (residual.delta / residual.horizon) ** 2 for residual in self.residuals
            ) / len(self.residuals)
        except OverflowError as error:
            raise ValueError(
                f"TTBBatchStats[{self.batch_id}].recomputed_batch_loss must be finite"
            ) from error
        expected_reward = math.fsum(residual.raw_reward for residual in self.residuals) / len(
            self.residuals
        )
        _require_finite(
            expected_loss,
            field_name=f"TTBBatchStats[{self.batch_id}].recomputed_batch_loss",
        )
        _require_close(
            self.batch_loss,
            expected_loss,
            field_name=f"TTBBatchStats[{self.batch_id}].batch_loss",
        )
        _require_close(
            self.mean_reward,
            expected_reward,
            field_name=f"TTBBatchStats[{self.batch_id}].mean_reward",
        )

    def to_value(self) -> dict[str, JsonValue]:
        """Return the canonical-JSON-compatible record value."""

        return {
            "batch_id": self.batch_id,
            "batch_loss": float(self.batch_loss),
            "created_at": self.created_at,
            "library_version": self.library_version,
            "mean_reward": float(self.mean_reward),
            "optimizer_step": self.optimizer_step,
            "residuals": [residual.to_value() for residual in self.residuals],
        }

    @classmethod
    def from_value(cls, value: object) -> TTBBatchStats:
        """Rebuild batch statistics from a JSON-compatible value and revalidate it."""

        record_name = cls.__name__
        item = _require_object(value, record_name=record_name)
        _require_exact_fields(
            item,
            fields=frozenset(
                {
                    "batch_id",
                    "batch_loss",
                    "created_at",
                    "library_version",
                    "mean_reward",
                    "optimizer_step",
                    "residuals",
                }
            ),
            record_name=record_name,
        )
        residual_values = _array(
            _required(item, "residuals", record_name=record_name),
            field_name=f"{record_name}.residuals",
        )
        return cls(
            batch_id=_string(
                _required(item, "batch_id", record_name=record_name),
                field_name=f"{record_name}.batch_id",
            ),
            optimizer_step=_integer(
                _required(item, "optimizer_step", record_name=record_name),
                field_name=f"{record_name}.optimizer_step",
            ),
            library_version=_string(
                _required(item, "library_version", record_name=record_name),
                field_name=f"{record_name}.library_version",
            ),
            residuals=tuple(
                TrajectoryResidual.from_value(residual) for residual in residual_values
            ),
            batch_loss=_number(
                _required(item, "batch_loss", record_name=record_name),
                field_name=f"{record_name}.batch_loss",
            ),
            mean_reward=_number(
                _required(item, "mean_reward", record_name=record_name),
                field_name=f"{record_name}.mean_reward",
            ),
            created_at=_string(
                _required(item, "created_at", record_name=record_name),
                field_name=f"{record_name}.created_at",
            ),
        )

    def _content_value(self) -> dict[str, JsonValue]:
        value = self.to_value()
        del value["created_at"]
        return value

    @property
    def content_hash(self) -> str:
        """Hash scientific content while excluding the audit-only timestamp."""

        return stable_hash(self._content_value())
