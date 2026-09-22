"""Evidence records for trajectory-balance phase transitions.

This module only defines immutable data and validation.  Computing windows or
deciding when to inspect them belongs to later implementation layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise

from .canonical import JsonValue, normalize_json, stable_hash
from .ttb_common import FLOAT_TOLERANCE, require_finite_number, require_non_empty_text

PHASE_TRANSITION_FORMAT = "skillev-phase-transition@2"


def _require_json_object(value: object, *, record: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{record} must be a JSON object")
    return normalized


def _require_exact_fields(
    value: dict[str, JsonValue],
    *,
    fields: frozenset[str],
    record: str,
) -> None:
    if set(value) != fields:
        raise ValueError(f"{record} has incompatible fields")


def _require_json_list(value: JsonValue, *, field: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return value


def _read_int(value: JsonValue, *, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _read_float(value: JsonValue, *, field: str) -> float:
    return require_finite_number(value, field=field)


def _read_bool(value: JsonValue, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _read_text(value: JsonValue, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    return value


class PhaseTriggerRule(StrEnum):
    """Closed rule selecting the evidence conjunction for one transition."""

    RESIDUAL_AND_ENTROPY = "residual-and-entropy"
    RESIDUAL_ONLY = "residual-only"
    ZERO_COVERAGE_COLD_START = "zero-coverage-generate@1"


@dataclass(frozen=True, slots=True)
class WindowStats:
    """Residual evidence from one closed optimizer-step window."""

    start_optimizer_step: int
    end_optimizer_step: int
    batch_count: int
    mean_squared_residual: float
    member_batch_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        location = (
            f"window[{self.start_optimizer_step}:{self.end_optimizer_step}]"
            if type(self.start_optimizer_step) is int and type(self.end_optimizer_step) is int
            else "window"
        )
        if type(self.start_optimizer_step) is not int or self.start_optimizer_step < 0:
            raise ValueError(f"{location}.start_optimizer_step must be a non-negative integer")
        if type(self.end_optimizer_step) is not int:
            raise ValueError(f"{location}.end_optimizer_step must be an integer")
        if self.end_optimizer_step < self.start_optimizer_step:
            raise ValueError(f"{location} has an invalid optimizer-step interval")
        if type(self.batch_count) is not int or self.batch_count < 1:
            raise ValueError(f"{location}.batch_count must be a positive integer")
        if not isinstance(self.member_batch_ids, tuple):
            raise ValueError(f"{location}.member_batch_ids must be a tuple")
        if self.batch_count != len(self.member_batch_ids):
            raise ValueError(f"{location}.batch_count does not match member_batch_ids")
        for batch_id in self.member_batch_ids:
            require_non_empty_text(batch_id, field="member_batch_ids", location=location)
        if len(set(self.member_batch_ids)) != len(self.member_batch_ids):
            raise ValueError(f"{location}.member_batch_ids must be unique")
        residual = require_finite_number(
            self.mean_squared_residual,
            field="mean_squared_residual",
            location=location,
        )
        if residual < 0:
            raise ValueError(f"{location}.mean_squared_residual cannot be negative")
        object.__setattr__(self, "mean_squared_residual", residual)

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "batch_count": self.batch_count,
            "end_optimizer_step": self.end_optimizer_step,
            "mean_squared_residual": self.mean_squared_residual,
            "member_batch_ids": list(self.member_batch_ids),
            "start_optimizer_step": self.start_optimizer_step,
        }

    @classmethod
    def from_value(cls, value: object) -> WindowStats:
        data = _require_json_object(value, record="WindowStats")
        _require_exact_fields(
            data,
            fields=frozenset(
                {
                    "batch_count",
                    "end_optimizer_step",
                    "mean_squared_residual",
                    "member_batch_ids",
                    "start_optimizer_step",
                }
            ),
            record="WindowStats",
        )
        raw_ids = _require_json_list(data["member_batch_ids"], field="member_batch_ids")
        return cls(
            start_optimizer_step=_read_int(
                data["start_optimizer_step"], field="start_optimizer_step"
            ),
            end_optimizer_step=_read_int(data["end_optimizer_step"], field="end_optimizer_step"),
            batch_count=_read_int(data["batch_count"], field="batch_count"),
            mean_squared_residual=_read_float(
                data["mean_squared_residual"], field="mean_squared_residual"
            ),
            member_batch_ids=tuple(_read_text(item, field="member_batch_ids") for item in raw_ids),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class EntropyObservation:
    """One skill-use Shannon-entropy observation."""

    window_end_step: int
    entropy: float
    total_invocations: int
    distinct_skills: int

    def __post_init__(self) -> None:
        location = (
            f"entropy_observation[{self.window_end_step}]"
            if type(self.window_end_step) is int
            else "entropy_observation"
        )
        if type(self.window_end_step) is not int or self.window_end_step < 0:
            raise ValueError(f"{location}.window_end_step must be a non-negative integer")
        entropy = require_finite_number(self.entropy, field="entropy", location=location)
        if entropy < 0:
            raise ValueError(f"{location}.entropy cannot be negative")
        object.__setattr__(self, "entropy", entropy)
        if type(self.total_invocations) is not int or self.total_invocations < 0:
            raise ValueError(f"{location}.total_invocations must be non-negative")
        if type(self.distinct_skills) is not int or self.distinct_skills < 0:
            raise ValueError(f"{location}.distinct_skills must be non-negative")
        if self.total_invocations == 0:
            if self.distinct_skills != 0 or entropy != 0.0:
                raise ValueError(f"{location} zero-use entropy must be exactly 0/0/0")
        elif not 1 <= self.distinct_skills <= self.total_invocations:
            raise ValueError(f"{location} positive-use entropy counts are inconsistent")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "distinct_skills": self.distinct_skills,
            "entropy": self.entropy,
            "total_invocations": self.total_invocations,
            "window_end_step": self.window_end_step,
        }

    @classmethod
    def from_value(cls, value: object) -> EntropyObservation:
        data = _require_json_object(value, record="EntropyObservation")
        _require_exact_fields(
            data,
            fields=frozenset(
                {
                    "distinct_skills",
                    "entropy",
                    "total_invocations",
                    "window_end_step",
                }
            ),
            record="EntropyObservation",
        )
        return cls(
            window_end_step=_read_int(data["window_end_step"], field="window_end_step"),
            entropy=_read_float(data["entropy"], field="entropy"),
            total_invocations=_read_int(data["total_invocations"], field="total_invocations"),
            distinct_skills=_read_int(data["distinct_skills"], field="distinct_skills"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


@dataclass(frozen=True, slots=True)
class PhaseTransitionEvent:
    """A true phase-transition trigger together with independently replayable evidence."""

    event_id: str
    triggered_at_step: int
    library_version: str
    previous_window: WindowStats
    current_window: WindowStats
    relative_improvement: float
    rho: float
    residual_condition_met: bool
    entropy_series: tuple[EntropyObservation, ...]
    trigger_rule: PhaseTriggerRule = field(
        default=PhaseTriggerRule.RESIDUAL_AND_ENTROPY,
        kw_only=True,
    )
    format: str = field(default=PHASE_TRANSITION_FORMAT, kw_only=True)
    required_consecutive_drops: int = field(default=2, kw_only=True)
    entropy_condition_met: bool
    triggered: bool

    def __post_init__(self) -> None:
        location = f"phase_transition[{self.event_id or '<missing>'}]"
        require_non_empty_text(self.event_id, field="event_id", location=location)
        require_non_empty_text(
            self.library_version,
            field="library_version",
            location=location,
        )
        if self.format != PHASE_TRANSITION_FORMAT:
            raise ValueError(f"{location}.format must be {PHASE_TRANSITION_FORMAT!r}")
        if type(self.triggered_at_step) is not int or self.triggered_at_step < 0:
            raise ValueError(f"{location}.triggered_at_step must be a non-negative integer")
        if not isinstance(self.previous_window, WindowStats) or not isinstance(
            self.current_window, WindowStats
        ):
            raise ValueError(f"{location}.previous_window/current_window must be WindowStats")
        if self.previous_window.end_optimizer_step >= self.current_window.start_optimizer_step:
            raise ValueError(f"{location} windows must be ordered and non-overlapping")

        previous_msr = self.previous_window.mean_squared_residual
        if previous_msr <= 0:
            raise ValueError(f"{location}.previous_window mean_squared_residual must be positive")
        relative_improvement = require_finite_number(
            self.relative_improvement,
            field="relative_improvement",
            location=location,
        )
        rho = require_finite_number(self.rho, field="rho", location=location)
        if rho <= 0:
            raise ValueError(f"{location}.rho must be positive")
        object.__setattr__(self, "relative_improvement", relative_improvement)
        object.__setattr__(self, "rho", rho)
        expected_improvement = (
            previous_msr - self.current_window.mean_squared_residual
        ) / previous_msr
        if not math.isclose(
            relative_improvement,
            expected_improvement,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise ValueError(f"{location}.relative_improvement does not match window evidence")

        if type(self.residual_condition_met) is not bool:
            raise ValueError(f"{location}.residual_condition_met must be a boolean")
        expected_residual_condition = relative_improvement < rho
        if self.residual_condition_met is not expected_residual_condition:
            raise ValueError(f"{location}.residual_condition_met does not match the evidence")

        if type(self.required_consecutive_drops) is not int or self.required_consecutive_drops < 1:
            raise ValueError(f"{location}.required_consecutive_drops must be a positive integer")
        if not isinstance(self.trigger_rule, PhaseTriggerRule):
            raise ValueError(f"{location}.trigger_rule must be a PhaseTriggerRule")
        if not isinstance(self.entropy_series, tuple) or any(
            not isinstance(item, EntropyObservation) for item in self.entropy_series
        ):
            raise ValueError(
                f"{location}.entropy_series must be a tuple of EntropyObservation records"
            )
        required_points = self.required_consecutive_drops + 1
        if (
            self.trigger_rule is PhaseTriggerRule.RESIDUAL_AND_ENTROPY
            and len(self.entropy_series) < required_points
        ):
            raise ValueError(f"{location}.entropy_series has too few observations")
        end_steps = [observation.window_end_step for observation in self.entropy_series]
        if any(current <= previous for previous, current in pairwise(end_steps)):
            raise ValueError(f"{location}.entropy_series must be strictly step-ordered")
        expected_entropy_condition = len(self.entropy_series) >= required_points and all(
            current.entropy < previous.entropy
            for previous, current in pairwise(self.entropy_series[-required_points:])
        )
        if type(self.entropy_condition_met) is not bool:
            raise ValueError(f"{location}.entropy_condition_met must be a boolean")
        if self.entropy_condition_met is not expected_entropy_condition:
            raise ValueError(f"{location}.entropy_condition_met does not match the evidence")

        if type(self.triggered) is not bool:
            raise ValueError(f"{location}.triggered must be a boolean")
        entropy_required = self.trigger_rule is PhaseTriggerRule.RESIDUAL_AND_ENTROPY
        expected_triggered = expected_residual_condition and (
            expected_entropy_condition if entropy_required else True
        )
        if self.triggered is not expected_triggered:
            raise ValueError(f"{location}.triggered does not match trigger_rule evidence")
        if not self.triggered:
            raise ValueError(f"{location} only records true phase transitions")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "current_window": self.current_window.to_value(),
            "entropy_condition_met": self.entropy_condition_met,
            "entropy_series": [item.to_value() for item in self.entropy_series],
            "event_id": self.event_id,
            "format": self.format,
            "library_version": self.library_version,
            "previous_window": self.previous_window.to_value(),
            "relative_improvement": self.relative_improvement,
            "required_consecutive_drops": self.required_consecutive_drops,
            "residual_condition_met": self.residual_condition_met,
            "rho": self.rho,
            "trigger_rule": self.trigger_rule.value,
            "triggered": self.triggered,
            "triggered_at_step": self.triggered_at_step,
        }

    @classmethod
    def from_value(cls, value: object) -> PhaseTransitionEvent:
        data = _require_json_object(value, record="PhaseTransitionEvent")
        _require_exact_fields(
            data,
            fields=frozenset(
                {
                    "current_window",
                    "entropy_condition_met",
                    "entropy_series",
                    "event_id",
                    "format",
                    "library_version",
                    "previous_window",
                    "relative_improvement",
                    "required_consecutive_drops",
                    "residual_condition_met",
                    "rho",
                    "trigger_rule",
                    "triggered",
                    "triggered_at_step",
                }
            ),
            record="PhaseTransitionEvent",
        )
        raw_entropy = _require_json_list(data["entropy_series"], field="entropy_series")
        return cls(
            event_id=_read_text(data["event_id"], field="event_id"),
            format=_read_text(data["format"], field="format"),
            triggered_at_step=_read_int(data["triggered_at_step"], field="triggered_at_step"),
            library_version=_read_text(data["library_version"], field="library_version"),
            previous_window=WindowStats.from_value(data["previous_window"]),
            current_window=WindowStats.from_value(data["current_window"]),
            relative_improvement=_read_float(
                data["relative_improvement"], field="relative_improvement"
            ),
            rho=_read_float(data["rho"], field="rho"),
            residual_condition_met=_read_bool(
                data["residual_condition_met"], field="residual_condition_met"
            ),
            entropy_series=tuple(EntropyObservation.from_value(item) for item in raw_entropy),
            required_consecutive_drops=_read_int(
                data["required_consecutive_drops"],
                field="required_consecutive_drops",
            ),
            entropy_condition_met=_read_bool(
                data["entropy_condition_met"], field="entropy_condition_met"
            ),
            trigger_rule=PhaseTriggerRule(_read_text(data["trigger_rule"], field="trigger_rule")),
            triggered=_read_bool(data["triggered"], field="triggered"),
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())
