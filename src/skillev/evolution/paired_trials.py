"""Frozen reference-policy skill interventions and complete paired outcomes.

These records describe an initial-condition intervention, not a clone of a live
agent. Execution must use an isolated, resettable environment for each arm. The
records validate declared execution identities; they cannot attest that an
external executor actually honored them.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ComparisonContext:
    """Conditions held fixed for every observation of one registered comparison."""

    task_family: str
    reference_snapshot_id: str
    executor_snapshot_id: str
    background_menu_id: str
    value_snapshot_id: str
    environment_config_id: str

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            _text(value, field)

    def to_value(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> ComparisonContext:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TrialInitialCondition:
    task_id: str
    reset_id: str
    first_role: str
    comparison_context: ComparisonContext

    def __post_init__(self) -> None:
        for field in ("task_id", "reset_id", "first_role"):
            _text(getattr(self, field), field)
        if not isinstance(self.comparison_context, ComparisonContext):
            raise TypeError("initial condition requires a ComparisonContext")

    def to_value(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> TrialInitialCondition:
        return cls(
            task_id=value["task_id"],
            reset_id=value["reset_id"],
            first_role=value["first_role"],
            comparison_context=ComparisonContext.from_value(value["comparison_context"]),
        )


@dataclass(frozen=True, slots=True)
class TrialArmSpec:
    pair_id: str
    comparison_id: str
    skill_id: str
    initial_condition: TrialInitialCondition
    arm: Literal["positive", "negative"]

    def __post_init__(self) -> None:
        for field in ("pair_id", "comparison_id", "skill_id"):
            _text(getattr(self, field), field)
        if not isinstance(self.initial_condition, TrialInitialCondition):
            raise TypeError("trial arm requires a TrialInitialCondition")
        if self.arm not in ("positive", "negative"):
            raise ValueError("trial arm must be positive or negative")

    @property
    def bound_skill_ids(self) -> tuple[str, ...]:
        return (self.skill_id,) if self.arm == "positive" else ()

    @property
    def excluded_skill_ids(self) -> tuple[str, ...]:
        """Executor must enforce this exclusion for the entire control rollout."""
        return (self.skill_id,) if self.arm == "negative" else ()

    @property
    def continuation_reference_id(self) -> str:
        return self.initial_condition.comparison_context.reference_snapshot_id

    @property
    def first_role(self) -> str:
        return self.initial_condition.first_role

    def to_value(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> TrialArmSpec:
        return cls(
            pair_id=value["pair_id"],
            comparison_id=value["comparison_id"],
            skill_id=value["skill_id"],
            initial_condition=TrialInitialCondition.from_value(value["initial_condition"]),
            arm=value["arm"],
        )


@dataclass(frozen=True, slots=True)
class PairedTrialSpec:
    positive: TrialArmSpec
    negative: TrialArmSpec

    def __post_init__(self) -> None:
        _validate_arms(self.positive, self.negative)

    @property
    def pair_id(self) -> str:
        return self.positive.pair_id


class PairScheduler:
    """Construct specifications only; never performs model or environment calls."""

    @staticmethod
    def plan(
        comparison_id: str,
        skill_id: str,
        initial: TrialInitialCondition,
        pair_id: str | None = None,
    ) -> PairedTrialSpec:
        # A repeated initial condition is the same pair by default. An explicit
        # pair ID distinguishes real replicates; it does not make them independent.
        if pair_id is not None:
            _text(pair_id, "pair_id")
        identity = (
            pair_id
            if pair_id is not None
            else _hash(
                {
                    "comparison_id": comparison_id,
                    "skill_id": skill_id,
                    "initial": initial.to_value(),
                }
            )
        )
        return PairedTrialSpec(
            positive=TrialArmSpec(identity, comparison_id, skill_id, initial, "positive"),
            negative=TrialArmSpec(identity, comparison_id, skill_id, initial, "negative"),
        )


def _validate_arms(positive: TrialArmSpec, negative: TrialArmSpec) -> None:
    if not isinstance(positive, TrialArmSpec) or not isinstance(negative, TrialArmSpec):
        raise TypeError("paired trials require two typed arms")
    if positive.arm != "positive" or negative.arm != "negative":
        raise ValueError("pair must contain one positive and one negative arm")
    for field in ("pair_id", "comparison_id", "skill_id", "initial_condition"):
        if getattr(positive, field) != getattr(negative, field):
            raise ValueError(f"paired trial arms disagree on {field}")


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """An execution report; observed identities may be supplied independently.

    If omitted, the caller attests execution honored the specification. A runner
    with independent execution receipts should pass ``observed_initial_condition``.
    A failed/incomplete arm may have no reward and never contributes evidence.
    Safety is never inferred from successful completion: side_effect_free must
    be explicitly supplied from the runner's verified risk assessment.
    """

    spec: TrialArmSpec
    reward: float | None
    completed: bool = True
    side_effect_free: bool = False
    observed_initial_condition: TrialInitialCondition | None = None
    control_exclusion_honored: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TrialArmSpec):
            raise TypeError("outcome requires its exact TrialArmSpec")
        for field in ("completed", "side_effect_free", "control_exclusion_honored"):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be boolean")
        if self.reward is not None and (
            isinstance(self.reward, bool)
            or not isinstance(self.reward, int | float)
            or not math.isfinite(self.reward)
            or not 0.0 <= self.reward <= 1.0
        ):
            raise ValueError("trial reward must be finite and in [0, 1]")
        if self.completed and self.reward is None:
            raise ValueError("a completed outcome requires its terminal reward")
        if self.observed_initial_condition is not None:
            if self.observed_initial_condition != self.spec.initial_condition:
                raise ValueError("observed execution identity differs from the frozen trial")
        if self.spec.arm == "negative" and not self.control_exclusion_honored:
            raise ValueError("candidate exclusion was not honored throughout the control")

    def to_value(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> TrialOutcome:
        observed = value["observed_initial_condition"]
        return cls(
            spec=TrialArmSpec.from_value(value["spec"]),
            reward=value["reward"],
            completed=value["completed"],
            side_effect_free=value["side_effect_free"],
            observed_initial_condition=(
                None if observed is None else TrialInitialCondition.from_value(observed)
            ),
            control_exclusion_honored=value["control_exclusion_honored"],
        )


@dataclass(frozen=True, slots=True)
class PairedTrialOutcome:
    positive: TrialOutcome
    negative: TrialOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.positive, TrialOutcome) or not isinstance(
            self.negative, TrialOutcome
        ):
            raise TypeError("paired outcome requires typed execution reports")
        _validate_arms(self.positive.spec, self.negative.spec)

    @property
    def pair_id(self) -> str:
        return self.positive.spec.pair_id

    @property
    def eligible(self) -> bool:
        return all(
            outcome.completed and outcome.side_effect_free
            for outcome in (self.positive, self.negative)
        )

    def to_value(self) -> dict[str, Any]:
        return {"positive": self.positive.to_value(), "negative": self.negative.to_value()}

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> PairedTrialOutcome:
        return cls(
            TrialOutcome.from_value(value["positive"]),
            TrialOutcome.from_value(value["negative"]),
        )


__all__ = [
    "ComparisonContext",
    "PairScheduler",
    "PairedTrialOutcome",
    "PairedTrialSpec",
    "TrialArmSpec",
    "TrialInitialCondition",
    "TrialOutcome",
]
