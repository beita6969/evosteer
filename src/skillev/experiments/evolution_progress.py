"""Answer-free source counts for one exact training evolution state."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.contracts import JsonValue, normalize_json


def _count(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class TrainingEvolutionCounts:
    """Exact source counts attached to an immutable evaluation state.

    These counts describe the completed training prefix that produced a
    frozen state.  They are not evaluation telemetry and never include work
    performed by the read-only evaluation itself.
    """

    training_step_count: int
    phase_count: int
    cycle_count: int
    action_count: int

    def __post_init__(self) -> None:
        for field in (
            "training_step_count",
            "phase_count",
            "cycle_count",
            "action_count",
        ):
            _count(getattr(self, field), field=field)
        if self.cycle_count > self.phase_count:
            raise ValueError("evolution cycle_count cannot exceed phase_count")
        if self.action_count < self.cycle_count:
            raise ValueError("evolution action_count cannot be below cycle_count")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "action_count": self.action_count,
            "cycle_count": self.cycle_count,
            "phase_count": self.phase_count,
            "training_step_count": self.training_step_count,
        }

    @classmethod
    def from_value(cls, value: object) -> TrainingEvolutionCounts:
        data = normalize_json(value)
        fields = {
            "action_count",
            "cycle_count",
            "phase_count",
            "training_step_count",
        }
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("training evolution counts have incompatible fields")
        return cls(
            training_step_count=_count(data["training_step_count"], field="training_step_count"),
            phase_count=_count(data["phase_count"], field="phase_count"),
            cycle_count=_count(data["cycle_count"], field="cycle_count"),
            action_count=_count(data["action_count"], field="action_count"),
        )


__all__ = ["TrainingEvolutionCounts"]
