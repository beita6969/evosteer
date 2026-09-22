"""Explicit, evidence-bound risk decisions for the EvoSteer batch gate.

Assessors are trusted environment code, never an LLM judge. The paper does not
specify a universal risk detector; the required decision and its provenance are
therefore explicit, while concrete environments enforce their own capabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrajectoryRiskAssessment:
    assessor_id: str
    evidence_id: str
    accepted: bool
    side_effect_free: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("assessor_id", "evidence_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"risk {name} must be nonempty")
        if type(self.accepted) is not bool or type(self.side_effect_free) is not bool:
            raise ValueError("risk flags require explicit booleans")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(reason, str) or not reason.strip() for reason in self.reasons
        ):
            raise ValueError("risk reasons must be immutable nonempty strings")
        if not self.accepted and not self.reasons:
            raise ValueError("rejected evidence must record a reason")

    def to_value(self) -> dict[str, Any]:
        return {**asdict(self), "reasons": list(self.reasons)}

    @classmethod
    def from_value(cls, value: object) -> TrajectoryRiskAssessment:
        if not isinstance(value, dict) or set(value) != {
            "assessor_id",
            "evidence_id",
            "accepted",
            "side_effect_free",
            "reasons",
        }:
            raise ValueError("incompatible trajectory risk record")
        if not isinstance(value["reasons"], list):
            raise ValueError("risk reasons must be a JSON array")
        return cls(
            value["assessor_id"],
            value["evidence_id"],
            value["accepted"],
            value["side_effect_free"],
            tuple(value["reasons"]),
        )
