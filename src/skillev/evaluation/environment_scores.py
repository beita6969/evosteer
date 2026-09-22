"""Native terminal measurements and learning projections are different values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NativeEnvironmentScore:
    raw_score: float
    termination_reason: str
    unit: str = "scienceworld-score/100"

    def __post_init__(self) -> None:
        if (
            type(self.raw_score) not in (int, float)
            or not math.isfinite(self.raw_score)
            or self.raw_score > 100
        ):
            raise ValueError("native ScienceWorld score must be finite and at most 100")
        if not self.termination_reason or self.unit != "scienceworld-score/100":
            raise ValueError("native score requires its termination and unit identity")

    @property
    def value(self) -> float:
        return self.raw_score / 100

    @classmethod
    def from_outcome(cls, outcome: Mapping[str, object]) -> NativeEnvironmentScore:
        raw = outcome.get("native_final_score")
        if not isinstance(raw, int | float) or isinstance(raw, bool):
            # Do not infer a native zero from a bounded reward or a done flag.
            raise RuntimeError("native ScienceWorld score evidence is unavailable")
        reason = outcome.get("termination_reason")
        if not isinstance(reason, str) or not reason:
            raise RuntimeError("native ScienceWorld termination evidence is unavailable")
        return cls(float(raw), reason)


@dataclass(frozen=True, slots=True)
class LearningRewardProjection:
    value: float
    success: bool
    projection: str = "scienceworld-zero-clipped-final-score@1"

    @classmethod
    def from_native(cls, native: NativeEnvironmentScore) -> LearningRewardProjection:
        return cls(max(0.0, native.value), native.raw_score == 100)
