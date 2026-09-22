"""Benchmark-native outcome separated from the scalar training projection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class NativeEvaluation:
    """One definitive or infrastructure-failed benchmark outcome.

    Headline metrics remain benchmark native. ``training_value`` is only the
    explicit scalar projection consumed by ``TerminalReward`` and must never
    replace the native publication metrics.
    """

    benchmark: str
    task_id: str
    metrics: Mapping[str, float]
    training_value: float
    success: bool | None
    candidate_valid: bool
    infrastructure_ok: bool

    def __post_init__(self) -> None:
        if not self.benchmark.strip() or not self.task_id.strip():
            raise ValueError("native evaluation identity is incomplete")
        normalized: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not name.strip() or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError("native evaluation metrics must be named finite values")
            normalized[name] = float(value)
        if isinstance(self.training_value, bool) or not math.isfinite(float(self.training_value)):
            raise ValueError("training value must be finite")
        if self.success is not None and type(self.success) is not bool:
            raise TypeError("success must be boolean or null")
        if type(self.candidate_valid) is not bool or type(self.infrastructure_ok) is not bool:
            raise TypeError("evaluation validity flags must be boolean")
        if not self.infrastructure_ok and self.candidate_valid:
            raise ValueError("an unresolved infrastructure failure cannot admit a candidate")
        object.__setattr__(self, "metrics", MappingProxyType(normalized))
        object.__setattr__(self, "training_value", float(self.training_value))


__all__ = ["NativeEvaluation"]
