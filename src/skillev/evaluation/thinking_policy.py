"""Frozen per-benchmark native-thinking condition, resolved before actor creation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .input_metric_contracts import (
    HISTORICAL_IID_BENCHMARKS,
    HISTORICAL_OOD_BENCHMARKS,
    IID_BENCHMARKS,
    OOD_BENCHMARKS,
)
from .step0_integrity import InferenceArm


@dataclass(frozen=True, slots=True)
class ThinkingPolicy:
    policy_id: str
    rows: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        # Frozen older maps remain readable without changing current panel membership.
        current = set(IID_BENCHMARKS)
        historical = set(HISTORICAL_IID_BENCHMARKS)
        legacy = historical | {"webshop"}
        ood = set(OOD_BENCHMARKS)
        old_ood = set(HISTORICAL_OOD_BENCHMARKS)
        legacy_ood = (old_ood - {"livemedbench"}) | {"gpqa-diamond-health"}
        bioorganic_ood = (old_ood - {"livemedbench"}) | {"gpqa-diamond-bioorganic"}
        named = {name for name, _ in self.rows}
        if (
            not self.policy_id
            or len(self.rows) != len(named)
            or named not in (current, historical, legacy, ood, old_ood, legacy_ood, bioorganic_ood)
        ):
            raise ValueError("thinking policy must identify each current IID benchmark once")
        if any(type(value) is not bool for _, value in self.rows):
            raise TypeError("native thinking values must be booleans")

    @classmethod
    def default(cls, *, ood: bool = False) -> ThinkingPolicy:
        """Current owner-selected defaults; explicit historic conditions are unchanged."""
        catalog = OOD_BENCHMARKS if ood else IID_BENCHMARKS
        if ood:
            return cls(
                "six-ood-math-gpqa-thinking-on@1",
                tuple((name, name in {"math-hard", "gpqa-diamond-bioorganic"}) for name in catalog),
            )
        return cls(
            "six-iid-aime-health-thinking-on@1",
            tuple((name, name in {"aime-2026", "healthbench"}) for name in catalog),
        )

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> ThinkingPolicy:
        if set(value) != {"policy_id", "thinking_by_benchmark"}:
            raise ValueError("thinking policy configuration has missing or unknown fields")
        return cls(value["policy_id"], tuple(value["thinking_by_benchmark"].items()))

    def resolve(self, benchmark: str, arm: InferenceArm) -> InferenceArm:
        return replace(arm, native_thinking=dict(self.rows)[benchmark])

    def to_value(self) -> dict[str, object]:
        return {"policy_id": self.policy_id, "thinking_by_benchmark": dict(self.rows)}
