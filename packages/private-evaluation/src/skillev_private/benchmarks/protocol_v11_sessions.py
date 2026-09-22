"""Exact task-to-session routing for Protocol 11."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from skillev.experiments.protocol_v11 import BenchmarkV11
from skillev.rollout import RolloutSessionBundle, RolloutTask


class ProtocolV11SessionBuilder(Protocol):
    def build(self, task: RolloutTask) -> RolloutSessionBundle: ...


@dataclass(frozen=True, slots=True)
class ProtocolV11SessionRegistry:
    builders: dict[BenchmarkV11, ProtocolV11SessionBuilder]

    def __post_init__(self) -> None:
        if set(self.builders) != set(BenchmarkV11):
            raise ValueError("Protocol 11 session registry must bind exactly ten builders")

    def create(self, benchmark: BenchmarkV11, task: RolloutTask) -> RolloutSessionBundle:
        return self.builders[benchmark].build(task)


__all__ = ["ProtocolV11SessionBuilder", "ProtocolV11SessionRegistry"]
