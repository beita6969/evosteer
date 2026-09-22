"""Protocol 11 evaluator registry bound to TerminalReward contracts."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.experiments.protocol_v11 import BenchmarkV11
from skillev.rollout.environment import TerminalEvaluator


@dataclass(frozen=True, slots=True)
class ProtocolV11EvaluatorRegistry:
    evaluators: dict[BenchmarkV11, TerminalEvaluator]

    def __post_init__(self) -> None:
        if set(self.evaluators) != set(BenchmarkV11):
            raise ValueError("Protocol 11 evaluator registry must bind exactly ten evaluators")

    def for_benchmark(self, benchmark: BenchmarkV11) -> TerminalEvaluator:
        return self.evaluators[benchmark]


__all__ = ["ProtocolV11EvaluatorRegistry"]
