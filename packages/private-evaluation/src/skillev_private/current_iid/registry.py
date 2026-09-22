"""Exact-set registry for Protocol 12 private runners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from skillev.evaluation.current_iid.catalog import (
    ACTIVE_CURRENT_IID_BENCHMARKS,
    CurrentIIDBenchmark,
)
from skillev.evaluation.current_iid.runner import CurrentIIDBenchmarkRunner


@dataclass(frozen=True, slots=True)
class CurrentIIDRunnerRegistry:
    runners: Mapping[CurrentIIDBenchmark, CurrentIIDBenchmarkRunner]

    def __post_init__(self) -> None:
        expected = set(ACTIVE_CURRENT_IID_BENCHMARKS)
        actual = set(self.runners)
        if actual != expected:
            missing = sorted(item.value for item in expected - actual)
            extra = sorted(item.value for item in actual - expected)
            raise ValueError(
                f"current-IID runner registry mismatch: missing={missing}, extra={extra}"
            )
        for benchmark, runner in self.runners.items():
            if runner.benchmark is not benchmark:
                raise ValueError("runner registry key differs from runner benchmark")

    def for_benchmark(self, benchmark: CurrentIIDBenchmark) -> CurrentIIDBenchmarkRunner:
        return self.runners[benchmark]


__all__ = ["CurrentIIDRunnerRegistry"]
