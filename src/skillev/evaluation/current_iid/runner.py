"""Uniform public runner interface for the nine heterogeneous evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .catalog import CurrentIIDBenchmark
from .conditions import ExecutionContract
from .receipts import Protocol12RunReceipt


@dataclass(frozen=True, slots=True)
class RunnerPreflightReceipt:
    benchmark: CurrentIIDBenchmark
    condition_id: str
    population_ready: bool
    model_ready: bool
    environment_ready: bool
    scorer_ready: bool
    grader_ready: bool
    message: str

    @property
    def passed(self) -> bool:
        return all(
            (
                self.population_ready,
                self.model_ready,
                self.environment_ready,
                self.scorer_ready,
                self.grader_ready,
            )
        )


class CurrentIIDBenchmarkRunner(Protocol):
    benchmark: CurrentIIDBenchmark

    async def preflight(self, contract: ExecutionContract) -> RunnerPreflightReceipt: ...

    async def run(self, contract: ExecutionContract) -> Protocol12RunReceipt: ...


__all__ = ["CurrentIIDBenchmarkRunner", "RunnerPreflightReceipt"]
