"""Protocol 10 SpreadsheetBench workspace and official-OJ session binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from skillev.contracts import JsonValue
from skillev.experiments.protocol_v10 import BenchmarkProtocolV10, BenchmarkV10
from skillev.rollout import RolloutSessionBundle, RolloutTask
from skillev.runtime import RolloutEnvironmentSession

from .catalog import PrivateSessionFactory
from .protocol_v10_evaluator import ProtocolV10TerminalEvaluator
from .protocol_v10_official import SpreadsheetBenchNativeBackend, SpreadsheetBenchOfficialOJ
from .protocol_v10_population import PrivateBenchmarkPopulation
from .protocol_v10_sessions import ProtocolV10PrivateRecord


class SpreadsheetBenchWorkspace(RolloutEnvironmentSession, SpreadsheetBenchOfficialOJ, Protocol):
    """One isolated editable workbook and its private official-OJ view."""

    async def close(self) -> None: ...


class SpreadsheetBenchWorkspaceFactory(Protocol):
    """Deployment boundary allowed to open private workbook assets."""

    def create(
        self,
        task: RolloutTask,
        private_payload: dict[str, JsonValue],
    ) -> SpreadsheetBenchWorkspace: ...


@dataclass(frozen=True, slots=True)
class _SpreadsheetRoute:
    task: RolloutTask
    private_payload: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _SpreadsheetSessionFactory:
    routes: tuple[_SpreadsheetRoute, ...]
    protocol: BenchmarkProtocolV10
    deployment: SpreadsheetBenchWorkspaceFactory

    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        matches = tuple(route for route in self.routes if route.task.task_id == task.task_id)
        if len(matches) != 1 or matches[0].task != task:
            raise ValueError("SpreadsheetBench task has no exact private workspace route")
        workspace = self.deployment.create(task, matches[0].private_payload)
        if workspace.environment_id != task.environment_id:
            raise ValueError("SpreadsheetBench workspace environment identity differs")
        if workspace.task_family != task.task_family:
            raise ValueError("SpreadsheetBench workspace task family differs")
        return RolloutSessionBundle(
            environment=workspace,
            evaluator=ProtocolV10TerminalEvaluator(
                self.protocol,
                SpreadsheetBenchNativeBackend(task.task_id, task.environment_id, workspace),
            ),
            retrieved_skills=(),
            cleanup=workspace.close,
        )


@dataclass(frozen=True, slots=True)
class ProtocolV10SpreadsheetSessionBuilder:
    """Bind every SpreadsheetBench population to an isolated workspace."""

    deployment: SpreadsheetBenchWorkspaceFactory

    def __post_init__(self) -> None:
        if not callable(getattr(self.deployment, "create", None)):
            raise TypeError("SpreadsheetBench builder requires a workspace factory")

    def build(
        self,
        protocol: BenchmarkProtocolV10,
        population: PrivateBenchmarkPopulation,
        records: tuple[ProtocolV10PrivateRecord, ...],
    ) -> PrivateSessionFactory:
        if protocol.benchmark is not BenchmarkV10.SPREADSHEETBENCH:
            raise ValueError("SpreadsheetBench builder received another benchmark")
        by_source = {record.source_id: record for record in records}
        routes = []
        for item in population.items:
            payload = by_source[item.source_id].private_payload
            if not isinstance(payload, dict):
                raise ValueError("SpreadsheetBench private workspace route is incompatible")
            routes.append(_SpreadsheetRoute(item.task, payload))
        return _SpreadsheetSessionFactory(tuple(routes), protocol, self.deployment)


__all__ = [
    "ProtocolV10SpreadsheetSessionBuilder",
    "SpreadsheetBenchWorkspace",
    "SpreadsheetBenchWorkspaceFactory",
]
