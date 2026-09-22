from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from skillev_private.benchmarks.protocol_v10_materialization import (
    ProtocolV10SourceRecord,
    materialize_protocol_v10_population,
)
from skillev_private.benchmarks.protocol_v10_official import SpreadsheetBenchGrade
from skillev_private.benchmarks.protocol_v10_sessions import ProtocolV10PrivateRecord
from skillev_private.benchmarks.protocol_v10_spreadsheet import (
    ProtocolV10SpreadsheetSessionBuilder,
)

from skillev.contracts import JsonValue, stable_hash
from skillev.experiments import BenchmarkV10, PopulationRole, load_active_protocol_v10
from skillev.rollout import (
    RolloutTask,
    RolloutTermination,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
)
from skillev.runtime import BudgetVector, EnvironmentObservation, StructuredAction

ROOT = Path(__file__).parents[2]


@dataclass(slots=True)
class _Workspace:
    task: RolloutTask
    payload: dict[str, JsonValue]
    closed: bool = False
    verifier_version: str = "spreadsheetbench-official-oj@fixture"

    @property
    def environment_id(self) -> str:
        return self.task.environment_id

    @property
    def task_family(self) -> str:
        return self.task.task_family

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        del action, step_index
        return EnvironmentObservation(
            public_value={"status": "workbook-updated"},
            observation_status="success",
            budget_usage=BudgetVector(tool_calls=1),
        )

    def validate_completion(self, submission: JsonValue) -> bool:
        return submission == {"workspace_id": self.task.task_id}

    async def grade(
        self,
        task_id: str,
        submitted_workbook: JsonValue,
    ) -> SpreadsheetBenchGrade:
        assert task_id == self.task.task_id
        assert submitted_workbook == {"workspace_id": self.task.task_id}
        assert self.payload["spreadsheet_relative_path"] == "private/input.xlsx"
        return SpreadsheetBenchGrade(4, 4)

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _WorkspaceFactory:
    workspace: _Workspace | None = None

    def create(
        self,
        task: RolloutTask,
        private_payload: dict[str, JsonValue],
    ) -> _Workspace:
        self.workspace = _Workspace(task, private_payload)
        return self.workspace


def test_spreadsheet_session_keeps_workbook_route_private_and_uses_official_oj() -> None:
    active = load_active_protocol_v10(ROOT / "configs/evaluation/protocol_v10.yaml")
    protocol = active.benchmark(BenchmarkV10.SPREADSHEETBENCH)
    spec = protocol.population(PopulationRole.FINAL_EVALUATION)[0]
    task = RolloutTask(
        task_id="spreadsheetbench/task-1",
        environment_id="benchmark:spreadsheetbench@fixture:official-oj",
        task_family="spreadsheetbench/spreadsheet-editing",
        context_id="spreadsheetbench:final",
        query="Update the workbook using the public instructions.",
        available_tools=("spreadsheet.execute", "submit"),
        public_context={
            "benchmark_id": "spreadsheetbench",
            "population_id": spec.population_id,
            "source_version": spec.source_version,
            "workbook_structure_id": "spreadsheetbench/public-workbook",
        },
    )
    source = ProtocolV10SourceRecord(
        task.task_id,
        task,
        {
            "answer_position": "private",
            "answer_sheet": "private",
            "data_position": "private",
            "spreadsheet_relative_path": "private/input.xlsx",
        },
    )
    population = materialize_protocol_v10_population(spec, (source,)).population
    deployment = _WorkspaceFactory()
    factory = ProtocolV10SpreadsheetSessionBuilder(deployment).build(
        protocol,
        population,
        (ProtocolV10PrivateRecord(source.source_id, source.private_payload),),
    )

    bundle = factory.create(task)
    assert "private/input.xlsx" not in task.query
    reward = asyncio.run(
        bundle.evaluator.evaluate(
            TerminalEvaluationRequest(
                trajectory_id="trajectory",
                task_id=task.task_id,
                termination=RolloutTermination.COMPLETED,
                evaluation_input=SubmittedTerminalValue({"workspace_id": task.task_id}),
                public_transcript_hash=stable_hash({"transcript": "public"}),
            )
        )
    )

    assert reward.value == 1.0
    assert reward.success is True
    assert bundle.cleanup is not None
    asyncio.run(bundle.cleanup())
    assert deployment.workspace is not None
    assert deployment.workspace.closed
