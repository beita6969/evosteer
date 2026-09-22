from __future__ import annotations

import asyncio
import stat
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from skillev_private.benchmarks.protocol_v10_official import SpreadsheetBenchGrade
from skillev_private.benchmarks.protocol_v10_spreadsheet_runtime import (
    IsolatedSpreadsheetWorkspaceFactory,
    ProtocolV10SpreadsheetAssetResolver,
    SpreadsheetExecutionResult,
    SpreadsheetWorkbookAssets,
)
from skillev_private.benchmarks.protocol_v10_spreadsheet_worker import _verified_oracle
from skillev_private.benchmarks.protocol_v10_workers import (
    SpreadsheetBenchProcessDeployment,
)

from skillev.contracts import JsonValue
from skillev.rollout import RolloutTask
from skillev.runtime import ActionKind, StructuredAction


@dataclass(frozen=True, slots=True)
class _Assets:
    workbook: Path

    def resolve(
        self,
        task: RolloutTask,
        private_payload: dict[str, JsonValue],
    ) -> SpreadsheetWorkbookAssets:
        assert private_payload == {"route": "private"}
        assert task.task_id == "sheet/task-1"
        return SpreadsheetWorkbookAssets(self.workbook)


@dataclass(slots=True)
class _Sandbox:
    calls: list[tuple[Path, str]] = field(default_factory=list)

    async def execute(self, workspace: Path, code: str) -> SpreadsheetExecutionResult:
        self.calls.append((workspace, code))
        (workspace / "input.xlsx").write_bytes(b"edited")
        return SpreadsheetExecutionResult(0, "saved", "")


@dataclass(slots=True)
class _OJ:
    verifier_version: str = "spreadsheetbench-official-oj@fixture"
    submissions: list[JsonValue] = field(default_factory=list)

    async def grade(self, task_id: str, submitted_workbook: JsonValue) -> SpreadsheetBenchGrade:
        assert task_id == "sheet/task-1"
        self.submissions.append(submitted_workbook)
        return SpreadsheetBenchGrade(3, 3)


def _task() -> RolloutTask:
    return RolloutTask(
        task_id="sheet/task-1",
        environment_id="benchmark:spreadsheetbench@fixture:official-oj",
        task_family="spreadsheetbench/spreadsheet-editing",
        context_id="spreadsheetbench:final",
        query="Update the workbook.",
        available_tools=("spreadsheet.execute", "submit"),
        public_context={
            "tool_schema": {
                "spreadsheet.execute": {"arguments": {"code": "Python code editing WORKBOOK_PATH"}}
            },
            "workbook_structure_id": "public/workbook",
        },
    )


def test_isolated_workspace_copies_only_input_executes_and_cleans_up(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"original")
    root = tmp_path / "workspaces"
    root.mkdir()
    sandbox = _Sandbox()
    oj = _OJ()
    workspace = IsolatedSpreadsheetWorkspaceFactory(
        workspace_root=root.resolve(),
        assets=_Assets(source.resolve()),
        sandbox=sandbox,
        oj=oj,
    ).create(_task(), {"route": "private"})

    observation = asyncio.run(
        workspace.execute(
            StructuredAction(
                kind=ActionKind.TOOL,
                name="execute",
                arguments={"code": "print(WORKBOOK_PATH)"},
                resource_id="spreadsheet",
            ),
            step_index=1,
        )
    )
    assert observation.observation_status == "success"
    assert observation.public_value["stdout"] == "saved"
    assert sandbox.calls[0][1] == "print(WORKBOOK_PATH)"

    submission = {"workspace_id": _task().task_id}
    assert workspace.validate_completion(submission)
    assert asyncio.run(workspace.grade(_task().task_id, submission)).all_cases_pass
    assert oj.submissions[0]["workspace_id"] == _task().task_id
    assert oj.submissions[0]["private_payload"] == {"route": "private"}
    workspace_directory = workspace.directory
    asyncio.run(workspace.close())
    assert not workspace_directory.exists()


def test_isolated_workspace_rejects_unregistered_tool_without_running_code(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"original")
    root = tmp_path / "workspaces"
    root.mkdir()
    sandbox = _Sandbox()
    workspace = IsolatedSpreadsheetWorkspaceFactory(
        workspace_root=root.resolve(),
        assets=_Assets(source.resolve()),
        sandbox=sandbox,
        oj=_OJ(),
    ).create(_task(), {"route": "private"})

    observation = asyncio.run(
        workspace.execute(
            StructuredAction(
                kind=ActionKind.TOOL,
                name="execute",
                arguments={"code": "print('no')"},
                resource_id="another-resource",
            ),
            step_index=1,
        )
    )
    assert observation.observation_status == "tool_error"
    assert sandbox.calls == []
    asyncio.run(workspace.close())


def test_asset_resolver_keeps_training_and_verified_inputs_separate(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "training.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("domain/task-1/input.xlsx", b"training-input")
        output.writestr("domain/task-1/target.xlsx", b"private-target")
    verified = tmp_path / "verified"
    verified_task = verified / "spreadsheet/13-1"
    verified_task.mkdir(parents=True)
    (verified_task / "13-1_init.xlsx").write_bytes(b"verified-input")
    (verified_task / "13-1_golden.xlsx").write_bytes(b"private-golden")
    alternate_task = verified / "spreadsheet/13-2"
    alternate_task.mkdir(parents=True)
    (alternate_task / "initial.xlsx").write_bytes(b"alternate-verified-input")
    (alternate_task / "13-2_golden.xlsx").write_bytes(b"alternate-private-golden")
    cache = tmp_path / "cache"
    cache.mkdir()
    resolver = ProtocolV10SpreadsheetAssetResolver(
        training_archive=archive.resolve(),
        verified_root=verified.resolve(),
        extraction_cache_root=cache.resolve(),
    )

    training = resolver.resolve(_task(), {"spreadsheet_task_route": "domain/task-1"})
    final = resolver.resolve(_task(), {"spreadsheet_relative_path": "spreadsheet/13-1"})
    alternate = resolver.resolve(_task(), {"spreadsheet_relative_path": "spreadsheet/13-2"})

    assert training.input_workbook.read_bytes() == b"training-input"
    assert final.input_workbook.read_bytes() == b"verified-input"
    assert alternate.input_workbook.read_bytes() == b"alternate-verified-input"
    assert not (training.input_workbook.parent / "target.xlsx").exists()


def test_verified_oracle_accepts_released_unprefixed_name(tmp_path: Path) -> None:
    task = tmp_path / "spreadsheet" / "13-2"
    task.mkdir(parents=True)
    oracle = task / "golden.xlsx"
    oracle.write_bytes(b"private-golden")

    assert _verified_oracle(tmp_path.resolve(), PurePosixPath("spreadsheet/13-2")) == oracle


def test_official_process_oj_recalculates_and_compares_private_workbook(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official"
    evaluation = source / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "evaluation.py").write_text(
        "from pathlib import Path\n"
        "def compare_workbooks(gt, proc, instruction_type, answer_position):\n"
        '    assert answer_position == "Sheet1\'!A1"\n'
        "    print('upstream diagnostic')\n"
        "    return Path(gt).read_bytes() == Path(proc).read_bytes(), ''\n",
        encoding="utf-8",
    )
    archive = tmp_path / "training.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("domain/task-1/target.xlsx", b"expected")
    verified = tmp_path / "verified"
    verified.mkdir()
    workspaces = tmp_path / "workspaces"
    workspace = workspaces / "case"
    workspace.mkdir(parents=True)
    submitted = workspace / "input.xlsx"
    submitted.write_bytes(b"expected")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    libreoffice = tmp_path / "soffice"
    libreoffice.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "out = sys.argv[sys.argv.index('--outdir') + 1]\n"
        "shutil.copy2(sys.argv[-1], out + '/submitted.xlsx')\n",
        encoding="utf-8",
    )
    libreoffice.chmod(libreoffice.stat().st_mode | stat.S_IXUSR)
    oj = SpreadsheetBenchProcessDeployment(
        interpreter_path=Path(sys.executable).resolve(),
        official_source_root=source.resolve(),
        training_archive=archive.resolve(),
        verified_root=verified.resolve(),
        libreoffice_path=libreoffice.resolve(),
        workspace_root=workspaces.resolve(),
        temporary_root=temporary.resolve(),
        timeout_seconds=10.0,
    ).build()

    grade = asyncio.run(
        oj.grade(
            "sheet/task-1",
            {
                "private_payload": {
                    "answer_position": "Sheet1'!A1",
                    "spreadsheet_task_route": "domain/task-1",
                },
                "submitted_workbook_path": str(submitted.resolve()),
                "workspace_id": "sheet/task-1",
            },
        )
    )

    assert grade.all_cases_pass
