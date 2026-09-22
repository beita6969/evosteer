from pathlib import Path

import pytest
from openpyxl import Workbook
from skillev_private.benchmarks.protocol_v10_spreadsheet_runtime import (
    IsolatedSpreadsheetWorkspace,
)
from skillev_private.benchmarks.protocol_v11_spreadsheet_runtime import (
    SpreadsheetInspectionRuntime,
    bind_protocol_v11_spreadsheet_session,
)

from skillev.rollout import RolloutSessionBundle, RolloutTask


def _workbook(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Summary"
    sheet.append(["Revenue", 10])
    sheet.append(["Cost", 4])
    workbook.save(path)
    workbook.close()


def test_inspection_tools_are_read_only_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "input.xlsx"
    _workbook(path)
    runtime = SpreadsheetInspectionRuntime(path)
    assert runtime.list_sheets() == {"sheets": ["Summary"]}
    preview = runtime.preview(
        {"sheet": "Summary", "range": "A1:B2", "max_rows": 2, "max_columns": 2}
    )
    assert [[cell["value"] for cell in row] for row in preview["rows"]] == [
        ["Revenue", 10],
        ["Cost", 4],
    ]
    assert preview["rows"][0][1]["data_type"] == "n"
    found = runtime.find({"query": "revenue", "sheet": "Summary"})
    assert found["matches"] == [{"sheet": "Summary", "cell": "A1", "value": "Revenue"}]
    assert found["next_start_after"] is None
    with pytest.raises(ValueError):
        runtime.preview({"sheet": "Summary", "max_rows": 51, "max_columns": 2})


def test_binding_adds_input_only_preview_and_v11_tools(tmp_path: Path) -> None:
    _workbook(tmp_path / "input.xlsx")
    task = RolloutTask(
        task_id="sheet-task",
        environment_id="sheet-env",
        task_family="spreadsheetbench/edit",
        context_id="sheet-context",
        query="Update the workbook.",
        available_tools=("spreadsheet:execute",),
        public_context={"instruction_type": "formatting", "answer_position": "Summary!B2"},
    )
    workspace = IsolatedSpreadsheetWorkspace(task, tmp_path, object(), object(), {})  # type: ignore[arg-type]
    evaluator = object()
    upgraded, session = bind_protocol_v11_spreadsheet_session(
        task,
        RolloutSessionBundle(workspace, evaluator, ()),  # type: ignore[arg-type]
    )
    assert upgraded.public_context["workbook_filename"] == "input.xlsx"  # type: ignore[index]
    first_row = upgraded.public_context["sheets"][0]["first_nonempty_rows"][0]  # type: ignore[index]
    assert [cell["value"] for cell in first_row] == ["Revenue", 10]
    assert [cell["coordinate"] for cell in first_row] == ["A1", "B1"]
    assert "spreadsheet:preview" in upgraded.available_tools
    assert upgraded.budget_profile is not None
    assert upgraded.budget_profile.max_turns == 12
    assert session.environment.environment_id == "sheet-env"
