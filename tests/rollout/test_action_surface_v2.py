from skillev.benchmarks.protocol_v11_action import protocol_v11_action_contract
from skillev.rollout import ToolActionSpecV2, admit_arguments
from skillev.runtime import ActionKind, StructuredAction


def test_spreadsheet_optional_fields_are_defaulted() -> None:
    surface, _ = protocol_v11_action_contract("spreadsheetbench")
    preview = next(tool for tool in surface.tools if tool.name == "preview")
    assert isinstance(preview, ToolActionSpecV2)
    result = admit_arguments(
        StructuredAction(
            ActionKind.TOOL,
            "preview",
            {"sheet": "Sheet1"},
            resource_id="spreadsheet",
        ),
        preview,
    )
    assert result.admitted
    assert result.normalized_action.arguments == {
        "sheet": "Sheet1",
        "range": None,
        "max_rows": 20,
        "max_columns": 12,
    }


def test_spreadsheet_v2_rejects_unknown_and_out_of_range_fields() -> None:
    surface, _ = protocol_v11_action_contract("spreadsheetbench")
    preview = next(tool for tool in surface.tools if tool.name == "preview")
    assert isinstance(preview, ToolActionSpecV2)
    unknown = admit_arguments(
        StructuredAction(
            ActionKind.TOOL,
            "preview",
            {"sheet": "Sheet1", "extra": True},
            resource_id="spreadsheet",
        ),
        preview,
    )
    assert unknown.error_code == "unknown_arguments"
    bounded = admit_arguments(
        StructuredAction(
            ActionKind.TOOL,
            "preview",
            {"sheet": "Sheet1", "max_rows": 51},
            resource_id="spreadsheet",
        ),
        preview,
    )
    assert bounded.error_code == "argument_out_of_range"
