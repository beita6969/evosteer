"""Answer-free action surfaces for the current Protocol 11 catalog."""

from __future__ import annotations

from skillev.benchmarks.protocol_v10_action import protocol_v10_action_contract
from skillev.rollout import (
    ACTION_SURFACE_FORMAT_V2,
    ActionSurface,
    ArgumentFieldSpec,
    ArgumentType,
    CompletionSpec,
    RolloutBudgetProfile,
    TerminalMode,
    ToolActionSpecV2,
)


def protocol_v11_action_contract(
    benchmark_id: str,
    *,
    max_steps: int | None = None,
) -> tuple[ActionSurface, RolloutBudgetProfile]:
    """Return a frozen public action domain without mutating Protocol 10."""

    if benchmark_id == "humaneval":
        return (
            ActionSurface(
                terminal_mode=TerminalMode.EXPLICIT_COMPLETION,
                completion=CompletionSpec(
                    value_schema={"answer": "complete Python source string"},
                    example_value={"answer": "def required_function(x):\n    return x"},
                ),
                public_instructions=(
                    "Return executable Python source for the declared function. Do not include "
                    "tests, markdown fences, or explanations.",
                ),
            ),
            RolloutBudgetProfile("humaneval-code-completion", 2, 1024, 4096),
        )
    if benchmark_id == "appworld":
        surface, _ = protocol_v10_action_contract("appworld", max_steps=max_steps)
        return surface, RolloutBudgetProfile("appworld-official-react-code", 40, 512, 4096)
    if benchmark_id in {"mbpp-plus", "mbpp-plus-hard"}:
        return protocol_v10_action_contract("mbpp-plus-fixed-100")
    if benchmark_id == "spreadsheetbench":
        spreadsheet_tools = (
            ToolActionSpecV2("spreadsheet", "list_sheets", {}, {}),
            ToolActionSpecV2(
                "spreadsheet",
                "preview",
                {
                    "sheet": ArgumentFieldSpec(ArgumentType.STRING, required=True),
                    "range": ArgumentFieldSpec(
                        ArgumentType.STRING, required=False, nullable=True, default=None
                    ),
                    "max_rows": ArgumentFieldSpec(
                        ArgumentType.INTEGER, required=False, default=20, minimum=1, maximum=50
                    ),
                    "max_columns": ArgumentFieldSpec(
                        ArgumentType.INTEGER, required=False, default=12, minimum=1, maximum=30
                    ),
                },
                {"sheet": "Sheet1", "range": "A1:F20", "max_rows": 20, "max_columns": 6},
            ),
            ToolActionSpecV2(
                "spreadsheet",
                "describe_range",
                {
                    "sheet": ArgumentFieldSpec(ArgumentType.STRING, required=True),
                    "range": ArgumentFieldSpec(ArgumentType.STRING, required=True),
                },
                {"sheet": "Sheet1", "range": "A1:F20"},
            ),
            ToolActionSpecV2(
                "spreadsheet",
                "find",
                {
                    "query": ArgumentFieldSpec(ArgumentType.STRING, required=True),
                    "sheet": ArgumentFieldSpec(
                        ArgumentType.STRING, required=False, nullable=True, default=None
                    ),
                    "start_after": ArgumentFieldSpec(
                        ArgumentType.STRING, required=False, nullable=True, default=None
                    ),
                    "limit": ArgumentFieldSpec(
                        ArgumentType.INTEGER, required=False, default=50, minimum=1, maximum=100
                    ),
                },
                {"query": "Revenue", "sheet": "Summary", "start_after": None, "limit": 50},
            ),
            ToolActionSpecV2(
                "spreadsheet",
                "execute",
                {"code": ArgumentFieldSpec(ArgumentType.STRING, required=True)},
                {"code": "from openpyxl import load_workbook\n..."},
            ),
        )
        return (
            ActionSurface(
                terminal_mode=TerminalMode.EXPLICIT_COMPLETION,
                tools=spreadsheet_tools,
                completion=CompletionSpec(
                    value_schema={"submit": True}, example_value={"submit": True}
                ),
                public_instructions=("Edit WORKBOOK_PATH in place, save it, then submit.",),
                format=ACTION_SURFACE_FORMAT_V2,
            ),
            RolloutBudgetProfile("spreadsheet-public-inspection", 12, 512, 2048),
        )
    if benchmark_id in {
        "hotpotqa",
        "triviaqa",
        "aime-2026",
        "healthbench",
        "webshop",
        "alfworld",
    }:
        return protocol_v10_action_contract(benchmark_id, max_steps=max_steps)
    raise ValueError(f"unsupported Protocol 11 benchmark: {benchmark_id}")


__all__ = ["protocol_v11_action_contract"]
