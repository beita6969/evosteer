"""Answer-free Protocol 10 action surfaces and preregistered budget profiles."""

from __future__ import annotations

from skillev.rollout import (
    ActionSurface,
    CompletionSpec,
    RolloutBudgetProfile,
    TerminalMode,
    ToolActionSpec,
)

_ANSWER_BENCHMARKS = {"hotpotqa", "triviaqa", "aime-2026", "healthbench"}

COMPLETION_WIRE_INSTRUCTION = (
    'Use exactly {"kind":"complete","name":"complete",'
    '"arguments":{"value":{"answer":"YOUR_ANSWER"}}}. '
    "kind and name belong at the root, not inside arguments. "
    "Do not put answer or value at the root or add resource_id/skill_id to completion. "
    "Encode the entire answer (including Python code) as one valid JSON string: "
    "escape newlines, double quotes and backslashes; no markdown fences or text outside JSON."
)


def protocol_v10_action_contract(
    benchmark_id: str,
    *,
    max_steps: int | None = None,
) -> tuple[ActionSurface, RolloutBudgetProfile]:
    """Return the frozen public action domain for one active benchmark."""

    if benchmark_id in _ANSWER_BENCHMARKS:
        instruction = {
            "aime-2026": (
                "The answer must be one canonical base-10 integer string from 0 through "
                "999, with no prose, LaTeX, sign, comma, or decimal point."
            ),
            "healthbench": (
                "Place the complete medical response in answer; give a useful, safe answer "
                "without markdown fences."
            ),
        }.get(benchmark_id, "Place only the final answer text in answer.")
        profile = (
            RolloutBudgetProfile("long-answer", 2, 512, 1536)
            if benchmark_id == "healthbench"
            else RolloutBudgetProfile("concise-answer", 2, 512, 192)
        )
        return (
            ActionSurface(
                terminal_mode=TerminalMode.EXPLICIT_COMPLETION,
                completion=CompletionSpec(
                    value_schema={"answer": "non-empty string"},
                    example_value={"answer": "YOUR_FINAL_ANSWER"},
                ),
                public_instructions=(instruction,),
            ),
            profile,
        )
    if benchmark_id == "mbpp-plus-fixed-100":
        return (
            ActionSurface(
                terminal_mode=TerminalMode.EXPLICIT_COMPLETION,
                completion=CompletionSpec(
                    value_schema={"answer": "complete Python source string"},
                    example_value={"answer": "def required_function(x): return x"},
                ),
                public_instructions=(
                    "Return complete Python source defining the function named by "
                    "code_signature; do not use a markdown fence.",
                    COMPLETION_WIRE_INSTRUCTION,
                ),
            ),
            RolloutBudgetProfile("code-completion", 2, 1024, 2048),
        )
    if benchmark_id == "webshop":
        return (
            ActionSurface(
                terminal_mode=TerminalMode.ENVIRONMENT,
                tools=(
                    ToolActionSpec(
                        "webshop",
                        "search",
                        {"query": "non-empty string"},
                        {"query": "SEARCH_QUERY"},
                    ),
                    ToolActionSpec(
                        "webshop",
                        "click",
                        {"target": "one current available action"},
                        {"target": "VISIBLE_TARGET"},
                    ),
                    ToolActionSpec("webshop", "purchase", {}, {}),
                ),
                dynamic_choice_fields=("initial_available_actions", "available_actions"),
                public_instructions=(
                    "purchase maps to the official click[buy now] action. Search and click "
                    "targets must come from the current public page state.",
                ),
            ),
            RolloutBudgetProfile("shopping", 20, 384, 256),
        )
    if benchmark_id == "alfworld":
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("ALFWorld action contract requires max_steps")
        return (
            ActionSurface(
                terminal_mode=TerminalMode.ENVIRONMENT,
                tools=(
                    ToolActionSpec(
                        "alfworld",
                        "act",
                        {"command": "one admissible command string"},
                        {"command": "ONE_ADMISSIBLE_COMMAND"},
                    ),
                ),
                dynamic_choice_fields=("admissible_commands",),
            ),
            RolloutBudgetProfile("embodied", max_steps, 384, 256),
        )
    if benchmark_id == "spreadsheetbench":
        return (
            ActionSurface(
                terminal_mode=TerminalMode.EXPLICIT_COMPLETION,
                tools=(
                    ToolActionSpec(
                        "spreadsheet",
                        "execute",
                        {"code": "Python source string"},
                        {"code": "PYTHON_CODE_EDITING_AND_SAVING_WORKBOOK_PATH"},
                    ),
                ),
                completion=CompletionSpec(
                    value_schema={"submit": True},
                    example_value={"submit": True},
                ),
                public_instructions=(
                    "Use WORKBOOK_PATH, edit the workbook in place, and save it before "
                    "submitting. You may inspect, edit, verify, repair, then submit.",
                ),
            ),
            RolloutBudgetProfile("spreadsheet", 12, 512, 2048),
        )
    if benchmark_id == "appworld":
        return (
            ActionSurface(
                terminal_mode=TerminalMode.EXPLICIT_COMPLETION,
                tools=(
                    ToolActionSpec(
                        "appworld",
                        "execute",
                        {"code": "Python source using public AppWorld APIs"},
                        {"code": "PYTHON_CODE_USING_PUBLIC_APPWORLD_APIS"},
                    ),
                ),
                completion=CompletionSpec(
                    value_schema={"submit": True},
                    example_value={"submit": True},
                ),
                public_instructions=(
                    "Use only the public apps and API signatures in public_api_surface. "
                    "State persists across execute calls.",
                ),
            ),
            RolloutBudgetProfile("appworld", 20, 512, 2048),
        )
    raise ValueError(f"unsupported Protocol 10 benchmark: {benchmark_id}")


__all__ = ["protocol_v10_action_contract"]
