from __future__ import annotations

import json

import pytest

from skillev.benchmarks.protocol_v10_action import protocol_v10_action_contract
from skillev.rollout import ActionParseStatus, StructuredJsonActionCodec, TerminalMode


@pytest.mark.parametrize(
    ("benchmark", "max_steps", "profile", "terminal"),
    [
        ("hotpotqa", None, "concise-answer", TerminalMode.EXPLICIT_COMPLETION),
        ("triviaqa", None, "concise-answer", TerminalMode.EXPLICIT_COMPLETION),
        ("aime-2026", None, "concise-answer", TerminalMode.EXPLICIT_COMPLETION),
        ("healthbench", None, "long-answer", TerminalMode.EXPLICIT_COMPLETION),
        ("webshop", None, "shopping", TerminalMode.ENVIRONMENT),
        ("alfworld", 31, "embodied", TerminalMode.ENVIRONMENT),
        ("spreadsheetbench", None, "spreadsheet", TerminalMode.EXPLICIT_COMPLETION),
        ("appworld", None, "appworld", TerminalMode.EXPLICIT_COMPLETION),
        ("mbpp-plus-fixed-100", None, "code-completion", TerminalMode.EXPLICIT_COMPLETION),
    ],
)
def test_all_protocol_v10_domains_have_explicit_surface_and_budget(
    benchmark: str,
    max_steps: int | None,
    profile: str,
    terminal: TerminalMode,
) -> None:
    surface, budget = protocol_v10_action_contract(benchmark, max_steps=max_steps)
    assert surface.terminal_mode is terminal
    assert budget.profile_id == profile
    if benchmark == "alfworld":
        assert budget.max_turns == max_steps

    codec = StructuredJsonActionCodec()
    examples = [
        {
            "arguments": tool.example_arguments,
            "kind": "tool",
            "name": tool.name,
            "resource_id": tool.resource_id,
        }
        for tool in surface.tools
    ]
    if surface.completion is not None:
        examples.append(
            {
                "arguments": {"value": surface.completion.example_value},
                "kind": "complete",
                "name": "complete",
            }
        )
    assert examples
    assert all(
        codec.parse(json.dumps(example)).status is ActionParseStatus.VALID for example in examples
    )


def test_stateful_surfaces_use_public_submit_sentinel() -> None:
    for benchmark in ("spreadsheetbench", "appworld"):
        surface, _ = protocol_v10_action_contract(benchmark)
        assert surface.completion is not None
        assert surface.completion.example_value == {"submit": True}
