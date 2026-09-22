import pytest

from skillev.benchmarks.protocol_v11_action import protocol_v11_action_contract


def test_humaneval_is_source_completion() -> None:
    surface, profile = protocol_v11_action_contract("humaneval")
    assert surface.completion is not None
    assert profile.max_action_tokens == 4096


def test_removed_swe_is_rejected() -> None:
    with pytest.raises(ValueError):
        protocol_v11_action_contract("swe-bench-verified")


def test_appworld_and_mbpp_hard_have_native_surfaces() -> None:
    appworld, _ = protocol_v11_action_contract("appworld")
    assert {tool.name for tool in appworld.tools} == {"execute"}
    mbpp, _ = protocol_v11_action_contract("mbpp-plus-hard")
    assert mbpp.completion is not None


def test_spreadsheet_has_read_only_inspection_tools() -> None:
    surface, _ = protocol_v11_action_contract("spreadsheetbench")
    assert {tool.name for tool in surface.tools} == {
        "list_sheets",
        "preview",
        "describe_range",
        "find",
        "execute",
    }
