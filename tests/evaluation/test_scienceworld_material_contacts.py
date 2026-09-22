"""Public contact syntax changes no circuit, object choice or task strategy."""

import pytest

from skillev.evaluation.capability_registry import native_tool_definitions
from skillev.evaluation.input_metric_contracts import CONTRACTS, IID_BENCHMARKS, OOD_BENCHMARKS
from skillev.evaluation.native_tool_instructions import native_tool_reference
from skillev.evaluation.scienceworld_commands import (
    MATERIAL_CONTACT_COMMAND_PROFILE,
    PERSISTENT_FOCUS_COMMAND_PROFILE,
    command_profile,
    commands_for_profile,
    native_functions,
)
from skillev.task_semantic_guidance import (
    PUBLIC_TASK_SEMANTICS_V13,
    PUBLIC_TASK_SEMANTICS_V14,
    phase_deliverable,
    public_task_semantics,
)


def test_only_generic_material_contact_reference_changes():
    old = commands_for_profile(PERSISTENT_FOCUS_COMMAND_PROFILE)
    new = commands_for_profile(MATERIAL_CONTACT_COMMAND_PROFILE)
    assert new[:4] == old[:4]
    assert new[5:] == old[5:]
    assert new[4].syntax == old[4].syntax
    assert new[4].meaning == old[4].meaning
    assert new[4].effects == old[4].effects
    assert new[4].evidence == old[4].evidence
    assert "Generic material objects" in new[4].reference
    assert "unconnected contact" in new[4].reference
    assert new[4].help() in native_tool_reference(
        "scienceworld", scienceworld_profile=MATERIAL_CONTACT_COMMAND_PROFILE
    )
    assert native_functions(MATERIAL_CONTACT_COMMAND_PROFILE) == native_functions(
        PERSISTENT_FOCUS_COMMAND_PROFILE
    )
    old_tools, new_tools = (
        {
            row["function"]["name"]: row["function"]
            for row in native_tool_definitions("scienceworld", scienceworld_profile=profile)
        }
        for profile in (PERSISTENT_FOCUS_COMMAND_PROFILE, MATERIAL_CONTACT_COMMAND_PROFILE)
    )
    for name in old_tools:
        assert old_tools[name]["parameters"] == new_tools[name]["parameters"]
    for name in ("inspect_object", "select_task_object"):
        assert old_tools[name]["description"] == new_tools[name]["description"]
    assert command_profile(PUBLIC_TASK_SEMANTICS_V14) == MATERIAL_CONTACT_COMMAND_PROFILE


@pytest.mark.parametrize(
    ("benchmark", "input_profile"),
    [
        (name, profile)
        for name in (*IID_BENCHMARKS, *OOD_BENCHMARKS)
        for profile in CONTRACTS[name].input_profiles
    ],
)
def test_material_api_condition_keeps_task_and_submission_meaning(benchmark, input_profile):
    assert public_task_semantics(
        benchmark, input_profile=input_profile, version=PUBLIC_TASK_SEMANTICS_V14
    ) == public_task_semantics(
        benchmark, input_profile=input_profile, version=PUBLIC_TASK_SEMANTICS_V13
    )
    assert phase_deliverable(benchmark, version=PUBLIC_TASK_SEMANTICS_V14) == phase_deliverable(
        benchmark, version=PUBLIC_TASK_SEMANTICS_V13
    )
