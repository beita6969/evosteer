"""Persistent-selection help changes no literal action, task or score semantics."""

import pytest

from skillev.evaluation.capability_registry import native_tool_definitions
from skillev.evaluation.input_metric_contracts import CONTRACTS, IID_BENCHMARKS, OOD_BENCHMARKS
from skillev.evaluation.native_tool_instructions import native_tool_reference
from skillev.evaluation.scienceworld_commands import (
    COMMANDS,
    PERSISTENT_FOCUS_COMMAND_PROFILE,
    SCOPED_TYPED_COMMAND_PROFILE,
    command_profile,
    commands_for_profile,
    native_functions,
)
from skillev.task_semantic_guidance import (
    PUBLIC_TASK_SEMANTICS_V12,
    PUBLIC_TASK_SEMANTICS_V13,
    phase_deliverable,
    public_task_semantics,
)


def test_focus_fact_is_shared_by_system_reference_and_both_literal_entrypoints():
    old_profile = SCOPED_TYPED_COMMAND_PROFILE
    new_profile = PERSISTENT_FOCUS_COMMAND_PROFILE
    old_specs, new_specs = commands_for_profile(old_profile), commands_for_profile(new_profile)
    assert old_specs == COMMANDS
    assert new_specs[1:] == old_specs[1:]
    assert new_specs[0].effects == old_specs[0].effects
    assert new_specs[0].reference == old_specs[0].reference
    assert new_specs[0].evidence == old_specs[0].evidence
    assert "replaces the previous focus" in new_specs[0].meaning
    assert "remains monitored across later commands" in new_specs[0].meaning
    old, new = (
        {
            row["function"]["name"]: row["function"]
            for row in native_tool_definitions("scienceworld", scienceworld_profile=profile)
        }
        for profile in (old_profile, new_profile)
    )
    assert native_functions(old_profile) == native_functions(new_profile)
    for name in native_functions(old_profile):
        assert old[name]["parameters"] == new[name]["parameters"]
    assert old["inspect_object"]["description"] == new["inspect_object"]["description"]
    for name in ("act", "select_task_object"):
        assert new_specs[0].help() in new[name]["description"]
        assert new_specs[0].meaning not in old[name]["description"]
    assert new_specs[0].help() in native_tool_reference(
        "scienceworld", scienceworld_profile=new_profile
    )
    assert command_profile(PUBLIC_TASK_SEMANTICS_V13) == new_profile
    assert command_profile(PUBLIC_TASK_SEMANTICS_V12) == old_profile


@pytest.mark.parametrize(
    ("benchmark", "input_profile"),
    [
        (name, profile)
        for name in (*IID_BENCHMARKS, *OOD_BENCHMARKS)
        for profile in CONTRACTS[name].input_profiles
    ],
)
def test_focus_help_profile_inherits_task_meaning_and_phase(benchmark, input_profile):
    assert public_task_semantics(
        benchmark, input_profile=input_profile, version=PUBLIC_TASK_SEMANTICS_V13
    ) == public_task_semantics(
        benchmark, input_profile=input_profile, version=PUBLIC_TASK_SEMANTICS_V12
    )
    assert phase_deliverable(benchmark, version=PUBLIC_TASK_SEMANTICS_V13) == phase_deliverable(
        benchmark, version=PUBLIC_TASK_SEMANTICS_V12
    )
