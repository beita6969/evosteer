"""Function-local public help must match the unchanged literal operation."""

import pytest

from skillev.evaluation.capability_registry import native_tool_definitions
from skillev.evaluation.input_metric_contracts import CONTRACTS, IID_BENCHMARKS, OOD_BENCHMARKS
from skillev.evaluation.scienceworld_commands import (
    COMMANDS,
    SCOPED_TYPED_COMMAND_PROFILE,
    TYPED_COMMAND_PROFILE,
    command_profile,
    native_functions,
    typed_description,
)
from skillev.task_semantic_guidance import (
    PUBLIC_TASK_SEMANTICS_V11,
    PUBLIC_TASK_SEMANTICS_V12,
    phase_deliverable,
    public_task_semantics,
)


def test_object_function_help_does_not_advertise_the_whole_observation_family():
    def functions(profile):
        return {
            row["function"]["name"]: row["function"]
            for row in native_tool_definitions("scienceworld", scienceworld_profile=profile)
        }

    old = functions(TYPED_COMMAND_PROFILE)
    new = functions(SCOPED_TYPED_COMMAND_PROFILE)
    old_help = old["inspect_object"]["description"]
    new_help = new["inspect_object"]["description"]
    assert COMMANDS[1].help() in old_help
    assert typed_description("inspect_object") in old_help
    assert COMMANDS[1].syntax not in new_help
    assert "look at TARGET" in new_help
    for command in ("look around", "inventory", "task"):
        assert f'act(command="{command}")' in new_help
    for name in native_functions(TYPED_COMMAND_PROFILE):
        assert new[name]["parameters"] == old[name]["parameters"]
        if name != "inspect_object":
            assert new[name]["description"] == old[name]["description"].replace(
                TYPED_COMMAND_PROFILE, SCOPED_TYPED_COMMAND_PROFILE
            )
    assert native_functions(SCOPED_TYPED_COMMAND_PROFILE) == native_functions(TYPED_COMMAND_PROFILE)
    assert command_profile(PUBLIC_TASK_SEMANTICS_V12) == SCOPED_TYPED_COMMAND_PROFILE
    assert command_profile(PUBLIC_TASK_SEMANTICS_V11) == TYPED_COMMAND_PROFILE


@pytest.mark.parametrize(
    ("benchmark", "input_profile"),
    [
        (name, profile)
        for name in (*IID_BENCHMARKS, *OOD_BENCHMARKS)
        for profile in CONTRACTS[name].input_profiles
    ],
)
def test_new_help_profile_inherits_public_task_meaning_and_phase(benchmark, input_profile):
    assert public_task_semantics(
        benchmark, input_profile=input_profile, version=PUBLIC_TASK_SEMANTICS_V12
    ) == public_task_semantics(
        benchmark, input_profile=input_profile, version=PUBLIC_TASK_SEMANTICS_V11
    )
    assert phase_deliverable(benchmark, version=PUBLIC_TASK_SEMANTICS_V12) == phase_deliverable(
        benchmark, version=PUBLIC_TASK_SEMANTICS_V11
    )
