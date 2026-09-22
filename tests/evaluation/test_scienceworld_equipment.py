"""Optional public help changes neither actions, old profiles, nor hidden observations."""

import asyncio
import json

import pytest

from skillev.evaluation.scienceworld_commands import command_profile
from skillev.evaluation.scienceworld_equipment import EQUIPMENT_PROFILE, equipment_reference
from skillev.evaluation.step0_integrity import InferenceArm, ToolCallMode
from skillev.task_semantic_guidance import (
    PUBLIC_TASK_SEMANTICS_V10,
    PUBLIC_TASK_SEMANTICS_V11,
    public_task_semantics,
)
from tests.evaluation.test_integrity_native_tool_calls import call
from tests.evaluation.test_scienceworld_owner_finish import FINISH, science_runtime


@pytest.mark.parametrize("profile", ["training-public-source-bridge@1", "released-ood-source@1"])
def test_help_is_public_task_independent_and_old_condition_is_unchanged(profile):
    for benchmark in (
        "musique",
        "nq-open",
        "math-hard",
        "gpqa-diamond-bioorganic",
        "apps-introductory",
    ):
        assert public_task_semantics(
            benchmark, input_profile=profile, version=PUBLIC_TASK_SEMANTICS_V11
        ) == public_task_semantics(
            benchmark, input_profile=profile, version=PUBLIC_TASK_SEMANTICS_V10
        )
    old = public_task_semantics(
        "scienceworld", input_profile=profile, version=PUBLIC_TASK_SEMANTICS_V10
    )
    new = public_task_semantics(
        "scienceworld", input_profile=profile, version=PUBLIC_TASK_SEMANTICS_V11
    )
    assert EQUIPMENT_PROFILE not in old
    assert new == old + equipment_reference()
    assert command_profile(PUBLIC_TASK_SEMANTICS_V11) == command_profile(PUBLIC_TASK_SEMANTICS_V10)


@pytest.mark.parametrize("version", [PUBLIC_TASK_SEMANTICS_V10, PUBLIC_TASK_SEMANTICS_V11])
def test_actual_owner_still_sends_its_literal_circuit_and_selection(monkeypatch, tmp_path, version):
    actions = ["connect battery anode to bulb anode", "focus on empty vessel"]
    instance, entry, environment = science_runtime(
        monkeypatch, tmp_path, [*(call("act", command=a) for a in actions), FINISH]
    )
    arm = InferenceArm(
        "instrument-help", tool_call_mode=ToolCallMode.QWEN_XML, task_semantic_guidance=version
    )
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert environment.actions == actions
        assert json.loads(final.text) == actions
        assert final.intervention_counts["model_calls"] == 3
        assert final.intervention_counts["peer_model_calls"] == 0
        requests = instance.journal.traces(
            ("synthetic", arm.arm_id, entry.task_id), "rendered-request"
        )
        for request in requests:
            system = request["messages"][0]["content"]
            assert (EQUIPMENT_PROFILE in system) is (version == PUBLIC_TASK_SEMANTICS_V11)
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        instance.journal.close()
