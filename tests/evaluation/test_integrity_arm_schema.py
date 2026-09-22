"""The YAML and broker JSON arm paths share one closed, typed schema."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from skillev.evaluation.step0_integrity import (
    ARM_V1_FORMAT,
    ARM_V2_FORMAT,
    ARM_V4_FORMAT,
    AgentTopology,
    InferenceArm,
    SkillMode,
    ToolCallMode,
    decode_integrity_arm,
    load_integrity_arm,
    validate_paired_intervention,
)


def test_v2_yaml_and_json_transport_decode_to_the_same_arm(tmp_path: Path) -> None:
    expected = InferenceArm("synthetic-clean")
    value = expected.to_value()
    path = tmp_path / "arm.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    assert value["format"] == ARM_V2_FORMAT
    assert load_integrity_arm(path) == expected
    assert decode_integrity_arm(value) == expected


def test_v1_is_an_explicit_compatibility_schema_without_hidden_context() -> None:
    value = InferenceArm("synthetic-v1").to_value()
    value["format"] = ARM_V1_FORMAT
    del value["precomputed_task_context"]
    del value["legacy"]

    decoded = decode_integrity_arm(value)
    assert decoded.precomputed_task_context is False
    assert decoded.legacy is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: {**value, "unexpected": False},
        lambda value: {key: item for key, item in value.items() if key != "legacy"},
        lambda value: {**value, "native_thinking": "false"},
        lambda value: {**value, "precomputed_task_context": True},
    ],
)
def test_v2_arm_rejects_unknown_missing_and_wrongly_typed_fields(mutate) -> None:
    with pytest.raises((TypeError, ValueError)):
        decode_integrity_arm(mutate(InferenceArm("synthetic-clean").to_value()))


def test_precomputed_context_requires_an_explicit_legacy_arm() -> None:
    value = InferenceArm("historical", precomputed_task_context=True, legacy=True).to_value()
    assert decode_integrity_arm(value).precomputed_task_context is True


def test_native_template_is_an_explicit_typed_arm_control(tmp_path):
    arm = InferenceArm("native", tool_call_mode=ToolCallMode.QWEN_XML)
    path = tmp_path / "native.yaml"
    path.write_text(yaml.safe_dump(arm.to_value()))
    assert load_integrity_arm(path) == decode_integrity_arm(arm.to_value()) == arm
    with pytest.raises(ValueError):
        decode_integrity_arm({**arm.to_value(), "tool_call_mode": "guessed-protocol"})
    with pytest.raises(ValueError):
        decode_integrity_arm({**arm.to_value(), "format": ARM_V2_FORMAT})
    with pytest.raises(ValueError):
        validate_paired_intervention(InferenceArm("plain"), arm)


def test_trained_arm_roundtrips_its_explicit_policy_and_native_tools(tmp_path):
    arm = InferenceArm(
        "trained",
        optimizer_steps=16,
        policy_id="synthetic-step-16",
        tool_call_mode=ToolCallMode.QWEN_XML,
    )
    path = tmp_path / "arm.yaml"
    path.write_text(yaml.safe_dump(arm.to_value()))
    assert arm.to_value()["format"] == ARM_V4_FORMAT
    assert load_integrity_arm(path) == decode_integrity_arm(arm.to_value()) == arm


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimizer_steps": 16},
        {"policy_id": "trained-without-step"},
        {"optimizer_steps": 16, "policy_id": ""},
    ],
)
def test_trained_policy_cannot_be_implied_by_an_arm_label(kwargs):
    with pytest.raises(ValueError):
        InferenceArm("trained", **kwargs)


def test_policy_and_skill_comparisons_keep_the_other_axes_fixed():
    base = InferenceArm("base", agent_topology=AgentTopology.MULTI_AGENT)
    trained = replace(base, arm_id="trained", optimizer_steps=16, policy_id="synthetic-step-16")
    skilled = replace(trained, arm_id="trained-with-skills", skill_mode=SkillMode.GENERIC_TEXT)
    assert validate_paired_intervention(base, trained) == "forward-policy"
    assert validate_paired_intervention(trained, skilled) == "skill-access"
    with pytest.raises(ValueError):
        validate_paired_intervention(base, skilled)
    with pytest.raises(ValueError):
        validate_paired_intervention(base, replace(trained, agent_topology=AgentTopology.SINGLE))
    with pytest.raises(ValueError):
        validate_paired_intervention(trained, replace(trained, optimizer_steps=17))
