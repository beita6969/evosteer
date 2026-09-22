"""Actual snapshot bodies and model-selected skill tools across the isolated actor broker."""

import asyncio
import json
from dataclasses import replace

import pytest
from skillev_private.evaluation.integrity_skills import read_skill_library

from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.integrity_results import require_paired_controls
from skillev.evaluation.skill_library_config import FrozenSkillLibrary
from skillev.evaluation.step0_integrity import (
    InferenceArm,
    SkillMode,
    ToolCallMode,
    decode_integrity_arm,
)
from skillev.evolution.skill_access import ACTIVE_APPLICABILITY_RULE
from skillev.experiments._evolution_preflight_seed import _seed_document, planned_seed_documents
from skillev.runtime import SkillLibraryState, SkillRequirement
from skillev.runtime.runtime_snapshot import RUNTIME_SNAPSHOT_FORMAT
from tests.evaluation.test_integrity_broker_boundary import runtime
from tests.evaluation.test_integrity_policy_runtime_controls import production_runtime
from tests.evaluation.test_step0_integrity_results import controls


def snapshot(library_id, body, *, matches=True, kind="evolved"):
    document = _seed_document(
        skill_id="synthetic-skill",
        title="Synthetic skill",
        summary="A public synthetic method.",
        instructions=body,
        requirements=(SkillRequirement("public", "Use public observations."),),
        contexts=("*",) if matches else ("different-public-context",),
    )
    return FrozenSkillLibrary(
        library_id,
        kind,
        SkillLibraryState.from_seed_documents((document,)),
        16 if kind == "evolved" else 0,
    )


def arm_for(skills, *, native=False):
    return InferenceArm(
        "skills-" + skills.library_id,
        skill_mode=SkillMode.LIBRARY,
        skill_library_id=skills.library_id,
        skill_retrieval_rule=skills.retrieval_rule,
        tool_call_mode=ToolCallMode.QWEN_XML if native else ToolCallMode.PLAIN_TEXT,
    )


def checkpoint(tmp_path, skills):
    directory = tmp_path / skills.library_id
    directory.mkdir()
    (directory / "COMPLETE").write_text("complete\n")
    (directory / "runtime_state.json").write_text(
        json.dumps(
            {
                "format": RUNTIME_SNAPSHOT_FORMAT,
                "optimizer_step": 16,
                "execution_state": {
                    "library": skills.state.to_value(),
                    "projections": {"training_only": "PRIVATE_TRAINING_DIAGNOSTIC_SENTINEL"},
                },
            }
        )
    )
    return {
        "kind": "evolved",
        "checkpoint_directory": str(directory),
        "retrieval_rule": ACTIVE_APPLICABILITY_RULE,
    }


def actor_runtime(tmp_path, skills, outputs, *, budget=5000):
    entry = PublicTaskView.from_record("case", "aime-2026", {"problem": "Synthetic integer task."})
    instance = runtime(tmp_path, entry, outputs, calls=len(outputs))
    instance.config["budgets"][entry.benchmark].update(
        skill_instruction_tokens=budget, calls_per_turn=len(outputs)
    )
    instance.skill_libraries = {skills.library_id: skills}
    instance.initial_skills = None
    return instance, entry


@pytest.mark.parametrize("native", [False, True])
def test_two_checkpoint_libraries_send_their_actual_distinct_bodies_to_the_model(tmp_path, native):
    bodies = (
        "SYNTHETIC_ALPHA_BODY: compare the public quantities.",
        "SYNTHETIC_BETA_BODY: use a symbolic relation if useful.",
    )
    for index, body in enumerate(bodies):
        original = snapshot(f"snapshot-{index}", body)
        settings = checkpoint(tmp_path, original)
        skills = read_skill_library(original.library_id, settings)
        # Neither optimizer files nor a trainer are needed for this projection.
        assert not (tmp_path / original.library_id / "optimizer.pt").exists()
        run_path = tmp_path / f"run-{index}"
        run_path.mkdir()
        instance, entry = actor_runtime(run_path, skills, ["Final answer: 42."])
        arm = arm_for(skills, native=native)
        try:
            final = asyncio.run(instance.generate(entry, arm, "synthetic"))
            requests = instance.journal.traces(
                ("synthetic", arm.arm_id, "case"), "rendered-request"
            )
            rendered = json.dumps(requests)
            assert body in rendered
            assert bodies[1 - index] not in rendered
            assert "PRIVATE_TRAINING_DIAGNOSTIC_SENTINEL" not in rendered
            assert final.intervention_counts["skill_access_episodes"] == 1
            assert final.intervention_counts["skill_retrieved"] == 1
            assert final.intervention_counts["skill_blocks_injected"] == 1
            assert final.intervention_counts["skill_calls"] == 0
            assert final.intervention_counts["model_calls"] == 1
            assert skills.state.to_value() == original.state.to_value()
        finally:
            instance.journal.close()


def call(name, **arguments):
    return (
        "<tool_call>\n<function="
        + name
        + ">\n"
        + "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items())
        + "</function>\n</tool_call>"
    )


@pytest.mark.parametrize("native", [False, True])
def test_model_can_discover_read_and_invoke_a_skill_without_a_task_word_match(tmp_path, native):
    body = "SYNTHETIC_DISCOVERABLE_BODY: try a representation of your choosing."
    skills = snapshot("discovery", body, matches=False)
    commands = [
        ("list_skills", {}),
        ("retrieve_skills", {}),
        ("read_skill", {"skill_id": "synthetic-skill"}),
        ("invoke_skill", {"skill_id": "synthetic-skill"}),
    ]
    outputs = [
        call(name, **args) if native else json.dumps({"kind": "skill", "operation": name, **args})
        for name, args in commands
    ]
    instance, entry = actor_runtime(tmp_path, skills, [*outputs, "42"])
    arm = arm_for(skills, native=native)
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        scope = ("synthetic", arm.arm_id, "case")
        requests = instance.journal.traces(scope, "rendered-request")
        assert body not in json.dumps(requests[0])
        assert "list_skills" in json.dumps(requests[0])
        assert body in json.dumps(requests[3])
        assert final.intervention_counts["skill_no_match"] == 2
        assert final.intervention_counts["skill_discovery_calls"] == 1
        assert final.intervention_counts["skill_read_calls"] == 2
        assert final.intervention_counts["skill_calls"] == 1
        assert final.intervention_counts["model_calls"] == 5
        assert final.intervention_counts["communication_repairs"] == 0
    finally:
        instance.journal.close()


@pytest.mark.parametrize("budget", [0, 1])
def test_budget_skip_is_not_reported_as_a_body_in_the_context(tmp_path, budget):
    body = "SYNTHETIC_TOO_LARGE_BODY"
    skills = snapshot("budget", body)
    instance, entry = actor_runtime(tmp_path, skills, ["42"], budget=budget)
    arm = arm_for(skills)
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        scope = ("synthetic", arm.arm_id, "case")
        assert body not in json.dumps(instance.journal.traces(scope, "rendered-request"))
        assert final.intervention_counts["skill_retrieved"] == 1
        assert final.intervention_counts["skill_budget_skips"] == 1
        assert final.intervention_counts["skill_blocks_injected"] == 0
        assert final.intervention_counts["skill_calls"] == 0
        assert (
            instance.journal.traces(scope, "skill-retrieval")[0]["bodies"][0]["status"]
            == "skill-token-budget"
        )
    finally:
        instance.journal.close()


def test_no_skill_removes_library_and_retrieval_but_keeps_base_tools(tmp_path):
    skills = snapshot("unused", "SYNTHETIC_UNUSED_BODY")
    instance, entry = actor_runtime(tmp_path, skills, ["42"])
    arm = InferenceArm("off", tool_call_mode=ToolCallMode.QWEN_XML)
    # The no-skill path must not even look up a configured library.
    instance.skill_libraries = None
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        scope = ("synthetic", arm.arm_id, "case")
        requests = instance.journal.traces(scope, "rendered-request")
        rendered = json.dumps(requests)
        assert "SYNTHETIC_UNUSED_BODY" not in rendered
        assert "list_skills" not in rendered
        assert {tool["function"]["name"] for tool in requests[0]["tools"]} >= {
            "history",
            "submit_answer",
        }
        assert all(
            value == 0
            for name, value in final.intervention_counts.items()
            if name.startswith("skill_")
        )
        assert not instance.journal.traces(scope, "skill-retrieval")
    finally:
        instance.journal.close()


def test_frozen_skills_are_a_separate_declared_axis_and_v5_roundtrips():
    skills = snapshot("frozen", "Synthetic public method.")
    arm = arm_for(skills)
    assert decode_integrity_arm(arm.to_value()) == arm
    left = controls()
    right = replace(left, skills=skills.to_value())
    assert require_paired_controls(left, right, InferenceArm("off"), arm) == "skill-access"
    with pytest.raises(ValueError):
        require_paired_controls(left, right, arm, arm)
    with pytest.raises(ValueError):
        replace(arm, skill_retrieval_rule="unimplemented-learned-ranker")


def test_production_constructor_loads_only_selected_read_only_libraries(tmp_path, monkeypatch):
    skills = snapshot("selected", "Synthetic checkpoint method.")
    settings = checkpoint(tmp_path, skills)
    arm = arm_for(skills)
    instance, _, _ = production_runtime(
        tmp_path,
        monkeypatch,
        config_overrides={
            "arms": [arm.to_value()],
            "skill_libraries": {skills.library_id: settings, "not-selected": {"kind": "invalid"}},
        },
    )
    try:
        assert instance._skill_binding(arm).state == skills.state
        before = instance._skill_binding(arm).to_value()
        # In-memory read-only snapshot remains frozen after source file changes.
        (tmp_path / "selected" / "runtime_state.json").write_text("{}")
        asyncio.run(instance.refresh())
        assert instance._skill_binding(arm).to_value() == before
    finally:
        asyncio.run(instance.close())


def test_no_skill_constructor_never_opens_even_a_configured_snapshot(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("no-skill opened a skill source")

    monkeypatch.setattr(
        "skillev_private.evaluation.integrity_runtime.read_skill_library", forbidden
    )
    instance, _, _ = production_runtime(
        tmp_path,
        monkeypatch,
        config_overrides={
            "skill_libraries": {"unread": {"kind": "evolved", "checkpoint_directory": "/absent"}},
        },
    )
    asyncio.run(instance.close())


def test_context_packing_omission_does_not_count_a_prepared_body_as_exposed():
    from skillev.evaluation.direct_baseline import DirectGenerationRequest
    from skillev.evaluation.integrity_generation import IntegrityGeneration
    from skillev.evaluation.public_context import PromptBlock, PublicPrompt
    from skillev.evaluation.step0_integrity import InterventionCounts
    from skillev.rollout import GenerationPhase
    from tests.evaluation.test_step0_architecture import _Generator, _profile
    from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer

    tokenizer = CleanTokenizer()
    generator = _Generator([(GenerationPhase.ACTION, "42")], tokenizer=tokenizer)
    counts = InterventionCounts()
    generation = IntegrityGeneration(generator, arm_for(snapshot("packing", "Method")), counts)
    required = ({"role": "user", "content": "Synthetic task."},)
    limit = len(tokenizer.encode_integrity_messages(required, enable_thinking=False)) + 10
    prompt = PublicPrompt.required(required).append(
        PromptBlock(
            "tool", "SYNTHETIC_OMITTED_BODY" * 100, required=False, skill_id="synthetic-skill"
        )
    )
    prompt = replace(prompt, maximum_input_tokens=limit)
    asyncio.run(
        generation.call(
            DirectGenerationRequest("case", required, _profile()),
            "case",
            prompt,
            phase=GenerationPhase.ACTION,
            maximum_tokens=16,
        )
    )
    assert counts.skill_blocks_injected == 0
    assert counts.skill_body_tokens == 0
    assert counts.skill_context_omissions == 1
    assert "SYNTHETIC_OMITTED_BODY" not in repr(tokenizer.messages[-1])


@pytest.mark.parametrize("mode", ["completion", "webshop", "alfworld"])
def test_base_tool_contracts_are_identical_with_and_without_skills(mode):
    from skillev.evaluation.capability_registry import native_tool_definitions

    base = native_tool_definitions(mode)
    skilled = native_tool_definitions(mode, skills=True)
    assert all(definition in skilled for definition in base)
    assert {definition["function"]["name"] for definition in skilled} >= {
        "list_skills",
        "read_skill",
        "retrieve_skills",
        "invoke_skill",
    }


def test_trained_forward_route_receives_the_selected_checkpoint_skill_body(tmp_path):
    from tests.evaluation.test_integrity_trained_policies import routed_runtime

    instance, binding, transport, entry = routed_runtime(tmp_path)
    body = "SYNTHETIC_TRAINED_SKILL_BODY: choose a useful public representation."
    skills = snapshot("trained-library", body)
    settings = checkpoint(tmp_path, skills)
    instance.skill_libraries = {skills.library_id: read_skill_library(skills.library_id, settings)}
    instance.config["budgets"][entry.benchmark]["skill_instruction_tokens"] = 5000
    transport.outputs[binding.adapter_name] = ["Final answer: 42."]
    arm = replace(
        arm_for(skills, native=True), optimizer_steps=16, policy_id=binding.policy.snapshot_id
    )
    try:
        final = asyncio.run(instance.generate(entry, arm, "synthetic"))
        assert final.policy_id == binding.policy.snapshot_id
        assert len(transport.calls) == 1
        assert transport.calls[0]["lora_path"] == binding.adapter_name
        assert body in instance.tokenizer.decode(tuple(transport.calls[0]["input_ids"]))
        assert final.intervention_counts["skill_blocks_injected"] == 1
        assert final.intervention_counts["model_calls"] == 1
        instance.validate_candidate(instance.journal, entry, arm, "synthetic")
    finally:
        asyncio.run(instance.close())


@pytest.mark.parametrize("exported", [False, True])
def test_fixed_initial_sources_are_distinct_from_evolved_checkpoint_sources(tmp_path, exported):
    original = snapshot("initial", "Synthetic initial advice.", kind="initial")
    if exported:
        path = tmp_path / "initial-library.json"
        path.write_text(json.dumps(original.state.to_value()))
        source = {"snapshot_file": str(path)}
    else:
        source = {"source": "planned-advisory-seeds@1"}
    loaded = read_skill_library(
        "initial", {"kind": "initial", "retrieval_rule": ACTIVE_APPLICABILITY_RULE, **source}
    )
    assert loaded.kind == "initial"
    assert loaded.source_optimizer_step == 0
    if exported:
        assert loaded.state == original.state
    else:
        assert set(loaded.state.active_skill_ids) == {
            document.manifest.skill_id for document in planned_seed_documents()
        }
    assert decode_integrity_arm(arm_for(loaded).to_value()).skill_library_id == "initial"


def test_model_projection_does_not_disclose_inactive_skill_lineage():
    active = snapshot("lineage", "Synthetic active advice.")
    inactive = _seed_document(
        skill_id="inactive-skill",
        title="Inactive",
        summary="Not active.",
        instructions="SYNTHETIC_INACTIVE_BODY",
        requirements=(SkillRequirement("public", "Use public observations."),),
    )
    state = replace(active.state, documents={**active.state.documents, "inactive-skill": inactive})
    projection = replace(active, state=state).to_value()
    assert "SYNTHETIC_INACTIVE_BODY" not in json.dumps(projection)
    assert (
        FrozenSkillLibrary.from_value(projection).state.active_skill_ids
        == active.state.active_skill_ids
    )


def test_a_weight_only_comparison_cannot_substitute_a_different_library_body():
    from tests.evaluation.test_integrity_trained_policies import policy_arms, policy_controls

    first = snapshot("same-label", "Synthetic first body.")
    second = snapshot("same-label", "Synthetic different body.")
    base, trained = (
        replace(
            arm,
            skill_mode=SkillMode.LIBRARY,
            skill_library_id=first.library_id,
            skill_retrieval_rule=first.retrieval_rule,
        )
        for arm in policy_arms()
    )
    left = replace(policy_controls(base), skills=first.to_value())
    right = replace(policy_controls(trained), skills=second.to_value())
    with pytest.raises(ValueError):
        require_paired_controls(left, right, base, trained)
    assert (
        require_paired_controls(left, replace(right, skills=first.to_value()), base, trained)
        == "forward-policy"
    )


@pytest.mark.parametrize("argument", ["kind", "operation", "unlisted"])
def test_skill_tool_arguments_follow_the_published_interface(argument):
    from skillev.evaluation.native_tool_calls import native_control_payload

    with pytest.raises(ValueError):
        native_control_payload(call("list_skills", **{argument: "review"}))


def test_production_structured_skill_invocation_is_accepted_by_the_same_actor(tmp_path):
    skills = snapshot("production-wire", "Synthetic text skill advice.")
    action = json.dumps(
        {
            "kind": "skill",
            "name": "invoke",
            "resource_id": "skill-runtime",
            "skill_id": "synthetic-skill",
            "arguments": {},
        }
    )
    instance, entry = actor_runtime(tmp_path, skills, [action, "42"])
    try:
        final = asyncio.run(instance.generate(entry, arm_for(skills), "synthetic"))
        assert final.intervention_counts["skill_calls"] == 1
        assert final.intervention_counts["model_calls"] == 2
        assert final.intervention_counts["communication_repairs"] == 0
    finally:
        instance.journal.close()
