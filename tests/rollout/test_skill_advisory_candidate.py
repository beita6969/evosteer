"""Synthetic capability boundaries; no private tasks or required skill counts."""

import asyncio
import json
from dataclasses import replace

from skillev.contracts import canonical_json
from skillev.evolution.retriever import TaskRetrievalFeatures, applicability_matches
from skillev.experiments._evolution_preflight_seed import planned_seed_documents
from skillev.policy.observed_history import history_facts
from skillev.policy.phase_context import (
    OBSERVED_PUBLIC_HISTORY,
    PhaseContextSpec,
    phase_chat_messages,
    skill_resource_instruction,
)
from skillev.policy.typed_history import _observed_state
from skillev.rollout.catalog import CatalogReadEnvironment
from skillev.runtime import ActionKind, StructuredAction
from skillev.scoring.rendering import render_forward_prefix_from_parts
from tests.rollout.engine_fakes import default_request
from tests.rollout.test_typed_alfworld_history import final_state


def test_candidates_are_new_optional_public_procedures_and_old_library_is_unchanged():
    old = planned_seed_documents()
    new = planned_seed_documents("public-procedure-advice@3")
    assert not {d.manifest.skill_id for d in old} & {d.manifest.skill_id for d in new}
    assert planned_seed_documents() == old
    assert all(d.manifest.license_id == "CC0-1.0" for d in new)
    assert all(len(d.instructions) > 500 for d in new)
    assert all(d.applicability.contexts != ("*",) for d in new)
    for domain in (
        "hotpotqa",
        "triviaqa",
        "aime-2026",
        "healthbench",
        "alfworld",
        "mbpp-plus",
        "humaneval",
    ):
        features = TaskRetrievalFeatures(
            "synthetic", f"{domain}/task", f"{domain}:task", (), domain, "public query"
        )
        offered = [
            d
            for d in new
            if applicability_matches(d.applicability, features, skill_id=d.manifest.skill_id)
        ]
        assert offered  # public split-independent scope, not a call quota
    assert all(
        not applicability_matches(
            d.applicability, replace(features, context="unrelated"), skill_id=d.manifest.skill_id
        )
        for d in new
    )


def test_read_returns_advice_without_executing_or_certifying_environment():
    class Environment:
        environment_id = "synthetic"
        task_family = "synthetic/task"

        async def execute(self, *args, **kwargs):
            raise AssertionError("a read must not execute the environment")

    skill = default_request().retrieved_skills[0]
    reader = CatalogReadEnvironment(Environment(), (skill,), "library", advisory_semantics=True)
    action = StructuredAction(
        kind=ActionKind.SKILL,
        name="invoke",
        arguments={},
        resource_id="skill-runtime",
        skill_id=skill.metadata.skill_id,
    )
    result = asyncio.run(reader.execute(action, step_index=1))
    assert result.public_value["content"] == skill.content
    assert result.public_value["environment_action_executed"] is False
    assert result.public_value["environment_state_checked"] is False
    assert result.terminal is False
    assert result.budget_usage.tool_calls == 1
    again = asyncio.run(reader.execute(action, step_index=2))
    assert again.budget_usage == result.budget_usage  # no silent discount/deduplication


def test_actual_terminal_unknown_and_false_are_not_replaced_by_successful_reads():
    observation = {"text": "A public object is visible.", "admissible_commands": ["look"]}
    assert (
        _observed_state(canonical_json(observation), preserve_terminal=True)[
            "native_episode_terminal"
        ]
        is None
    )
    spec = PhaseContextSpec(
        history_format=OBSERVED_PUBLIC_HISTORY,
        initial_public_state_json=canonical_json({**observation, "terminal": False}),
        max_turns=25,
    )
    prefix = render_forward_prefix_from_parts(
        spec.wrap("Public synthetic goal"), (), 1, "I think I finished"
    )
    messages, _ = phase_chat_messages(prefix.text)
    assert final_state(messages)["latest_environment_state"]["native_episode_terminal"] is False
    assert "I think I finished" not in messages[-1]["content"]
    read = {"status": "skill-read", "skill_id": "advice", "version": "1", "library_version": "lib"}
    rows = [
        {"step": i, "execution_feedback": {"status": "success", "observation": json.dumps(read)}}
        for i in (1, 2)
    ]
    facts = history_facts(rows)
    assert facts["skill_reads_used"] == 2
    assert facts["repeat_same_version_read_count"] == 1
    assert facts["native_environment_observation_count"] == 0
    assert facts["last_native_command"] is None


def test_optional_early_resource_choice_is_versioned_and_has_no_unscored_route():
    spec = PhaseContextSpec(
        skill_advice=True,
        tools_json=canonical_json([{"function": {"name": "read_skill"}}]),
        reasoning_token_cap=1024,
        action_token_cap=2048,
    )
    decoded, body = PhaseContextSpec.split(spec.wrap("task"))
    assert decoded == spec
    assert body == "task"
    instruction = skill_resource_instruction(spec, reasoning=True)
    assert "need not solve the whole task" in instruction
    assert "action phase" in instruction
    assert "no call quota" in instruction
    assert skill_resource_instruction(replace(spec, skill_advice=False), reasoning=True) == ""
    assert skill_resource_instruction(replace(spec, tools_json="[]"), reasoning=True) == ""
