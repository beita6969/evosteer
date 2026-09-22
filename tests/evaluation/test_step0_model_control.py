"""Synthetic offline regressions: these do not measure model benchmark quality."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from skillev.evaluation.direct_baseline import (
    DirectGenerationRequest,
    WebShopPublicCatalogCandidate,
)
from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.interactive_tasks import (
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativeInteractiveTask,
    NativePublicState,
    run_native_interactive_task,
)
from skillev.evaluation.direct_baseline.parsing import parse_visible_react
from skillev.evaluation.step0_architecture import (
    StepZeroActionMode,
    StepZeroArchitectureDirectClient,
    StepZeroCompletionMode,
    StepZeroReasoningMode,
)
from skillev.evaluation.step0_completion import native_action_constraint, parse_native_tool_action
from skillev.evaluation.step0_integrity import AgentTopology, InferenceArm, ReasoningPass, SkillMode
from skillev.evaluation.step0_public_state import PublicEpisodeState
from skillev.experiments._evolution_preflight_seed import planned_seed_documents
from skillev.rollout import GenerationPhase
from skillev.runtime import ActionParseStatus
from tests.evaluation.test_step0_architecture import _client as _fixture_client
from tests.evaluation.test_step0_architecture import _Generator, _profile


def _client(generator: _Generator, **kwargs: Any) -> StepZeroArchitectureDirectClient:
    arm = InferenceArm(
        "synthetic-two-pass-clean",
        reasoning_pass=ReasoningPass.REQUIRED,
        agent_topology=AgentTopology.TWO_PASS,
        skill_mode=SkillMode.GENERIC_TEXT if kwargs.get("with_skills", True) else SkillMode.OFF,
    )
    return _fixture_client(generator, arm=arm, **kwargs)


def _interactive_task(benchmark: str, task: str) -> NativeInteractiveTask:
    return NativeInteractiveTask(
        task_id="case-1",
        benchmark=DirectBenchmark(benchmark),
        task=task,
        profile=_profile(),
        max_steps=3,
        prompt_profile_id=f"{benchmark}-step0-skill-owned-react-memory@1",
        parser_profile_id="native-action-memory-v6@1",
        population_id="synthetic",
        run_seed=42,
        history_window_steps=4,
        history_maximum_characters=24000,
        webshop_catalog_candidates=(
            (WebShopPublicCatalogCandidate("first", "First suggestion", 5.0, "first query"),)
            if benchmark == "webshop"
            else ()
        ),
    )


@pytest.mark.parametrize("with_skills", [False, True])
@pytest.mark.parametrize(
    "benchmark",
    [
        "hotpotqa",
        "triviaqa",
        "aime-2026",
        "healthbench",
        "webshop",
        "alfworld",
        "mbpp-plus",
        "humaneval",
    ],
)
def test_all_eight_skill_and_no_skill_routes_reach_model_and_native_output(
    benchmark: str,
    with_skills: bool,
) -> None:
    interactive = benchmark in {"webshop", "alfworld"}
    code = benchmark in {"mbpp-plus", "humaneval"}
    thinking = code or benchmark == "aime-2026"
    output = (
        "Action: click[second]"
        if benchmark == "webshop"
        else "Action: look"
        if benchmark == "alfworld"
        else "Final answer: 042"
        if benchmark == "aime-2026"
        else "```python\ndef solve():\n    return 2\n```"
        if code
        else "Synthetic answer"
    )
    generator = _Generator(
        [
            (GenerationPhase.REASONING, "Choose my own approach; the advice can be revised."),
            (GenerationPhase.ACTION, output),
        ]
    )
    mode = (
        StepZeroActionMode.WEB_SHOP
        if benchmark == "webshop"
        else StepZeroActionMode.ALF_WORLD
        if benchmark == "alfworld"
        else StepZeroActionMode.COMPLETION
    )
    completion = (
        StepZeroCompletionMode.PYTHON_SOURCE
        if code
        else StepZeroCompletionMode.AIME_BOXED_INTEGER
        if benchmark == "aime-2026"
        else StepZeroCompletionMode.NATURAL_LANGUAGE
        if benchmark == "healthbench"
        else StepZeroCompletionMode.SHORT_ANSWER
    )
    client = _client(
        generator,
        mode=mode,
        benchmark=benchmark,
        completion_mode=completion,
        reasoning_mode=(
            StepZeroReasoningMode.QWEN_THINKING if thinking else StepZeroReasoningMode.STANDARD
        ),
        with_skills=with_skills,
        skill_instruction_token_budget=4096,
    )
    if interactive:
        actions = (
            ("click[first]", "click[second]")
            if benchmark == "webshop"
            else ("look", "go to shelf 2")
        )
        asyncio.run(
            client.begin_interactive_episode(
                _interactive_task(benchmark, "public root task"),
                NativePublicState("Current state", actions),
            )
        )
    result = asyncio.run(
        client.generate(
            DirectGenerationRequest(
                "case-1:step:1" if interactive else "case-1",
                ({"role": "user", "content": "Synthetic public task only"},),
                _profile(enable_thinking=thinking),
            )
        )
    )
    assert result.text
    assert len(generator.profiles) == 2
    assert client.diagnostics.generations == 2
    assert client.diagnostics.public_catalog_actions == 0
    assert bool(client.architecture_identity.seed_skill_ids) is with_skills
    assert bool(client._implementation.counts.skill_blocks_injected) is with_skills
    initial_body = planned_seed_documents()[0].instructions
    assert (initial_body in "\n".join(generator.tokenizer.thinking_prompts)) is with_skills
    if interactive:
        action = parse_visible_react(result.text).action.value
        assert action == ("click[second]" if benchmark == "webshop" else "look")
        asyncio.run(
            client.observe_interactive_episode(
                "case-1",
                action,
                NativeEnvironmentStep("Done", True, 1.0, True, True),
            )
        )
        asyncio.run(
            client.finish_interactive_episode("case-1", NativeEnvironmentOutcome(1, True, True))
        )
    assert not generator.active_episodes


def test_live_state_and_free_form_messages_survive_without_narrowing_actions() -> None:
    notes = "Goal: revise my plan\nAction: do not click[Buy Now]\n通信\uff1a先检查另一件商品。"
    generator = _Generator(
        [
            (GenerationPhase.REASONING, notes),
            (GenerationPhase.ACTION, "Action: 点击[second]"),
            (GenerationPhase.REASONING, "Now make my own second decision."),
            (GenerationPhase.ACTION, "Action: click[Back to Search]"),
        ]
    )
    root = "public task " * 120 + "MANDATORY FINAL QUALIFIER"
    client = _client(
        generator, mode=StepZeroActionMode.WEB_SHOP, benchmark="webshop", root_query=root
    )
    asyncio.run(
        client.begin_interactive_episode(
            _interactive_task("webshop", root),
            NativePublicState("Initial public page", ("click[first]", "click[second]")),
        )
    )

    def request(turn: int) -> DirectGenerationRequest:
        return DirectGenerationRequest(
            f"case-1:step:{turn}",
            ({"role": "user", "content": "LEGACY FORCED STRATEGY"},),
            _profile(),
        )

    first = asyncio.run(client.generate(request(1)))
    assert parse_visible_react(first.text).action.value == "click[second]"
    assert first.reasoning_text == notes
    actions = ("click[< Prev]", "click[Buy Now]", "click[Back to Search]")
    asyncio.run(
        client.observe_interactive_episode(
            "case-1",
            "click[second]",
            NativeEnvironmentStep(
                "LATEST PUBLIC PAGE", False, 0.7319, False, True, available_actions=actions
            ),
        )
    )
    memory = client.current_interactive_memory("case-1")
    assert root in memory
    assert notes in memory
    assert "LATEST PUBLIC PAGE" in memory
    second = asyncio.run(client.generate(request(2)))
    assert parse_visible_react(second.text).action.value == "click[Back to Search]"
    assert all(
        "LEGACY FORCED STRATEGY" not in prompt for prompt in generator.tokenizer.thinking_prompts
    )
    latest_prompt = generator.tokenizer.thinking_prompts[-1]
    assert "LATEST PUBLIC PAGE" in latest_prompt
    assert "MANDATORY FINAL QUALIFIER" in latest_prompt
    assert "0.7319" not in latest_prompt
    # Mentioning (even negating) an action in reasoning must not reduce the surface to it.
    assert generator.constraints[-1] is None
    assert all(action in latest_prompt for action in actions)


@pytest.mark.parametrize(
    "text",
    [
        "Action: click[item-2]",
        "点击[item-2]",
        '{"kind":"tool","resource_id":"webshop","name":"点击","arguments":{"target":"item-2"}}',
    ],
)
def test_native_and_json_tool_wires_preserve_the_exact_argument(text: str) -> None:
    result = parse_native_tool_action("webshop", text)
    assert result.status is ActionParseStatus.VALID
    assert result.action is not None
    assert result.action.name == "click"
    assert result.action.arguments == {"target": "item-2"}


def test_ambiguous_actions_are_not_repaired_into_a_favorable_choice() -> None:
    result = parse_native_tool_action("webshop", "Action: click[first]\nAction: click[second]")
    assert result.action is None
    schema = native_action_constraint(
        "webshop", available_actions=("click[< Prev]", "click[Back to Search]")
    ).json_schema
    assert "Back to Search" in json.dumps(schema)


def test_history_boundaries_keep_complete_latest_observation_and_model_notes() -> None:
    state = PublicEpisodeState("full task", history_window_steps=2, history_maximum_characters=20)
    state = state.observe("look", "old observation").observe(
        "look", "latest object 123 at shelf 234"
    )
    rendered = state.render()
    assert "full task" in rendered
    assert "latest object 123 at shelf 234" in rendered
    assert "old observation" not in rendered


def test_direct_no_skill_runner_uses_the_same_native_verb_boundary() -> None:
    class Client:
        async def generate(self, request):
            from skillev.evaluation.direct_baseline import DirectGenerationResult

            return DirectGenerationResult(request.request_id, "点击[item-2]", "stop", 1, 1)

    class Environment:
        async def reset(self):
            return NativePublicState("Public product", ("click[item-2]",))

        async def step(self, action):
            assert action == "click[item-2]"
            return NativeEnvironmentStep("Done", True, 0.5, False, True)

        async def outcome(self):
            return NativeEnvironmentOutcome(0.5, False, True)

        async def close(self):
            pass

    attempt = asyncio.run(
        run_native_interactive_task(
            Client(), _interactive_task("webshop", "public task"), Environment()
        )
    )
    assert attempt.reward == 0.5
    assert attempt.success is False
    assert attempt.infrastructure_error is None
