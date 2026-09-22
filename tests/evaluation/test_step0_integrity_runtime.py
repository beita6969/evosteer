"""Synthetic integrity regressions; no licensed data or inference service."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from skillev.evaluation.agent_messages import AgentMessage, EpisodeMailbox, MessageKind
from skillev.evaluation.decision_transport import (
    DecisionChannel,
    ExplicitDecision,
    PublicSurface,
    normalize_decision,
    public_repair_feedback,
)
from skillev.evaluation.direct_baseline import DirectGenerationRequest
from skillev.evaluation.integrity_controller import IntegrityStepZeroClient
from skillev.evaluation.native_tool_calls import NativeTools
from skillev.evaluation.sealed_candidates import CandidateJournal, CandidateReader, FinalCandidate
from skillev.evaluation.step0_integrity import (
    AgentTopology,
    InferenceArm,
    ReasoningPass,
    SkillMode,
    load_integrity_arm,
)
from skillev.evaluation.step0_public_state import ExecutionStatus, PublicEpisodeState
from skillev.evaluation.step0_types import (
    StepZeroActionMode,
    StepZeroArchitectureConfig,
    StepZeroCompletionMode,
    StepZeroTaskBinding,
)
from skillev.rollout import GenerationPhase
from skillev.runtime import SkillLibraryState
from tests.evaluation.test_step0_architecture import _Generator, _profile, _Tokenizer


@dataclass(frozen=True, slots=True)
class CleanTokenizer(_Tokenizer):
    messages: list[tuple[tuple[dict[str, str], ...], bool]] = field(default_factory=list)

    def encode_integrity_messages(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        enable_thinking: bool,
        tools: NativeTools = (),
    ) -> list[int]:
        self.messages.append((messages, enable_thinking))
        return self.encode(repr(messages) + (json.dumps(tools) if tools else ""))


def clean_client(
    generator: _Generator, arm: InferenceArm, *, benchmark: str = "aime-2026"
) -> IntegrityStepZeroClient:
    return IntegrityStepZeroClient(
        generator=generator,
        library_state=SkillLibraryState.from_seed_documents(()),
        bindings=(
            StepZeroTaskBinding(
                task_id="case",
                benchmark_id=benchmark,
                task_family="public",
                panel_index=0,
                action_mode=StepZeroActionMode.COMPLETION,
                completion_mode=StepZeroCompletionMode.AIME_BOXED_INTEGER,
            ),
        ),
        config=StepZeroArchitectureConfig(3, 256, 256, 4096, 0, "Qwen3.5-9B", "synthetic", arm=arm),
    )


@pytest.mark.parametrize("thinking", [False, True])
def test_aime_thinking_is_configurable_without_a_serializer_or_reference_fallback(
    thinking: bool,
) -> None:
    tokenizer = CleanTokenizer()
    response = "<think>calculation</think>42" if thinking else "42"
    generator = _Generator([(GenerationPhase.ACTION, response)], tokenizer=tokenizer)
    client = clean_client(generator, InferenceArm("clean-aime", native_thinking=thinking))
    request = DirectGenerationRequest(
        "case", ({"role": "user", "content": "Compute the requested integer."},), _profile()
    )
    result = asyncio.run(client.generate(request))
    assert result.text == r"\boxed{42}"
    assert asyncio.run(client.generate(request)) == result
    assert len(generator.profiles) == 1
    assert generator.profiles[0].enable_thinking is thinking
    assert tokenizer.messages[0][1] is thinking
    assert client.counts.extra_model_calls_for_serialization == 0
    assert client.architecture_identity.seed_skill_ids == ()
    assert "RetrievedSkills" not in repr(tokenizer.messages)


def test_explicit_two_pass_is_not_called_multi_agent() -> None:
    generator = _Generator(
        [(GenerationPhase.REASONING, "Check the calculation."), (GenerationPhase.ACTION, "42")],
        tokenizer=CleanTokenizer(),
    )
    arm = InferenceArm(
        "two-pass", reasoning_pass=ReasoningPass.REQUIRED, agent_topology=AgentTopology.TWO_PASS
    )
    client = clean_client(generator, arm)
    asyncio.run(
        client.generate(
            DirectGenerationRequest("case", ({"role": "user", "content": "Compute."},), _profile())
        )
    )
    assert client.counts.model_calls == 2
    assert client.architecture_identity.inference_arm is not None
    assert client.architecture_identity.inference_arm["agent_topology"] == "two-pass-single"


def test_no_skill_never_constructs_production_skill_access(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = _Generator([(GenerationPhase.ACTION, "42")], tokenizer=CleanTokenizer())

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("no-skill read the advice catalog")

    monkeypatch.setattr("skillev.evolution.skill_access.ReadOnlySkillAccess.__init__", forbidden)
    client = clean_client(generator, InferenceArm("off"))
    asyncio.run(
        client.generate(
            DirectGenerationRequest("case", ({"role": "user", "content": "Compute."},), _profile())
        )
    )
    assert client.counts.skill_body_tokens == 0


def test_historical_multi_agent_arm_is_readable_but_not_executable() -> None:
    arm = InferenceArm("historical-multi", agent_topology=AgentTopology.MULTI_AGENT)
    generator = _Generator([], tokenizer=CleanTokenizer())
    with pytest.raises(ValueError):
        clean_client(generator, arm)
    assert not generator.profiles
    assert arm.to_value()["agent_topology"] == "multi-agent"


@pytest.mark.parametrize(
    "action",
    [
        "Action: look",
        'Action: act(command="look")',
        'Action: act\nArgument: command: "look"',
        'Action: act\nArgument: command="look"',
        'Action: act with command "look"',
        'Action: act with command "look".',
        'Action: act\n{"kind":"tool","resource_id":"alfworld",'
        '"name":"act","arguments":{"command":"look"}}',
        "Action: `look`",
    ],
)
def test_review_is_not_executed_and_explicit_native_action_remains_available(action: str) -> None:
    from skillev.evaluation.direct_baseline.interactive_tasks import NativePublicState
    from tests.evaluation.test_step0_architecture import _client
    from tests.evaluation.test_step0_model_control import _interactive_task

    tokenizer = CleanTokenizer()
    generator = _Generator(
        [
            (GenerationPhase.ACTION, '{"kind":"review","body":"I will look around."}'),
            (GenerationPhase.ACTION, action),
        ],
        tokenizer=tokenizer,
    )
    client = _client(
        generator,
        mode=StepZeroActionMode.ALF_WORLD,
        benchmark="alfworld",
        arm=InferenceArm("single-native"),
        with_skills=False,
    )
    asyncio.run(
        client.begin_interactive_episode(
            _interactive_task("alfworld", "Synthetic public task"),
            NativePublicState("Unchanged initial observation", ("look", "inventory")),
        )
    )
    response = asyncio.run(
        client.generate(
            DirectGenerationRequest(
                "case-1:step:1", ({"role": "user", "content": "Synthetic task"},), _profile()
            )
        )
    )
    assert response.text == "Action: look"
    assert generator.constraints == [None, None]
    assert client.architecture_identity.intervention_counts["communication_repairs"] == 0
    assert "no environment tool was invoked" in repr(tokenizer.messages[-1])
    assert "Unchanged initial observation" in repr(tokenizer.messages[-1])
    # Caller has not executed the selected action yet.
    assert client.architecture_identity.intervention_counts["tool_calls"] == 0


@pytest.mark.parametrize(
    "text",
    [
        "Do not click[Buy Now]",
        "Example: click[Buy Now]",
        "click[Next] then click[Buy Now]",
        "Action: click[Next]\nAction: click[Buy Now]",
        "Example:\nAction: click[Buy Now]",
        "Do not execute this:\nAction: click[Buy Now]",
        "Example:\n```text\nAction: click[Buy Now]\n```",
        "> Action: click[Buy Now]",
        "Action: click[Next]\nDo not execute this.",
        "click[Next]\nAction: click[Buy Now]",
        'Action: act\nArgument: {"command": "click[Buy Now]"}',
        '{"kind":"tool","resource_id":"webshop","name":"act",'
        '"arguments":{"command":"click[Buy Now]"}}',
        "click[buy now]",
    ],
)
def test_mentions_multiple_calls_and_target_rewrites_are_not_actions(text: str) -> None:
    result = normalize_decision(
        ExplicitDecision(text, 3), PublicSurface("webshop", 3, ("click[Buy Now]", "click[Next]"))
    )
    assert result.action is None


def test_alias_is_syntax_only_and_revision_and_channel_are_enforced() -> None:
    surface = PublicSurface(
        "webshop", 3, ("click[Blue XL]", "click[Back to Search]", "click[< Prev]")
    )
    decision = ExplicitDecision("点击[Blue XL]", 3)
    assert normalize_decision(decision, surface).action == "click[Blue XL]"
    assert normalize_decision(replace(decision, state_revision=2), surface).action is None
    assert (
        normalize_decision(replace(decision, channel=DecisionChannel.MESSAGE), surface).action
        is None
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I will examine the visible page.\n\nAction: click[Blue XL]", "click[Blue XL]"),
        ('Action: click\nArgument: {"target": "Blue XL"}', "click[Blue XL]"),
        ('Action: 搜索\nArguments: {"query": "blue shoes"}', "search[blue shoes]"),
        ("```text\nAction: click[Blue XL]\n```", "click[Blue XL]"),
        ('Action: search(query="blue shoes")', "search[blue shoes]"),
        ('search("blue shoes")', "search[blue shoes]"),
        ('webshop.search(query="blue shoes")', "search[blue shoes]"),
        ('Action: 点击(target="Blue XL")', "click[Blue XL]"),
        ('Action: search\nArgument: query: "blue shoes"', "search[blue shoes]"),
        ('Action: search\nArgument: query="blue shoes"', "search[blue shoes]"),
        ('Action: search\nArgument: query = "blue shoes"', "search[blue shoes]"),
        ('Action: search with query "blue shoes"', "search[blue shoes]"),
        ('Action: search[query] with query "blue shoes"', "search[blue shoes]"),
        ('Action: 点击[target] with target "Blue XL"', "click[Blue XL]"),
        ('Action: search query="blue shoes"', "search[blue shoes]"),
        ('Action: search for "blue shoes"', "search[blue shoes]"),
        ('Action: Search for "blue shoes".', "search[blue shoes]"),
        ('Action: search\nArgument: query="blue shoes".', "search[blue shoes]"),
        ('Action: search with query "blue shoes.".', "search[blue shoes.]"),
        ('Action: search "blue shoes"', "search[blue shoes]"),
        ('Action: click on "Blue XL"', "click[Blue XL]"),
        ('Action: click target: "Blue XL"', "click[Blue XL]"),
        ("Action: search\nArgument: query: blue shoes", "search[blue shoes]"),
        ("Action: search\nquery: blue shoes", "search[blue shoes]"),
        ("Action: search\nArgument: blue shoes", "search[blue shoes]"),
        ('Action: search\nArgument: "blue shoes"', "search[blue shoes]"),
        ("Action: click\ntarget: Blue XL", "click[Blue XL]"),
        ('Action: click\nArgument: target: "Blue XL"', "click[Blue XL]"),
        ("Action: `click[Blue XL]`", "click[Blue XL]"),
        ('```python\nsearch(query="blue shoes")\n```', "search[blue shoes]"),
        (
            'Action: click[Blue XL]\n\n{"kind":"tool","resource_id":"webshop",'
            '"name":"click","arguments":{"target":"Blue XL"}}',
            "click[Blue XL]",
        ),
        (
            'Action: click(target="Blue XL")\n\n```json\n'
            '{"kind":"tool","resource_id":"webshop",'
            '"name":"click","arguments":{"target":"Blue XL"}}\n```',
            "click[Blue XL]",
        ),
        (
            'Action: 点击[Blue XL]\n{"kind":"tool","resource_id":"webshop",'
            '"name":"click","arguments":{"target":"Blue XL"}}',
            "click[Blue XL]",
        ),
        (
            'Action: search\n{"kind":"tool","resource_id":"webshop",'
            '"name":"search","arguments":{"query":"blue shoes"}}',
            "search[blue shoes]",
        ),
        (
            'Action: 点击\n```json\n{"kind":"tool","resource_id":"webshop",'
            '"name":"click","arguments":{"target":"Blue XL"}}\n```',
            "click[Blue XL]",
        ),
        (
            'Action: Search for "blue shoes".\n{"kind":"tool","resource_id":"webshop",'
            '"name":"search","arguments":{"query":"blue shoes"}}',
            "search[blue shoes]",
        ),
    ],
)
def test_unique_explicit_action_fields_preserve_intent_and_arguments(
    text: str, expected: str
) -> None:
    surface = PublicSurface("webshop", 3, ("click[Blue XL]", "search"))
    assert normalize_decision(ExplicitDecision(text, 3), surface).action == expected


@pytest.mark.parametrize(
    "text",
    [
        'search(query="blue shoes")',
        'Action: search\nArgument: query: "blue shoes"',
        'click(target="blue xl")',
        'act(command="click[Blue XL]")',
        'another_resource.click(target="Blue XL")',
        'click(target="Blue XL", extra="value")',
        'click(target="Blue" + " XL")',
        'click(target=str("Blue XL"))',
        'click(**{"target":"Blue XL"})',
        'click(target="Blue XL"); click(target="Other")',
        'click(target="Blue XL")\nAction: click[Other]',
        "`click[Blue XL]`\nAction: click[Other]",
        'Action: click\nArgument: target: "Blue XL"\nquery: "Other"',
        'Action: click[Blue XL]\n{"kind":"tool","resource_id":"webshop",'
        '"name":"click","arguments":{"target":"Other"}}',
        'Example:\nAction: click(target="Blue XL")',
        'Do not execute this:\nAction: click(target="Blue XL")',
        'Action: search with query "blue shoes"',
        'Action: click with target "Blue XL" + "Other"',
        'Action: click target="Blue XL"\nAction: click[Other]',
        'Action: click\nArgument: target="Blue XL", query="Other"',
        'Action: click\nArgument: target = str("Blue XL")',
        'Action: search\n{"kind":"tool","resource_id":"webshop",'
        '"name":"click","arguments":{"target":"Blue XL"}}',
        'Action: click\n{"kind":"tool","resource_id":"another_resource",'
        '"name":"click","arguments":{"target":"Blue XL"}}',
        'Action: choose the first item\n{"kind":"tool","resource_id":"webshop",'
        '"name":"click","arguments":{"target":"Blue XL"}}',
        'Do not execute this:\nAction: click\n{"kind":"tool","resource_id":"webshop",'
        '"name":"click","arguments":{"target":"Blue XL"}}',
    ],
)
def test_call_notation_does_not_expand_permissions_evaluate_code_or_choose_between_calls(
    text: str,
) -> None:
    surface = PublicSurface("webshop", 3, ("click[Blue XL]", "click[Other]"))
    assert normalize_decision(ExplicitDecision(text, 3), surface).action is None


def test_call_notation_preserves_literal_punctuation_and_escaped_quotes() -> None:
    surface = PublicSurface("webshop", 3, ("search",))
    result = normalize_decision(
        ExplicitDecision(r'search(query="a \"blue\" mug (XL)")', 3), surface
    )
    assert result.action == 'search[a "blue" mug (XL)]'


@pytest.mark.parametrize(
    "text",
    [
        'Action: search\nArgument: query = str("blue shoes")',
        'Action: search\nArgument: query = "blue" + " shoes"',
        'Action: search with query "blue" + " shoes"',
        'Action: search[query] with query "blue" + " shoes"',
        'Action: search[query] with target "blue shoes"',
        'Action: search[target] with target "blue shoes"',
        'Action: search[query] with query "blue shoes" or "red shoes"',
        'Action: search[query] with query "blue shoes"\nAction: search[red shoes]',
        'Do not execute this:\nAction: search[query] with query "blue shoes"',
        'Example:\nAction: search[query] with query "blue shoes"',
        'Action: search for "blue shoes" or "red shoes"',
        'Action: search for "blue shoes" then click[Other]',
        'Do not execute this:\nAction: search for "blue shoes"',
        'Action: search not for "blue shoes"',
        'search for "blue shoes"\nAction: search[red shoes]',
        'Action: search for "blue shoes"..',
        'Action: search for "blue shoes". or "red shoes".',
        'Action: search for "blue shoes".\nDo not execute this.',
    ],
)
def test_literal_search_cannot_turn_expressions_or_multiple_intents_into_a_query(text: str) -> None:
    surface = PublicSurface("webshop", 3, ("search",))
    assert normalize_decision(ExplicitDecision(text, 3), surface).action is None


def test_alfworld_analysis_does_not_make_a_valid_final_command_unavailable() -> None:
    surface = PublicSurface("alfworld", 3, ("look", "go to shelf 2"))
    response = "I can inspect the room, then decide what to do.\n\nAction: go to shelf 2"
    assert normalize_decision(ExplicitDecision(response, 3), surface).action == "go to shelf 2"
    assert normalize_decision(ExplicitDecision(response, 2), surface).action is None


def test_unsubmitted_prose_gets_representation_feedback_without_mining_its_last_command() -> None:
    surface = PublicSurface("alfworld", 3, ("look", "inventory"))
    response = (
        "I could look around first, but I am still discussing the possibilities.\n\ninventory"
    )
    result = normalize_decision(ExplicitDecision(response, 3), surface)
    assert result.action is None
    feedback = public_repair_feedback(result, surface)
    assert "Action:" in feedback
    assert "Discussion alone" in feedback
    assert normalize_decision(ExplicitDecision("inventory", 3), surface).action == "inventory"
    assert (
        normalize_decision(ExplicitDecision("Action: inventory", 3), surface).action == "inventory"
    )


def test_wrong_resource_feedback_names_only_the_public_resource() -> None:
    surface = PublicSurface("alfworld", 3, ("inventory",))
    result = normalize_decision(
        ExplicitDecision(
            '{"kind":"tool","resource_id":"different_resource","name":"act",'
            '"arguments":{"command":"inventory"}}',
            3,
        ),
        surface,
    )
    assert result.action is None
    assert 'resource_id is "alfworld"' in public_repair_feedback(result, surface)


def test_state_keeps_full_task_and_failed_action_is_not_a_fact() -> None:
    task = "full public task " * 600
    state = PublicEpisodeState(task).observe(
        "heat apple 1", "Could not heat it.", execution_status=ExecutionStatus.REJECTED
    )
    assert task in state.render()
    assert state.history[0].execution_status is ExecutionStatus.REJECTED
    assert "Executed action" not in state.render()
    assert state.revision == 1


def test_messages_round_trip_and_route_without_turning_opinions_into_actions() -> None:
    mailbox = EpisodeMailbox("r", "a", "e", ("owner", "peer"))
    request = AgentMessage(
        "r", "a", "e", "m1", "owner", "peer", MessageKind.REQUEST, "Check this.\n\\frac{a}{b}", 1
    )
    mailbox.deliver(request, current_revision=1)
    reply = AgentMessage(
        "r",
        "a",
        "e",
        "m2",
        "peer",
        "owner",
        MessageKind.OPINION,
        "点击[Blue XL]\n```python\nx = '中文'\n```",
        1,
        parent_id="m1",
    )
    delivered = mailbox.deliver(reply, current_revision=2)
    assert delivered.body == reply.body
    assert delivered.late
    assert mailbox.deliver(reply, current_revision=2) is delivered
    with pytest.raises(ValueError):
        mailbox.deliver(replace(reply, message_id="m3", episode_id="another"), current_revision=2)
    with pytest.raises(ValueError):
        mailbox.deliver(replace(reply, message_id="m4", recipient="peer"), current_revision=2)


def test_final_is_durable_and_cannot_be_replaced_by_a_vote_or_baseline(tmp_path: Path) -> None:
    path = tmp_path / "private" / "candidates.sqlite"
    journal = CandidateJournal(path)
    candidate = FinalCandidate(
        "r", "a", "e", "qwen-policy", "owner-final", "wrong answer", "frozen-parser", 10, 2
    )
    journal.seal(candidate)
    journal.close()
    reopened = CandidateJournal(path)
    with pytest.raises(ValueError):
        reopened.seal(replace(candidate, text="better baseline answer"))
    reader = CandidateReader(path)
    assert reader.get("r", "a", "e") == candidate
    reader.close()
    reopened.close()


def test_unknown_execution_cannot_be_replayed_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "actions.sqlite"
    scope = ("r", "a", "e")
    journal = CandidateJournal(path)
    journal.intent(scope, "d1", "click[Buy Now]", 2)
    journal.close()
    journal = CandidateJournal(path)
    with pytest.raises(RuntimeError):
        journal.intent(scope, "d2", "click[Buy Now]", 2)
    journal.acknowledge(scope, "d1", status="confirmed", observation="Purchase complete.")
    with pytest.raises(ValueError):
        journal.intent(scope, "d1", "click[Buy Now]", 2)
    journal.close()


def test_primary_config_is_literal_no_thinking_and_no_skills() -> None:
    arm = load_integrity_arm(Path("configs/evaluation/step0_integrity_v2.yaml"))
    arm.validate_primary_no_skill()
    assert arm.skill_mode is SkillMode.OFF
    assert not arm.native_thinking
