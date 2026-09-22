from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from skillev.evaluation.direct_baseline.client import (
    DirectGenerationRequest,
    DirectGenerationResult,
)
from skillev.evaluation.direct_baseline.config import DirectBenchmark, DirectDecodingProfile
from skillev.evaluation.direct_baseline.context_budget import (
    ContextBudget,
    newest_contiguous_history,
)
from skillev.evaluation.direct_baseline.interactive_tasks import (
    InvalidCandidatePolicy,
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativeInteractiveTask,
    NativePublicState,
    run_native_interactive_task,
)
from skillev.evaluation.direct_baseline.parsing import parse_visible_react
from skillev.evaluation.direct_baseline.prompts import (
    PublicInteractiveTurn,
    render_interactive_messages,
)
from skillev.evaluation.interactive_prompt_assets import (
    InteractivePromptAsset,
    InteractivePromptExample,
    InteractiveReplayStep,
)


def _profile() -> DirectDecodingProfile:
    return DirectDecodingProfile("memory@1", False, 0.7, 0.8, 20, 0, 1.5, 1, 200)


@dataclass
class _RecordingClient:
    outputs: list[str]
    requests: list[DirectGenerationRequest] = field(default_factory=list)

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        self.requests.append(request)
        return DirectGenerationResult(request.request_id, self.outputs.pop(0), "stop", 1, 1)


@dataclass
class _TwoStepEnvironment:
    actions: list[str] = field(default_factory=list)
    step_count: int = 0

    async def reset(self) -> NativePublicState:
        return NativePublicState("start state", ("search[item]",))

    async def step(self, action: str) -> NativeEnvironmentStep:
        self.actions.append(action)
        self.step_count += 1
        if self.step_count == 1:
            return NativeEnvironmentStep(
                "returned state one",
                False,
                0,
                False,
                True,
                available_actions=("click[item]",),
            )
        return NativeEnvironmentStep("done", True, 1, True, True)

    async def outcome(self) -> NativeEnvironmentOutcome:
        return NativeEnvironmentOutcome(0, False, False)

    async def close(self) -> None:
        return None


def _task(*, max_steps: int = 2) -> NativeInteractiveTask:
    return NativeInteractiveTask(
        "webshop:test",
        DirectBenchmark.WEB_SHOP,
        "buy an item under 20 dollars",
        _profile(),
        max_steps,
        prompt_profile_id="webshop-native-react-memory-v5@1",
        parser_profile_id="native-action-memory-v5@1",
        population_id="test",
        run_seed=42,
        invalid_candidate_policy=InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
        history_window_steps=None,
        history_maximum_characters=None,
    )


def _response(memory: str, action: str, thought: str = "advance the goal") -> str:
    return f"Memory:\n{memory}\n\nThought:\n{thought}\n\nAction:\n{action}"


def test_returned_state_and_memory_enter_the_next_turn_without_action_rewrite() -> None:
    client = _RecordingClient(
        [
            _response("Ready to purchase: no", "search[item]"),
            _response("Ready to purchase: yes", "click[item]"),
        ]
    )
    environment = _TwoStepEnvironment()
    attempt = asyncio.run(run_native_interactive_task(client, _task(), environment))
    second_prompt = client.requests[1].messages[-1]["content"]
    assert (
        "Accumulated structured memory before this action:\nReady to purchase: no" in second_prompt
    )
    assert "Environment returned state: returned state one" in second_prompt
    assert "click[item]" in second_prompt
    assert (
        "Admissible actions:\n- search[item]\n\nAdmissible actions before the action:"
        not in second_prompt
    )
    assert environment.actions == ["search[item]", "click[item]"]
    assert attempt.trace[0].observation_before == "start state"
    assert attempt.trace[0].available_actions_before == ("search[item]",)
    assert attempt.trace[0].structured_memory_after == "Ready to purchase: no"


def test_missing_memory_carries_previous_memory_and_does_not_invalidate_action() -> None:
    client = _RecordingClient(
        [
            _response("Ready to purchase: no", "search[item]"),
            "Thought:\ninspect it\n\nAction:\nclick[item]",
        ]
    )
    attempt = asyncio.run(run_native_interactive_task(client, _task(), _TwoStepEnvironment()))
    assert attempt.success is True
    assert attempt.trace[1].memory_update_status == "missing-carried"
    assert attempt.trace[1].structured_memory_after == "Ready to purchase: no"


def test_invalid_response_consumes_a_step_without_losing_public_state_or_memory() -> None:
    client = _RecordingClient(
        [
            "Memory:\nReady to purchase: no\n\nThought:\nneed a valid action",
            _response("Ready to purchase: no", "search[item]"),
        ]
    )
    environment = _TwoStepEnvironment()
    attempt = asyncio.run(run_native_interactive_task(client, _task(), environment))
    second_prompt = client.requests[1].messages[-1]["content"]
    assert "start state" in second_prompt
    assert "prior response did not contain one valid native action" in second_prompt
    assert "Ready to purchase: no" in second_prompt
    assert environment.actions == ["search[item]"]
    assert attempt.invalid_actions == 1


def test_unbounded_history_keeps_steps_older_than_eight_when_context_fits() -> None:
    history = tuple(
        PublicInteractiveTurn(
            _response(f"memory {index}", f"click[{index}]"),
            f"click[{index}]",
            f"state {index}",
            f"returned {index}",
            (f"click[{index + 1}]",),
            (f"click[{index}]",),
            f"memory {index - 1}",
            f"memory {index}",
            f"thought {index}",
            "updated",
        )
        for index in range(10)
    )
    text = render_interactive_messages(
        "webshop-native-react-memory-v5@1",
        task="buy item",
        current_observation="latest",
        available_actions=("click[done]",),
        history=history,
        history_window_steps=None,
        history_maximum_characters=None,
        structured_memory="memory 9",
    )[-1]["content"]
    assert "State before action: state 0" in text
    assert text.index("state 0") < text.index("state 9") < text.index("latest")


@dataclass(frozen=True)
class _CharacterCounter:
    def count(self, messages: tuple[dict[str, str], ...], *, enable_thinking: bool) -> int:
        assert enable_thinking is False
        return sum(len(message["content"]) for message in messages)


def test_context_overflow_keeps_only_a_contiguous_newest_raw_suffix() -> None:
    history = ("old", "middle", "new")

    def render(items: tuple[str, ...]) -> tuple[dict[str, str], ...]:
        return ({"role": "user", "content": "memory|" + "|".join(items)},)

    selected = newest_contiguous_history(
        render=render,
        history=history,
        counter=_CharacterCounter(),
        budget=ContextBudget(context_length=22, output_reserve=3, enable_thinking=False),
    )
    assert selected == ("middle", "new")


def test_structured_parser_keeps_action_valid_when_memory_is_absent() -> None:
    parsed = parse_visible_react("Thought:\nmove forward\n\nAction:\nopen fridge 1")
    assert parsed.memory is None
    assert parsed.thought == "move forward"
    assert parsed.action.value == "open fridge 1"


def test_alfworld_demonstrations_are_selected_by_task_type() -> None:
    step = InteractiveReplayStep(
        "before",
        ("look", "go to unused location 1"),
        "look",
        "after",
        ("go to table 1",),
    )
    examples = tuple(
        InteractivePromptExample(
            f"train/{task_type}",
            f"source/{task_type}",
            "train",
            f"task {task_type}",
            ("legacy",),
            task_type,
            (step,),
        )
        for task_type in ("pick_heat_then_place", "pick_clean_then_place")
    )
    asset = InteractivePromptAsset(
        "alfworld",
        "asset-v5",
        "official-train-replay",
        "train",
        "revision",
        "visible-react",
        examples,
    )
    rendered = asset.render_demonstrations(
        task="heat task",
        task_type="pick_heat_then_place",
        structured_memory=True,
    )
    assert len(rendered) == 1
    assert "task pick_heat_then_place" in rendered[0]
    assert "task pick_clean_then_place" not in rendered[0]
    assert "Memory:" in rendered[0]
    assert "Thought:" in rendered[0]
    assert "Initial public state:\nbefore" in rendered[0]
    assert "Environment feedback:\nafter" in rendered[0]
    assert "replay-verified against the native public action surface" in rendered[0]
    assert "go to unused location 1" not in rendered[0]
    assert "go to table 1" not in rendered[0]


def test_webshop_demonstration_retains_the_native_click_surface() -> None:
    step = InteractiveReplayStep(
        "search results",
        ("click[item]", "click[Back to Search]"),
        "click[item]",
        "item page",
        ("click[Buy Now]",),
    )
    asset = InteractivePromptAsset(
        "webshop",
        "asset-v5",
        "official-train-replay",
        "train",
        "revision",
        "visible-react",
        (
            InteractivePromptExample(
                "train/item",
                "goal-2000",
                "train",
                "buy an item",
                ("legacy",),
                replay_steps=(step,),
            ),
        ),
    )
    rendered = asset.render_demonstrations(task="buy an item", structured_memory=True)[0]
    assert "State before action 1:\nsearch results" in rendered
    assert "Admissible actions before action 1:" in rendered
    assert "click[Back to Search]" in rendered
    assert "Returned admissible actions:\n- click[Buy Now]" in rendered


def test_alfworld_structured_prompt_uses_only_the_public_task_type_decomposition() -> None:
    text = render_interactive_messages(
        "alfworld-native-react-memory-v5@1",
        task="heat an object and place it",
        current_observation="You are in a room.",
        available_actions=("go to counter 1",),
        task_type="pick_heat_then_place",
    )[-1]["content"]
    assert "Task decomposition:" in text
    assert "heating appliance" in text
    assert "second object" not in text
    assert (
        "one short line per field"
        in render_interactive_messages(
            "alfworld-native-react-memory-v5@1",
            task="heat an object and place it",
            current_observation="You are in a room.",
            available_actions=("go to counter 1",),
            task_type="pick_heat_then_place",
        )[0]["content"]
    )


def test_alfworld_structured_prompt_rejects_missing_task_type() -> None:
    with pytest.raises(ValueError, match="known task type"):
        render_interactive_messages(
            "alfworld-native-react-memory-v5@1",
            task="complete the task",
            current_observation="You are in a room.",
            available_actions=("look",),
        )
