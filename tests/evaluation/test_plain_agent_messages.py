"""Free-text agent bodies preserve math/code, and failed controls are not finals."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from skillev.evaluation.agent_communication import control_payload, peer_request
from skillev.evaluation.direct_baseline import DirectGenerationRequest
from skillev.evaluation.direct_baseline.interactive_tasks import (
    NativeEnvironmentStep,
    NativePublicState,
)
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.integrity_controller import IntegrityStepZeroClient
from skillev.evaluation.native_tool_instructions import native_tool_instructions
from skillev.evaluation.step0_completion import StepZeroTerminalMode, project_terminal_candidate
from skillev.evaluation.step0_integrity import InferenceArm
from skillev.evaluation.step0_public_state import ExecutionStatus, PublicEpisodeState
from skillev.evaluation.step0_types import (
    StepZeroActionMode,
    StepZeroArchitectureConfig,
    StepZeroCompletionMode,
    StepZeroTaskBinding,
)
from skillev.rollout import GenerationPhase
from tests.evaluation.test_step0_architecture import _Generator, _profile
from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer, clean_client
from tests.evaluation.test_step0_model_control import _interactive_task


@pytest.mark.parametrize("peer", ["solver", "researcher"])
def test_plain_message_body_is_verbatim(peer: str) -> None:
    body = '\\frac{a}{b} and \\text{"quoted"}\n    return "中文"\n'
    assert peer_request(f"Message to {peer}:\n{body}") == (peer, body)
    assert control_payload(f"Review:\n{body}")["body"] == body


@pytest.mark.parametrize(
    "text",
    [
        r'{"kind":"message","recipient":"solver","body":"Compute \frac{a}{b} with \alpha"}',
        '{"kind":"message","recipient":"solver","body":"First line\nsecond line"}',
        r'{"kind":"review","body":"Compute \alpha"}',
        "Message to solver:\n",
        "Message to unavailable:\nPlease help.",
    ],
)
def test_explicit_invalid_controls_do_not_silently_become_final_answers(text: str) -> None:
    with pytest.raises(ValueError):
        peer_request(text)


def test_plain_prose_mention_is_not_a_peer_call() -> None:
    assert peer_request("An example would be:\nMessage to solver:\nPlease help.") is None
    assert peer_request("The answer is 42.") is None
    assert peer_request("Message to solver: Please help.") == ("solver", "Please help.")
    assert peer_request('{"kind":"message","recipient":"solver","body":"Help."}') == (
        "solver",
        "Help.",
    )


def test_invalid_message_feedback_stays_inside_original_call_budget() -> None:
    bad = r'{"kind":"message","recipient":"solver","body":"Compute \alpha"}'
    generator = _Generator([(GenerationPhase.ACTION, bad)] * 3, tokenizer=CleanTokenizer())
    client = clean_client(generator, InferenceArm("multi"))
    result = asyncio.run(
        client.generate(
            DirectGenerationRequest("case", ({"role": "user", "content": "Compute."},), _profile())
        )
    )
    assert result.text == ""
    assert client.counts.model_calls == 3
    assert client.counts.peer_model_calls == 0
    assert client.counts.communication_repairs == 2  # only admitted repair model calls


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The calculation is verified.\n\n42", r"\boxed{42}"),
        ("First compute 84 divided by 2.\n042\n", r"\boxed{42}"),
        ("There are 42 items in this intermediate step.", None),
        ("42\nThis is only an intermediate quantity.", None),
        ("Final answer: 7\n42", None),
        ("\\boxed{7}\n42", None),
    ],
)
def test_standalone_final_integer_is_not_confused_with_intermediate_values(text, expected) -> None:
    assert project_terminal_candidate(StepZeroTerminalMode.AIME_INTEGER, text) == expected


def test_thinking_off_visible_discussion_survives_the_next_environment_step() -> None:
    tokenizer = CleanTokenizer()
    generator = _Generator(
        [
            (GenerationPhase.ACTION, "Review:\nRemember the blue label for later."),
            (GenerationPhase.ACTION, "Action: look"),
            (GenerationPhase.ACTION, "Action: inventory"),
        ],
        tokenizer=tokenizer,
    )
    client = IntegrityStepZeroClient(
        generator=generator,
        library_state=None,
        bindings=(
            StepZeroTaskBinding(
                task_id="case-1",
                benchmark_id="alfworld",
                task_family="public",
                panel_index=0,
                action_mode=StepZeroActionMode.ALF_WORLD,
            ),
        ),
        config=StepZeroArchitectureConfig(
            3,
            256,
            256,
            10000,
            0,
            "Qwen3.5-9B",
            "synthetic",
            arm=InferenceArm("multi"),
        ),
    )

    async def episode() -> None:
        task = _interactive_task("alfworld", "Synthetic public task")
        await client.begin_interactive_episode(
            task,
            NativePublicState("Initial room", ("look", "inventory")),
        )
        first = await client.generate(
            DirectGenerationRequest(
                "case-1:step:1", ({"role": "user", "content": "Synthetic task"},), _profile()
            )
        )
        assert first.text == "Action: look"
        await client.observe_interactive_episode(
            "case-1",
            "look",
            NativeEnvironmentStep(
                "The attempted inspection failed.",
                False,
                0.0,
                False,
                False,
                available_actions=("look", "inventory"),
            ),
        )
        second = await client.generate(
            DirectGenerationRequest(
                "case-1:step:2", ({"role": "user", "content": "Synthetic task"},), _profile()
            )
        )
        assert second.text == "Action: inventory"
        assert f"Remaining environment actions: {task.max_steps - 1}." in repr(
            tokenizer.messages[-1][0]
        )

    asyncio.run(episode())
    next_input = repr(tokenizer.messages[-1][0])
    assert "blue label" in next_input
    assert "attempted inspection failed" in next_input
    assert client._interactive["case-1"].memory.history[-1].execution_status is (
        ExecutionStatus.REJECTED
    )
    assert client.counts.model_calls == 3
    assert client.counts.peer_model_calls == 0
    assert all(not thinking for _, thinking in tokenizer.messages)
    assert "action: 2." in repr(tokenizer.messages[1][0])


def test_long_visible_notes_do_not_displace_the_task_or_current_observation() -> None:
    state = PublicEpisodeState("Complete task")
    state = state.observe(None, "Initial observation").remember("owner", "old " * 2000)
    state = state.observe("look", "Latest observation").remember("solver", "Recent advice")
    rendered = state.render(maximum_tokens=900, count_tokens=len)
    assert "Complete task" in rendered
    assert "Latest observation" in rendered
    assert "Recent advice" in rendered
    assert "old old old" not in rendered
    assert len(state.discussions) == 2  # The archive is not overwritten by rendering.
    assert len(rendered) <= 900


def test_observations_follow_the_decisions_that_produced_them() -> None:
    state = PublicEpisodeState("Complete task", history_window_steps=None)
    state = state.observe(None, "Initially at the desk.")
    state = state.remember("solver", "Consider opening the cabinet.")
    state = state.remember("owner", "Action: go to cabinet 1")
    state = state.observe("go to cabinet 1", "You are now at cabinet 1.")
    state = state.remember("owner", "Action: examine desk 1")
    state = state.observe(
        "examine desk 1",
        "That action failed; still at cabinet 1.",
        execution_status=ExecutionStatus.REJECTED,
    )
    rendered = state.render()
    assert rendered.index("Consider opening") < rendered.index("You are now")
    assert rendered.index("Action: examine") < rendered.index("That action failed")
    assert rendered.rstrip().endswith("That action failed; still at cabinet 1.")
    assert "Initially at the desk." in rendered
    assert len(state.discussions) == 3


def test_shopping_owner_receives_the_simulated_purchase_interface() -> None:
    tokenizer = CleanTokenizer()
    generator = _Generator(
        [
            (GenerationPhase.ACTION, "Action: click[buy now]"),
        ],
        tokenizer=tokenizer,
    )
    client = IntegrityStepZeroClient(
        generator=generator,
        library_state=None,
        bindings=(
            StepZeroTaskBinding(
                task_id="case-1",
                benchmark_id="webshop",
                task_family="public",
                panel_index=0,
                action_mode=StepZeroActionMode.WEB_SHOP,
            ),
        ),
        config=StepZeroArchitectureConfig(
            3,
            256,
            256,
            10000,
            0,
            "Qwen3.5-9B",
            "synthetic",
            arm=InferenceArm("multi"),
        ),
    )

    async def episode() -> None:
        await client.begin_interactive_episode(
            _interactive_task("webshop", "Buy a fictional blue mug."),
            NativePublicState("Blue mug on the current page.", ("click[buy now]",)),
        )
        result = await client.generate(
            DirectGenerationRequest(
                "case-1:step:1", ({"role": "user", "content": "Buy a fictional mug."},), _profile()
            )
        )
        assert result.text == "Action: click[buy now]"

    asyncio.run(episode())
    for messages, _ in tokenizer.messages[:2]:
        context = "\n".join(item["content"] for item in messages)
        assert "simulated shopping" in context
        assert "no real purchases" in context
        assert "ends the episode" in context
        assert "Blue mug on the current page." in context
        system = next(item["content"] for item in messages if item["role"] == "system")
        assert native_tool_instructions("webshop") in system
        assert "Blue mug on the current page." not in system
        assert any(
            item["role"] == "tool" and "Blue mug on the current page." in item["content"]
            for item in messages
        )
    assert client.counts.skill_body_tokens == 0


def test_alfworld_owner_receives_public_api_without_a_task_recipe() -> None:
    tokenizer = CleanTokenizer()
    responses = [(GenerationPhase.ACTION, "Action: inventory")]
    client = IntegrityStepZeroClient(
        generator=_Generator(responses, tokenizer=tokenizer),
        library_state=None,
        bindings=(
            StepZeroTaskBinding(
                task_id="case-1",
                benchmark_id="alfworld",
                task_family="public",
                panel_index=0,
                action_mode=StepZeroActionMode.ALF_WORLD,
            ),
        ),
        config=StepZeroArchitectureConfig(
            3,
            256,
            256,
            10000,
            0,
            "Qwen3.5-9B",
            "synthetic",
            arm=InferenceArm(
                "synthetic-api",
            ),
        ),
    )

    async def episode() -> None:
        await client.begin_interactive_episode(
            _interactive_task("alfworld", "A fictional public task with the label Indigo."),
            NativePublicState("Only the fictional marker Veridian is visible.", ("inventory",)),
        )
        result = await client.generate(
            DirectGenerationRequest(
                "case-1:step:1",
                ({"role": "user", "content": "A fictional public task with the label Indigo."},),
                _profile(),
            )
        )
        assert result.text == "Action: inventory"

    asyncio.run(episode())
    for messages, _ in tokenizer.messages:
        system = next(item["content"] for item in messages if item["role"] == "system")
        assert native_tool_instructions("alfworld") in system
        assert "Indigo" not in system
        assert "Veridian" not in system
        assert any(item["role"] == "user" and "Indigo" in item["content"] for item in messages)
        assert any(item["role"] == "tool" and "Veridian" in item["content"] for item in messages)
    assert client.counts.skill_body_tokens == 0
    assert client.counts.extra_model_calls_for_serialization == 0


def test_short_answer_projection_preserves_the_owners_explicit_answer_not_a_reference() -> None:
    for text in ("blue", "I considered red, but selected blue.\nFinal answer: blue"):
        assert project_terminal_candidate(StepZeroTerminalMode.SHORT_ANSWER, text) == "blue"


@pytest.mark.parametrize(("turn_limit", "episode_limit"), [(4, None), (8, 4)])
def test_undeliverable_peer_request_does_not_discard_a_remaining_owner_call(
    turn_limit, episode_limit
) -> None:
    generator = _Generator(
        [
            (GenerationPhase.ACTION, "Review:\nCheck the arithmetic."),
            (GenerationPhase.ACTION, "Review:\nThe calculation is ready."),
            (GenerationPhase.ACTION, "Message to solver:\nPlease verify."),
            (GenerationPhase.ACTION, "42"),
        ],
        tokenizer=CleanTokenizer(),
    )
    client = clean_client(generator, InferenceArm("multi"))
    client._config = replace(client._config, max_turns=turn_limit)
    client.generation.call_limit = episode_limit
    result = asyncio.run(
        client.generate(
            DirectGenerationRequest("case", ({"role": "user", "content": "Compute."},), _profile())
        )
    )
    assert result.text == r"\boxed{42}"
    assert client.counts.model_calls == 4
    assert client.counts.peer_model_calls == 0
    assert client.counts.communication_repairs == 1


@pytest.mark.parametrize("prefix", ["", "```python\n", "```\n"])
@pytest.mark.parametrize("suffix", ["", "\n```"])
def test_single_code_payload_accepts_one_sided_boundary_fences(prefix, suffix) -> None:
    source = "def synthetic_double(x):\n    return x * 2\n"
    assert project_terminal_candidate(
        StepZeroTerminalMode.PYTHON_SOURCE, prefix + source + suffix
    ) == (source.rstrip("\n"))


def test_code_projection_uses_final_block_without_rewriting_unframed_prose() -> None:
    assert (
        project_terminal_candidate(
            StepZeroTerminalMode.PYTHON_SOURCE, "```python\nx = 1\n```\n```python\nx = 2\n```"
        )
        == "x = 2"
    )
    source = "x = 1\n\nReview: This is prose, not Python."
    assert project_terminal_candidate(StepZeroTerminalMode.PYTHON_SOURCE, source) == source


@pytest.mark.parametrize("structured", [False, True])
def test_hotpot_interface_keeps_all_passages_and_the_question(structured) -> None:
    bodies = [f'Full paragraph {i}, with "quotes", \\math and\na second line.' for i in range(10)]
    passages = [[f"Title {i}", [body]] for i, body in enumerate(bodies)] if structured else bodies
    question = "Which fictional object is described?"
    entry = PublicTaskView.from_record(
        "synthetic-context", "hotpotqa", {"question": question, "context": passages}
    )
    rendered = entry.render()
    assert rendered.endswith(question)
    for body in bodies:
        assert body in rendered
    if structured:
        for i in range(10):
            assert f"Title {i}" in rendered


@pytest.mark.parametrize(
    ("mode", "ambiguous", "submitted", "expected"),
    [
        (
            StepZeroCompletionMode.PYTHON_SOURCE,
            "```python\nx = 1\n```\n```python\nx = 2",
            "```python\nx = 3\n```",
            "x = 3",
        ),
        (
            StepZeroCompletionMode.AIME_BOXED_INTEGER,
            "Final answer: 41\nFinal answer: 42",
            "43",
            r"\boxed{43}",
        ),
        (
            StepZeroCompletionMode.SHORT_ANSWER,
            "Final answer: blue\nFinal answer: red",
            "green",
            "green",
        ),
    ],
)
def test_unsubmitted_terminal_payload_is_returned_to_its_owner(
    mode, ambiguous, submitted, expected
) -> None:
    tokenizer = CleanTokenizer()
    generator = _Generator(
        [(GenerationPhase.ACTION, ambiguous), (GenerationPhase.ACTION, submitted)],
        tokenizer=tokenizer,
    )
    client = clean_client(generator, InferenceArm("multi"))
    client._bindings["case"] = replace(client._bindings["case"], completion_mode=mode)
    request = DirectGenerationRequest(
        "case", ({"role": "user", "content": "Complete the synthetic task."},), _profile()
    )
    result = asyncio.run(client.generate(request))
    # The owner may submit something different from every earlier draft. The
    # interface never picks an earlier candidate or reads a reference answer.
    assert result.text == expected
    assert any(item["content"] == ambiguous for item in tokenizer.messages[1][0])
    assert "not submitted" in repr(tokenizer.messages[1][0])
    assert any(
        item["role"] == "tool" and "not submitted" in item["content"]
        for item in tokenizer.messages[1][0]
    )
    assert client.counts.model_calls == 2
    assert client.counts.communication_repairs == 1
    assert client.counts.peer_model_calls == 0
    assert client.counts.extra_model_calls_for_serialization == 0
    assert asyncio.run(client.generate(request)) == result
    assert client.counts.model_calls == 2


def test_framework_controls_do_not_impersonate_the_user() -> None:
    tokenizer = CleanTokenizer()
    answer = "Your fictional parcel is ready for collection."
    responses = [(GenerationPhase.ACTION, answer)]
    generator = _Generator(responses, tokenizer=tokenizer)
    arm = InferenceArm("single")
    client = clean_client(generator, arm, benchmark="healthbench")
    client._bindings["case"] = replace(
        client._bindings["case"], completion_mode=StepZeroCompletionMode.NATURAL_LANGUAGE
    )
    client.generation.call_limit = 4
    client.generation.output_token_limit = 4096
    original = (
        {"role": "system", "content": "Help the person with their fictional parcel."},
        {"role": "user", "content": "The courier says my parcel is ready."},
        {"role": "assistant", "content": "Do you want to collect it?"},
        {"role": "user", "content": "Yes. What is its current status?"},
    )
    result = asyncio.run(client.generate(DirectGenerationRequest("case", original, _profile())))
    assert result.text == answer
    for index in (0,):
        messages, thinking = tokenizer.messages[index]
        # Original conversational content and ordering survive unchanged. No
        # framework notice or third-party opinion becomes another user turn.
        assert [item for item in messages if item["role"] == "user"] == [
            item for item in original if item["role"] == "user"
        ]
        positions = [messages.index(item) for item in original if item["role"] != "system"]
        assert positions == sorted(positions)
        assert not thinking
        system = [item for item in messages if item["role"] == "system"]
        assert len(system) == 1
        assert messages[0] is system[0]
        assert system[0]["content"].startswith(original[0]["content"])
        for counter in ("Calls remaining", "Remaining system-wide"):
            # Runtime budgets belong to the opening system instructions, not
            # fabricated tool responses or additional user turns.
            assert system[0]["content"].count(counter) == 1
            assert all(counter not in item["content"] for item in messages[1:])
        assert "Available peers:" not in system[0]["content"]
    assert client.counts.peer_model_calls == 0
    assert client.counts.model_calls == 1
    assert client.counts.extra_model_calls_for_serialization == 0


@pytest.mark.parametrize(("turn_limit", "episode_limit"), [(2, None), (8, 2)])
def test_terminal_submission_feedback_cannot_extend_the_original_call_budget(
    turn_limit, episode_limit
) -> None:
    ambiguous = "Final answer: 41\nFinal answer: 42"
    generator = _Generator([(GenerationPhase.ACTION, ambiguous)] * 2, tokenizer=CleanTokenizer())
    client = clean_client(generator, InferenceArm("multi"))
    client._config = replace(client._config, max_turns=turn_limit)
    client.generation.call_limit = episode_limit
    result = asyncio.run(
        client.generate(
            DirectGenerationRequest("case", ({"role": "user", "content": "Compute."},), _profile())
        )
    )
    assert result.text == ""
    assert client.counts.model_calls == 2
    assert client.counts.communication_repairs == 1
