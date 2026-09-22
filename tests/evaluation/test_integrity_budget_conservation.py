"""Owner decisions, review and failed transport spend the same frozen budget."""

import asyncio
from dataclasses import replace

import pytest

from skillev.evaluation.direct_baseline import DirectGenerationRequest
from skillev.evaluation.direct_baseline.interactive_tasks import (
    NativeEnvironmentStep,
    NativePublicState,
)
from skillev.evaluation.integrity_controller import IntegrityStepZeroClient
from skillev.evaluation.integrity_generation import EvaluationBudgetExhausted
from skillev.evaluation.public_context import PromptBlock, PublicPrompt
from skillev.evaluation.step0_integrity import InferenceArm, ReasoningPass, SkillMode
from skillev.evaluation.step0_types import (
    StepZeroActionMode,
    StepZeroArchitectureConfig,
    StepZeroTaskBinding,
)
from skillev.experiments._evolution_preflight_seed import _seed_document
from skillev.rollout import GenerationPhase
from skillev.runtime import SkillLibraryState, SkillRequirement
from tests.evaluation.test_step0_architecture import _Generator, _profile
from tests.evaluation.test_step0_integrity_runtime import CleanTokenizer, clean_client
from tests.evaluation.test_step0_model_control import _interactive_task


def request():
    return DirectGenerationRequest(
        "case", ({"role": "user", "content": "Calculate a synthetic integer."},), _profile()
    )


def test_optional_review_consumes_the_local_turn_budget_too():
    generator = _Generator(
        [(GenerationPhase.REASONING, "An optional check.")], tokenizer=CleanTokenizer()
    )
    client = clean_client(
        generator, InferenceArm("optional", reasoning_pass=ReasoningPass.OPTIONAL)
    )
    client._config = replace(client._config, max_turns=1)
    client.request_reasoning("case")
    result = asyncio.run(client.generate(request()))
    assert result.text == ""
    assert client.counts.model_calls == 1
    assert len(generator.profiles) == 1


def test_rejected_consultation_and_owner_final_share_one_call_ledger():
    generator = _Generator(
        [
            (GenerationPhase.ACTION, "Message to solver:\nPlease calculate this."),
            (GenerationPhase.ACTION, "Final answer: 42"),
        ],
        tokenizer=CleanTokenizer(),
    )
    client = clean_client(generator, InferenceArm("A2"))
    client.generation.call_limit = 2
    result = asyncio.run(client.generate(request()))
    assert result.text == r"\boxed{42}"
    assert client.counts.model_calls == 2
    assert client.counts.peer_model_calls == 0
    assert [call.participant for call in client.generation.calls] == ["owner", "owner"]
    assert all(call.transport_status == "completed" for call in client.generation.calls)


def test_live_budgets_are_system_metadata_not_unrequested_tool_results():
    tokenizer = CleanTokenizer()
    malformed = "Message to solver:"
    generator = _Generator(
        [(GenerationPhase.ACTION, malformed), (GenerationPhase.ACTION, "Final answer: 42")],
        tokenizer=tokenizer,
    )
    client = clean_client(generator, InferenceArm("A2"))
    client.generation.call_limit = 2
    client.generation.output_token_limit = 4096
    result = asyncio.run(client.generate(request()))
    assert result.text == r"\boxed{42}"
    first, second = (item[0] for item in tokenizer.messages)
    assert first[0]["role"] == second[0]["role"] == "system"
    assert first[-1] == request().messages[-1]
    assert not any(message["role"] == "tool" for message in first)
    assert "2 model calls and 4096 output tokens" in first[0]["content"]
    assert f"1 model calls and {4096 - len(malformed)} output tokens" in second[0]["content"]
    assert "Calls remaining" in first[0]["content"]
    assert "Calls remaining" in second[0]["content"]
    assert client.generation.call_limit == client.counts.model_calls == 2


def test_runtime_budget_refresh_preserves_the_complete_history_prefix():
    prefix = (
        {"role": "system", "content": "Use the supplied public task and native tools."},
        {"role": "user", "content": "Synthetic public task."},
        {"role": "tool", "content": "Long public history. " * 1000},
    )
    prompt = PublicPrompt.required(prefix).append(
        PromptBlock("tool", "Current runtime action surface.", runtime_metadata=True)
    )
    first = prompt.with_runtime_metadata("Remaining model calls: 8.")
    second = prompt.with_runtime_metadata("Remaining model calls: 7.")
    before = tuple(block.message() for block in first.blocks)
    after = tuple(block.message() for block in second.blocks)
    assert before[:-1] == after[:-1] == prefix
    assert len(before) == len(after) == len(prompt.blocks)
    assert before[-1]["role"] == after[-1]["role"] == "tool"
    assert before[-1]["content"] != after[-1]["content"]
    assert prompt.blocks[-1].content == "Current runtime action surface."


def test_interactive_budgets_refresh_in_the_existing_surface_and_remain_enforced():
    tokenizer = CleanTokenizer()
    responses = [
        (GenerationPhase.ACTION, "Action: look"),
        (GenerationPhase.ACTION, "Action: inventory"),
    ]
    expected_output_tokens = sum(len(text) for _, text in responses)
    generator = _Generator(responses, tokenizer=tokenizer)
    client = clean_client(generator, InferenceArm("single"), benchmark="alfworld")
    client._bindings["case"] = replace(
        client._bindings["case"],
        action_mode=StepZeroActionMode.ALF_WORLD,
        # This fixture counts characters, not native tokenizer subwords.
        maximum_h0_tokens=16384,
    )
    client.generation.call_limit = 2
    client.generation.output_token_limit = 4096

    async def episode():
        task = replace(_interactive_task("alfworld", "Synthetic public task."), task_id="case")
        await client.begin_interactive_episode(
            task, NativePublicState("Synthetic initial room.", ("look", "inventory"))
        )
        first = await client.generate(
            DirectGenerationRequest(
                "case:step:1", ({"role": "user", "content": task.task},), _profile()
            )
        )
        assert first.text == "Action: look"
        await client.observe_interactive_episode(
            "case",
            "look",
            NativeEnvironmentStep(
                "Synthetic room after inspection.",
                False,
                0.0,
                False,
                True,
                available_actions=("look", "inventory"),
            ),
        )
        second = await client.generate(
            DirectGenerationRequest(
                "case:step:2", ({"role": "user", "content": task.task},), _profile()
            )
        )
        assert second.text == "Action: inventory"
        exhausted = await client.generate(
            DirectGenerationRequest(
                "case:step:3", ({"role": "user", "content": task.task},), _profile()
            )
        )
        assert not exhausted.text

    asyncio.run(episode())
    first, second = (item[0] for item in tokenizer.messages)
    assert first[0] == second[0]
    assert [item for item in first if item["role"] == "user"] == [
        item for item in second if item["role"] == "user"
    ]
    for messages, remaining in ((first, 2), (second, 1)):
        assert sum(item["role"] == "system" for item in messages) == 1
        surface = [item for item in messages if "Available actions" in item["content"]]
        assert len(surface) == 1
        assert surface[0]["role"] == "tool"
        assert "Remaining environment actions" in surface[0]["content"]
        assert f"{remaining} model calls" in surface[0]["content"]
        assert "Calls remaining" in surface[0]["content"]
        assert "Remaining system-wide" not in messages[0]["content"]
    assert client.counts.model_calls == client.generation.call_limit == 2
    assert client.generation.output_tokens == expected_output_tokens
    assert len(generator.profiles) == 2
    assert client.counts.peer_model_calls == 0


def test_local_action_allowance_renews_without_resetting_episode_budgets():
    tokenizer = CleanTokenizer()
    responses = [
        (GenerationPhase.ACTION, "I have not selected a command yet."),
        (GenerationPhase.ACTION, "Action: look"),
        (GenerationPhase.ACTION, "Action: inventory"),
    ]
    client = clean_client(_Generator(list(responses), tokenizer=tokenizer), InferenceArm("single"))
    client._config = replace(client._config, max_turns=2)
    client._bindings["case"] = replace(
        client._bindings["case"],
        benchmark_id="alfworld",
        action_mode=StepZeroActionMode.ALF_WORLD,
        maximum_h0_tokens=16384,
    )
    client.generation.call_limit = 4
    client.generation.output_token_limit = 4096

    async def episode():
        task = replace(
            _interactive_task("alfworld", "Synthetic public task."), task_id="case", max_steps=200
        )
        await client.begin_interactive_episode(task, NativePublicState("Room.", ("look",)))
        first = await client.generate(
            DirectGenerationRequest(
                "case:step:1", ({"role": "user", "content": task.task},), _profile()
            )
        )
        assert first.text == "Action: look"
        await client.observe_interactive_episode(
            "case",
            "look",
            NativeEnvironmentStep(
                "Room inspected.", False, 0.0, False, True, available_actions=("inventory",)
            ),
        )
        second = await client.generate(
            DirectGenerationRequest(
                "case:step:2", ({"role": "user", "content": task.task},), _profile()
            )
        )
        assert second.text == "Action: inventory"

    asyncio.run(episode())
    spent = 0
    for (messages, _), local, global_calls, actions, (_, response) in zip(
        tokenizer.messages, (2, 1, 2), (4, 3, 2), (200, 200, 199), responses, strict=True
    ):
        surface = next(
            row["content"] for row in messages if "Remaining environment actions" in row["content"]
        )
        assert f"one environment action: {local}" in surface
        assert "NOT the remaining environment actions" in surface
        assert "budgets do not reset" in surface
        assert f"{global_calls} model calls and {4096 - spent} output tokens" in surface
        assert f"Remaining environment actions: {actions}" in surface
        spent += len(response)
    assert client.generation.output_tokens == spent
    assert client.counts.model_calls == 3
    assert client.counts.peer_model_calls == 0


@pytest.mark.parametrize("limit", [256, 12000])
def test_response_limit_is_shown_and_one_full_call_is_not_chunked(limit):
    tokenizer = CleanTokenizer()
    generator = _Generator([(GenerationPhase.ACTION, "Final answer: 42")], tokenizer=tokenizer)
    client = clean_client(generator, InferenceArm("full-call"))
    client._config = replace(client._config, max_action_tokens=limit)
    client.generation.call_limit = 1
    client.generation.output_token_limit = limit
    asyncio.run(client.generate(request()))
    profile = generator.profiles[0]
    assert profile.max_new_tokens == limit
    messages = tokenizer.messages[0][0]
    assert any(
        f"at most {profile.max_new_tokens} output tokens" in row["content"] for row in messages
    )
    assert len(generator.profiles) == client.counts.model_calls == 1
    assert client.generation.native_chunk_tokens is None


@pytest.mark.parametrize("error_type", [TimeoutError, asyncio.CancelledError])
def test_unknown_transport_usage_is_not_zero_and_call_is_not_refunded(error_type):
    class FailingGenerator(_Generator):
        async def generate_evaluation(self, *args, **kwargs):
            raise error_type("synthetic transport completion unknown")

    client = clean_client(FailingGenerator([], tokenizer=CleanTokenizer()), InferenceArm("A2"))
    client.generation.call_limit = 1
    with pytest.raises(error_type):
        asyncio.run(client.generate(request()))
    assert client.counts.model_calls == 1
    call = client.generation.calls[0]
    assert call.transport_status == "unknown"
    assert call.input_tokens is None
    assert call.output_tokens is None


def test_repair_not_admitted_by_output_budget_is_not_counted_as_a_model_call():
    malformed = "Message to solver:"
    generator = _Generator([(GenerationPhase.ACTION, malformed)], tokenizer=CleanTokenizer())
    client = clean_client(generator, InferenceArm("A2"))
    client.generation.output_token_limit = len(malformed)
    with pytest.raises(EvaluationBudgetExhausted):
        asyncio.run(client.generate(request()))
    assert client.counts.model_calls == 1
    assert client.counts.communication_repairs == 0


@pytest.mark.parametrize("budget", [0, 64, 256, 1024])
def test_advice_respects_the_actual_binding_budget_and_never_truncates_a_body(budget):
    generator = _Generator([(GenerationPhase.ACTION, "42")], tokenizer=CleanTokenizer())
    original = clean_client(generator, InferenceArm("advice", skill_mode=SkillMode.GENERIC_TEXT))
    binding: StepZeroTaskBinding = replace(
        original._bindings["case"], skill_instruction_token_budget=budget
    )
    config: StepZeroArchitectureConfig = original._config
    body = "Consider the public evidence without guessing facts. " * 2
    library = SkillLibraryState.from_seed_documents(
        (
            _seed_document(
                skill_id="synthetic",
                title="Synthetic advice",
                summary="calculate",
                instructions=body,
                requirements=(SkillRequirement("public", "Use public information."),),
            ),
        )
    )
    client = IntegrityStepZeroClient(
        generator=generator,
        library_state=library,
        bindings=(binding,),
        config=config,
    )
    asyncio.run(client.generate(request()))
    assert client.counts.skill_body_tokens <= budget
    rendered = repr(generator.tokenizer.messages)
    assert (body in rendered) == (client.counts.skill_blocks_injected == 1)
    if budget == 0:
        assert "Optional advice" not in rendered
