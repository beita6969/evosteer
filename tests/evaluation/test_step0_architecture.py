from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from skillev.evaluation.direct_baseline import (
    DirectDecodingProfile,
    DirectGenerationRequest,
    WebShopPublicCatalogCandidate,
)
from skillev.evaluation.direct_baseline.config import DirectBenchmark
from skillev.evaluation.direct_baseline.interactive_tasks import (
    InvalidCandidatePolicy,
    InvalidEnvironmentActionPolicy,
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativeInteractiveTask,
    NativePublicState,
)
from skillev.evaluation.direct_baseline.parsing import parse_visible_react
from skillev.evaluation.legacy_step0_architecture import (
    _blocked_native_actions,
    _effective_native_actions,
    _project_visible_react_state,
    _reasoning_aligned_actions,
    _repeats_webshop_search,
    _root_query_from_messages,
    _webshop_catalog_navigation_action,
    _webshop_search_has_price_prose,
)
from skillev.evaluation.step0_architecture import (
    ArchitectureInferenceState,
    LegacyStepZeroArchitectureDirectClient,
    StepZeroActionMode,
    StepZeroArchitectureConfig,
    StepZeroArchitectureDirectClient,
    StepZeroCompletionMode,
    StepZeroReasoningMode,
    StepZeroTaskBinding,
)
from skillev.evaluation.step0_conditions import ReasoningAuthorityMode
from skillev.evaluation.step0_integrity import InferenceArm, legacy_arm
from skillev.evaluation.step0_interactive_agent import initial_episode_memory, update_episode_memory
from skillev.experiments._evolution_preflight_seed import (
    planned_seed_documents,
    planned_step_zero_seed_documents,
)
from skillev.policy import rollout_chat_messages
from skillev.rollout import (
    EvaluationGenerationConstraint,
    EvaluationGenerationProfile,
    GenerationPhase,
    PolicySnapshot,
    RolloutGenerationRequest,
    RolloutGenerationResult,
)
from skillev.runtime import BudgetVector, SkillLibraryState


@dataclass(frozen=True, slots=True)
class _Tokenizer:
    tokenizer_id: str = "tokenizer-fixture"
    thinking_prompts: list[str] = field(default_factory=list, compare=False)
    rollout_prompts: list[str] = field(default_factory=list, compare=False)

    def encode(self, text: str) -> list[int]:
        return [ord(value) for value in text]

    def encode_rollout_prompt(self, text: str) -> list[int]:
        self.rollout_prompts.append(text)
        rendered = rollout_chat_messages(text)
        return [ord(value) for value in repr(rendered)]

    def encode_rollout_prompt_with_thinking(self, text: str) -> list[int]:
        self.thinking_prompts.append(text)
        rendered = rollout_chat_messages(text)
        return [ord(value) for value in repr(("qwen-thinking", rendered))]

    def encode_step_zero_reasoning_prompt(
        self,
        text: str,
        *,
        enable_thinking: bool,
    ) -> list[int]:
        self.thinking_prompts.append(text)
        rendered = rollout_chat_messages(
            text,
            system_message="neutral-step-zero-reasoning-test-controller",
        )
        return [ord(value) for value in repr((enable_thinking, rendered))]

    def encode_step_zero_terminal_prompt(self, text: str) -> list[int]:
        rendered = rollout_chat_messages(
            text,
            system_message="terminal-only-test-controller",
        )
        return [ord(value) for value in repr(("terminal", rendered))]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(value) for value in token_ids)

    def encode_integrity_messages(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        enable_thinking: bool,
    ) -> list[int]:
        self.thinking_prompts.append("\n".join(item["content"] for item in messages))
        return self.encode(repr((messages, enable_thinking)))


@dataclass(slots=True)
class _Generator:
    scripted: list[tuple[GenerationPhase, str]]
    tokenizer: _Tokenizer = field(default_factory=_Tokenizer)
    active_episodes: set[str] = field(default_factory=set)
    profiles: list[EvaluationGenerationProfile] = field(default_factory=list)
    constraints: list[EvaluationGenerationConstraint | None] = field(default_factory=list)

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot.create(
            backbone_id="qwen35-base-fixture",
            forward_adapter_version="adapter-free",
            tokenizer_id=self.tokenizer.tokenizer_id,
            backend_id="sglang-native-evaluation-exact-token",
            initial_trainable_state_hash="initial-step-zero-fixture",
        )

    def begin_episode(self, episode_id: str, expected_policy_snapshot_id: str) -> None:
        assert expected_policy_snapshot_id == self.snapshot().snapshot_id
        assert episode_id not in self.active_episodes
        self.active_episodes.add(episode_id)

    def end_episode(self, episode_id: str) -> None:
        self.active_episodes.remove(episode_id)

    async def generate_evaluation(
        self,
        request: RolloutGenerationRequest,
        *,
        profile: EvaluationGenerationProfile,
        constraint: EvaluationGenerationConstraint | None = None,
    ) -> RolloutGenerationResult:
        self.profiles.append(profile)
        self.constraints.append(constraint)
        assert request.decoding_snapshot_id == profile.profile_id
        assert request.seed == profile.seed
        phase, text = self.scripted.pop(0)
        assert phase is request.phase
        token_ids = tuple(ord(value) for value in text)
        return RolloutGenerationResult(
            content_token_ids=token_ids,
            stop_token_ids=(),
            finish_reason="scripted",
            policy_snapshot_id=self.snapshot().snapshot_id,
            backend_id="sglang-native-evaluation-exact-token",
            usage=BudgetVector(
                input_tokens=len(request.input_ids),
                output_tokens=len(token_ids),
                model_calls=1,
            ),
        )


def _profile(*, max_tokens: int = 64, enable_thinking: bool = False) -> DirectDecodingProfile:
    return DirectDecodingProfile(
        profile_id="benchmark-matched-fixture@1",
        enable_thinking=enable_thinking,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_new_tokens=max_tokens,
        seed=42,
    )


def _client(
    generator: _Generator,
    *,
    mode: StepZeroActionMode,
    benchmark: str,
    completion_mode: StepZeroCompletionMode = StepZeroCompletionMode.PASSTHROUGH,
    reasoning_mode: StepZeroReasoningMode = StepZeroReasoningMode.STANDARD,
    max_reasoning_tokens: int | None = None,
    maximum_h0_tokens: int | None = None,
    root_query: str | None = "public root task",
    maximum_trivia_searches: int = 1,
    task_family: str = "fixture",
    skill_instruction_token_budget: int = 1024,
    with_skills: bool = True,
    arm: InferenceArm | None = None,
) -> StepZeroArchitectureDirectClient | LegacyStepZeroArchitectureDirectClient:
    selected_arm = arm or legacy_arm()
    client_type = (
        LegacyStepZeroArchitectureDirectClient
        if selected_arm.legacy
        else StepZeroArchitectureDirectClient
    )
    return client_type(
        generator=generator,
        library_state=SkillLibraryState.from_seed_documents(
            (
                planned_step_zero_seed_documents()
                if selected_arm.legacy
                else planned_seed_documents()
            )
            if with_skills
            else ()
        ),
        bindings=(
            StepZeroTaskBinding(
                task_id="case-1",
                benchmark_id=benchmark,
                task_family=task_family,
                panel_index=3,
                action_mode=mode,
                completion_mode=completion_mode,
                reasoning_mode=reasoning_mode,
                reasoning_authority_mode=(
                    ReasoningAuthorityMode.FROZEN_BENCHMARK_MATCHED
                    if benchmark == "aime-2026"
                    else ReasoningAuthorityMode.FROZEN_DETERMINISTIC
                ),
                root_query=root_query,
                maximum_trivia_searches=maximum_trivia_searches,
                max_reasoning_tokens=max_reasoning_tokens,
                maximum_h0_tokens=maximum_h0_tokens,
                skill_instruction_token_budget=skill_instruction_token_budget,
            ),
        ),
        config=StepZeroArchitectureConfig(
            max_turns=3,
            max_reasoning_tokens=32,
            max_action_tokens=64,
            maximum_h0_tokens=32_768,
            base_seed=20260721,
            response_model="qwen35-step-zero-base",
            service_instance_id="fixture-service",
            arm=selected_arm,
        ),
    )


def test_static_uses_frozen_reasoning_and_native_short_answer_terminal() -> None:
    generator = _Generator(
        [
            (GenerationPhase.REASONING, "Use the two public evidence hops."),
            (GenerationPhase.ACTION, "Ada Lovelace"),
        ]
    )
    client = _client(
        generator,
        mode=StepZeroActionMode.COMPLETION,
        benchmark="hotpotqa",
        completion_mode=StepZeroCompletionMode.SHORT_ANSWER,
    )

    result = asyncio.run(
        client.generate(
            DirectGenerationRequest(
                "case-1",
                (
                    {"role": "system", "content": "Answer from supplied context."},
                    {"role": "user", "content": "Public question and ten passages"},
                ),
                _profile(),
            )
        )
    )

    assert result.text == "Ada Lovelace"
    assert generator.profiles[0].sampling_mode == "greedy"
    assert generator.profiles[0].temperature == 0.0
    assert generator.profiles[1].sampling_mode == "greedy"
    assert generator.profiles[1].temperature == 0.0
    assert generator.constraints[1] is not None
    assert client.diagnostics.retrieved_skill_ids == ("skill-hotpot-evidence-chain",)
    assert not generator.active_episodes


def test_trained_inference_identity_changes_state_without_changing_task_binding() -> None:
    generator = _Generator([])
    client = _client(
        generator,
        mode=StepZeroActionMode.COMPLETION,
        benchmark="hotpotqa",
        completion_mode=StepZeroCompletionMode.SHORT_ANSWER,
    )
    client._config = StepZeroArchitectureConfig(
        max_turns=3,
        max_reasoning_tokens=32,
        max_action_tokens=64,
        maximum_h0_tokens=32_768,
        base_seed=20260721,
        response_model="trained-forward-step-16",
        service_instance_id="fixture-service",
        inference_state=ArchitectureInferenceState(
            method_id="skillev-bayesian-improve-trained-step-16@1",
            optimizer_steps=16,
            forward_adapter_active=True,
            posterior_active=True,
            calibration_active=True,
            action_policy_authority=("retrieved-skill-orchestration-plus-trained-forward-adapter"),
        ),
        arm=legacy_arm(),
    )

    receipt = client.architecture_identity
    assert receipt.optimizer_steps == 16
    assert receipt.forward_adapter_active is True
    assert receipt.reasoning_authority == "trained-forward-policy-deterministic"


def test_hotpot_root_query_excludes_the_ten_passage_source_envelope() -> None:
    source = (
        "Context passages:\n\nPassage 1:\npublic evidence"
        "\n\nQuestion:\nWhich public entity connects the passages?"
    )

    assert (
        _root_query_from_messages(
            ({"role": "user", "content": source},),
            benchmark_id="hotpotqa",
        )
        == "Which public entity connects the passages?"
    )


def test_aime_uses_native_thinking_and_integer_only_terminal() -> None:
    generator = _Generator(
        [
            (GenerationPhase.REASONING, r"The verified result is \boxed{42}."),
            (GenerationPhase.ACTION, "042"),
        ]
    )
    client = _client(
        generator,
        mode=StepZeroActionMode.COMPLETION,
        benchmark="aime-2026",
        completion_mode=StepZeroCompletionMode.AIME_BOXED_INTEGER,
        reasoning_mode=StepZeroReasoningMode.QWEN_THINKING,
        max_reasoning_tokens=128,
    )

    result = asyncio.run(
        client.generate(
            DirectGenerationRequest(
                "case-1",
                ({"role": "user", "content": "A public math problem"},),
                _profile(max_tokens=256, enable_thinking=True),
            )
        )
    )

    assert result.text == r"\boxed{42}"
    assert len(generator.tokenizer.thinking_prompts) == 1
    thinking_prompt = generator.tokenizer.thinking_prompts[0]
    assert "strategy comes from the retrieved skill" not in thinking_prompt
    assert "concise reasoning text only" not in thinking_prompt
    assert generator.profiles[0].max_new_tokens == 81_920
    assert generator.profiles[0].sampling_mode == "sampling"
    assert generator.profiles[0].temperature == 0.7
    assert generator.profiles[1].sampling_mode == "greedy"
    assert generator.profiles[1].temperature == 0.0
    assert client.architecture_identity.reasoning_authority == "frozen-benchmark-matched"
    assert client.architecture_identity.action_policy_authority == (
        "retrieved-skill-orchestration-plus-adapter-free-evaluation-matched-base"
    )
    assert generator.constraints[1] is not None
    assert generator.constraints[1].regex is not None


def test_aime_and_code_binding_metadata_does_not_forbid_non_thinking_arms() -> None:
    for benchmark in ("aime-2026", "mbpp-plus", "humaneval"):
        binding = StepZeroTaskBinding(
            task_id=f"{benchmark}-case",
            benchmark_id=benchmark,
            task_family="fixture",
            panel_index=0,
            action_mode=StepZeroActionMode.COMPLETION,
            completion_mode=(
                StepZeroCompletionMode.AIME_BOXED_INTEGER
                if benchmark == "aime-2026"
                else StepZeroCompletionMode.PYTHON_SOURCE
            ),
            reasoning_mode=StepZeroReasoningMode.STANDARD,
        )
        assert binding.reasoning_mode is StepZeroReasoningMode.STANDARD


def test_trivia_complementary_searches_then_terminal_use_only_the_trivia_skill() -> None:
    generator = _Generator(
        [
            (GenerationPhase.REASONING, "Search the key public relationship."),
            (
                GenerationPhase.ACTION,
                '{"arguments":{"query":"public entity relationship"},"kind":"tool",'
                '"name":"search","resource_id":"triviaqa"}',
            ),
            (GenerationPhase.REASONING, "Search the unresolved alternate entity name."),
            (
                GenerationPhase.ACTION,
                '{"arguments":{"query":"alternate entity relationship"},"kind":"tool",'
                '"name":"search","resource_id":"triviaqa"}',
            ),
            (GenerationPhase.REASONING, "Synthesize both sets of public snippets."),
            (GenerationPhase.ACTION, "Example"),
        ]
    )
    client = _client(
        generator,
        mode=StepZeroActionMode.TRIVIA_SEARCH,
        benchmark="triviaqa",
        completion_mode=StepZeroCompletionMode.SHORT_ANSWER,
        maximum_trivia_searches=2,
    )

    search = asyncio.run(
        client.generate(
            DirectGenerationRequest(
                "case-1:search-turn:1",
                ({"role": "user", "content": "Search this public question."},),
                _profile(),
            )
        )
    )
    second_search = asyncio.run(
        client.generate(
            DirectGenerationRequest(
                "case-1:search-turn:2",
                ({"role": "user", "content": "Question plus local Wikipedia snippets."},),
                _profile(),
            )
        )
    )
    final = asyncio.run(
        client.generate(
            DirectGenerationRequest(
                "case-1:search-turn:3",
                (
                    {
                        "role": "user",
                        "content": "Question plus two sets of local Wikipedia snippets.",
                    },
                ),
                _profile(),
            )
        )
    )

    assert search.text == "Search: public entity relationship"
    assert second_search.text == "Search: alternate entity relationship"
    assert final.text == "Example"
    assert client.diagnostics.retrieved_skill_ids == ("skill-trivia-search-synthesis",)
    assert all(
        "Make exactly one task-local search" not in item
        for item in generator.tokenizer.rollout_prompts
    )


def test_non_trivia_binding_cannot_change_the_trivia_search_budget() -> None:
    with pytest.raises(ValueError):
        StepZeroTaskBinding(
            task_id="hotpot-case",
            benchmark_id="hotpotqa",
            task_family="qa",
            panel_index=0,
            action_mode=StepZeroActionMode.COMPLETION,
            maximum_trivia_searches=2,
        )


def test_trivia_search_requires_an_explicit_stable_answer_free_root_query() -> None:
    with pytest.raises(ValueError):
        StepZeroTaskBinding(
            task_id="trivia-case",
            benchmark_id="triviaqa",
            task_family="qa",
            panel_index=0,
            action_mode=StepZeroActionMode.TRIVIA_SEARCH,
            completion_mode=StepZeroCompletionMode.SHORT_ANSWER,
            root_query=None,
        )


def test_interactive_episode_refreshes_current_prompt_and_surface_while_pinning_policy() -> None:
    generator = _Generator(
        [
            (
                GenerationPhase.REASONING,
                "Memory:\nGoal / hard constraints:\ncopied blue plate demonstration\n"
                "Current location / surface: search\nReady to purchase: no\n\n"
                "Thought:\nSearch with the required product and price constraints.\n\n"
                "Action:\nsearch[red mug]",
            ),
            (
                GenerationPhase.ACTION,
                '{"arguments":{"query":"red mug"},"kind":"tool",'
                '"name":"search","resource_id":"webshop"}',
            ),
            (
                GenerationPhase.REASONING,
                "Memory:\nGoal / hard constraints: red mug below 20 dollars\n"
                "Current location / surface: results\nCurrent candidate: b012345678\n"
                "Ready to purchase: no\n\nThought:\nOpen the current matching product.\n\n"
                "Action:\nclick[b012345678]",
            ),
            (
                GenerationPhase.ACTION,
                '{"arguments":{"target":"b012345678"},"kind":"tool",'
                '"name":"click","resource_id":"webshop"}',
            ),
        ]
    )
    client = _client(
        generator,
        mode=StepZeroActionMode.WEB_SHOP,
        benchmark="webshop",
        root_query="buy a red mug below 20 dollars",
    )
    task = NativeInteractiveTask(
        task_id="case-1",
        benchmark=DirectBenchmark.WEB_SHOP,
        task="buy a red mug below 20 dollars",
        profile=_profile(),
        max_steps=2,
        prompt_profile_id="webshop-step0-skill-owned-react-memory@1",
        parser_profile_id="native-action-memory-v6@1",
        population_id="fixture",
        run_seed=42,
        invalid_candidate_policy=InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
        invalid_environment_action_policy=(InvalidEnvironmentActionPolicy.DELEGATE_TO_OFFICIAL_ENV),
    )
    asyncio.run(client.begin_interactive_episode(task, NativePublicState("search", ("search",))))

    first = asyncio.run(
        client.generate(
            DirectGenerationRequest(
                "case-1:step:1",
                ({"role": "user", "content": "Task and current search surface"},),
                _profile(),
            )
        )
    )
    parsed = parse_visible_react(first.text)
    assert parsed.action.value == "search[red mug]"
    assert parsed.memory is not None
    assert "Constraints: buy a red mug below 20 dollars" in parsed.memory
    assert (
        "Environment status entering this decision: NOT TERMINAL; task incomplete" in parsed.memory
    )
    assert "Goal / hard constraints:" not in parsed.memory
    assert "copied blue plate demonstration" not in parsed.memory
    assert "Model working memory (cannot override" not in parsed.memory
    assert parsed.thought == "Search with the required product and price constraints."
    assert "memory" not in generator.scripted
    asyncio.run(
        client.observe_interactive_episode(
            "case-1",
            parsed.action.value,
            NativeEnvironmentStep(
                "results",
                False,
                0.0,
                False,
                True,
                available_actions=("click[b012345678]",),
            ),
        )
    )
    second = asyncio.run(
        client.generate(
            DirectGenerationRequest(
                "case-1:step:2",
                ({"role": "user", "content": "LATEST RESULTS SURFACE MARKER"},),
                _profile(),
            )
        )
    )
    parsed_second = parse_visible_react(second.text)
    assert parsed_second.action.value == "click[b012345678]"
    assert parsed_second.thought == "Open the current matching product."
    assert any(
        "LATEST RESULTS SURFACE MARKER" in item for item in generator.tokenizer.rollout_prompts
    )
    assert any(
        '"current_native_action_surface":["click[b012345678]"]' in item
        for item in generator.tokenizer.rollout_prompts
    )
    assert any(
        '"current_controller_memory":"Authoritative current task (verbatim): buy a red mug '
        "below 20 dollars"
        in item
        and "Last native action: search[red mug]" in item
        for item in generator.tokenizer.rollout_prompts
    )
    first_action_schema = generator.constraints[1]
    second_action_schema = generator.constraints[3]
    assert first_action_schema is not None
    assert second_action_schema is not None
    assert first_action_schema.json_schema["properties"]["name"] == {"const": "search"}  # type: ignore[index,union-attr]
    target_schema = second_action_schema.json_schema["properties"]["arguments"][  # type: ignore[index,union-attr]
        "properties"
    ]["target"]
    assert target_schema == {"enum": ["b012345678"], "type": "string"}
    asyncio.run(
        client.observe_interactive_episode(
            "case-1",
            parsed_second.action.value,
            NativeEnvironmentStep("product", True, 1.0, True, True),
        )
    )
    artifact = asyncio.run(
        client.finish_interactive_episode(
            "case-1",
            NativeEnvironmentOutcome(1.0, True, True),
        )
    )
    assert artifact.outer_steps == 2
    assert artifact.architecture_turns == 2
    assert not generator.active_episodes


def test_model_memory_cannot_replace_current_alfworld_target_or_completion_state() -> None:
    memory, thought = _project_visible_react_state(
        "Memory:\n"
        "Goal: put demonstration-object in demonstration-destination\n"
        "Remaining subgoals:\n"
        "- locate demonstration-object\n"
        "- put it in demonstration-destination\n"
        "Known object locations: current-object is on shelf 1\n"
        "Current location: shelf 1\n\n"
        "Thought:\nTake the current object from the observed shelf.\n\n"
        "Action:\ntake current-object from shelf 1",
        fallback_memory=(
            "Authoritative current task (verbatim): put current-object in current-destination\n"
            "Environment status entering this decision: NOT TERMINAL; task incomplete"
        ),
    )

    assert "demonstration-object" not in memory
    assert "demonstration-destination" not in memory
    assert "current-object is on shelf 1" in memory
    assert "current-destination" in memory
    assert thought == "Take the current object from the observed shelf."


def test_webshop_public_catalog_operator_routes_only_search_and_current_clicks() -> None:
    candidates = (
        WebShopPublicCatalogCandidate(
            product_id="X000000001",
            title="Synthetic cobalt watch case",
            price=20.0,
            suggested_search_query="Synthetic cobalt watch case",
            options=(("color", ("cobalt", "navy")),),
        ),
        WebShopPublicCatalogCandidate(
            product_id="X000000002",
            title="Synthetic cobalt watch band",
            price=18.0,
            suggested_search_query="Synthetic cobalt watch band",
        ),
    )
    memory = initial_episode_memory(
        benchmark="webshop",
        task_text="Find a cobalt watch accessory below 30 dollars.",
        task_type=None,
    )
    assert _webshop_catalog_navigation_action(candidates, memory, ("search",)) == (
        "search[Synthetic cobalt watch case]"
    )
    searched = update_episode_memory(
        memory,
        action="search[Synthetic cobalt watch case]",
        observation="Page 1",
        available_actions=("click[x000000001]", "click[x000000002]"),
    )
    assert (
        _webshop_catalog_navigation_action(
            candidates,
            searched,
            ("click[x000000001]", "click[x000000002]"),
        )
        == "click[x000000001]"
    )
    paged_search = update_episode_memory(
        memory,
        action="search[Synthetic cobalt watch case]",
        observation="Page 1 without the catalog candidate",
        available_actions=("click[next >]", "click[x000000099]"),
    )
    assert (
        _webshop_catalog_navigation_action(
            candidates,
            paged_search,
            ("click[next >]", "click[x000000099]"),
        )
        == "click[next >]"
    )
    deep_page = paged_search
    for page in range(2, 6):
        deep_page = update_episode_memory(
            deep_page,
            action="click[next >]",
            observation=f"Page {page} without the catalog candidate",
            available_actions=(
                "click[next >]",
                "click[back to search]",
                "click[x000000099]",
            ),
        )
    assert (
        _webshop_catalog_navigation_action(
            candidates,
            deep_page,
            ("click[next >]", "click[back to search]", "click[x000000099]"),
        )
        == "click[back to search]"
    )
    search_box = update_episode_memory(
        deep_page,
        action="click[back to search]",
        observation="Search box",
        available_actions=("search", "click[search]"),
    )
    assert (
        _webshop_catalog_navigation_action(
            candidates,
            search_box,
            ("search", "click[search]"),
        )
        == "search[Synthetic cobalt watch band]"
    )
    opened = update_episode_memory(
        searched,
        action="click[x000000001]",
        observation=(
            "Synthetic cobalt watch case [SEP] color [SEP] cobalt [SEP] navy [SEP] $20.00"
        ),
        available_actions=("click[< Prev]", "click[cobalt]", "click[navy]", "click[buy now]"),
    )
    assert (
        _webshop_catalog_navigation_action(
            candidates,
            opened,
            ("click[< Prev]", "click[cobalt]", "click[navy]", "click[buy now]"),
        )
        == "click[cobalt]"
    )
    selected = update_episode_memory(
        opened,
        action="click[cobalt]",
        observation=(
            "Synthetic cobalt watch case [SEP] color [SEP] cobalt [SEP] navy [SEP] $20.00"
        ),
        available_actions=("click[< Prev]", "click[cobalt]", "click[navy]", "click[buy now]"),
    )
    assert (
        _webshop_catalog_navigation_action(
            candidates,
            selected,
            ("click[< Prev]", "click[cobalt]", "click[navy]", "click[buy now]"),
        )
        == "click[buy now]"
    )
    returned = update_episode_memory(
        opened,
        action="click[< Prev]",
        observation="Page 1",
        available_actions=("click[x000000001]", "click[x000000002]"),
        terminal_success=False,
    )
    assert (
        _webshop_catalog_navigation_action(
            candidates,
            returned,
            ("click[x000000002]",),
        )
        == "click[x000000002]"
    )


def test_webshop_public_catalog_is_executed_only_by_the_retrieved_skill_operator() -> None:
    generator = _Generator([])
    client = _client(
        generator,
        mode=StepZeroActionMode.WEB_SHOP,
        benchmark="webshop",
        root_query="buy a cobalt watch case below 30 dollars",
        task_family="webshop",
        skill_instruction_token_budget=4096,
    )
    task = NativeInteractiveTask(
        task_id="case-1",
        benchmark=DirectBenchmark.WEB_SHOP,
        task="buy a cobalt watch case below 30 dollars",
        profile=_profile(),
        max_steps=3,
        prompt_profile_id="webshop-step0-skill-owned-react-memory@1",
        parser_profile_id="native-action-memory-v6@1",
        population_id="fixture",
        run_seed=42,
        webshop_catalog_candidates=(
            WebShopPublicCatalogCandidate(
                product_id="X000000001",
                title="Synthetic cobalt watch case",
                price=20.0,
                suggested_search_query="Synthetic cobalt watch case",
            ),
        ),
    )
    asyncio.run(client.begin_interactive_episode(task, NativePublicState("search", ("search",))))

    result = asyncio.run(
        client.generate(
            DirectGenerationRequest(
                "case-1:step:1",
                ({"role": "user", "content": "current search surface"},),
                _profile(),
            )
        )
    )
    parsed = parse_visible_react(result.text)

    assert parsed.action.value == "search[Synthetic cobalt watch case]"
    assert parsed.thought is not None
    assert "retrieved WebShop skill" in parsed.thought
    assert result.finish_reason == "step-zero-retrieved-skill-operator-action"
    assert result.response_model == "skill-webshop-constraint-ledger/operator@1"
    assert generator.profiles == []
    assert generator.tokenizer.thinking_prompts == []
    assert client.diagnostics.generations == 0
    assert client.diagnostics.public_catalog_actions == 1


def test_webshop_catalog_operator_selects_one_exact_public_option_per_group() -> None:
    candidates = (
        WebShopPublicCatalogCandidate(
            product_id="X000000001",
            title="Synthetic black watch case",
            price=20.0,
            suggested_search_query="Synthetic black watch case",
            options=(
                ("color", ("2 black", "black", "black-black", "blue")),
                ("size", ("10 inch (pack of 1)", "12 inch (pack of 1)")),
            ),
        ),
    )
    memory = initial_episode_memory(
        benchmark="webshop",
        task_text="Find a black 12 inch watch accessory below 30 dollars.",
        task_type=None,
    )
    observation = (
        "Instruction [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] "
        "2 black [SEP] black [SEP] black-black [SEP] blue [SEP] "
        "size [SEP] 10 inch (pack of 1) [SEP] 12 inch (pack of 1) [SEP] "
        "Synthetic black watch case [SEP] Price: $20.00 [SEP] Buy Now"
    )
    actions = (
        "click[< Prev]",
        "click[2 black]",
        "click[black]",
        "click[black-black]",
        "click[blue]",
        "click[10 inch (pack of 1)]",
        "click[12 inch (pack of 1)]",
        "click[buy now]",
    )
    opened = update_episode_memory(
        memory,
        action="click[x000000001]",
        observation=observation,
        available_actions=actions,
    )

    assert opened.webshop is not None
    assert opened.webshop.purchase_ready is False
    assert (
        _webshop_catalog_navigation_action(candidates, opened, actions, observation)
        == "click[black]"
    )

    selected = update_episode_memory(
        opened,
        action="click[black]",
        observation=observation,
        available_actions=actions,
    )
    assert selected.webshop is not None
    assert selected.webshop.selected_options == (("color", "black"),)
    assert selected.webshop.purchase_ready is False
    after_color_actions = tuple(action for action in actions if action != "click[black]")
    assert (
        _webshop_catalog_navigation_action(
            candidates,
            selected,
            after_color_actions,
            observation,
            option_surface=actions,
        )
        == "click[12 inch (pack of 1)]"
    )

    sized = update_episode_memory(
        selected,
        action="click[12 inch (pack of 1)]",
        observation=observation,
        available_actions=actions,
    )
    assert sized.webshop is not None
    assert sized.webshop.selected_options == (
        ("color", "black"),
        ("size", "12 inch (pack of 1)"),
    )
    assert sized.webshop.purchase_ready is True
    after_options = tuple(
        action for action in actions if action not in {"click[black]", "click[12 inch (pack of 1)]"}
    )
    assert (
        _webshop_catalog_navigation_action(
            candidates,
            sized,
            after_options,
            observation,
            option_surface=actions,
        )
        == "click[buy now]"
    )


def test_webshop_catalog_operator_matches_connectors_and_packaging_prose() -> None:
    candidates = (
        WebShopPublicCatalogCandidate(
            product_id="X000000003",
            title="Synthetic mineral blend",
            price=20.0,
            suggested_search_query="Synthetic mineral blend",
            options=(
                ("blend", ("cedar", "cedar + citrus mineral")),
                ("size", ("8 gram (pack of 4)", "24 gram (pack of 4)")),
                ("style", ("jar", "bag")),
            ),
        ),
    )
    memory = initial_episode_memory(
        benchmark="webshop",
        task_text=(
            "Find a pack of 4 bags of cedar and citrus mineral blend, with 24 grams in each bag."
        ),
        task_type=None,
    )
    actions = (
        "click[cedar]",
        "click[cedar + citrus mineral]",
        "click[8 gram (pack of 4)]",
        "click[24 gram (pack of 4)]",
        "click[jar]",
        "click[bag]",
        "click[buy now]",
    )
    opened = update_episode_memory(
        memory,
        action="click[x000000003]",
        observation="Synthetic product page",
        available_actions=actions,
    )

    assert (
        _webshop_catalog_navigation_action(
            candidates,
            opened,
            actions,
            "Synthetic product page",
            option_surface=actions,
        )
        == "click[cedar + citrus mineral]"
    )
    blended = update_episode_memory(
        opened,
        action="click[cedar + citrus mineral]",
        observation="Synthetic product page",
        available_actions=actions,
        webshop_option_groups=candidates[0].options,
    )
    assert (
        _webshop_catalog_navigation_action(
            candidates,
            blended,
            tuple(action for action in actions if action != "click[cedar + citrus mineral]"),
            "Synthetic product page",
            option_surface=actions,
        )
        == "click[24 gram (pack of 4)]"
    )
    sized = update_episode_memory(
        blended,
        action="click[24 gram (pack of 4)]",
        observation="Synthetic product page",
        available_actions=actions,
        webshop_option_groups=candidates[0].options,
    )
    assert (
        _webshop_catalog_navigation_action(
            candidates,
            sized,
            tuple(
                action
                for action in actions
                if action not in {"click[cedar + citrus mineral]", "click[24 gram (pack of 4)]"}
            ),
            "Synthetic product page",
            option_surface=actions,
        )
        == "click[bag]"
    )


def test_webshop_catalog_operator_prefers_full_conjunctive_option_over_short_match() -> None:
    candidate = WebShopPublicCatalogCandidate(
        product_id="X000000005",
        title="Synthetic freeze-dried fruit bundle",
        price=15.0,
        suggested_search_query="Synthetic freeze-dried fruit bundle",
        options=(("flavor", ("mangoes", "strawberries + mangos")),),
    )
    memory = initial_episode_memory(
        benchmark="webshop",
        task_text="Find a bundle of freeze-dried strawberries with mango flavor below 20 dollars.",
        task_type=None,
    )
    actions = ("click[mangoes]", "click[strawberries + mangos]", "click[buy now]")
    opened = update_episode_memory(
        memory,
        action="click[x000000005]",
        observation="Synthetic product page",
        available_actions=actions,
    )

    assert (
        _webshop_catalog_navigation_action(
            (candidate,),
            opened,
            actions,
            "Synthetic product page",
            option_surface=actions,
        )
        == "click[strawberries + mangos]"
    )


def test_webshop_catalog_operator_uses_public_recommendation_only_when_task_is_silent() -> None:
    candidate = WebShopPublicCatalogCandidate(
        product_id="X000000004",
        title="Synthetic desk organizer",
        price=20.0,
        suggested_search_query="Synthetic desk organizer",
        options=(
            ("size", ("small", "large")),
            ("finish", ("matte", "gloss")),
        ),
        suggested_options=(("finish", "matte"),),
    )
    memory = initial_episode_memory(
        benchmark="webshop",
        task_text="Find a large desk organizer below 30 dollars.",
        task_type=None,
    )
    actions = ("click[small]", "click[large]", "click[matte]", "click[gloss]", "click[buy now]")
    opened = update_episode_memory(
        memory,
        action="click[x000000004]",
        observation="Synthetic product page",
        available_actions=actions,
    )
    assert (
        _webshop_catalog_navigation_action(
            (candidate,),
            opened,
            actions,
            "Synthetic product page",
            option_surface=actions,
        )
        == "click[large]"
    )
    sized = update_episode_memory(
        opened,
        action="click[large]",
        observation="Synthetic product page",
        available_actions=actions,
        webshop_option_groups=candidate.options,
    )
    assert (
        _webshop_catalog_navigation_action(
            (candidate,),
            sized,
            tuple(action for action in actions if action != "click[large]"),
            "Synthetic product page",
            option_surface=actions,
        )
        == "click[matte]"
    )


def test_webshop_effective_surface_skips_price_but_preserves_ambiguous_accessories() -> None:
    binding = StepZeroTaskBinding(
        task_id="webshop-case",
        benchmark_id="webshop",
        task_family="webshop",
        panel_index=0,
        action_mode=StepZeroActionMode.WEB_SHOP,
    )
    memory = initial_episode_memory(
        benchmark="webshop",
        task_text="cobalt synthetic watch device with price lower than 25 dollars",
        task_type=None,
    )
    actions = (
        "click[x000000001]",
        "click[x000000002]",
        "click[next >]",
        "click[back to search]",
    )
    observation = (
        "Page 1 [SEP] X000000001 [SEP] Synthetic Watch Device (Cobalt) [SEP] $75.00 "
        "[SEP] X000000002 [SEP] Screen Protector for Synthetic Watch Device [SEP] $6.00"
    )

    assert _effective_native_actions(binding, memory, actions, observation) == (
        "click[x000000002]",
        "click[next >]",
    )

    mixed_actions = ("click[x000000003]", "click[next >]")
    assert (
        _reasoning_aligned_actions(
            binding,
            mixed_actions,
            "The current page contains only accessories; there is no viable product.",
        )
        == mixed_actions
    )
    assert _reasoning_aligned_actions(
        binding,
        mixed_actions,
        "I will use the current legal action `click[next >]`.",
    ) == ("click[next >]",)
    last_page_actions = ("click[x000000003]", "click[< Prev]", "click[back to search]")
    assert (
        _reasoning_aligned_actions(
            binding,
            last_page_actions,
            "All visible products are hard violations; there is no viable product.",
        )
        == last_page_actions
    )
    assert _reasoning_aligned_actions(
        binding,
        last_page_actions,
        "All are violations. Action: click[back to search]",
    ) == ("click[back to search]",)
    assert _effective_native_actions(
        binding,
        memory,
        last_page_actions,
        "Page 3 [SEP] X000000003 [SEP] Synthetic Watch Device [SEP] $79.00",
    ) == ("click[back to search]",)

    searched = update_episode_memory(
        memory,
        action="search[cobalt synthetic watch device]",
        observation="results",
        available_actions=actions,
    )
    assert _repeats_webshop_search(
        binding,
        "search[  COBALT synthetic   watch device ]",
        searched,
    )
    assert _webshop_search_has_price_prose(
        binding,
        "search[cobalt synthetic watch device under 25 dollars]",
    )
    assert not _webshop_search_has_price_prose(
        binding,
        "search[cobalt synthetic watch device]",
    )
    paged = searched
    for page in range(1, 6):
        paged = update_episode_memory(
            paged,
            action="click[next >]",
            observation=f"Page {page} results",
            available_actions=actions,
        )
    assert _effective_native_actions(
        binding,
        paged,
        (
            "click[x000000001]",
            "click[next >]",
            "click[< Prev]",
            "click[back to search]",
        ),
        "Page 5 [SEP] X000000001 [SEP] Synthetic Watch Device [SEP] $75.00",
    ) == ("click[back to search]",)
    candidate_limited = searched
    for asin in (
        "x000000004",
        "x000000005",
        "x000000006",
        "x000000007",
        "x000000008",
        "x000000009",
    ):
        candidate_limited = update_episode_memory(
            candidate_limited,
            action=f"click[{asin}]",
            observation="Synthetic Watch Device [SEP] Price: $20.00 [SEP] Buy Now",
            available_actions=("click[< Prev]", "click[buy now]"),
        )
        candidate_limited = update_episode_memory(
            candidate_limited,
            action="click[< Prev]",
            observation="Page 2 results",
            available_actions=actions,
        )
    assert _effective_native_actions(
        binding,
        candidate_limited,
        (
            "click[x000000010]",
            "click[next >]",
            "click[< Prev]",
            "click[back to search]",
        ),
        "Page 2 [SEP] X000000010 [SEP] Cobalt Synthetic Watch Device [SEP] $20.00",
    ) == ("click[back to search]",)

    opened = update_episode_memory(
        memory,
        action="click[x000000008]",
        observation="Cobalt synthetic watch device [SEP] Price: $20.00 [SEP] Buy Now",
        available_actions=("click[description]", "click[buy now]"),
    )
    assert opened.webshop is not None
    assert opened.webshop.required_information_tabs_remaining == ("description",)
    assert "click[buy now]" not in _blocked_native_actions(binding, opened)
    assert _reasoning_aligned_actions(
        binding,
        ("click[description]", "click[buy now]"),
        "The mandatory material is not confirmed by the title. Action: click[description]",
        memory=opened,
    ) == ("click[description]",)
    described = update_episode_memory(
        opened,
        action="click[description]",
        observation="Description page confirms synthetic material and cobalt finish.",
        available_actions=("click[< Prev]",),
    )
    assert "click[description]" in _blocked_native_actions(binding, described)
    assert described.webshop is not None
    assert described.webshop.product_information_evidence == (
        (
            "x000000008",
            "description",
            "Description page confirms synthetic material and cobalt finish.",
        ),
    )
    assert "persistent public information evidence" in described.render().casefold()
    assert "synthetic material" in described.render()

    option_page = (
        "Instruction [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] cobalt [SEP] "
        "navy [SEP] Synthetic Watch Device (Cobalt) [SEP] Price: $20.00 [SEP] Buy Now"
    )
    option_actions = (
        "click[back to search]",
        "click[< Prev]",
        "click[cobalt]",
        "click[navy]",
        "click[buy now]",
    )
    option_opened = update_episode_memory(
        memory,
        action="click[x000000008]",
        observation=option_page,
        available_actions=option_actions,
    )
    assert option_opened.webshop is not None
    assert option_opened.webshop.purchase_ready is False
    option_selected = update_episode_memory(
        option_opened,
        action="click[cobalt]",
        observation=option_page,
        available_actions=option_actions,
    )
    assert option_selected.webshop is not None
    assert option_selected.webshop.selected_options == (("color", "cobalt"),)
    assert option_selected.webshop.purchase_ready is True
    assert "click[cobalt]" in _blocked_native_actions(binding, option_selected)
    assert _reasoning_aligned_actions(
        binding,
        ("click[< Prev]", "click[buy now]"),
        "The mandatory synthetic material has not been confirmed and lacks positive public "
        "evidence. Action: click[buy now]",
        memory=option_selected,
    ) == ("click[buy now]",)


def test_alfworld_effective_surface_advances_visible_target_and_typed_subgoals() -> None:
    binding = StepZeroTaskBinding(
        task_id="alfworld-case",
        benchmark_id="alfworld",
        task_family="alfworld",
        panel_index=0,
        action_mode=StepZeroActionMode.ALF_WORLD,
    )
    memory = initial_episode_memory(
        benchmark="alfworld",
        task_text="Place a cooled target-object on a destination-surface.",
        task_type="pick_cool_then_place",
    )
    at_fridge = update_episode_memory(
        memory,
        action="go to fridge 1",
        observation="You arrive at fridge 1. The fridge 1 is closed.",
        available_actions=("open fridge 1", "go to destination-surface 1"),
    )
    assert _effective_native_actions(
        binding,
        at_fridge,
        ("open fridge 1", "go to destination-surface 1"),
        "The fridge 1 is closed.",
    ) == ("open fridge 1",)

    opened = update_episode_memory(
        at_fridge,
        action="open fridge 1",
        observation="The fridge is open. In it, you see a target-object 1 and a distractor 1.",
        available_actions=(
            "take target-object 1 from fridge 1",
            "take distractor 1 from fridge 1",
            "go to destination-surface 1",
        ),
    )
    assert _effective_native_actions(
        binding,
        opened,
        (
            "take target-object 1 from fridge 1",
            "take distractor 1 from fridge 1",
            "go to destination-surface 1",
        ),
        "The open fridge contains the synthetic target and a distractor.",
    ) == ("take target-object 1 from fridge 1",)

    held = update_episode_memory(
        opened,
        action="take target-object 1 from fridge 1",
        observation="You pick up target-object 1.",
        available_actions=(
            "cool target-object 1 with fridge 1",
            "go to destination-surface 1",
        ),
    )
    assert _effective_native_actions(
        binding,
        held,
        ("cool target-object 1 with fridge 1", "go to destination-surface 1"),
        "Inventory: target-object 1.",
    ) == ("cool target-object 1 with fridge 1",)

    cooled = update_episode_memory(
        held,
        action="cool target-object 1 with fridge 1",
        observation="You cool target-object 1.",
        available_actions=(
            "open fridge 1",
            "go to fridge 1",
            "go to destination-surface 1",
        ),
    )
    assert _effective_native_actions(
        binding,
        cooled,
        ("open fridge 1", "go to fridge 1", "go to destination-surface 1"),
        "The target-object is cool.",
    ) == ("go to destination-surface 1",)

    search_memory = initial_episode_memory(
        benchmark="alfworld",
        task_text="Put one synthetic-token on destination-a.",
        task_type="pick_and_place",
    )
    exhausted = update_episode_memory(
        search_memory,
        action="go to source-a 1",
        observation="On source-a 1, you see a distractor 1.",
        available_actions=("go to source-a 1", "go to destination-a 1"),
    )
    assert _effective_native_actions(
        binding,
        exhausted,
        ("go to source-a 1", "go to destination-a 1"),
        "The source has no synthetic-token.",
    ) == ("go to destination-a 1",)

    recovery_memory = initial_episode_memory(
        benchmark="alfworld",
        task_text="Place a heated target-token in destination-box.",
        task_type="pick_heat_then_place",
    )
    assert _effective_native_actions(
        binding,
        recovery_memory,
        (
            "take distractor-token 1 from source-box 1",
            "go to destination-box 1",
        ),
        "A distractor is visible.",
    ) == ("go to destination-box 1",)
    wrong_held = update_episode_memory(
        recovery_memory,
        action="take distractor-token 1 from source-box 1",
        observation="You pick up distractor-token 1.",
        available_actions=("move distractor-token 1 to source-box 1",),
    )
    assert wrong_held.alfworld is not None
    assert wrong_held.alfworld.target_object is None
    assert "clear inventory" in wrong_held.alfworld.next_subgoal
    assert _effective_native_actions(
        binding,
        wrong_held,
        ("move distractor-token 1 to source-box 1",),
        "Inventory contains distractor-token 1.",
    ) == ("move distractor-token 1 to source-box 1",)
    recovered = update_episode_memory(
        wrong_held,
        action="move distractor-token 1 to source-box 1",
        observation="You put down distractor-token 1.",
        available_actions=("go to destination-box 1",),
    )
    assert recovered.alfworld is not None
    assert recovered.alfworld.held_object is None
    assert recovered.alfworld.placed_count == 0

    look_memory = initial_episode_memory(
        benchmark="alfworld",
        task_text="Look at target-token under a desk lamp.",
        task_type="look_at_obj",
    )
    look_held = update_episode_memory(
        look_memory,
        action="take target-token 1 from source-a 1",
        observation="You pick up target-token 1.",
        available_actions=("go to desklamp 1",),
    )
    assert _effective_native_actions(
        binding,
        look_held,
        (
            "examine target-token 1",
            "use desklamp 1",
            "go to source-a 1",
        ),
        "You are carrying the target beside a desk lamp.",
    ) == ("use desklamp 1",)

    two_memory = initial_episode_memory(
        benchmark="alfworld",
        task_text="Put two target-tokens in destination-box.",
        task_type="pick_two_obj",
    )
    at_source = update_episode_memory(
        two_memory,
        action="go to source-a 1",
        observation="Two target tokens are visible.",
        available_actions=(
            "take target-token 1 from source-a 1",
            "take target-token 2 from source-a 1",
        ),
    )
    first_held = update_episode_memory(
        at_source,
        action="take target-token 1 from source-a 1",
        observation="You pick up target-token 1.",
        available_actions=("go to destination-box 1",),
    )
    at_destination = update_episode_memory(
        first_held,
        action="go to destination-box 1",
        observation="The destination box is open.",
        available_actions=("move target-token 1 to destination-box 1",),
    )
    first_placed = update_episode_memory(
        at_destination,
        action="move target-token 1 to destination-box 1",
        observation="You place target-token 1 in destination-box 1.",
        available_actions=(
            "take target-token 1 from destination-box 1",
            "go to source-a 1",
        ),
    )
    assert first_placed.alfworld is not None
    assert first_placed.alfworld.placed_objects == ("target-token 1",)
    assert first_placed.alfworld.target_source_location == "source-a 1"
    assert _effective_native_actions(
        binding,
        first_placed,
        (
            "take target-token 1 from destination-box 1",
            "go to source-a 1",
        ),
        "The first target is already in the destination.",
    ) == ("go to source-a 1",)
    returned_to_source = update_episode_memory(
        first_placed,
        action="go to source-a 1",
        observation="Both target instances are visible in public state.",
        available_actions=(
            "take target-token 1 from source-a 1",
            "take target-token 2 from source-a 1",
        ),
    )
    assert _effective_native_actions(
        binding,
        returned_to_source,
        (
            "take target-token 1 from source-a 1",
            "take target-token 2 from source-a 1",
        ),
        "One unplaced target remains.",
    ) == ("take target-token 2 from source-a 1",)


def test_alfworld_controller_preserves_location_instances_and_public_aliases() -> None:
    binding = StepZeroTaskBinding(
        task_id="alfworld-alias-case",
        benchmark_id="alfworld",
        task_family="alfworld",
        panel_index=0,
        action_mode=StepZeroActionMode.ALF_WORLD,
    )
    search = initial_episode_memory(
        benchmark="alfworld",
        task_text="Put a synthetic-token in a destination-box.",
        task_type="pick_and_place",
    )
    first_drawer = update_episode_memory(
        search,
        action="go to drawer 1",
        observation="Drawer 1 is open and empty.",
        available_actions=("go to drawer 1", "go to drawer 2"),
    )
    assert _effective_native_actions(
        binding,
        first_drawer,
        ("go to drawer 1", "go to drawer 2"),
        "Drawer 1 is open and empty.",
    ) == ("go to drawer 2",)

    shakers = initial_episode_memory(
        benchmark="alfworld",
        task_text="Move two synthetic shakers into a synthetic drawer.",
        task_type="pick_two_obj",
    )
    assert _effective_native_actions(
        binding,
        shakers,
        (
            "take peppershaker 1 from shelf 1",
            "take spoon 1 from shelf 1",
            "go to drawer 1",
        ),
        "A shaker and a spoon are visible.",
    ) == ("take peppershaker 1 from shelf 1",)

    explicit_pepper = initial_episode_memory(
        benchmark="alfworld",
        task_text="Put some pepper shaker in a drawer.",
        task_type="pick_and_place",
    )
    assert _effective_native_actions(
        binding,
        explicit_pepper,
        (
            "take saltshaker 1 from shelf 1",
            "take peppershaker 1 from shelf 1",
            "go to drawer 1",
        ),
        "A salt shaker and a pepper shaker are visible.",
    ) == ("take peppershaker 1 from shelf 1",)
    rendered = explicit_pepper.render()
    assert "Required target phrase from the authoritative task: Put some pepper shaker" in rendered

    explicit_salt = initial_episode_memory(
        benchmark="alfworld",
        task_text="Put some salt shaker in a drawer.",
        task_type="pick_and_place",
    )
    initial_salt_candidates = _effective_native_actions(
        binding,
        explicit_salt,
        (
            "take peppershaker 1 from shelf 1",
            "take saltshaker 1 from shelf 1",
            "go to drawer 1",
        ),
        "A pepper shaker and a salt shaker are visible.",
    )
    assert initial_salt_candidates == (
        "take peppershaker 1 from shelf 1",
        "take saltshaker 1 from shelf 1",
    )
    held_salt = update_episode_memory(
        explicit_salt,
        action="take saltshaker 1 from shelf 1",
        observation="Saltshaker 1 is in inventory.",
        available_actions=("move saltshaker 1 to drawer 1",),
    )
    disproven_salt = update_episode_memory(
        held_salt,
        action="move saltshaker 1 to drawer 1",
        observation="The drawer now contains saltshaker 1, but the task is not complete.",
        available_actions=("take peppershaker 1 from shelf 1",),
        terminal_success=False,
    )
    assert _effective_native_actions(
        binding,
        disproven_salt,
        ("take peppershaker 1 from shelf 1",),
        "A pepper shaker remains on shelf 1.",
    ) == ("take peppershaker 1 from shelf 1",)

    sponge = initial_episode_memory(
        benchmark="alfworld",
        task_text="Clean one synthetic sponge, then place it on a synthetic metal rack.",
        task_type="pick_clean_then_place",
    )
    assert _effective_native_actions(
        binding,
        sponge,
        (
            "take dishsponge 1 from bathtubbasin 1",
            "take soapbar 1 from bathtubbasin 1",
        ),
        "A dish sponge and soap are visible.",
    ) == ("take dishsponge 1 from bathtubbasin 1",)


def test_alfworld_destination_span_excludes_transform_appliance_and_compound_prefix() -> None:
    binding = StepZeroTaskBinding(
        task_id="alfworld-destination-case",
        benchmark_id="alfworld",
        task_family="alfworld",
        panel_index=0,
        action_mode=StepZeroActionMode.ALF_WORLD,
    )
    cool = initial_episode_memory(
        benchmark="alfworld",
        task_text=(
            "Cool synthetic bread with a synthetic fridge, then place it on the counter left "
            "of a synthetic stove."
        ),
        task_type="pick_cool_then_place",
    )
    held = update_episode_memory(
        cool,
        action="take bread 1 from countertop 2",
        observation="Bread is in inventory.",
        available_actions=("cool bread 1 with fridge 1",),
    )
    cooled = update_episode_memory(
        held,
        action="cool bread 1 with fridge 1",
        observation="The bread is chilled.",
        available_actions=(
            "go to fridge 1",
            "go to countertop 1",
            "go to countertop 2",
        ),
    )
    assert _effective_native_actions(
        binding,
        cooled,
        (
            "go to fridge 1",
            "go to countertop 1",
            "go to countertop 2",
        ),
        "The bread is chilled.",
    ) == ("go to countertop 1", "go to countertop 2")

    paper = initial_episode_memory(
        benchmark="alfworld",
        task_text="Place synthetic toilet paper onto a synthetic toilet paper holder.",
        task_type="pick_and_place",
    )
    paper_held = update_episode_memory(
        paper,
        action="take toiletpaper 1 from shelf 1",
        observation="Toilet paper is in inventory.",
        available_actions=("go to toilet 1", "go to toiletpaperhanger 1"),
    )
    assert _effective_native_actions(
        binding,
        paper_held,
        ("go to toilet 1", "go to toiletpaperhanger 1"),
        "Toilet paper is in inventory.",
    ) == ("go to toiletpaperhanger 1",)


def test_alfworld_look_task_keeps_target_and_returns_to_publicly_seen_lamp() -> None:
    binding = StepZeroTaskBinding(
        task_id="alfworld-look-case",
        benchmark_id="alfworld",
        task_family="alfworld",
        panel_index=0,
        action_mode=StepZeroActionMode.ALF_WORLD,
    )
    memory = initial_episode_memory(
        benchmark="alfworld",
        task_text="Turn on a light with a synthetic-box in hand.",
        task_type="look_at_obj",
    )
    lamp_seen = update_episode_memory(
        memory,
        action="go to dresser 1",
        observation="A desk lamp 1 is on dresser 1.",
        available_actions=("go to diningtable 1",),
    )
    at_target = update_episode_memory(
        lamp_seen,
        action="go to diningtable 1",
        observation="A synthetic-box 1 is on diningtable 1.",
        available_actions=("take synthetic-box 1 from diningtable 1",),
    )
    held = update_episode_memory(
        at_target,
        action="take synthetic-box 1 from diningtable 1",
        observation="The synthetic box is in inventory.",
        available_actions=(
            "go to dresser 1",
            "move synthetic-box 1 to diningtable 1",
        ),
    )
    assert held.alfworld is not None
    assert held.alfworld.look_source_location == "dresser 1"
    assert _effective_native_actions(
        binding,
        held,
        ("go to dresser 1", "move synthetic-box 1 to diningtable 1"),
        "The synthetic box is in inventory.",
    ) == ("go to dresser 1",)


def test_alfworld_look_task_does_not_use_lamp_before_acquiring_target() -> None:
    binding = StepZeroTaskBinding(
        task_id="alfworld-look-search-case",
        benchmark_id="alfworld",
        task_family="alfworld",
        panel_index=0,
        action_mode=StepZeroActionMode.ALF_WORLD,
    )
    memory = initial_episode_memory(
        benchmark="alfworld",
        task_text="Look at a synthetic-box under a desk lamp.",
        task_type="look_at_obj",
    )
    at_lamp = update_episode_memory(
        memory,
        action="go to desk 1",
        observation="A desk lamp 1 is on desk 1, but the target is not here.",
        available_actions=(
            "use desklamp 1",
            "go to shelf 1",
            "go to cabinet 1",
        ),
    )
    assert at_lamp.alfworld is not None
    assert at_lamp.alfworld.held_object is None
    assert _effective_native_actions(
        binding,
        at_lamp,
        (
            "use desklamp 1",
            "go to shelf 1",
            "go to cabinet 1",
        ),
        "A desk lamp 1 is on desk 1, but the target is not here.",
    ) == ("go to shelf 1", "go to cabinet 1")


def test_single_object_placement_is_not_completed_until_environment_confirms_success() -> None:
    memory = initial_episode_memory(
        benchmark="alfworld",
        task_text="Put a target-token in a destination-box.",
        task_type="pick_and_place",
    )
    held = update_episode_memory(
        memory,
        action="take target-token 1 from source-box 1",
        observation="Target is in inventory.",
        available_actions=("move target-token 1 to destination-box 1",),
    )
    unconfirmed = update_episode_memory(
        held,
        action="move target-token 1 to destination-box 1",
        observation="The environment remains active.",
        available_actions=("go to source-box 1",),
        terminal_success=False,
    )
    confirmed = update_episode_memory(
        held,
        action="move target-token 1 to destination-box 1",
        observation="The task is complete.",
        available_actions=(),
        terminal_success=True,
    )

    assert unconfirmed.alfworld is not None
    assert unconfirmed.alfworld.placed_count == 0
    assert unconfirmed.alfworld.rejected_target_objects == ("target-token 1",)
    assert "remained nonterminal" in unconfirmed.alfworld.next_subgoal
    assert confirmed.alfworld is not None
    assert confirmed.alfworld.placed_count == 1


def test_alfworld_opens_current_destination_before_routing_to_another_instance() -> None:
    binding = StepZeroTaskBinding(
        task_id="alfworld-open-destination",
        benchmark_id="alfworld",
        task_family="alfworld",
        panel_index=0,
        action_mode=StepZeroActionMode.ALF_WORLD,
    )
    memory = initial_episode_memory(
        benchmark="alfworld",
        task_text="Move two synthetic shakers into a synthetic drawer.",
        task_type="pick_two_obj",
    )
    held = update_episode_memory(
        memory,
        action="take peppershaker 1 from shelf 1",
        observation="A synthetic shaker is in inventory.",
        available_actions=("go to drawer 1",),
    )
    at_closed_destination = update_episode_memory(
        held,
        action="go to drawer 1",
        observation="Drawer 1 is closed.",
        available_actions=("open drawer 1", "go to drawer 2"),
    )

    assert _effective_native_actions(
        binding,
        at_closed_destination,
        ("open drawer 1", "go to drawer 2"),
        "Drawer 1 is closed.",
    ) == ("open drawer 1",)


def test_alfworld_public_destination_phrases_cover_plain_table_and_carry_to() -> None:
    binding = StepZeroTaskBinding(
        task_id="alfworld-public-destination",
        benchmark_id="alfworld",
        task_family="alfworld",
        panel_index=0,
        action_mode=StepZeroActionMode.ALF_WORLD,
    )
    table_task = initial_episode_memory(
        benchmark="alfworld",
        task_text="Clean a synthetic spatula, then place it on a synthetic table.",
        task_type="pick_clean_then_place",
    )
    table_held = update_episode_memory(
        table_task,
        action="take spatula 1 from shelf 1",
        observation="Spatula is in inventory.",
        available_actions=("clean spatula 1 with sinkbasin 1",),
    )
    table_clean = update_episode_memory(
        table_held,
        action="clean spatula 1 with sinkbasin 1",
        observation="The spatula is clean.",
        available_actions=("go to diningtable 1", "go to sidetable 1"),
    )
    assert _effective_native_actions(
        binding,
        table_clean,
        ("go to diningtable 1", "go to sidetable 1"),
        "The spatula is clean.",
    ) == ("go to diningtable 1",)

    carry_task = initial_episode_memory(
        benchmark="alfworld",
        task_text="Clean a synthetic mug and carry it to a synthetic coffee machine.",
        task_type="pick_clean_then_place",
    )
    carry_held = update_episode_memory(
        carry_task,
        action="take mug 1 from shelf 1",
        observation="Mug is in inventory.",
        available_actions=("clean mug 1 with sinkbasin 1",),
    )
    carry_clean = update_episode_memory(
        carry_held,
        action="clean mug 1 with sinkbasin 1",
        observation="The mug is clean.",
        available_actions=("go to coffeemachine 1", "go to countertop 1"),
    )
    assert _effective_native_actions(
        binding,
        carry_clean,
        ("go to coffeemachine 1", "go to countertop 1"),
        "The mug is clean.",
    ) == ("go to coffeemachine 1",)


def test_alfworld_nonterminal_single_placement_rejects_that_public_instance() -> None:
    binding = StepZeroTaskBinding(
        task_id="alfworld-target-recovery",
        benchmark_id="alfworld",
        task_family="alfworld",
        panel_index=0,
        action_mode=StepZeroActionMode.ALF_WORLD,
    )
    memory = initial_episode_memory(
        benchmark="alfworld",
        task_text="Clean a synthetic knife and place it in a synthetic drawer.",
        task_type="pick_clean_then_place",
    )
    held = update_episode_memory(
        memory,
        action="take knife 1 from shelf 1",
        observation="Knife is in inventory.",
        available_actions=("clean knife 1 with sinkbasin 1",),
    )
    cleaned = update_episode_memory(
        held,
        action="clean knife 1 with sinkbasin 1",
        observation="The knife is clean.",
        available_actions=("move knife 1 to drawer 1",),
    )
    rejected = update_episode_memory(
        cleaned,
        action="move knife 1 to drawer 1",
        observation="The environment remains active.",
        available_actions=(
            "take knife 1 from drawer 1",
            "take butterknife 2 from drawer 1",
        ),
        terminal_success=False,
    )

    assert _effective_native_actions(
        binding,
        rejected,
        (
            "take knife 1 from drawer 1",
            "take butterknife 2 from drawer 1",
        ),
        "Two public knife instances are visible.",
    ) == ("take butterknife 2 from drawer 1",)
