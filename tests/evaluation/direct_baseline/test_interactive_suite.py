from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from skillev.evaluation.direct_baseline.client import (
    DirectGenerationRequest,
    DirectGenerationResult,
)
from skillev.evaluation.direct_baseline.config import DirectBenchmark, DirectDecodingProfile
from skillev.evaluation.direct_baseline.interactive_tasks import (
    InvalidCandidatePolicy,
    NativeEnvironmentOutcome,
    NativeEnvironmentStep,
    NativeInteractiveTask,
    NativePublicState,
    run_native_interactive_task,
)
from skillev.evaluation.direct_baseline.prompts import WebShopPublicCatalogCandidate


def _profile() -> DirectDecodingProfile:
    return DirectDecodingProfile("native@1", False, 0.7, 0.8, 20, 0, 1.5, 1, 200)


@dataclass
class _Client:
    outputs: list[str]
    requests: list[DirectGenerationRequest] = field(default_factory=list)

    async def generate(self, request: DirectGenerationRequest) -> DirectGenerationResult:
        self.requests.append(request)
        return DirectGenerationResult(request.request_id, self.outputs.pop(0), "stop", 1, 1)


@dataclass
class _PersistentMemoryClient(_Client):
    live_memory: str = "LIVE CONTROLLER STATE 0"

    async def begin_interactive_episode(
        self,
        task: NativeInteractiveTask,
        state: NativePublicState,
    ) -> None:
        del task, state

    async def observe_interactive_episode(
        self,
        task_id: str,
        action: str | None,
        result: NativeEnvironmentStep | NativePublicState,
    ) -> None:
        del task_id, action, result
        self.live_memory = "LIVE CONTROLLER STATE 1"

    def current_interactive_memory(self, task_id: str) -> str:
        del task_id
        return self.live_memory

    async def finish_interactive_episode(
        self,
        task_id: str,
        outcome: NativeEnvironmentOutcome,
    ) -> None:
        del task_id, outcome


@dataclass
class _Environment:
    closed: bool = False
    step_count: int = 0
    terminal: bool = True
    close_error: bool = False
    action_valid: bool | None = True

    async def reset(self) -> NativePublicState:
        return NativePublicState("search page", ("search", "click[search]"))

    async def step(self, action: str) -> NativeEnvironmentStep:
        self.step_count += 1
        return NativeEnvironmentStep(
            "purchased",
            self.terminal,
            1.0,
            self.terminal,
            self.action_valid,
            available_actions=("search", "click[search]"),
        )

    async def outcome(self) -> NativeEnvironmentOutcome:
        return NativeEnvironmentOutcome(0.25, False, False)

    async def close(self) -> None:
        if self.close_error:
            raise RuntimeError("worker cleanup failed")
        self.closed = True


def _task(
    task_id: str,
    max_steps: int,
    *,
    invalid_candidate_policy: InvalidCandidatePolicy = InvalidCandidatePolicy.TERMINATE_ZERO,
) -> NativeInteractiveTask:
    return NativeInteractiveTask(
        task_id,
        DirectBenchmark.WEB_SHOP,
        "buy item",
        _profile(),
        max_steps,
        prompt_profile_id="webshop-native-react@2",
        parser_profile_id="native-action@2",
        population_id="test-panel@1",
        run_seed=42,
        invalid_candidate_policy=invalid_candidate_policy,
    )


def test_native_interaction_reaches_terminal_and_cleans_up() -> None:
    environment = _Environment()
    attempt = asyncio.run(
        run_native_interactive_task(
            _Client(["Thought: locate it\nAction: search[item]"]),
            _task("webshop:1", 5),
            environment,
        )
    )
    assert attempt.reward == 1
    assert attempt.success is True
    assert attempt.valid_actions == 1
    assert attempt.infrastructure_error is None
    assert attempt.trace[0].prompt_tokens == 1
    assert attempt.trace[0].completion_tokens == 1
    assert environment.closed


def test_alfworld_reset_task_overrides_a_stale_manifest_annotation() -> None:
    @dataclass
    class Environment(_Environment):
        async def reset(self) -> NativePublicState:
            return NativePublicState(
                "-= Welcome =-\n\nYour task is to: put a clean plate on the countertop.",
                ("look",),
            )

    task = NativeInteractiveTask(
        "alfworld:1",
        DirectBenchmark.ALF_WORLD,
        "wash the dirty bowl before putting it on the counter",
        _profile(),
        1,
        prompt_profile_id="alfworld-native-react@2",
        parser_profile_id="native-action@2",
        population_id="test-panel@1",
        run_seed=42,
        task_type="pick_clean_then_place",
    )
    client = _Client(["Thought: inspect the room\nAction: look"])

    attempt = asyncio.run(run_native_interactive_task(client, task, Environment()))

    prompt = client.requests[0].messages[-1]["content"]
    assert attempt.success is True
    assert "put a clean plate on the countertop" in prompt
    assert "dirty bowl" not in prompt


def test_invalid_action_is_candidate_failure_without_resampling() -> None:
    environment = _Environment()
    client = _Client(["I cannot\ndecide", "Action: search[item]"])
    attempt = asyncio.run(
        run_native_interactive_task(
            client,
            _task("webshop:2", 5),
            environment,
        )
    )
    assert attempt.reward == 0
    assert attempt.invalid_actions == 1
    assert len(client.outputs) == 1
    assert environment.step_count == 0
    assert environment.closed


def test_horizon_preserves_native_partial_score() -> None:
    environment = _Environment(terminal=False)
    attempt = asyncio.run(
        run_native_interactive_task(
            _Client(["search[item]"]),
            _task("webshop:3", 1),
            environment,
        )
    )
    assert attempt.reward == 0.25
    assert attempt.terminal_reached is False
    assert attempt.terminated_by_horizon is True
    assert attempt.termination_reason == "horizon"
    assert environment.closed


def test_cleanup_failure_is_classified_as_infrastructure() -> None:
    environment = _Environment(close_error=True)
    attempt = asyncio.run(
        run_native_interactive_task(
            _Client(["search[item]"]),
            _task("webshop:4", 1),
            environment,
        )
    )
    assert attempt.reward == 1
    assert attempt.infrastructure_error is None
    assert attempt.cleanup_error == "RuntimeError"


def test_invalid_candidate_can_consume_a_step_without_resampling() -> None:
    environment = _Environment()
    client = _Client(["I cannot\ndecide", "Action: search[item]"])
    task = _task(
        "webshop:5",
        2,
        invalid_candidate_policy=InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
    )
    attempt = asyncio.run(run_native_interactive_task(client, task, environment))
    assert attempt.reward == 1
    assert attempt.steps == 2
    assert attempt.invalid_actions == 1
    assert environment.step_count == 1


def test_nonterminal_status_is_explicit_in_the_next_model_input() -> None:
    environment = _Environment(terminal=False)
    client = _Client(["search[item]", "search[item]"])

    asyncio.run(run_native_interactive_task(client, _task("webshop:6", 2), environment))

    assert len(client.requests) == 2
    assert "Environment status: NOT TERMINAL" in client.requests[1].messages[-1]["content"]


def test_public_surface_drives_invalid_action_count_when_environment_omits_validity() -> None:
    environment = _Environment(terminal=False, action_valid=None)
    attempt = asyncio.run(
        run_native_interactive_task(
            _Client(["click[stale-product]"]),
            _task("webshop:7", 1),
            environment,
        )
    )

    assert attempt.invalid_actions == 1
    assert attempt.valid_actions == 0
    assert attempt.trace[0].action_listed_before is False
    assert attempt.trace[0].official_action_valid is None


def test_persistent_client_live_memory_replaces_the_prior_pre_action_copy() -> None:
    environment = _Environment(terminal=False)
    client = _PersistentMemoryClient(
        [
            "Memory:\nMODEL PRE-ACTION 0\n\nThought:\nsearch\n\nAction:\nsearch[item]",
            "Memory:\nMODEL PRE-ACTION 1\n\nThought:\nsearch\n\nAction:\nsearch[item]",
        ]
    )
    task = NativeInteractiveTask(
        "webshop:live-memory",
        DirectBenchmark.WEB_SHOP,
        "buy synthetic item",
        _profile(),
        2,
        prompt_profile_id="webshop-native-react-memory-v18@1",
        parser_profile_id="native-action-memory-v6@1",
        population_id="test-panel@1",
        run_seed=42,
        invalid_candidate_policy=InvalidCandidatePolicy.CONSUME_STEP_AND_CONTINUE,
    )

    attempt = asyncio.run(run_native_interactive_task(client, task, environment))

    assert "LIVE CONTROLLER STATE 0" in client.requests[0].messages[-1]["content"]
    second_prompt = client.requests[1].messages[-1]["content"]
    accumulated = (
        second_prompt.split(
            "Accumulated structured memory before this action:\n",
            1,
        )[1]
        .split("\nCurrent decision policy:", 1)[0]
        .strip()
    )
    assert accumulated == "LIVE CONTROLLER STATE 1"
    assert attempt.trace[0].structured_memory_after == "LIVE CONTROLLER STATE 1"
    assert attempt.trace[0].memory_update_status == "controller-observed"


def test_public_catalog_candidate_is_answer_independent_navigation_context() -> None:
    candidate = WebShopPublicCatalogCandidate(
        product_id="PUBLIC-ITEM-1",
        title="Synthetic cobalt travel bottle six-pack",
        price=18.5,
        suggested_search_query="Synthetic cobalt travel bottle six-pack",
        attributes=("leak proof", "BPA free"),
        options=(("color", ("amber", "clear")), ("count", ("pack of 6",))),
    )
    environment = _Environment()
    client = _Client(
        [
            "Memory:\nReady to purchase: no\n\n"
            "Thought:\nSearch the public catalog suggestion.\n\n"
            "Action:\nsearch[Synthetic cobalt travel bottle six-pack]"
        ]
    )
    task = NativeInteractiveTask(
        "webshop:catalog-context",
        DirectBenchmark.WEB_SHOP,
        "buy an amber leak-proof six-pack",
        _profile(),
        1,
        prompt_profile_id="webshop-native-react-memory-v18@1",
        parser_profile_id="native-action-memory-v6@1",
        population_id="test-panel@1",
        run_seed=42,
        webshop_catalog_candidates=(candidate,),
    )

    attempt = asyncio.run(run_native_interactive_task(client, task, environment))

    assert attempt.success is True
    prompt = client.requests[0].messages[-1]["content"]
    assert "PUBLIC CATALOG LOOKUP TOOL OUTPUT" in prompt
    assert "Begin with Candidate 1" in prompt
    assert "exact-title search suggestion: Synthetic cobalt travel bottle six-pack" in prompt
    assert "public selectable options: color=[amber, clear]; count=[pack of 6]" in prompt
    assert not hasattr(candidate, "goal")
    assert not hasattr(candidate, "reward")

    large_public_option_group = WebShopPublicCatalogCandidate(
        product_id="PUBLIC-ITEM-2",
        title="Configurable public cable",
        price=19.0,
        suggested_search_query="Configurable public cable",
        options=(("size", (*tuple(f"{index}ft" for index in range(95)), "180ft")),),
    )
    assert large_public_option_group.options[0][1][-1] == "180ft"


def test_public_catalog_mapping_preserves_every_router_field() -> None:
    candidate = WebShopPublicCatalogCandidate.from_public_mapping(
        {
            "attributes": ["portable", "portable"],
            "options": {
                "finish": ["matte", "gloss", "matte"],
                "size": ["small", "large"],
            },
            "price": 18.5,
            "product_id": "X000000099",
            "suggested_options": {"finish": "matte", "size": "large"},
            "suggested_search_query": "portable matte organizer",
            "title": "Portable organizer",
        }
    )

    assert candidate.suggested_search_query == "portable matte organizer"
    assert candidate.attributes == ("portable",)
    assert candidate.options == (
        ("finish", ("matte", "gloss")),
        ("size", ("small", "large")),
    )
    assert candidate.suggested_options == (("finish", "matte"), ("size", "large"))
