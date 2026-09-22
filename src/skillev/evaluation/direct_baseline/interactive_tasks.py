"""Benchmark-native sequential interaction under explicit environment contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from .client import (
    ChatTokenCounter,
    DirectGenerationClient,
    DirectGenerationError,
    DirectGenerationRequest,
)
from .config import DirectBenchmark, DirectDecodingProfile
from .parsing import (
    PARSER_REGISTRY,
    ParsedVisibleReAct,
    canonical_native_action,
    parse_visible_react,
)
from .prompts import (
    PublicInteractiveTurn,
    WebShopPublicCatalogCandidate,
    is_source_action_only_profile,
    is_structured_memory_profile,
    render_interactive_messages,
)


class InvalidCandidatePolicy(StrEnum):
    TERMINATE_ZERO = "terminate-zero"
    CONSUME_STEP_AND_CONTINUE = "consume-step-and-continue"


class InvalidEnvironmentActionPolicy(StrEnum):
    DELEGATE_TO_OFFICIAL_ENV = "delegate-to-official-env"
    TERMINATE_ZERO = "terminate-zero"


class InteractiveReasoningMode(StrEnum):
    HIDDEN_THINKING = "hidden-thinking"
    VISIBLE_REACT = "visible-react"
    ACTION_ONLY = "action-only"


def action_matches_public_surface(
    benchmark: DirectBenchmark,
    action: str,
    available_actions: tuple[str, ...],
) -> bool:
    """Return whether an action is represented by the public native surface.

    The official WebShop bridge exposes a parameterized search box as the
    capability token ``search``.  Concrete model actions necessarily carry a
    query (``search[...]``), while clickable actions are listed verbatim.
    """

    if action in available_actions:
        return True
    if benchmark is not DirectBenchmark.WEB_SHOP or "search" not in available_actions:
        return False
    return (
        action.startswith("search[")
        and action.endswith("]")
        and bool(action[len("search[") : -1].strip())
        and "[" not in action[len("search[") : -1]
        and "]" not in action[len("search[") : -1]
    )


@dataclass(frozen=True, slots=True)
class InteractiveProtocol:
    prompt_profile_id: str
    parser_profile_id: str
    max_steps: int
    invalid_candidate_policy: InvalidCandidatePolicy
    invalid_environment_action_policy: InvalidEnvironmentActionPolicy
    history_window_steps: int | None
    include_reasoning_in_history: bool = False
    reasoning_mode: InteractiveReasoningMode = InteractiveReasoningMode.ACTION_ONLY
    history_maximum_characters: int | None = 30_000

    def __post_init__(self) -> None:
        if not self.prompt_profile_id.strip() or not self.parser_profile_id.strip():
            raise ValueError("interactive prompt and parser profiles are required")
        if self.max_steps <= 0:
            raise ValueError("interactive max_steps must be positive")
        if self.history_window_steps is not None and self.history_window_steps <= 0:
            raise ValueError("history_window_steps must be positive or null")
        if self.history_maximum_characters is not None and self.history_maximum_characters <= 0:
            raise ValueError("history character budget must be positive")
        if self.reasoning_mode is InteractiveReasoningMode.HIDDEN_THINKING and (
            self.include_reasoning_in_history
        ):
            raise ValueError("hidden reasoning cannot enter public history")


@dataclass(frozen=True, slots=True)
class NativeEnvironmentStep:
    observation: str
    terminal: bool
    reward: float
    success: bool
    action_valid: bool | None
    invalid_reason: str | None = None
    available_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation.strip():
            raise ValueError("native environment observation must be non-empty")
        if not 0 <= self.reward <= 1:
            raise ValueError("native environment reward must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class NativeEnvironmentOutcome:
    reward: float
    success: bool
    terminal_reached: bool
    terminated_by_horizon: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.reward <= 1:
            raise ValueError("native environment reward must lie in [0, 1]")


class NativeInteractiveEnvironment(Protocol):
    async def reset(self) -> str | NativePublicState: ...
    async def step(self, action: str) -> NativeEnvironmentStep: ...
    async def outcome(self) -> NativeEnvironmentOutcome: ...
    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class NativeInteractiveTask:
    task_id: str
    benchmark: DirectBenchmark
    task: str
    profile: DirectDecodingProfile
    max_steps: int
    prompt_profile_id: str
    parser_profile_id: str
    population_id: str
    run_seed: int
    invalid_candidate_policy: InvalidCandidatePolicy = InvalidCandidatePolicy.TERMINATE_ZERO
    invalid_environment_action_policy: InvalidEnvironmentActionPolicy = (
        InvalidEnvironmentActionPolicy.DELEGATE_TO_OFFICIAL_ENV
    )
    history_window_steps: int | None = None
    include_reasoning_in_history: bool = False
    reasoning_mode: InteractiveReasoningMode = InteractiveReasoningMode.ACTION_ONLY
    history_maximum_characters: int | None = 30_000
    demonstrations: tuple[str, ...] = ()
    panel_index: int = 0
    service_slot: int = 0
    history_token_counter: ChatTokenCounter | None = None
    context_length: int | None = None
    task_type: str | None = None
    webshop_catalog_candidates: tuple[WebShopPublicCatalogCandidate, ...] = ()

    def __post_init__(self) -> None:
        if self.benchmark not in {
            DirectBenchmark.WEB_SHOP,
            DirectBenchmark.ALF_WORLD,
            DirectBenchmark.SCIENCE_WORLD,
            DirectBenchmark.MIND2WEB,
        }:
            raise ValueError("task does not use a native interactive benchmark")
        if not self.task_id.strip() or not self.task.strip() or self.max_steps <= 0:
            raise ValueError("interactive task identity, text, and positive horizon are required")
        if self.run_seed != self.profile.seed:
            raise ValueError("task run_seed must equal request profile seed")
        if self.history_maximum_characters is not None and self.history_maximum_characters <= 0:
            raise ValueError("history character budget must be positive")
        if self.reasoning_mode is InteractiveReasoningMode.HIDDEN_THINKING and (
            self.include_reasoning_in_history
        ):
            raise ValueError("hidden reasoning cannot enter public history")
        if type(self.panel_index) is not int or self.panel_index < 0:
            raise ValueError("interactive panel index must be non-negative")
        if type(self.service_slot) is not int or self.service_slot < 0:
            raise ValueError("interactive service slot must be non-negative")
        if (self.history_token_counter is None) != (self.context_length is None):
            raise ValueError("interactive token budget identity is incomplete")
        if self.context_length is not None and self.context_length <= 0:
            raise ValueError("interactive context length must be positive")
        if self.task_type is not None and (
            self.benchmark is not DirectBenchmark.ALF_WORLD or not self.task_type.strip()
        ):
            raise ValueError("task_type is supported only for ALFWorld and must be non-empty")
        if self.webshop_catalog_candidates and self.benchmark is not DirectBenchmark.WEB_SHOP:
            raise ValueError("public catalog candidates are supported only for WebShop")
        if len(self.webshop_catalog_candidates) > 8 or any(
            not isinstance(item, WebShopPublicCatalogCandidate)
            for item in self.webshop_catalog_candidates
        ):
            raise ValueError("WebShop public catalog context requires at most eight typed items")


@dataclass(frozen=True, slots=True)
class NativePublicState:
    observation_text: str
    available_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.observation_text.strip():
            raise ValueError("native public observation must be non-empty")

    def render(self) -> str:
        if not self.available_actions:
            return self.observation_text
        actions = "\n".join(f"- {item}" for item in self.available_actions)
        return f"{self.observation_text}\n\nAdmissible actions:\n{actions}"


@dataclass(frozen=True, slots=True)
class InteractiveStepRecordV2:
    step_index: int
    observation_before: str
    available_actions_before: tuple[str, ...]
    raw_text: str
    reasoning_text: str | None
    parsed_action: str | None
    parse_reason: str
    action_listed_before: bool | None
    official_action_valid: bool | None
    observation_after: str | None
    available_actions_after: tuple[str, ...]
    native_reward_after: float | None
    official_terminal_after: bool
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    response_model: str | None = None
    response_id: str | None = None
    service_instance_id: str | None = None
    service_slot: int = 0
    structured_memory_before: str | None = None
    structured_memory_after: str | None = None
    visible_thought: str | None = None
    memory_update_status: str = "not-applicable"
    repeated_action: bool = False
    unchanged_state: bool = False
    revisited_product_or_location: bool = False
    purchase_with_unmet_constraints: bool = False

    @property
    def observation(self) -> str:
        return self.observation_after or self.observation_before

    @property
    def action_valid(self) -> bool | None:
        return self.official_action_valid

    @property
    def reward(self) -> float | None:
        return self.native_reward_after

    @property
    def terminal(self) -> bool:
        return self.official_terminal_after


InteractiveStepRecord = InteractiveStepRecordV2


@dataclass(frozen=True, slots=True)
class NativeInteractiveAttempt:
    task_id: str
    benchmark: DirectBenchmark
    reward: float | None
    success: bool | None
    steps: int
    valid_actions: int
    invalid_actions: int
    submission_produced: bool
    terminal_reached: bool
    infrastructure_error: str | None
    terminated_by_horizon: bool = False
    cleanup_error: str | None = None
    termination_reason: str | None = None
    trace: tuple[InteractiveStepRecordV2, ...] = ()
    budget_exhausted: bool = False
    terminal_success: bool | None = None
    unfinished_subgoal_at_horizon: str | None = None


async def run_native_interactive_task(
    client: DirectGenerationClient,
    task: NativeInteractiveTask,
    environment: NativeInteractiveEnvironment,
) -> NativeInteractiveAttempt:
    attempt = await _run_without_cleanup(client, task, environment)
    finish = getattr(client, "finish_interactive_episode", None)
    if callable(finish):
        try:
            await finish(
                task.task_id,
                NativeEnvironmentOutcome(
                    reward=attempt.reward or 0.0,
                    success=bool(attempt.success),
                    terminal_reached=attempt.terminal_reached,
                    terminated_by_horizon=attempt.terminated_by_horizon,
                ),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            attempt = replace(attempt, cleanup_error=type(exc).__name__)
    try:
        await environment.close()
    except (OSError, RuntimeError) as exc:
        return replace(attempt, cleanup_error=type(exc).__name__)
    return attempt


async def _run_without_cleanup(
    client: DirectGenerationClient,
    task: NativeInteractiveTask,
    environment: NativeInteractiveEnvironment,
) -> NativeInteractiveAttempt:
    history: list[PublicInteractiveTurn] = []
    trace: list[InteractiveStepRecordV2] = []
    valid_actions = invalid_actions = 0
    submission_produced = False
    structured_memory: str | None = None
    visited_entities: set[str] = set()
    try:
        reset = await environment.reset()
        state = reset if isinstance(reset, NativePublicState) else NativePublicState(reset)
        task = task_with_authoritative_reset_instruction(task, state)
        begin = getattr(client, "begin_interactive_episode", None)
        if callable(begin):
            await begin(task, state)
        parser = PARSER_REGISTRY[task.parser_profile_id]
        for step_index in range(1, task.max_steps + 1):
            source_action_only = is_source_action_only_profile(task.prompt_profile_id)
            structured = is_structured_memory_profile(task.prompt_profile_id)
            live_memory = _current_client_interactive_memory(client, task.task_id)
            if structured and live_memory is not None:
                structured_memory = live_memory
            try:
                generation = await client.generate(
                    DirectGenerationRequest(
                        request_id=f"{task.task_id}:step:{step_index}",
                        messages=render_interactive_messages(
                            task.prompt_profile_id,
                            task=task.task,
                            current_observation=(
                                state.observation_text
                                if source_action_only or structured
                                else state.render()
                            ),
                            history=tuple(history),
                            history_window_steps=task.history_window_steps,
                            history_maximum_characters=task.history_maximum_characters,
                            demonstrations=task.demonstrations,
                            structured_memory=structured_memory,
                            remaining_steps=task.max_steps - step_index + 1,
                            available_actions=state.available_actions,
                            history_token_counter=task.history_token_counter,
                            context_length=task.context_length,
                            output_reserve=(
                                task.profile.max_new_tokens
                                if task.history_token_counter is not None
                                else None
                            ),
                            enable_thinking=(
                                task.profile.enable_thinking
                                if task.history_token_counter is not None
                                else None
                            ),
                            task_type=task.task_type,
                            webshop_catalog_candidates=task.webshop_catalog_candidates,
                        ),
                        profile=task.profile,
                        service_slot=task.service_slot,
                    )
                )
            except DirectGenerationError as exc:
                return _infra(task, step_index - 1, valid_actions, invalid_actions, exc, trace)
            visible_react: ParsedVisibleReAct | None = (
                parse_visible_react(generation.text) if structured else None
            )
            parsed = visible_react.action if visible_react is not None else parser(generation.text)
            if task.benchmark is DirectBenchmark.WEB_SHOP and parsed.value is not None:
                parsed = replace(parsed, value=canonical_native_action(parsed.value))
            memory_before = structured_memory
            memory_after = (
                visible_react.memory
                if visible_react is not None and visible_react.memory
                else memory_before
            )
            memory_status = (
                "not-applicable"
                if not structured
                else "updated"
                if visible_react is not None and visible_react.memory is not None
                else "missing-carried"
                if memory_before is not None
                else "missing-empty"
            )
            visible_thought = visible_react.thought if visible_react is not None else None
            if parsed.value is None:
                invalid_actions += 1
                returned_state = NativePublicState(
                    f"{state.observation_text}\n\nEnvironment feedback: The prior response did "
                    "not contain one valid native action. Choose one action under the same "
                    "public contract.",
                    state.available_actions,
                )
                observe = getattr(client, "observe_interactive_episode", None)
                if callable(observe):
                    await observe(task.task_id, None, returned_state)
                live_memory_after = _current_client_interactive_memory(client, task.task_id)
                if structured and live_memory_after is not None:
                    memory_after = live_memory_after
                    memory_status = "controller-observed"
                trace.append(
                    InteractiveStepRecordV2(
                        step_index=step_index,
                        observation_before=state.observation_text,
                        available_actions_before=state.available_actions,
                        raw_text=generation.text,
                        reasoning_text=generation.reasoning_text,
                        parsed_action=None,
                        parse_reason=parsed.reason.value,
                        action_listed_before=None,
                        official_action_valid=False,
                        observation_after=None,
                        available_actions_after=(),
                        native_reward_after=0.0,
                        official_terminal_after=(
                            task.invalid_candidate_policy is InvalidCandidatePolicy.TERMINATE_ZERO
                        ),
                        finish_reason=generation.finish_reason,
                        prompt_tokens=generation.prompt_tokens,
                        completion_tokens=generation.completion_tokens,
                        response_model=generation.response_model,
                        response_id=generation.response_id,
                        service_instance_id=generation.service_instance_id,
                        service_slot=task.service_slot,
                        structured_memory_before=memory_before,
                        structured_memory_after=memory_after,
                        visible_thought=visible_thought,
                        memory_update_status=memory_status,
                    )
                )
                if task.invalid_candidate_policy is InvalidCandidatePolicy.TERMINATE_ZERO:
                    return _candidate_attempt(
                        task, step_index, valid_actions, invalid_actions, submission_produced, trace
                    )
                history.append(
                    PublicInteractiveTurn(
                        "Action: <candidate response was invalid>",
                        None,
                        state.observation_text,
                        returned_state.observation_text,
                        returned_state.available_actions,
                        state.available_actions,
                        memory_before,
                        memory_after,
                        visible_thought,
                        memory_status,
                    )
                )
                structured_memory = memory_after
                state = returned_state
                continue
            submission_produced = True
            prior_state = state
            action_listed_before = (
                action_matches_public_surface(
                    task.benchmark,
                    parsed.value,
                    prior_state.available_actions,
                )
                if prior_state.available_actions
                else None
            )
            result = await environment.step(parsed.value)
            next_observation = _environment_observation_for_next_turn(result)
            observe = getattr(client, "observe_interactive_episode", None)
            if callable(observe):
                await observe(task.task_id, parsed.value, result)
            live_memory_after = _current_client_interactive_memory(client, task.task_id)
            if structured and live_memory_after is not None:
                memory_after = live_memory_after
                memory_status = "controller-observed"
            repeated_action = any(
                row.parsed_action == parsed.value
                and _same_public_observation(
                    row.observation_before,
                    prior_state.observation_text,
                )
                for row in trace
            )
            unchanged_state = _same_public_observation(
                prior_state.observation_text,
                result.observation,
            )
            entity = _visited_entity(task.benchmark, parsed.value, result.observation)
            revisited = entity is not None and entity in visited_entities
            if entity is not None:
                visited_entities.add(entity)
            purchase_with_unmet = (
                task.benchmark is DirectBenchmark.WEB_SHOP
                and parsed.value.casefold() == "click[buy now]"
                and _memory_reports_unmet_requirements(memory_after)
            )
            diagnostic_action_valid = (
                result.action_valid if result.action_valid is not None else action_listed_before
            )
            if diagnostic_action_valid is True:
                valid_actions += 1
            elif diagnostic_action_valid is False:
                invalid_actions += 1
            trace.append(
                InteractiveStepRecordV2(
                    step_index=step_index,
                    observation_before=prior_state.observation_text,
                    available_actions_before=prior_state.available_actions,
                    raw_text=generation.text,
                    reasoning_text=generation.reasoning_text,
                    parsed_action=parsed.value,
                    parse_reason=parsed.reason.value,
                    action_listed_before=action_listed_before,
                    official_action_valid=result.action_valid,
                    observation_after=result.observation,
                    available_actions_after=result.available_actions,
                    native_reward_after=result.reward,
                    official_terminal_after=result.terminal,
                    finish_reason=generation.finish_reason,
                    prompt_tokens=generation.prompt_tokens,
                    completion_tokens=generation.completion_tokens,
                    response_model=generation.response_model,
                    response_id=generation.response_id,
                    service_instance_id=generation.service_instance_id,
                    service_slot=task.service_slot,
                    structured_memory_before=memory_before,
                    structured_memory_after=memory_after,
                    visible_thought=visible_thought,
                    memory_update_status=memory_status,
                    repeated_action=repeated_action,
                    unchanged_state=unchanged_state,
                    revisited_product_or_location=revisited,
                    purchase_with_unmet_constraints=purchase_with_unmet,
                )
            )
            if (
                diagnostic_action_valid is False
                and task.invalid_environment_action_policy
                is InvalidEnvironmentActionPolicy.TERMINATE_ZERO
            ):
                return _candidate_attempt(
                    task, step_index, valid_actions, invalid_actions, submission_produced, trace
                )
            history.append(
                PublicInteractiveTurn(
                    (
                        generation.text
                        if task.reasoning_mode is InteractiveReasoningMode.VISIBLE_REACT
                        else parsed.value
                        if source_action_only
                        else f"Action: {parsed.value}"
                    ),
                    parsed.value,
                    prior_state.observation_text,
                    next_observation,
                    result.available_actions,
                    prior_state.available_actions,
                    memory_before,
                    memory_after,
                    visible_thought,
                    memory_status,
                )
            )
            structured_memory = memory_after
            state = NativePublicState(
                next_observation,
                result.available_actions,
            )
            if result.terminal:
                return NativeInteractiveAttempt(
                    task.task_id,
                    task.benchmark,
                    result.reward,
                    result.success,
                    step_index,
                    valid_actions,
                    invalid_actions,
                    submission_produced,
                    True,
                    None,
                    termination_reason="official-terminal",
                    trace=tuple(trace),
                    budget_exhausted=(step_index == task.max_steps and not result.success),
                    terminal_success=result.success,
                    unfinished_subgoal_at_horizon=(
                        _remaining_subgoals(memory_after)
                        if step_index == task.max_steps and not result.success
                        else None
                    ),
                )
        outcome = await environment.outcome()
        return NativeInteractiveAttempt(
            task.task_id,
            task.benchmark,
            outcome.reward,
            outcome.success,
            task.max_steps,
            valid_actions,
            invalid_actions,
            submission_produced,
            outcome.terminal_reached,
            None,
            terminated_by_horizon=not outcome.terminal_reached,
            termination_reason=("horizon" if not outcome.terminal_reached else "official-terminal"),
            trace=tuple(trace),
            budget_exhausted=(not outcome.success and task.max_steps == len(trace)),
            terminal_success=(outcome.success if outcome.terminal_reached else None),
            unfinished_subgoal_at_horizon=(
                _remaining_subgoals(structured_memory) if not outcome.success else None
            ),
        )
    except (OSError, RuntimeError) as exc:
        return _infra(task, len(trace), valid_actions, invalid_actions, exc, trace)


_ALFWORLD_TASK_LINE = re.compile(r"(?im)^\s*Your task is to:\s*(\S.*)\s*$")


def task_with_authoritative_reset_instruction(
    task: NativeInteractiveTask,
    state: NativePublicState,
) -> NativeInteractiveTask:
    """Use the task emitted by the pinned ALFWorld reset as model input.

    Released ALFWorld games can contain several crowd annotations, including
    annotations whose object does not match the game identity.  The simulator
    selects one annotation at reset time and prints it in the public initial
    observation.  A separately frozen manifest annotation is therefore not an
    authoritative substitute and previously caused the controller to pursue a
    different object than the live environment requested.
    """

    if task.benchmark is not DirectBenchmark.ALF_WORLD:
        return task
    matches = tuple(_ALFWORLD_TASK_LINE.findall(state.observation_text))
    if len(matches) != 1:
        raise RuntimeError("ALFWorld reset did not expose one authoritative task line")
    instruction = " ".join(matches[0].split())
    return replace(task, task=instruction)


def _current_client_interactive_memory(
    client: DirectGenerationClient,
    task_id: str,
) -> str | None:
    """Read optional client-owned state after its latest environment observation.

    Direct baselines keep carrying model-authored memory in this runner. A
    persistent architecture client observes the transition after that response
    and therefore owns newer memory than the pre-action Memory it emitted.
    """

    getter = getattr(client, "current_interactive_memory", None)
    if not callable(getter):
        return None
    memory = getter(task_id)
    if not isinstance(memory, str) or not memory.strip():
        raise ValueError("interactive client memory must be a non-empty string")
    return memory


_NONTERMINAL_STATUS = (
    "Environment status: NOT TERMINAL. The current task is still incomplete; continue with "
    "a different current admissible action when the last action made no progress."
)


def _environment_observation_for_next_turn(result: NativeEnvironmentStep) -> str:
    if result.terminal:
        return result.observation
    return f"{result.observation}\n\n{_NONTERMINAL_STATUS}"


def _same_public_observation(before: str, after: str) -> bool:
    normalized_before = before.removesuffix(f"\n\n{_NONTERMINAL_STATUS}")
    return " ".join(normalized_before.split()).casefold() == " ".join(after.split()).casefold()


def _visited_entity(
    benchmark: DirectBenchmark,
    action: str,
    returned_observation: str,
) -> str | None:
    normalized = action.strip().casefold()
    if benchmark is DirectBenchmark.ALF_WORLD and normalized.startswith("go to "):
        return f"location:{normalized.removeprefix('go to ').strip()}"
    if benchmark is DirectBenchmark.WEB_SHOP and normalized.startswith("click["):
        if any(
            marker in returned_observation.casefold()
            for marker in ("price:", "buy now", "description")
        ):
            return f"product:{normalized}"
    return None


def _memory_reports_unmet_requirements(memory: str | None) -> bool:
    if memory is None:
        return False
    lowered = memory.casefold()
    if "ready to purchase:" in lowered:
        ready = lowered.split("ready to purchase:", 1)[1].splitlines()[0].strip()
        if ready.startswith("no"):
            return True
    if "unmet requirements:" not in lowered:
        return False
    value = lowered.split("unmet requirements:", 1)[1].split("ready to purchase:", 1)[0]
    normalized = value.strip(" \n\t-:.;")
    return bool(normalized and normalized not in {"none", "n/a", "no unmet requirements"})


def _remaining_subgoals(memory: str | None) -> str | None:
    if memory is None or "remaining subgoals:" not in memory.casefold():
        return None
    start = memory.casefold().index("remaining subgoals:") + len("remaining subgoals:")
    value = memory[start:]
    marker = value.casefold().find("repeated/no-progress actions:")
    if marker >= 0:
        value = value[:marker]
    normalized = value.strip()
    return normalized or None


def _candidate_attempt(
    task: NativeInteractiveTask,
    steps: int,
    valid_actions: int,
    invalid_actions: int,
    submission_produced: bool,
    trace: list[InteractiveStepRecordV2],
) -> NativeInteractiveAttempt:
    return NativeInteractiveAttempt(
        task.task_id,
        task.benchmark,
        0.0,
        False,
        steps,
        valid_actions,
        invalid_actions,
        submission_produced,
        False,
        None,
        termination_reason="candidate-invalid",
        trace=tuple(trace),
    )


def _infra(
    task: NativeInteractiveTask,
    steps: int,
    valid_actions: int,
    invalid_actions: int,
    error: BaseException,
    trace: list[InteractiveStepRecordV2],
) -> NativeInteractiveAttempt:
    return NativeInteractiveAttempt(
        task.task_id,
        task.benchmark,
        None,
        None,
        steps,
        valid_actions,
        invalid_actions,
        False,
        False,
        type(error).__name__,
        termination_reason="infrastructure",
        trace=tuple(trace),
    )
