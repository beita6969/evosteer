"""Frozen tool-using nodes over one shared public task environment.

This executor never owns a trusted evaluator. COMPLETE closes only the current
role invocation. Environment mutations persist across nodes and reruns. Local
reserve/settle accounting bounds the whole invocation; GraphRuntime charges
that aggregate once on the team ledger, avoiding a second global debit.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Protocol, cast

from skillev.contracts.canonical import JsonValue, canonical_json, normalize_json, stable_hash
from skillev.rollout.action_contract import ActionContract, completion_example
from skillev.rollout.action_surface import (
    ActionSurface,
    CompletionSpec,
    TerminalMode,
)
from skillev.rollout.codec import StructuredJsonActionCodec
from skillev.runtime.bounded_agent import (
    EnvironmentMethodFailedError,
    EnvironmentMethodTimeoutError,
    EnvironmentSkillInvocationMismatchError,
)
from skillev.runtime.budget_ledger import BudgetLedger
from skillev.runtime.contracts import (
    ActionKind,
    BudgetReservation,
    BudgetSettlement,
    BudgetVector,
    StructuredAction,
)
from skillev.runtime.execution import EnvironmentObservation, RolloutEnvironmentSession

from .evosteer_features import answer_present
from .graph import NodeExecutionRequest, NodeExecutionResult

FORMAT = "evosteer-frozen-tool-executor@2"
ToolPermission = tuple[str, str]


class FrozenTextPolicy(Protocol):
    @property
    def reference_id(self) -> str: ...

    def frozen_text(
        self,
        text: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        input_limit: int | None = None,
    ) -> tuple[str, int, int] | Awaitable[tuple[str, int, int]]: ...


class ToolExecutorPoisonedError(RuntimeError):
    """An external call failed with potentially unknown effects or resource use."""


def _public_object(value: object) -> dict[str, JsonValue]:
    result = normalize_json(value)
    if not isinstance(result, dict):
        raise TypeError("public execution record must be a JSON object")
    return result


def _node_completion(value: JsonValue) -> bool:
    # Empty model output is legitimate; this is shape validation, never grading.
    return isinstance(value, dict) and set(value) == {"answer"} and type(value["answer"]) is str


def _project_completion(value: JsonValue) -> str:
    return cast(str, cast(dict[str, JsonValue], value)["answer"])


def _public_observation(observation: EnvironmentObservation) -> dict[str, JsonValue]:
    # Submission payload is for final evaluator routing, not model observation.
    return {
        key: value for key, value in observation.to_value().items() if key != "terminal_submission"
    }


class FrozenToolExecutor:
    """A frozen local agent using existing JSON action and environment contracts.

    Permission entries are exact (resource_id, action_name) pairs. Every role
    must be explicitly configured, including roles allowed no task tools. Bound
    skill reads additionally require ("skill-runtime", "invoke"). Unknown or
    forbidden actions remain public model failures and never reach the world.

    The synchronous production policy runs on the calling thread so adapter
    selection cannot race a concurrent training/scoring operation. The enclosing
    application must keep all operations on its shared model serialized.
    """

    def __init__(
        self,
        policy: FrozenTextPolicy,
        environment: RolloutEnvironmentSession,
        action_surface: ActionSurface,
        *,
        role_permissions: Mapping[str, tuple[ToolPermission, ...]],
        max_turns: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        tool_call_maximum: BudgetVector | None = None,
        initial_observation: JsonValue = None,
        completion_projector: Callable[[JsonValue], str] | None = None,
        completion_projector_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        environment_step_offset: int = 0,
        phase_steps: int | None = 8,
        max_phases: int | None = 12,
    ) -> None:
        if not isinstance(action_surface, ActionSurface):
            raise TypeError("action_surface must be the existing public ActionSurface")
        if max_turns is not None and (type(max_turns) is not int or max_turns < 1):
            raise ValueError("debug max_turns must be positive or None")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if type(environment_step_offset) is not int or environment_step_offset < 0:
            raise ValueError("environment_step_offset must be nonnegative")
        if (phase_steps is None) != (max_phases is None) or any(
            value is not None and (type(value) is not int or value < 1)
            for value in (phase_steps, max_phases)
        ):
            raise ValueError("phase_steps/max_phases must both be positive integers or both None")
        if (
            phase_steps is not None
            and max_phases is not None
            and (environment_step_offset > phase_steps * max_phases)
        ):
            raise ValueError("environment step offset exceeds the interactive phase budget")
        if completion_projector is not None and (
            type(completion_projector_id) is not str or not completion_projector_id.strip()
        ):
            raise ValueError("a custom completion projector requires a pinned identity")
        if completion_projector is None and completion_projector_id is not None:
            raise ValueError("completion projector identity supplied without its implementation")
        if type(policy.reference_id) is not str or not policy.reference_id:
            raise ValueError("frozen policy requires its immutable reference identity")
        if not environment.environment_id or not environment.task_family:
            raise ValueError("a public environment identity is required")
        maximum = tool_call_maximum or BudgetVector(
            tool_calls=1,
            wall_time_milliseconds=30_000,
        )
        if not isinstance(maximum, BudgetVector) or maximum.tool_calls != 1:
            raise ValueError("a single environment dispatch must reserve exactly one tool call")
        known = {(tool.resource_id, tool.name) for tool in action_surface.tools}
        known.add(("skill-runtime", "invoke"))
        permissions: dict[str, tuple[ToolPermission, ...]] = {}
        for role_id, entries in role_permissions.items():
            if type(role_id) is not str or not role_id.strip() or not isinstance(entries, tuple):
                raise ValueError("role permissions require nonempty role IDs and immutable tuples")
            if any(
                not isinstance(item, tuple)
                or len(item) != 2
                or any(type(part) is not str or not part for part in item)
                for item in entries
            ):
                raise ValueError("permissions must contain exact resource/action pairs")
            if len(set(entries)) != len(entries) or not set(entries) <= known:
                raise ValueError("role permissions repeat or reference unavailable tools")
            permissions[role_id] = tuple(sorted(entries))
        if not permissions:
            raise ValueError("at least one explicitly configured role is required")
        self._policy = policy
        self._environment = environment
        self._surface = ActionSurface.from_value(action_surface.to_value())
        self._permissions = dict(sorted(permissions.items()))
        self._max_turns = max_turns
        self._max_new_tokens = max_new_tokens
        self._temperature = float(temperature)
        self._tool_maximum = maximum
        self._initial_observation = normalize_json(initial_observation)
        self._project = completion_projector or _project_completion
        self._clock = clock
        self._step_index = environment_step_offset
        self._observed_steps = environment_step_offset
        self._phase_steps = phase_steps
        self._max_phases = max_phases
        self._last_outcome: dict[str, JsonValue] | None = None
        self._terminal = False
        self._history: list[dict[str, JsonValue]] = []
        self._invocations: set[tuple[str, int]] = set()
        self._runtime_id: str | None = None
        self._poisoned = False
        self._lock = asyncio.Lock()
        self._configuration_id = stable_hash(
            {
                "format": FORMAT,
                "surface": self._surface.to_value(),
                "role_permissions": {
                    key: [list(item) for item in value] for key, value in self._permissions.items()
                },
                "max_turns": max_turns,
                "max_new_tokens": max_new_tokens,
                "temperature": self._temperature,
                "tool_call_maximum": maximum.to_value(),
                "environment_id": environment.environment_id,
                "task_family": environment.task_family,
                "completion": "local-answer-string@1",
                "completion_projector": completion_projector_id or "answer-field@1",
                "phase_steps": phase_steps,
                "max_phases": max_phases,
            }
        )
        self._pinned_identity = self.frozen_identity

    @property
    def frozen_identity(self) -> str:
        return f"{self._policy.reference_id}/{self._configuration_id}"

    @property
    def environment_step_index(self) -> int:
        return self._step_index

    @property
    def environment_terminal(self) -> bool:
        return self._terminal

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def phase_budget_exhausted(self) -> bool:
        return (
            self._phase_steps is not None
            and self._max_phases is not None
            and self._observed_steps >= self._phase_steps * self._max_phases
        )

    def public_environment_state(self) -> dict[str, JsonValue]:
        """Observed dispatch progress, separate from task grading and native termination.

        One phase step is one completed public environment dispatch, including a
        measured error/timeout. Skill document reads and rejected local actions
        do not advance the world. Phase index is zero-based; at exhaustion it
        equals max_phases. The 8/12 protocol is an explicit runtime convention
        for the paper's stated phase lengths, not a benchmark-specific reset API.
        """
        enabled = self._phase_steps is not None and self._max_phases is not None
        phase_index = self._observed_steps // self._phase_steps if self._phase_steps else 0
        step_cap = (
            self._phase_steps * self._max_phases
            if self._phase_steps is not None and self._max_phases is not None
            else None
        )
        return _public_object(
            {
                "enabled": enabled,
                "steps_per_phase": self._phase_steps,
                "max_phases": self._max_phases,
                "steps_completed": self._observed_steps,
                "phase_index": phase_index,
                "phases_completed": phase_index,
                "steps_in_phase": self._observed_steps % self._phase_steps
                if self._phase_steps
                else 0,
                "remaining_steps": None if step_cap is None else step_cap - self._observed_steps,
                "dispatches_started": self._step_index,
                "poisoned": self._poisoned,
                "phase_budget_exhausted": self.phase_budget_exhausted,
                "environment_terminal": self._terminal,
                "last_outcome": self._last_outcome,
            }
        )

    @property
    def public_environment_history(self) -> tuple[dict[str, JsonValue], ...]:
        return tuple(_public_object(event) for event in self._history)

    def _contract(self, request: NodeExecutionRequest) -> ActionContract:
        permissions = self._permissions[request.role.role_id]
        skill_ids = tuple(sorted(skill_id for skill_id, _ in request.skills))
        # Node COMPLETE deliberately overrides a task-level environment-terminal
        # interface. It submits a local answer only and never ends/grades the team.
        local_surface = replace(
            self._surface,
            terminal_mode=TerminalMode.EXPLICIT_COMPLETION,
            completion=CompletionSpec({"answer": "string"}, {"answer": "Local result"}),
            tools=tuple(
                tool
                for tool in self._surface.tools
                if (tool.resource_id, tool.name) in permissions
                and not self._terminal
                and not self.phase_budget_exhausted
            ),
        )
        readable = skill_ids if ("skill-runtime", "invoke") in permissions else ()
        return ActionContract.freeze(
            local_surface,
            retrieved_skill_ids=readable,
            active_skill_ids=readable,
        )

    def _prompt(
        self,
        request: NodeExecutionRequest,
        contract: ActionContract,
        turns: list[dict[str, JsonValue]],
        ledger: BudgetLedger,
    ) -> str:
        prompt: str = canonical_json(
            {
                "instruction": (
                    "Execute this node role using its allowed public tools. Return exactly one "
                    "structured JSON action at a time, with no separate commentary. COMPLETE ends "
                    "only this node invocation; it does not grade or submit the team's task. "
                    "Tool effects persist in the shared environment. A skill supplies advice only."
                ),
                "role": request.role.instruction,
                "task": request.task_prompt,
                "bound_skills": [{"skill_id": key, "body": body} for key, body in request.skills],
                "inbound_messages": list(request.messages),
                "previous_node_output": request.previous_output,
                "initial_environment_observation": self._initial_observation,
                "shared_environment_history": self._history,
                "environment_terminal": self._terminal,
                "environment": self.public_environment_state(),
                "node_turns": turns,
                "action_instructions": list(contract.render_public_instruction()),
                "skill_read": (
                    {
                        "kind": "skill",
                        "name": "invoke",
                        "resource_id": "skill-runtime",
                        "skill_id": contract.retrieved_skill_ids[0],
                        "arguments": {},
                    }
                    if contract.retrieved_skill_ids
                    else None
                ),
                "local_completion_example": completion_example({"answer": "Local result"}),
                "remaining_invocation_budget": ledger.available.to_value(),
            }
        )
        return prompt

    @staticmethod
    def _reserve(
        ledger: BudgetLedger,
        invocation: str,
        label: str,
        maximum: BudgetVector,
    ) -> str:
        identifier = f"{invocation}:{label}"
        ledger.reserve(
            BudgetReservation(
                identifier,
                ledger.run_id,
                ledger.attempt_id,
                invocation,
                maximum,
            )
        )
        return identifier

    async def _generate(
        self,
        prompt: str,
        request: NodeExecutionRequest,
        turn: int,
        ledger: BudgetLedger,
        invocation: str,
    ) -> tuple[str, BudgetVector]:
        remaining = ledger.available
        maximum = BudgetVector(
            input_tokens=remaining.input_tokens,
            output_tokens=min(self._max_new_tokens, remaining.output_tokens),
            model_calls=1,
            agent_turns=1,
            wall_time_milliseconds=remaining.wall_time_milliseconds,
        )
        reservation_id = self._reserve(ledger, invocation, f"model:{turn}", maximum)
        start = self._clock()
        generated = self._policy.frozen_text(
            prompt,
            max_new_tokens=maximum.output_tokens,
            temperature=self._temperature,
            seed=int(
                stable_hash({"node_seed": request.seed, "turn": turn}).split(":", 1)[-1][:16], 16
            ),
            input_limit=maximum.input_tokens,
        )
        response = await generated if inspect.isawaitable(generated) else generated
        elapsed = self._clock() - start
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("model timing source moved backward or became nonfinite")
        if not isinstance(response, tuple) or len(response) != 3:
            raise TypeError("frozen policy must return text and exact input/output token counts")
        text, input_count, output_count = response
        if type(text) is not str or type(input_count) is not int or type(output_count) is not int:
            raise TypeError("invalid frozen policy response fields")
        if input_count < 1 or output_count < 0 or (text and output_count == 0):
            raise ValueError("frozen policy token usage is invalid")
        usage = BudgetVector(
            input_tokens=input_count,
            output_tokens=output_count,
            model_calls=1,
            agent_turns=1,
            wall_time_milliseconds=math.ceil(elapsed * 1000),
        )
        ledger.settle(BudgetSettlement(reservation_id, usage))
        if self.frozen_identity != self._pinned_identity:
            raise RuntimeError("frozen policy identity changed during a node call")
        return text, usage

    async def _dispatch(
        self,
        action: StructuredAction,
        expected_skills: tuple[str, ...],
        request: NodeExecutionRequest,
        turn: int,
        ledger: BudgetLedger,
        invocation: str,
    ) -> EnvironmentObservation:
        reservation_id = self._reserve(ledger, invocation, f"tool:{turn}", self._tool_maximum)
        if action.kind is ActionKind.SKILL:
            body = dict(request.skills)[cast(str, action.skill_id)]
            # A real deterministic document-read operation, not an environment step.
            observation = EnvironmentObservation(
                {
                    "status": "skill-read",
                    "skill_id": action.skill_id,
                    "content": body,
                    "environment_action_executed": False,
                },
                "success",
                expected_skills,
                budget_usage=BudgetVector(tool_calls=1),
            )
        else:
            if self.phase_budget_exhausted or self._terminal:
                raise RuntimeError("environment dispatch attempted after its execution boundary")
            self._step_index += 1
            try:
                observation = await self._environment.execute(action, step_index=self._step_index)
            except EnvironmentMethodTimeoutError as error:
                observation = EnvironmentObservation(
                    {"error": error.public_error_code},
                    "timeout",
                    invoked_skill_ids=expected_skills,
                    budget_usage=error.budget_usage,
                )
            except EnvironmentMethodFailedError as error:
                observation = EnvironmentObservation(
                    {"error": error.public_error_code},
                    "tool_error",
                    invoked_skill_ids=expected_skills,
                    budget_usage=error.budget_usage,
                )
            if not isinstance(observation, EnvironmentObservation):
                raise TypeError("environment returned an incompatible public observation")
        if observation.invoked_skill_ids != expected_skills:
            raise EnvironmentSkillInvocationMismatchError("environment invented skill credit")
        if observation.budget_usage.tool_calls != 1:
            raise ValueError("one environment dispatch must report exactly one tool call")
        ledger.settle(BudgetSettlement(reservation_id, observation.budget_usage))
        if action.kind is not ActionKind.SKILL:
            self._observed_steps += 1
            self._terminal = self._terminal or observation.terminal
            self._last_outcome = _public_object(
                {
                    "status": observation.observation_status,
                    "terminal": observation.terminal,
                    "public_value": observation.public_value,
                }
            )
            self._history.append(
                _public_object(
                    {
                        "environment_step_index": self._step_index,
                        "node_id": request.node_id,
                        "execution_index": request.execution_index,
                        "action": action.to_value(),
                        "observation": _public_observation(observation),
                    }
                )
            )
        return observation

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        async with self._lock:
            if self._poisoned:
                raise ToolExecutorPoisonedError("shared environment has an uncertain prior effect")
            if self.frozen_identity != self._pinned_identity:
                raise RuntimeError("frozen executor identity changed before dispatch")
            if request.role.role_id not in self._permissions:
                raise ValueError("node role lacks an explicit tool permission configuration")
            if self._runtime_id is not None and self._runtime_id != request.runtime_id:
                raise ValueError("one task environment cannot be shared across different teams")
            key = (request.runtime_id, request.execution_index)
            if key in self._invocations:
                raise ValueError(
                    "node invocation cannot be executed twice against the shared world"
                )
            cap = request.role.model_maximum
            if (
                cap.input_tokens < 1
                or cap.output_tokens < 1
                or cap.model_calls < 1
                or cap.agent_turns < 1
                or cap.wall_time_milliseconds < 1
            ):
                raise ValueError("node envelope must admit at least one frozen model turn")
            self._invocations.add(key)
            self._runtime_id = request.runtime_id
            ledger = BudgetLedger(
                run_id=request.runtime_id,
                attempt_id=f"node-{request.execution_index}",
                cap=cap,
            )
            invocation = f"{request.runtime_id}:{request.node_id}:{request.execution_index}"
            codec = StructuredJsonActionCodec()
            turns: list[dict[str, JsonValue]] = []
            output = ""
            termination = "turn-limit"
            try:
                turn = 0
                while self._max_turns is None or turn < self._max_turns:
                    turn += 1
                    remaining = ledger.available
                    if (
                        remaining.model_calls < 1
                        or remaining.agent_turns < 1
                        or remaining.input_tokens < 1
                        or remaining.output_tokens < 1
                        or remaining.wall_time_milliseconds < 1
                    ):
                        termination = "budget-limit"
                        break
                    contract = self._contract(request)
                    prompt = self._prompt(request, contract, turns, ledger)
                    step_before = self._step_index
                    text, model_usage = await self._generate(
                        prompt, request, turn, ledger, invocation
                    )
                    parsed = codec.parse(text)
                    row: dict[str, JsonValue] = {
                        "turn": turn,
                        "prompt": prompt,
                        "raw_text": text,
                        "parse": parsed.to_value(),
                        "model_usage": model_usage.to_value(),
                    }
                    if parsed.action is None:
                        observation = EnvironmentObservation(
                            {"error": parsed.public_error_code or "invalid_action"},
                            "parse_error"
                            if parsed.status.value == "parse-error"
                            else "schema_invalid",
                        )
                        output = text
                    else:
                        action = parsed.action
                        permission = (action.resource_id, action.name)
                        allowed = (
                            action.kind is ActionKind.COMPLETE
                            or permission in self._permissions[request.role.role_id]
                        )
                        admitted = contract.validate(action, validate_completion=_node_completion)
                        if not allowed:
                            observation = EnvironmentObservation(
                                {"error": "role_tool_forbidden"}, "schema_invalid"
                            )
                        elif action.kind is ActionKind.TOOL and (
                            self._terminal or self.phase_budget_exhausted
                        ):
                            observation = EnvironmentObservation(
                                {
                                    "error": "environment_already_terminal"
                                    if self._terminal
                                    else "environment_phase_budget_exhausted"
                                },
                                "tool_error",
                            )
                        elif admitted.error is not None:
                            status, error = admitted.error
                            observation = EnvironmentObservation({"error": error}, status)
                        elif admitted.completion:
                            output = self._project(admitted.submission)
                            if type(output) is not str:
                                raise TypeError("local completion projector must return text")
                            termination = "local-complete"
                            row["observation"] = {
                                "status": "local-complete",
                                "value": admitted.submission,
                            }
                            turns.append(_public_object(row))
                            break
                        elif action.kind is ActionKind.SKILL and (
                            action.resource_id != "skill-runtime"
                            or action.name != "invoke"
                            or action.arguments != {}
                        ):
                            observation = EnvironmentObservation(
                                {"error": "invalid_skill_read"}, "schema_invalid"
                            )
                        elif self._terminal and action.kind is not ActionKind.SKILL:
                            observation = EnvironmentObservation(
                                {"error": "environment_already_terminal"}, "tool_error"
                            )
                        elif not self._tool_maximum.fits_within(ledger.available):
                            observation = EnvironmentObservation(
                                {"error": "node_tool_budget_exhausted"}, "tool_error"
                            )
                        else:
                            observation = await self._dispatch(
                                admitted.action,
                                admitted.invoked_skill_ids,
                                request,
                                turn,
                                ledger,
                                invocation,
                            )
                        output = observation.observation_text
                    row["observation"] = _public_observation(observation)
                    turns.append(_public_object(row))
                    if (
                        self._step_index > step_before
                        and self._phase_steps is not None
                        and (self._step_index % self._phase_steps == 0)
                    ):
                        termination = (
                            "environment-phase-budget"
                            if self.phase_budget_exhausted
                            else "environment-phase-boundary"
                        )
                        break
                ledger.assert_fully_settled()
            except BaseException:
                self._poisoned = True
                raise
            last_observation = turns[-1].get("observation") if turns else None
            last_status = (
                last_observation.get("observation_status")
                if isinstance(last_observation, dict)
                else None
            )
            failure_kind = (
                last_status
                if last_status
                in {
                    "parse_error",
                    "schema_invalid",
                    "tool_error",
                    "timeout",
                }
                else None
            )
            completed = termination == "local-complete"
            execution_status = "answered" if completed else "yielded"
            if failure_kind is not None:
                execution_status = "failed"
            return NodeExecutionResult(
                output,
                ledger.settled,
                _public_object(
                    {
                        "format": FORMAT,
                        "termination": termination,
                        "completed": completed,
                        "execution_outcome": {
                            "status": execution_status,
                            # A node that yielded has not answered; one that
                            # completed answered only if its text states one.
                            "answer_present": completed and answer_present(output),
                            "failure_kind": failure_kind,
                        },
                        "turns": turns,
                        "environment_terminal": self._terminal,
                        "environment_step_index": self._step_index,
                        "environment": self.public_environment_state(),
                        "task_success": None,  # Only the team's later trusted evaluator can know.
                        "local_budget_entries": [
                            {
                                "reservation_id": entry.reservation.reservation_id,
                                "maximum": entry.reservation.maximum.to_value(),
                                "actual": entry.settlement.actual.to_value()
                                if entry.settlement
                                else None,
                            }
                            for entry in ledger.entries
                        ],
                    }
                ),
            )


__all__ = ["FORMAT", "FrozenTextPolicy", "FrozenToolExecutor", "ToolExecutorPoisonedError"]
