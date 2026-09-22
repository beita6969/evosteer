from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import skillev.runtime as runtime
from skillev.contracts import JsonValue, canonical_json
from skillev.runtime import (
    ActionKind,
    ActionParseResult,
    ActionParseStatus,
    AgentTurnsExhaustedError,
    BoundedAgent,
    BoundedAgentPolicy,
    BoundedAgentState,
    BoundedAgentTurnRequest,
    BudgetLedger,
    BudgetVector,
    EnvironmentMethodFailedError,
    EnvironmentMethodTimeoutError,
    EnvironmentObservation,
    EnvironmentSkillInvocationMismatchError,
    EventType,
    LiveAttemptEventLog,
    RuntimeEventEmitter,
    StructuredAction,
)
from skillev.runtime.event_log_reader import read_event_history


@dataclass(slots=True)
class ScriptedEnvironment:
    results: list[EnvironmentObservation | BaseException]
    environment_id: str = "public-debug-environment"
    task_family: str = "debug"
    calls: list[tuple[StructuredAction, int]] = field(default_factory=list)
    completion_error: BaseException | None = None

    async def execute(
        self,
        action: StructuredAction,
        *,
        step_index: int,
    ) -> EnvironmentObservation:
        self.calls.append((action, step_index))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def validate_completion(self, submission: JsonValue) -> bool:
        if self.completion_error is not None:
            raise self.completion_error
        return isinstance(submission, dict) and set(submission) == {"answer"}


def _agent(
    tmp_path: Path,
    environment: ScriptedEnvironment,
    *,
    max_turns: int = 3,
) -> tuple[BoundedAgent, BudgetLedger, LiveAttemptEventLog]:
    ledger = BudgetLedger(
        run_id="run",
        attempt_id="attempt",
        cap=BudgetVector(tool_calls=4, wall_time_milliseconds=100),
    )
    log = LiveAttemptEventLog(
        tmp_path / "events.jsonl",
        run_id="run",
        attempt_id="attempt",
    )
    emitter = RuntimeEventEmitter(
        log=log,
        producer_id="execution",
        clock=lambda: "2026-01-01T00:00:00Z",
    )
    return (
        BoundedAgent(
            environment=environment,
            policy=BoundedAgentPolicy(max_turns=max_turns),
            ledger=ledger,
            tool_call_maximum=BudgetVector(
                tool_calls=1,
                wall_time_milliseconds=25,
            ),
            emitter=emitter,
        ),
        ledger,
        log,
    )


def _skill_action(skill_id: str = "model-claimed-skill") -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.SKILL,
        name="lookup",
        arguments={"query": "public request"},
        resource_id="public-tool",
        skill_id=skill_id,
    )


def _complete_action(value: object) -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.COMPLETE,
        name="complete",
        arguments={"value": value},
    )


def _tool_action() -> StructuredAction:
    return StructuredAction(
        kind=ActionKind.TOOL,
        name="lookup",
        arguments={"query": "public request"},
        resource_id="public-tool",
    )


def _request(
    *,
    step: int,
    parse_result: ActionParseResult,
    retrieved_skill_ids: tuple[str, ...] = ("model-claimed-skill",),
    active_skill_ids: tuple[str, ...] = ("model-claimed-skill",),
) -> BoundedAgentTurnRequest:
    action_text = (
        canonical_json(parse_result.action.to_value())
        if parse_result.action is not None
        else "not-json"
    )
    return BoundedAgentTurnRequest(
        trajectory_id="trajectory-one",
        step_index=step,
        action_text=action_text,
        action_token_ids=(101, 102),
        parse_result=parse_result,
        retrieved_skill_ids=retrieved_skill_ids,
        active_skill_ids=active_skill_ids,
    )


def test_bounded_agent_public_api_is_execution_only() -> None:
    parameters = set(inspect.signature(BoundedAgent).parameters)
    assert {"environment", "policy", "ledger", "tool_call_maximum", "emitter"} <= parameters
    assert "model" not in parameters
    assert "terminal_evaluator" not in parameters
    for retired_name in (
        "ActionParser",
        "BackwardTokenScorer",
        "BoundedAgentLoopState",
        "PendingTrajectoryStep",
        "PromptBuilder",
        "TrajectoryStep",
    ):
        assert retired_name not in runtime.__all__
        assert not hasattr(runtime, retired_name)


def test_execution_contracts_round_trip_exactly() -> None:
    parse_result = ActionParseResult(ActionParseStatus.VALID, _skill_action(), None)
    assert ActionParseResult.from_value(parse_result.to_value()) == parse_result
    observation = EnvironmentObservation(
        public_value={"z": 1, "a": [2, 3]},
        observation_status="success",
        invoked_skill_ids=("actual-skill",),
        budget_usage=BudgetVector(tool_calls=1),
    )
    assert EnvironmentObservation.from_value(observation.to_value()) == observation
    with pytest.raises(ValueError):
        BoundedAgentState(
            invocation_id="invocation",
            completed=False,
            completion_value={"answer": 1},
        )


def test_valid_execution_settles_exact_measured_usage(tmp_path: Path) -> None:
    environment = ScriptedEnvironment(
        [
            EnvironmentObservation(
                public_value={"result": 6},
                observation_status="success",
                invoked_skill_ids=("model-claimed-skill",),
                budget_usage=BudgetVector(tool_calls=1, wall_time_milliseconds=4),
            )
        ]
    )
    agent, ledger, log = _agent(tmp_path, environment)
    result = asyncio.run(
        agent.execute_turn(
            BoundedAgentState(invocation_id="invocation"),
            _request(
                step=1,
                parse_result=ActionParseResult(
                    ActionParseStatus.VALID,
                    _skill_action(),
                    None,
                ),
            ),
        )
    )
    assert result.state == BoundedAgentState(invocation_id="invocation", turns_used=1)
    assert result.observation.invoked_skill_ids == ("model-claimed-skill",)
    assert ledger.settled == BudgetVector(tool_calls=1, wall_time_milliseconds=4)
    event_types = tuple(event.event_type for event in read_event_history(log.path))
    assert EventType.BUDGET_RESERVED in event_types
    assert EventType.BUDGET_SETTLED in event_types
    assert EventType.AGENT_STEP_RECORDED in event_types


@pytest.mark.parametrize(
    ("status", "observation_status"),
    [
        (ActionParseStatus.PARSE_ERROR, "parse_error"),
        (ActionParseStatus.SCHEMA_INVALID, "schema_invalid"),
    ],
)
def test_invalid_action_is_data_without_execution_or_resampling(
    tmp_path: Path,
    status: ActionParseStatus,
    observation_status: str,
) -> None:
    environment = ScriptedEnvironment([])
    agent, ledger, _ = _agent(tmp_path, environment)
    result = asyncio.run(
        agent.execute_turn(
            BoundedAgentState(invocation_id="invocation"),
            _request(
                step=1,
                parse_result=ActionParseResult(status, None, "public-invalid-action"),
            ),
        )
    )
    assert environment.calls == []
    assert result.observation.observation_status == observation_status
    assert result.state.turns_used == 1
    assert ledger.entries == ()


@pytest.mark.parametrize(
    ("skill_id", "retrieved_skill_ids", "active_skill_ids"),
    [
        ("skill-unknown", ("model-claimed-skill",), ("model-claimed-skill",)),
        ("skill-inactive", ("model-claimed-skill",), ("model-claimed-skill",)),
        (
            "skill-active-not-retrieved",
            ("model-claimed-skill",),
            ("model-claimed-skill", "skill-active-not-retrieved"),
        ),
    ],
)
def test_unavailable_skill_action_is_agent_data_without_environment_credit(
    tmp_path: Path,
    skill_id: str,
    retrieved_skill_ids: tuple[str, ...],
    active_skill_ids: tuple[str, ...],
) -> None:
    environment = ScriptedEnvironment([])
    agent, ledger, _ = _agent(tmp_path, environment)

    result = asyncio.run(
        agent.execute_turn(
            BoundedAgentState(invocation_id="invocation"),
            _request(
                step=1,
                parse_result=ActionParseResult(
                    ActionParseStatus.VALID,
                    _skill_action(skill_id),
                    None,
                ),
                retrieved_skill_ids=retrieved_skill_ids,
                active_skill_ids=active_skill_ids,
            ),
        )
    )

    assert environment.calls == []
    assert ledger.entries == ()
    assert result.observation.observation_status == "schema_invalid"
    assert result.observation.invoked_skill_ids == ()


@pytest.mark.parametrize(
    "action",
    [_skill_action(), _tool_action()],
)
def test_environment_cannot_inject_skill_credit(tmp_path: Path, action: StructuredAction) -> None:
    environment = ScriptedEnvironment(
        [
            EnvironmentObservation(
                public_value={"result": "public"},
                observation_status="success",
                invoked_skill_ids=("other-skill",),
                budget_usage=BudgetVector(tool_calls=1),
            )
        ]
    )
    agent, ledger, _ = _agent(tmp_path, environment)

    with pytest.raises(EnvironmentSkillInvocationMismatchError):
        asyncio.run(
            agent.execute_turn(
                BoundedAgentState(invocation_id="invocation"),
                _request(
                    step=1,
                    parse_result=ActionParseResult(ActionParseStatus.VALID, action, None),
                ),
            )
        )

    assert len(environment.calls) == 1
    assert ledger.entries[0].state is runtime.ReservationState.RESERVED


def test_completion_validation_failure_is_not_reclassified_as_agent_data(
    tmp_path: Path,
) -> None:
    environment = ScriptedEnvironment([], completion_error=RuntimeError("PRIVATE failure"))
    agent, ledger, log = _agent(tmp_path, environment)
    with pytest.raises(RuntimeError, match="PRIVATE"):
        asyncio.run(
            agent.execute_turn(
                BoundedAgentState(invocation_id="invocation"),
                _request(
                    step=1,
                    parse_result=ActionParseResult(
                        ActionParseStatus.VALID,
                        _complete_action({"answer": 6}),
                        None,
                    ),
                ),
            )
        )
    assert ledger.entries == ()
    assert "PRIVATE" not in canonical_json(
        [event.to_value() for event in read_event_history(log.path)]
    )


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (
            EnvironmentMethodTimeoutError(
                budget_usage=BudgetVector(tool_calls=1, wall_time_milliseconds=7)
            ),
            "timeout",
        ),
        (
            EnvironmentMethodFailedError(
                budget_usage=BudgetVector(tool_calls=1, wall_time_milliseconds=7)
            ),
            "tool_error",
        ),
    ],
)
def test_typed_environment_outcomes_settle_exact_usage(
    tmp_path: Path,
    failure: BaseException,
    expected_status: str,
) -> None:
    agent, ledger, _ = _agent(tmp_path, ScriptedEnvironment([failure]))
    result = asyncio.run(
        agent.execute_turn(
            BoundedAgentState(invocation_id="invocation"),
            _request(
                step=1,
                parse_result=ActionParseResult(
                    ActionParseStatus.VALID,
                    _skill_action(),
                    None,
                ),
            ),
        )
    )
    assert result.observation.observation_status == expected_status
    assert ledger.settled == BudgetVector(tool_calls=1, wall_time_milliseconds=7)


@pytest.mark.parametrize(
    "failure",
    [
        AssertionError("PRIVATE assertion"),
        TypeError("PRIVATE type error"),
        TimeoutError("PRIVATE untyped timeout"),
    ],
)
def test_unknown_environment_failure_terminates_with_unsettled_reservation(
    tmp_path: Path,
    failure: BaseException,
) -> None:
    agent, ledger, log = _agent(tmp_path, ScriptedEnvironment([failure]))
    with pytest.raises(type(failure), match="PRIVATE"):
        asyncio.run(
            agent.execute_turn(
                BoundedAgentState(invocation_id="invocation"),
                _request(
                    step=1,
                    parse_result=ActionParseResult(
                        ActionParseStatus.VALID,
                        _skill_action(),
                        None,
                    ),
                ),
            )
        )
    assert ledger.entries[0].state is runtime.ReservationState.RESERVED
    assert ledger.settled == BudgetVector()
    events = read_event_history(log.path)
    assert EventType.BUDGET_SETTLED not in {event.event_type for event in events}
    assert "PRIVATE" not in canonical_json([event.to_value() for event in events])


def test_zero_tool_call_measurement_is_rejected_not_rewritten(tmp_path: Path) -> None:
    agent, ledger, _ = _agent(
        tmp_path,
        ScriptedEnvironment(
            [
                EnvironmentObservation(
                    public_value={"result": 6},
                    observation_status="success",
                    invoked_skill_ids=("model-claimed-skill",),
                    budget_usage=BudgetVector(wall_time_milliseconds=4),
                )
            ]
        ),
    )
    with pytest.raises(ValueError):
        asyncio.run(
            agent.execute_turn(
                BoundedAgentState(invocation_id="invocation"),
                _request(
                    step=1,
                    parse_result=ActionParseResult(
                        ActionParseStatus.VALID,
                        _skill_action(),
                        None,
                    ),
                ),
            )
        )
    assert ledger.entries[0].state is runtime.ReservationState.RESERVED


def test_turn_order_and_horizon_are_enforced(tmp_path: Path) -> None:
    agent, _, _ = _agent(tmp_path, ScriptedEnvironment([]), max_turns=1)
    invalid = ActionParseResult(ActionParseStatus.PARSE_ERROR, None, "invalid-json")
    state = BoundedAgentState(invocation_id="invocation")
    with pytest.raises(ValueError):
        asyncio.run(agent.execute_turn(state, _request(step=2, parse_result=invalid)))
    first = asyncio.run(agent.execute_turn(state, _request(step=1, parse_result=invalid)))
    with pytest.raises(AgentTurnsExhaustedError):
        asyncio.run(agent.execute_turn(first.state, _request(step=2, parse_result=invalid)))


@pytest.mark.parametrize(
    "kind", ["parse-error", "unavailable-skill", "complete", "tool-error", "terminal"]
)
def test_explicit_action_assessment_does_not_confuse_execution_and_task_success(tmp_path, kind):
    observation = EnvironmentObservation(
        public_value={"status": "public-feedback"},
        observation_status="tool_error" if kind == "tool-error" else "success",
        terminal=kind == "terminal",
        terminal_submission={"answer": "submitted"} if kind == "terminal" else None,
        budget_usage=BudgetVector(tool_calls=1),
    )
    environment = ScriptedEnvironment([observation])
    agent, _, log = _agent(tmp_path, environment)
    action = _tool_action()
    if kind == "unavailable-skill":
        action = _skill_action("unavailable")
    elif kind == "complete":
        action = _complete_action({"answer": "submitted"})
    parsed = (
        ActionParseResult(ActionParseStatus.PARSE_ERROR, None, "action_not_json")
        if kind == "parse-error"
        else ActionParseResult(ActionParseStatus.VALID, action, None)
    )
    result = asyncio.run(
        agent.execute_turn(
            BoundedAgentState(invocation_id="invocation"), _request(step=1, parse_result=parsed)
        )
    )
    event = next(
        e for e in read_event_history(log.path) if e.event_type is EventType.AGENT_STEP_RECORDED
    )
    assessment = event.payload["assessment"]
    assert assessment["executed"] is (kind in ("tool-error", "terminal"))
    assert assessment["admitted"] is (kind in ("tool-error", "terminal", "complete"))
    assert assessment["accepted_submission"] is (kind == "complete")
    assert assessment["environment_terminal"] is (kind == "terminal")
    assert assessment["terminal_task_success"] is None
    assert event.payload["observation_text"] == result.observation.observation_text
    assert event.payload["action_token_ids"] == [101, 102]
