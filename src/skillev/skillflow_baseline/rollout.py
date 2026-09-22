"""Protocol 10 task bridge that never embeds trusted answers."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, cast

from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.rollout import RolloutTask
from skillev.rollout.environment import (
    NoSubmissionReason,
    NoTerminalSubmission,
    SubmittedTerminalValue,
    TerminalEvaluationRequest,
)
from skillev.rollout.types import RolloutTermination
from skillev.runtime import ActionKind, EnvironmentMethodFailedError, StructuredAction
from skillev.runtime.sglang_gateway import SGLangGateway
from src.executor.openai_request_policy import ContextBudgetExceeded

_TASK_TYPES = {
    "aime-2026": "math_reasoning",
    "alfworld": "alfworld",
    "appworld": "interactive_agent",
    "healthbench": "science_qa",
    "hotpotqa": "multi_hop_qa",
    "mbpp-plus-fixed-100": "code_generation",
    "spreadsheetbench": "interactive_agent",
    "triviaqa": "factual_qa",
    "webshop": "webshop",
}


@dataclass(frozen=True, slots=True)
class ExactSkillFlowTaskAdapter:
    """Preserve episode/source/position while projecting to upstream task fields."""

    task: RolloutTask
    sequence_position: int

    def __post_init__(self) -> None:
        if self.benchmark_id not in _TASK_TYPES:
            raise ValueError("Protocol 10 task family has no exact SkillFlow mapping")
        if type(self.sequence_position) is not int or self.sequence_position < 0:
            raise ValueError("SkillFlow sequence position must be non-negative")

    @property
    def benchmark_id(self) -> str:
        return self.task.task_family.split("/", 1)[0]

    def to_upstream(self) -> dict[str, Any]:
        context = self.task.public_context
        source_id = context.get("source_id") if isinstance(context, dict) else None
        return {
            "answer": "",
            "context": [],
            "extra": {
                "protocol_v10_episode_id": self.task.task_id,
                "protocol_v10_sequence_position": self.sequence_position,
                "source": self.benchmark_id,
                "source_id": source_id,
            },
            "question": self.task.query,
            "task_type": _TASK_TYPES[self.benchmark_id],
            "_protocol_v10_task": self.task,
        }


@dataclass(frozen=True, slots=True)
class ProtocolV10SkillFlowEpisodeRunner:
    """Run upstream prompt/action semantics against a trusted Protocol 10 session."""

    session_factory: object
    gateway: SGLangGateway | None = None

    def run(self, question: dict[str, Any], trainer: Any) -> object:
        from training.batch_inference import supervisor_call
        from training.environment import GenericTaskEnvironment
        from training.trajectory import Trajectory, Turn

        task = question.get("_protocol_v10_task")
        if not isinstance(task, RolloutTask):
            raise TypeError("exact SkillFlow episode lost its Protocol 10 task")
        create = getattr(self.session_factory, "create", None)
        if not callable(create):
            raise TypeError("exact SkillFlow session factory is unavailable")
        bundle = create(task)
        benchmark_id = task.task_family.split("/", 1)[0]
        if benchmark_id in {"webshop", "alfworld"}:
            return self._run_react(question, task, trainer, bundle)
        workspace = trainer.workspace
        config = trainer.config
        render_environment = GenericTaskEnvironment(
            m_exec=trainer.m_exec,
            max_episode_steps=int(trainer.max_episode_steps),
            epsilon_min=float(trainer.epsilon_min),
            skill_workspace=workspace,
            max_obs_chars=int(config.get("max_obs_chars", 0)),
            max_context_chars=int(config.get("max_context_chars", 0)),
            reward_mode="outcome_only",
            skill_mode="policy_action",
        )
        # Prompt rendering is reused from upstream, but the legacy environment
        # must never instantiate a second benchmark runtime or see an answer.
        render_environment._ragen_adapter = None
        messages, trajectory = render_environment.reset(question)
        if not isinstance(trajectory, Trajectory):  # pragma: no cover - upstream boundary
            raise TypeError("upstream prompt renderer returned another trajectory type")
        trajectory.gold_answer = ""
        evaluation_input: SubmittedTerminalValue | NoTerminalSubmission = NoTerminalSubmission(
            NoSubmissionReason.HORIZON_EXHAUSTED
        )
        termination = RolloutTermination.HORIZON_EXHAUSTED
        environment_step_index = 1
        try:
            for _step_index in range(1, int(trainer.max_episode_steps) + 1):
                snapshot = [dict(message) for message in messages]
                if self.gateway is not None:
                    self.gateway.begin_supervisor_rollout()
                try:
                    content, tool_name, tool_args = supervisor_call(
                        messages=messages,
                        api_base=str(config["supervisor_api_base"]),
                        model=str(config.get("supervisor_model", "supervisor_theta")),
                        tools=render_environment._tools,
                        max_tokens=int(config.get("supervisor_max_output_tokens", 512)),
                        temperature=float(config.get("supervisor_temperature", 0.8)),
                        enable_thinking=False,
                        request_policy=trainer.supervisor_request_policy,
                        request_timeout_seconds=float(
                            config.get("supervisor_request_timeout_seconds", 60)
                        ),
                    )
                except ContextBudgetExceeded as error:
                    render_environment.terminate_context_budget(
                        trajectory,
                        error,
                        failed_step=_step_index - 1,
                    )
                    break
                finally:
                    if self.gateway is not None:
                        self.gateway.end_supervisor_rollout()
                arguments = normalize_json(tool_args or {})
                if not isinstance(arguments, dict):
                    arguments = {}
                output = content or GenericTaskEnvironment._canonical_tool_call_text(
                    tool_name or "parse_error",
                    arguments,
                )
                context_text = "\n".join(
                    f"[{message['role']}] {message.get('content', '')!s}"
                    for message in snapshot[-6:]
                )
                observation_text: str
                invoked_skill: str | None = None
                recorded_by_renderer = False
                if tool_name == "answer":
                    answer = arguments.get("response", arguments.get("answer", ""))
                    submission: JsonValue = {"answer": str(answer)}
                    if bundle.environment.validate_completion(submission):
                        evaluation_input = SubmittedTerminalValue(submission)
                        termination = RolloutTermination.COMPLETED
                        trajectory.final_answer = str(answer)
                        observation_text = "accepted_for_evaluation"
                    else:
                        observation_text = "[SCHEMA_INVALID] completion was rejected"
                elif tool_name is None:
                    observation_text = "[PARSE_ERROR] no tool call was generated"
                else:
                    (
                        observation_text,
                        invoked_skill,
                        terminal_submission,
                        recorded_by_renderer,
                        environment_advanced,
                    ) = _execute_protocol_or_upstream_tool(
                        bundle=bundle,
                        render_environment=render_environment,
                        trajectory=trajectory,
                        output=output,
                        tool_name=tool_name,
                        arguments=arguments,
                        step_index=environment_step_index,
                    )
                    if environment_advanced:
                        environment_step_index += 1
                    if terminal_submission is not None:
                        evaluation_input = SubmittedTerminalValue(terminal_submission)
                        termination = RolloutTermination.COMPLETED
                if not recorded_by_renderer:
                    trajectory.add_turn(
                        Turn(
                            supervisor_input=context_text,
                            supervisor_output=output,
                            action_type=tool_name or "parse_error",
                            skill_id=invoked_skill,
                            tool_args=dict(arguments),
                            messages_snapshot=snapshot,
                            observation=observation_text,
                            parse_error=tool_name is None,
                        )
                    )
                    messages.append({"role": "assistant", "content": output})
                    messages.append({"role": "user", "content": observation_text})
                if termination is RolloutTermination.COMPLETED:
                    break
            reward = asyncio.run(
                bundle.evaluator.evaluate(
                    TerminalEvaluationRequest(
                        trajectory_id=trajectory.traj_id,
                        task_id=task.task_id,
                        termination=termination,
                        evaluation_input=evaluation_input,
                        public_transcript_hash=stable_hash(
                            {
                                "task_id": task.task_id,
                                "turns": [
                                    {
                                        "action": turn.supervisor_output,
                                        "observation": turn.observation,
                                    }
                                    for turn in trajectory.turns
                                ],
                            }
                        ),
                    )
                )
            )
            trajectory.reward = reward.value
            trajectory.answer_reward = reward.value
            trajectory.r_tilde = max(reward.value, float(trainer.epsilon_min))
            trajectory.completed = True
            return trajectory
        finally:
            if bundle.cleanup is not None:
                asyncio.run(bundle.cleanup())

    def _run_react(
        self,
        question: dict[str, Any],
        task: RolloutTask,
        trainer: Any,
        bundle: Any,
    ) -> object:
        from training.batch_inference import react_call
        from training.gflownet_trainer import TASK_TYPE_TO_ID
        from training.react_prompts import (
            _ALFWORLD_EXAMPLE,
            ALFWORLD_TEMPLATE,
            ALFWORLD_TEMPLATE_NO_HIS,
            WEBSHOP_TEMPLATE,
            WEBSHOP_TEMPLATE_NO_HIS,
        )
        from training.trajectory import Trajectory, Turn

        config = trainer.config
        family = task.task_family.split("/", 1)[0]
        observation = _initial_observation(task)
        available = _available_actions(task.public_context)
        trajectory = Trajectory(
            question=task.query,
            gold_answer="",
            task_type=family,
        )
        trajectory.task_type_id = TASK_TYPE_TO_ID.get(family, 9)
        history: list[tuple[str, str]] = []
        termination = RolloutTermination.HORIZON_EXHAUSTED
        evaluation_input: SubmittedTerminalValue | NoTerminalSubmission = NoTerminalSubmission(
            NoSubmissionReason.HORIZON_EXHAUSTED
        )
        environment_step_index = 1
        try:
            for _step_index in range(1, int(trainer.max_episode_steps) + 1):
                prompt = _react_prompt(
                    family=family,
                    task_description=task.query,
                    observation=observation,
                    available=available,
                    history=history,
                    webshop_initial=WEBSHOP_TEMPLATE_NO_HIS,
                    webshop_later=WEBSHOP_TEMPLATE,
                    alfworld_initial=ALFWORLD_TEMPLATE_NO_HIS,
                    alfworld_later=ALFWORLD_TEMPLATE,
                    alfworld_example=_ALFWORLD_EXAMPLE,
                )
                if self.gateway is not None:
                    self.gateway.begin_supervisor_rollout()
                try:
                    full_content, action_text = react_call(
                        prompt=prompt,
                        api_base=str(config["supervisor_api_base"]),
                        model=str(config.get("supervisor_model", "supervisor_theta")),
                        max_tokens=int(config.get("supervisor_max_output_tokens", 512)),
                        temperature=float(config.get("supervisor_temperature", 0.8)),
                        request_policy=trainer.supervisor_request_policy,
                        request_timeout_seconds=float(
                            config.get("supervisor_request_timeout_seconds", 60)
                        ),
                    )
                finally:
                    if self.gateway is not None:
                        self.gateway.end_supervisor_rollout()
                action = _react_action(family, action_text)
                if action is None:
                    public = {"error": "unparseable_react_action"}
                    terminal = False
                    terminal_submission = None
                else:
                    result = asyncio.run(
                        bundle.environment.execute(action, step_index=environment_step_index)
                    )
                    environment_step_index += 1
                    public = result.public_value
                    terminal = result.terminal
                    terminal_submission = result.terminal_submission
                observation = json.dumps(public, ensure_ascii=False, sort_keys=True)
                available = _available_actions(public)
                trajectory.add_turn(
                    Turn(
                        supervisor_input=prompt,
                        supervisor_output=full_content,
                        action_type=(action.name if action is not None else "parse_error"),
                        tool_args=(
                            action.arguments
                            if action is not None and isinstance(action.arguments, dict)
                            else {}
                        ),
                        messages_snapshot=[{"role": "user", "content": prompt}],
                        observation=observation,
                        parse_error=action is None,
                    )
                )
                history.append((observation, action_text or "invalid"))
                if terminal:
                    termination = RolloutTermination.COMPLETED
                    evaluation_input = SubmittedTerminalValue(cast(JsonValue, terminal_submission))
                    break
            reward = asyncio.run(
                bundle.evaluator.evaluate(
                    TerminalEvaluationRequest(
                        trajectory_id=trajectory.traj_id,
                        task_id=task.task_id,
                        termination=termination,
                        evaluation_input=evaluation_input,
                        public_transcript_hash=stable_hash(
                            {
                                "task_id": task.task_id,
                                "turns": [
                                    {
                                        "action": turn.supervisor_output,
                                        "observation": turn.observation,
                                    }
                                    for turn in trajectory.turns
                                ],
                            }
                        ),
                    )
                )
            )
            trajectory.reward = reward.value
            trajectory.answer_reward = reward.value
            trajectory.r_tilde = max(reward.value, float(trainer.epsilon_min))
            trajectory.completed = True
            return trajectory
        finally:
            if bundle.cleanup is not None:
                asyncio.run(bundle.cleanup())


def _execute_protocol_or_upstream_tool(
    *,
    bundle: Any,
    render_environment: Any,
    trajectory: Any,
    output: str,
    tool_name: str,
    arguments: dict[str, Any],
    step_index: int,
) -> tuple[str, str | None, JsonValue | None, bool, bool]:
    """Keep upstream internal tools local while reserving terminals for Protocol 10."""
    action = _structured_action(tool_name, arguments)
    try:
        observation = asyncio.run(bundle.environment.execute(action, step_index=step_index))
    except EnvironmentMethodFailedError as error:
        if error.public_error_code != "unsupported_benchmark_action":
            raise
        _, _, result = render_environment.step(output, tool_name, arguments, trajectory)
        rendered = result.get("observation", "[TOOL_COMPLETED]")
        invoked_skill = (
            getattr(trajectory.turns[-1], "skill_id", None) if trajectory.turns else None
        )
        return str(rendered), invoked_skill, None, True, False
    return (
        json.dumps(observation.public_value, ensure_ascii=False, sort_keys=True),
        action.skill_id,
        cast(JsonValue, observation.terminal_submission) if observation.terminal else None,
        False,
        True,
    )


def _structured_action(name: str, arguments: dict[str, Any]) -> StructuredAction:
    if name == "skill_invoke":
        skill_id = arguments.get("skill_id")
        if type(skill_id) is not str or not skill_id:
            raise ValueError("SkillFlow skill invocation has no skill ID")
        return StructuredAction(
            kind=ActionKind.SKILL,
            name=name,
            arguments=arguments,
            resource_id=skill_id,
            skill_id=skill_id,
        )
    return StructuredAction(
        kind=ActionKind.TOOL,
        name=name,
        arguments=arguments,
        resource_id=name,
    )


def _initial_observation(task: RolloutTask) -> str:
    context = task.public_context
    if isinstance(context, dict):
        payload = context.get("payload")
        for candidate in (context, payload):
            if isinstance(candidate, dict):
                initial = candidate.get("initial_observation")
                if type(initial) is str:
                    return initial
    return task.query


def _available_actions(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        for key in ("available_actions", "admissible_actions", "clickables"):
            actions = value.get(key)
            if isinstance(actions, list | tuple) and all(type(item) is str for item in actions):
                return tuple(actions)
    return ()


def _react_prompt(
    *,
    family: str,
    task_description: str,
    observation: str,
    available: tuple[str, ...],
    history: list[tuple[str, str]],
    webshop_initial: str,
    webshop_later: str,
    alfworld_initial: str,
    alfworld_later: str,
    alfworld_example: str,
) -> str:
    rendered_actions = "\n".join(available) if family == "webshop" else ", ".join(available)
    if not history:
        template = webshop_initial if family == "webshop" else alfworld_initial
        values: dict[str, object] = {
            "admissible_actions": rendered_actions,
            "available_actions": rendered_actions,
            "current_observation": observation,
            "example": alfworld_example,
            "task_description": task_description,
        }
    else:
        template = webshop_later if family == "webshop" else alfworld_later
        values = {
            "action_history": "\n".join(
                f"Observation: {seen}\nAction: {action}" for seen, action in history[-4:]
            ),
            "admissible_actions": rendered_actions,
            "available_actions": rendered_actions,
            "current_observation": observation,
            "current_step": len(history) + 1,
            "example": alfworld_example,
            "history_length": min(len(history), 4),
            "step_count": len(history),
            "task_description": task_description,
        }
    return template.format(**values)


def _react_action(family: str, action_text: str | None) -> StructuredAction | None:
    if type(action_text) is not str or not action_text.strip():
        return None
    action = action_text.strip()
    if family == "alfworld":
        return StructuredAction(
            kind=ActionKind.TOOL,
            name="act",
            arguments={"command": action},
            resource_id="alfworld",
        )
    if action.startswith("search[") and action.endswith("]"):
        return StructuredAction(
            kind=ActionKind.TOOL,
            name="search",
            arguments={"query": action[7:-1]},
            resource_id="webshop",
        )
    if action.startswith("click[") and action.endswith("]"):
        target = action[6:-1]
        if target.casefold() == "buy now":
            return StructuredAction(
                kind=ActionKind.TOOL,
                name="purchase",
                arguments={},
                resource_id="webshop",
            )
        return StructuredAction(
            kind=ActionKind.TOOL,
            name="click",
            arguments={"target": target},
            resource_id="webshop",
        )
    return None


__all__ = ["ExactSkillFlowTaskAdapter", "ProtocolV10SkillFlowEpisodeRunner"]
