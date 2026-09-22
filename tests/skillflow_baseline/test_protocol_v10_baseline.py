from __future__ import annotations

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from skillev.contracts import SuccessRule, TerminalReward
from skillev.rollout import EnvironmentObservation, RolloutTask
from skillev.runtime import BudgetVector, EnvironmentMethodFailedError, ExactAttemptRunPlan
from skillev.skillflow_baseline import (
    ExactSkillFlowProtocolConfig,
    ExactSkillFlowTaskAdapter,
    build_exact_skillflow_protocol_v10_application,
    compute_upstream_ttb_scalars,
)
from skillev.skillflow_baseline.rollout import (
    ProtocolV10SkillFlowEpisodeRunner,
    _execute_protocol_or_upstream_tool,
)
from src.executor.openai_request_policy import ChatTokenBudget, ContextBudgetExceeded


def _projection() -> dict[str, object]:
    document = yaml.safe_load(
        Path("configs/baseline/skillflow_upstream_semantics.yaml").read_text(encoding="utf-8")
    )
    return {
        "backward_lora_rank": document["parameterization"]["phi_lora"]["rank"],
        "batch_size": 16,
        "beta": document["objective"]["beta"],
        "epsilon_min": document["objective"]["epsilon_min"],
        "kl_coeff": document["objective"]["kl_coefficient"],
        "lora_alpha": document["parameterization"]["theta_lora"]["alpha"],
        "lora_rank": document["parameterization"]["theta_lora"]["rank"],
        "lora_target_modules": document["parameterization"]["theta_lora"]["targets"],
        "max_grad_norm": document["optimization"]["gradient_clip_norm"],
        "max_steps": 288,
        "n_trajectories_per_question": 1,
        "plateau_window_size": document["evolution"]["plateau_window"],
        "reward_mode": "outcome_only",
        "skill_mode": "policy_action",
        "ttb_edge_normalization": document["objective"]["edge_normalization"],
        "ttb_length_normalization": document["objective"]["trajectory_normalization"],
    }


def _task(index: int) -> RolloutTask:
    return RolloutTask(
        task_id=f"episode-{index}",
        environment_id="benchmark:triviaqa",
        task_family="triviaqa",
        context_id=f"private:{index}",
        query="Answer this trusted-evaluator task.",
        available_tools=(),
        public_context={"source_id": f"source-{index}"},
    )


def test_config_freezes_upstream_semantics() -> None:
    exact = ExactSkillFlowProtocolConfig(_projection())
    assert exact.values["kl_coeff"] == 0.01
    changed = _projection()
    changed["max_grad_norm"] = 1.0
    with pytest.raises(ValueError):
        ExactSkillFlowProtocolConfig(changed)


def test_task_bridge_preserves_identity_and_hides_answer() -> None:
    projected = ExactSkillFlowTaskAdapter(_task(7), 7).to_upstream()
    assert projected["answer"] == ""
    assert projected["extra"] == {
        "protocol_v10_episode_id": "episode-7",
        "protocol_v10_sequence_position": 7,
        "source": "triviaqa",
        "source_id": "source-7",
    }


def test_completion_benchmark_runs_upstream_internal_tool_locally() -> None:
    trajectory = SimpleNamespace(turns=[])

    class CompletionEnvironment:
        async def execute(self, action: Any, *, step_index: int) -> object:
            assert action.name == "analyze"
            assert step_index == 1
            raise EnvironmentMethodFailedError(
                budget_usage=BudgetVector(tool_calls=1),
                public_error_code="unsupported_benchmark_action",
            )

    class UpstreamRenderer:
        def step(
            self,
            output: str,
            tool_name: str,
            arguments: dict[str, Any],
            seen_trajectory: Any,
        ) -> tuple[float, bool, dict[str, str]]:
            assert output == "analysis"
            assert tool_name == "analyze"
            assert arguments == {"instruction": "reason"}
            assert seen_trajectory is trajectory
            trajectory.turns.append(SimpleNamespace(skill_id=None))
            return 0.0, False, {"observation": "local-tool-result"}

    result = _execute_protocol_or_upstream_tool(
        bundle=SimpleNamespace(environment=CompletionEnvironment()),
        render_environment=UpstreamRenderer(),
        trajectory=trajectory,
        output="analysis",
        tool_name="analyze",
        arguments={"instruction": "reason"},
        step_index=1,
    )

    assert result == ("local-tool-result", None, None, True, False)


def test_react_parse_error_does_not_advance_protocol_environment_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import training.batch_inference as batch_inference

    generations = iter((("unparseable", None), ("search", "search[public query]")))
    monkeypatch.setattr(batch_inference, "react_call", lambda **_: next(generations))

    task = RolloutTask(
        task_id="webshop-episode",
        environment_id="benchmark:webshop",
        task_family="webshop",
        context_id="webshop-public",
        query="Find a public product.",
        available_tools=(),
        public_context={
            "initial_observation": "store home",
            "available_actions": ["search[public query]"],
        },
    )

    class Environment:
        def __init__(self) -> None:
            self.step_indices: list[int] = []

        async def execute(self, action: Any, *, step_index: int) -> EnvironmentObservation:
            self.step_indices.append(step_index)
            assert action.name == "search"
            return EnvironmentObservation(
                public_value={"available_actions": []},
                observation_status="success",
                terminal_submission={"status": "finished"},
                terminal=True,
                budget_usage=BudgetVector(tool_calls=1),
            )

    class Evaluator:
        async def evaluate(self, request: Any) -> TerminalReward:
            del request
            return TerminalReward(
                value=0.0,
                success=False,
                success_rule=SuccessRule.R_AT_THRESHOLD,
                success_threshold=0.5,
                native_metric_name="webshop-score",
                native_payload={},
                environment_id=task.environment_id,
                verifier_version="test-webshop-v1",
            )

    environment = Environment()
    bundle = SimpleNamespace(environment=environment, evaluator=Evaluator(), cleanup=None)
    runner = ProtocolV10SkillFlowEpisodeRunner(
        session_factory=SimpleNamespace(create=lambda _: bundle)
    )
    trainer = SimpleNamespace(
        config={
            "supervisor_api_base": "http://unused",
            "supervisor_max_output_tokens": 32,
            "supervisor_temperature": 0.8,
        },
        epsilon_min=0.1,
        max_episode_steps=2,
        supervisor_request_policy=SimpleNamespace(),
    )

    trajectory = runner.run({"_protocol_v10_task": task}, trainer)

    assert environment.step_indices == [1]
    assert trajectory.n_parse_errors == 1
    assert trajectory.completed is True


def test_completion_context_budget_is_a_closed_zero_reward_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import training.batch_inference as batch_inference
    import training.environment as training_environment
    from training.trajectory import Trajectory, Turn

    budget_error = ContextBudgetExceeded(
        ChatTokenBudget(
            prompt_tokens=32_700,
            completion_tokens=512,
            safety_tokens=64,
            context_limit=32_768,
        ),
        request_kind="supervisor",
        source="local_preflight",
    )

    def exceed_budget(**_: object) -> tuple[str, None, None]:
        raise budget_error

    monkeypatch.setattr(batch_inference, "supervisor_call", exceed_budget)

    class Renderer:
        def __init__(self, **_: object) -> None:
            self._tools: list[object] = []

        def reset(self, question: dict[str, object]) -> tuple[list[dict[str, str]], Trajectory]:
            return ([{"role": "user", "content": str(question["question"])}], Trajectory())

        def terminate_context_budget(
            self,
            trajectory: Trajectory,
            error: ContextBudgetExceeded,
            *,
            failed_step: int,
        ) -> None:
            assert error is budget_error
            assert failed_step == 0
            trajectory.add_turn(
                Turn(
                    supervisor_input="",
                    supervisor_output="",
                    action_type="context_budget_stop",
                    observation="[CONTEXT_BUDGET_EXCEEDED]",
                    parse_error=True,
                )
            )
            trajectory.completed = True
            trajectory.truncated = True

    monkeypatch.setattr(training_environment, "GenericTaskEnvironment", Renderer)

    task = _task(3)
    requests: list[Any] = []

    class Evaluator:
        async def evaluate(self, request: Any) -> TerminalReward:
            requests.append(request)
            return TerminalReward(
                value=0.0,
                success=False,
                success_rule=SuccessRule.R_AT_THRESHOLD,
                success_threshold=0.5,
                native_metric_name="triviaqa-score",
                native_payload={},
                environment_id=task.environment_id,
                verifier_version="test-triviaqa-v1",
            )

    bundle = SimpleNamespace(
        environment=SimpleNamespace(),
        evaluator=Evaluator(),
        cleanup=None,
    )
    runner = ProtocolV10SkillFlowEpisodeRunner(
        session_factory=SimpleNamespace(create=lambda _: bundle)
    )
    trainer = SimpleNamespace(
        config={
            "supervisor_api_base": "http://unused",
            "supervisor_max_output_tokens": 512,
            "supervisor_temperature": 0.8,
        },
        epsilon_min=0.1,
        m_exec=SimpleNamespace(),
        max_episode_steps=1,
        supervisor_request_policy=SimpleNamespace(),
        workspace=SimpleNamespace(),
    )

    trajectory = runner.run(
        {"_protocol_v10_task": task, "question": task.query, "task_type": "factual_qa"},
        trainer,
    )

    assert trajectory.truncated is True
    assert trajectory.completed is True
    assert trajectory.reward == 0.0
    assert requests[0].termination.value == "horizon-exhausted"


def test_upstream_ttb_scalar_projection() -> None:
    scalars = compute_upstream_ttb_scalars(
        log_z=0.4,
        forward_logprob=-1.2,
        backward_logprob=-0.7,
        reward=0.5,
        beta=1.0,
        effective_steps=2,
        action_token_count=4,
        batch_size=16,
        epsilon_min=0.1,
    )
    raw = 0.4 - 1.2 - math.log(0.5) + 0.7
    normalized = raw / 2
    assert scalars.raw_residual == pytest.approx(raw)
    assert scalars.normalized_residual == pytest.approx(normalized)
    assert scalars.loss == pytest.approx(normalized**2)
    assert scalars.z_scale == pytest.approx(2 * normalized / 32)
    assert scalars.theta_scale_per_token == pytest.approx(scalars.z_scale / 4)
    assert scalars.phi_scale_per_token == pytest.approx(-scalars.z_scale / 4)


class FakeGFlowNetTrainer:
    max_steps = 288

    def __init__(self) -> None:
        self._current_step = 0
        self.train_data: list[dict[str, object]] = []

    def setup(
        self,
        train_data: list[dict[str, object]],
        val_data: list[dict[str, object]],
    ) -> None:
        assert val_data == []
        self.train_data = train_data

    def train(self) -> None:
        self._current_step = self.max_steps


def test_application_consumes_the_frozen_order_once() -> None:
    trainer = FakeGFlowNetTrainer()
    application = build_exact_skillflow_protocol_v10_application(
        trainer=trainer,
        ordered_tasks=tuple(
            ExactSkillFlowTaskAdapter(_task(index), index) for index in range(4608)
        ),
        config=ExactSkillFlowProtocolConfig(_projection()),
    )
    summary = asyncio.run(
        application.run(
            ExactAttemptRunPlan(phase_search_steps=266, closure_steps=22, maximum_cycles=13)
        )
    )
    assert summary.final_optimizer_step == 288
    assert len(trainer.train_data) == 4608
    assert [
        item["extra"]["protocol_v10_sequence_position"]  # type: ignore[index]
        for item in trainer.train_data
    ] == list(range(4608))


def test_application_can_commit_one_non_formal_smoke_step() -> None:
    trainer = FakeGFlowNetTrainer()
    application = build_exact_skillflow_protocol_v10_application(
        trainer=trainer,
        ordered_tasks=tuple(
            ExactSkillFlowTaskAdapter(_task(index), index) for index in range(4608)
        ),
        config=ExactSkillFlowProtocolConfig(_projection()),
    )

    summary = asyncio.run(
        application.run(
            ExactAttemptRunPlan(phase_search_steps=266, closure_steps=22, maximum_cycles=13),
            maximum_steps_this_attempt=1,
        )
    )

    assert summary.initial_optimizer_step == 0
    assert summary.final_optimizer_step == 1
