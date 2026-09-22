"""Fail-closed, observer-only checks over persisted rollout dataflow."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.diagnostics.rollout_trace import RolloutTraceEvent, RolloutTraceStage
from skillev.rollout.artifact import RolloutArtifact
from skillev.scoring import render_forward_prefix_from_parts, render_reasoning_prefix


class TrainingDataflowInvariantError(RuntimeError):
    """A persisted rollout cannot be joined into the claimed training flow."""


@dataclass(frozen=True, slots=True)
class TrainingDataflowInvariantChecker:
    require_empty_skills_for_condition: frozenset[str]

    def check_rollout(
        self,
        artifact: RolloutArtifact,
        *,
        condition_id: str,
        trace: tuple[RolloutTraceEvent, ...] | None = None,
    ) -> None:
        if artifact.manifest.condition_id != condition_id:
            raise TrainingDataflowInvariantError("artifact condition identity changed")
        task_id = artifact.manifest.task_id
        context_task_id = artifact.initial_context.contract.meta.get("task_id")
        if task_id != context_task_id:
            raise TrainingDataflowInvariantError("task identity changed between H0 and manifest")
        if artifact.record.trajectory_id != artifact.manifest.trajectory_id:
            raise TrainingDataflowInvariantError("trajectory identity changed")
        if condition_id in self.require_empty_skills_for_condition and (
            artifact.initial_context.contract.active_skill_ids
            or artifact.initial_context.contract.retrieved_skill_ids
        ):
            raise TrainingDataflowInvariantError("no-skill condition received a skill")
        expected_steps = tuple(range(1, artifact.record.horizon + 1))
        if tuple(step.index for step in artifact.record.steps) != expected_steps:
            raise TrainingDataflowInvariantError("trajectory step sequence is not contiguous")
        manifest = artifact.manifest
        if len(manifest.reasoning_finish_reasons) != artifact.record.horizon:
            raise TrainingDataflowInvariantError("reasoning finish reasons do not align")
        if len(manifest.action_finish_reasons) != artifact.record.horizon:
            raise TrainingDataflowInvariantError("action finish reasons do not align")
        if trace is not None:
            self._check_trace(artifact, trace)

    @staticmethod
    def _check_trace(artifact: RolloutArtifact, trace: tuple[RolloutTraceEvent, ...]) -> None:
        if any(event.trajectory_id != artifact.manifest.trajectory_id for event in trace):
            raise TrainingDataflowInvariantError("trace contains another trajectory")
        by_step: dict[int, list[RolloutTraceEvent]] = {}
        for event in trace:
            if event.step_index is not None:
                by_step.setdefault(event.step_index, []).append(event)
        for step in artifact.record.steps:
            events = by_step.get(step.index, [])
            counts = {
                stage: sum(event.stage is stage for event in events)
                for stage in (
                    RolloutTraceStage.REASONING_REQUEST,
                    RolloutTraceStage.ACTION_REQUEST,
                    RolloutTraceStage.ACTION_RESULT,
                    RolloutTraceStage.ACTION_PARSED,
                    RolloutTraceStage.ENVIRONMENT_RESULT,
                )
            }
            if any(value != 1 for value in counts.values()):
                raise TrainingDataflowInvariantError("trace stage is missing or duplicated")
            reasoning = next(
                event for event in events if event.stage is RolloutTraceStage.REASONING_REQUEST
            )
            expected_reasoning = render_reasoning_prefix(
                artifact.initial_context.text,
                artifact.record.steps[: step.index - 1],
                step.index,
            ).text
            if reasoning.public_payload.get("prompt_text") != expected_reasoning:
                raise TrainingDataflowInvariantError(
                    "next reasoning request differs from canonical trajectory state"
                )
            action_request = next(
                event for event in events if event.stage is RolloutTraceStage.ACTION_REQUEST
            )
            expected_action = render_forward_prefix_from_parts(
                artifact.initial_context.text,
                artifact.record.steps[: step.index - 1],
                step.index,
                step.reasoning_text,
            ).text
            if action_request.public_payload.get("prompt_text") != expected_action:
                raise TrainingDataflowInvariantError("action request differs from canonical prefix")
            action_result = next(
                event for event in events if event.stage is RolloutTraceStage.ACTION_RESULT
            )
            parsed = next(
                event for event in events if event.stage is RolloutTraceStage.ACTION_PARSED
            )
            if action_result.public_payload.get("raw_text") != parsed.public_payload.get(
                "raw_text"
            ):
                raise TrainingDataflowInvariantError("parsed action differs from generated action")
            environment = next(
                event for event in events if event.stage is RolloutTraceStage.ENVIRONMENT_RESULT
            )
            invoked = environment.public_payload.get("invoked_skill_ids", [])
            if not isinstance(invoked, list) or tuple(invoked) != step.invoked_skill_ids:
                raise TrainingDataflowInvariantError("trace skill invocation differs from artifact")
        terminal_requests = tuple(
            item for item in trace if item.stage is RolloutTraceStage.TERMINAL_REQUEST
        )
        terminal_results = tuple(
            item for item in trace if item.stage is RolloutTraceStage.TERMINAL_RESULT
        )
        if len(terminal_requests) != 1 or len(terminal_results) != 1:
            raise TrainingDataflowInvariantError("trace terminal lifecycle is incomplete")
        if terminal_requests[0].public_payload.get("termination") != (
            artifact.manifest.termination.value
        ):
            raise TrainingDataflowInvariantError("trace termination differs from manifest")


__all__ = ["TrainingDataflowInvariantChecker", "TrainingDataflowInvariantError"]
