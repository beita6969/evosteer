"""Group only one trajectory and one adapter; preserve each action denominator."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from typing import Any

from skillev.policy import AdapterRole, PolicyBackbone
from skillev.policy.interface import PolicyScoringMemoryError
from skillev.policy.scoring_execution import TeacherForcingConfig

from .edge_plan import PreparedEdge, PreparedEdgePlan


def edge_groups(
    plan: PreparedEdgePlan, role: AdapterRole, config: TeacherForcingConfig
) -> Iterator[tuple[PreparedEdge, ...]]:
    edges = [edge for edge in plan.edges if edge.role is role]
    # Do not reorder the baseline one-edge path. Grouped execution is explicit.
    if config.microbatch_size > 1:
        edges.sort(
            key=lambda edge: (edge.full_length.bit_length(), edge.full_length, edge.step_index)
        )
    group: list[PreparedEdge] = []
    for edge in edges:
        if group and (
            len(group) == config.microbatch_size
            or max(edge.full_length, group[-1].full_length) * (len(group) + 1)
            > config.microbatch_max_tokens
            or group[0].full_length.bit_length() != edge.full_length.bit_length()
        ):
            yield tuple(group)
            group = []
        group.append(edge)
    if group:
        yield tuple(group)


def backward_edge_groups(
    backbone: PolicyBackbone, plan: PreparedEdgePlan, role: AdapterRole
) -> list[float]:
    from skillev.policy.teacher_forcing import backward_action_logprob_means

    config = getattr(backbone, "teacher_forcing_config", TeacherForcingConfig())
    score_batch = getattr(backbone, "score_edge_microbatch", None)
    if score_batch is None:
        raise TypeError("edge microbatching requires a backend with explicit padded scoring")
    values = [0.0] * plan.record.horizon
    for group in edge_groups(plan, role, config):
        timer: Any = getattr(backbone, "edge_scoring_timer", nullcontext)
        with timer():
            try:
                scores = score_batch(
                    tuple((edge.prefix_ids, edge.action_ids) for edge in group), role
                )
            except PolicyScoringMemoryError as error:
                from .objective import ScoringDirection, TrajectoryScoringMemoryError

                longest = max(group, key=lambda edge: edge.full_length)
                failure = TrajectoryScoringMemoryError(
                    trajectory_id=plan.record.trajectory_id,
                    step_index=longest.step_index,
                    direction=(
                        ScoringDirection.FORWARD
                        if role is AdapterRole.FORWARD_POLICY
                        else ScoringDirection.HINDSIGHT
                    ),
                    prefix_token_count=len(longest.prefix_ids),
                    action_token_count=len(longest.action_ids),
                )
                failure.microbatch_steps = tuple(edge.step_index for edge in group)
                raise failure from error
            if len(scores) != len(group) or any(
                score.ndim != 1 or len(score) != len(edge.action_ids)
                for score, edge in zip(scores, group, strict=True)
            ):
                raise ValueError("microbatch scorer changed the action token spans")
            means = backward_action_logprob_means(
                scores,
                tuple(len(edge.action_ids) for edge in group),
                forward=role is AdapterRole.FORWARD_POLICY,
            )
            for edge, value in zip(group, means, strict=True):
                values[edge.step_index - 1] = value
            finish_profile = getattr(backbone, "finish_edge_backward_profile", None)
            if finish_profile is not None:
                finish_profile()
    return values
