"""Whole-batch join of trajectory records and scoring-time edge evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from skillev.contracts import EdgeLogprobRecord, TrajectoryRecord, TTBBatchStats


@dataclass(frozen=True, slots=True)
class TrajectoryFlowInput:
    """The single join result consumed by the trajectory-flow estimator."""

    trajectory_id: str
    horizon: int
    edges: tuple[EdgeLogprobRecord, ...]
    invoked_skill_ids_by_step: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        location = f"TrajectoryFlowInput[{self.trajectory_id}]"
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id:
            raise ValueError("TrajectoryFlowInput.trajectory_id must be non-empty")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError(f"{location}.horizon must be a positive integer")
        if not isinstance(self.edges, tuple) or len(self.edges) != self.horizon:
            raise ValueError(f"{location}.edges must contain exactly horizon edges")
        if (
            not isinstance(self.invoked_skill_ids_by_step, tuple)
            or len(self.invoked_skill_ids_by_step) != self.horizon
        ):
            raise ValueError(
                f"{location}.invoked_skill_ids_by_step must contain exactly horizon entries"
            )
        for expected_step, (edge, skill_ids) in enumerate(
            zip(self.edges, self.invoked_skill_ids_by_step, strict=True),
            start=1,
        ):
            step_location = f"{location}, step_index={expected_step}"
            if not isinstance(edge, EdgeLogprobRecord):
                raise ValueError(f"{step_location}: edge must be an EdgeLogprobRecord")
            if edge.trajectory_id != self.trajectory_id:
                raise ValueError(f"{step_location}: edge targets another trajectory")
            if edge.step_index != expected_step:
                raise ValueError(f"{step_location}: edge step index differs")
            if not isinstance(skill_ids, tuple) or any(
                not isinstance(skill_id, str) or not skill_id for skill_id in skill_ids
            ):
                raise ValueError(f"{step_location}: invoked skill IDs must be non-empty strings")
            if len(set(skill_ids)) != len(skill_ids):
                raise ValueError(f"{step_location}: invoked skill IDs must be unique")


@dataclass(frozen=True, slots=True)
class BatchFlowInput:
    """One optimizer batch joined to all trajectory-level flow inputs."""

    stats: TTBBatchStats
    trajectories: tuple[TrajectoryFlowInput, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stats, TTBBatchStats):
            raise ValueError("BatchFlowInput.stats must be a TTBBatchStats record")
        if not isinstance(self.trajectories, tuple):
            raise ValueError("BatchFlowInput.trajectories must be a tuple")
        by_id: dict[str, TrajectoryFlowInput] = {}
        for trajectory in self.trajectories:
            if not isinstance(trajectory, TrajectoryFlowInput):
                raise ValueError("BatchFlowInput requires TrajectoryFlowInput values")
            if trajectory.trajectory_id in by_id:
                raise ValueError(
                    f"batch {self.stats.batch_id!r} repeats trajectory {trajectory.trajectory_id!r}"
                )
            by_id[trajectory.trajectory_id] = trajectory
        residuals = {item.trajectory_id: item for item in self.stats.residuals}
        if set(by_id) != set(residuals):
            raise ValueError(f"batch {self.stats.batch_id!r} trajectory IDs differ from residuals")
        for trajectory_id, trajectory in by_id.items():
            if trajectory.horizon != residuals[trajectory_id].horizon:
                raise ValueError(
                    f"batch {self.stats.batch_id!r}, trajectory {trajectory_id!r}: "
                    "horizon differs from residual"
                )


def assemble_batch_flow_input(
    stats: TTBBatchStats,
    records_by_id: Mapping[str, TrajectoryRecord],
    edges: tuple[EdgeLogprobRecord, ...],
) -> BatchFlowInput:
    """Join contracts in residual order.

    This is the sole join authority for online projections and offline audit
    recomputation.
    """

    if not isinstance(stats, TTBBatchStats):
        raise ValueError("stats must be a TTBBatchStats record")
    if not isinstance(records_by_id, Mapping):
        raise ValueError("records_by_id must be a mapping")
    if not isinstance(edges, tuple):
        raise ValueError(f"batch {stats.batch_id!r}: edges must be a tuple")

    expected_ids = tuple(item.trajectory_id for item in stats.residuals)
    expected_set = set(expected_ids)
    missing_records = sorted(expected_set - set(records_by_id))
    extra_records = sorted(set(records_by_id) - expected_set)
    if missing_records or extra_records:
        raise ValueError(
            f"batch {stats.batch_id!r}: trajectory records differ; "
            f"missing={missing_records!r}, extra={extra_records!r}"
        )

    edges_by_key: dict[tuple[str, int], EdgeLogprobRecord] = {}
    scoring_policies: set[tuple[str, str, str]] = set()
    for edge in edges:
        if not isinstance(edge, EdgeLogprobRecord):
            raise ValueError("edges must contain EdgeLogprobRecord values")
        context = edge.context
        if (
            context is None
            or context.batch_id != stats.batch_id
            or context.library_version != stats.library_version
        ):
            raise ValueError(
                f"batch {stats.batch_id!r}, edge {edge.trajectory_id}:{edge.step_index}: "
                "scoring context differs"
            )
        scoring_policies.add(
            (
                context.policy_snapshot_id,
                edge.forward_adapter_version,
                edge.backward_adapter_version,
            )
        )
        if edge.trajectory_id not in expected_set:
            raise ValueError(
                f"batch {stats.batch_id!r}: edge targets extra trajectory {edge.trajectory_id!r}"
            )
        key = (edge.trajectory_id, edge.step_index)
        if key in edges_by_key:
            raise ValueError(
                f"batch {stats.batch_id!r}: duplicate edge {edge.trajectory_id}:{edge.step_index}"
            )
        edges_by_key[key] = edge
    if len(scoring_policies) != 1:
        raise ValueError(f"batch {stats.batch_id!r}: forward/backward scores mix policy versions")

    trajectories: list[TrajectoryFlowInput] = []
    expected_keys: set[tuple[str, int]] = set()
    for residual in stats.residuals:
        record = records_by_id[residual.trajectory_id]
        if not isinstance(record, TrajectoryRecord):
            raise ValueError("records_by_id values must be TrajectoryRecord records")
        if record.trajectory_id != residual.trajectory_id:
            raise ValueError("trajectory record mapping key differs from record identity")
        if record.initial_context.meta["library_version"] != stats.library_version:
            raise ValueError("trajectory and scored batch use different skill libraries")
        if residual.raw_reward != record.reward.value:
            raise ValueError("TTB reward differs from the terminal reward projection")
        if record.horizon != residual.horizon:
            raise ValueError(
                f"batch {stats.batch_id!r}, trajectory {record.trajectory_id!r}: "
                "record horizon differs from residual"
            )
        trajectory_edges: list[EdgeLogprobRecord] = []
        for step_index in range(1, record.horizon + 1):
            key = (record.trajectory_id, step_index)
            expected_keys.add(key)
            try:
                edge = edges_by_key[key]
                if (
                    edge.context is None
                    or edge.context.action_token_count
                    != record.steps[step_index - 1].action_token_count
                    or edge.context.action_token_ids
                    != record.steps[step_index - 1].action_token_ids
                ):
                    raise ValueError(
                        f"batch {stats.batch_id!r}, edge {key}: "
                        "scored action span or length denominator differs"
                    )
                trajectory_edges.append(edge)
            except KeyError as error:
                raise ValueError(
                    f"batch {stats.batch_id!r}, trajectory {record.trajectory_id!r}, "
                    f"step_index={step_index}: missing edge"
                ) from error
        for actual, expected in (
            (
                residual.sum_forward,
                math.fsum(edge.forward_logprob_per_token for edge in trajectory_edges),
            ),
            (
                residual.sum_backward,
                math.fsum(edge.backward_logprob_per_token for edge in trajectory_edges),
            ),
        ):
            if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"batch {stats.batch_id!r}, trajectory {record.trajectory_id!r}: "
                    "residual and edge scores differ"
                )
        trajectories.append(
            TrajectoryFlowInput(
                trajectory_id=record.trajectory_id,
                horizon=record.horizon,
                edges=tuple(trajectory_edges),
                invoked_skill_ids_by_step=tuple(step.invoked_skill_ids for step in record.steps),
            )
        )
    extra_keys = sorted(set(edges_by_key) - expected_keys)
    if extra_keys:
        raise ValueError(f"batch {stats.batch_id!r}: extra edges {extra_keys!r}")
    return BatchFlowInput(stats=stats, trajectories=tuple(trajectories))
