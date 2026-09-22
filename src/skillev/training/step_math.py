"""Shared TTB optimizer mathematics without projection or source-event shape."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import torch

from skillev.contracts import (
    EdgeLogprobRecord,
    JsonValue,
    TrainingStepReportValue,
    TrajectoryResidual,
    TTBBatchStats,
)
from skillev.policy import AdapterRole, PolicyBackbone, PolicyParameterGroups
from skillev.rollout import PolicySnapshot, RolloutArtifact
from skillev.scoring import (
    ScoringConfig,
    backward_trajectory_delta_streaming,
    materialize_edge_records,
    materialize_residual,
)
from skillev.scoring.edge_plan import PreparedEdgePlan, prepare_edge_plan

from .config import OptimizerConfig
from .optimizer_observation import OptimizerTransitionObservation
from .planning import CollectedTrainingBatch

if TYPE_CHECKING:
    from .provisional_math import ProvisionalGradientPool


@dataclass(frozen=True, slots=True)
class ComponentGradientNorms:
    forward: float
    backward: float
    z_head: float


@dataclass(frozen=True, slots=True)
class PreparedTTBStep:
    batch: CollectedTrainingBatch
    snapshot_before: PolicySnapshot
    residuals: tuple[TrajectoryResidual, ...]
    edges: tuple[EdgeLogprobRecord, ...]
    stats: TTBBatchStats
    detached_losses: tuple[float, ...]
    gradient_norms: ComponentGradientNorms
    started_at: str


class TTBStepPreparer(Protocol):
    """Replaceable gradient boundary used by local and process workers."""

    def prepare(
        self,
        *,
        backbone: PolicyBackbone,
        optimizer: torch.optim.Optimizer,
        parameters: PolicyParameterGroups,
        batch: CollectedTrainingBatch,
        snapshot_before: PolicySnapshot,
        temperature_beta: float,
        clock: Callable[[], str],
    ) -> PreparedTTBStep: ...


@dataclass(frozen=True, slots=True)
class TTBArtifactMath:
    """Detached per-trajectory math returned by one gradient worker."""

    position: int
    trajectory_id: str
    residual: TrajectoryResidual
    edges: tuple[EdgeLogprobRecord, ...]
    loss: float
    scoring_metrics: tuple[dict[str, JsonValue], ...] = field(default=(), compare=False)


@dataclass(frozen=True, slots=True)
class TTBGradientShard:
    """Globally scaled gradient contribution for a disjoint trajectory subset."""

    batch_id: str
    optimizer_step: int
    global_batch_size: int
    artifacts: tuple[TTBArtifactMath, ...]
    gradients: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not self.artifacts or not self.gradients:
            raise ValueError("TTB gradient shard cannot be empty")
        positions = tuple(item.position for item in self.artifacts)
        if len(positions) != len(set(positions)):
            raise ValueError("TTB gradient shard repeats a trajectory position")
        require_finite_gradients(self.gradients)


def create_ttb_optimizer(
    backbone: PolicyBackbone,
    config: OptimizerConfig,
) -> tuple[torch.optim.Optimizer, PolicyParameterGroups]:
    """Create the sole three-group AdamW optimizer used by full and arm loops."""

    groups = backbone.parameter_groups()
    optimizer = torch.optim.AdamW(
        [
            {
                "name": AdapterRole.FORWARD_POLICY.value,
                "params": groups.forward,
                "lr": config.adapter_learning_rate,
            },
            {
                "name": AdapterRole.BACKWARD_POLICY.value,
                "params": groups.backward,
                "lr": config.backward_learning_rate or config.adapter_learning_rate,
            },
            {
                "name": "z-head",
                "params": groups.z_head,
                "lr": config.z_learning_rate,
            },
        ],
        weight_decay=0.0,
    )
    if config.stability is not None:
        for group in optimizer.param_groups:
            group["stability_condition"] = config.stability.to_value()
    return optimizer, groups


def prepare_ttb_step(
    *,
    backbone: PolicyBackbone,
    optimizer: torch.optim.Optimizer,
    parameters: PolicyParameterGroups,
    batch: CollectedTrainingBatch,
    snapshot_before: PolicySnapshot,
    temperature_beta: float,
    clock: Callable[[], str],
) -> PreparedTTBStep:
    """Score and backpropagate one exact batch without stepping or publishing."""

    started_at = clock()
    optimizer.zero_grad(set_to_none=True)
    shard = compute_ttb_gradient_shard(
        backbone=backbone,
        parameters=parameters,
        batch=batch,
        positions=tuple(range(len(batch.artifacts))),
        global_batch_size=len(batch.artifacts),
        temperature_beta=temperature_beta,
    )
    return merge_ttb_gradient_shards(
        parameters=parameters,
        batch=batch,
        snapshot_before=snapshot_before,
        shards=(shard,),
        clock=clock,
        started_at=started_at,
    )


def compute_ttb_gradient_shard(
    *,
    backbone: PolicyBackbone,
    parameters: PolicyParameterGroups,
    batch: CollectedTrainingBatch,
    positions: tuple[int, ...],
    global_batch_size: int,
    temperature_beta: float,
) -> TTBGradientShard:
    """Backpropagate a disjoint subset with the global mean-loss scaling."""

    if not positions or len(positions) != len(set(positions)):
        raise ValueError("TTB gradient positions must be non-empty and unique")
    if global_batch_size != len(batch.artifacts):
        raise ValueError("TTB gradient global batch size differs from the sealed batch")
    if any(position < 0 or position >= global_batch_size for position in positions):
        raise ValueError("TTB gradient position lies outside the sealed batch")
    accumulated: dict[str, torch.Tensor] = {}
    artifact_math: list[TTBArtifactMath] = []
    for position in positions:
        shard = compute_ttb_artifact_contribution(
            backbone=backbone,
            parameters=parameters,
            artifact=batch.artifacts[position],
            position=position,
            batch_id=batch.batch_id,
            optimizer_step=batch.optimizer_step,
            policy_snapshot_id=batch.policy_snapshot_id,
            library_version=batch.library_version,
            global_batch_size=global_batch_size,
            temperature_beta=temperature_beta,
        )
        accumulate_ttb_gradients(accumulated, shard.gradients)
        artifact_math.extend(shard.artifacts)
    return TTBGradientShard(
        batch.batch_id, batch.optimizer_step, global_batch_size, tuple(artifact_math), accumulated
    )


def accumulate_ttb_gradients(
    accumulated: dict[str, torch.Tensor],
    contributions: Mapping[str, torch.Tensor],
) -> None:
    for name, contribution in contributions.items():
        if name not in accumulated:
            accumulated[name] = contribution.clone()
        else:
            accumulated[name].add_(contribution)


def require_finite_gradients(gradients: Mapping[str, torch.Tensor]) -> None:
    flags: dict[torch.device, list[torch.Tensor]] = {}
    for tensor in gradients.values():
        flags.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    if not flags or any(not bool(torch.stack(values).all().item()) for values in flags.values()):
        raise ValueError("gradient contribution contains non-finite values")


def compute_ttb_artifact_contribution(
    *,
    backbone: PolicyBackbone,
    parameters: PolicyParameterGroups,
    artifact: RolloutArtifact,
    position: int,
    batch_id: str,
    optimizer_step: int,
    policy_snapshot_id: str,
    library_version: str,
    global_batch_size: int,
    temperature_beta: float,
    prepared_edges: PreparedEdgePlan | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
    provisional: ProvisionalGradientPool | None = None,
) -> TTBGradientShard:
    """One trajectory's exact globally normalized contribution, never an update."""
    if type(global_batch_size) is not int or not 0 <= position < global_batch_size:
        raise ValueError("gradient position lies outside the planned batch")
    if (
        artifact.manifest.policy_snapshot.snapshot_id != policy_snapshot_id
        or artifact.manifest.library_version != library_version
        or artifact.manifest.sampling_coordinate.optimizer_step_or_anchor_ordinal != optimizer_step
    ):
        raise ValueError("gradient artifact belongs to another policy, library or step")
    named = named_ttb_parameters(parameters)
    drain_scoring = getattr(backbone, "drain_scoring_metrics", list)
    drain_scoring()
    forward_version = backbone.adapter_version(AdapterRole.FORWARD_POLICY)
    backward_version = backbone.adapter_version(AdapterRole.BACKWARD_POLICY)
    if forward_version != artifact.manifest.policy_snapshot.forward_adapter_version:
        raise ValueError("gradient scorer uses another forward policy")
    for parameter in named.values():
        parameter.grad = None
    try:
        edges = prepared_edges or prepare_edge_plan(
            backbone.tokenizer, artifact.record, artifact.initial_context.text
        )
        prior_metrics: tuple[dict[str, JsonValue], ...] = ()
        if provisional is not None and position in provisional.partials:
            score, prior_metrics = provisional.finalize(
                position, artifact, edges, ScoringConfig(temperature_beta=temperature_beta)
            )
        else:
            score = backward_trajectory_delta_streaming(
                backbone,
                artifact.record,
                artifact.initial_context.text,
                ScoringConfig(temperature_beta=temperature_beta),
                prepared=edges,
                progress=progress,
            )
        if (
            backbone.adapter_version(AdapterRole.FORWARD_POLICY) != forward_version
            or backbone.adapter_version(AdapterRole.BACKWARD_POLICY) != backward_version
        ):
            raise ValueError("policy changed during forward/backward scoring")
        coefficient = score.gradient_coefficient / global_batch_size
        gradients = {
            name: p.grad.detach().mul(coefficient)
            for name, p in named.items()
            if p.grad is not None
        }
        if set(gradients) != set(named):
            raise RuntimeError("artifact did not produce all trainable gradients")
        item = TTBArtifactMath(
            position,
            artifact.record.trajectory_id,
            materialize_residual(score, raw_reward=artifact.record.reward.value),
            materialize_edge_records(
                score,
                forward_adapter_version=forward_version,
                backward_adapter_version=backward_version,
                batch_id=batch_id,
                policy_snapshot_id=policy_snapshot_id,
                library_version=library_version,
                action_token_ids=tuple(step.action_token_ids for step in artifact.record.steps),
                action_token_counts=tuple(
                    step.action_token_count for step in artifact.record.steps
                ),
            ),
            score.loss,
            (*prior_metrics, *drain_scoring()),
        )
        return TTBGradientShard(batch_id, optimizer_step, global_batch_size, (item,), gradients)
    finally:
        for parameter in named.values():
            parameter.grad = None


def merge_ttb_gradient_shards(
    *,
    parameters: PolicyParameterGroups,
    batch: CollectedTrainingBatch,
    snapshot_before: PolicySnapshot,
    shards: tuple[TTBGradientShard, ...],
    clock: Callable[[], str],
    started_at: str,
) -> PreparedTTBStep:
    """Verify exact coverage, sum globally scaled gradients, and prepare commit."""

    if not shards:
        raise ValueError("TTB gradient merge requires shards")
    if snapshot_before.snapshot_id != batch.policy_snapshot_id:
        raise ValueError("gradient merge policy differs from the collected batch")
    expected_positions = tuple(range(len(batch.artifacts)))
    indexed = {item.position: item for shard in shards for item in shard.artifacts}
    artifact_count = sum(len(shard.artifacts) for shard in shards)
    if tuple(sorted(indexed)) != expected_positions or artifact_count != len(indexed):
        raise ValueError("TTB gradient shards do not cover each trajectory exactly once")
    for shard in shards:
        if (
            shard.batch_id != batch.batch_id
            or shard.optimizer_step != batch.optimizer_step
            or shard.global_batch_size != len(batch.artifacts)
        ):
            raise ValueError("TTB gradient shard belongs to another sealed batch")
        for item in shard.artifacts:
            if item.trajectory_id != batch.artifacts[item.position].record.trajectory_id:
                raise ValueError("TTB gradient shard trajectory identity differs")
    named_parameters = named_ttb_parameters(parameters)
    if any(set(shard.gradients) != set(named_parameters) for shard in shards):
        raise ValueError("TTB gradient shards use different parameter sets")
    for name, parameter in named_parameters.items():
        combined = torch.zeros_like(parameter)
        for shard in shards:
            combined.add_(shard.gradients[name].to(device=parameter.device, dtype=parameter.dtype))
        parameter.grad = combined
    require_finite_gradients({n: p.grad for n, p in named_parameters.items() if p.grad is not None})
    ordered = tuple(indexed[position] for position in expected_positions)
    residual_tuple = tuple(item.residual for item in ordered)
    edges = tuple(edge for item in ordered for edge in item.edges)
    losses = tuple(item.loss for item in ordered)
    stats = TTBBatchStats(
        batch_id=batch.batch_id,
        optimizer_step=batch.optimizer_step,
        library_version=batch.library_version,
        residuals=residual_tuple,
        batch_loss=(
            math.fsum((item.delta / item.horizon) ** 2 for item in residual_tuple)
            / len(residual_tuple)
        ),
        mean_reward=(math.fsum(item.raw_reward for item in residual_tuple) / len(residual_tuple)),
        created_at=clock(),
    )
    # Merge once before diagnostics/calibration. Never normalize each worker's flow.
    from skillev.diagnostics import assemble_batch_flow_input

    assemble_batch_flow_input(
        stats,
        {artifact.record.trajectory_id: artifact.record for artifact in batch.artifacts},
        edges,
    )
    return PreparedTTBStep(
        batch=batch,
        snapshot_before=snapshot_before,
        residuals=residual_tuple,
        edges=edges,
        stats=stats,
        detached_losses=losses,
        gradient_norms=component_gradient_norms(parameters),
        started_at=started_at,
    )


def named_ttb_parameters(
    parameters: PolicyParameterGroups,
) -> dict[str, torch.nn.Parameter]:
    result: dict[str, torch.nn.Parameter] = {}
    for group_name, group in (
        ("forward", parameters.forward),
        ("backward", parameters.backward),
        ("z_head", parameters.z_head),
    ):
        for index, parameter in enumerate(group):
            result[f"{group_name}.{index}"] = parameter
    if len({id(parameter) for parameter in result.values()}) != len(result):
        raise ValueError("TTB trainable parameter groups overlap")
    return result


def apply_optimizer_step(
    *,
    optimizer: torch.optim.Optimizer,
    backbone: PolicyBackbone,
    prepared: PreparedTTBStep,
    clock: Callable[[], str],
    observe_transition: bool = False,
) -> TrainingStepReportValue:
    """Apply the single optimizer transition represented by ``prepared``."""

    observation = (
        OptimizerTransitionObservation.capture(optimizer, backbone.parameter_groups())
        if observe_transition
        else None
    )
    optimizer.step()
    transition = observation.finish() if observation is not None else None
    backbone.mark_policy_update(prepared.batch.optimizer_step)
    norms = prepared.gradient_norms
    return TrainingStepReportValue(
        format="skillev-training-step-report@5"
        if observe_transition
        else "skillev-training-step-report@3",
        optimizer_transition=transition,
        optimizer_step=prepared.batch.optimizer_step,
        batch_id=prepared.batch.batch_id,
        torch_batch_loss=(math.fsum(prepared.detached_losses) / len(prepared.detached_losses)),
        audited_batch_loss=prepared.stats.batch_loss,
        mean_reward=prepared.stats.mean_reward,
        grad_norm_forward=norms.forward,
        grad_norm_backward=norms.backward,
        grad_norm_z=norms.z_head,
        forward_adapter_version=backbone.adapter_version(AdapterRole.FORWARD_POLICY),
        backward_adapter_version=backbone.adapter_version(AdapterRole.BACKWARD_POLICY),
        z_version=backbone.z_version,
        started_at=prepared.started_at,
        completed_at=clock(),
    )


def component_gradient_norms(parameters: PolicyParameterGroups) -> ComponentGradientNorms:
    return ComponentGradientNorms(
        forward=_gradient_l2_norm(parameters.forward),
        backward=_gradient_l2_norm(parameters.backward),
        z_head=_gradient_l2_norm(parameters.z_head),
    )


def _gradient_l2_norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
    values: dict[torch.device, list[torch.Tensor]] = {}
    for parameter in parameters:
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            values.setdefault(grad.device, []).append(grad.to(dtype=torch.float32).pow(2).sum())
    if not values:
        raise RuntimeError("trainable component produced no gradients")
    # Preserve per-tensor float32 sum and Python double fsum, but synchronize once/device.
    squared = [float(v) for group in values.values() for v in torch.stack(group).cpu().tolist()]
    return math.sqrt(math.fsum(squared))


__all__ = [
    "ComponentGradientNorms",
    "PreparedTTBStep",
    "TTBArtifactMath",
    "TTBGradientShard",
    "TTBStepPreparer",
    "apply_optimizer_step",
    "component_gradient_norms",
    "compute_ttb_gradient_shard",
    "create_ttb_optimizer",
    "merge_ttb_gradient_shards",
    "named_ttb_parameters",
    "prepare_ttb_step",
]
