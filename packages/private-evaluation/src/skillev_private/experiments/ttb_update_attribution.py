"""Private, isolated TTB gradient/Adam diagnosis using the production scorer.

The factory must load a NEW model and optimizer at the checkpoint being studied.
There is no production-loop integration, sample filtering, reward rewrite or
four-independent-Adam decomposition. Reports may contain private trajectory IDs
and must not be committed or uploaded as W&B tables.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch

from skillev.contracts import TrajectoryRecord
from skillev.policy import AdapterRole, PolicyBackbone
from skillev.rollout.codec import codec_for_initial_meta
from skillev.runtime.execution import ActionParseStatus
from skillev.training.planning import CollectedTrainingBatch
from skillev.training.step_math import (
    accumulate_ttb_gradients,
    compute_ttb_artifact_contribution,
    named_ttb_parameters,
)
from skillev.training.update_attribution import direct_coefficients

TensorMap = dict[str, torch.Tensor]
GROUPS = ("forward", "backward", "z_head")
CATEGORIES = tuple(
    f"structure-{valid}/terminal-success-{success}"
    for valid in (False, True)
    for success in (False, True)
)


@dataclass(frozen=True)
class ProtocolProbe:
    """An independently declared, synthetically valid action; not a chosen answer."""

    probe_id: str
    prefix_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]


def _norm(values: Mapping[str, torch.Tensor], component: str) -> float:
    return math.sqrt(
        math.fsum(
            value.double().square().sum().item()
            for name, value in values.items()
            if name.startswith(component + ".")
        )
    )


def _norms(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {component: _norm(values, component) for component in GROUPS}


def _comparison(reference: TensorMap, actual: TensorMap) -> dict[str, dict[str, float | None]]:
    errors = {
        name: actual.get(name, torch.zeros_like(value)).cpu() - value.cpu()
        for name, value in reference.items()
    }
    result = {}
    for component in GROUPS:
        scale, error = _norm(reference, component), _norm(errors, component)
        maximum = max(
            (
                value.abs().max().item()
                for name, value in errors.items()
                if name.startswith(component + ".")
            ),
            default=0.0,
        )
        result[component] = {
            "relative_l2": error / scale if scale else None,
            "max_absolute": maximum,
        }
    return result


class _Capture:
    """Bounded CPU accumulators, preserving incoming hook order per trajectory.

    Category sums change addition grouping, so their measured reconstruction
    error is reported separately from the canonical incoming-gradient sum.
    Nothing is returned from a hook: autograd's actual gradient stays untouched.
    """

    def __init__(
        self, named: Mapping[str, torch.nn.Parameter], categories: tuple[str, ...]
    ) -> None:
        self.categories = categories
        self.current: str | None = None
        self.ordered: TensorMap = {}
        self.grouped: dict[str, TensorMap] = {}
        self.handles = [
            p.register_hook(self._hook(name))  # type: ignore[no-untyped-call]
            for name, p in named.items()
        ]

    def _hook(self, name: str) -> Callable[[torch.Tensor], None]:
        def capture(gradient: torch.Tensor) -> None:
            if self.current is None:
                raise RuntimeError("gradient arrived outside a declared scorer stage")
            value = gradient.detach().to(device="cpu", copy=True)
            accumulate_ttb_gradients(self.ordered, {name: value})
            accumulate_ttb_gradients(self.grouped.setdefault(self.current, {}), {name: value})

        return capture

    def progress(self, value: dict[str, object]) -> None:
        if value["stage"] == "z":
            self.current = "partition"
        elif value["stage"] == "score-and-backward":
            self.current = self.categories[cast(int, value["step_index"]) - 1]
        else:
            self.current = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _probe_nll(backbone: PolicyBackbone, panel: tuple[ProtocolProbe, ...]) -> dict[str, float]:
    result = {}
    with torch.no_grad():
        for probe in panel:
            if not probe.action_token_ids or probe.probe_id in result:
                raise ValueError(
                    "protocol panel needs unique IDs and nonempty original action spans"
                )
            values = backbone.score(
                probe.prefix_ids, probe.action_token_ids, AdapterRole.FORWARD_POLICY
            )
            result[probe.probe_id] = -float(values.mean().item())
    return result


def _edge_categories(record: TrajectoryRecord) -> tuple[str, ...]:
    codec = codec_for_initial_meta(record.initial_context.meta)
    return tuple(
        f"structure-{codec.parse(step.action_text).status is ActionParseStatus.VALID}"
        f"/terminal-success-{record.reward.success}"
        for step in record.steps
    )


def attribute_update(
    *,
    replica_factory: Callable[[], tuple[PolicyBackbone, torch.optim.Optimizer]],
    batch: CollectedTrainingBatch,
    expected_batch_size: int,
    temperature_beta: float,
    relative_l2_tolerance: float,
    absolute_tolerance: float,
    protocol_panel: tuple[ProtocolProbe, ...] = (),
    observe: Callable[[str, PolicyBackbone, torch.optim.Optimizer], dict[str, Any]] | None = None,
    progress: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, Any]:
    """Compute all samples, then exactly ONE Adam update on the isolated replica.

    Tolerances are mandatory input, frozen before looking at this result. The
    optional fixed protocol panel measures target-action NLL, not generation
    legality or task accuracy. An optional read-only observer can persist the
    same replica and run a separately declared synthetic panel at both boundaries.
    Observers must not update parameters, optimizer state or gradients.
    """
    trajectory_ids = tuple(a.record.trajectory_id for a in batch.artifacts)
    if (
        len(batch.artifacts) != expected_batch_size
        or expected_batch_size < 1
        or len(set(trajectory_ids)) != expected_batch_size
    ):
        raise ValueError("attribution requires the complete declared batch")
    if any(not math.isfinite(t) or t < 0 for t in (relative_l2_tolerance, absolute_tolerance)):
        raise ValueError("attribution tolerances must be finite and nonnegative")
    backbone, optimizer = replica_factory()
    execution = getattr(backbone, "teacher_forcing_config", None)
    if execution is not None and execution.microbatch_size != 1:
        raise ValueError("per-edge attribution requires the declared single-edge scorer")
    parameters = backbone.parameter_groups()
    named = named_ttb_parameters(parameters)
    before = {name: p.detach().cpu().clone() for name, p in named.items()}
    before_nll = _probe_nll(backbone, protocol_panel)
    observations = {"before": observe("before", backbone, optimizer)} if observe else {}
    optimizer.zero_grad(set_to_none=True)
    total: TensorMap = {}
    ordered: TensorMap = {}
    grouped: dict[str, TensorMap] = {category: {} for category in (*CATEGORIES, "partition")}
    math_rows, checks = [], []
    completed_edges = 0
    total_edges = sum(artifact.record.horizon for artifact in batch.artifacts)
    for position, artifact in enumerate(batch.artifacts):
        categories = _edge_categories(artifact.record)
        capture = _Capture(named, categories)
        try:
            shard = compute_ttb_artifact_contribution(
                backbone=backbone,
                parameters=parameters,
                artifact=artifact,
                position=position,
                batch_id=batch.batch_id,
                optimizer_step=batch.optimizer_step,
                policy_snapshot_id=batch.policy_snapshot_id,
                library_version=batch.library_version,
                global_batch_size=expected_batch_size,
                temperature_beta=temperature_beta,
                progress=capture.progress,
            )
        finally:
            capture.close()
        row = shard.artifacts[0]
        coefficient = 2 * row.residual.delta / (expected_batch_size * row.residual.horizon**2)
        original = dict(shard.gradients)
        reconstructed = {name: value.mul(coefficient) for name, value in capture.ordered.items()}
        checks.append(_comparison(original, reconstructed))
        accumulate_ttb_gradients(total, original)
        accumulate_ttb_gradients(ordered, reconstructed)
        for category, values in capture.grouped.items():
            accumulate_ttb_gradients(
                grouped[category], {name: value.mul(coefficient) for name, value in values.items()}
            )
        math_rows.append(row)
        completed_edges += artifact.record.horizon
        if progress:
            progress(
                {
                    "position": position,
                    "completed_trajectories": position + 1,
                    "completed_edges": completed_edges,
                    "total_trajectories": expected_batch_size,
                    "total_edges": total_edges,
                }
            )
    # Check both original-order and grouped reconstruction before the diagnostic Adam.
    canonical_error = _comparison(total, ordered)
    group_sum: TensorMap = {}
    for values in grouped.values():
        accumulate_ttb_gradients(group_sum, values)
    category_error = _comparison(total, group_sum)
    for check in [*checks, canonical_error, category_error]:
        for value in check.values():
            relative = value["relative_l2"]
            absolute = cast(float, value["max_absolute"])
            if absolute > absolute_tolerance and (
                relative is None or relative > relative_l2_tolerance
            ):
                raise ValueError("attribution reconstruction exceeds the declared numerical limits")
    for name, parameter in named.items():
        parameter.grad = total[name].to(parameter.device)
    optimizer.step()
    backbone.mark_policy_update(batch.optimizer_step)
    if observe:
        observations["after"] = observe("after", backbone, optimizer)
    updates = {name: p.detach().cpu() - before[name] for name, p in named.items()}
    relative_updates = {
        group: _norm(updates, group) / _norm(before, group) if _norm(before, group) else None
        for group in GROUPS
    }
    residuals = tuple(row.residual for row in math_rows)
    edges = direct_coefficients(tuple(a.record for a in batch.artifacts), residuals)
    delta_by_trajectory = {row.trajectory_id: row.delta for row in residuals}
    category_evidence = {}
    for category in CATEGORIES:
        members = [
            edge
            for edge in edges
            if f"structure-{edge.structure_valid}/terminal-success-{edge.terminal_success}"
            == category
        ]
        category_evidence[category] = {
            "edge_count": len(members),
            "trajectory_count": len({edge.trajectory_id for edge in members}),
            "evidence_status": "observed" if members else "no-evidence",
            "edge_weighted_deltas": [delta_by_trajectory[e.trajectory_id] for e in members],
            "gradient_norms": _norms(grouped[category]) if members else None,
        }
    return {
        "format": "skillev-gradient-attribution@1",
        "purpose": "isolated-diagnostic-only",
        "batch_id": batch.batch_id,
        "policy_snapshot_id": batch.policy_snapshot_id,
        "global_batch_size": expected_batch_size,
        "category_evidence": category_evidence,
        "boundary_observations": observations,
        "diagnostic_optimizer_updates": 1,
        "relative_l2_tolerance": relative_l2_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "gradient_norms": _norms(total),
        "adam_update_norms": _norms(updates),
        "relative_parameter_change": relative_updates,
        "canonical_reconstruction_error": canonical_error,
        "category_reassociation_error": category_error,
        "category_gradient_norms": {
            category: _norms(values) for category, values in grouped.items()
        },
        "partition_attribution": "Z belongs to the trajectory, not a fabricated action class",
        "trajectories": [
            {
                "residual": row.residual.to_value(),
                "success": artifact.record.reward.success,
                "gradient_coefficient": 2
                * row.residual.delta
                / (expected_batch_size * row.residual.horizon**2),
            }
            for row, artifact in zip(math_rows, batch.artifacts, strict=True)
        ],
        "edges": [
            {**asdict(edge), "admission_status": None, "execution_status": None} for edge in edges
        ],
        "scored_edges": [edge.to_value() for row in math_rows for edge in row.edges],
        "fixed_action_nll_before": before_nll,
        "fixed_action_nll_after": _probe_nll(backbone, protocol_panel),
        "limitations": (
            "NLL is not generated legality; category gradients are not additive Adam updates"
        ),
    }
