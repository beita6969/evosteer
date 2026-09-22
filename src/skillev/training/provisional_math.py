"""Per-trajectory ∇Delta storage; terminal reward is admitted only at finalization.

F/B/Z parameter groups are disjoint. Interleaving trajectories/directions keeps
exactly the original edge-add order inside each parameter group. All live CUDA
state is owned by the calling gradient worker, not the rollout executor.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import cast

import torch

from skillev.contracts import JsonValue, TrajectoryStep
from skillev.policy import AdapterRole, PolicyBackbone, PolicyParameterGroups
from skillev.policy.interface import encode_policy_prompt
from skillev.policy.versions import TrainableVersions
from skillev.rollout import RolloutArtifact
from skillev.rollout.provisional import ProvisionalStep
from skillev.scoring.edge_plan import PreparedEdge, PreparedEdgePlan
from skillev.scoring.objective import (
    ScoringConfig,
    StreamingTrajectoryScore,
    _detached_floats,
    _encoded_initial_context,
    compose_streaming_score,
)
from skillev.scoring.rendering import render_forward_prefix, render_hindsight_prefix

from .gradient_buckets import PackedGradients, pack_gradients
from .step_math import named_ttb_parameters


@dataclass(frozen=True, slots=True)
class PreparedProvisionalStep:
    trajectory_id: str
    task_id: str
    policy_snapshot_id: str
    library_version: str
    versions: TrainableVersions
    query: str
    initial_text: str
    query_ids: tuple[int, ...]
    step: TrajectoryStep
    forward: PreparedEdge
    backward: PreparedEdge

    @property
    def token_cost(self) -> int:
        return self.forward.full_length + self.backward.full_length

    def to_value(self) -> dict[str, object]:
        from dataclasses import asdict

        return {
            **{
                k: getattr(self, k)
                for k in (
                    "trajectory_id",
                    "task_id",
                    "policy_snapshot_id",
                    "library_version",
                    "query",
                    "initial_text",
                )
            },
            "versions": asdict(self.versions),
            "query_ids": list(self.query_ids),
            "step": self.step.to_value(),
            "forward": list(self.forward.prefix_ids),
            "backward": list(self.backward.prefix_ids),
        }

    @classmethod
    def from_value(cls, value: dict[str, object]) -> PreparedProvisionalStep:
        step = TrajectoryStep.from_value(value["step"])

        def ids(key: str) -> tuple[int, ...]:
            raw = value[key]
            if (
                not isinstance(raw, list)
                or not raw
                or any(type(t) is not int or t < 0 for t in raw)
            ):
                raise ValueError("invalid provisional token IDs")
            return tuple(raw)

        return cls(
            **{
                k: cast(str, value[k])
                for k in (
                    "trajectory_id",
                    "task_id",
                    "policy_snapshot_id",
                    "library_version",
                    "query",
                    "initial_text",
                )
            },
            versions=TrainableVersions(**cast(dict[str, str], value["versions"])),
            query_ids=ids("query_ids"),
            step=step,
            forward=PreparedEdge(
                step.index, AdapterRole.FORWARD_POLICY, ids("forward"), step.action_token_ids
            ),
            backward=PreparedEdge(
                step.index, AdapterRole.BACKWARD_POLICY, ids("backward"), step.action_token_ids
            ),
        )


def prepare_provisional_step(
    backbone: PolicyBackbone, value: ProvisionalStep, versions: TrainableVersions
) -> PreparedProvisionalStep:
    tokenizer = backbone.tokenizer
    if value.policy.tokenizer_id != tokenizer.tokenizer_id:
        raise ValueError("provisional tokenizer differs from scoring tokenizer")
    steps = (*value.previous_steps, value.step)
    edges = []
    for role, render, expected in (
        (AdapterRole.FORWARD_POLICY, render_forward_prefix, value.step.forward_prefix_hash),
        (AdapterRole.BACKWARD_POLICY, render_hindsight_prefix, value.step.hindsight_prefix_hash),
    ):
        rendered = render(value.initial_text, steps, value.step.index)
        if rendered.prefix_hash != expected:
            raise ValueError("materialized provisional scoring prefix differs")
        encoded = value.forward_input if role is AdapterRole.FORWARD_POLICY else None
        if encoded is None:
            encoded = encode_policy_prompt(
                tokenizer,
                rendered.text,
                initial_text=value.initial_text,
                window=value.input_window,
            )
        if (
            not encoded.ids
            or any(type(token) is not int or token < 0 for token in encoded.ids)
            or encoded.original_tokens < len(encoded.ids)
            or (value.input_window is not None and len(encoded.ids) > value.input_window.max_tokens)
        ):
            raise ValueError("invalid admitted provisional input")
        edges.append(
            PreparedEdge(
                value.step.index,
                role,
                encoded.ids,
                value.step.action_token_ids,
            )
        )
    query = tuple(tokenizer.encode(value.query))
    if not query:
        raise ValueError("empty provisional Z query")
    return PreparedProvisionalStep(
        value.trajectory_id,
        value.task_id,
        value.policy.snapshot_id,
        value.library_version,
        versions,
        value.query,
        value.initial_text,
        query,
        value.step,
        *edges,
    )


@dataclass(slots=True)
class _Partial:
    first: PreparedProvisionalStep
    gradients: PackedGradients | dict[str, torch.Tensor] | None = None
    steps: list[PreparedProvisionalStep] = field(default_factory=list)
    scalars: list[torch.Tensor] = field(default_factory=list)
    metrics: list[dict[str, JsonValue]] = field(default_factory=list)


class ProvisionalGradientPool:
    def __init__(
        self,
        backbone: PolicyBackbone,
        parameters: PolicyParameterGroups,
        *,
        device_budget_bytes: int = 0,
    ) -> None:
        if type(device_budget_bytes) is not int or device_budget_bytes < 0:
            raise ValueError("device gradient budget must be a non-negative byte count")
        self.device_budget_bytes = device_budget_bytes
        self._resident: OrderedDict[int, int] = OrderedDict()
        self.device_bytes = self.peak_device_bytes = self.spilled_bytes = 0
        self.backbone = backbone
        self.named = named_ttb_parameters(parameters)  # Also rejects overlapping F/B/Z groups.
        self.partials: dict[int, _Partial] = {}
        self.versions = TrainableVersions.from_backbone(backbone)
        config = getattr(backbone, "teacher_forcing_config", None)
        if config is not None and config.microbatch_size != 1:
            raise ValueError("provisional gradients require the qualified single-edge scorer")

    def clear(self) -> None:
        self.partials.clear()
        self._resident.clear()
        self.device_bytes = 0
        for parameter in self.named.values():
            parameter.grad = None

    def advance(
        self,
        position: int,
        edge: PreparedProvisionalStep,
        progress: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if (
            self.versions != edge.versions
            or TrainableVersions.from_backbone(self.backbone) != self.versions
        ):
            raise ValueError("provisional scorer policy/Z version changed")
        partial = self.partials.setdefault(position, _Partial(edge))
        first = partial.first
        if (
            any(
                getattr(first, k) != getattr(edge, k)
                for k in (
                    "trajectory_id",
                    "task_id",
                    "policy_snapshot_id",
                    "library_version",
                    "query",
                    "initial_text",
                    "query_ids",
                    "versions",
                )
            )
            or edge.step.index != len(partial.steps) + 1
        ):
            raise ValueError(
                "provisional step is repeated, missing, or belongs to another trajectory"
            )
        self.device_bytes -= self._resident.pop(position, 0)
        restored = self._restore(partial.gradients)
        for name, parameter in self.named.items():
            parameter.grad = restored.get(name)
        drain = getattr(self.backbone, "drain_scoring_metrics", list)
        drain()
        try:
            if not partial.steps:
                z = self.backbone.z_value(edge.query_ids)
                partial.scalars.append(z.detach())
                z.backward()  # type: ignore[no-untyped-call]
            session = getattr(self.backbone, "scoring_session", lambda role: nullcontext())
            for prepared in (edge.forward, edge.backward):
                if progress is not None:
                    progress(
                        {
                            "stage": "provisional-score-and-backward",
                            "step_index": edge.step.index,
                            "direction": prepared.role.value,
                            "prefix_tokens": len(prepared.prefix_ids),
                            "action_tokens": len(prepared.action_ids),
                        }
                    )
                with session(prepared.role):
                    scores = self.backbone.score(
                        prepared.prefix_ids, prepared.action_ids, prepared.role
                    )
                    if scores.ndim != 1 or len(scores) != len(prepared.action_ids):
                        raise ValueError("provisional score does not cover exact action tokens")
                    value = scores.sum() / len(prepared.action_ids)
                    partial.scalars.append(value.detach())
                    signed = value if prepared.role is AdapterRole.FORWARD_POLICY else -value
                    signed.backward()  # type: ignore[no-untyped-call]
                    del signed, value, scores
                    finish = getattr(self.backbone, "finish_edge_backward_profile", None)
                    if finish is not None:
                        finish()
                    if progress is not None:
                        progress(
                            {
                                "stage": "edge-backward-returned",
                                "step_index": edge.step.index,
                                "direction": prepared.role.value,
                            }
                        )
            if TrainableVersions.from_backbone(self.backbone) != self.versions:
                raise ValueError("policy changed while scoring a provisional edge")
            gradients = {n: p.grad for n, p in self.named.items() if p.grad is not None}
            if set(gradients) != set(self.named):
                raise RuntimeError("provisional edge did not produce every trainable gradient")
            self._store(position, gradients)
            partial.steps.append(edge)
            partial.metrics.extend(drain())
            if progress is not None:
                progress({"provisional_storage": self.storage_metrics()})
        finally:
            for parameter in self.named.values():
                parameter.grad = None

    def finalize(
        self,
        position: int,
        artifact: RolloutArtifact,
        plan: PreparedEdgePlan,
        config: ScoringConfig,
    ) -> tuple[StreamingTrajectoryScore, tuple[dict[str, JsonValue], ...]]:
        partial = self.partials[position]
        first, record = partial.first, artifact.record
        _encoded_initial_context(self.backbone.tokenizer, record, artifact.initial_context.text)
        if (
            record.trajectory_id != first.trajectory_id
            or artifact.manifest.task_id != first.task_id
            or artifact.manifest.policy_snapshot.snapshot_id != first.policy_snapshot_id
            or artifact.manifest.library_version != first.library_version
            or artifact.initial_context.text != first.initial_text
            or record.initial_context.query != first.query
            or len(partial.steps) != record.horizon
            or TrainableVersions.from_backbone(self.backbone) != first.versions
        ):
            raise ValueError("terminal artifact does not match provisional trajectory")
        for provisional, step in zip(partial.steps, record.steps, strict=True):
            if (
                provisional.step != step
                or provisional.forward != plan.edge(step.index, AdapterRole.FORWARD_POLICY)
                or provisional.backward != plan.edge(step.index, AdapterRole.BACKWARD_POLICY)
            ):
                raise ValueError("terminal artifact changed a provisional edge")
        values = _detached_floats(partial.scalars)
        score = compose_streaming_score(record, config, values[0], values[1::2], values[2::2])
        assert partial.gradients is not None
        self.device_bytes -= self._resident.pop(position, 0)
        restored = self._restore(partial.gradients)
        for name, parameter in self.named.items():
            parameter.grad = restored[name]
        del self.partials[position]
        return score, tuple(partial.metrics)

    def _restore(
        self, gradients: PackedGradients | dict[str, torch.Tensor] | None
    ) -> dict[str, torch.Tensor]:
        if gradients is None:
            return {}
        return (
            gradients.on_device(self.named) if isinstance(gradients, PackedGradients) else gradients
        )

    def _store(self, position: int, gradients: dict[str, torch.Tensor]) -> None:
        size = sum(g.numel() * g.element_size() for g in gradients.values())
        partial = self.partials[position]
        if size > self.device_budget_bytes:
            partial.gradients = pack_gradients(gradients)
            self.spilled_bytes += size
            return
        while self.device_bytes + size > self.device_budget_bytes:
            old, count = self._resident.popitem(last=False)
            victim = self.partials[old]
            assert isinstance(victim.gradients, dict)
            victim.gradients = pack_gradients(victim.gradients)
            self.device_bytes -= count
            self.spilled_bytes += count
        # Own these tensors after .grad is cleared. Neither another trajectory
        # nor the final loss coefficient mutates this unscaled accumulation.
        partial.gradients = gradients
        self._resident[position] = size
        self.device_bytes += size
        self.peak_device_bytes = max(self.peak_device_bytes, self.device_bytes)

    def storage_metrics(self) -> dict[str, int]:
        return {
            "device_budget_bytes": self.device_budget_bytes,
            "resident_bytes": self.device_bytes,
            "peak_resident_bytes": self.peak_device_bytes,
            "cpu_spilled_bytes": self.spilled_bytes,
        }
