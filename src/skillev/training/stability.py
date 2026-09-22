"""Optional reference-policy optimization condition, not the original bare TTB.

The regularizer is categorical KL(current F || frozen reference F) averaged
by token within edge, edge within trajectory, then over the full batch. No
sampled log-ratio surrogate, no worker averaging, no effect on B or Z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from skillev.contracts import JsonValue

if TYPE_CHECKING:
    import torch

    from skillev.policy import PolicyBackbone, PolicyParameterGroups
    from skillev.scoring.edge_plan import PreparedEdge

    from .planning import CollectedTrainingBatch


@dataclass(frozen=True, slots=True)
class PolicyStabilityConfig:
    reference_id: str | None = None
    coefficient: float = 0.0
    forward_max_norm: float | None = None
    backward_max_norm: float | None = None
    z_max_norm: float | None = None
    format: str = "reference-categorical-kl-group-clip@1"

    def __post_init__(self) -> None:
        if self.format != "reference-categorical-kl-group-clip@1":
            raise ValueError("unsupported stability condition")
        if (
            isinstance(self.coefficient, bool)
            or not math.isfinite(self.coefficient)
            or self.coefficient < 0
        ):
            raise ValueError("stability coefficient must be finite and nonnegative")
        if self.coefficient > 0 and (
            not isinstance(self.reference_id, str) or not self.reference_id.strip()
        ):
            raise ValueError("reference identity required for nonzero KL")
        if self.coefficient == 0 and self.reference_id is not None:
            raise ValueError("disabled KL must not imply an active reference")
        for value in (self.forward_max_norm, self.backward_max_norm, self.z_max_norm):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError("clip threshold must be positive or disabled")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "format": self.format,
            "reference_id": self.reference_id,
            "coefficient": self.coefficient,
            "forward_max_norm": self.forward_max_norm,
            "backward_max_norm": self.backward_max_norm,
            "z_max_norm": self.z_max_norm,
        }

    @classmethod
    def from_value(cls, value: object) -> PolicyStabilityConfig:
        if not isinstance(value, dict):
            raise TypeError("stability config must be an object")
        return cls(**value)


def categorical_reference_kl(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    support: torch.Tensor | None = None,
    vocabulary_chunk_size: int = 4096,
) -> torch.Tensor:
    """Exact per-context KL; optional shared public support, never token selection."""
    import torch

    if logits.shape != reference_logits.shape or logits.ndim < 2:
        raise ValueError("categorical logits shapes must match")
    if type(vocabulary_chunk_size) is not int or vocabulary_chunk_size < 1:
        raise ValueError("invalid categorical chunk size")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    current = logits.to(dtype)
    reference = reference_logits.detach().to(device=logits.device, dtype=dtype)
    if not torch.isfinite(current).all() or not torch.isfinite(reference).all():
        raise ValueError("nonfinite unmasked categorical logits")
    if support is not None:
        if (
            support.shape != logits.shape
            or support.dtype is not torch.bool
            or not support.any(dim=-1).all()
        ):
            raise ValueError("both policies require one identical nonempty public support")
        support = support.to(logits.device)
        current = current.masked_fill(~support, -torch.inf)
        reference = reference.masked_fill(~support, -torch.inf)
    current_normalizer = current.logsumexp(-1, keepdim=True)
    reference_normalizer = reference.logsumexp(-1, keepdim=True)
    result = current.new_zeros(current.shape[:-1])
    for start in range(0, current.shape[-1], vocabulary_chunk_size):
        stop = start + vocabulary_chunk_size
        lp = current[..., start:stop] - current_normalizer
        lr = reference[..., start:stop] - reference_normalizer
        if support is not None:
            allowed = support[..., start:stop]
            # Finite zero values off support prevent 0 * nan and nan gradients.
            lp = lp.masked_fill(~allowed, 0)
            lr = lr.masked_fill(~allowed, 0)
            probability = lp.exp().masked_fill(~allowed, 0)
        else:
            probability = lp.exp()
        result = result + (probability * (lp - lr)).sum(-1)
    return result


class ReferencePolicyScorer(Protocol):
    """An exact-logit scorer, never an answer generator or a task consultant."""

    reference_id: str

    def logits(self, edge: PreparedEdge) -> tuple[torch.Tensor, torch.Tensor]:
        """Return differentiable current F and detached frozen F logits, [K,V]."""
        ...


def add_reference_gradients(
    *,
    batch: CollectedTrainingBatch,
    backbone: PolicyBackbone,
    config: PolicyStabilityConfig,
    scorer: ReferencePolicyScorer | None,
) -> float:
    from skillev.policy import AdapterRole
    from skillev.scoring.edge_plan import prepare_edge_plan

    if config.coefficient == 0:
        return 0.0
    if scorer is None or scorer.reference_id != config.reference_id:
        raise ValueError("frozen reference scorer does not match the declared condition")
    if not batch.artifacts:
        raise ValueError("reference KL requires the complete nonempty batch")
    loss = 0.0
    for artifact in batch.artifacts:
        plan = prepare_edge_plan(backbone.tokenizer, artifact.record, artifact.initial_context.text)
        for edge in plan.edges:
            if edge.role is not AdapterRole.FORWARD_POLICY:
                continue
            current, reference = scorer.logits(edge)
            if current.ndim != 2 or current.shape[0] != len(edge.action_ids):
                raise ValueError(
                    "reference scorer does not cover exactly the sampled action contexts"
                )
            term = categorical_reference_kl(current, reference).mean() * (
                config.coefficient / (len(batch.artifacts) * artifact.record.horizon)
            )
            term.backward()  # type: ignore[no-untyped-call]
            loss += float(term.detach())
    return loss


def clip_full_batch_groups(
    parameters: PolicyParameterGroups, config: PolicyStabilityConfig
) -> dict[str, JsonValue]:
    import torch

    report: dict[str, JsonValue] = {}
    for name, values, maximum in (
        ("forward", parameters.forward, config.forward_max_norm),
        ("backward", parameters.backward, config.backward_max_norm),
        ("z", parameters.z_head, config.z_max_norm),
    ):
        norm = math.sqrt(
            math.fsum(
                float(p.grad.detach().double().square().sum()) for p in values if p.grad is not None
            )
        )
        if not math.isfinite(norm):
            raise ValueError("nonfinite full-batch gradient norm")
        scale = 1.0 if maximum is None or norm <= maximum else maximum / norm
        if scale < 1:
            with torch.no_grad():
                for parameter in values:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
        report[name] = {"pre_clip_norm": norm, "scale": scale, "threshold": maximum}
    return report
