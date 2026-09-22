"""Direct TTB coefficients, not a causal decomposition of an Adam update.

No gradients are changed or failures filtered. A positive F derivative lowers
that sampled log probability under independent-coordinate gradient descent;
shared weights, other samples and Adam can change the actual parameter effect.
"""

from dataclasses import dataclass

from skillev.contracts import JsonValue, TrajectoryRecord, TrajectoryResidual
from skillev.rollout.codec import codec_for_initial_meta
from skillev.runtime.execution import ActionParseStatus


@dataclass(frozen=True)
class EdgeAttribution:
    trajectory_id: str
    step_index: int
    terminal_success: bool
    structure_valid: bool
    action_tokens: int
    forward_logprob_derivative: float
    backward_logprob_derivative: float


def direct_coefficients(
    records: tuple[TrajectoryRecord, ...], residuals: tuple[TrajectoryResidual, ...]
) -> tuple[EdgeAttribution, ...]:
    if not records or tuple(r.trajectory_id for r in records) != tuple(
        r.trajectory_id for r in residuals
    ):
        raise ValueError("attribution requires the complete ordered batch")
    if len({r.trajectory_id for r in records}) != len(records):
        raise ValueError("attribution repeats a trajectory")
    results = []
    for record, residual in zip(records, residuals, strict=True):
        codec = codec_for_initial_meta(record.initial_context.meta)
        if len(record.steps) != residual.horizon:
            raise ValueError("attribution horizon differs from the scored trajectory")
        coefficient = 2 * residual.delta / (len(records) * residual.horizon**2)
        for step in record.steps:
            forward = coefficient / step.action_token_count
            results.append(
                EdgeAttribution(
                    record.trajectory_id,
                    step.index,
                    record.reward.success,
                    codec.parse(step.action_text).status is ActionParseStatus.VALID,
                    step.action_token_count,
                    forward,
                    -forward,
                )
            )
    return tuple(results)


def attribution_summary(edges: tuple[EdgeAttribution, ...]) -> dict[str, JsonValue]:
    groups: dict[str, JsonValue] = {}
    for valid in (False, True):
        for success in (False, True):
            members = [
                e for e in edges if e.structure_valid == valid and e.terminal_success == success
            ]
            groups[f"structure-{valid}/terminal-success-{success}"] = {
                "edge_count": len(members),
                "forward_probability_increase_direct_term": sum(
                    e.forward_logprob_derivative < 0 for e in members
                ),
                "forward_probability_decrease_direct_term": sum(
                    e.forward_logprob_derivative > 0 for e in members
                ),
            }
    return {
        "interpretation": "direct-logprob-coordinates-not-Adam-parameter-causality",
        "groups": groups,
    }
