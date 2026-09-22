"""Independent scalar diagnostic oracle, not an alternative production trainer."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TTBReference:
    forward: float
    backward: float
    log_reward: float
    delta: float
    loss: float
    z_derivative: float
    forward_sum_derivatives: tuple[float, ...]
    backward_sum_derivatives: tuple[float, ...]


def ttb_reference(
    *,
    log_z: float,
    forward_sums: tuple[float, ...],
    backward_sums: tuple[float, ...],
    token_counts: tuple[int, ...],
    reward: float,
    epsilon: float,
    beta: float,
    batch_size: int = 1,
) -> TTBReference:
    if not token_counts or not len(token_counts) == len(forward_sums) == len(backward_sums):
        raise ValueError("incomplete edge coverage")
    if (
        any(type(count) is not int or count <= 0 for count in token_counts)
        or type(batch_size) is not int
        or batch_size <= 0
    ):
        raise ValueError("counts must be positive integers")
    if (
        not all(
            math.isfinite(v) for v in (log_z, reward, epsilon, beta, *forward_sums, *backward_sums)
        )
        or not 0 <= reward <= 1
        or epsilon <= 0
        or beta <= 0
    ):
        raise ValueError("invalid finite reward condition")
    forward = math.fsum(
        value / count for value, count in zip(forward_sums, token_counts, strict=True)
    )
    backward = math.fsum(
        value / count for value, count in zip(backward_sums, token_counts, strict=True)
    )
    log_reward = beta * math.log(reward + epsilon)
    delta = math.fsum((log_z, forward, -log_reward, -backward))
    coefficient = 2 * delta / (len(token_counts) ** 2 * batch_size)
    return TTBReference(
        forward,
        backward,
        log_reward,
        delta,
        (delta / len(token_counts)) ** 2 / batch_size,
        coefficient,
        tuple(coefficient / count for count in token_counts),
        tuple(-coefficient / count for count in token_counts),
    )
