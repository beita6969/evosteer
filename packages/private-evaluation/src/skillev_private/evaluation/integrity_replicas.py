"""Balance independent model requests, not task difficulty or evaluation outcomes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


class ReplicaLoadBalancer:
    policy_id = "least-inflight-round-robin-ties@1"

    def __init__(self, size: int, *, context_lengths: tuple[int, ...] | None = None) -> None:
        if size < 1:
            raise ValueError("inference requires at least one replica")
        if context_lengths is not None and (
            len(context_lengths) != size
            or any(type(limit) is not int or limit < 1 for limit in context_lengths)
        ):
            raise ValueError("each replica needs its positive observed context capacity")
        self.context_lengths = context_lengths
        if context_lengths is not None:
            self.policy_id = "context-fit-least-inflight-round-robin-ties@2"
        self.inflight = [0] * size
        self.next_index = 0

    @contextmanager
    def acquire(self, *, required_context_tokens: int = 0) -> Iterator[int]:
        # Reservation happens without an await on the coordinator event loop.
        # Only request capacity and current load influence selection, not task
        # identity, answer content, difficulty or an evaluation outcome.
        if type(required_context_tokens) is not int or required_context_tokens < 0:
            raise ValueError("required request capacity must be nonnegative")
        size = len(self.inflight)
        eligible = [
            index
            for index in range(size)
            if self.context_lengths is None
            or self.context_lengths[index] >= required_context_tokens
        ]
        if not eligible:
            raise ValueError("no serving replica can fit the unchanged request budget")
        index = min(
            eligible,
            key=lambda i: (self.inflight[i], (i - self.next_index) % size),
        )
        self.inflight[index] += 1
        self.next_index = (index + 1) % size
        try:
            yield index
        finally:
            self.inflight[index] -= 1
