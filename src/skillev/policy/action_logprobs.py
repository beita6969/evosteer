"""Per-original-token full-vocabulary probabilities, without a K-by-V FP32 save."""

from __future__ import annotations

from typing import Any, cast

import torch


class _ChunkedTargetLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(logits, targets)
        result = torch.empty(targets.shape, dtype=torch.float32, device=logits.device)
        for start in range(0, len(targets), 16):
            stop = min(start + 16, len(targets))
            values = logits[start:stop].float()
            # Keep the reference full-vocabulary normalization and target IDs.
            result[start:stop] = (
                torch.log_softmax(values, -1).gather(-1, targets[start:stop, None]).squeeze(-1)
            )
        return result

    @staticmethod
    def backward(ctx: Any, upstream: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits, targets = ctx.saved_tensors
        gradient = torch.empty_like(logits)
        for start in range(0, len(targets), 16):
            stop = min(start + 16, len(targets))
            values = -torch.softmax(logits[start:stop].float(), -1)
            values.scatter_add_(
                -1,
                targets[start:stop, None],
                torch.ones_like(targets[start:stop, None], dtype=values.dtype),
            )
            # Arbitrary signed per-token upstream, including the TTB residual
            # coefficient and action-specific K; never substitute a token mean.
            values.mul_(upstream[start:stop, None])
            gradient[start:stop] = values.to(logits.dtype)
        return gradient, None


def action_token_logprobs(
    logits: torch.Tensor, targets: torch.Tensor, *, implementation: str = "reference"
) -> torch.Tensor:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("action probabilities require one target per original action position")
    if implementation == "reference":
        return torch.log_softmax(logits.float(), -1).gather(-1, targets[:, None]).squeeze(-1)
    if implementation == "chunked-target@1":
        return cast(torch.Tensor, _ChunkedTargetLogprobs.apply(logits, targets))  # type: ignore[no-untyped-call]
    raise ValueError("unsupported action log-probability execution implementation")
