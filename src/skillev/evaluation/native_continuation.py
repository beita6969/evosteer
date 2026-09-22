"""Same-stream native thinking budgets and token-prefix continuation state."""

from __future__ import annotations

from dataclasses import dataclass

from .native_channels import ChannelStatus
from .public_context import PackedPrompt

NATIVE_CONTINUATION_POLICY = "same-owner-token-stream-through-final-stop@2"
BOUNDED_THINKING_POLICY = "same-owner-token-stream-bounded-thinking@1"


def thinking_boundary_at_reserve(
    *,
    enabled: bool,
    continuing: bool,
    remaining: int,
    reserve: int,
    channel_status: ChannelStatus | None,
    previous_boundary: str | None = None,
) -> str | None:
    """Emit only the native newline/end/separator framing, not answer tokens.

    The newlines matter: serving smoke tests reproduced a second </think> when
    an interrupted draft was closed with a bare delimiter mid-sentence.
    """
    if not enabled or not continuing or remaining <= 0:
        return None
    if previous_boundary == "\n":
        return "</think>"
    if previous_boundary == "</think>":
        return "\n\n"
    if previous_boundary == "\n\n":
        return None
    if (
        enabled
        and continuing
        and 0 < remaining <= reserve
        and channel_status is ChannelStatus.REASONING_UNFINISHED
    ):
        return "\n"
    return None


def native_call_allowance(remaining: int, chunk: int, reserve: int) -> int:
    """Uniform admission rule; the reserve is eventually available, never removed."""
    if chunk <= 0 or reserve <= 0:
        raise ValueError("native continuation chunk and final reserve must be positive")
    return min(chunk, remaining - reserve) if remaining > reserve else remaining


@dataclass(frozen=True, slots=True)
class NativeContinuation:
    call_id: str
    prompt: PackedPrompt
    output_token_ids: tuple[int, ...]

    @property
    def input_token_ids(self) -> tuple[int, ...]:
        return self.prompt.input_ids + self.output_token_ids
