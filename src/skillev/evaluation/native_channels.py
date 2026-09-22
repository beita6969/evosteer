"""One token-boundary decoder for the native Qwen actor and trusted broker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from skillev.rollout.generator import RolloutTokenizerProtocol


class ChannelStatus(StrEnum):
    COMPLETE = "complete"
    REASONING_UNFINISHED = "reasoning-unfinished"
    FINAL_UNFINISHED = "final-unfinished"
    MALFORMED = "malformed"
    FINAL_EMPTY = "final-empty"


@dataclass(frozen=True, slots=True)
class NativeChannels:
    reasoning_token_ids: tuple[int, ...]
    final_token_ids: tuple[int, ...]
    status: ChannelStatus


def _positions(tokens: tuple[int, ...], marker: tuple[int, ...]) -> list[int]:
    if not marker:
        raise ValueError("native channel markers require the actual tokenizer encoding")
    return [
        index
        for index in range(len(tokens) - len(marker) + 1)
        if tokens[index : index + len(marker)] == marker
    ]


def split_native_channels(
    tokens: tuple[int, ...],
    *,
    enabled: bool,
    tokenizer: RolloutTokenizerProtocol,
    finish_reason: str | None = None,
) -> NativeChannels:
    """Decode the native token path whose thinking-on template opens reasoning.

    An unclosed reasoning stream has no final channel, regardless of any boxed
    draft inside it. Marker encodings come from this run's fixed tokenizer;
    this does not reinterpret a serving API's separate reasoning/content fields.
    """
    if not enabled:
        return NativeChannels(
            (), tokens, ChannelStatus.COMPLETE if tokens else ChannelStatus.FINAL_EMPTY
        )
    end = tuple(tokenizer.encode("</think>"))
    start = tuple(tokenizer.encode("<think>"))
    ends = _positions(tokens, end)
    if not ends:
        return NativeChannels(tokens, (), ChannelStatus.REASONING_UNFINISHED)
    if len(ends) != 1:
        return NativeChannels(tokens, (), ChannelStatus.MALFORMED)
    cut = ends[0]
    reasoning, final = tokens[:cut], tokens[cut + len(end) :]
    starts = _positions(tokens, start)
    if starts not in ([], [0]):
        return NativeChannels(tokens, (), ChannelStatus.MALFORMED)
    if starts:
        reasoning = reasoning[len(start) :]
    status = ChannelStatus.FINAL_EMPTY
    if tokenizer.decode(final).strip():
        # A length boundary is a pause, even if the partial final already looks
        # parseable (e.g. the first digit of a three-digit answer).
        status = (
            ChannelStatus.FINAL_UNFINISHED if finish_reason == "length" else ChannelStatus.COMPLETE
        )
    return NativeChannels(reasoning, final, status)
