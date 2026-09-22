"""Thinking-aware context selection for interactive public transition memory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .client import ChatTokenCounter

TurnT = TypeVar("TurnT")


@dataclass(frozen=True, slots=True)
class ContextBudget:
    context_length: int
    output_reserve: int
    enable_thinking: bool

    def __post_init__(self) -> None:
        if self.context_length <= 0 or self.output_reserve <= 0:
            raise ValueError("context length and output reserve must be positive")
        if self.output_reserve >= self.context_length:
            raise ValueError("output reserve must be smaller than the context length")


def newest_contiguous_history(
    *,
    render: Callable[[tuple[TurnT, ...]], tuple[dict[str, str], ...]],
    history: tuple[TurnT, ...],
    counter: ChatTokenCounter,
    budget: ContextBudget,
) -> tuple[TurnT, ...]:
    """Keep all history when possible, otherwise the longest complete newest suffix.

    Callers place cumulative structured memory in ``render(())``.  It therefore
    survives genuine context overflow while raw transitions are removed only
    from the oldest edge; a newer transition is never skipped to retain an
    older, shorter one.
    """

    base_tokens = counter.count(render(()), enable_thinking=budget.enable_thinking)
    if base_tokens + budget.output_reserve > budget.context_length:
        raise ValueError("interactive prompt base exceeds the context budget")
    if not history:
        return ()
    full_tokens = counter.count(render(history), enable_thinking=budget.enable_thinking)
    if full_tokens + budget.output_reserve <= budget.context_length:
        return history
    for start in range(1, len(history) + 1):
        candidate = history[start:]
        tokens = counter.count(render(candidate), enable_thinking=budget.enable_thinking)
        if tokens + budget.output_reserve <= budget.context_length:
            return candidate
    return ()


__all__ = ["ContextBudget", "newest_contiguous_history"]
