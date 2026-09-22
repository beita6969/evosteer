from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from skillev.evaluation.direct_baseline.context_budget import (
    ContextBudget,
    newest_contiguous_history,
)


@dataclass
class _Counter:
    tokenizer_id: str = "fake"
    thinking_modes: list[bool] = field(default_factory=list)

    def count(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        enable_thinking: bool,
    ) -> int:
        self.thinking_modes.append(enable_thinking)
        return sum(len(item["content"]) for item in messages)


def _render(history: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    return ({"role": "user", "content": "base" + "".join(history)},)


def test_newest_contiguous_history_keeps_only_a_suffix() -> None:
    counter = _Counter()
    retained = newest_contiguous_history(
        render=_render,
        history=("1111", "22", "3"),
        counter=counter,
        budget=ContextBudget(context_length=10, output_reserve=2, enable_thinking=True),
    )
    assert retained == ("22", "3")
    assert set(counter.thinking_modes) == {True}


def test_base_prompt_must_fit_before_history_is_considered() -> None:
    with pytest.raises(ValueError, match="base"):
        newest_contiguous_history(
            render=_render,
            history=("x",),
            counter=_Counter(),
            budget=ContextBudget(context_length=5, output_reserve=2, enable_thinking=False),
        )
