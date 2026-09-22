"""Regression coverage for unambiguous Step-0 binding construction."""

from __future__ import annotations

import pytest

from skillev.evaluation.step0_types import StepZeroActionMode, StepZeroTaskBinding


def binding(*, skill_instruction_token_budget: int) -> StepZeroTaskBinding:
    return StepZeroTaskBinding(
        task_id="synthetic-task",
        benchmark_id="aime-2026",
        task_family="public-task",
        panel_index=7,
        action_mode=StepZeroActionMode.COMPLETION,
        skill_instruction_token_budget=skill_instruction_token_budget,
    )


@pytest.mark.parametrize("budget", [0, 64, 256, 1024])
def test_skill_instruction_budget_is_preserved_including_explicit_zero(budget: int) -> None:
    assert binding(skill_instruction_token_budget=budget).skill_instruction_token_budget == budget


def test_binding_constructor_is_keyword_only_and_panel_position_is_independent() -> None:
    with pytest.raises(TypeError):
        StepZeroTaskBinding(
            "synthetic-task", "aime-2026", "public-task", 7, StepZeroActionMode.COMPLETION
        )

    selected = binding(skill_instruction_token_budget=64)
    assert selected.panel_index == 7
    assert selected.skill_instruction_token_budget == 64


def test_negative_skill_instruction_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        binding(skill_instruction_token_budget=-1)
