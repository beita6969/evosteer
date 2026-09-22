from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from skillev.scoring import (
    prefix_content_hash,
    render_forward_prefix,
    render_forward_prefix_from_parts,
    render_hindsight_prefix,
    render_hindsight_prefix_from_parts,
    render_reasoning_prefix,
    render_step_zero_reasoning_prefix,
)


def test_renderers_encode_current_condition_and_complete_history_exactly(
    scoring_case: Any,
) -> None:
    initial_text = scoring_case.initial_text
    first, second = scoring_case.record.steps

    forward = render_forward_prefix(initial_text, scoring_case.record.steps, 2)
    hindsight = render_hindsight_prefix(initial_text, scoring_case.record.steps, 2)
    expected_history = (
        "### Step 1\n"
        f"Reasoning:\n{first.reasoning_text}\n"
        f"Action:\n{first.action_text}\n"
        f"Observation:\n{first.observation_text}\n"
    )

    assert forward.text.startswith(
        initial_text + expected_history + "### Step 2\n" + f"Reasoning:\n{second.reasoning_text}\n"
    )
    assert hindsight.text.startswith(
        initial_text
        + expected_history
        + "### Step 2\n"
        + f"Observation:\n{second.observation_text}\n"
    )

    # Both directions retain the complete r/a/o history.
    for historical_value in (
        first.reasoning_text,
        first.action_text,
        first.observation_text,
    ):
        assert historical_value in forward.text
        assert historical_value in hindsight.text

    # Only the current condition differs: r_t for pi, o_t for hindsight phi.
    assert second.reasoning_text in forward.text
    assert second.observation_text not in forward.text
    assert second.observation_text in hindsight.text
    assert second.reasoning_text not in hindsight.text
    assert second.action_text not in forward.text
    assert second.action_text not in hindsight.text
    assert forward.text.endswith("Action:\n")
    assert hindsight.text.endswith("Action:\n")
    assert "Available Actions" in forward.text
    assert "Available Actions" in hindsight.text


def test_sampling_and_scoring_renderers_use_identical_conditions(
    scoring_case: Any,
) -> None:
    initial_text = scoring_case.initial_text
    steps = scoring_case.record.steps

    for step in steps:
        previous_steps = steps[: step.index - 1]
        forward = render_forward_prefix(initial_text, steps, step.index)
        forward_from_parts = render_forward_prefix_from_parts(
            initial_text,
            previous_steps,
            step.index,
            step.reasoning_text,
        )
        hindsight = render_hindsight_prefix(initial_text, steps, step.index)
        hindsight_from_parts = render_hindsight_prefix_from_parts(
            initial_text,
            previous_steps,
            step.index,
            step.observation_text,
        )

        assert forward_from_parts.text == forward.text
        assert hindsight_from_parts.text == hindsight.text


def test_reasoning_prompt_contains_only_complete_history_before_current_step(
    scoring_case: Any,
) -> None:
    initial_text = scoring_case.initial_text
    first, second = scoring_case.record.steps
    prompt = render_reasoning_prefix(initial_text, (first,), 2)
    expected = (
        initial_text
        + "### Step 1\n"
        + f"Reasoning:\n{first.reasoning_text}\n"
        + f"Action:\n{first.action_text}\n"
        + f"Observation:\n{first.observation_text}\n"
        + "### Step 2\n"
    )

    assert prompt.text.startswith(expected)
    assert prompt.text.endswith("Reasoning:\n")
    assert second.reasoning_text not in prompt.text
    assert second.action_text not in prompt.text
    assert second.observation_text not in prompt.text


def test_step_zero_reasoning_prompt_is_strategy_neutral(scoring_case: Any) -> None:
    initial_text = scoring_case.initial_text
    first, second = scoring_case.record.steps
    prompt = render_step_zero_reasoning_prefix(initial_text, (first,), 2)

    assert "strategy comes from" not in prompt.text
    assert "concise reasoning" not in prompt.text
    assert first.reasoning_text in prompt.text
    assert first.action_text in prompt.text
    assert first.observation_text in prompt.text
    assert second.reasoning_text not in prompt.text
    assert prompt.text.endswith("Reasoning:\n")


def test_part_based_current_conditions_do_not_cross_directions(scoring_case: Any) -> None:
    initial_text = scoring_case.initial_text
    first, second = scoring_case.record.steps

    forward = render_forward_prefix_from_parts(
        initial_text,
        (first,),
        2,
        second.reasoning_text,
    )
    hindsight = render_hindsight_prefix_from_parts(
        initial_text,
        (first,),
        2,
        second.observation_text,
    )

    assert second.reasoning_text in forward.text
    assert second.observation_text not in forward.text
    assert second.observation_text in hindsight.text
    assert second.reasoning_text not in hindsight.text


@pytest.mark.parametrize(
    ("previous_steps_slice", "step_index"),
    [
        (slice(0, 0), 2),
        (slice(0, 1), 1),
        (slice(0, 2), 2),
    ],
)
def test_part_based_renderers_reject_missing_or_extra_history(
    scoring_case: Any,
    previous_steps_slice: slice,
    step_index: int,
) -> None:
    previous_steps = scoring_case.record.steps[previous_steps_slice]

    with pytest.raises(ValueError):
        render_reasoning_prefix(scoring_case.initial_text, previous_steps, step_index)
    with pytest.raises(ValueError):
        render_forward_prefix_from_parts(
            scoring_case.initial_text,
            previous_steps,
            step_index,
            "current reasoning",
        )
    with pytest.raises(ValueError):
        render_hindsight_prefix_from_parts(
            scoring_case.initial_text,
            previous_steps,
            step_index,
            "current observation",
        )


def test_part_based_renderers_reject_noncontiguous_history(scoring_case: Any) -> None:
    malformed_history = (replace(scoring_case.record.steps[0], index=2),)

    with pytest.raises(ValueError):
        render_reasoning_prefix(scoring_case.initial_text, malformed_history, 2)
    with pytest.raises(ValueError):
        render_forward_prefix_from_parts(
            scoring_case.initial_text,
            malformed_history,
            2,
            "current reasoning",
        )
    with pytest.raises(ValueError):
        render_hindsight_prefix_from_parts(
            scoring_case.initial_text,
            malformed_history,
            2,
            "current observation",
        )


def test_empty_current_reasoning_preserves_the_canonical_structure(
    make_scoring_case: Any,
) -> None:
    case = make_scoring_case(reasoning_texts=("", "reasontwo thoughttwo"))

    forward = render_forward_prefix(case.initial_text, case.record.steps, 1)
    hindsight = render_hindsight_prefix(case.initial_text, case.record.steps, 1)

    assert "### Step 1\nReasoning:\n\n" in forward.text
    assert "### Step 1\nObservation:\nobserveone resultone\n" in hindsight.text
    assert forward.text.endswith("Action:\n")
    assert hindsight.text.endswith("Action:\n")


def test_prefix_hash_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError):
        prefix_content_hash(kind="sideways", text="prefix")


def test_forward_renderer_rejects_zero_step(scoring_case: Any) -> None:
    with pytest.raises(ValueError):
        render_forward_prefix(
            scoring_case.initial_text,
            scoring_case.record.steps,
            0,
        )


def test_hindsight_renderer_rejects_step_past_horizon(scoring_case: Any) -> None:
    with pytest.raises(ValueError):
        render_hindsight_prefix(
            scoring_case.initial_text,
            scoring_case.record.steps,
            3,
        )
