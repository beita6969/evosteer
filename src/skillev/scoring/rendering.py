"""Canonical text rendering for trajectory-balance conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

from skillev.contracts.action_wire import NATIVE_CARRIER_WIRES, NATIVE_TOOL_WIRES
from skillev.contracts.canonical import stable_hash
from skillev.contracts.ttb_trajectory import TrajectoryStep
from skillev.policy.typed_history import render_typed_phase

TEMPLATE_VERSION: Final = "ttb-render@5"

_REASONING_GENERATION_REMINDER: Final = (
    "Reasoning pass: reason about the task and current public state.\n"
)
_STEP_ZERO_REASONING_GENERATION_REMINDER: Final = (
    "Reasoning pass: solve the public task using any helpful approach.\n"
)
_ACTION_GENERATION_REMINDER: Final = (
    "Action pass: send one JSON action using an interface in Available Actions.\n"
)

PrefixKind: TypeAlias = Literal["forward", "hindsight"]

_PREFIX_KINDS: Final = frozenset({"forward", "hindsight"})


def _require_prefix_kind(kind: str) -> PrefixKind:
    if kind not in _PREFIX_KINDS:
        raise ValueError("prefix kind must be 'forward' or 'hindsight'")
    return cast(PrefixKind, kind)


def _step_at(
    steps: tuple[TrajectoryStep, ...],
    step_index: int,
) -> TrajectoryStep:
    if type(step_index) is not int or not 1 <= step_index <= len(steps):
        raise ValueError("step_index must use one-based indexing within steps")
    for expected_index, step in enumerate(steps[:step_index], start=1):
        if step.index != expected_index:
            raise ValueError("steps must have contiguous one-based indices")
    return steps[step_index - 1]


def _require_previous_steps(
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
) -> None:
    if type(step_index) is not int or step_index < 1:
        raise ValueError("step_index must use one-based indexing")
    if len(previous_steps) != step_index - 1:
        raise ValueError("previous_steps must contain exactly the history before step_index")
    for expected_index, step in enumerate(previous_steps, start=1):
        if step.index != expected_index:
            raise ValueError("previous_steps must have contiguous one-based indices")


def _render_history(
    previous_steps: tuple[TrajectoryStep, ...],
) -> str:
    return "".join(
        (
            f"### Step {history_index}\n"
            f"Reasoning:\n{step.reasoning_text}\n"
            f"Action:\n{step.action_text}\n"
            f"Observation:\n{step.observation_text}\n"
        )
        for history_index, step in enumerate(previous_steps, start=1)
    )


@dataclass(frozen=True, slots=True)
class RenderedPrefix:
    """One canonical rendered condition and its content hash."""

    kind: PrefixKind
    step_index: int
    text: str
    prefix_hash: str

    def __post_init__(self) -> None:
        _require_prefix_kind(self.kind)
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("step_index must use one-based indexing")


@dataclass(frozen=True, slots=True)
class RenderedReasoningPrompt:
    """The canonical prompt for the reasoning generation pass."""

    step_index: int
    text: str
    prompt_hash: str

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("step_index must use one-based indexing")


def assembled_context_hash(initial_text: str) -> str:
    """Return the versioned content hash promised by an initial context."""

    return stable_hash(
        {
            "initial_text": initial_text,
            "template_version": TEMPLATE_VERSION,
        }
    )


def prefix_content_hash(*, kind: str, text: str) -> str:
    """Return the versioned content hash for one rendered prefix."""

    checked_kind = _require_prefix_kind(kind)
    return stable_hash(
        {
            "kind": checked_kind,
            "template_version": TEMPLATE_VERSION,
            "text": text,
        }
    )


def render_reasoning_prefix(
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
) -> RenderedReasoningPrompt:
    """Render the history-only prompt used to generate the current reasoning."""

    return _render_reasoning_prefix(
        initial_text,
        previous_steps,
        step_index,
        reminder=_REASONING_GENERATION_REMINDER,
    )


def render_step_zero_reasoning_prefix(
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
) -> RenderedReasoningPrompt:
    """Render a strategy-neutral reasoning phase for seeded Step-0 inference.

    This adapter-level suffix identifies only the phase, without shortening
    reasoning or requiring a retrieved skill's solution strategy.
    """

    return _render_reasoning_prefix(
        initial_text,
        previous_steps,
        step_index,
        reminder=_STEP_ZERO_REASONING_GENERATION_REMINDER,
    )


def _render_reasoning_prefix(
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
    *,
    reminder: str,
) -> RenderedReasoningPrompt:
    """Render one reasoning prefix from an explicit phase-only reminder."""

    _require_previous_steps(previous_steps, step_index)
    text = render_typed_phase(
        initial_text,
        previous_steps,
        step_index,
        phase="reasoning",
        instruction=reminder,
    ) or (
        initial_text
        + _render_history(previous_steps)
        + f"### Step {step_index}\n"
        + reminder
        + "Reasoning:\n"
    )
    return RenderedReasoningPrompt(
        step_index=step_index,
        text=text,
        prompt_hash=stable_hash(
            {
                "step_index": step_index,
                "template_version": TEMPLATE_VERSION,
                "text": text,
            }
        ),
    )


def _action_reminder(initial_text: str) -> str:
    from skillev.policy.phase_context import PhaseContextSpec, native_action_instruction

    spec, _ = PhaseContextSpec.split(initial_text)
    if spec is not None and spec.action_wire in NATIVE_CARRIER_WIRES:
        return native_action_instruction(spec) + "\n"
    if spec is not None and spec.action_wire in NATIVE_TOOL_WIRES:
        return "Action pass: emit exactly one native tool call from the declared tools.\n"
    return _ACTION_GENERATION_REMINDER


def render_forward_prefix_from_parts(
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
    reasoning_text: str,
) -> RenderedPrefix:
    """Render the forward condition before the current action exists."""

    _require_previous_steps(previous_steps, step_index)
    text = render_typed_phase(
        initial_text,
        previous_steps,
        step_index,
        phase="forward",
        instruction=_action_reminder(initial_text),
        current_text=reasoning_text,
    ) or (
        initial_text
        + _render_history(previous_steps)
        + f"### Step {step_index}\n"
        + f"Reasoning:\n{reasoning_text}\n"
        + _action_reminder(initial_text)
        + "Action:\n"
    )
    return RenderedPrefix(
        kind="forward",
        step_index=step_index,
        text=text,
        prefix_hash=prefix_content_hash(kind="forward", text=text),
    )


def render_hindsight_prefix_from_parts(
    initial_text: str,
    previous_steps: tuple[TrajectoryStep, ...],
    step_index: int,
    observation_text: str,
) -> RenderedPrefix:
    """Render the hindsight condition without consulting current reasoning."""

    _require_previous_steps(previous_steps, step_index)
    text = render_typed_phase(
        initial_text,
        previous_steps,
        step_index,
        phase="hindsight",
        instruction=_action_reminder(initial_text),
        current_text=observation_text,
    ) or (
        initial_text
        + _render_history(previous_steps)
        + f"### Step {step_index}\n"
        + f"Observation:\n{observation_text}\n"
        + _action_reminder(initial_text)
        + "Action:\n"
    )
    return RenderedPrefix(
        kind="hindsight",
        step_index=step_index,
        text=text,
        prefix_hash=prefix_content_hash(kind="hindsight", text=text),
    )


def render_forward_prefix(
    initial_text: str,
    steps: tuple[TrajectoryStep, ...],
    step_index: int,
) -> RenderedPrefix:
    """Render history plus the current reasoning as the forward condition."""

    current_step = _step_at(steps, step_index)
    return render_forward_prefix_from_parts(
        initial_text,
        steps[: step_index - 1],
        step_index,
        current_step.reasoning_text,
    )


def render_hindsight_prefix(
    initial_text: str,
    steps: tuple[TrajectoryStep, ...],
    step_index: int,
) -> RenderedPrefix:
    """Render history plus the current observation as the hindsight condition."""

    current_step = _step_at(steps, step_index)
    # **绝不读当前步的 reasoning_text**
    return render_hindsight_prefix_from_parts(
        initial_text,
        steps[: step_index - 1],
        step_index,
        current_step.observation_text,
    )
