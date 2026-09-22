"""Trusted evaluator outcome to TTB terminal-reward side-channel adapter."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass

from .outcomes import EvaluationStatus, TerminalRewardSignal, TrustedEvaluatorOutcome


@dataclass(frozen=True, slots=True)
class TerminalRewardAdapter:
    """Map an answer-free evaluator projection to ``R(tau)`` in ``[0, 1]``.

    ``epsilon_min`` is intentionally not applied here: the methodology adds it
    when constructing ``tilde R`` inside the TTB objective.
    """

    invalid_reward: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.invalid_reward, bool) or not isinstance(
            self.invalid_reward, int | float
        ):
            raise TypeError("invalid reward must be numeric")
        if float(self.invalid_reward) != 0:
            raise ValueError("invalid submissions receive zero reward, not configurable credit")

    def adapt(self, outcome: TrustedEvaluatorOutcome) -> TerminalRewardSignal:
        if outcome.status is EvaluationStatus.EVALUATOR_ERROR:
            raise RuntimeError("evaluator infrastructure failed; no terminal reward is available")
        value = (
            float(outcome.score)
            if outcome.status is EvaluationStatus.COMPLETED
            else float(self.invalid_reward)
        )
        return TerminalRewardSignal(outcome.trajectory_id, value)

    def write_side_channel(
        self,
        outcome: TrustedEvaluatorOutcome,
        destination: MutableMapping[str, float],
    ) -> TerminalRewardSignal:
        """Write only ``trajectory_id -> scalar`` to a trainer-owned mapping."""

        signal = self.adapt(outcome)
        if signal.trajectory_id in destination:
            raise KeyError("terminal reward for trajectory already exists")
        destination[signal.trajectory_id] = float(signal.value)
        return signal


__all__ = ["TerminalRewardAdapter"]
