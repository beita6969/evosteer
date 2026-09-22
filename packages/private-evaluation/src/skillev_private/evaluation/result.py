"""Private evaluation result with an explicitly safe projection."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.evaluation.outcomes import TrustedEvaluatorOutcome


@dataclass(frozen=True, slots=True)
class PrivateEvaluationResult:
    outcome: TrustedEvaluatorOutcome
    diagnostics: tuple[tuple[str, object], ...]

    def public_projection(self) -> TrustedEvaluatorOutcome:
        """Return the only value allowed to leave the evaluator process."""

        return self.outcome


__all__ = ["PrivateEvaluationResult"]
