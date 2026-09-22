"""Closed domain failures that a child attempt publishes without private detail.

The attempt worker is the only place that turns one of these errors into a
public ``AttemptFailed`` record.  The errors themselves are deliberately
terminal: they carry no retry or recovery behaviour.
"""

from __future__ import annotations

from .attempt_protocol import AttemptFailureCode, AttemptFailureStage


class AttemptDomainError(RuntimeError):
    """One known, terminal attempt-domain failure classification."""

    def __init__(
        self,
        *,
        code: AttemptFailureCode,
        stage: AttemptFailureStage,
        private_detail: str,
    ) -> None:
        if not isinstance(code, AttemptFailureCode):
            raise TypeError("attempt domain failure code must be AttemptFailureCode")
        if not isinstance(stage, AttemptFailureStage):
            raise TypeError("attempt domain failure stage must be AttemptFailureStage")
        if type(private_detail) is not str or not private_detail:
            raise ValueError("attempt domain failure requires private detail")
        super().__init__(private_detail)
        self.code = code
        self.stage = stage


class EvolutionMutationFailedError(AttemptDomainError):
    """A complete Phi mutation could not be constructed."""

    def __init__(self, private_detail: str) -> None:
        super().__init__(
            code=AttemptFailureCode.MUTATION_FAILED,
            stage=AttemptFailureStage.EXECUTION,
            private_detail=private_detail,
        )


class LibraryApplyFailedError(AttemptDomainError):
    """The prepared library state could not be made live."""

    def __init__(self, private_detail: str) -> None:
        super().__init__(
            code=AttemptFailureCode.LIBRARY_APPLY_FAILED,
            stage=AttemptFailureStage.EXECUTION,
            private_detail=private_detail,
        )


class PartitionResetFailedError(AttemptDomainError):
    """The deterministic Z-partition reset could not complete."""

    def __init__(self, private_detail: str) -> None:
        super().__init__(
            code=AttemptFailureCode.PARTITION_RESET_FAILED,
            stage=AttemptFailureStage.EXECUTION,
            private_detail=private_detail,
        )


class EventAppendFailedError(AttemptDomainError):
    """An authoritative source event could not be durably appended."""

    def __init__(self, private_detail: str) -> None:
        super().__init__(
            code=AttemptFailureCode.EVENT_APPEND_FAILED,
            stage=AttemptFailureStage.EXECUTION,
            private_detail=private_detail,
        )


__all__ = [
    "AttemptDomainError",
    "EventAppendFailedError",
    "EvolutionMutationFailedError",
    "LibraryApplyFailedError",
    "PartitionResetFailedError",
]
