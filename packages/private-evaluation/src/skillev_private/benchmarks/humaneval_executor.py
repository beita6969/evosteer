"""Conserved scorer receipt for the original HumanEval test suite."""

from __future__ import annotations

from dataclasses import dataclass

from .humaneval_identity import CodeTestSuite


@dataclass(frozen=True, slots=True)
class HumanEvalExecutionReceipt:
    test_suite: CodeTestSuite
    planned_count: int
    submitted_count: int
    syntax_valid_count: int
    definitive_verdict_count: int
    timeout_count: int
    assertion_failure_count: int
    pass_count: int
    scorer_infrastructure_count: int

    def validate_formal(self) -> None:
        if self.test_suite is not CodeTestSuite.HUMANEVAL_ORIGINAL:
            raise ValueError("formal HumanEval used another test suite")
        if self.planned_count != 128 or self.definitive_verdict_count != 128:
            raise ValueError("HumanEval verdict coverage differs")
        if self.scorer_infrastructure_count:
            raise ValueError("HumanEval scorer infrastructure is not clear")
        if self.pass_count + self.assertion_failure_count + self.timeout_count != 128:
            raise ValueError("HumanEval definitive taxonomy does not conserve the panel")


__all__ = ["HumanEvalExecutionReceipt"]
