"""Result-blind B1 admission for the exact run plan, budgets, and terminal.

This module deliberately does not change the ordinary attempt runtime.  B1
adds this immutable admission beside the existing formal execution freeze so
that a formal input must use the conservative F2/F3 capacity proof and a
formal success must demonstrate a real library-changing cycle followed by the
predeclared closure tail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

from skillev.application_config import ApplicationConfig
from skillev.contracts import JsonValue, normalize_json, stable_hash
from skillev.contracts.identity import validate_sha256
from skillev.runtime import BudgetVector, ExactAttemptRunPlan
from skillev.runtime.attempt_protocol import AttemptRunSummary

from .evolution_preflight import EvolutionPreflightProof

B1_RUN_ADMISSION_FORMAT: Final = "skillev-b1-run-admission@2"


class B1ExactAttemptProjection(Protocol):
    """The formal-input fields consumed by B1 admission."""

    @property
    def run_plan(self) -> ExactAttemptRunPlan: ...

    @property
    def phi_per_cycle_maximum(self) -> BudgetVector: ...

    @property
    def attempt_budget(self) -> BudgetVector: ...

    @property
    def application(self) -> ApplicationConfig: ...

    @property
    def initial_library_version(self) -> str: ...


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _object(value: object, *, fields: frozenset[str], label: str) -> dict[str, JsonValue]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or frozenset(normalized) != fields:
        raise ValueError(f"{label} has an incompatible field set")
    return normalized


@dataclass(frozen=True, slots=True)
class B1TerminalRequirements:
    """Success conditions that distinguish a completed formal method attempt.

    The ordinary runtime continues to permit non-formal experiments that do
    not trigger a phase.  The B1 formal protocol instead requires at least one
    non-empty mutation and requires every closure slot to train under the
    final, changed library.
    """

    planned_training_steps: int
    closure_steps: int
    minimum_committed_cycles: int
    minimum_committed_actions: int
    initial_library_version: str

    def __post_init__(self) -> None:
        _positive_int(self.planned_training_steps, field="planned_training_steps")
        _positive_int(self.closure_steps, field="closure_steps")
        _positive_int(self.minimum_committed_cycles, field="minimum_committed_cycles")
        _positive_int(self.minimum_committed_actions, field="minimum_committed_actions")
        validate_sha256(self.initial_library_version)
        if self.closure_steps >= self.planned_training_steps:
            raise ValueError("closure tail must follow at least one phase-search step")

    def require_success(
        self,
        summary: AttemptRunSummary,
        *,
        ordered_training_library_versions: tuple[str, ...],
    ) -> None:
        """Admit a terminal only when its source training sequence meets B1.

        ``ordered_training_library_versions`` is the ordered projection of the
        authoritative full-method or no-Bayesian training-step source records,
        not a value inferred from the outcome summary.
        """

        if summary.planned_training_steps_this_attempt != self.planned_training_steps:
            raise ValueError("terminal planned steps differ from the B1 run plan")
        if summary.completed_training_steps_this_attempt != self.planned_training_steps:
            raise ValueError("terminal did not complete every B1 training step")
        if summary.cycles_committed_this_attempt < self.minimum_committed_cycles:
            raise ValueError("terminal lacks the required committed evolution cycle")
        if summary.actions_committed_this_attempt < self.minimum_committed_actions:
            raise ValueError("terminal lacks the required committed evolution action")
        if summary.final_library_version == self.initial_library_version:
            raise ValueError("terminal library did not change from the B1 initial library")
        if len(ordered_training_library_versions) != self.planned_training_steps:
            raise ValueError("terminal source training sequence is incomplete")
        if any(type(item) is not str or not item for item in ordered_training_library_versions):
            raise ValueError("terminal source library versions must be non-empty text")
        closure_versions = ordered_training_library_versions[-self.closure_steps :]
        if any(item != summary.final_library_version for item in closure_versions):
            raise ValueError("B1 closure tail was not executed under the final library")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "closure_steps": self.closure_steps,
            "initial_library_version": self.initial_library_version,
            "minimum_committed_actions": self.minimum_committed_actions,
            "minimum_committed_cycles": self.minimum_committed_cycles,
            "planned_training_steps": self.planned_training_steps,
        }

    @classmethod
    def from_value(cls, value: object) -> B1TerminalRequirements:
        data = _object(
            value,
            fields=frozenset(
                {
                    "closure_steps",
                    "initial_library_version",
                    "minimum_committed_actions",
                    "minimum_committed_cycles",
                    "planned_training_steps",
                }
            ),
            label="B1 terminal requirements",
        )
        if type(data["initial_library_version"]) is not str:
            raise TypeError("B1 initial library version must be text")
        return cls(
            planned_training_steps=_positive_int(
                data["planned_training_steps"], field="planned_training_steps"
            ),
            closure_steps=_positive_int(data["closure_steps"], field="closure_steps"),
            minimum_committed_cycles=_positive_int(
                data["minimum_committed_cycles"], field="minimum_committed_cycles"
            ),
            minimum_committed_actions=_positive_int(
                data["minimum_committed_actions"], field="minimum_committed_actions"
            ),
            initial_library_version=data["initial_library_version"],
        )


@dataclass(frozen=True, slots=True)
class B1RunAdmission:
    """The exact B1 projection of the existing result-blind F2/F3 proof."""

    f2_f3_proof_hash: str
    exact_admission_report_hash: str
    run_plan: ExactAttemptRunPlan
    phi_per_cycle_maximum: BudgetVector
    total_phi_maximum: BudgetVector
    attempt_budget: BudgetVector
    closure_tail_budget: BudgetVector
    maximum_library_size: int
    maximum_h0_tokens: int
    maximum_applicable_skills_per_task: int
    maximum_complete_rendered_skill_block_tokens: int
    maximum_complete_rendered_skill_block_tokens_in_h0: int
    available_h0_task_and_wrapper_tokens: int
    measured_task_and_wrapper_maximum_tokens: int
    terminal: B1TerminalRequirements
    format: str = B1_RUN_ADMISSION_FORMAT

    def __post_init__(self) -> None:
        validate_sha256(self.f2_f3_proof_hash)
        validate_sha256(self.exact_admission_report_hash)
        if not isinstance(self.run_plan, ExactAttemptRunPlan):
            raise TypeError("B1 admission requires an ExactAttemptRunPlan")
        for field in (
            "phi_per_cycle_maximum",
            "total_phi_maximum",
            "attempt_budget",
            "closure_tail_budget",
        ):
            if not isinstance(getattr(self, field), BudgetVector):
                raise TypeError(f"{field} must be a BudgetVector")
        _positive_int(self.maximum_library_size, field="maximum_library_size")
        _positive_int(self.maximum_h0_tokens, field="maximum_h0_tokens")
        _positive_int(
            self.maximum_applicable_skills_per_task,
            field="maximum_applicable_skills_per_task",
        )
        _positive_int(
            self.maximum_complete_rendered_skill_block_tokens,
            field="maximum_complete_rendered_skill_block_tokens",
        )
        _positive_int(
            self.maximum_complete_rendered_skill_block_tokens_in_h0,
            field="maximum_complete_rendered_skill_block_tokens_in_h0",
        )
        _positive_int(
            self.available_h0_task_and_wrapper_tokens,
            field="available_h0_task_and_wrapper_tokens",
        )
        _positive_int(
            self.measured_task_and_wrapper_maximum_tokens,
            field="measured_task_and_wrapper_maximum_tokens",
        )
        if not isinstance(self.terminal, B1TerminalRequirements):
            raise TypeError("B1 admission requires terminal requirements")
        if self.format != B1_RUN_ADMISSION_FORMAT:
            raise ValueError("unsupported B1 run admission format")
        if self.total_phi_maximum != self.phi_per_cycle_maximum.scale(self.run_plan.maximum_cycles):
            raise ValueError("B1 total Phi budget is not the conservative per-cycle envelope")
        if not self.total_phi_maximum.fits_within(self.attempt_budget):
            raise ValueError("B1 attempt budget does not contain its complete Phi envelope")
        if not self.closure_tail_budget.fits_within(self.attempt_budget):
            raise ValueError("B1 attempt budget does not contain its closure tail")
        if self.maximum_complete_rendered_skill_block_tokens_in_h0 != (
            self.maximum_applicable_skills_per_task
            * self.maximum_complete_rendered_skill_block_tokens
        ):
            raise ValueError("B1 complete rendered skill-block envelope is inconsistent")
        if (
            self.maximum_complete_rendered_skill_block_tokens_in_h0
            + (self.available_h0_task_and_wrapper_tokens)
            != self.maximum_h0_tokens
        ):
            raise ValueError("B1 task/wrapper capacity differs from its H0 cap")
        if (
            self.maximum_complete_rendered_skill_block_tokens_in_h0
            + (self.measured_task_and_wrapper_maximum_tokens)
            > self.maximum_h0_tokens
        ):
            raise ValueError("B1 measured task/wrapper maximum exceeds its H0 capacity")
        if self.terminal.planned_training_steps != self.run_plan.total_training_steps:
            raise ValueError("B1 terminal steps differ from the exact run plan")
        if self.terminal.closure_steps != self.run_plan.closure_steps:
            raise ValueError("B1 terminal closure differs from the exact run plan")
        if self.terminal.minimum_committed_cycles > self.run_plan.maximum_cycles:
            raise ValueError("B1 minimum cycles exceed the exact run plan")

    @classmethod
    def from_preflight(
        cls,
        proof: EvolutionPreflightProof,
        *,
        exact_admission_report_hash: str,
        measured_task_and_wrapper_maximum_tokens: int,
    ) -> B1RunAdmission:
        if not isinstance(proof, EvolutionPreflightProof):
            raise TypeError("B1 run admission requires EvolutionPreflightProof")
        return cls(
            f2_f3_proof_hash=proof.content_hash,
            exact_admission_report_hash=exact_admission_report_hash,
            run_plan=proof.inputs.run_plan,
            phi_per_cycle_maximum=proof.phi_per_cycle_maximum,
            total_phi_maximum=proof.total_phi_maximum,
            attempt_budget=proof.attempt_budget,
            closure_tail_budget=proof.closure_tail_budget,
            maximum_library_size=proof.maximum_library_size,
            maximum_h0_tokens=proof.maximum_h0_tokens,
            maximum_applicable_skills_per_task=proof.maximum_applicable_skills_per_task,
            maximum_complete_rendered_skill_block_tokens=(
                proof.maximum_complete_rendered_skill_block_tokens
            ),
            maximum_complete_rendered_skill_block_tokens_in_h0=(
                proof.maximum_complete_rendered_skill_block_tokens_in_h0
            ),
            available_h0_task_and_wrapper_tokens=(proof.available_h0_task_and_wrapper_tokens),
            measured_task_and_wrapper_maximum_tokens=(measured_task_and_wrapper_maximum_tokens),
            terminal=B1TerminalRequirements(
                planned_training_steps=proof.inputs.run_plan.total_training_steps,
                closure_steps=proof.inputs.run_plan.closure_steps,
                minimum_committed_cycles=proof.inputs.minimum_required_committed_cycles,
                minimum_committed_actions=1,
                initial_library_version=proof.inputs.initial_library.current_version,
            ),
        )

    def require_preflight(
        self,
        proof: EvolutionPreflightProof,
        *,
        exact_admission_report_hash: str,
        measured_task_and_wrapper_maximum_tokens: int,
    ) -> None:
        if self != B1RunAdmission.from_preflight(
            proof,
            exact_admission_report_hash=exact_admission_report_hash,
            measured_task_and_wrapper_maximum_tokens=(measured_task_and_wrapper_maximum_tokens),
        ):
            raise ValueError("B1 run admission differs from the frozen F2/F3 proof")

    def require_exact_input(self, exact: B1ExactAttemptProjection) -> None:
        """Require one formal exact input to use this plan and capacity verbatim."""

        if exact.run_plan != self.run_plan:
            raise ValueError("formal exact input uses another B1 run plan")
        if exact.phi_per_cycle_maximum != self.phi_per_cycle_maximum:
            raise ValueError("formal exact input uses another per-cycle Phi budget")
        if exact.attempt_budget != self.attempt_budget:
            raise ValueError("formal exact input uses another attempt budget")
        if exact.application.maximum_h0_tokens != self.maximum_h0_tokens:
            raise ValueError("formal exact input uses another maximum H0 size")
        if exact.initial_library_version != self.terminal.initial_library_version:
            raise ValueError("formal exact input uses another initial skill library")

    @property
    def content_hash(self) -> str:
        return stable_hash(self._body_value())

    @property
    def initial_library_version(self) -> str:
        return self.terminal.initial_library_version

    def _body_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_budget": self.attempt_budget.to_value(),
            "closure_tail_budget": self.closure_tail_budget.to_value(),
            "exact_admission_report_hash": self.exact_admission_report_hash,
            "f2_f3_proof_hash": self.f2_f3_proof_hash,
            "format": self.format,
            "available_h0_task_and_wrapper_tokens": (self.available_h0_task_and_wrapper_tokens),
            "maximum_applicable_skills_per_task": self.maximum_applicable_skills_per_task,
            "maximum_complete_rendered_skill_block_tokens": (
                self.maximum_complete_rendered_skill_block_tokens
            ),
            "maximum_complete_rendered_skill_block_tokens_in_h0": (
                self.maximum_complete_rendered_skill_block_tokens_in_h0
            ),
            "maximum_h0_tokens": self.maximum_h0_tokens,
            "maximum_library_size": self.maximum_library_size,
            "measured_task_and_wrapper_maximum_tokens": (
                self.measured_task_and_wrapper_maximum_tokens
            ),
            "phi_per_cycle_maximum": self.phi_per_cycle_maximum.to_value(),
            "run_plan": self.run_plan.to_value(),
            "terminal": self.terminal.to_value(),
            "total_phi_maximum": self.total_phi_maximum.to_value(),
        }

    def to_value(self) -> dict[str, JsonValue]:
        return {**self._body_value(), "content_hash": self.content_hash}

    @classmethod
    def from_value(cls, value: object) -> B1RunAdmission:
        data = _object(
            value,
            fields=frozenset(
                {
                    "attempt_budget",
                    "available_h0_task_and_wrapper_tokens",
                    "closure_tail_budget",
                    "content_hash",
                    "exact_admission_report_hash",
                    "f2_f3_proof_hash",
                    "format",
                    "maximum_applicable_skills_per_task",
                    "maximum_complete_rendered_skill_block_tokens",
                    "maximum_complete_rendered_skill_block_tokens_in_h0",
                    "maximum_h0_tokens",
                    "maximum_library_size",
                    "measured_task_and_wrapper_maximum_tokens",
                    "phi_per_cycle_maximum",
                    "run_plan",
                    "terminal",
                    "total_phi_maximum",
                }
            ),
            label="B1 run admission",
        )
        result = cls(
            f2_f3_proof_hash=_text(data["f2_f3_proof_hash"], field="f2_f3_proof_hash"),
            exact_admission_report_hash=_text(
                data["exact_admission_report_hash"],
                field="exact_admission_report_hash",
            ),
            run_plan=ExactAttemptRunPlan.from_value(data["run_plan"]),
            phi_per_cycle_maximum=BudgetVector.from_value(data["phi_per_cycle_maximum"]),
            total_phi_maximum=BudgetVector.from_value(data["total_phi_maximum"]),
            attempt_budget=BudgetVector.from_value(data["attempt_budget"]),
            closure_tail_budget=BudgetVector.from_value(data["closure_tail_budget"]),
            maximum_library_size=_positive_int(
                data["maximum_library_size"], field="maximum_library_size"
            ),
            maximum_h0_tokens=_positive_int(data["maximum_h0_tokens"], field="maximum_h0_tokens"),
            maximum_applicable_skills_per_task=_positive_int(
                data["maximum_applicable_skills_per_task"],
                field="maximum_applicable_skills_per_task",
            ),
            maximum_complete_rendered_skill_block_tokens=_positive_int(
                data["maximum_complete_rendered_skill_block_tokens"],
                field="maximum_complete_rendered_skill_block_tokens",
            ),
            maximum_complete_rendered_skill_block_tokens_in_h0=_positive_int(
                data["maximum_complete_rendered_skill_block_tokens_in_h0"],
                field="maximum_complete_rendered_skill_block_tokens_in_h0",
            ),
            available_h0_task_and_wrapper_tokens=_positive_int(
                data["available_h0_task_and_wrapper_tokens"],
                field="available_h0_task_and_wrapper_tokens",
            ),
            measured_task_and_wrapper_maximum_tokens=_positive_int(
                data["measured_task_and_wrapper_maximum_tokens"],
                field="measured_task_and_wrapper_maximum_tokens",
            ),
            terminal=B1TerminalRequirements.from_value(data["terminal"]),
            format=_text(data["format"], field="format"),
        )
        if _text(data["content_hash"], field="content_hash") != result.content_hash:
            raise ValueError("B1 run admission content hash differs")
        return result


__all__ = [
    "B1_RUN_ADMISSION_FORMAT",
    "B1ExactAttemptProjection",
    "B1RunAdmission",
    "B1TerminalRequirements",
]
