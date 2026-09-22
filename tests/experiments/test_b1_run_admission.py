"""B1 result-blind run-plan, budget, and terminal admission coverage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast

import pytest

from skillev.application_config import ApplicationConfig
from skillev.contracts import TrainingStepReportValue, stable_hash
from skillev.experiments.b1_run_admission import B1RunAdmission
from skillev.experiments.evolution_preflight import build_planned_evolution_preflight
from skillev.runtime import BudgetVector, ExactAttemptRunPlan, FullAttemptSummary

_EXACT_ADMISSION_HASH = stable_hash("test exact H0 and Generate admission")
_MEASURED_TASK_AND_WRAPPER_MAXIMUM = 2_048


def _admission() -> B1RunAdmission:
    return B1RunAdmission.from_preflight(
        build_planned_evolution_preflight(),
        exact_admission_report_hash=_EXACT_ADMISSION_HASH,
        measured_task_and_wrapper_maximum_tokens=(_MEASURED_TASK_AND_WRAPPER_MAXIMUM),
    )


@dataclass(frozen=True)
class _ExactInputProjection:
    run_plan: ExactAttemptRunPlan
    phi_per_cycle_maximum: BudgetVector
    attempt_budget: BudgetVector
    application: ApplicationConfig
    initial_library_version: str


def _report(optimizer_step: int) -> TrainingStepReportValue:
    return TrainingStepReportValue(
        optimizer_step=optimizer_step,
        batch_id=f"formal-batch-{optimizer_step}",
        torch_batch_loss=0.25,
        audited_batch_loss=0.25,
        mean_reward=0.5,
        grad_norm_forward=1.0,
        grad_norm_backward=1.0,
        grad_norm_z=1.0,
        forward_adapter_version=f"forward@{optimizer_step}",
        backward_adapter_version=f"backward@{optimizer_step}",
        z_version=f"z@{optimizer_step}",
        started_at="2026-08-02T00:00:00Z",
        completed_at="2026-08-02T00:00:01Z",
    )


def _successful_summary(admission: B1RunAdmission) -> FullAttemptSummary:
    planned = admission.run_plan.total_training_steps
    return FullAttemptSummary(
        reports=tuple(_report(step) for step in range(1, planned + 1)),
        planned_training_steps_this_attempt=planned,
        completed_training_steps_this_attempt=planned,
        actions_committed_this_attempt=1,
        cycles_committed_this_attempt=1,
        cycles_committed_in_run=1,
        initial_optimizer_step=0,
        final_optimizer_step=planned,
        final_library_version="sha256:" + "2" * 64,
        final_policy_snapshot_id="formal-final-policy",
    )


def _library_versions(admission: B1RunAdmission, *, final: str) -> tuple[str, ...]:
    search = admission.run_plan.phase_search_steps
    return (admission.terminal.initial_library_version,) * search + (
        final,
    ) * admission.run_plan.closure_steps


def test_b1_admission_binds_the_conservative_f2_f3_envelope() -> None:
    proof = build_planned_evolution_preflight()
    admission = B1RunAdmission.from_preflight(
        proof,
        exact_admission_report_hash=_EXACT_ADMISSION_HASH,
        measured_task_and_wrapper_maximum_tokens=(_MEASURED_TASK_AND_WRAPPER_MAXIMUM),
    )

    assert admission.f2_f3_proof_hash == proof.content_hash
    assert admission.run_plan.phase_search_steps == 4606
    assert admission.run_plan.closure_steps == 2
    assert admission.run_plan.maximum_cycles == 2
    assert admission.terminal.minimum_committed_cycles == 1
    assert admission.phi_per_cycle_maximum == BudgetVector(
        input_tokens=1_687_552,
        output_tokens=843_776,
        model_calls=206,
    )
    assert admission.total_phi_maximum == BudgetVector(
        input_tokens=3_375_104,
        output_tokens=1_687_552,
        model_calls=412,
    )
    assert admission.attempt_budget == BudgetVector(
        input_tokens=9_063_071_744,
        output_tokens=107_855_872,
        model_calls=138_652,
        agent_turns=69_120,
        tool_calls=69_120,
        wall_time_milliseconds=20_736_000_000,
    )
    assert admission.maximum_library_size == 206
    assert admission.maximum_h0_tokens == 32_768
    assert admission.maximum_applicable_skills_per_task == 5
    assert admission.maximum_complete_rendered_skill_block_tokens == 3_400
    assert admission.maximum_complete_rendered_skill_block_tokens_in_h0 == 17_000
    assert admission.available_h0_task_and_wrapper_tokens == 15_768
    assert admission.measured_task_and_wrapper_maximum_tokens == _MEASURED_TASK_AND_WRAPPER_MAXIMUM
    assert B1RunAdmission.from_value(admission.to_value()) == admission
    admission.require_preflight(
        proof,
        exact_admission_report_hash=_EXACT_ADMISSION_HASH,
        measured_task_and_wrapper_maximum_tokens=(_MEASURED_TASK_AND_WRAPPER_MAXIMUM),
    )


def test_b1_admission_rejects_a_cycle_specific_phi_total_as_the_formal_cap() -> None:
    proof = build_planned_evolution_preflight()

    with pytest.raises(ValueError):
        replace(
            B1RunAdmission.from_preflight(
                proof,
                exact_admission_report_hash=_EXACT_ADMISSION_HASH,
                measured_task_and_wrapper_maximum_tokens=(_MEASURED_TASK_AND_WRAPPER_MAXIMUM),
            ),
            total_phi_maximum=proof.cycle_specific_total_phi_maximum,
        )


def test_b1_admission_rejects_a_measured_wrapper_over_complete_block_capacity() -> None:
    proof = build_planned_evolution_preflight()

    with pytest.raises(ValueError):
        B1RunAdmission.from_preflight(
            proof,
            exact_admission_report_hash=_EXACT_ADMISSION_HASH,
            measured_task_and_wrapper_maximum_tokens=(
                proof.available_h0_task_and_wrapper_tokens + 1
            ),
        )


def test_b1_admission_requires_exact_formal_input_budget_and_library() -> None:
    admission = _admission()
    exact = _ExactInputProjection(
        run_plan=admission.run_plan,
        phi_per_cycle_maximum=admission.phi_per_cycle_maximum,
        attempt_budget=admission.attempt_budget,
        application=cast(
            ApplicationConfig,
            SimpleNamespace(maximum_h0_tokens=admission.maximum_h0_tokens),
        ),
        initial_library_version=admission.terminal.initial_library_version,
    )

    admission.require_exact_input(exact)

    with pytest.raises(ValueError):
        admission.require_exact_input(
            replace(exact, attempt_budget=exact.attempt_budget.add(BudgetVector(model_calls=1)))
        )


def test_b1_terminal_requires_mutation_and_final_library_closure() -> None:
    admission = _admission()
    summary = _successful_summary(admission)
    versions = _library_versions(admission, final=summary.final_library_version)

    admission.terminal.require_success(
        summary,
        ordered_training_library_versions=versions,
    )

    with pytest.raises(ValueError):
        admission.terminal.require_success(
            replace(
                summary,
                cycles_committed_this_attempt=0,
                cycles_committed_in_run=0,
                actions_committed_this_attempt=0,
            ),
            ordered_training_library_versions=versions,
        )
    with pytest.raises(ValueError):
        admission.terminal.require_success(
            replace(
                summary,
                final_library_version=admission.terminal.initial_library_version,
            ),
            ordered_training_library_versions=(admission.terminal.initial_library_version,)
            * admission.run_plan.total_training_steps,
        )
    with pytest.raises(ValueError):
        admission.terminal.require_success(
            summary,
            ordered_training_library_versions=(
                *versions[:-1],
                admission.terminal.initial_library_version,
            ),
        )


def test_b1_terminal_rejects_an_incomplete_source_training_sequence() -> None:
    admission = _admission()
    summary = _successful_summary(admission)
    versions = _library_versions(admission, final=summary.final_library_version)

    with pytest.raises(ValueError):
        admission.terminal.require_success(
            summary,
            ordered_training_library_versions=versions[:-1],
        )
