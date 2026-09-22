"""Formal execution freeze regression coverage."""

from __future__ import annotations

from dataclasses import replace

import pytest

from skillev.contracts import stable_hash
from skillev.evolution import SkillAuthoringAuthority
from skillev.experiments import Benchmark, BenchmarkTaskCount, SchedulePurpose
from skillev.experiments.protocol import FrozenTaskSequenceIdentity
from skillev.policy import TrainableStateIdentity
from skillev.runtime import BudgetVector, ExactAttemptRunPlan
from tests.experiments.formal_execution_helpers import (
    make_formal_application,
    make_formal_execution,
)


def _authority() -> SkillAuthoringAuthority:
    return SkillAuthoringAuthority(
        input_schema_id="formal-input@1",
        output_schema_id="formal-output@1",
        license_id="unit-test",
        allowed_task_families=("debug/task-family",),
        allowed_tools=(),
    )


def _freeze():
    application = make_formal_application()
    plan = ExactAttemptRunPlan(phase_search_steps=2, closure_steps=1, maximum_cycles=1)
    sequence = FrozenTaskSequenceIdentity(
        purpose=SchedulePurpose.IID_TRAINING,
        ordered_task_ids_hash=stable_hash("formal-training-sequence"),
        task_count=plan.total_training_steps,
        benchmark_counts=(
            BenchmarkTaskCount(benchmark=Benchmark.HOTPOT_QA, count=plan.total_training_steps),
        ),
        schedule_algorithm="formal-execution-test@1",
    )
    deployment_hash = stable_hash("formal-deployment")
    return make_formal_execution(
        application=application,
        run_plan=plan,
        training_sequence=sequence,
        backbone_deployment_hash=deployment_hash,
        initial_trainable_state=TrainableStateIdentity.create(
            backbone_deployment_hash=deployment_hash,
            forward_adapter_hash=stable_hash("formal-forward"),
            backward_adapter_hash=stable_hash("formal-backward"),
            z_head_hash=stable_hash("formal-z"),
        ),
        initial_library_version=stable_hash("formal-library"),
        initial_skill_library_state_hash=stable_hash("formal-library-state"),
        authoring_authority=_authority(),
        attempt_budget=BudgetVector(input_tokens=100, output_tokens=100, model_calls=100),
        phi_per_cycle_maximum=BudgetVector(input_tokens=4, output_tokens=4, model_calls=1),
        catalog_freeze_hash=stable_hash("formal-catalog"),
    )


def test_formal_execution_round_trips_and_excludes_only_artifact_namespace() -> None:
    freeze = _freeze()

    assert type(freeze).from_value(freeze.to_value()) == freeze
    assert (
        "experiment_id"
        not in freeze.to_value()["application_scientific_config"]["trainer"]["execution"]
    )

    renamed_namespace = replace(
        make_formal_application(),
        trainer=replace(
            make_formal_application().trainer,
            execution=replace(
                make_formal_application().trainer.execution,
                experiment_id="another-storage-namespace",
            ),
        ),
    )
    freeze.requires_application(renamed_namespace)


def test_formal_execution_rejects_scientific_config_seed_and_schedule_drift() -> None:
    freeze = _freeze()

    changed_optimizer = replace(
        make_formal_application(),
        trainer=replace(
            make_formal_application().trainer,
            optimizer=replace(
                make_formal_application().trainer.optimizer,
                adapter_learning_rate=2e-3,
            ),
        ),
    )
    with pytest.raises(ValueError):
        freeze.requires_application(changed_optimizer)
    with pytest.raises(ValueError):
        replace(freeze, seed=freeze.seed + 1)
    with pytest.raises(ValueError):
        replace(
            freeze,
            training_sequence=replace(freeze.training_sequence, task_count=1),
        )


def test_formal_execution_preregisters_exact_progress_anchor_cycles() -> None:
    freeze = _freeze()

    assert type(freeze).from_value(freeze.to_value()) == freeze
    assert freeze.progress_anchor_cycle_ordinals == (1,)
    freeze.requires_progress_anchor_cycle(1)
    with pytest.raises(ValueError):
        freeze.requires_progress_anchor_cycle(2)
    with pytest.raises(ValueError):
        replace(freeze, progress_anchor_cycle_ordinals=(0,))
    with pytest.raises(ValueError):
        replace(freeze, progress_anchor_cycle_ordinals=(1, 1))
    with pytest.raises(ValueError):
        replace(freeze, progress_anchor_cycle_ordinals=())
