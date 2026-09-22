from __future__ import annotations

from skillev.contracts import stable_hash
from skillev.evolution import SkillAuthoringAuthority
from skillev.experiments.evolution_errors import (
    EvolutionErrorEvaluationProtocol,
    FrozenIIDEvaluationSelection,
)
from skillev.experiments.protocol import (
    BENCHMARK_SPECS,
    FIXED_SEED,
    BenchmarkRole,
    BenchmarkTaskCount,
    DatasetSnapshotIdentity,
    FrozenTaskSequenceIdentity,
    ProtocolFreeze,
    SchedulePurpose,
    TrainingUse,
    create_protocol,
)
from skillev.policy import TrainableStateIdentity
from skillev.runtime import BudgetVector, ExactAttemptRunPlan
from tests.experiments.formal_execution_helpers import (
    make_formal_application,
    make_formal_execution,
)


def _hash(label: str) -> str:
    return stable_hash({"label": label})


def _schedules() -> tuple[FrozenTaskSequenceIdentity, ...]:
    memberships = {
        SchedulePurpose.IID_TRAINING: tuple(
            item.benchmark
            for item in BENCHMARK_SPECS
            if item.training_use is TrainingUse.TRAINING_MIX
        ),
        SchedulePurpose.IID_EVALUATION: tuple(
            item.benchmark for item in BENCHMARK_SPECS if item.role is BenchmarkRole.IID
        ),
        SchedulePurpose.OOD_EVALUATION: tuple(
            item.benchmark for item in BENCHMARK_SPECS if item.role is BenchmarkRole.OOD
        ),
    }
    return tuple(
        FrozenTaskSequenceIdentity(
            purpose=purpose,
            ordered_task_ids_hash=_hash(f"schedule-{purpose.value}"),
            task_count=len(benchmarks),
            benchmark_counts=tuple(
                BenchmarkTaskCount(benchmark=benchmark, count=1)
                for benchmark in sorted(benchmarks, key=lambda item: item.value)
            ),
            schedule_algorithm="test-result-blind-schedule@1",
        )
        for purpose, benchmarks in memberships.items()
    )


def _formal_execution():
    plan = ExactAttemptRunPlan(phase_search_steps=8, closure_steps=1, maximum_cycles=1)
    deployment_hash = _hash("formal-deployment")
    authority = SkillAuthoringAuthority(
        input_schema_id="formal-input@1",
        output_schema_id="formal-output@1",
        license_id="unit-test",
        allowed_task_families=("debug/task-family",),
        allowed_tools=(),
    )
    return make_formal_execution(
        application=make_formal_application(),
        run_plan=plan,
        training_sequence=next(
            item for item in _schedules() if item.purpose is SchedulePurpose.IID_TRAINING
        ),
        backbone_deployment_hash=deployment_hash,
        initial_trainable_state=TrainableStateIdentity.create(
            backbone_deployment_hash=deployment_hash,
            forward_adapter_hash=_hash("forward"),
            backward_adapter_hash=_hash("backward"),
            z_head_hash=_hash("z"),
        ),
        initial_library_version=_hash("library"),
        initial_skill_library_state_hash=_hash("library-state"),
        authoring_authority=authority,
        attempt_budget=BudgetVector(input_tokens=100, output_tokens=100, model_calls=100),
        phi_per_cycle_maximum=BudgetVector(input_tokens=4, output_tokens=4, model_calls=1),
        catalog_freeze_hash=_hash("catalog"),
    )


def test_evolution_error_protocol_round_trips_against_v3_protocol_freeze() -> None:
    protocol = create_protocol(
        tuple(
            DatasetSnapshotIdentity(
                name=spec.dataset_snapshot_name,
                version="official-snapshot-v1",
                snapshot_hash=_hash(spec.dataset_snapshot_name),
            )
            for spec in BENCHMARK_SPECS
        ),
        schedule_identities=_schedules(),
        formal_execution=_formal_execution(),
    )
    freeze = ProtocolFreeze.create(protocol)
    selections = tuple(
        FrozenIIDEvaluationSelection(
            benchmark=spec.benchmark,
            selection_hash=_hash(f"selection-{spec.benchmark.value}"),
            selection_count=2,
        )
        for spec in BENCHMARK_SPECS
        if spec.role is BenchmarkRole.IID
    )
    error_protocol = EvolutionErrorEvaluationProtocol(
        benchmark_protocol_hash=protocol.content_hash,
        protocol_freeze_id=freeze.freeze_id,
        seed=FIXED_SEED,
        selections=selections,
        decoding_snapshot_id=_hash("decoding"),
        state_policy_identity=_hash("policy"),
        post_window_optimizer_steps=2,
        post_window_trajectory_count=4,
        reward_tolerance=0.1,
        minimum_product_invocations=1,
    )

    assert EvolutionErrorEvaluationProtocol.from_value(error_protocol.to_value()) == error_protocol
