"""Result-blind B1 schedule, manifest, and zero-result composition tests."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from skillev_private.benchmarks.catalog import (
    PrivateBenchmarkCatalog,
    PrivateBenchmarkWorkload,
)
from skillev_private.experiments.b1_freeze import (
    B1EvaluationPlan,
    B1FreezeSummary,
    B1ResultCounts,
    build_b1_freeze_artifacts,
    build_b1_schedules,
    write_b1_freeze_artifacts,
)
from skillev_private.experiments.benchmark_attempt_input import PrivateBenchmarkAttemptInput

from skillev.application import ApplicationConfig
from skillev.contracts import stable_hash
from skillev.evolution import SkillAuthoringAuthority
from skillev.experiments import (
    BENCHMARK_SPECS,
    FIXED_SEED,
    DatasetSnapshotIdentity,
    FixedSubsampleEvaluation,
    FormalExecutionFreeze,
    ProtocolFreeze,
    SchedulePurpose,
    TrainingUse,
    create_protocol,
)
from skillev.experiments.b1_run_admission import B1RunAdmission, B1TerminalRequirements
from skillev.policy import TrainableStateIdentity
from skillev.rollout import RolloutTask
from skillev.runtime import AttemptBuilderKind, BudgetVector, ExactAttemptRunPlan
from skillev.training import RolloutWorkflowBinding
from tests.experiments.formal_execution_helpers import (
    make_formal_application,
    make_formal_artifacts,
)

if TYPE_CHECKING:
    from skillev.training import RolloutSessionBundle


@dataclass(frozen=True, slots=True)
class _UnusedSessionFactory:
    def create(self, task: RolloutTask) -> RolloutSessionBundle:
        raise AssertionError(f"B1 schedule construction opened {task.task_id}")


def _task(benchmark_id: str, ordinal: int) -> RolloutTask:
    return RolloutTask(
        task_id=f"b1-{benchmark_id}-{ordinal:04d}",
        environment_id=f"b1:{benchmark_id}",
        task_family=f"{benchmark_id}/synthetic",
        context_id=f"b1:{benchmark_id}:{ordinal:04d}",
        query="Public synthetic B1 schedule fixture.",
        available_tools=(),
        public_context={"benchmark_id": benchmark_id},
    )


def _catalog() -> PrivateBenchmarkCatalog:
    return PrivateBenchmarkCatalog(
        tuple(
            PrivateBenchmarkWorkload(
                benchmark=spec.benchmark,
                tasks=tuple(
                    _task(spec.benchmark.value, ordinal)
                    for ordinal in range(
                        1
                        if spec.training_use is TrainingUse.TRAINING_MIX
                        else (
                            500
                            if isinstance(spec.evaluation_sampling, FixedSubsampleEvaluation)
                            else 6
                        )
                    )
                ),
                session_factory=_UnusedSessionFactory(),
            )
            for spec in BENCHMARK_SPECS
        )
    )


def _formal_execution(
    *,
    application: ApplicationConfig,
    run_plan: ExactAttemptRunPlan,
    training_sequence: object,
) -> FormalExecutionFreeze:
    artifacts = make_formal_artifacts()
    deployment_hash = stable_hash("b1-deployment")
    state = TrainableStateIdentity.create(
        backbone_deployment_hash=deployment_hash,
        forward_adapter_hash=stable_hash("b1-forward"),
        backward_adapter_hash=stable_hash("b1-backward"),
        z_head_hash=stable_hash("b1-z"),
    )
    initial_library_version = stable_hash("b1-library")
    attempt_budget = BudgetVector(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        model_calls=100_000,
        agent_turns=100_000,
        tool_calls=100_000,
        wall_time_milliseconds=100_000_000,
    )
    phi_per_cycle_maximum = BudgetVector(
        input_tokens=10_000,
        output_tokens=10_000,
        model_calls=100,
    )
    run_admission = B1RunAdmission(
        f2_f3_proof_hash=stable_hash("f2-f3-proof"),
        exact_admission_report_hash=stable_hash("b1-exact-admission-report"),
        run_plan=run_plan,
        phi_per_cycle_maximum=phi_per_cycle_maximum,
        total_phi_maximum=phi_per_cycle_maximum.scale(run_plan.maximum_cycles),
        attempt_budget=attempt_budget,
        closure_tail_budget=BudgetVector(model_calls=1),
        maximum_library_size=3,
        maximum_h0_tokens=application.maximum_h0_tokens,
        maximum_applicable_skills_per_task=1,
        maximum_complete_rendered_skill_block_tokens=64,
        maximum_complete_rendered_skill_block_tokens_in_h0=64,
        available_h0_task_and_wrapper_tokens=application.maximum_h0_tokens - 64,
        measured_task_and_wrapper_maximum_tokens=32,
        terminal=B1TerminalRequirements(
            planned_training_steps=run_plan.total_training_steps,
            closure_steps=run_plan.closure_steps,
            minimum_committed_cycles=1,
            minimum_committed_actions=1,
            initial_library_version=initial_library_version,
        ),
    )
    return FormalExecutionFreeze.create(
        seed=FIXED_SEED,
        backbone_kind="qwen-causal",
        backbone_deployment_hash=deployment_hash,
        base_model_artifact=artifacts.base_model,
        tokenizer_artifact=artifacts.tokenizer,
        implementation_build=artifacts.implementation,
        application=application,
        run_plan=run_plan,
        initial_trainable_state=state,
        initial_library_version=initial_library_version,
        initial_skill_library_state_hash=stable_hash("b1-library-state"),
        authoring_authority=SkillAuthoringAuthority(
            input_schema_id="b1-input@1",
            output_schema_id="b1-output@1",
            license_id="CC0-1.0",
            allowed_task_families=("synthetic",),
            allowed_tools=(),
        ),
        attempt_budget=attempt_budget,
        phi_per_cycle_maximum=phi_per_cycle_maximum,
        b1_run_admission=run_admission,
        training_sequence=training_sequence,  # type: ignore[arg-type]
        catalog_freeze_hash=stable_hash("b1-catalog"),
    )


@dataclass(frozen=True, slots=True)
class _PublicIdentity:
    content_hash: str


@dataclass(frozen=True, slots=True)
class _FakeExactInput:
    builder_kind: AttemptBuilderKind
    protocol: object
    protocol_freeze: ProtocolFreeze
    training_sequence: object
    run_plan: ExactAttemptRunPlan
    phi_per_cycle_maximum: BudgetVector
    attempt_budget: BudgetVector
    application: ApplicationConfig
    initial_library_version: str
    rollout_workflow: RolloutWorkflowBinding = field(default_factory=RolloutWorkflowBinding)
    common_cross_arm_identity: str = "one-common-cross-arm-identity"

    def to_value(self) -> dict[str, object]:
        return {
            "builder_kind": self.builder_kind.value,
            "protocol_hash": self.protocol.content_hash,  # type: ignore[attr-defined]
            "training_sequence": self.training_sequence.to_value(),  # type: ignore[attr-defined]
        }

    def public_identity(self, *, exact_input_sha256: str) -> _PublicIdentity:
        return _PublicIdentity(
            content_hash=stable_hash(
                {"builder_kind": self.builder_kind.value, "input": exact_input_sha256}
            )
        )

    def cross_arm_execution_identity(self, *, exact_input_sha256: str) -> str:
        del exact_input_sha256
        return self.common_cross_arm_identity


def _artifacts() -> object:
    run_plan = ExactAttemptRunPlan(phase_search_steps=8, closure_steps=1, maximum_cycles=2)
    application = make_formal_application(batch_size=1)
    schedules = build_b1_schedules(_catalog(), training_task_count=9)
    formal = _formal_execution(
        application=application,
        run_plan=run_plan,
        training_sequence=schedules[0].identity,
    )
    snapshots = tuple(
        DatasetSnapshotIdentity(
            name=spec.dataset_snapshot_name,
            version="b1-fixture@1",
            snapshot_hash=stable_hash({"snapshot": spec.benchmark.value}),
        )
        for spec in BENCHMARK_SPECS
    )
    protocol = create_protocol(
        dataset_snapshots=snapshots,
        schedule_identities=tuple(item.identity for item in schedules),
        formal_execution=formal,
    )
    freeze = ProtocolFreeze.create(protocol)
    fake_inputs = tuple(
        _FakeExactInput(
            builder_kind=kind,
            protocol=protocol,
            protocol_freeze=freeze,
            training_sequence=schedules[0],
            run_plan=formal.run_plan,
            phi_per_cycle_maximum=formal.phi_per_cycle_maximum,
            attempt_budget=formal.attempt_budget,
            application=application,
            initial_library_version=formal.initial_library_version,
        )
        for kind in AttemptBuilderKind
    )
    return build_b1_freeze_artifacts(
        schedules=schedules,
        protocol=protocol,
        protocol_freeze=freeze,
        exact_inputs=cast(tuple[PrivateBenchmarkAttemptInput, ...], fake_inputs),
        b0_report_hash=stable_hash("b0-report"),
        f2_f3_proof_hash=stable_hash("f2-f3-proof"),
    )


def test_b1_builds_three_deterministic_result_blind_schedules() -> None:
    catalog = _catalog()
    first = build_b1_schedules(catalog, training_task_count=9)
    second = build_b1_schedules(catalog, training_task_count=9)

    assert first == second
    assert tuple(item.identity.purpose for item in first) == (
        SchedulePurpose.IID_TRAINING,
        SchedulePurpose.IID_EVALUATION,
        SchedulePurpose.OOD_EVALUATION,
    )
    assert first[0].identity.task_count == 9
    assert all(item.identity.task_count >= 1 for item in first)


def test_b1_composes_seven_training_slots_and_every_evaluation_coordinate() -> None:
    artifacts = _artifacts()

    assert len(artifacts.run_manifest.slots) == 7  # type: ignore[attr-defined]
    assert len(artifacts.evaluation_plan.slots) == 42  # type: ignore[attr-defined]
    assert artifacts.summary.result_counts == B1ResultCounts()  # type: ignore[attr-defined]
    assert (
        B1EvaluationPlan.from_value(artifacts.evaluation_plan.to_value())  # type: ignore[attr-defined]
        == artifacts.evaluation_plan  # type: ignore[attr-defined]
    )
    assert (
        B1FreezeSummary.from_value(artifacts.summary.to_value())  # type: ignore[attr-defined]
        == artifacts.summary  # type: ignore[attr-defined]
    )
    coordinates = {
        (item.kind.value, item.anchor_ordinal)
        for item in artifacts.evaluation_plan.slots  # type: ignore[attr-defined]
    }
    assert coordinates == {
        ("initial-iid", 0),
        ("initial-ood", 0),
        ("progress-iid", 1),
        ("progress-iid", 2),
        ("final-iid", 0),
        ("final-ood", 0),
    }


def test_b1_writes_one_fresh_freeze_without_claims_or_results(tmp_path: Path) -> None:
    artifacts = _artifacts()
    output = (tmp_path / "b1").resolve()

    write_b1_freeze_artifacts(artifacts, output)  # type: ignore[arg-type]

    assert (output / "formal-run-manifest.json").is_file()
    assert (output / "formal-evaluation-plan.json").is_file()
    assert (output / "zero-result-counts.json").is_file()
    assert sorted(item.name for item in (output / "exact-inputs").iterdir()) == sorted(
        f"{kind.value}.json" for kind in AttemptBuilderKind
    )
    assert not (output / "ledger").exists()
    with pytest.raises(FileExistsError):
        write_b1_freeze_artifacts(artifacts, output)  # type: ignore[arg-type]


def test_b1_rejects_results_and_cross_arm_control_drift() -> None:
    with pytest.raises(ValueError):
        B1ResultCounts(training_terminal_count=1)

    artifacts = _artifacts()
    exact_inputs = artifacts.exact_inputs  # type: ignore[attr-defined]
    drifted = (
        *exact_inputs[:-1],
        replace(exact_inputs[-1], common_cross_arm_identity="drifted"),
    )
    with pytest.raises(ValueError):
        build_b1_freeze_artifacts(
            schedules=artifacts.schedules,  # type: ignore[attr-defined]
            protocol=artifacts.protocol,  # type: ignore[attr-defined]
            protocol_freeze=artifacts.protocol_freeze,  # type: ignore[attr-defined]
            exact_inputs=cast(tuple[PrivateBenchmarkAttemptInput, ...], drifted),
            b0_report_hash=stable_hash("b0-report"),
            f2_f3_proof_hash=stable_hash("f2-f3-proof"),
        )
