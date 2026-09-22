"""Closed formal-benchmark builders using only the private catalog route.

The public completion-smoke builder is intentionally not imported here.  The
fixed worker reads the private runtime pins from its exact input.  A B2 child
hydrates exactly the source benchmarks referenced by its frozen training
sequence; evaluation workers retain the complete eighteen-benchmark loader.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import assert_never

from skillev.application import SKILLEVApplication, TerminalComponents
from skillev.contracts import TrainingStepCommit
from skillev.evolution import PhiBudgetAuthority
from skillev.experiments import (
    AblationArm,
    FormalExecutionHardwareAttestation,
    FormalImplementationBuildAttestation,
)
from skillev.experiments.arm_events import (
    ArmEventType,
    LiveArmEventLog,
    read_arm_event_history,
)
from skillev.experiments.arms.builders import (
    ArmApplicationInputs,
    FlowOnlyArmApplicationInputs,
    build_capped_flow_weight_application,
    build_clipped_importance_application,
    build_no_bayesian_calibration_application,
    build_no_flow_weighting_application,
    build_posterior_mean_decision_application,
    build_residual_only_phase_application,
)
from skillev.experiments.arms.no_bayesian_contracts import FlowOnlyTrainingStepCommit
from skillev.experiments.checkpoint_namespace import checkpoint_storage_for_attempt
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptRequest,
    AttemptRunProgress,
    BudgetLedger,
    EventType,
    LiveAttemptEventLog,
    RuntimeEventEmitter,
)
from skillev.runtime.attempt_builders import BuiltAttempt, ExactAttemptApplication
from skillev.runtime.attempt_protocol import AttemptRunSummary, expected_source_log_layout
from skillev.runtime.event_log_reader import read_event_history
from skillev_private.benchmarks.catalog import OrderedPublicTaskProvider
from skillev_private.benchmarks.semantic_selection import (
    materialize_iid_training_subcatalog,
)

from .benchmark_attempt_input import PrivateBenchmarkAttemptInput
from .implementation_build import measure_formal_execution_hardware


def _clock() -> str:
    return "1970-01-01T00:00:00Z"


def _arm_builder(
    kind: AttemptBuilderKind,
) -> tuple[AblationArm, Callable[[ArmApplicationInputs], ExactAttemptApplication]]:
    match kind:
        case AttemptBuilderKind.NO_BAYESIAN:
            raise ValueError("no-Bayesian uses its independent flow-only builder")
        case AttemptBuilderKind.UNIT_FLOW:
            return AblationArm.NO_FLOW_WEIGHTING, build_no_flow_weighting_application
        case AttemptBuilderKind.CAPPED_FLOW:
            return AblationArm.CAPPED_FLOW_WEIGHT, build_capped_flow_weight_application
        case AttemptBuilderKind.CLIPPED_IMPORTANCE:
            return AblationArm.CLIPPED_IMPORTANCE, build_clipped_importance_application
        case AttemptBuilderKind.POSTERIOR_MEAN:
            return AblationArm.LCB_TO_MEAN, build_posterior_mean_decision_application
        case AttemptBuilderKind.RESIDUAL_ONLY_PHASE:
            return AblationArm.AND_TO_SINGLE_CONDITION, build_residual_only_phase_application
        case AttemptBuilderKind.FULL:
            raise ValueError("the full arm does not have an ablation builder")
        case _ as unreachable:
            assert_never(unreachable)


def _b1_terminal_validator(
    *,
    exact: PrivateBenchmarkAttemptInput,
    event_log_path: Path,
    arm_event_log_path: Path,
) -> Callable[[AttemptRunSummary], None]:
    """Project authoritative source-step library versions into B1 admission."""

    def validate(summary: AttemptRunSummary) -> None:
        if exact.builder_kind is AttemptBuilderKind.NO_BAYESIAN:
            versions = tuple(
                FlowOnlyTrainingStepCommit.from_value(event.payload).library_version
                for event in read_arm_event_history(arm_event_log_path)
                if event.event_type is ArmEventType.FLOW_ONLY_TRAINING_STEP_COMMITTED
            )
        else:
            versions = tuple(
                TrainingStepCommit.from_value(event.payload).library_version
                for event in read_event_history(event_log_path)
                if event.event_type is EventType.TRAINING_STEP_COMMITTED
            )
        exact.b1_run_admission.terminal.require_success(
            summary,
            ordered_training_library_versions=versions,
        )

    return validate


def build_private_benchmark_attempt(
    request: AttemptRequest,
) -> BuiltAttempt:
    """Build one formal graph from its private input and pinned catalog runtime."""

    if not isinstance(request, AttemptRequest):
        raise TypeError("formal benchmark builder requires AttemptRequest")
    exact = PrivateBenchmarkAttemptInput.read_verified(
        request.exact_input_path,
        expected_sha256=request.exact_input_sha256,
    )
    if exact.builder_kind is not request.builder_kind:
        raise ValueError("request builder differs from private exact input")
    identity = exact.public_identity(exact_input_sha256=request.exact_input_sha256)
    event_log_path = request.private_bundle_directory / "events.jsonl"
    arm_event_log_path = request.private_bundle_directory / "arm-events.jsonl"
    measured_build = exact.runtime.implementation_build_deployment.require_exact(
        exact.implementation_build
    )
    measured_hardware = measure_formal_execution_hardware()
    attestation = FormalImplementationBuildAttestation(
        attempt_id=request.attempt_id,
        builder_kind=request.builder_kind,
        exact_input_sha256=request.exact_input_sha256,
        public_identity_content_hash=identity.content_hash,
        formal_execution_content_hash=exact.protocol.formal_execution.content_hash,
        expected_implementation_build=exact.implementation_build,
        measured_implementation_build=measured_build,
    )
    event_log = LiveAttemptEventLog(
        event_log_path,
        run_id=request.run_id,
        attempt_id=request.attempt_id,
    )
    emitter = RuntimeEventEmitter(
        log=event_log,
        producer_id="skillev-formal-build",
        clock=_clock,
    )
    emitter.emit(EventType.FORMAL_IMPLEMENTATION_BUILD_ATTESTED, attestation.to_value())
    emitter.emit(
        EventType.FORMAL_EXECUTION_HARDWARE_ATTESTED,
        FormalExecutionHardwareAttestation(
            attempt_id=request.attempt_id,
            builder_kind=request.builder_kind,
            exact_input_sha256=request.exact_input_sha256,
            public_identity_content_hash=identity.content_hash,
            formal_execution_content_hash=exact.protocol.formal_execution.content_hash,
            measured_hardware=measured_hardware,
        ).to_value(),
    )
    loaded_catalog = exact.runtime.load_training_catalog(exact.catalog_bundle)
    try:
        catalog = materialize_iid_training_subcatalog(
            loaded_catalog.catalog,
            exact.runtime.load_iid_semantic_selection(),
            ordered_episode_ids=exact.training_sequence.task_ids,
        )
        tasks = exact.training_sequence.resolve(catalog)
        provider = OrderedPublicTaskProvider(tasks)
        sessions = catalog.route(tasks)
        checkpoint_storage = checkpoint_storage_for_attempt(
            exact.checkpoint_storage,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            builder_kind=request.builder_kind,
            exact_input_sha256=request.exact_input_sha256,
        )
        terminal = TerminalComponents(
            ledger=BudgetLedger(
                run_id=request.run_id,
                attempt_id=request.attempt_id,
                cap=exact.attempt_budget,
            ),
            authoring_authority=exact.authoring_authority,
            phi_budget=PhiBudgetAuthority(exact.phi_per_cycle_maximum),
        )
        if request.builder_kind is AttemptBuilderKind.FULL:
            full_application = SKILLEVApplication.build(
                backbone_config=exact.backbone,
                task_provider=provider,
                base_session_factory=sessions,
                seed_documents=exact.seed_documents,
                terminal_components=terminal,
                checkpoint_storage=checkpoint_storage,
                initial_checkpoint=exact.initial_checkpoint,
                public_identity=identity,
                event_log=event_log,
                clock=_clock,
                workflow_binding=exact.rollout_workflow,
            )
            return BuiltAttempt(
                application=full_application,
                run_plan=exact.run_plan,
                public_identity=identity,
                source_logs=expected_source_log_layout(request.builder_kind),
                close_source_logs_callback=loaded_catalog.close,
                validate_summary_callback=_b1_terminal_validator(
                    exact=exact,
                    event_log_path=event_log_path,
                    arm_event_log_path=arm_event_log_path,
                ),
            )

        run_progress = AttemptRunProgress.from_state(
            exact.run_plan,
            identity.initial_run_cursor,
        )
        snapshot_identity = identity.runtime_snapshot_identity()
        arm_log: LiveArmEventLog | None = None
        arm_application: ExactAttemptApplication
        if request.builder_kind is AttemptBuilderKind.NO_BAYESIAN:
            arm = AblationArm.SKILLFLOW_DISABLED
            arm_log = LiveArmEventLog(
                arm_event_log_path,
                arm=arm,
                run_id=request.run_id,
                attempt_id=request.attempt_id,
            )
            arm_application = build_no_bayesian_calibration_application(
                FlowOnlyArmApplicationInputs(
                    backbone_config=exact.backbone,
                    task_provider=provider,
                    base_session_factory=sessions,
                    seed_documents=exact.seed_documents,
                    terminal_components=terminal,
                    checkpoint_storage=checkpoint_storage,
                    initial_checkpoint=exact.initial_checkpoint,
                    public_identity=identity,
                    run_progress=run_progress,
                    snapshot_identity=snapshot_identity,
                    event_log=event_log,
                    clock=_clock,
                    rollout_workflow=exact.rollout_workflow,
                    arm_event_log=arm_log,
                )
            )
        else:
            arm, builder = _arm_builder(request.builder_kind)
            arm_application = builder(
                ArmApplicationInputs(
                    backbone_config=exact.backbone,
                    task_provider=provider,
                    base_session_factory=sessions,
                    seed_documents=exact.seed_documents,
                    terminal_components=terminal,
                    checkpoint_storage=checkpoint_storage,
                    initial_checkpoint=exact.initial_checkpoint,
                    public_identity=identity,
                    run_progress=run_progress,
                    snapshot_identity=snapshot_identity,
                    event_log=event_log,
                    clock=_clock,
                    rollout_workflow=exact.rollout_workflow,
                )
            )

        def close() -> None:
            if arm_log is not None:
                arm_log.close()
            loaded_catalog.close()

        return BuiltAttempt(
            application=arm_application,
            run_plan=exact.run_plan,
            public_identity=identity,
            source_logs=expected_source_log_layout(request.builder_kind),
            close_source_logs_callback=close,
            validate_summary_callback=_b1_terminal_validator(
                exact=exact,
                event_log_path=event_log_path,
                arm_event_log_path=arm_event_log_path,
            ),
        )
    except BaseException:
        loaded_catalog.close()
        raise


__all__ = ["build_private_benchmark_attempt"]
