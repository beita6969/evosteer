"""Six explicit one-axis application builders for the declared ablations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from skillev.application import (
    ApplicationConfig,
    ApplicationPublicIdentity,
    RolloutGeneratorFactory,
    SkillAuthorFactory,
    SKILLEVApplication,
    TerminalComponents,
)
from skillev.calibration import CalibrationEngine
from skillev.diagnostics import OnlineFlowDiagnostics
from skillev.evolution import (
    AuthoringCallMaximum,
    BaseModelSkillAuthor,
    BaseRolloutSessionFactory,
    EvolutionDecisionPolicy,
    EvolutionPhaseDetector,
    FullEvolutionPolicy,
    PhaseTransitionDetector,
    RetrievingRolloutSessionFactory,
    TaskConditionedSkillRetriever,
)
from skillev.experiments.arm_events import ArmEventType, LiveArmEventLog
from skillev.policy import (
    PolicyBackbone,
    PrivateInitialCheckpointBinding,
    QwenDeploymentConfig,
)
from skillev.policy.hf_backbone import build_qwen_policy_backbone
from skillev.rollout import (
    LocalPolicyGenerator,
    RolloutGenerator,
)
from skillev.runtime import (
    AttemptRunProgress,
    FlowOnlyRuntimeExecutionState,
    LiveAttemptEventLog,
    RuntimeEventEmitter,
    RuntimeSnapshotIdentity,
    RuntimeSnapshotStore,
    SkillDocument,
    SkillLibrary,
    SkillLibraryState,
    StepAdapterPublisher,
    StepTransactionJournal,
)
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan
from skillev.training import (
    FilesystemTrainingCheckpointStore,
    MethodProjectionPipeline,
    OnlineDiagnosticsProjection,
    PrivateCheckpointStorageBinding,
    RolloutWorkflowBinding,
    RolloutWorkflowResources,
    TaskProvider,
    TTBStepPreparer,
)

from ..protocol import CappedFlowWeightArmProtocol, ClippedImportanceArmProtocol
from .capped_flow_weight import CappedFlowCalibrationEngine, CappedFlowWeightConfig
from .clipped_importance import ClippedOnlineFlowDiagnostics
from .no_bayesian_calibration import FlowOnlyEvolutionPolicy, FlowOnlyProjectionPipeline
from .no_bayesian_contracts import FlowOnlyLibraryInitialized
from .no_bayesian_loop import ArmFlowOnlyResultSink, FlowOnlyEvolutionLoop
from .no_bayesian_training import FlowOnlyTrainingLoop
from .no_flow_weighting import UnitFlowCalibrationEngine
from .posterior_mean_decision import PosteriorMeanEvolutionPolicy
from .residual_only_phase import ResidualOnlyPhaseDetector


@dataclass(frozen=True, slots=True)
class ArmApplicationInputs:
    backbone_config: QwenDeploymentConfig
    task_provider: TaskProvider
    base_session_factory: BaseRolloutSessionFactory
    seed_documents: tuple[SkillDocument, ...]
    terminal_components: TerminalComponents
    checkpoint_storage: PrivateCheckpointStorageBinding
    initial_checkpoint: PrivateInitialCheckpointBinding
    public_identity: ApplicationPublicIdentity
    run_progress: AttemptRunProgress
    snapshot_identity: RuntimeSnapshotIdentity
    event_log: LiveAttemptEventLog
    clock: Callable[[], str]
    rollout_workflow: RolloutWorkflowBinding

    def __post_init__(self) -> None:
        if not self.seed_documents:
            raise ValueError("an ablation application requires a non-empty seed library")
        if self.run_progress.plan != self.public_identity.run_plan:
            raise ValueError("arm run progress differs from public run plan")
        if self.snapshot_identity != self.public_identity.runtime_snapshot_identity():
            raise ValueError("arm snapshot identity differs from public identity")
        if not isinstance(self.checkpoint_storage, PrivateCheckpointStorageBinding):
            raise TypeError("arm inputs require private checkpoint storage")
        if not isinstance(self.initial_checkpoint, PrivateInitialCheckpointBinding):
            raise TypeError("arm inputs require private initial checkpoint")
        if not isinstance(self.rollout_workflow, RolloutWorkflowBinding):
            raise TypeError("arm inputs require a rollout workflow binding")

    @property
    def config(self) -> ApplicationConfig:
        return self.public_identity.application_config

    @property
    def run_plan(self) -> ExactAttemptRunPlan:
        return self.public_identity.run_plan


@dataclass(frozen=True, slots=True)
class FlowOnlyArmApplicationInputs(ArmApplicationInputs):
    """Extra independent source stream used only by the no-Bayesian arm."""

    arm_event_log: LiveArmEventLog | None
    generator: RolloutGenerator | None = None
    generator_factory: RolloutGeneratorFactory | None = None
    gradient_preparer: TTBStepPreparer | None = None
    workflow_resources: RolloutWorkflowResources | None = None
    step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher] | None = None
    skill_author_factory: SkillAuthorFactory | None = None
    resume_snapshot_directory: Path | None = None
    restored_execution_state: FlowOnlyRuntimeExecutionState | None = None

    def __post_init__(self) -> None:
        ArmApplicationInputs.__post_init__(self)
        if (self.resume_snapshot_directory is None) != (self.restored_execution_state is None):
            raise ValueError("flow-only resume snapshot and execution state must be paired")


@dataclass(slots=True)
class FlowOnlyApplication:
    backbone: PolicyBackbone
    library: SkillLibrary
    projections: FlowOnlyProjectionPipeline
    training_loop: FlowOnlyTrainingLoop
    detector: PhaseTransitionDetector
    evolution_loop: FlowOnlyEvolutionLoop
    emitter: RuntimeEventEmitter
    arm_event_log: LiveArmEventLog | None

    @property
    def final_training_snapshot_directory(self) -> Path:
        return self.evolution_loop.final_training_snapshot_directory


def _backbone_and_library(
    inputs: ArmApplicationInputs,
) -> tuple[PolicyBackbone, SkillLibrary]:
    backbone = build_qwen_policy_backbone(inputs.backbone_config)
    backbone.load_checkpoint(inputs.initial_checkpoint.directory)
    backbone.bind_initial_trainable_state(inputs.initial_checkpoint.trainable_state)
    return backbone, SkillLibrary(SkillLibraryState.from_seed_documents(inputs.seed_documents))


def build_no_bayesian_calibration_application(
    inputs: FlowOnlyArmApplicationInputs,
) -> FlowOnlyApplication:
    backbone, fresh_library = _backbone_and_library(inputs)
    restored = inputs.restored_execution_state
    library = fresh_library if restored is None else SkillLibrary(restored.library)
    projections = (
        FlowOnlyProjectionPipeline.fresh(
            inputs.config.diagnostics,
            library_version=library.current_version,
        )
        if restored is None
        else FlowOnlyProjectionPipeline.from_runtime_state(
            inputs.config.diagnostics,
            restored.projections,
        )
    )
    emitter = RuntimeEventEmitter(
        log=inputs.event_log,
        producer_id="skillev-bayesian-improve-no-calibration",
        clock=inputs.clock,
    )
    if inputs.generator is not None and inputs.generator_factory is not None:
        raise ValueError("provide either a flow-only generator or generator factory")
    active_workflow_resources = (
        inputs.workflow_resources
        if inputs.workflow_resources is not None
        else RolloutWorkflowResources(inputs.rollout_workflow)
    )
    if active_workflow_resources.binding != inputs.rollout_workflow:
        raise ValueError("flow-only workflow resources differ from the binding")
    generator = (
        inputs.generator
        if inputs.generator is not None
        else (
            inputs.generator_factory(backbone, active_workflow_resources)
            if inputs.generator_factory is not None
            else LocalPolicyGenerator(
                backbone,
                episode_cache_enabled=(inputs.rollout_workflow.max_resident_trajectories == 1),
            )
        )
    )
    if generator.tokenizer.tokenizer_id != backbone.tokenizer.tokenizer_id:
        raise ValueError("rollout generator and training backbone use different tokenizers")
    retriever = TaskConditionedSkillRetriever(
        library=library,
    )
    session_factory = RetrievingRolloutSessionFactory(
        base_factory=inputs.base_session_factory,
        retriever=retriever,
    )
    if restored is None and inputs.arm_event_log is not None:
        inputs.arm_event_log.append(
            ArmEventType.FLOW_ONLY_LIBRARY_INITIALIZED,
            FlowOnlyLibraryInitialized(
                documents=tuple(item.to_value() for item in library.all_documents()),
                active_skill_ids=library.active_skill_ids,
                library_version=library.current_version,
                initial_optimizer_step=0,
                method_identity_hash=inputs.public_identity.method_identity_hash,
                run_cursor=inputs.run_progress.state.to_source_value(),
            ).to_value(),
        )
    checkpoint_store = FilesystemTrainingCheckpointStore(root=inputs.checkpoint_storage.directory)
    training_loop = FlowOnlyTrainingLoop(
        backbone=backbone,
        generator=generator,
        task_provider=inputs.task_provider,
        session_factory=session_factory,
        context_assembler=inputs.config.trainer.rollout.context_assembler(
            maximum_h0_tokens=inputs.config.maximum_h0_tokens
        ),
        library=library,
        config=inputs.config.trainer,
        ledger=inputs.terminal_components.ledger,
        emitter=emitter,
        clock=inputs.clock,
        projections=projections,
        checkpoint_store=checkpoint_store,
        arm_event_log=inputs.arm_event_log,
        sampling_schedule_hash=inputs.public_identity.sampling_schedule_hash,
        ordered_task_sequence_hash=inputs.public_identity.ordered_task_sequence_hash,
        gradient_preparer=inputs.gradient_preparer,
        workflow_binding=inputs.rollout_workflow,
        workflow_resources=active_workflow_resources,
    )
    detector = (
        PhaseTransitionDetector.fresh(
            evolution_config=inputs.config.evolution,
            diagnostics_config=inputs.config.diagnostics,
            library_version=library.current_version,
        )
        if restored is None
        else PhaseTransitionDetector.from_runtime_state(
            evolution_config=inputs.config.evolution,
            diagnostics_config=inputs.config.diagnostics,
            expected_library_version=library.current_version,
            state=restored.detector,
        )
    )
    evolution_loop = FlowOnlyEvolutionLoop(
        training_loop=training_loop,
        projections=projections,
        detector=detector,
        library=library,
        author=(
            BaseModelSkillAuthor(
                backbone=backbone,
                config=inputs.config.evolution,
                sampling=inputs.config.authoring_sampling,
                ledger=inputs.terminal_components.ledger,
                emitter=emitter,
                maximum=AuthoringCallMaximum(
                    input_tokens=inputs.config.evolution.max_authoring_prompt_tokens,
                    output_tokens=inputs.config.evolution.max_authoring_completion_tokens,
                ),
            )
            if inputs.skill_author_factory is None
            else inputs.skill_author_factory(
                backbone,
                inputs.config.evolution,
                inputs.config.authoring_sampling,
                inputs.terminal_components.ledger,
                emitter,
                AuthoringCallMaximum(
                    input_tokens=inputs.config.evolution.max_authoring_prompt_tokens,
                    output_tokens=inputs.config.evolution.max_authoring_completion_tokens,
                ),
            )
        ),
        policy=FlowOnlyEvolutionPolicy(),
        authority=inputs.terminal_components.authoring_authority,
        emitter=emitter,
        result_sink=(
            None if inputs.arm_event_log is None else ArmFlowOnlyResultSink(inputs.arm_event_log)
        ),
        phi_budget=inputs.terminal_components.phi_budget,
        snapshot_store=RuntimeSnapshotStore(checkpoint_store),
        config=inputs.config.evolution,
        run_progress=inputs.run_progress,
        snapshot_identity=inputs.snapshot_identity,
        phase_checkpoint_cycle_ordinals=inputs.public_identity.phase_checkpoint_cycle_ordinals,
        step_adapter_publisher=(
            None
            if inputs.step_adapter_publisher_factory is None
            else inputs.step_adapter_publisher_factory(backbone)
        ),
        step_transaction_journal=(
            None
            if inputs.step_adapter_publisher_factory is None
            else StepTransactionJournal(
                (Path(inputs.checkpoint_storage.directory) / "step-transactions").resolve()
            )
        ),
    )
    application = FlowOnlyApplication(
        backbone=backbone,
        library=library,
        projections=projections,
        training_loop=training_loop,
        detector=detector,
        evolution_loop=evolution_loop,
        emitter=emitter,
        arm_event_log=inputs.arm_event_log,
    )
    if inputs.resume_snapshot_directory is not None:
        application.training_loop.restore_policy_optimizer_exact(
            inputs.resume_snapshot_directory,
            expected_identity=inputs.snapshot_identity,
        )
    return application


def build_no_flow_weighting_application(
    inputs: ArmApplicationInputs,
) -> SKILLEVApplication:
    backbone, library = _backbone_and_library(inputs)
    return _build_full_shape(
        inputs,
        backbone=backbone,
        library=library,
        diagnostics=OnlineFlowDiagnostics.fresh(
            inputs.config.diagnostics,
            library_version=library.current_version,
        ),
        calibration=UnitFlowCalibrationEngine(inputs.config.calibration),
        detector=PhaseTransitionDetector.fresh(
            evolution_config=inputs.config.evolution,
            diagnostics_config=inputs.config.diagnostics,
            library_version=library.current_version,
        ),
        policy=FullEvolutionPolicy(),
    )


def build_capped_flow_weight_application(
    inputs: ArmApplicationInputs,
) -> SKILLEVApplication:
    backbone, library = _backbone_and_library(inputs)
    arm_protocol = CappedFlowWeightArmProtocol()
    return _build_full_shape(
        inputs,
        backbone=backbone,
        library=library,
        diagnostics=OnlineFlowDiagnostics.fresh(
            inputs.config.diagnostics,
            library_version=library.current_version,
        ),
        calibration=CappedFlowCalibrationEngine(
            inputs.config.calibration,
            CappedFlowWeightConfig(cap=arm_protocol.CAP),
        ),
        detector=PhaseTransitionDetector.fresh(
            evolution_config=inputs.config.evolution,
            diagnostics_config=inputs.config.diagnostics,
            library_version=library.current_version,
        ),
        policy=FullEvolutionPolicy(),
    )


def build_clipped_importance_application(
    inputs: ArmApplicationInputs,
) -> SKILLEVApplication:
    backbone, library = _backbone_and_library(inputs)
    arm_protocol = ClippedImportanceArmProtocol()
    return _build_full_shape(
        inputs,
        backbone=backbone,
        library=library,
        diagnostics=ClippedOnlineFlowDiagnostics.fresh(
            inputs.config.diagnostics,
            clip=arm_protocol.CLIP,
            library_version=library.current_version,
        ),
        calibration=CalibrationEngine(inputs.config.calibration),
        detector=PhaseTransitionDetector.fresh(
            evolution_config=inputs.config.evolution,
            diagnostics_config=inputs.config.diagnostics,
            library_version=library.current_version,
        ),
        policy=FullEvolutionPolicy(),
    )


def build_posterior_mean_decision_application(
    inputs: ArmApplicationInputs,
) -> SKILLEVApplication:
    backbone, library = _backbone_and_library(inputs)
    return _build_full_shape(
        inputs,
        backbone=backbone,
        library=library,
        diagnostics=OnlineFlowDiagnostics.fresh(
            inputs.config.diagnostics,
            library_version=library.current_version,
        ),
        calibration=CalibrationEngine(inputs.config.calibration),
        detector=PhaseTransitionDetector.fresh(
            evolution_config=inputs.config.evolution,
            diagnostics_config=inputs.config.diagnostics,
            library_version=library.current_version,
        ),
        policy=PosteriorMeanEvolutionPolicy(),
    )


def build_residual_only_phase_application(
    inputs: ArmApplicationInputs,
) -> SKILLEVApplication:
    backbone, library = _backbone_and_library(inputs)
    return _build_full_shape(
        inputs,
        backbone=backbone,
        library=library,
        diagnostics=OnlineFlowDiagnostics.fresh(
            inputs.config.diagnostics,
            library_version=library.current_version,
        ),
        calibration=CalibrationEngine(inputs.config.calibration),
        detector=ResidualOnlyPhaseDetector.fresh(
            inputs.config.diagnostics,
            library_version=library.current_version,
        ),
        policy=FullEvolutionPolicy(),
    )


def _build_full_shape(
    inputs: ArmApplicationInputs,
    *,
    backbone: PolicyBackbone,
    library: SkillLibrary,
    diagnostics: OnlineDiagnosticsProjection,
    calibration: CalibrationEngine,
    detector: EvolutionPhaseDetector,
    policy: EvolutionDecisionPolicy,
) -> SKILLEVApplication:
    projections = MethodProjectionPipeline.from_fresh_components(
        diagnostics=diagnostics,
        calibration=calibration,
    )
    return SKILLEVApplication.build_from_components(
        backbone=backbone,
        task_provider=inputs.task_provider,
        base_session_factory=inputs.base_session_factory,
        terminal_components=inputs.terminal_components,
        checkpoint_storage=inputs.checkpoint_storage,
        public_identity=inputs.public_identity,
        event_log=inputs.event_log,
        clock=inputs.clock,
        library=library,
        projections=projections,
        detector=detector,
        policy=policy,
        run_progress=inputs.run_progress,
        workflow_binding=inputs.rollout_workflow,
    )


__all__ = [
    "ArmApplicationInputs",
    "FlowOnlyApplication",
    "build_capped_flow_weight_application",
    "build_clipped_importance_application",
    "build_no_bayesian_calibration_application",
    "build_no_flow_weighting_application",
    "build_posterior_mean_decision_application",
    "build_residual_only_phase_application",
]
