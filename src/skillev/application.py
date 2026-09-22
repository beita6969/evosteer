"""The single explicit full-method application composition root."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from skillev.application_config import ApplicationConfig
from skillev.application_continuation import (
    ActionWireContinuation,
    HorizonContinuation,
    PlanContinuation,
    ReasoningContinuation,
    SkillExposureContinuation,
    TokenBudgetNoticeContinuation,
)
from skillev.application_snapshot import (
    ApplicationSnapshotFactory,
    require_full_execution_coherence,
)
from skillev.calibration import CalibrationEngine, CellQuery, ConfidenceMultiplier
from skillev.contracts import ContextFeature, LibraryInitialized
from skillev.domain_subset_continuation import DomainSubsetContinuation
from skillev.evolution import (
    AuthoringCallMaximum,
    AuthoringSamplingConfig,
    BaseModelSkillAuthor,
    BaseRolloutSessionFactory,
    EvolutionConfig,
    EvolutionDecisionPolicy,
    EvolutionLoop,
    EvolutionPhaseDetector,
    EvolutionProjectionView,
    FullEvolutionPolicy,
    PhaseTransitionDetector,
    PhiBudgetAuthority,
    RetrievingRolloutSessionFactory,
    SkillAuthor,
    SkillAuthoringAuthority,
    TaskConditionedSkillRetriever,
)
from skillev.format_review_continuation import FormatReviewContinuation
from skillev.healthbench_judge_continuation import HealthBenchJudgeContinuation
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
    AttemptRunCursorState,
    AttemptRunProgress,
    BudgetLedger,
    EventType,
    FullRuntimeExecutionState,
    LiveAttemptEventLog,
    OrderedTaskCursorState,
    RuntimeEventEmitter,
    RuntimeSnapshotIdentity,
    RuntimeSnapshotStore,
    SkillDocument,
    SkillLibrary,
    SkillLibraryState,
    StepAdapterPublisher,
    StepTransactionJournal,
    require_seed_library,
)
from skillev.runtime.attempt_run_plan import ExactAttemptRunPlan
from skillev.training import (
    FilesystemTrainingCheckpointStore,
    MethodProjectionPipeline,
    PosteriorEventProvenance,
    PrivateCheckpointStorageBinding,
    RolloutWorkflowBinding,
    RolloutWorkflowResources,
    TaskProvider,
    TrainingLoop,
    TrainingProjectionPipeline,
    TTBStepPreparer,
)
from skillev.training.stability import ReferencePolicyScorer


class ApplicationProjectionPipeline(
    TrainingProjectionPipeline,
    EvolutionProjectionView,
    Protocol,
):
    @property
    def posterior_provenance(self) -> PosteriorEventProvenance: ...

    def query_configured(self, skill_id: str, z: ContextFeature) -> CellQuery: ...

    def query_at_k(
        self, skill_id: str, z: ContextFeature, k: ConfidenceMultiplier
    ) -> CellQuery: ...


class TaskProviderFactory(Protocol):
    def from_exact_state(self, state: OrderedTaskCursorState) -> TaskProvider: ...


class RolloutGeneratorFactory(Protocol):
    def __call__(
        self,
        backbone: PolicyBackbone,
        resources: RolloutWorkflowResources,
    ) -> RolloutGenerator: ...


class SkillAuthorFactory(Protocol):
    def __call__(
        self,
        backbone: PolicyBackbone,
        config: EvolutionConfig,
        sampling: AuthoringSamplingConfig,
        ledger: BudgetLedger,
        emitter: RuntimeEventEmitter,
        maximum: AuthoringCallMaximum,
    ) -> SkillAuthor: ...


@dataclass(frozen=True, slots=True)
class FormalSharedRuntimeDependencies:
    """Method-neutral SGLang, workflow, adapter, and authoring dependencies."""

    rollout_generator_factory: RolloutGeneratorFactory
    workflow_resources: RolloutWorkflowResources
    step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]
    skill_author_factory: SkillAuthorFactory

    def __post_init__(self) -> None:
        if not callable(self.rollout_generator_factory):
            raise TypeError("formal shared runtime requires a rollout generator factory")
        if not isinstance(self.workflow_resources, RolloutWorkflowResources):
            raise TypeError("formal shared runtime requires workflow resources")
        if not callable(self.step_adapter_publisher_factory):
            raise TypeError("formal shared runtime requires an adapter publisher factory")
        if not callable(self.skill_author_factory):
            raise TypeError("formal shared runtime requires an external author factory")


@dataclass(frozen=True, slots=True)
class BayesianGradientRuntime:
    """BayesianImprove full/no-calibration distributed TTB transport."""

    coordinator: TTBStepPreparer

    def __post_init__(self) -> None:
        from skillev.training.distributed_ttb import DistributedTTBGradientCoordinator

        if not isinstance(self.coordinator, DistributedTTBGradientCoordinator):
            raise TypeError("Bayesian gradient runtime requires DistributedTTB")


@dataclass(frozen=True, slots=True)
class SkillFlowGradientRuntime:
    """Exact SkillFlow distributed gradient transport, separate from Bayesian TTB."""

    coordinator: object

    def __post_init__(self) -> None:
        from training.distributed_gradient import DistributedGradientClient

        if not isinstance(self.coordinator, DistributedGradientClient):
            raise TypeError("SkillFlow gradient runtime requires the upstream-derived client")


@dataclass(frozen=True, slots=True)
class FormalRuntimeDependencies:
    """Mandatory infrastructure for a fail-closed Protocol 10 application."""

    rollout_generator_factory: RolloutGeneratorFactory
    gradient_preparer: TTBStepPreparer
    workflow_resources: RolloutWorkflowResources
    step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]
    skill_author_factory: SkillAuthorFactory

    @property
    def shared(self) -> FormalSharedRuntimeDependencies:
        return FormalSharedRuntimeDependencies(
            rollout_generator_factory=self.rollout_generator_factory,
            workflow_resources=self.workflow_resources,
            step_adapter_publisher_factory=self.step_adapter_publisher_factory,
            skill_author_factory=self.skill_author_factory,
        )

    @property
    def bayesian_gradient(self) -> BayesianGradientRuntime:
        return BayesianGradientRuntime(self.gradient_preparer)

    def __post_init__(self) -> None:
        from skillev.training.distributed_ttb import DistributedTTBGradientCoordinator

        if not callable(self.rollout_generator_factory):
            raise TypeError("formal runtime requires a rollout generator factory")
        if not isinstance(self.gradient_preparer, DistributedTTBGradientCoordinator):
            raise TypeError("formal runtime requires the distributed TTB coordinator")
        if not isinstance(self.workflow_resources, RolloutWorkflowResources):
            raise TypeError("formal runtime requires shared workflow resources")
        if not callable(self.step_adapter_publisher_factory):
            raise TypeError("formal runtime requires an adapter publisher factory")
        if not callable(self.skill_author_factory):
            raise TypeError("formal runtime requires an external skill author factory")

    def require_bound(self, application: SKILLEVApplication) -> None:
        from skillev.evolution.external_sglang_authoring import ExternalSGLangSkillAuthor
        from skillev.rollout.external_sglang import ExternalSGLangRolloutGenerator
        from skillev.runtime.sglang_step_publisher import SGLangStepAdapterPublisher

        if not isinstance(application.generator, ExternalSGLangRolloutGenerator):
            raise TypeError("formal application did not bind external exact-token SGLang")
        if application.training_loop.gradient_preparer is not self.gradient_preparer:
            raise TypeError("formal application did not bind the distributed TTB coordinator")
        if application.training_loop.workflow_resources is not self.workflow_resources:
            raise TypeError("formal application did not bind the shared workflow resources")
        if not isinstance(
            application.evolution_loop.step_adapter_publisher,
            SGLangStepAdapterPublisher,
        ):
            raise TypeError("formal application did not bind the SGLang adapter publisher")
        if application.evolution_loop.step_transaction_journal is None:
            raise TypeError("formal application did not bind the step transaction journal")
        if not isinstance(application.evolution_loop.author, ExternalSGLangSkillAuthor):
            raise TypeError("formal application did not bind external base-model authoring")

    def require_flow_only_bound(self, application: object) -> None:
        """Reject every local fallback in the formal no-calibration arm."""

        from skillev.evolution.external_sglang_authoring import ExternalSGLangSkillAuthor
        from skillev.experiments.arms.builders import FlowOnlyApplication
        from skillev.rollout.external_sglang import ExternalSGLangRolloutGenerator
        from skillev.runtime.sglang_step_publisher import SGLangStepAdapterPublisher

        if not isinstance(application, FlowOnlyApplication):
            raise TypeError("formal no-calibration builder returned another application")
        generator = application.training_loop.generator
        if not isinstance(generator, ExternalSGLangRolloutGenerator):
            raise TypeError("formal no-calibration application did not bind external SGLang")
        if application.training_loop.gradient_preparer is not self.gradient_preparer:
            raise TypeError("formal no-calibration application did not bind distributed TTB")
        if application.training_loop.workflow_resources is not self.workflow_resources:
            raise TypeError("formal no-calibration application did not bind workflow resources")
        if not isinstance(
            application.evolution_loop.step_adapter_publisher,
            SGLangStepAdapterPublisher,
        ):
            raise TypeError("formal no-calibration application did not bind adapter publication")
        if application.evolution_loop.step_transaction_journal is None:
            raise TypeError("formal no-calibration application did not bind step transactions")
        if not isinstance(application.evolution_loop.author, ExternalSGLangSkillAuthor):
            raise TypeError("formal no-calibration application did not bind external authoring")


class ApplicationPublicIdentity(Protocol):
    """Dependency-light public identity required by the composition root.

    The concrete identity intentionally lives under ``experiments`` and imports
    :class:`ApplicationConfig`; expressing this boundary as a protocol keeps the
    application root free of that import cycle while making all executable
    controls derive from the one published identity.
    """

    @property
    def application_config(self) -> ApplicationConfig: ...

    @property
    def run_plan(self) -> ExactAttemptRunPlan: ...

    @property
    def initial_run_cursor(self) -> AttemptRunCursorState: ...

    @property
    def method_identity_hash(self) -> str: ...

    @property
    def sampling_schedule_hash(self) -> str: ...

    @property
    def ordered_task_sequence_hash(self) -> str: ...

    @property
    def phase_checkpoint_cycle_ordinals(self) -> tuple[int, ...]: ...

    def runtime_snapshot_identity(self) -> RuntimeSnapshotIdentity: ...


@dataclass(frozen=True, slots=True)
class TerminalComponents:
    ledger: BudgetLedger
    authoring_authority: SkillAuthoringAuthority
    phi_budget: PhiBudgetAuthority


@dataclass(slots=True)
class SKILLEVApplication:
    backbone: PolicyBackbone
    generator: RolloutGenerator
    library: SkillLibrary
    retriever: TaskConditionedSkillRetriever
    projections: ApplicationProjectionPipeline
    training_loop: TrainingLoop
    detector: EvolutionPhaseDetector
    evolution_loop: EvolutionLoop
    snapshot_store: RuntimeSnapshotStore
    emitter: RuntimeEventEmitter
    public_identity: ApplicationPublicIdentity
    run_progress: AttemptRunProgress
    snapshot_identity: RuntimeSnapshotIdentity

    @property
    def final_training_snapshot_directory(self) -> Path:
        """Private mandatory final checkpoint for the completed attempt."""

        return self.evolution_loop.final_training_snapshot_directory

    @classmethod
    def build(
        cls,
        *,
        backbone_config: QwenDeploymentConfig,
        task_provider: TaskProvider,
        base_session_factory: BaseRolloutSessionFactory,
        seed_documents: tuple[SkillDocument, ...],
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        initial_checkpoint: PrivateInitialCheckpointBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        generator: RolloutGenerator | None = None,
        generator_factory: RolloutGeneratorFactory | None = None,
        gradient_preparer: TTBStepPreparer | None = None,
        workflow_binding: RolloutWorkflowBinding | None = None,
        workflow_resources: RolloutWorkflowResources | None = None,
        step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]
        | None = None,
        skill_author_factory: SkillAuthorFactory | None = None,
        reference_scorer_factory: Callable[[PolicyBackbone], ReferencePolicyScorer] | None = None,
        step_transaction_journal: StepTransactionJournal | None = None,
    ) -> SKILLEVApplication:
        require_seed_library(seed_documents)
        config = _require_application_config(public_identity)
        run_plan = _require_run_plan(public_identity)
        backbone = build_qwen_policy_backbone(backbone_config)
        backbone.load_checkpoint(initial_checkpoint.directory)
        backbone.bind_initial_trainable_state(initial_checkpoint.trainable_state)
        library = SkillLibrary(SkillLibraryState.from_seed_documents(seed_documents))
        projections = MethodProjectionPipeline.fresh(
            diagnostics_config=config.diagnostics,
            calibration=CalibrationEngine(config.calibration),
            library_version=library.current_version,
        )
        detector = PhaseTransitionDetector.fresh(
            evolution_config=config.evolution,
            diagnostics_config=config.diagnostics,
            library_version=library.current_version,
        )
        return cls.build_from_components(
            backbone=backbone,
            task_provider=task_provider,
            base_session_factory=base_session_factory,
            terminal_components=terminal_components,
            checkpoint_storage=checkpoint_storage,
            event_log=event_log,
            clock=clock,
            library=library,
            projections=projections,
            detector=detector,
            policy=FullEvolutionPolicy(),
            public_identity=public_identity,
            run_progress=AttemptRunProgress.from_state(
                run_plan,
                _require_initial_cursor(public_identity, run_plan),
            ),
            generator=generator,
            generator_factory=generator_factory,
            gradient_preparer=gradient_preparer,
            workflow_binding=workflow_binding,
            workflow_resources=workflow_resources,
            step_adapter_publisher_factory=step_adapter_publisher_factory,
            skill_author_factory=skill_author_factory,
            reference_scorer_factory=reference_scorer_factory,
            step_transaction_journal=step_transaction_journal,
        )

    @classmethod
    def build_formal(
        cls,
        *,
        backbone_config: QwenDeploymentConfig,
        task_provider: TaskProvider,
        base_session_factory: BaseRolloutSessionFactory,
        seed_documents: tuple[SkillDocument, ...],
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        initial_checkpoint: PrivateInitialCheckpointBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        runtime: FormalRuntimeDependencies,
    ) -> SKILLEVApplication:
        """Build Protocol 10 full method without any local/default fallback."""

        application = cls.build(
            backbone_config=backbone_config,
            task_provider=task_provider,
            base_session_factory=base_session_factory,
            seed_documents=seed_documents,
            terminal_components=terminal_components,
            checkpoint_storage=checkpoint_storage,
            initial_checkpoint=initial_checkpoint,
            public_identity=public_identity,
            event_log=event_log,
            clock=clock,
            generator_factory=runtime.rollout_generator_factory,
            gradient_preparer=runtime.gradient_preparer,
            workflow_binding=runtime.workflow_resources.binding,
            workflow_resources=runtime.workflow_resources,
            step_adapter_publisher_factory=runtime.step_adapter_publisher_factory,
            skill_author_factory=runtime.skill_author_factory,
            step_transaction_journal=StepTransactionJournal(
                (Path(checkpoint_storage.directory) / "step-transactions").resolve()
            ),
        )
        runtime.require_bound(application)
        return application

    @classmethod
    def resume(
        cls,
        *,
        snapshot_directory: Path,
        backbone_config: QwenDeploymentConfig,
        task_provider_factory: TaskProviderFactory,
        base_session_factory: BaseRolloutSessionFactory,
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        initial_checkpoint: PrivateInitialCheckpointBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        generator: RolloutGenerator | None = None,
        generator_factory: RolloutGeneratorFactory | None = None,
        gradient_preparer: TTBStepPreparer | None = None,
        workflow_binding: RolloutWorkflowBinding | None = None,
        workflow_resources: RolloutWorkflowResources | None = None,
        step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]
        | None = None,
        skill_author_factory: SkillAuthorFactory | None = None,
        reference_scorer_factory: Callable[[PolicyBackbone], ReferencePolicyScorer] | None = None,
        step_transaction_journal: StepTransactionJournal | None = None,
        plan_continuation: PlanContinuation | None = None,
        horizon_continuation: HorizonContinuation | None = None,
        reasoning_continuation: ReasoningContinuation | None = None,
        action_wire_continuation: ActionWireContinuation | None = None,
        token_budget_continuation: TokenBudgetNoticeContinuation | None = None,
        skill_exposure_continuation: SkillExposureContinuation | None = None,
        domain_subset_continuation: DomainSubsetContinuation | None = None,
        healthbench_judge_continuation: HealthBenchJudgeContinuation | None = None,
        format_review_continuation: FormatReviewContinuation | None = None,
    ) -> SKILLEVApplication:
        """Hydrate an exact snapshot, optionally declaring one narrow condition change."""

        if (
            sum(
                item is not None
                for item in (
                    plan_continuation,
                    horizon_continuation,
                    reasoning_continuation,
                    action_wire_continuation,
                    token_budget_continuation,
                    skill_exposure_continuation,
                    domain_subset_continuation,
                    healthbench_judge_continuation,
                    format_review_continuation,
                )
            )
            > 1
        ):
            raise ValueError("select only one explicit condition continuation")
        continuation = (
            plan_continuation
            or horizon_continuation
            or reasoning_continuation
            or action_wire_continuation
            or token_budget_continuation
            or skill_exposure_continuation
            or domain_subset_continuation
            or healthbench_judge_continuation
            or format_review_continuation
        )
        config = _require_application_config(public_identity)
        run_plan = _require_run_plan(public_identity)
        snapshot_identity = public_identity.runtime_snapshot_identity()
        restore_identity = (
            snapshot_identity
            if continuation is None
            else continuation.require_target(public_identity)
        )

        artifact_store = FilesystemTrainingCheckpointStore(root=Path(checkpoint_storage.directory))
        metadata = artifact_store.load_metadata(snapshot_directory.resolve())
        if metadata.identity != restore_identity:
            raise ValueError("snapshot identity differs from public attempt identity")
        if continuation is not None:
            if metadata.optimizer_step != continuation.optimizer_step:
                raise ValueError("condition transition targets another saved step")
            if step_transaction_journal is None or step_transaction_journal.pending():
                raise ValueError("finish the original transaction before changing conditions")
        state = metadata.execution_state
        if not isinstance(state, FullRuntimeExecutionState):
            raise TypeError("full application requires a full runtime snapshot")
        require_full_execution_coherence(state, metadata.optimizer_step)
        if domain_subset_continuation is not None and (
            state.task_cursor.cursor != domain_subset_continuation.completed_tasks
        ):
            raise ValueError("domain removal differs from the saved task cursor")
        if plan_continuation is not None and public_identity.initial_run_cursor != (
            plan_continuation.migrate_cursor(state.run_cursor, run_plan)
        ):
            raise ValueError("append declaration must preserve the actual complete cursor")
        backbone = build_qwen_policy_backbone(backbone_config)
        backbone.load_checkpoint(initial_checkpoint.directory)
        backbone.bind_initial_trainable_state(initial_checkpoint.trainable_state)
        task_provider = task_provider_factory.from_exact_state(state.task_cursor)
        library = SkillLibrary(state.library)
        projections = MethodProjectionPipeline.from_runtime_state(
            diagnostics_config=config.diagnostics,
            calibration_config=config.calibration,
            state=state.projections,
        )
        detector = PhaseTransitionDetector.from_runtime_state(
            evolution_config=config.evolution,
            diagnostics_config=config.diagnostics,
            expected_library_version=library.current_version,
            state=state.detector,
        )
        application = cls._assemble(
            backbone=backbone,
            task_provider=task_provider,
            base_session_factory=base_session_factory,
            terminal_components=terminal_components,
            checkpoint_storage=checkpoint_storage,
            event_log=event_log,
            clock=clock,
            library=library,
            projections=projections,
            detector=detector,
            policy=FullEvolutionPolicy(),
            public_identity=public_identity,
            run_progress=AttemptRunProgress.from_state(
                run_plan,
                state.run_cursor
                if plan_continuation is None
                else plan_continuation.migrate_cursor(state.run_cursor, run_plan),
            ),
            generator=generator,
            generator_factory=generator_factory,
            gradient_preparer=gradient_preparer,
            workflow_binding=workflow_binding,
            workflow_resources=workflow_resources,
            step_adapter_publisher_factory=step_adapter_publisher_factory,
            skill_author_factory=skill_author_factory,
            reference_scorer_factory=reference_scorer_factory,
            step_transaction_journal=step_transaction_journal,
        )
        application.training_loop.restore_policy_optimizer_exact(
            snapshot_directory,
            expected_identity=restore_identity,
        )
        from skillev.application_recovery import reconcile_application_step

        journal = application.evolution_loop.step_transaction_journal
        if journal is None:
            raise RuntimeError("application resume requires its durable step journal")
        reconcile_application_step(application, journal, snapshot_directory)
        application.record_method_state()
        return application

    @classmethod
    def resume_formal(
        cls,
        *,
        snapshot_directory: Path,
        backbone_config: QwenDeploymentConfig,
        task_provider_factory: TaskProviderFactory,
        base_session_factory: BaseRolloutSessionFactory,
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        initial_checkpoint: PrivateInitialCheckpointBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        runtime: FormalRuntimeDependencies,
        horizon_continuation: HorizonContinuation | None = None,
        reasoning_continuation: ReasoningContinuation | None = None,
        action_wire_continuation: ActionWireContinuation | None = None,
        token_budget_continuation: TokenBudgetNoticeContinuation | None = None,
        skill_exposure_continuation: SkillExposureContinuation | None = None,
        domain_subset_continuation: DomainSubsetContinuation | None = None,
        healthbench_judge_continuation: HealthBenchJudgeContinuation | None = None,
        format_review_continuation: FormatReviewContinuation | None = None,
    ) -> SKILLEVApplication:
        """Resume Protocol 10 full method without any local/default fallback."""

        journal = StepTransactionJournal(
            (Path(checkpoint_storage.directory) / "step-transactions").resolve()
        )
        application = cls.resume(
            snapshot_directory=snapshot_directory,
            backbone_config=backbone_config,
            task_provider_factory=task_provider_factory,
            base_session_factory=base_session_factory,
            terminal_components=terminal_components,
            checkpoint_storage=checkpoint_storage,
            initial_checkpoint=initial_checkpoint,
            public_identity=public_identity,
            event_log=event_log,
            clock=clock,
            generator_factory=runtime.rollout_generator_factory,
            gradient_preparer=runtime.gradient_preparer,
            workflow_binding=runtime.workflow_resources.binding,
            workflow_resources=runtime.workflow_resources,
            step_adapter_publisher_factory=runtime.step_adapter_publisher_factory,
            skill_author_factory=runtime.skill_author_factory,
            step_transaction_journal=journal,
            horizon_continuation=horizon_continuation,
            reasoning_continuation=reasoning_continuation,
            action_wire_continuation=action_wire_continuation,
            token_budget_continuation=token_budget_continuation,
            skill_exposure_continuation=skill_exposure_continuation,
            domain_subset_continuation=domain_subset_continuation,
            healthbench_judge_continuation=healthbench_judge_continuation,
            format_review_continuation=format_review_continuation,
        )
        runtime.require_bound(application)
        return application

    @classmethod
    def build_from_components(
        cls,
        *,
        backbone: PolicyBackbone,
        task_provider: TaskProvider,
        base_session_factory: BaseRolloutSessionFactory,
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        library: SkillLibrary,
        projections: ApplicationProjectionPipeline,
        detector: EvolutionPhaseDetector,
        policy: EvolutionDecisionPolicy,
        run_progress: AttemptRunProgress,
        generator: RolloutGenerator | None = None,
        generator_factory: RolloutGeneratorFactory | None = None,
        gradient_preparer: TTBStepPreparer | None = None,
        workflow_binding: RolloutWorkflowBinding | None = None,
        workflow_resources: RolloutWorkflowResources | None = None,
        step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]
        | None = None,
        skill_author_factory: SkillAuthorFactory | None = None,
        reference_scorer_factory: Callable[[PolicyBackbone], ReferencePolicyScorer] | None = None,
        step_transaction_journal: StepTransactionJournal | None = None,
    ) -> SKILLEVApplication:
        run_plan = _require_run_plan(public_identity)
        if run_progress.plan != run_plan:
            raise ValueError("application run progress differs from public run plan")
        application = cls._assemble(
            backbone=backbone,
            task_provider=task_provider,
            base_session_factory=base_session_factory,
            terminal_components=terminal_components,
            checkpoint_storage=checkpoint_storage,
            event_log=event_log,
            clock=clock,
            library=library,
            projections=projections,
            detector=detector,
            policy=policy,
            public_identity=public_identity,
            run_progress=run_progress,
            generator=generator,
            generator_factory=generator_factory,
            gradient_preparer=gradient_preparer,
            workflow_binding=workflow_binding,
            workflow_resources=workflow_resources,
            step_adapter_publisher_factory=step_adapter_publisher_factory,
            skill_author_factory=skill_author_factory,
            reference_scorer_factory=reference_scorer_factory,
            step_transaction_journal=step_transaction_journal,
        )
        application._emit_library_initialized()
        application.evolution_loop.save_initial_snapshot()
        return application

    @classmethod
    def _assemble(
        cls,
        *,
        backbone: PolicyBackbone,
        task_provider: TaskProvider,
        base_session_factory: BaseRolloutSessionFactory,
        terminal_components: TerminalComponents,
        checkpoint_storage: PrivateCheckpointStorageBinding,
        public_identity: ApplicationPublicIdentity,
        event_log: LiveAttemptEventLog,
        clock: Callable[[], str],
        library: SkillLibrary,
        projections: ApplicationProjectionPipeline,
        detector: EvolutionPhaseDetector,
        policy: EvolutionDecisionPolicy,
        run_progress: AttemptRunProgress,
        generator: RolloutGenerator | None = None,
        generator_factory: RolloutGeneratorFactory | None = None,
        gradient_preparer: TTBStepPreparer | None = None,
        workflow_binding: RolloutWorkflowBinding | None = None,
        workflow_resources: RolloutWorkflowResources | None = None,
        step_adapter_publisher_factory: Callable[[PolicyBackbone], StepAdapterPublisher]
        | None = None,
        skill_author_factory: SkillAuthorFactory | None = None,
        reference_scorer_factory: Callable[[PolicyBackbone], ReferencePolicyScorer] | None = None,
        step_transaction_journal: StepTransactionJournal | None = None,
    ) -> SKILLEVApplication:
        # All public build/resume paths own the same durable step transaction.
        # Omitting a transport-specific journal must not disable recovery.
        if step_transaction_journal is None:
            step_transaction_journal = StepTransactionJournal(
                (Path(checkpoint_storage.directory) / "step-transactions").resolve()
            )
        config = _require_application_config(public_identity)
        run_plan = _require_run_plan(public_identity)
        if run_progress.plan != run_plan:
            raise ValueError("application run progress differs from public run plan")
        _validate_library_task_family_universe(
            library,
            terminal_components.authoring_authority.allowed_task_families,
        )
        snapshot_identity = public_identity.runtime_snapshot_identity()
        # The actual scorer and the persisted run identity must describe the same
        # reward condition; this also covers public resume's assembly path.
        scoring = getattr(base_session_factory, "terminal_evaluation_conditions_json", None)
        if scoring != snapshot_identity.terminal_evaluation_conditions_json:
            raise ValueError("terminal evaluator configuration differs from the training identity")
        if (
            getattr(base_session_factory, "task_feature_mapping_version", None)
            != snapshot_identity.task_feature_mapping_version
        ):
            raise ValueError("task feature mapping differs from the training identity")

        emitter = RuntimeEventEmitter(
            log=event_log,
            producer_id="skillev-application",
            clock=clock,
        )
        if generator is not None and generator_factory is not None:
            raise ValueError("provide either a rollout generator or a generator factory")
        active_workflow_binding = (
            workflow_resources.binding
            if workflow_binding is None and workflow_resources is not None
            else workflow_binding or RolloutWorkflowBinding()
        )
        if workflow_resources is not None and workflow_resources.binding != active_workflow_binding:
            raise ValueError("rollout workflow resources differ from the binding")
        active_workflow_resources = (
            workflow_resources
            if workflow_resources is not None
            else RolloutWorkflowResources(active_workflow_binding)
        )
        active_generator = (
            generator
            if generator is not None
            else (
                generator_factory(backbone, active_workflow_resources)
                if generator_factory is not None
                else LocalPolicyGenerator(
                    backbone,
                    # The Transformers cache is intentionally a single-episode
                    # optimization. Concurrent resident trajectories can
                    # interleave requests even when model calls are serialized,
                    # so they must use independent full prefills.
                    episode_cache_enabled=(active_workflow_binding.max_resident_trajectories == 1),
                )
            )
        )
        if active_generator.tokenizer.tokenizer_id != backbone.tokenizer.tokenizer_id:
            raise ValueError("rollout generator and training backbone use different tokenizers")
        retriever = TaskConditionedSkillRetriever(
            library=library,
        )
        session_factory = RetrievingRolloutSessionFactory(
            base_factory=base_session_factory,
            retriever=retriever,
        )
        artifact_store = FilesystemTrainingCheckpointStore(root=Path(checkpoint_storage.directory))
        snapshot_store = RuntimeSnapshotStore(artifact_store)
        training_loop = TrainingLoop(
            reference_scorer=None
            if reference_scorer_factory is None
            else reference_scorer_factory(backbone),
            backbone=backbone,
            generator=active_generator,
            task_provider=task_provider,
            session_factory=session_factory,
            context_assembler=config.trainer.rollout.context_assembler(
                maximum_h0_tokens=config.maximum_h0_tokens,
            ),
            library=library,
            config=config.trainer,
            ledger=terminal_components.ledger,
            emitter=emitter,
            clock=clock,
            projections=projections,
            checkpoint_store=artifact_store,
            sampling_schedule_hash=public_identity.sampling_schedule_hash,
            ordered_task_sequence_hash=public_identity.ordered_task_sequence_hash,
            gradient_preparer=gradient_preparer,
            workflow_binding=active_workflow_binding,
            workflow_resources=active_workflow_resources,
        )
        author_maximum = AuthoringCallMaximum(
            input_tokens=config.evolution.max_authoring_prompt_tokens,
            output_tokens=config.evolution.max_authoring_completion_tokens,
        )
        author = (
            BaseModelSkillAuthor(
                backbone=backbone,
                config=config.evolution,
                sampling=config.authoring_sampling,
                ledger=terminal_components.ledger,
                emitter=emitter,
                maximum=author_maximum,
            )
            if skill_author_factory is None
            else skill_author_factory(
                backbone,
                config.evolution,
                config.authoring_sampling,
                terminal_components.ledger,
                emitter,
                author_maximum,
            )
        )
        snapshot_factory = ApplicationSnapshotFactory(
            training_loop=training_loop,
            task_provider=task_provider,
            run_progress=run_progress,
            snapshot_identity=snapshot_identity,
            library=library,
            projections=projections,
            detector=detector,
        )
        step_adapter_publisher = (
            None
            if step_adapter_publisher_factory is None
            else step_adapter_publisher_factory(backbone)
        )
        evolution_loop = EvolutionLoop(
            training_loop=training_loop,
            projections=projections,
            detector=detector,
            library=library,
            author=author,
            policy=policy,
            authority=terminal_components.authoring_authority,
            phi_budget=terminal_components.phi_budget,
            config=config.evolution,
            emitter=emitter,
            snapshot_store=snapshot_store,
            snapshot_factory=snapshot_factory,
            run_progress=run_progress,
            step_adapter_publisher=step_adapter_publisher,
            phase_checkpoint_cycle_ordinals=public_identity.phase_checkpoint_cycle_ordinals,
            step_transaction_journal=step_transaction_journal,
        )
        return cls(
            backbone=backbone,
            generator=active_generator,
            library=library,
            retriever=retriever,
            projections=projections,
            training_loop=training_loop,
            detector=detector,
            evolution_loop=evolution_loop,
            snapshot_store=snapshot_store,
            emitter=emitter,
            public_identity=public_identity,
            run_progress=run_progress,
            snapshot_identity=snapshot_identity,
        )

    def _emit_library_initialized(self) -> None:
        state = self.library.state
        self.emitter.emit(
            EventType.LIBRARY_INITIALIZED,
            LibraryInitialized(
                documents=tuple(document.to_value() for document in self.library.all_documents()),
                active_skill_ids=state.active_skill_ids,
                library_version=state.current_version,
                initial_optimizer_step=self.training_loop.optimizer_step,
                method_identity_hash=self.public_identity.method_identity_hash,
                run_cursor=self.run_progress.state.to_source_value(),
            ).to_value(),
        )
        self.record_method_state()

    def record_method_state(self) -> None:
        """Log resolved controls and identities, not a historical config filename."""
        from skillev.application_reporting import resolved_method_state

        self.emitter.emit(EventType.METHOD_STATE_RECORDED, resolved_method_state(self))


def _validate_library_task_family_universe(
    library: SkillLibrary,
    task_family_universe: tuple[str, ...],
) -> None:
    if not task_family_universe or "*" in task_family_universe:
        raise ValueError("authoring authority requires a finite task-family universe")
    universe = set(task_family_universe)
    for document in library.active_documents():
        families = document.applicability.task_families
        if families != ("*",) and not set(families) <= universe:
            raise ValueError("active skill targets a family outside the formal universe")


def _require_application_config(identity: ApplicationPublicIdentity) -> ApplicationConfig:
    config = identity.application_config
    if not isinstance(config, ApplicationConfig):
        raise TypeError("public identity application config is incompatible")
    return config


def _require_run_plan(identity: ApplicationPublicIdentity) -> ExactAttemptRunPlan:
    plan = identity.run_plan
    if not isinstance(plan, ExactAttemptRunPlan):
        raise TypeError("public identity run plan is incompatible")
    return plan


def _require_initial_cursor(
    identity: ApplicationPublicIdentity,
    plan: ExactAttemptRunPlan,
) -> AttemptRunCursorState:
    cursor = identity.initial_run_cursor
    if not isinstance(cursor, AttemptRunCursorState):
        raise TypeError("public identity initial run cursor is incompatible")
    cursor.require_plan(plan)
    return cursor


__all__ = [
    "ApplicationConfig",
    "ApplicationPublicIdentity",
    "RolloutGeneratorFactory",
    "SKILLEVApplication",
    "TaskProviderFactory",
    "TerminalComponents",
]
