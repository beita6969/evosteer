"""Private read-only worker helper for frozen progress/OOD evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillev.audit import (
    AuditResources,
    FormalArtifactResolver,
    audit_published_attempt,
    require_exact_formal_artifacts,
)
from skillev.audit.tokenizer import ResolvedAuditTokenizer, TokenizerArtifactResolver
from skillev.contracts import stable_hash
from skillev.experiments import (
    ExecutionHardwareIdentity,
    FormalRunLedger,
    ImplementationBuildIdentity,
)
from skillev.policy import PublicTokenizerIdentity, QwenDeploymentConfig
from skillev.runtime import PublishedSuccessfulAttemptBundle
from skillev_private.evaluation.frozen_runner import (
    FrozenEvaluationExecutionConfig,
    FrozenEvaluationRunner,
)
from skillev_private.evaluation.result_contracts import PrivateEvaluationEpisodeResult
from skillev_private.frozen_state import FrozenEvaluationInput, FrozenInferenceState
from skillev_private.initial_baseline import (
    InitialBaselineEvaluationInput,
    InitialBaselineInferenceState,
)
from skillev_private.phase_anchor import (
    PhaseAnchorEvaluationInput,
    PhaseAnchorInferenceState,
    PhaseCheckpointArtifactResolver,
)

from .benchmark_attempt_input import PrivateBenchmarkAttemptInput
from .formal_evaluation_manifest import (
    FormalEvaluationKind,
    FormalEvaluationLaunch,
    FormalEvaluationLedger,
    FormalEvaluationSlot,
    FormalEvaluationTerminal,
)
from .implementation_build import measure_formal_execution_hardware


@dataclass(frozen=True, slots=True)
class FormalEvaluationExecutionResult:
    episodes: tuple[PrivateEvaluationEpisodeResult, ...]
    terminal: FormalEvaluationTerminal


@dataclass(frozen=True, slots=True)
class _AdmittedEvaluation:
    ledger: FormalEvaluationLedger
    slot: FormalEvaluationSlot
    implementation_build: ImplementationBuildIdentity
    execution_hardware: ExecutionHardwareIdentity


@dataclass(frozen=True, slots=True)
class FrozenEvaluationDeployment:
    """Private locations needed to reconstruct the sole formal evaluation graph."""

    source_training_input_path: Path
    published_training_bundle_directory: Path
    formal_run_ledger_directory: Path
    output_directory: Path
    run_id: str
    formal_artifact_resolver: FormalArtifactResolver
    formal_evaluation_launch: FormalEvaluationLaunch

    def __post_init__(self) -> None:
        for field in (
            "source_training_input_path",
            "published_training_bundle_directory",
            "formal_run_ledger_directory",
            "output_directory",
        ):
            path = getattr(self, field)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{field} must be an absolute private path")
        if type(self.run_id) is not str or not self.run_id.strip():
            raise ValueError("frozen evaluation run_id must be non-empty")
        if not isinstance(self.formal_artifact_resolver, FormalArtifactResolver):
            raise TypeError("frozen evaluation requires a formal artifact resolver")
        if not isinstance(self.formal_evaluation_launch, FormalEvaluationLaunch):
            raise TypeError("frozen evaluation requires a formal evaluation launch")


@dataclass(frozen=True, slots=True)
class InitialBaselineEvaluationDeployment:
    """Private files needed to evaluate the exact pre-training baseline.

    The baseline still requires its source arm's terminal success.  This
    prevents a hand-created manifest slot from being evaluated as if it had
    participated in the preregistered formal group.
    """

    source_training_input_path: Path
    published_training_bundle_directory: Path
    formal_run_ledger_directory: Path
    output_directory: Path
    run_id: str
    formal_artifact_resolver: FormalArtifactResolver
    formal_evaluation_launch: FormalEvaluationLaunch

    def __post_init__(self) -> None:
        for field in (
            "source_training_input_path",
            "published_training_bundle_directory",
            "formal_run_ledger_directory",
            "output_directory",
        ):
            path = getattr(self, field)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{field} must be an absolute private path")
        if type(self.run_id) is not str or not self.run_id.strip():
            raise ValueError("initial baseline evaluation run_id must be non-empty")
        if not isinstance(self.formal_artifact_resolver, FormalArtifactResolver):
            raise TypeError("initial baseline evaluation requires a formal artifact resolver")
        if not isinstance(self.formal_evaluation_launch, FormalEvaluationLaunch):
            raise TypeError("initial baseline evaluation requires a formal evaluation launch")


@dataclass(frozen=True, slots=True)
class PhaseAnchorEvaluationDeployment:
    """Private files needed to evaluate one exact post-cycle progress anchor."""

    source_training_input_path: Path
    published_training_bundle_directory: Path
    formal_run_ledger_directory: Path
    phase_checkpoint_directory: Path
    output_directory: Path
    run_id: str
    formal_artifact_resolver: FormalArtifactResolver
    formal_evaluation_launch: FormalEvaluationLaunch

    def __post_init__(self) -> None:
        for field in (
            "source_training_input_path",
            "published_training_bundle_directory",
            "formal_run_ledger_directory",
            "phase_checkpoint_directory",
            "output_directory",
        ):
            path = getattr(self, field)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{field} must be an absolute private path")
        if type(self.run_id) is not str or not self.run_id.strip():
            raise ValueError("phase anchor evaluation run_id must be non-empty")
        if not isinstance(self.formal_artifact_resolver, FormalArtifactResolver):
            raise TypeError("phase anchor evaluation requires a formal artifact resolver")
        if not isinstance(self.formal_evaluation_launch, FormalEvaluationLaunch):
            raise TypeError("phase anchor evaluation requires a formal evaluation launch")


@dataclass(frozen=True, slots=True)
class _DeploymentPhaseCheckpointResolver(PhaseCheckpointArtifactResolver):
    directory: Path

    def resolve_phase_checkpoint_artifact(self, **_: object) -> Path:
        return self.directory


@dataclass(frozen=True, slots=True)
class _SourceTokenizerResolver(TokenizerArtifactResolver):
    """Resolve the one tokenizer pinned by the verified formal input."""

    backbone: QwenDeploymentConfig

    def resolve(self, identity: PublicTokenizerIdentity) -> ResolvedAuditTokenizer:
        from skillev.policy import QwenTokenizerAdapter

        tokenizer = QwenTokenizerAdapter.from_config(self.backbone)
        if tokenizer.public_identity != identity:
            raise ValueError("formal source tokenizer differs from published identity")
        return tokenizer


def _admit_evaluation(
    *,
    launch: FormalEvaluationLaunch,
    source: PrivateBenchmarkAttemptInput,
    state: FrozenInferenceState | InitialBaselineInferenceState | PhaseAnchorInferenceState,
    task_sequence: object,
    expected_kind: FormalEvaluationKind,
    anchor_ordinal: int,
) -> _AdmittedEvaluation:
    from skillev_private.benchmarks.curriculum import PrivateFrozenTaskSequence

    if not isinstance(task_sequence, PrivateFrozenTaskSequence):
        raise TypeError("formal evaluation requires a private frozen task sequence")
    ledger = FormalEvaluationLedger.open(launch.ledger_directory)
    slot = ledger.consume(launch)
    source_identity = source.public_identity(exact_input_sha256=state.source_exact_input_sha256)
    source_attempt_id = (
        state.source_attempt_id
        if isinstance(state, InitialBaselineInferenceState)
        else state.source_training_attempt_id
    )
    source_identity_hash = (
        state.source_identity_hash
        if isinstance(state, InitialBaselineInferenceState)
        else state.source_training_identity_hash
    )
    if (
        slot.source_formal_run_group_id != state.source_formal_run_group_id
        or slot.source_builder_kind is not source_identity.builder_kind
        or slot.source_training_attempt_id != source_attempt_id
        or slot.source_training_identity_hash != source_identity_hash
        or source_identity_hash != source_identity.content_hash
        or slot.kind is not expected_kind
        or slot.anchor_ordinal != anchor_ordinal
        or slot.frozen_state_hash != state.content_hash
        or slot.policy_snapshot_id != state.policy_snapshot_id
        or slot.library_version != state.library.current_version
        or slot.task_sequence_identity != task_sequence.identity
        or slot.sampling_schedule_hash != source_identity.sampling_schedule_hash
        or slot.implementation_build_hash != source.implementation_build.content_hash
    ):
        ledger.record_failure(slot=slot)
        raise ValueError("formal evaluation launch differs from the admitted state and schedule")
    try:
        measured_build = source.runtime.implementation_build_deployment.require_exact(
            source.implementation_build
        )
        measured_hardware = measure_formal_execution_hardware()
    except BaseException:
        ledger.record_failure(slot=slot)
        raise
    return _AdmittedEvaluation(ledger, slot, measured_build, measured_hardware)


def _terminalize_evaluation(
    admitted: _AdmittedEvaluation,
    episodes: tuple[PrivateEvaluationEpisodeResult, ...],
) -> FormalEvaluationExecutionResult:
    terminal = admitted.ledger.record_success(
        slot=admitted.slot,
        outcome_hash=stable_hash([item.to_value() for item in episodes]),
        implementation_build=admitted.implementation_build,
        execution_hardware=admitted.execution_hardware,
    )
    return FormalEvaluationExecutionResult(episodes=episodes, terminal=terminal)


async def execute_frozen_evaluation(
    request: FrozenEvaluationInput,
    *,
    deployment: FrozenEvaluationDeployment,
) -> FormalEvaluationExecutionResult:
    """Run the exact Qwen/retriever/official evaluation graph once.

    The worker resolves the training input by its published SHA-256, rebuilds
    the one pinned private catalog, and never accepts an arbitrary episode
    callable.  No optimizer, calibration update, or evolution graph exists
    on this path.
    """

    if not isinstance(request, FrozenEvaluationInput):
        raise TypeError("frozen evaluation worker requires FrozenEvaluationInput")
    if not isinstance(deployment, FrozenEvaluationDeployment):
        raise TypeError("frozen evaluation worker requires FrozenEvaluationDeployment")
    source = PrivateBenchmarkAttemptInput.read_verified(
        deployment.source_training_input_path,
        expected_sha256=request.state.source_exact_input_sha256,
    )
    identity = source.public_identity(
        exact_input_sha256=request.state.source_exact_input_sha256,
    )
    bundle = PublishedSuccessfulAttemptBundle.open_exact(
        deployment.published_training_bundle_directory
    )
    ledger = FormalRunLedger.open(directory=deployment.formal_run_ledger_directory)
    audit_result = audit_published_attempt(
        bundle,
        resources=AuditResources(
            _SourceTokenizerResolver(source.backbone),
            formal_artifact_resolver=deployment.formal_artifact_resolver,
        ),
        formal_run_ledger=ledger,
    )
    request.state.require_audited_published_training(bundle, audit_result, ledger)
    if request.task_sequence.identity not in source.protocol.schedule_identities:
        raise ValueError("frozen evaluation sequence is absent from the formal protocol")
    expected_kind = (
        FormalEvaluationKind.FINAL_IID
        if request.task_sequence.identity.purpose.value == "iid-evaluation"
        else FormalEvaluationKind.FINAL_OOD
        if request.task_sequence.identity.purpose.value == "ood-evaluation"
        else None
    )
    if expected_kind is None:
        raise ValueError("final frozen evaluation requires an IID or OOD evaluation schedule")
    admitted = _admit_evaluation(
        launch=deployment.formal_evaluation_launch,
        source=source,
        state=request.state,
        task_sequence=request.task_sequence,
        expected_kind=expected_kind,
        anchor_ordinal=0,
    )
    loaded_catalog = None
    try:
        execution = FrozenEvaluationExecutionConfig(
            backbone=source.backbone,
            trainer=source.application.trainer,
            maximum_h0_tokens=source.application.maximum_h0_tokens,
            initial_checkpoint=source.initial_checkpoint,
            source_identity=identity,
            output_directory=deployment.output_directory,
            run_id=deployment.run_id,
        )
        loaded_catalog = source.runtime.load_complete_catalog(source.catalog_bundle)
        runner = FrozenEvaluationRunner(
            state=request.state,
            catalog=loaded_catalog.catalog,
            execution=execution,
        )
        episodes = await runner.run_sequence(request.task_sequence)
    except BaseException:
        admitted.ledger.record_failure(slot=admitted.slot)
        raise
    finally:
        if loaded_catalog is not None:
            loaded_catalog.close()
    return _terminalize_evaluation(admitted, episodes)


async def execute_initial_baseline_evaluation(
    request: InitialBaselineEvaluationInput,
    *,
    deployment: InitialBaselineEvaluationDeployment,
) -> FormalEvaluationExecutionResult:
    """Evaluate exactly the shared pre-training state, with no training artifact."""

    if not isinstance(request, InitialBaselineEvaluationInput):
        raise TypeError("initial baseline worker requires InitialBaselineEvaluationInput")
    if not isinstance(deployment, InitialBaselineEvaluationDeployment):
        raise TypeError("initial baseline worker requires InitialBaselineEvaluationDeployment")
    source = PrivateBenchmarkAttemptInput.read_verified(
        deployment.source_training_input_path,
        expected_sha256=request.state.source_exact_input_sha256,
    )
    formal_execution = source.protocol.formal_execution
    require_exact_formal_artifacts(
        deployment.formal_artifact_resolver,
        base_model=formal_execution.base_model_artifact,
        tokenizer=formal_execution.tokenizer_artifact,
        implementation_build=formal_execution.implementation_build,
    )
    identity = source.public_identity(
        exact_input_sha256=request.state.source_exact_input_sha256,
    )
    bundle = PublishedSuccessfulAttemptBundle.open_exact(
        deployment.published_training_bundle_directory
    )
    ledger = FormalRunLedger.open(directory=deployment.formal_run_ledger_directory)
    request.state.require_formal_initial(
        initial_checkpoint=source.initial_checkpoint,
        seed_documents=source.seed_documents,
        source_identity=identity,
        source_bundle=bundle,
        formal_run_ledger=ledger,
    )
    if request.task_sequence.identity not in source.protocol.schedule_identities:
        raise ValueError("initial baseline sequence is absent from the formal protocol")
    expected_kind = (
        FormalEvaluationKind.INITIAL_IID
        if request.task_sequence.identity.purpose.value == "iid-evaluation"
        else FormalEvaluationKind.INITIAL_OOD
        if request.task_sequence.identity.purpose.value == "ood-evaluation"
        else None
    )
    if expected_kind is None:
        raise ValueError("initial baseline requires an IID or OOD evaluation schedule")
    admitted = _admit_evaluation(
        launch=deployment.formal_evaluation_launch,
        source=source,
        state=request.state,
        task_sequence=request.task_sequence,
        expected_kind=expected_kind,
        anchor_ordinal=0,
    )
    loaded_catalog = None
    try:
        execution = FrozenEvaluationExecutionConfig(
            backbone=source.backbone,
            trainer=source.application.trainer,
            maximum_h0_tokens=source.application.maximum_h0_tokens,
            initial_checkpoint=source.initial_checkpoint,
            source_identity=identity,
            output_directory=deployment.output_directory,
            run_id=deployment.run_id,
        )
        loaded_catalog = source.runtime.load_complete_catalog(source.catalog_bundle)
        runner = FrozenEvaluationRunner(
            state=request.state,
            catalog=loaded_catalog.catalog,
            execution=execution,
        )
        episodes = await runner.run_sequence(request.task_sequence)
    except BaseException:
        admitted.ledger.record_failure(slot=admitted.slot)
        raise
    finally:
        if loaded_catalog is not None:
            loaded_catalog.close()
    return _terminalize_evaluation(admitted, episodes)


async def execute_phase_anchor_evaluation(
    request: PhaseAnchorEvaluationInput,
    *,
    deployment: PhaseAnchorEvaluationDeployment,
) -> FormalEvaluationExecutionResult:
    """Run a predeclared no-update IID progress sequence at one phase checkpoint."""

    if not isinstance(request, PhaseAnchorEvaluationInput):
        raise TypeError("phase anchor worker requires PhaseAnchorEvaluationInput")
    if not isinstance(deployment, PhaseAnchorEvaluationDeployment):
        raise TypeError("phase anchor worker requires PhaseAnchorEvaluationDeployment")
    source = PrivateBenchmarkAttemptInput.read_verified(
        deployment.source_training_input_path,
        expected_sha256=request.state.source_exact_input_sha256,
    )
    identity = source.public_identity(
        exact_input_sha256=request.state.source_exact_input_sha256,
    )
    bundle = PublishedSuccessfulAttemptBundle.open_exact(
        deployment.published_training_bundle_directory
    )
    ledger = FormalRunLedger.open(directory=deployment.formal_run_ledger_directory)
    audit_result = audit_published_attempt(
        bundle,
        resources=AuditResources(
            _SourceTokenizerResolver(source.backbone),
            formal_artifact_resolver=deployment.formal_artifact_resolver,
        ),
        formal_run_ledger=ledger,
    )
    if audit_result.formal_artifacts_verified is not True:
        raise ValueError("phase anchor audit did not measure formal artifacts")
    request.state.require_published_phase(
        bundle,
        artifact_resolver=_DeploymentPhaseCheckpointResolver(deployment.phase_checkpoint_directory),
        formal_run_ledger=ledger,
    )
    if request.task_sequence.identity not in source.protocol.schedule_identities:
        raise ValueError("phase anchor sequence is absent from the formal protocol")
    admitted = _admit_evaluation(
        launch=deployment.formal_evaluation_launch,
        source=source,
        state=request.state,
        task_sequence=request.task_sequence,
        expected_kind=FormalEvaluationKind.PROGRESS_IID,
        anchor_ordinal=request.state.evolution_counts.cycle_count,
    )
    loaded_catalog = None
    try:
        execution = FrozenEvaluationExecutionConfig(
            backbone=source.backbone,
            trainer=source.application.trainer,
            maximum_h0_tokens=source.application.maximum_h0_tokens,
            initial_checkpoint=source.initial_checkpoint,
            source_identity=identity,
            output_directory=deployment.output_directory,
            run_id=deployment.run_id,
        )
        loaded_catalog = source.runtime.load_complete_catalog(source.catalog_bundle)
        runner = FrozenEvaluationRunner(
            state=request.state,
            catalog=loaded_catalog.catalog,
            execution=execution,
        )
        episodes = await runner.run_sequence(request.task_sequence)
    except BaseException:
        admitted.ledger.record_failure(slot=admitted.slot)
        raise
    finally:
        if loaded_catalog is not None:
            loaded_catalog.close()
    return _terminalize_evaluation(admitted, episodes)


__all__ = [
    "FormalEvaluationExecutionResult",
    "FrozenEvaluationDeployment",
    "InitialBaselineEvaluationDeployment",
    "PhaseAnchorEvaluationDeployment",
    "execute_frozen_evaluation",
    "execute_initial_baseline_evaluation",
    "execute_phase_anchor_evaluation",
]
