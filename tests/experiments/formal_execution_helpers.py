"""Small result-blind identities shared by formal-execution tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillev.application import ApplicationConfig
from skillev.audit import FormalArtifactResolver
from skillev.calibration import CalibrationConfig
from skillev.contracts import canonical_json, stable_hash
from skillev.diagnostics import DiagnosticsConfig
from skillev.evolution import (
    AuthoringSamplingConfig,
    EvolutionConfig,
    SkillAuthoringAuthority,
)
from skillev.experiments import PublishedAttemptIdentity
from skillev.experiments.b1_run_admission import B1RunAdmission, B1TerminalRequirements
from skillev.experiments.execution_hardware import (
    ExecutionHardwareIdentity,
    FormalExecutionHardwareAttestation,
)
from skillev.experiments.formal_execution import (
    FormalExecutionFreeze,
    FormalImplementationBuildAttestation,
    ImplementationBuildIdentity,
)
from skillev.experiments.formal_run_manifest import (
    FormalRunLedger,
    FormalRunManifest,
    FormalRunSlot,
)
from skillev.experiments.protocol import FIXED_SEED, FrozenTaskSequenceIdentity, ProtocolFreeze
from skillev.policy import (
    ArtifactFileIdentity,
    BaseModelArtifactIdentity,
    PublicTokenizerIdentity,
    PublicTokenizerKind,
    TokenizerArtifactIdentity,
    TrainableStateIdentity,
)
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptPublisher,
    AttemptRequest,
    AttemptSucceeded,
    BudgetVector,
    EventEnvelope,
    EventType,
    ExactAttemptRunPlan,
    FinalTrainingArtifact,
    FlowOnlyAttemptSummary,
    FullAttemptSummary,
    PublishedSuccessfulAttemptBundle,
    UnpublishedAttemptBundle,
)
from skillev.training import (
    CheckpointConfig,
    OptimizerConfig,
    PolicyRolloutConfig,
    TrainerConfig,
    TrainingExecutionConfig,
    TTBMethodConfig,
    conservative_rollout_maximum,
)


@dataclass(frozen=True, slots=True)
class FormalArtifactFixture:
    """Synthetic byte identities; no model files or benchmark contents."""

    base_model: BaseModelArtifactIdentity
    tokenizer: TokenizerArtifactIdentity
    implementation: ImplementationBuildIdentity

    @property
    def public_tokenizer(self) -> PublicTokenizerIdentity:
        return PublicTokenizerIdentity(
            kind=self.tokenizer.kind,
            tokenizer_id=self.tokenizer.tokenizer_id,
            revision=self.tokenizer.revision,
            content_hash=self.tokenizer.content_hash,
        )


@dataclass(frozen=True, slots=True)
class FormalArtifactResolverFixture(FormalArtifactResolver):
    """Exact measured identities for a formal-worker contract fixture."""

    artifacts: FormalArtifactFixture

    def resolve_base_model(
        self,
        expected: BaseModelArtifactIdentity,
    ) -> BaseModelArtifactIdentity:
        del expected
        return self.artifacts.base_model

    def resolve_tokenizer(
        self,
        expected: TokenizerArtifactIdentity,
    ) -> TokenizerArtifactIdentity:
        del expected
        return self.artifacts.tokenizer

    def resolve_implementation_build(
        self,
        expected: ImplementationBuildIdentity,
    ) -> ImplementationBuildIdentity:
        del expected
        return self.artifacts.implementation


def _artifact_file(relative_path: str, label: str) -> ArtifactFileIdentity:
    return ArtifactFileIdentity(
        relative_path=relative_path,
        size_bytes=1,
        sha256=stable_hash({"formal-test-file": label}),
    )


def make_formal_artifacts() -> FormalArtifactFixture:
    """Return one stable, fully populated formal execution artifact set."""

    base_model = BaseModelArtifactIdentity.create(
        backend_class="transformers.AutoModelForCausalLM",
        upstream_revision="test-upstream-revision",
        dtype_conversion_policy="from_pretrained:bfloat16;module_to:bfloat16",
        model_config=_artifact_file("config.json", "model-config"),
        generation_config=_artifact_file("generation_config.json", "generation-config"),
        weight_index=_artifact_file("model.safetensors.index.json", "weight-index"),
        weight_shards=(_artifact_file("model-00001-of-00001.safetensors", "weight-shard"),),
    )
    tokenizer = TokenizerArtifactIdentity.create(
        kind=PublicTokenizerKind.QWEN,
        tokenizer_id="formal-test-tokenizer",
        revision="test-tokenizer-revision",
        backend_serialization_hash=stable_hash("formal-tokenizer-backend"),
        tokenizer_config_hash=stable_hash("formal-tokenizer-config"),
        chat_template_hash=stable_hash("formal-tokenizer-template"),
        special_tokens_hash=stable_hash("formal-tokenizer-special-tokens"),
        added_tokens_hash=stable_hash("formal-tokenizer-added-tokens"),
        transformers_version="5.14.1",
        tokenizers_version="0.22.2",
    )
    implementation = ImplementationBuildIdentity(
        source_commit="1" * 40,
        source_tree_hash=stable_hash("formal-source-tree"),
        public_wheel_hash=stable_hash("formal-public-wheel"),
        private_evaluation_wheel_hash=stable_hash("formal-private-wheel"),
        lockfile_hash=stable_hash("formal-lockfile"),
        python_version="3.13.0",
        torch_version="2.13.0",
        transformers_version="5.14.1",
        peft_version="0.19.1",
        tokenizers_version="0.22.2",
        deterministic_algorithms=True,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        matmul_allow_tf32=False,
    )
    return FormalArtifactFixture(
        base_model=base_model,
        tokenizer=tokenizer,
        implementation=implementation,
    )


def make_formal_artifact_resolver(
    artifacts: FormalArtifactFixture | None = None,
) -> FormalArtifactResolverFixture:
    """Return the explicit measured-artifact boundary for worker tests."""

    return FormalArtifactResolverFixture(artifacts=artifacts or make_formal_artifacts())


def make_formal_application(*, batch_size: int = 1) -> ApplicationConfig:
    """Build a small application whose seed is mechanically protocol-fixed."""

    rollout_maximum = conservative_rollout_maximum(
        max_turns=1,
        max_reasoning_tokens=2,
        max_action_tokens=2,
        max_model_input_tokens=64,
        max_tool_wall_time_milliseconds=10,
    )
    return ApplicationConfig(
        trainer=TrainerConfig(
            method=TTBMethodConfig(epsilon_min=0.01, temperature_beta=1.0),
            rollout=PolicyRolloutConfig(
                base_seed=FIXED_SEED,
                max_turns=1,
                max_reasoning_tokens=2,
                max_action_tokens=2,
                per_rollout_maximum=rollout_maximum,
            ),
            optimizer=OptimizerConfig(
                adapter_learning_rate=1e-3,
                z_learning_rate=2e-3,
                weight_decay=0.0,
            ),
            execution=TrainingExecutionConfig(
                experiment_id="formal-execution-test-namespace",
                batch_size=batch_size,
            ),
            checkpoint=CheckpointConfig(every_n_steps=1),
        ),
        diagnostics=DiagnosticsConfig(window_size=2),
        calibration=CalibrationConfig(),
        evolution=EvolutionConfig(generate_min_absolute_log_importance=0.1),
        authoring_sampling=AuthoringSamplingConfig(temperature=1.0, top_p=1.0),
        maximum_h0_tokens=128,
    )


def make_formal_execution(
    *,
    application: ApplicationConfig,
    run_plan: ExactAttemptRunPlan,
    training_sequence: FrozenTaskSequenceIdentity,
    backbone_deployment_hash: str,
    initial_trainable_state: TrainableStateIdentity,
    initial_library_version: str,
    initial_skill_library_state_hash: str,
    authoring_authority: SkillAuthoringAuthority,
    attempt_budget: BudgetVector,
    phi_per_cycle_maximum: BudgetVector,
    catalog_freeze_hash: str,
    artifacts: FormalArtifactFixture | None = None,
    b1_run_admission: B1RunAdmission | None = None,
) -> FormalExecutionFreeze:
    """Construct the one typed freeze required by formal-identity tests."""

    selected = artifacts or make_formal_artifacts()
    selected_b1 = b1_run_admission or B1RunAdmission(
        f2_f3_proof_hash=stable_hash("formal-test-f2-f3-proof"),
        exact_admission_report_hash=stable_hash("formal-test-exact-admission"),
        run_plan=run_plan,
        phi_per_cycle_maximum=phi_per_cycle_maximum,
        total_phi_maximum=phi_per_cycle_maximum.scale(run_plan.maximum_cycles),
        attempt_budget=attempt_budget,
        closure_tail_budget=BudgetVector(),
        maximum_library_size=1,
        maximum_h0_tokens=application.maximum_h0_tokens,
        maximum_applicable_skills_per_task=1,
        maximum_complete_rendered_skill_block_tokens=1,
        maximum_complete_rendered_skill_block_tokens_in_h0=1,
        available_h0_task_and_wrapper_tokens=application.maximum_h0_tokens - 1,
        measured_task_and_wrapper_maximum_tokens=1,
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
        backbone_deployment_hash=backbone_deployment_hash,
        base_model_artifact=selected.base_model,
        tokenizer_artifact=selected.tokenizer,
        implementation_build=selected.implementation,
        application=application,
        run_plan=run_plan,
        initial_trainable_state=initial_trainable_state,
        initial_library_version=initial_library_version,
        initial_skill_library_state_hash=initial_skill_library_state_hash,
        authoring_authority=authoring_authority,
        attempt_budget=attempt_budget,
        phi_per_cycle_maximum=phi_per_cycle_maximum,
        b1_run_admission=selected_b1,
        training_sequence=training_sequence,
        catalog_freeze_hash=catalog_freeze_hash,
    )


def make_formal_run_ledger_for_identity(
    tmp_path: Path,
    identity: PublishedAttemptIdentity,
) -> FormalRunLedger:
    """Create a seven-slot ledger for one formal test identity."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    if identity.formal_execution is None:
        raise ValueError("formal run fixture requires a formal identity")
    freeze = ProtocolFreeze(
        protocol_hash=identity.protocol_hash,
        freeze_id=identity.protocol_freeze_id,
    )
    slots = tuple(
        FormalRunSlot(
            builder_kind=kind,
            attempt_id=f"formal-{kind.value}",
            exact_input_sha256=(
                identity.exact_input_sha256
                if kind is identity.builder_kind
                else stable_hash({"formal-helper-input": kind.value})
            ),
            public_identity_content_hash=(
                identity.content_hash
                if kind is identity.builder_kind
                else stable_hash({"formal-helper-identity": kind.value})
            ),
        )
        for kind in AttemptBuilderKind
    )
    manifest = FormalRunManifest.create(protocol_freeze=freeze, slots=slots)
    return FormalRunLedger.create(
        directory=tmp_path / "formal-run-ledger",
        manifest=manifest,
    )


def _formal_hardware_identity() -> ExecutionHardwareIdentity:
    """Stable CUDA-only identity used by result-blind formal bundle fixtures."""

    return ExecutionHardwareIdentity(
        accelerator_name="NVIDIA H200",
        compute_capability="9.0",
        visible_device_count=1,
        nvidia_driver_version="570.12",
        cuda_runtime_version="12.8",
        cudnn_version="91000",
        nccl_version="2.27.5",
        kernel_release="6.12.0-test",
        safetensors_version="0.5.3",
    )


def _formal_source_event_lines(
    *,
    identity: PublishedAttemptIdentity,
    request: AttemptRequest,
    hardware_identity: ExecutionHardwareIdentity | None = None,
) -> str:
    """Write the two mandatory pre-model formal provenance events."""

    if identity.formal_execution is None:
        raise ValueError("formal helper source requires a formal execution freeze")
    build = FormalImplementationBuildAttestation(
        attempt_id=request.attempt_id,
        builder_kind=request.builder_kind,
        exact_input_sha256=request.exact_input_sha256,
        public_identity_content_hash=identity.content_hash,
        formal_execution_content_hash=identity.formal_execution.content_hash,
        expected_implementation_build=identity.formal_execution.implementation_build,
        measured_implementation_build=identity.formal_execution.implementation_build,
    )
    hardware = FormalExecutionHardwareAttestation(
        attempt_id=request.attempt_id,
        builder_kind=request.builder_kind,
        exact_input_sha256=request.exact_input_sha256,
        public_identity_content_hash=identity.content_hash,
        formal_execution_content_hash=identity.formal_execution.content_hash,
        measured_hardware=hardware_identity or _formal_hardware_identity(),
    )
    events = (
        EventEnvelope.create(
            event_type=EventType.FORMAL_IMPLEMENTATION_BUILD_ATTESTED,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            producer_id="skillev-formal-build",
            producer_seq=1,
            occurred_at="1970-01-01T00:00:00Z",
            payload=build.to_value(),
        ),
        EventEnvelope.create(
            event_type=EventType.FORMAL_EXECUTION_HARDWARE_ATTESTED,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            producer_id="skillev-formal-build",
            producer_seq=2,
            occurred_at="1970-01-01T00:00:00Z",
            payload=hardware.to_value(),
        ),
    )
    return "".join(canonical_json(event.to_value()) + "\n" for event in events)


def publish_formal_success_for_identity(
    tmp_path: Path,
    identity: PublishedAttemptIdentity,
    ledger: FormalRunLedger,
    *,
    hardware_identity: ExecutionHardwareIdentity | None = None,
) -> PublishedSuccessfulAttemptBundle:
    """Create one real formal published bundle for source-admission tests.

    The helper creates only answer-free synthetic source logs.  It exercises
    the production order claim → private launch → terminal → publication.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)

    slot = next(
        slot for slot in ledger.manifest.slots if slot.builder_kind is identity.builder_kind
    )
    if (
        slot.exact_input_sha256 != identity.exact_input_sha256
        or slot.public_identity_content_hash != identity.content_hash
    ):
        raise ValueError("formal helper ledger differs from the source identity")
    exact_input = tmp_path / "formal-input.json"
    exact_input.write_text("{}\n", encoding="utf-8")
    private_root = tmp_path / "formal-private"
    private_root.mkdir(exist_ok=True)
    private = UnpublishedAttemptBundle.create(private_root / slot.attempt_id)
    request = AttemptRequest(
        run_id="formal-helper-run",
        attempt_id=slot.attempt_id,
        builder_kind=slot.builder_kind,
        exact_input_path=exact_input,
        exact_input_sha256=slot.exact_input_sha256,
        private_bundle_directory=private.directory,
    )
    private.event_log_path.write_text(
        _formal_source_event_lines(
            identity=identity,
            request=request,
            hardware_identity=hardware_identity,
        ),
        encoding="utf-8",
    )
    if request.builder_kind is AttemptBuilderKind.NO_BAYESIAN:
        private.arm_event_log_path.write_text(
            '{"source":"formal-helper-flow-only"}\n',
            encoding="utf-8",
        )
    identity_sha256, identity_content_hash = private.write_public_identity_once(identity.to_value())
    source_logs = private.source_log_digests(request.builder_kind)
    summary = (
        FlowOnlyAttemptSummary(
            reports=(),
            planned_training_steps_this_attempt=0,
            completed_training_steps_this_attempt=0,
            actions_committed_this_attempt=0,
            cycles_committed_this_attempt=0,
            cycles_committed_in_run=0,
            initial_optimizer_step=0,
            final_optimizer_step=0,
            final_library_version="formal-helper-library",
            final_policy_snapshot_id="formal-helper-snapshot",
        )
        if request.builder_kind is AttemptBuilderKind.NO_BAYESIAN
        else FullAttemptSummary(
            reports=(),
            planned_training_steps_this_attempt=0,
            completed_training_steps_this_attempt=0,
            actions_committed_this_attempt=0,
            cycles_committed_this_attempt=0,
            cycles_committed_in_run=0,
            initial_optimizer_step=0,
            final_optimizer_step=0,
            final_library_version="formal-helper-library",
            final_policy_snapshot_id="formal-helper-snapshot",
        )
    )
    artifact = FinalTrainingArtifact(
        artifact_sha256=stable_hash({"formal-helper": "artifact", "arm": slot.builder_kind.value}),
        runtime_state_sha256=stable_hash(
            {"formal-helper": "runtime", "arm": slot.builder_kind.value}
        ),
        policy_snapshot_id=summary.final_policy_snapshot_id,
        library_version=summary.final_library_version,
        optimizer_step=summary.final_optimizer_step,
    )
    claim = ledger.claim(request)
    ledger.consume_private_launch(ledger.launch_for(request=request, claim_secret=claim))
    private.write_outcome_once(
        AttemptSucceeded(
            attempt_id=request.attempt_id,
            builder_kind=request.builder_kind,
            exact_input_sha256=request.exact_input_sha256,
            public_identity_sha256=identity_sha256,
            public_identity_content_hash=identity_content_hash,
            source_logs=source_logs,
            summary=summary,
            final_training_artifact=artifact,
            formal_run_group_id=ledger.manifest.run_group_id,
        ).to_value()
    )
    publisher = AttemptPublisher(
        published_root=tmp_path / "formal-published",
        quarantine_root=tmp_path / "formal-quarantine",
    )
    prepared = publisher.prepare_success(private, request=request)
    admission = ledger.record_success(request=request, prepared=prepared)
    return publisher.publish_prepared(prepared, formal_publication=admission)


__all__ = [
    "FormalArtifactFixture",
    "FormalArtifactResolverFixture",
    "make_formal_application",
    "make_formal_artifact_resolver",
    "make_formal_artifacts",
    "make_formal_execution",
    "make_formal_run_ledger_for_identity",
    "publish_formal_success_for_identity",
]
