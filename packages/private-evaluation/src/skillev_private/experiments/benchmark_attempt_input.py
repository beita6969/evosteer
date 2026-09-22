"""Private formal-attempt input with answer-free public identity projection.

This file may contain private catalog paths and frozen task IDs.  Its public
projection deliberately excludes those values and only exposes their already
frozen hashes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillev.application_config import ApplicationConfig
from skillev.contracts import JsonValue, normalize_json
from skillev.evolution import SkillAuthoringAuthority
from skillev.experiments import (
    AttemptPurpose,
    CrossArmExecutionIdentity,
    ExperimentProtocol,
    FormalExecutionFreeze,
    FormalTrainingBinding,
    ImplementationBuildIdentity,
    ProtocolFreeze,
    PublishedAttemptIdentity,
    SchedulePurpose,
    arm_protocol_for_builder_kind,
)
from skillev.experiments.b1_run_admission import B1RunAdmission
from skillev.policy import (
    PrivateInitialCheckpointBinding,
    PublicTokenizerIdentity,
    PublicTokenizerKind,
    QwenBackboneConfig,
    QwenDeploymentConfig,
    QwenMultimodalBackboneConfig,
    TokenizerArtifactIdentity,
    public_qwen_deployment_hash,
)
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptRunCursorState,
    BudgetVector,
    ExactAttemptRunPlan,
    SkillDocument,
    SkillLibraryState,
)
from skillev.runtime.attempt_publication import sha256_bytes
from skillev.training import (
    FixedAttemptBudgetPlan,
    PrivateCheckpointStorageBinding,
    RolloutWorkflowBinding,
)
from skillev_private.benchmarks.catalog_freeze import FrozenProductionCatalogBundle
from skillev_private.benchmarks.curriculum import PrivateFrozenTaskSequence

from .benchmark_runtime import PrivateBenchmarkRuntime

PRIVATE_BENCHMARK_ATTEMPT_INPUT_FORMAT = "skillev-private-benchmark-attempt-input@5"


def _object(value: object, *, fields: frozenset[str], label: str) -> dict[str, Any]:
    normalized = normalize_json(value)
    if not isinstance(normalized, dict) or set(normalized) != fields:
        raise ValueError(f"{label} has incompatible fields")
    return normalized


def _sha256(value: object, *, field: str) -> str:
    if type(value) is not str or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{field} must be a SHA-256 identity")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 identity") from error
    return value


@dataclass(frozen=True, slots=True)
class PrivateBenchmarkAttemptInput:
    """One formal private input for a fixed preregistered arm and schedule."""

    builder_kind: AttemptBuilderKind
    backbone: QwenDeploymentConfig
    backbone_kind: str
    tokenizer_artifact: TokenizerArtifactIdentity
    implementation_build: ImplementationBuildIdentity
    application: ApplicationConfig
    checkpoint_storage: PrivateCheckpointStorageBinding
    initial_checkpoint: PrivateInitialCheckpointBinding
    run_plan: ExactAttemptRunPlan
    seed_documents: tuple[SkillDocument, ...]
    attempt_budget: BudgetVector
    phi_per_cycle_maximum: BudgetVector
    b1_run_admission: B1RunAdmission
    authoring_authority: SkillAuthoringAuthority
    protocol: ExperimentProtocol
    protocol_freeze: ProtocolFreeze
    catalog_bundle: FrozenProductionCatalogBundle
    runtime: PrivateBenchmarkRuntime
    training_sequence: PrivateFrozenTaskSequence
    rollout_workflow: RolloutWorkflowBinding = field(default_factory=RolloutWorkflowBinding)
    format: str = PRIVATE_BENCHMARK_ATTEMPT_INPUT_FORMAT

    def __post_init__(self) -> None:
        if self.format != PRIVATE_BENCHMARK_ATTEMPT_INPUT_FORMAT:
            raise ValueError("unsupported private benchmark attempt input format")
        if not isinstance(self.builder_kind, AttemptBuilderKind):
            raise TypeError("private input requires a closed builder kind")
        if self.backbone_kind == "qwen-causal":
            if type(self.backbone) is not QwenBackboneConfig:
                raise TypeError("causal private input requires QwenBackboneConfig")
        elif self.backbone_kind == "qwen-multimodal":
            if type(self.backbone) is not QwenMultimodalBackboneConfig:
                raise TypeError("multimodal private input requires QwenMultimodalBackboneConfig")
        else:
            raise ValueError("private input backbone_kind is unsupported")
        if not isinstance(self.application, ApplicationConfig):
            raise TypeError("private input application must be ApplicationConfig")
        if not isinstance(self.protocol, ExperimentProtocol):
            raise TypeError("private input protocol must be ExperimentProtocol")
        if self.application.trainer.rollout.base_seed != self.protocol.seed:
            raise ValueError("formal rollout seed must equal the fixed protocol seed")
        if not isinstance(self.tokenizer_artifact, TokenizerArtifactIdentity):
            raise TypeError("private input requires a tokenizer artifact identity")
        if not isinstance(self.implementation_build, ImplementationBuildIdentity):
            raise TypeError("private input requires an implementation build identity")
        if self.backbone.base_model_artifact is None:
            raise ValueError("formal private input requires a base model artifact identity")
        formal_execution = self.protocol.formal_execution
        if not isinstance(formal_execution, FormalExecutionFreeze):  # pragma: no cover
            raise TypeError("protocol formal execution type is invalid")
        if formal_execution.backbone_kind != self.backbone_kind:
            raise ValueError("formal execution freeze has another backbone kind")
        if formal_execution.backbone_deployment_hash != self.backbone_deployment_hash:
            raise ValueError("formal execution freeze has another backbone deployment")
        if formal_execution.base_model_artifact != self.backbone.base_model_artifact:
            raise ValueError("formal execution freeze has another base model artifact")
        if self.tokenizer_artifact.content_hash != self.backbone.tokenizer_content_hash:
            raise ValueError("private tokenizer artifact differs from backbone config")
        if formal_execution.tokenizer_artifact != self.tokenizer_artifact:
            raise ValueError("formal execution freeze has another tokenizer artifact")
        if formal_execution.implementation_build != self.implementation_build:
            raise ValueError("formal execution freeze has another implementation build")
        if formal_execution.seed != self.application.trainer.rollout.base_seed:
            raise ValueError("formal execution seed differs from rollout seed")
        formal_execution.requires_application(self.application)
        if not isinstance(self.checkpoint_storage, PrivateCheckpointStorageBinding):
            raise TypeError("private input requires checkpoint storage")
        if not isinstance(self.initial_checkpoint, PrivateInitialCheckpointBinding):
            raise TypeError("private input requires an initial policy checkpoint")
        if not isinstance(self.run_plan, ExactAttemptRunPlan):
            raise TypeError("private input run_plan must be ExactAttemptRunPlan")
        if not isinstance(self.attempt_budget, BudgetVector):
            raise TypeError("private input attempt_budget must be BudgetVector")
        if not isinstance(self.phi_per_cycle_maximum, BudgetVector):
            raise TypeError("private input phi_per_cycle_maximum must be BudgetVector")
        if not isinstance(self.authoring_authority, SkillAuthoringAuthority):
            raise TypeError("private input authoring_authority is invalid")
        if not isinstance(self.protocol_freeze, ProtocolFreeze):
            raise TypeError("private input protocol_freeze must be ProtocolFreeze")
        if self.protocol_freeze.protocol_hash != self.protocol.content_hash:
            raise ValueError("protocol freeze differs from private attempt protocol")
        if not isinstance(self.catalog_bundle, FrozenProductionCatalogBundle):
            raise TypeError("private input catalog_bundle must be frozen")
        if not isinstance(self.runtime, PrivateBenchmarkRuntime):
            raise TypeError("private input runtime must be PrivateBenchmarkRuntime")
        if not isinstance(self.training_sequence, PrivateFrozenTaskSequence):
            raise TypeError("private input training_sequence must be frozen")
        if not isinstance(self.rollout_workflow, RolloutWorkflowBinding):
            raise TypeError("private input rollout workflow binding is invalid")
        if self.training_sequence.identity.purpose is not SchedulePurpose.IID_TRAINING:
            raise ValueError("training attempt requires an IID training sequence")
        expected_schedule = next(
            item
            for item in self.protocol.schedule_identities
            if item.purpose is SchedulePurpose.IID_TRAINING
        )
        if self.training_sequence.identity != expected_schedule:
            raise ValueError("private training sequence differs from protocol schedule identity")
        expected_tasks = (
            self.application.trainer.execution.batch_size * self.run_plan.total_training_steps
        )
        if self.training_sequence.identity.task_count != expected_tasks:
            raise ValueError("training sequence does not fill the exact run plan")
        if formal_execution.run_plan != self.run_plan:
            raise ValueError("formal execution freeze has another run plan")
        if formal_execution.training_sequence != self.training_sequence.identity:
            raise ValueError("formal execution freeze has another training sequence")
        if formal_execution.catalog_freeze_hash != self.catalog_bundle.public_freeze_hash:
            raise ValueError("formal execution freeze has another catalog freeze")
        if not isinstance(self.seed_documents, tuple) or not self.seed_documents:
            raise ValueError("formal benchmark attempt requires a seed library")
        if any(not isinstance(item, SkillDocument) for item in self.seed_documents):
            raise TypeError("private input seed_documents contain an invalid skill")
        self.required_attempt_budget().validate_against(self.attempt_budget)
        snapshots = {
            source.benchmark: source.snapshot for source in self.catalog_bundle.non_process.sources
        }
        snapshots.update(
            {source.benchmark: source.snapshot for source in self.catalog_bundle.process.sources}
        )
        if len(snapshots) != len(self.protocol.dataset_snapshots):
            raise ValueError("private catalog bundle does not contain the complete benchmark suite")
        by_name = {snapshot.name: snapshot for snapshot in snapshots.values()}
        if tuple(snapshot.name for snapshot in self.protocol.dataset_snapshots) != tuple(
            spec.dataset_snapshot_name for spec in self.protocol.benchmarks
        ):
            raise ValueError("protocol snapshots are not in declared benchmark order")
        if any(
            by_name.get(snapshot.name) != snapshot for snapshot in self.protocol.dataset_snapshots
        ):
            raise ValueError("private catalog snapshots differ from the frozen protocol")
        if formal_execution.initial_trainable_state != self.initial_checkpoint.trainable_state:
            raise ValueError("formal execution freeze has another initial trainable state")
        if formal_execution.initial_library_version != self.initial_library_version:
            raise ValueError("formal execution freeze has another initial library")
        if (
            formal_execution.initial_skill_library_state_hash
            != self.initial_skill_library_state_hash
        ):
            raise ValueError("formal execution freeze has another initial library state")
        if formal_execution.authoring_authority != self.authoring_authority:
            raise ValueError("formal execution freeze has another authoring authority")
        if formal_execution.attempt_budget != self.attempt_budget:
            raise ValueError("formal execution freeze has another attempt budget")
        if formal_execution.phi_per_cycle_maximum != self.phi_per_cycle_maximum:
            raise ValueError("formal execution freeze has another Phi budget")
        if not isinstance(self.b1_run_admission, B1RunAdmission):
            raise TypeError("private input requires the B1 run admission")
        if formal_execution.b1_run_admission != self.b1_run_admission:
            raise ValueError("private input B1 admission differs from the formal freeze")
        self.b1_run_admission.require_exact_input(self)

    def required_attempt_budget(self) -> FixedAttemptBudgetPlan:
        return FixedAttemptBudgetPlan.from_trainer_and_run_plan(
            trainer=self.application.trainer,
            run_plan=self.run_plan,
            phi_per_cycle_maximum=self.phi_per_cycle_maximum,
        )

    @property
    def initial_library_version(self) -> str:
        return SkillLibraryState.from_seed_documents(self.seed_documents).current_version

    @property
    def initial_skill_library_state_hash(self) -> str:
        return SkillLibraryState.from_seed_documents(self.seed_documents).state_hash

    @property
    def backbone_deployment_hash(self) -> str:
        return public_qwen_deployment_hash(
            self.backbone,
            backend_kind=self.backbone_kind,
        )

    def public_identity(self, *, exact_input_sha256: str) -> PublishedAttemptIdentity:
        return PublishedAttemptIdentity(
            builder_kind=self.builder_kind,
            purpose=AttemptPurpose.FORMAL_BENCHMARK_TRAINING,
            arm_protocol=arm_protocol_for_builder_kind(self.builder_kind),
            application_config=self.application,
            run_plan=self.run_plan,
            initial_optimizer_step=0,
            initial_run_cursor=AttemptRunCursorState.fresh(self.run_plan),
            authoring_authority=self.authoring_authority,
            attempt_budget=self.attempt_budget,
            phi_per_cycle_maximum=self.phi_per_cycle_maximum,
            protocol_hash=self.protocol.content_hash,
            protocol_freeze_id=self.protocol_freeze.freeze_id,
            exact_input_sha256=_sha256(exact_input_sha256, field="exact_input_sha256"),
            backbone_deployment_hash=self.backbone_deployment_hash,
            initial_trainable_state=self.initial_checkpoint.trainable_state,
            tokenizer_identity=PublicTokenizerIdentity(
                kind=PublicTokenizerKind.QWEN,
                tokenizer_id=self.backbone.tokenizer_id,
                revision=self.backbone.revision,
                content_hash=self.backbone.tokenizer_content_hash,
            ),
            initial_library_version=self.initial_library_version,
            initial_skill_library_state_hash=self.initial_skill_library_state_hash,
            ordered_task_sequence_hash=self.training_sequence.identity.ordered_task_ids_hash,
            formal_training_binding=FormalTrainingBinding(
                training_sequence=self.training_sequence.identity,
                catalog_freeze_hash=self.catalog_bundle.public_freeze_hash,
            ),
            formal_execution=self.protocol.formal_execution,
            base_model_artifact=self.backbone.base_model_artifact,
            tokenizer_artifact=self.tokenizer_artifact,
            implementation_build=self.implementation_build,
        )

    def cross_arm_execution_identity(
        self,
        *,
        exact_input_sha256: str,
    ) -> CrossArmExecutionIdentity:
        """Return the non-ablation controls used to gate seven-arm aggregation."""

        return CrossArmExecutionIdentity.from_public_identity(
            self.public_identity(exact_input_sha256=exact_input_sha256)
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "application": self.application.to_value(),
            "attempt_budget": self.attempt_budget.to_value(),
            "authoring_authority": self.authoring_authority.to_value(),
            "backbone": self.backbone.to_value(),
            "backbone_kind": self.backbone_kind,
            "b1_run_admission": self.b1_run_admission.to_value(),
            "implementation_build": self.implementation_build.to_value(),
            "builder_kind": self.builder_kind.value,
            "catalog_bundle": self.catalog_bundle.to_value(),
            "checkpoint_storage": self.checkpoint_storage.to_value(),
            "format": self.format,
            "initial_checkpoint": self.initial_checkpoint.to_value(),
            "phi_per_cycle_maximum": self.phi_per_cycle_maximum.to_value(),
            "protocol": self.protocol.to_value(),
            "protocol_freeze": self.protocol_freeze.to_value(),
            "runtime": self.runtime.to_value(),
            "run_plan": self.run_plan.to_value(),
            "rollout_workflow": self.rollout_workflow.to_value(),
            "seed_documents": [item.to_value() for item in self.seed_documents],
            "training_sequence": self.training_sequence.to_value(),
            "tokenizer_artifact": self.tokenizer_artifact.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> PrivateBenchmarkAttemptInput:
        data = _object(
            value,
            fields=frozenset(
                {
                    "application",
                    "attempt_budget",
                    "authoring_authority",
                    "backbone",
                    "backbone_kind",
                    "b1_run_admission",
                    "builder_kind",
                    "catalog_bundle",
                    "checkpoint_storage",
                    "format",
                    "implementation_build",
                    "initial_checkpoint",
                    "phi_per_cycle_maximum",
                    "protocol",
                    "protocol_freeze",
                    "runtime",
                    "run_plan",
                    "rollout_workflow",
                    "seed_documents",
                    "training_sequence",
                    "tokenizer_artifact",
                }
            ),
            label="private benchmark attempt input",
        )
        if type(data["backbone_kind"]) is not str or type(data["format"]) is not str:
            raise ValueError("private benchmark input kind and format must be text")
        raw_documents = data["seed_documents"]
        if not isinstance(raw_documents, list):
            raise ValueError("private benchmark seed_documents must be an array")
        backbone_kind = data["backbone_kind"]
        backbone = (
            QwenBackboneConfig.from_value(data["backbone"])
            if backbone_kind == "qwen-causal"
            else QwenMultimodalBackboneConfig.from_value(data["backbone"])
        )
        return cls(
            builder_kind=AttemptBuilderKind(data["builder_kind"]),
            backbone=backbone,
            backbone_kind=backbone_kind,
            tokenizer_artifact=TokenizerArtifactIdentity.from_value(data["tokenizer_artifact"]),
            implementation_build=ImplementationBuildIdentity.from_value(
                data["implementation_build"]
            ),
            application=ApplicationConfig.from_value(data["application"]),
            checkpoint_storage=PrivateCheckpointStorageBinding.from_value(
                data["checkpoint_storage"]
            ),
            initial_checkpoint=PrivateInitialCheckpointBinding.from_value(
                data["initial_checkpoint"]
            ),
            run_plan=ExactAttemptRunPlan.from_value(data["run_plan"]),
            seed_documents=tuple(SkillDocument.from_value(item) for item in raw_documents),
            attempt_budget=BudgetVector.from_value(data["attempt_budget"]),
            phi_per_cycle_maximum=BudgetVector.from_value(data["phi_per_cycle_maximum"]),
            b1_run_admission=B1RunAdmission.from_value(data["b1_run_admission"]),
            authoring_authority=SkillAuthoringAuthority.from_value(data["authoring_authority"]),
            protocol=ExperimentProtocol.from_value(data["protocol"]),
            protocol_freeze=ProtocolFreeze.from_value(data["protocol_freeze"]),
            catalog_bundle=FrozenProductionCatalogBundle.from_value(data["catalog_bundle"]),
            runtime=PrivateBenchmarkRuntime.from_value(data["runtime"]),
            training_sequence=PrivateFrozenTaskSequence.from_value(data["training_sequence"]),
            rollout_workflow=RolloutWorkflowBinding.from_value(data["rollout_workflow"]),
            format=data["format"],
        )

    @classmethod
    def read_verified(cls, path: Path, *, expected_sha256: str) -> PrivateBenchmarkAttemptInput:
        data = path.resolve().read_bytes()
        if sha256_bytes(data) != _sha256(expected_sha256, field="expected_sha256"):
            raise ValueError("private benchmark input content changed after request capture")
        return cls.from_value(json.loads(data.decode("utf-8")))


__all__ = ["PRIVATE_BENCHMARK_ATTEMPT_INPUT_FORMAT", "PrivateBenchmarkAttemptInput"]
