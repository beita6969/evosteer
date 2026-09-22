"""Answer-free algorithm identity for one exact published attempt."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from skillev.application_config import ApplicationConfig
from skillev.contracts import (
    SCIENTIFIC_SAMPLING_ALGORITHM,
    JsonValue,
    normalize_json,
    scientific_sampling_schedule_hash,
    stable_hash,
)
from skillev.contracts.identity import validate_sha256
from skillev.evolution import SkillAuthoringAuthority
from skillev.experiments.protocol import (
    AblationArm,
    ArmProtocol,
    CappedFlowWeightArmProtocol,
    ClippedImportanceArmProtocol,
    FrozenTaskSequenceIdentity,
    FullArmProtocol,
    NoBayesianCalibrationArmProtocol,
    NoFlowWeightingArmProtocol,
    PosteriorMeanArmProtocol,
    ResidualOnlyPhaseArmProtocol,
    SchedulePurpose,
    arm_protocol_for_builder_kind,
    arm_protocol_from_value,
)
from skillev.policy import (
    BaseModelArtifactIdentity,
    PublicTokenizerIdentity,
    TokenizerArtifactIdentity,
    TrainableStateIdentity,
)
from skillev.runtime import (
    AttemptRunCursorState,
    BudgetVector,
    ExactAttemptRunPlan,
    RuntimeSnapshotIdentity,
)
from skillev.runtime.attempt_protocol import AttemptBuilderKind

from .formal_execution import FormalExecutionFreeze, ImplementationBuildIdentity

PUBLISHED_ATTEMPT_IDENTITY_FORMAT: Final = "skillev-attempt-public-identity@7"
FORMAL_TRAINING_BINDING_FORMAT: Final = "skillev-formal-training-binding@1"


class AttemptPurpose(StrEnum):
    """Closed publication purpose; fixtures cannot enter formal aggregates."""

    CORRECTNESS_FIXTURE = "correctness-fixture"
    FORMAL_BENCHMARK_TRAINING = "formal-benchmark-training"


_ARM_PROTOCOL_TYPES = (
    FullArmProtocol,
    NoBayesianCalibrationArmProtocol,
    NoFlowWeightingArmProtocol,
    CappedFlowWeightArmProtocol,
    ClippedImportanceArmProtocol,
    PosteriorMeanArmProtocol,
    ResidualOnlyPhaseArmProtocol,
)


def _non_empty_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class FormalTrainingBinding:
    """Path-free identity of the formal training schedule and catalog freeze.

    A correctness fixture deliberately has no such binding.  A formal attempt
    must carry this complete public projection so that it cannot be promoted
    into a paper aggregate merely by changing an attempt-purpose label.
    """

    training_sequence: FrozenTaskSequenceIdentity
    catalog_freeze_hash: str
    format: str = FORMAL_TRAINING_BINDING_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.training_sequence, FrozenTaskSequenceIdentity):
            raise TypeError("formal training binding requires a frozen task sequence")
        if self.training_sequence.purpose is not SchedulePurpose.IID_TRAINING:
            raise ValueError("formal training binding requires an IID training sequence")
        validate_sha256(self.catalog_freeze_hash)
        if self.format != FORMAL_TRAINING_BINDING_FORMAT:
            raise ValueError("unsupported formal training binding format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @property
    def training_sequence_content_hash(self) -> str:
        return self.training_sequence.content_hash

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "catalog_freeze_hash": self.catalog_freeze_hash,
            "format": self.format,
            "training_sequence": self.training_sequence.to_value(),
            "training_sequence_content_hash": self.training_sequence_content_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> FormalTrainingBinding:
        normalized = normalize_json(value)
        fields = {
            "catalog_freeze_hash",
            "format",
            "training_sequence",
            "training_sequence_content_hash",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("formal training binding has incompatible fields")
        if type(normalized["catalog_freeze_hash"]) is not str:
            raise TypeError("formal training binding catalog_freeze_hash must be text")
        if type(normalized["format"]) is not str:
            raise TypeError("formal training binding format must be text")
        if type(normalized["training_sequence_content_hash"]) is not str:
            raise TypeError("formal training binding sequence content hash must be text")
        result = cls(
            catalog_freeze_hash=normalized["catalog_freeze_hash"],
            training_sequence=FrozenTaskSequenceIdentity.from_value(
                normalized["training_sequence"]
            ),
            format=normalized["format"],
        )
        if normalized["training_sequence_content_hash"] != result.training_sequence_content_hash:
            raise ValueError("formal training binding sequence content hash differs from sequence")
        return result


@dataclass(frozen=True, slots=True)
class PublishedAttemptIdentity:
    """All public controls required to reproduce and audit one method attempt.

    It intentionally excludes task/query text, rewards, verifier payloads,
    model paths, credentials, and per-item results.
    """

    builder_kind: AttemptBuilderKind
    purpose: AttemptPurpose
    arm_protocol: ArmProtocol
    application_config: ApplicationConfig
    run_plan: ExactAttemptRunPlan
    initial_optimizer_step: int
    initial_run_cursor: AttemptRunCursorState
    authoring_authority: SkillAuthoringAuthority
    attempt_budget: BudgetVector
    phi_per_cycle_maximum: BudgetVector
    protocol_hash: str
    protocol_freeze_id: str
    exact_input_sha256: str
    backbone_deployment_hash: str
    initial_trainable_state: TrainableStateIdentity
    tokenizer_identity: PublicTokenizerIdentity
    initial_library_version: str
    initial_skill_library_state_hash: str
    ordered_task_sequence_hash: str
    formal_training_binding: FormalTrainingBinding | None = None
    formal_execution: FormalExecutionFreeze | None = None
    base_model_artifact: BaseModelArtifactIdentity | None = None
    tokenizer_artifact: TokenizerArtifactIdentity | None = None
    implementation_build: ImplementationBuildIdentity | None = None
    format: str = PUBLISHED_ATTEMPT_IDENTITY_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.builder_kind, AttemptBuilderKind):
            raise TypeError("builder_kind must be AttemptBuilderKind")
        if not isinstance(self.purpose, AttemptPurpose):
            raise TypeError("attempt purpose must be AttemptPurpose")
        if not isinstance(self.arm_protocol, _ARM_PROTOCOL_TYPES):
            raise TypeError("arm_protocol has an unsupported variant")
        if arm_protocol_from_value(self.arm_protocol.to_value()) != self.arm_protocol:
            raise ValueError("arm_protocol differs from its frozen wire identity")
        if arm_protocol_for_builder_kind(self.builder_kind) != self.arm_protocol:
            raise ValueError("builder kind and arm protocol differ")
        if not isinstance(self.application_config, ApplicationConfig):
            raise TypeError("application_config must be ApplicationConfig")
        if not isinstance(self.run_plan, ExactAttemptRunPlan):
            raise TypeError("run_plan must be ExactAttemptRunPlan")
        if type(self.initial_optimizer_step) is not int or self.initial_optimizer_step < 0:
            raise ValueError("initial_optimizer_step must be non-negative")
        if not isinstance(self.initial_run_cursor, AttemptRunCursorState):
            raise TypeError("initial_run_cursor must be AttemptRunCursorState")
        self.initial_run_cursor.require_plan(self.run_plan)
        if not isinstance(self.authoring_authority, SkillAuthoringAuthority):
            raise TypeError("authoring_authority must be SkillAuthoringAuthority")
        if not isinstance(self.attempt_budget, BudgetVector):
            raise TypeError("attempt_budget must be BudgetVector")
        if not isinstance(self.phi_per_cycle_maximum, BudgetVector):
            raise TypeError("phi_per_cycle_maximum must be BudgetVector")
        if not isinstance(self.tokenizer_identity, PublicTokenizerIdentity):
            raise TypeError("tokenizer_identity must be PublicTokenizerIdentity")
        if self.base_model_artifact is not None and not isinstance(
            self.base_model_artifact,
            BaseModelArtifactIdentity,
        ):
            raise TypeError("base_model_artifact must be BaseModelArtifactIdentity or None")
        if self.tokenizer_artifact is not None and not isinstance(
            self.tokenizer_artifact,
            TokenizerArtifactIdentity,
        ):
            raise TypeError("tokenizer_artifact must be TokenizerArtifactIdentity or None")
        if self.implementation_build is not None and not isinstance(
            self.implementation_build,
            ImplementationBuildIdentity,
        ):
            raise TypeError("implementation_build must be ImplementationBuildIdentity or None")
        if (
            self.tokenizer_artifact is not None
            and self.tokenizer_artifact.content_hash != self.tokenizer_identity.content_hash
        ):
            raise ValueError("tokenizer artifact differs from the public tokenizer identity")
        if not isinstance(self.initial_trainable_state, TrainableStateIdentity):
            raise TypeError("published identity requires initial trainable state")
        if self.initial_trainable_state.backbone_deployment_hash != self.backbone_deployment_hash:
            raise ValueError("initial trainable state targets another backbone deployment")
        for field in (
            "protocol_hash",
            "protocol_freeze_id",
            "exact_input_sha256",
            "backbone_deployment_hash",
            "ordered_task_sequence_hash",
        ):
            validate_sha256(getattr(self, field))
        validate_sha256(self.initial_library_version)
        validate_sha256(self.initial_skill_library_state_hash)
        if self.purpose is AttemptPurpose.FORMAL_BENCHMARK_TRAINING:
            if not isinstance(self.formal_training_binding, FormalTrainingBinding):
                raise ValueError("formal attempt requires a formal training binding")
            if not isinstance(self.formal_execution, FormalExecutionFreeze):
                raise ValueError("formal attempt requires a formal execution freeze")
            if not isinstance(self.base_model_artifact, BaseModelArtifactIdentity):
                raise ValueError("formal attempt requires a base model artifact identity")
            if not isinstance(self.tokenizer_artifact, TokenizerArtifactIdentity):
                raise ValueError("formal attempt requires a tokenizer artifact identity")
            if not isinstance(self.implementation_build, ImplementationBuildIdentity):
                raise ValueError("formal attempt requires an implementation build identity")
            if (
                self.formal_training_binding.training_sequence.ordered_task_ids_hash
                != self.ordered_task_sequence_hash
            ):
                raise ValueError("formal training binding differs from ordered task sequence")
            expected_task_count = (
                self.application_config.trainer.execution.batch_size
                * self.run_plan.total_training_steps
            )
            if self.formal_training_binding.training_sequence.task_count != expected_task_count:
                raise ValueError("formal training binding does not fill the exact run plan")
            self._require_formal_execution_alignment()
        elif any(
            value is not None
            for value in (
                self.formal_training_binding,
                self.formal_execution,
                self.base_model_artifact,
                self.tokenizer_artifact,
                self.implementation_build,
            )
        ):
            raise ValueError("correctness fixtures cannot carry formal execution identity")
        if self.format != PUBLISHED_ATTEMPT_IDENTITY_FORMAT:
            raise ValueError("unsupported published attempt identity format")

    @property
    def arm(self) -> AblationArm:
        return self.arm_protocol.arm

    @property
    def application_config_hash(self) -> str:
        return self.application_config.content_hash

    @property
    def authoring_authority_hash(self) -> str:
        return stable_hash(self.authoring_authority.to_value())

    @property
    def attempt_budget_hash(self) -> str:
        return stable_hash(self.attempt_budget.to_value())

    @property
    def phi_per_cycle_budget_hash(self) -> str:
        return stable_hash(self.phi_per_cycle_maximum.to_value())

    @property
    def formal_execution_hash(self) -> str | None:
        return self.formal_execution.content_hash if self.formal_execution is not None else None

    @property
    def sampling_schedule_algorithm(self) -> str:
        return SCIENTIFIC_SAMPLING_ALGORITHM

    @property
    def sampling_schedule_hash(self) -> str:
        return scientific_sampling_schedule_hash(
            base_seed=self.application_config.trainer.rollout.base_seed
        )

    @property
    def method_identity_hash(self) -> str:
        return stable_hash(
            {
                "application_config_hash": self.application_config_hash,
                "arm_protocol": self.arm_protocol.to_value(),
                "authoring_authority": self.authoring_authority.to_value(),
                "backbone_deployment_hash": self.backbone_deployment_hash,
                "base_model_artifact": (
                    self.base_model_artifact.to_value() if self.base_model_artifact else None
                ),
                "builder_kind": self.builder_kind.value,
                "initial_library_version": self.initial_library_version,
                "initial_skill_library_state_hash": self.initial_skill_library_state_hash,
                "initial_trainable_state": self.initial_trainable_state.to_value(),
                "initial_optimizer_step": self.initial_optimizer_step,
                "initial_run_cursor": self.initial_run_cursor.to_value(),
                "ordered_task_sequence_hash": self.ordered_task_sequence_hash,
                "formal_training_binding": (
                    self.formal_training_binding.to_value()
                    if self.formal_training_binding is not None
                    else None
                ),
                "formal_execution": (
                    self.formal_execution.to_value() if self.formal_execution else None
                ),
                "implementation_build": (
                    self.implementation_build.to_value() if self.implementation_build else None
                ),
                "phi_per_cycle_maximum": self.phi_per_cycle_maximum.to_value(),
                "protocol_freeze_id": self.protocol_freeze_id,
                "protocol_hash": self.protocol_hash,
                "purpose": self.purpose.value,
                "run_plan_hash": self.run_plan.content_hash,
                "sampling_schedule_algorithm": self.sampling_schedule_algorithm,
                "sampling_schedule_hash": self.sampling_schedule_hash,
                "tokenizer_identity": self.tokenizer_identity.to_value(),
                "tokenizer_artifact": (
                    self.tokenizer_artifact.to_value() if self.tokenizer_artifact else None
                ),
            }
        )

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    @property
    def phase_checkpoint_cycle_ordinals(self) -> tuple[int, ...]:
        """Exact formal phase ordinals that must publish a progress checkpoint."""

        if self.formal_execution is None:
            return ()
        return self.formal_execution.progress_anchor_cycle_ordinals

    def runtime_snapshot_identity(self) -> RuntimeSnapshotIdentity:
        return RuntimeSnapshotIdentity(
            builder_kind=self.builder_kind,
            public_identity_content_hash=self.content_hash,
            application_config_hash=self.application_config_hash,
            method_identity_hash=self.method_identity_hash,
            protocol_hash=self.protocol_hash,
            protocol_freeze_id=self.protocol_freeze_id,
            run_plan_hash=self.run_plan.content_hash,
            initial_library_version=self.initial_library_version,
            initial_skill_library_state_hash=self.initial_skill_library_state_hash,
            initial_trainable_state_hash=self.initial_trainable_state.content_hash,
            ordered_task_sequence_hash=self.ordered_task_sequence_hash,
            sampling_schedule_algorithm=self.sampling_schedule_algorithm,
            sampling_schedule_hash=self.sampling_schedule_hash,
            formal_execution_hash=self.formal_execution_hash,
            base_model_artifact_hash=(
                self.base_model_artifact.content_hash if self.base_model_artifact else None
            ),
            tokenizer_artifact_hash=(
                self.tokenizer_artifact.content_hash if self.tokenizer_artifact else None
            ),
            implementation_build_hash=(
                self.implementation_build.content_hash if self.implementation_build else None
            ),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "application_config": self.application_config.to_value(),
            "arm_protocol": self.arm_protocol.to_value(),
            "attempt_budget": self.attempt_budget.to_value(),
            "authoring_authority": self.authoring_authority.to_value(),
            "backbone_deployment_hash": self.backbone_deployment_hash,
            "base_model_artifact": (
                self.base_model_artifact.to_value() if self.base_model_artifact else None
            ),
            "builder_kind": self.builder_kind.value,
            "exact_input_sha256": self.exact_input_sha256,
            "format": self.format,
            "formal_training_binding": (
                self.formal_training_binding.to_value()
                if self.formal_training_binding is not None
                else None
            ),
            "formal_execution": (
                self.formal_execution.to_value() if self.formal_execution else None
            ),
            "implementation_build": (
                self.implementation_build.to_value() if self.implementation_build else None
            ),
            "initial_library_version": self.initial_library_version,
            "initial_skill_library_state_hash": self.initial_skill_library_state_hash,
            "initial_optimizer_step": self.initial_optimizer_step,
            "initial_run_cursor": self.initial_run_cursor.to_value(),
            "initial_trainable_state": self.initial_trainable_state.to_value(),
            "ordered_task_sequence_hash": self.ordered_task_sequence_hash,
            "phi_per_cycle_maximum": self.phi_per_cycle_maximum.to_value(),
            "protocol_freeze_id": self.protocol_freeze_id,
            "protocol_hash": self.protocol_hash,
            "purpose": self.purpose.value,
            "run_plan": self.run_plan.to_value(),
            "sampling_schedule_algorithm": self.sampling_schedule_algorithm,
            "sampling_schedule_hash": self.sampling_schedule_hash,
            "tokenizer_identity": self.tokenizer_identity.to_value(),
            "tokenizer_artifact": (
                self.tokenizer_artifact.to_value() if self.tokenizer_artifact else None
            ),
        }

    @classmethod
    def from_value(cls, value: object) -> PublishedAttemptIdentity:
        normalized = normalize_json(value)
        fields = {
            "application_config",
            "arm_protocol",
            "attempt_budget",
            "authoring_authority",
            "backbone_deployment_hash",
            "base_model_artifact",
            "builder_kind",
            "exact_input_sha256",
            "format",
            "formal_training_binding",
            "formal_execution",
            "implementation_build",
            "initial_library_version",
            "initial_skill_library_state_hash",
            "initial_optimizer_step",
            "initial_run_cursor",
            "initial_trainable_state",
            "ordered_task_sequence_hash",
            "phi_per_cycle_maximum",
            "protocol_freeze_id",
            "protocol_hash",
            "purpose",
            "run_plan",
            "sampling_schedule_algorithm",
            "sampling_schedule_hash",
            "tokenizer_identity",
            "tokenizer_artifact",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("PublishedAttemptIdentity has incompatible fields")
        text_fields = {
            "backbone_deployment_hash",
            "builder_kind",
            "exact_input_sha256",
            "format",
            "initial_library_version",
            "initial_skill_library_state_hash",
            "ordered_task_sequence_hash",
            "protocol_freeze_id",
            "protocol_hash",
            "purpose",
            "sampling_schedule_algorithm",
            "sampling_schedule_hash",
        }
        if any(type(normalized[field]) is not str for field in text_fields):
            raise TypeError("published identity text fields have incompatible types")
        if type(normalized["initial_optimizer_step"]) is not int:
            raise TypeError("initial_optimizer_step must be an integer")
        application_config = ApplicationConfig.from_value(normalized["application_config"])
        if normalized["sampling_schedule_algorithm"] != SCIENTIFIC_SAMPLING_ALGORITHM:
            raise ValueError("published identity sampling algorithm is incompatible")
        if normalized["sampling_schedule_hash"] != scientific_sampling_schedule_hash(
            base_seed=application_config.trainer.rollout.base_seed
        ):
            raise ValueError("published identity sampling schedule differs from config")
        raw_formal_training = normalized["formal_training_binding"]
        if raw_formal_training is not None and not isinstance(raw_formal_training, dict):
            raise TypeError("formal_training_binding must be an object or null")
        nullable_object_fields = (
            "base_model_artifact",
            "formal_execution",
            "implementation_build",
            "tokenizer_artifact",
        )
        if any(
            normalized[field] is not None and not isinstance(normalized[field], dict)
            for field in nullable_object_fields
        ):
            raise TypeError("published formal artifact identities must be objects or null")
        return cls(
            builder_kind=AttemptBuilderKind(normalized["builder_kind"]),
            purpose=AttemptPurpose(normalized["purpose"]),
            arm_protocol=arm_protocol_from_value(normalized["arm_protocol"]),
            application_config=application_config,
            run_plan=ExactAttemptRunPlan.from_value(normalized["run_plan"]),
            initial_optimizer_step=normalized["initial_optimizer_step"],
            initial_run_cursor=AttemptRunCursorState.from_value(normalized["initial_run_cursor"]),
            authoring_authority=SkillAuthoringAuthority.from_value(
                normalized["authoring_authority"]
            ),
            attempt_budget=BudgetVector.from_value(normalized["attempt_budget"]),
            phi_per_cycle_maximum=BudgetVector.from_value(normalized["phi_per_cycle_maximum"]),
            protocol_hash=normalized["protocol_hash"],
            protocol_freeze_id=normalized["protocol_freeze_id"],
            exact_input_sha256=normalized["exact_input_sha256"],
            backbone_deployment_hash=normalized["backbone_deployment_hash"],
            initial_trainable_state=TrainableStateIdentity.from_value(
                normalized["initial_trainable_state"]
            ),
            tokenizer_identity=PublicTokenizerIdentity.from_value(normalized["tokenizer_identity"]),
            initial_library_version=normalized["initial_library_version"],
            initial_skill_library_state_hash=normalized["initial_skill_library_state_hash"],
            ordered_task_sequence_hash=normalized["ordered_task_sequence_hash"],
            formal_training_binding=(
                FormalTrainingBinding.from_value(raw_formal_training)
                if raw_formal_training is not None
                else None
            ),
            formal_execution=(
                FormalExecutionFreeze.from_value(normalized["formal_execution"])
                if normalized["formal_execution"] is not None
                else None
            ),
            base_model_artifact=(
                BaseModelArtifactIdentity.from_value(normalized["base_model_artifact"])
                if normalized["base_model_artifact"] is not None
                else None
            ),
            tokenizer_artifact=(
                TokenizerArtifactIdentity.from_value(normalized["tokenizer_artifact"])
                if normalized["tokenizer_artifact"] is not None
                else None
            ),
            implementation_build=(
                ImplementationBuildIdentity.from_value(normalized["implementation_build"])
                if normalized["implementation_build"] is not None
                else None
            ),
            format=normalized["format"],
        )

    def _require_formal_execution_alignment(self) -> None:
        """Require the public attempt to be an exact projection of its freeze."""

        assert self.formal_execution is not None  # narrowed by __post_init__
        assert self.formal_training_binding is not None
        assert self.base_model_artifact is not None
        assert self.tokenizer_artifact is not None
        assert self.implementation_build is not None
        freeze = self.formal_execution
        if freeze.backbone_deployment_hash != self.backbone_deployment_hash:
            raise ValueError("formal execution freeze targets another backbone deployment")
        if freeze.base_model_artifact != self.base_model_artifact:
            raise ValueError("formal execution freeze has another base model artifact")
        if freeze.tokenizer_artifact != self.tokenizer_artifact:
            raise ValueError("formal execution freeze has another tokenizer artifact")
        if freeze.implementation_build != self.implementation_build:
            raise ValueError("formal execution freeze has another implementation build")
        if freeze.run_plan != self.run_plan:
            raise ValueError("formal execution freeze has another run plan")
        if freeze.initial_trainable_state != self.initial_trainable_state:
            raise ValueError("formal execution freeze has another initial trainable state")
        if freeze.initial_library_version != self.initial_library_version:
            raise ValueError("formal execution freeze has another initial library")
        if freeze.initial_skill_library_state_hash != self.initial_skill_library_state_hash:
            raise ValueError("formal execution freeze has another initial library state")
        if freeze.authoring_authority != self.authoring_authority:
            raise ValueError("formal execution freeze has another authoring authority")
        if freeze.attempt_budget != self.attempt_budget:
            raise ValueError("formal execution freeze has another attempt budget")
        if freeze.phi_per_cycle_maximum != self.phi_per_cycle_maximum:
            raise ValueError("formal execution freeze has another Phi budget")
        if freeze.training_sequence != self.formal_training_binding.training_sequence:
            raise ValueError("formal execution freeze has another training sequence")
        if freeze.catalog_freeze_hash != self.formal_training_binding.catalog_freeze_hash:
            raise ValueError("formal execution freeze has another catalog freeze")
        if freeze.seed != self.application_config.trainer.rollout.base_seed:
            raise ValueError("formal execution freeze seed differs from rollout seed")
        if (
            freeze.sampling_schedule_algorithm != self.sampling_schedule_algorithm
            or freeze.sampling_schedule_hash != self.sampling_schedule_hash
        ):
            raise ValueError("formal execution freeze has another sampling schedule")
        freeze.requires_application(self.application_config)


__all__ = [
    "FORMAL_TRAINING_BINDING_FORMAT",
    "PUBLISHED_ATTEMPT_IDENTITY_FORMAT",
    "AttemptPurpose",
    "FormalTrainingBinding",
    "PublishedAttemptIdentity",
]
