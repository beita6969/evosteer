"""Immutable, result-blind execution identity for formal seven-arm runs."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.application_config import ApplicationConfig
from skillev.contracts import (
    SCIENTIFIC_SAMPLING_ALGORITHM,
    JsonValue,
    canonical_json,
    normalize_json,
    parse_canonical_json,
    scientific_sampling_schedule_hash,
    stable_hash,
)
from skillev.contracts.identity import validate_sha256
from skillev.evolution import SkillAuthoringAuthority
from skillev.policy import (
    BaseModelArtifactIdentity,
    TokenizerArtifactIdentity,
    TrainableStateIdentity,
)
from skillev.runtime import AttemptBuilderKind, BudgetVector, ExactAttemptRunPlan

from .b1_run_admission import B1RunAdmission
from .protocol import FrozenTaskSequenceIdentity, SchedulePurpose

IMPLEMENTATION_BUILD_IDENTITY_FORMAT = "skillev-implementation-build@2"
FORMAL_IMPLEMENTATION_BUILD_ATTESTATION_FORMAT = "skillev-formal-implementation-build-attestation@1"
FORMAL_EXECUTION_FREEZE_FORMAT = "skillev-formal-execution-freeze@6"


def formal_application_config_value(config: ApplicationConfig) -> dict[str, JsonValue]:
    """Return every scientific application control without artifact namespaces.

    ``experiment_id`` selects a storage namespace only.  Every other typed
    application control, including the rollout seed, batch population, and
    checkpoint cadence, remains an explicit part of the formal freeze.
    """

    if not isinstance(config, ApplicationConfig):
        raise TypeError("formal application config requires ApplicationConfig")
    value = normalize_json(config.to_value())
    if not isinstance(value, dict):  # pragma: no cover - typed source invariant
        raise TypeError("application config did not serialize to an object")
    trainer = value["trainer"]
    if not isinstance(trainer, dict):  # pragma: no cover - typed source invariant
        raise TypeError("application trainer did not serialize to an object")
    execution = trainer["execution"]
    if not isinstance(execution, dict):  # pragma: no cover - typed source invariant
        raise TypeError("application execution did not serialize to an object")
    if set(execution) != {"batch_size", "experiment_id", "format"}:
        raise ValueError("application execution config has incompatible fields")
    return {
        **value,
        "trainer": {
            **trainer,
            "execution": {
                "batch_size": execution["batch_size"],
                "format": execution["format"],
            },
        },
    }


def _scientific_rollout_seed(value: dict[str, JsonValue]) -> int:
    """Read the one rollout seed retained in a path-free formal config."""

    trainer = value.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("formal application config lacks a trainer object")
    rollout = trainer.get("rollout")
    if not isinstance(rollout, dict):
        raise ValueError("formal application config lacks a rollout object")
    seed = rollout.get("base_seed")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("formal application config rollout seed is invalid")
    return seed


def _scientific_batch_size(value: dict[str, JsonValue]) -> int:
    """Read the exact batch population retained in a formal config."""

    trainer = value.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("formal application config lacks a trainer object")
    execution = trainer.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("formal application config lacks an execution object")
    batch_size = execution.get("batch_size")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("formal application config batch size is invalid")
    return batch_size


@dataclass(frozen=True, slots=True)
class ImplementationBuildIdentity:
    """Path-free identity of the executable source, wheels, lock, and runtime."""

    source_commit: str
    source_tree_hash: str
    public_wheel_hash: str
    private_evaluation_wheel_hash: str
    lockfile_hash: str
    python_version: str
    torch_version: str
    transformers_version: str
    peft_version: str
    tokenizers_version: str
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    matmul_allow_tf32: bool
    format: str = IMPLEMENTATION_BUILD_IDENTITY_FORMAT

    def __post_init__(self) -> None:
        if (
            type(self.source_commit) is not str
            or len(self.source_commit) != 40
            or any(character not in "0123456789abcdef" for character in self.source_commit)
        ):
            raise ValueError("source_commit must be a full lowercase Git commit")
        for field in (
            "source_tree_hash",
            "public_wheel_hash",
            "private_evaluation_wheel_hash",
            "lockfile_hash",
        ):
            validate_sha256(getattr(self, field))
        for field in (
            "python_version",
            "torch_version",
            "transformers_version",
            "peft_version",
            "tokenizers_version",
        ):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise ValueError(f"{field} must be non-empty text")
        for field in (
            "deterministic_algorithms",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "matmul_allow_tf32",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be a boolean")
        if self.format != IMPLEMENTATION_BUILD_IDENTITY_FORMAT:
            raise ValueError("unsupported implementation build identity format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "cudnn_benchmark": self.cudnn_benchmark,
            "cudnn_deterministic": self.cudnn_deterministic,
            "deterministic_algorithms": self.deterministic_algorithms,
            "format": self.format,
            "lockfile_hash": self.lockfile_hash,
            "matmul_allow_tf32": self.matmul_allow_tf32,
            "peft_version": self.peft_version,
            "private_evaluation_wheel_hash": self.private_evaluation_wheel_hash,
            "public_wheel_hash": self.public_wheel_hash,
            "python_version": self.python_version,
            "source_commit": self.source_commit,
            "source_tree_hash": self.source_tree_hash,
            "tokenizers_version": self.tokenizers_version,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
        }

    @classmethod
    def from_value(cls, value: object) -> ImplementationBuildIdentity:
        normalized = normalize_json(value)
        fields = {
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
            "format",
            "lockfile_hash",
            "matmul_allow_tf32",
            "peft_version",
            "private_evaluation_wheel_hash",
            "public_wheel_hash",
            "python_version",
            "source_commit",
            "source_tree_hash",
            "tokenizers_version",
            "torch_version",
            "transformers_version",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("implementation build identity has incompatible fields")
        text_fields = fields - {
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
            "matmul_allow_tf32",
        }
        if any(type(normalized[field]) is not str for field in text_fields):
            raise TypeError("implementation build identity text fields must be strings")
        if any(type(normalized[field]) is not bool for field in fields - text_fields):
            raise TypeError("implementation build identity backend flags must be booleans")
        return cls(
            source_commit=normalized["source_commit"],
            source_tree_hash=normalized["source_tree_hash"],
            public_wheel_hash=normalized["public_wheel_hash"],
            private_evaluation_wheel_hash=normalized["private_evaluation_wheel_hash"],
            lockfile_hash=normalized["lockfile_hash"],
            python_version=normalized["python_version"],
            torch_version=normalized["torch_version"],
            transformers_version=normalized["transformers_version"],
            peft_version=normalized["peft_version"],
            tokenizers_version=normalized["tokenizers_version"],
            deterministic_algorithms=normalized["deterministic_algorithms"],
            cudnn_deterministic=normalized["cudnn_deterministic"],
            cudnn_benchmark=normalized["cudnn_benchmark"],
            matmul_allow_tf32=normalized["matmul_allow_tf32"],
            format=normalized["format"],
        )


@dataclass(frozen=True, slots=True)
class FormalImplementationBuildAttestation:
    """One path-free proof that a formal child measured its executing build.

    The event is emitted before the builder can load a model.  It exists
    because formal inputs can otherwise name one implementation build while a
    child process executes another one.
    """

    attempt_id: str
    builder_kind: AttemptBuilderKind
    exact_input_sha256: str
    public_identity_content_hash: str
    formal_execution_content_hash: str
    expected_implementation_build: ImplementationBuildIdentity
    measured_implementation_build: ImplementationBuildIdentity
    format: str = FORMAL_IMPLEMENTATION_BUILD_ATTESTATION_FORMAT

    def __post_init__(self) -> None:
        if (
            type(self.attempt_id) is not str
            or not self.attempt_id.strip()
            or "\x00" in self.attempt_id
        ):
            raise ValueError("formal build attestation attempt_id must be non-empty text")
        if not isinstance(self.builder_kind, AttemptBuilderKind):
            raise TypeError("formal build attestation requires a closed builder kind")
        for field in (
            "exact_input_sha256",
            "public_identity_content_hash",
            "formal_execution_content_hash",
        ):
            validate_sha256(getattr(self, field))
        if not isinstance(self.expected_implementation_build, ImplementationBuildIdentity):
            raise TypeError("formal build attestation expected build is invalid")
        if not isinstance(self.measured_implementation_build, ImplementationBuildIdentity):
            raise TypeError("formal build attestation measured build is invalid")
        if self.measured_implementation_build != self.expected_implementation_build:
            raise ValueError("formal child measured another implementation build")
        if self.format != FORMAL_IMPLEMENTATION_BUILD_ATTESTATION_FORMAT:
            raise ValueError("unsupported formal implementation build attestation format")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_id": self.attempt_id,
            "builder_kind": self.builder_kind.value,
            "exact_input_sha256": self.exact_input_sha256,
            "expected_implementation_build": self.expected_implementation_build.to_value(),
            "formal_execution_content_hash": self.formal_execution_content_hash,
            "format": self.format,
            "measured_implementation_build": self.measured_implementation_build.to_value(),
            "public_identity_content_hash": self.public_identity_content_hash,
        }

    @classmethod
    def from_value(cls, value: object) -> FormalImplementationBuildAttestation:
        normalized = normalize_json(value)
        fields = {
            "attempt_id",
            "builder_kind",
            "exact_input_sha256",
            "expected_implementation_build",
            "formal_execution_content_hash",
            "format",
            "measured_implementation_build",
            "public_identity_content_hash",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("formal implementation build attestation has incompatible fields")
        for field in (
            "attempt_id",
            "builder_kind",
            "exact_input_sha256",
            "formal_execution_content_hash",
            "format",
            "public_identity_content_hash",
        ):
            if type(normalized[field]) is not str:
                raise TypeError("formal implementation build attestation text fields must be text")
        return cls(
            attempt_id=normalized["attempt_id"],
            builder_kind=AttemptBuilderKind(normalized["builder_kind"]),
            exact_input_sha256=normalized["exact_input_sha256"],
            public_identity_content_hash=normalized["public_identity_content_hash"],
            formal_execution_content_hash=normalized["formal_execution_content_hash"],
            expected_implementation_build=ImplementationBuildIdentity.from_value(
                normalized["expected_implementation_build"]
            ),
            measured_implementation_build=ImplementationBuildIdentity.from_value(
                normalized["measured_implementation_build"]
            ),
            format=normalized["format"],
        )


@dataclass(frozen=True, slots=True)
class FormalExecutionFreeze:
    """All result-affecting controls fixed before a formal run begins."""

    seed: int
    backbone_kind: str
    backbone_deployment_hash: str
    base_model_artifact: BaseModelArtifactIdentity
    tokenizer_artifact: TokenizerArtifactIdentity
    implementation_build: ImplementationBuildIdentity
    application_scientific_config_json: str
    application_scientific_config_hash: str
    run_plan: ExactAttemptRunPlan
    initial_trainable_state: TrainableStateIdentity
    initial_library_version: str
    initial_skill_library_state_hash: str
    authoring_authority: SkillAuthoringAuthority
    attempt_budget: BudgetVector
    phi_per_cycle_maximum: BudgetVector
    b1_run_admission: B1RunAdmission
    training_sequence: FrozenTaskSequenceIdentity
    catalog_freeze_hash: str
    progress_anchor_cycle_ordinals: tuple[int, ...]
    sampling_schedule_algorithm: str
    sampling_schedule_hash: str
    format: str = FORMAL_EXECUTION_FREEZE_FORMAT

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("formal execution seed must be an unsigned 64-bit integer")
        if type(self.backbone_kind) is not str or not self.backbone_kind:
            raise ValueError("backbone_kind must be non-empty text")
        validate_sha256(self.initial_library_version)
        validate_sha256(self.initial_skill_library_state_hash)
        for field in (
            "backbone_deployment_hash",
            "application_scientific_config_hash",
            "catalog_freeze_hash",
            "sampling_schedule_hash",
        ):
            validate_sha256(getattr(self, field))
        if self.sampling_schedule_algorithm != SCIENTIFIC_SAMPLING_ALGORITHM:
            raise ValueError("formal execution uses another sampling algorithm")
        if self.sampling_schedule_hash != scientific_sampling_schedule_hash(base_seed=self.seed):
            raise ValueError("formal execution sampling schedule differs from its seed")
        if not isinstance(self.base_model_artifact, BaseModelArtifactIdentity):
            raise TypeError("formal execution requires a base model artifact identity")
        if not isinstance(self.tokenizer_artifact, TokenizerArtifactIdentity):
            raise TypeError("formal execution requires a tokenizer artifact identity")
        if not isinstance(self.implementation_build, ImplementationBuildIdentity):
            raise TypeError("formal execution requires an implementation build identity")
        if not isinstance(self.run_plan, ExactAttemptRunPlan):
            raise TypeError("formal execution requires an exact run plan")
        if not isinstance(self.initial_trainable_state, TrainableStateIdentity):
            raise TypeError("formal execution requires an initial trainable state")
        if self.initial_trainable_state.backbone_deployment_hash != self.backbone_deployment_hash:
            raise ValueError("formal initial trainable state targets another deployment")
        if not isinstance(self.authoring_authority, SkillAuthoringAuthority):
            raise TypeError("formal execution requires an authoring authority")
        if not isinstance(self.attempt_budget, BudgetVector):
            raise TypeError("formal execution requires an attempt budget")
        if not isinstance(self.phi_per_cycle_maximum, BudgetVector):
            raise TypeError("formal execution requires a Phi budget")
        if not isinstance(self.b1_run_admission, B1RunAdmission):
            raise TypeError("formal execution requires a B1 run admission")
        if self.b1_run_admission.run_plan != self.run_plan:
            raise ValueError("B1 run admission has another exact run plan")
        if self.b1_run_admission.attempt_budget != self.attempt_budget:
            raise ValueError("B1 run admission has another attempt budget")
        if self.b1_run_admission.phi_per_cycle_maximum != self.phi_per_cycle_maximum:
            raise ValueError("B1 run admission has another per-cycle Phi budget")
        if self.b1_run_admission.initial_library_version != self.initial_library_version:
            raise ValueError("B1 run admission has another initial library")
        if not isinstance(self.training_sequence, FrozenTaskSequenceIdentity):
            raise TypeError("formal execution requires a frozen training sequence")
        if self.training_sequence.purpose is not SchedulePurpose.IID_TRAINING:
            raise ValueError("formal execution requires an IID training sequence")
        if (
            not isinstance(self.progress_anchor_cycle_ordinals, tuple)
            or any(
                type(item) is not int or item < 1 for item in self.progress_anchor_cycle_ordinals
            )
            or tuple(sorted(set(self.progress_anchor_cycle_ordinals)))
            != self.progress_anchor_cycle_ordinals
        ):
            raise ValueError("progress anchor cycle ordinals must be sorted unique positives")
        expected_anchor_cycles = tuple(range(1, self.run_plan.maximum_cycles + 1))
        if self.progress_anchor_cycle_ordinals != expected_anchor_cycles:
            raise ValueError("formal execution requires every run-plan phase checkpoint")
        if self.format != FORMAL_EXECUTION_FREEZE_FORMAT:
            raise ValueError("unsupported formal execution freeze format")
        try:
            parsed_config = parse_canonical_json(self.application_scientific_config_json)
        except ValueError as error:
            raise ValueError("formal application config must be canonical JSON") from error
        if not isinstance(parsed_config, dict):
            raise ValueError("formal application config must be a JSON object")
        if stable_hash(parsed_config) != self.application_scientific_config_hash:
            raise ValueError("formal application config hash differs from its canonical value")
        if _scientific_rollout_seed(parsed_config) != self.seed:
            raise ValueError("formal application config rollout seed differs from formal seed")
        if parsed_config.get("maximum_h0_tokens") != self.b1_run_admission.maximum_h0_tokens:
            raise ValueError("formal application H0 cap differs from the B1 admission")
        if self.training_sequence.task_count != (
            _scientific_batch_size(parsed_config) * self.run_plan.total_training_steps
        ):
            raise ValueError("formal training sequence does not fill the exact run plan")

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        backbone_kind: str,
        backbone_deployment_hash: str,
        base_model_artifact: BaseModelArtifactIdentity,
        tokenizer_artifact: TokenizerArtifactIdentity,
        implementation_build: ImplementationBuildIdentity,
        application: ApplicationConfig,
        run_plan: ExactAttemptRunPlan,
        initial_trainable_state: TrainableStateIdentity,
        initial_library_version: str,
        initial_skill_library_state_hash: str,
        authoring_authority: SkillAuthoringAuthority,
        attempt_budget: BudgetVector,
        phi_per_cycle_maximum: BudgetVector,
        b1_run_admission: B1RunAdmission,
        training_sequence: FrozenTaskSequenceIdentity,
        catalog_freeze_hash: str,
    ) -> FormalExecutionFreeze:
        application_value = formal_application_config_value(application)
        if _scientific_rollout_seed(application_value) != seed:
            raise ValueError("formal execution seed differs from application rollout seed")
        if training_sequence.task_count != (
            _scientific_batch_size(application_value) * run_plan.total_training_steps
        ):
            raise ValueError("formal training sequence does not fill the exact run plan")
        return cls(
            seed=seed,
            backbone_kind=backbone_kind,
            backbone_deployment_hash=backbone_deployment_hash,
            base_model_artifact=base_model_artifact,
            tokenizer_artifact=tokenizer_artifact,
            implementation_build=implementation_build,
            application_scientific_config_json=canonical_json(application_value),
            application_scientific_config_hash=stable_hash(application_value),
            run_plan=run_plan,
            initial_trainable_state=initial_trainable_state,
            initial_library_version=initial_library_version,
            initial_skill_library_state_hash=initial_skill_library_state_hash,
            authoring_authority=authoring_authority,
            attempt_budget=attempt_budget,
            phi_per_cycle_maximum=phi_per_cycle_maximum,
            b1_run_admission=b1_run_admission,
            training_sequence=training_sequence,
            catalog_freeze_hash=catalog_freeze_hash,
            progress_anchor_cycle_ordinals=tuple(range(1, run_plan.maximum_cycles + 1)),
            sampling_schedule_algorithm=SCIENTIFIC_SAMPLING_ALGORITHM,
            sampling_schedule_hash=scientific_sampling_schedule_hash(base_seed=seed),
        )

    @property
    def application_scientific_config(self) -> dict[str, JsonValue]:
        parsed = parse_canonical_json(self.application_scientific_config_json)
        if not isinstance(parsed, dict):  # pragma: no cover - post-init invariant
            raise RuntimeError("formal application config lost object shape")
        return parsed

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())

    def requires_application(self, application: ApplicationConfig) -> None:
        if formal_application_config_value(application) != self.application_scientific_config:
            raise ValueError(
                "application scientific config differs from the formal execution freeze"
            )

    def requires_progress_anchor_cycle(self, cycle_ordinal: int) -> None:
        """Require this phase ordinal to have been fixed before result ingestion."""

        if type(cycle_ordinal) is not int or cycle_ordinal < 1:
            raise ValueError("progress anchor cycle ordinal must be positive")
        if cycle_ordinal not in self.progress_anchor_cycle_ordinals:
            raise ValueError("phase cycle is not preregistered for a progress anchor")

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "application_scientific_config": self.application_scientific_config,
            "application_scientific_config_hash": self.application_scientific_config_hash,
            "attempt_budget": self.attempt_budget.to_value(),
            "authoring_authority": self.authoring_authority.to_value(),
            "b1_run_admission": self.b1_run_admission.to_value(),
            "backbone_deployment_hash": self.backbone_deployment_hash,
            "backbone_kind": self.backbone_kind,
            "base_model_artifact": self.base_model_artifact.to_value(),
            "catalog_freeze_hash": self.catalog_freeze_hash,
            "format": self.format,
            "implementation_build": self.implementation_build.to_value(),
            "initial_library_version": self.initial_library_version,
            "initial_skill_library_state_hash": self.initial_skill_library_state_hash,
            "initial_trainable_state": self.initial_trainable_state.to_value(),
            "phi_per_cycle_maximum": self.phi_per_cycle_maximum.to_value(),
            "progress_anchor_cycle_ordinals": list(self.progress_anchor_cycle_ordinals),
            "run_plan": self.run_plan.to_value(),
            "sampling_schedule_algorithm": self.sampling_schedule_algorithm,
            "sampling_schedule_hash": self.sampling_schedule_hash,
            "seed": self.seed,
            "tokenizer_artifact": self.tokenizer_artifact.to_value(),
            "training_sequence": self.training_sequence.to_value(),
        }

    @classmethod
    def from_value(cls, value: object) -> FormalExecutionFreeze:
        normalized = normalize_json(value)
        fields = {
            "application_scientific_config",
            "application_scientific_config_hash",
            "attempt_budget",
            "authoring_authority",
            "b1_run_admission",
            "backbone_deployment_hash",
            "backbone_kind",
            "base_model_artifact",
            "catalog_freeze_hash",
            "format",
            "implementation_build",
            "initial_library_version",
            "initial_skill_library_state_hash",
            "initial_trainable_state",
            "phi_per_cycle_maximum",
            "progress_anchor_cycle_ordinals",
            "run_plan",
            "sampling_schedule_algorithm",
            "sampling_schedule_hash",
            "seed",
            "tokenizer_artifact",
            "training_sequence",
        }
        if not isinstance(normalized, dict) or set(normalized) != fields:
            raise ValueError("formal execution freeze has incompatible fields")
        text_fields = {
            "application_scientific_config_hash",
            "backbone_deployment_hash",
            "backbone_kind",
            "catalog_freeze_hash",
            "format",
            "initial_library_version",
            "initial_skill_library_state_hash",
            "sampling_schedule_algorithm",
            "sampling_schedule_hash",
        }
        if any(type(normalized[field]) is not str for field in text_fields):
            raise TypeError("formal execution freeze text fields must be strings")
        if type(normalized["seed"]) is not int:
            raise TypeError("formal execution freeze seed must be an integer")
        raw_anchor_cycles = normalized["progress_anchor_cycle_ordinals"]
        if not isinstance(raw_anchor_cycles, list):
            raise TypeError("formal execution progress anchor cycles must be an array")
        application_value = normalized["application_scientific_config"]
        if not isinstance(application_value, dict):
            raise TypeError("formal execution application config must be an object")
        return cls(
            seed=normalized["seed"],
            backbone_kind=normalized["backbone_kind"],
            backbone_deployment_hash=normalized["backbone_deployment_hash"],
            base_model_artifact=BaseModelArtifactIdentity.from_value(
                normalized["base_model_artifact"]
            ),
            tokenizer_artifact=TokenizerArtifactIdentity.from_value(
                normalized["tokenizer_artifact"]
            ),
            implementation_build=ImplementationBuildIdentity.from_value(
                normalized["implementation_build"]
            ),
            application_scientific_config_json=canonical_json(application_value),
            application_scientific_config_hash=normalized["application_scientific_config_hash"],
            run_plan=ExactAttemptRunPlan.from_value(normalized["run_plan"]),
            initial_trainable_state=TrainableStateIdentity.from_value(
                normalized["initial_trainable_state"]
            ),
            initial_library_version=normalized["initial_library_version"],
            initial_skill_library_state_hash=normalized["initial_skill_library_state_hash"],
            authoring_authority=SkillAuthoringAuthority.from_value(
                normalized["authoring_authority"]
            ),
            attempt_budget=BudgetVector.from_value(normalized["attempt_budget"]),
            phi_per_cycle_maximum=BudgetVector.from_value(normalized["phi_per_cycle_maximum"]),
            b1_run_admission=B1RunAdmission.from_value(normalized["b1_run_admission"]),
            training_sequence=FrozenTaskSequenceIdentity.from_value(
                normalized["training_sequence"]
            ),
            catalog_freeze_hash=normalized["catalog_freeze_hash"],
            progress_anchor_cycle_ordinals=tuple(raw_anchor_cycles),
            sampling_schedule_algorithm=normalized["sampling_schedule_algorithm"],
            sampling_schedule_hash=normalized["sampling_schedule_hash"],
            format=normalized["format"],
        )


__all__ = [
    "FORMAL_EXECUTION_FREEZE_FORMAT",
    "IMPLEMENTATION_BUILD_IDENTITY_FORMAT",
    "FormalExecutionFreeze",
    "ImplementationBuildIdentity",
    "formal_application_config_value",
]
