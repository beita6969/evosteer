"""Path-free scientific identity gates for seven-arm comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass

from skillev.contracts import JsonValue, stable_hash
from skillev.runtime import AttemptBuilderKind, EventEnvelope, PublishedSuccessfulAttemptBundle

from .attempt_identity import AttemptPurpose, PublishedAttemptIdentity
from .execution_hardware import (
    ExecutionHardwareIdentity,
    require_formal_execution_hardware_attestation,
)
from .formal_run_manifest import FormalRunLedger
from .protocol import AblationArm


@dataclass(frozen=True, slots=True)
class CrossArmScientificIdentity:
    """Every scientific control shared by the declared seven-arm comparison.

    Artifact directories, attempt IDs, event-log namespaces, and execution
    labels are intentionally absent.  Each arm protocol is the one permitted
    substitution and is checked separately by :func:`require_cross_arm_comparable`.
    """

    protocol_hash: str
    protocol_freeze_id: str
    ordered_task_sequence_hash: str
    formal_training_binding_hash: str
    run_plan_hash: str
    initial_trainable_state_hash: str
    initial_library_version: str
    initial_skill_library_state_hash: str
    attempt_budget_hash: str
    phi_per_cycle_maximum_hash: str
    authoring_authority_hash: str
    method_config_hash: str
    rollout_config_hash: str
    optimizer_config_hash: str
    execution_config_hash: str
    diagnostics_config_hash: str
    calibration_config_hash: str
    evolution_config_hash: str
    authoring_sampling_hash: str
    maximum_h0_tokens: int
    formal_execution_hash: str | None
    base_model_artifact_hash: str | None
    tokenizer_artifact_hash: str | None
    implementation_build_hash: str | None

    def __post_init__(self) -> None:
        for name in (
            "protocol_hash",
            "protocol_freeze_id",
            "ordered_task_sequence_hash",
            "formal_training_binding_hash",
            "run_plan_hash",
            "initial_trainable_state_hash",
            "initial_library_version",
            "initial_skill_library_state_hash",
            "attempt_budget_hash",
            "phi_per_cycle_maximum_hash",
            "authoring_authority_hash",
            "method_config_hash",
            "rollout_config_hash",
            "optimizer_config_hash",
            "execution_config_hash",
            "diagnostics_config_hash",
            "calibration_config_hash",
            "evolution_config_hash",
            "authoring_sampling_hash",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        for name in (
            "formal_execution_hash",
            "base_model_artifact_hash",
            "tokenizer_artifact_hash",
            "implementation_build_hash",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"{name} must be non-empty text or None")
        if type(self.maximum_h0_tokens) is not int or self.maximum_h0_tokens < 1:
            raise ValueError("maximum_h0_tokens must be positive")

    @classmethod
    def from_public_identity(cls, identity: PublishedAttemptIdentity) -> CrossArmScientificIdentity:
        if not isinstance(identity, PublishedAttemptIdentity):
            raise TypeError("cross-arm identity requires PublishedAttemptIdentity")
        config = identity.application_config
        # ``experiment_id`` is an artifact namespace, not a control.  Batch
        # population and checkpoint cadence remain explicit controls.
        execution = {
            "batch_size": config.trainer.execution.batch_size,
            "checkpoint_cadence": config.trainer.checkpoint.to_value(),
        }
        formal_training_binding_hash = stable_hash(
            {
                "purpose": identity.purpose.value,
                "formal_training_binding": (
                    identity.formal_training_binding.to_value()
                    if identity.formal_training_binding is not None
                    else None
                ),
            }
        )
        return cls(
            protocol_hash=identity.protocol_hash,
            protocol_freeze_id=identity.protocol_freeze_id,
            ordered_task_sequence_hash=identity.ordered_task_sequence_hash,
            formal_training_binding_hash=formal_training_binding_hash,
            run_plan_hash=identity.run_plan.content_hash,
            initial_trainable_state_hash=identity.initial_trainable_state.content_hash,
            initial_library_version=identity.initial_library_version,
            initial_skill_library_state_hash=identity.initial_skill_library_state_hash,
            attempt_budget_hash=identity.attempt_budget_hash,
            phi_per_cycle_maximum_hash=identity.phi_per_cycle_budget_hash,
            authoring_authority_hash=identity.authoring_authority_hash,
            method_config_hash=stable_hash(config.trainer.method.to_value()),
            rollout_config_hash=stable_hash(config.trainer.rollout.to_value()),
            optimizer_config_hash=stable_hash(config.trainer.optimizer.to_value()),
            execution_config_hash=stable_hash(execution),
            diagnostics_config_hash=stable_hash(config.diagnostics.to_value()),
            calibration_config_hash=stable_hash(config.calibration.to_value()),
            evolution_config_hash=stable_hash(config.evolution.to_value()),
            authoring_sampling_hash=stable_hash(config.authoring_sampling.to_value()),
            maximum_h0_tokens=config.maximum_h0_tokens,
            formal_execution_hash=identity.formal_execution_hash,
            base_model_artifact_hash=(
                identity.base_model_artifact.content_hash
                if identity.base_model_artifact is not None
                else None
            ),
            tokenizer_artifact_hash=(
                identity.tokenizer_artifact.content_hash
                if identity.tokenizer_artifact is not None
                else None
            ),
            implementation_build_hash=(
                identity.implementation_build.content_hash
                if identity.implementation_build is not None
                else None
            ),
        )

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_budget_hash": self.attempt_budget_hash,
            "authoring_authority_hash": self.authoring_authority_hash,
            "authoring_sampling_hash": self.authoring_sampling_hash,
            "base_model_artifact_hash": self.base_model_artifact_hash,
            "calibration_config_hash": self.calibration_config_hash,
            "diagnostics_config_hash": self.diagnostics_config_hash,
            "evolution_config_hash": self.evolution_config_hash,
            "execution_config_hash": self.execution_config_hash,
            "formal_execution_hash": self.formal_execution_hash,
            "formal_training_binding_hash": self.formal_training_binding_hash,
            "initial_library_version": self.initial_library_version,
            "initial_skill_library_state_hash": self.initial_skill_library_state_hash,
            "initial_trainable_state_hash": self.initial_trainable_state_hash,
            "implementation_build_hash": self.implementation_build_hash,
            "maximum_h0_tokens": self.maximum_h0_tokens,
            "method_config_hash": self.method_config_hash,
            "optimizer_config_hash": self.optimizer_config_hash,
            "ordered_task_sequence_hash": self.ordered_task_sequence_hash,
            "phi_per_cycle_maximum_hash": self.phi_per_cycle_maximum_hash,
            "protocol_freeze_id": self.protocol_freeze_id,
            "protocol_hash": self.protocol_hash,
            "rollout_config_hash": self.rollout_config_hash,
            "run_plan_hash": self.run_plan_hash,
            "tokenizer_artifact_hash": self.tokenizer_artifact_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(self.to_value())


# Retained as aliases for existing private-input producers; they refer to the
# one path-free representation rather than preserving an old, weaker wire.
CrossArmComparableIdentity = CrossArmScientificIdentity
CrossArmExecutionIdentity = CrossArmScientificIdentity


def _require_cross_arm_controls(
    identities: tuple[PublishedAttemptIdentity, ...],
) -> CrossArmScientificIdentity:
    """Require seven ordered arms with identical non-ablation controls."""

    if not isinstance(identities, tuple) or len(identities) != len(AblationArm):
        raise ValueError("aggregate requires the exact seven formal arms")
    if any(not isinstance(identity, PublishedAttemptIdentity) for identity in identities):
        raise TypeError("aggregate identities must be published attempt identities")
    if tuple(identity.arm for identity in identities) != tuple(AblationArm):
        raise ValueError("aggregate requires exact seven-arm order")
    purposes = {identity.purpose for identity in identities}
    if len(purposes) != 1:
        raise ValueError("aggregate cannot mix attempt purposes")
    common = tuple(CrossArmScientificIdentity.from_public_identity(item) for item in identities)
    if any(item != common[0] for item in common[1:]):
        raise ValueError("arm results have different scientific identities")
    return common[0]


def require_cross_arm_comparable(
    identities: tuple[PublishedAttemptIdentity, ...],
) -> CrossArmScientificIdentity:
    """Compare correctness-fixture arms without admitting formal aggregation.

    Formal benchmark attempts need their write-once manifest and terminal
    ledger.  Keeping that proof out of this lightweight helper would recreate
    the exact rerun-until-lucky gap found in the audit.
    """

    common = _require_cross_arm_controls(identities)
    if identities[0].purpose is AttemptPurpose.FORMAL_BENCHMARK_TRAINING:
        raise ValueError("formal aggregate requires a formal run ledger")
    return common


def require_formal_cross_arm_comparable(
    identities: tuple[PublishedAttemptIdentity, ...],
    *,
    formal_run_ledger: FormalRunLedger,
    bundles: tuple[PublishedSuccessfulAttemptBundle, ...],
) -> CrossArmScientificIdentity:
    """Accept only the frozen formal manifest's seven terminal successes."""

    common = _require_cross_arm_controls(identities)
    if identities[0].purpose is not AttemptPurpose.FORMAL_BENCHMARK_TRAINING:
        raise ValueError("formal aggregate requires formal-benchmark-training attempts")
    if any(
        getattr(common, field) is None
        for field in (
            "formal_execution_hash",
            "base_model_artifact_hash",
            "tokenizer_artifact_hash",
            "implementation_build_hash",
        )
    ):
        raise ValueError("formal aggregate requires complete frozen execution artifacts")
    if not isinstance(formal_run_ledger, FormalRunLedger):
        raise TypeError("formal aggregate requires FormalRunLedger")
    if formal_run_ledger.manifest.protocol_hash != common.protocol_hash:
        raise ValueError("formal run manifest belongs to another protocol")
    if formal_run_ledger.manifest.protocol_freeze_id != common.protocol_freeze_id:
        raise ValueError("formal run manifest belongs to another protocol freeze")

    selected = formal_run_ledger.require_complete_successes(bundles)
    identities_by_kind = {identity.builder_kind: identity for identity in identities}
    if len(identities_by_kind) != len(identities):  # protected by the arm gate
        raise RuntimeError("formal cross-arm identities repeat a builder kind")
    for bundle in selected:
        identity = identities_by_kind.get(bundle.builder_kind)
        if identity is None:  # protected by the closed manifest and arm gate
            raise RuntimeError("formal run manifest has an unknown builder kind")
        if (
            bundle.exact_input_sha256 != identity.exact_input_sha256
            or bundle.public_identity_content_hash != identity.content_hash
        ):
            raise ValueError("formal aggregate bundle differs from its public identity")
    _require_same_formal_execution_hardware(
        identities_by_kind=identities_by_kind,
        bundles=selected,
    )
    return common


def _require_same_formal_execution_hardware(
    *,
    identities_by_kind: dict[AttemptBuilderKind, PublishedAttemptIdentity],
    bundles: tuple[PublishedSuccessfulAttemptBundle, ...],
) -> ExecutionHardwareIdentity:
    """Require the seven selected source logs to report one CUDA identity."""

    measured: list[ExecutionHardwareIdentity] = []
    for bundle in bundles:
        exact = PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
        identity = identities_by_kind.get(exact.builder_kind)
        if identity is None:  # protected by the closed manifest and arm gate
            raise RuntimeError("formal hardware source has an unknown builder kind")
        envelopes = tuple(
            EventEnvelope.from_value(json.loads(line))
            for line in exact.event_log_path.read_text(encoding="utf-8").splitlines()
        )
        attestation = require_formal_execution_hardware_attestation(
            envelopes,
            identity=identity,
            attempt_id=exact.attempt_id,
        )
        if attestation is None:  # protected by the formal aggregate gate
            raise RuntimeError("formal aggregate lost its execution hardware attestation")
        measured.append(attestation.measured_hardware)
    if not measured:  # protected by the exact seven-arm gate
        raise RuntimeError("formal aggregate has no selected hardware identities")
    if any(item != measured[0] for item in measured[1:]):
        raise ValueError("formal aggregate mixes execution hardware/runtime identities")
    return measured[0]


__all__ = [
    "CrossArmComparableIdentity",
    "CrossArmExecutionIdentity",
    "CrossArmScientificIdentity",
    "require_cross_arm_comparable",
    "require_formal_cross_arm_comparable",
]
