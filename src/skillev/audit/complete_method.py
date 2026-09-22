"""Bundle-gated audit for a published full-shaped attempt."""

from __future__ import annotations

import json
from dataclasses import dataclass

from skillev.contracts import JsonValue, stable_hash
from skillev.experiments.attempt_identity import PublishedAttemptIdentity
from skillev.experiments.evolution_progress import TrainingEvolutionCounts
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptSucceeded,
    FullAttemptSummary,
    PublishedSuccessfulAttemptBundle,
    attempt_outcome_from_value,
)

from .evolution_decision import verify_cycle_authoring_usage
from .final_state import AuditedFinalTrainingState
from .formal_artifacts import FormalArtifactResolver, require_exact_formal_artifacts
from .formal_build_attestation import require_formal_implementation_build_attestation
from .formal_hardware_attestation import require_formal_execution_hardware_attestation
from .full_recompute import (
    full_shaped_kernel_for_identity,
    recompute_full_timeline,
)
from .published_events import PublishedEventHistory, expected_rollout_reservation_ids
from .published_identity import read_published_attempt_identity
from .source_reducer import AuditEvidenceMismatchError, AuditSourceReducer, CommittedLibrarySegment
from .tokenizer import TokenizerArtifactResolver, require_exact_tokenizer

COMPLETE_METHOD_AUDIT_FORMAT = "skillev-complete-method-audit@3"


@dataclass(frozen=True, slots=True)
class AuditResources:
    tokenizer_resolver: TokenizerArtifactResolver
    formal_artifact_resolver: FormalArtifactResolver | None = None


@dataclass(frozen=True, slots=True)
class CompleteMethodAuditResult:
    attempt_id: str
    builder_kind: AttemptBuilderKind
    public_identity_content_hash: str
    training_step_count: int
    phase_count: int
    cycle_count: int
    final_library_version: str
    diagnostics_content_hashes: tuple[str, ...]
    posterior_batch_content_hashes: tuple[str, ...]
    posterior_cell_content_hashes: tuple[str, ...]
    decision_content_hashes: tuple[str, ...]
    proposal_content_hashes: tuple[str, ...]
    formal_artifacts_verified: bool
    final_training_state: AuditedFinalTrainingState
    format: str = COMPLETE_METHOD_AUDIT_FORMAT

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_id": self.attempt_id,
            "builder_kind": self.builder_kind.value,
            "cycle_count": self.cycle_count,
            "decision_content_hashes": list(self.decision_content_hashes),
            "diagnostics_content_hashes": list(self.diagnostics_content_hashes),
            "final_library_version": self.final_library_version,
            "formal_artifacts_verified": self.formal_artifacts_verified,
            "format": self.format,
            "phase_count": self.phase_count,
            "posterior_batch_content_hashes": list(self.posterior_batch_content_hashes),
            "posterior_cell_content_hashes": list(self.posterior_cell_content_hashes),
            "proposal_content_hashes": list(self.proposal_content_hashes),
            "public_identity_content_hash": self.public_identity_content_hash,
            "training_step_count": self.training_step_count,
        }


def audit_full_shaped_attempt(
    bundle: PublishedSuccessfulAttemptBundle,
    *,
    resources: AuditResources,
) -> CompleteMethodAuditResult:
    """Audit one full-shaped arm with the kernel pinned by its identity."""

    exact = PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
    if exact.builder_kind is AttemptBuilderKind.NO_BAYESIAN:
        raise TypeError("full-shaped audit rejects no-Bayesian bundles")
    identity = read_published_attempt_identity(exact)
    kernel = full_shaped_kernel_for_identity(identity)
    tokenizer = require_exact_tokenizer(resources.tokenizer_resolver, identity.tokenizer_identity)
    formal_artifacts_verified = _verify_formal_artifacts(identity, resources)
    event_history = PublishedEventHistory.read_full_or_operational_log(exact)
    require_formal_implementation_build_attestation(
        event_history.envelopes,
        identity=identity,
        attempt_id=exact.attempt_id,
    )
    require_formal_execution_hardware_attestation(
        event_history.envelopes,
        identity=identity,
        attempt_id=exact.attempt_id,
    )
    segments = AuditSourceReducer(identity).consume_published_bundle(exact)
    timeline = recompute_full_timeline(
        segments,
        identity=identity,
        kernel=kernel,
        tokenizer=tokenizer,
    )
    _verify_cycle_resets_and_continuation(segments)
    settlements = event_history.budget_settlements()
    cycles = tuple(segment.cycle for segment in segments if segment.cycle is not None)
    if len(cycles) != len(timeline.authoring_reservation_ids_by_cycle):
        raise AuditEvidenceMismatchError("recomputed Phi cycles do not align with source cycles")
    authoring_reservation_ids: list[str] = []
    for cycle, expected_reservation_ids in zip(
        cycles,
        timeline.authoring_reservation_ids_by_cycle,
        strict=True,
    ):
        verify_cycle_authoring_usage(
            cycle,
            settlements,
            expected_reservation_ids=expected_reservation_ids,
        )
        authoring_reservation_ids.extend(expected_reservation_ids)
    rollout_reservation_ids = expected_rollout_reservation_ids(
        record
        for segment in segments
        for training_step in segment.training_steps
        for record in training_step.records
    )
    event_history.require_exact_budget_reservations(
        (*rollout_reservation_ids, *authoring_reservation_ids)
    )
    _verify_outcome_summary(exact, _expected_full_summary(identity=identity, segments=segments))
    phase_count = sum(segment.phase is not None for segment in segments)
    cycle_count = sum(segment.cycle is not None for segment in segments)
    proposal_content_hashes = tuple(
        proposal_hash
        for item in timeline.decisions
        for proposal_hash in item.proposal_content_hashes
    )
    evolution_counts = TrainingEvolutionCounts(
        training_step_count=len(timeline.steps),
        phase_count=phase_count,
        cycle_count=cycle_count,
        action_count=len(proposal_content_hashes),
    )
    return CompleteMethodAuditResult(
        attempt_id=exact.attempt_id,
        builder_kind=exact.builder_kind,
        public_identity_content_hash=identity.content_hash,
        training_step_count=evolution_counts.training_step_count,
        phase_count=evolution_counts.phase_count,
        cycle_count=evolution_counts.cycle_count,
        final_library_version=segments[-1].library_version,
        diagnostics_content_hashes=tuple(
            stable_hash(item.diagnostic.to_value()) for item in timeline.steps
        ),
        posterior_batch_content_hashes=tuple(
            commit.posterior_batch.content_hash
            for segment in segments
            for commit in segment.training_steps
        ),
        posterior_cell_content_hashes=tuple(item.content_hash for item in timeline.final_cells),
        decision_content_hashes=tuple(item.content_hash for item in timeline.decisions),
        proposal_content_hashes=proposal_content_hashes,
        formal_artifacts_verified=formal_artifacts_verified,
        final_training_state=AuditedFinalTrainingState(
            library=segments[-1].library_state,
            calibration_cells=timeline.final_cells,
            final_policy_snapshot_id=segments[-1].training_steps[-1].policy_snapshot_after,
            final_optimizer_step=segments[-1].training_steps[-1].optimizer_step,
            evolution_counts=evolution_counts,
        ),
    )


def _verify_formal_artifacts(
    identity: PublishedAttemptIdentity,
    resources: AuditResources,
) -> bool:
    """Optionally measure formal execution bytes without inventing verification.

    Correctness fixtures carry no formal artifact manifest.  Formal attempts
    have one by contract; when the caller supplies a measurement resolver, a
    mismatch is terminal and a successful result explicitly records that the
    actual artifact set was checked.
    """

    if identity.formal_execution is None:
        return False
    if resources.formal_artifact_resolver is None:
        return False
    assert identity.base_model_artifact is not None
    assert identity.tokenizer_artifact is not None
    assert identity.implementation_build is not None
    require_exact_formal_artifacts(
        resources.formal_artifact_resolver,
        base_model=identity.base_model_artifact,
        tokenizer=identity.tokenizer_artifact,
        implementation_build=identity.implementation_build,
    )
    return True


def _expected_full_summary(
    *,
    identity: PublishedAttemptIdentity,
    segments: tuple[CommittedLibrarySegment, ...],
) -> FullAttemptSummary:
    commits = tuple(commit for segment in segments for commit in segment.training_steps)
    cycles = tuple(segment.cycle for segment in segments if segment.cycle is not None)
    if not commits:
        raise AuditEvidenceMismatchError("successful attempt has no training commits")
    final_cursor = commits[-1].run_cursor_after
    return FullAttemptSummary(
        reports=tuple(item.report for item in commits),
        planned_training_steps_this_attempt=(
            identity.run_plan.total_training_steps
            - identity.initial_run_cursor.completed_training_steps
        ),
        completed_training_steps_this_attempt=len(commits),
        actions_committed_this_attempt=sum(len(item.mutation.actions) for item in cycles),
        cycles_committed_this_attempt=len(cycles),
        cycles_committed_in_run=final_cursor.committed_cycles,
        initial_optimizer_step=identity.initial_optimizer_step,
        final_optimizer_step=commits[-1].optimizer_step,
        final_library_version=segments[-1].library_version,
        final_policy_snapshot_id=commits[-1].policy_snapshot_after,
    )


def _verify_outcome_summary(
    exact: PublishedSuccessfulAttemptBundle, expected: FullAttemptSummary
) -> None:
    outcome = attempt_outcome_from_value(json.loads(exact.outcome_path.read_text(encoding="utf-8")))
    if not isinstance(outcome, AttemptSucceeded):
        raise AuditEvidenceMismatchError("successful bundle contains a failed outcome")
    if outcome.summary.to_value() != expected.to_value():
        raise AuditEvidenceMismatchError("AttemptSucceeded summary differs from source truth")


def _verify_cycle_resets_and_continuation(segments: tuple[CommittedLibrarySegment, ...]) -> None:
    for index, segment in enumerate(segments):
        cycle = segment.cycle
        if cycle is None:
            continue
        expected_reset_version = f"z-reset-{cycle.z_reset_seed:016x}@0"
        if cycle.z_version_after_reset != expected_reset_version:
            raise AuditEvidenceMismatchError("cycle Z version differs from reset seed")
        if index + 1 == len(segments):
            raise AuditEvidenceMismatchError("cycle has no following library segment")
        next_segment = segments[index + 1]
        if next_segment.library_version != cycle.library_version_after:
            raise AuditEvidenceMismatchError("next segment does not use the committed library")
        if not next_segment.training_steps:
            raise AuditEvidenceMismatchError("cycle has no following training step")
        z_version = next_segment.training_steps[0].report.z_version
        if z_version.rpartition("@")[0] != expected_reset_version.rpartition("@")[0]:
            raise AuditEvidenceMismatchError("next training step does not descend from reset Z")


__all__ = [
    "COMPLETE_METHOD_AUDIT_FORMAT",
    "AuditResources",
    "CompleteMethodAuditResult",
    "audit_full_shaped_attempt",
]
