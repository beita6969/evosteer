"""Bundle-gated audit for the independent no-Bayesian arm."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from skillev.contracts import JsonValue, PhaseTransitionEvent, stable_hash
from skillev.diagnostics import BatchDiagnostics
from skillev.evolution import (
    PhaseTransitionDetected,
    PhaseTransitionDetector,
    TrajectoryEvidenceView,
    WindowFlowView,
    authoring_reservation_id,
)
from skillev.evolution.authoring import (
    AuthoredSkillDraft,
    AuthoringResult,
    GenerateAuthoringRequest,
    RetainAuthoringRequest,
    validate_authoring_result,
)
from skillev.experiments.arms.no_bayesian_calibration import (
    FlowOnlyEvolutionPolicy,
    FlowOnlyProjectionPipeline,
    FlowOnlyProjectionTransition,
    FlowOnlyTrainingSource,
)
from skillev.experiments.arms.no_bayesian_contracts import (
    FlowOnlyCycleResult,
    FlowOnlyDecision,
    FlowOnlyRetainProposal,
    flow_only_proposal_content_hash,
)
from skillev.experiments.arms.no_bayesian_execution import (
    flow_only_authoring_request_for_proposal,
    flow_only_generated_document_for_draft,
    flow_only_source_document_for_draft,
)
from skillev.experiments.attempt_identity import PublishedAttemptIdentity
from skillev.experiments.evolution_progress import TrainingEvolutionCounts
from skillev.policy import AuthoringTokenizerProtocol
from skillev.runtime import (
    AttemptBuilderKind,
    AttemptSucceeded,
    BudgetVector,
    FlowOnlyAttemptSummary,
    PublishedSuccessfulAttemptBundle,
    RunSlotKind,
    SkillDocument,
    SkillLibraryState,
    attempt_outcome_from_value,
)

from .complete_method import AuditResources, _verify_formal_artifacts
from .final_state import AuditedFinalTrainingState
from .formal_build_attestation import require_formal_implementation_build_attestation
from .formal_hardware_attestation import require_formal_execution_hardware_attestation
from .no_bayesian_source_reducer import CommittedFlowOnlySegment, FlowOnlyAuditSourceReducer
from .published_events import PublishedEventHistory, expected_rollout_reservation_ids
from .published_identity import read_published_attempt_identity
from .source_reducer import AuditEvidenceMismatchError
from .tokenizer import require_exact_tokenizer

NO_BAYESIAN_AUDIT_FORMAT = "skillev-no-bayesian-audit@3"


@dataclass(frozen=True, slots=True)
class NoBayesianAuditResult:
    attempt_id: str
    public_identity_content_hash: str
    training_step_count: int
    phase_count: int
    cycle_count: int
    final_library_version: str
    diagnostics_content_hashes: tuple[str, ...]
    decision_content_hashes: tuple[str, ...]
    proposal_content_hashes: tuple[str, ...]
    formal_artifacts_verified: bool
    final_training_state: AuditedFinalTrainingState
    format: str = NO_BAYESIAN_AUDIT_FORMAT

    def to_value(self) -> dict[str, JsonValue]:
        return {
            "attempt_id": self.attempt_id,
            "cycle_count": self.cycle_count,
            "decision_content_hashes": list(self.decision_content_hashes),
            "diagnostics_content_hashes": list(self.diagnostics_content_hashes),
            "final_library_version": self.final_library_version,
            "formal_artifacts_verified": self.formal_artifacts_verified,
            "format": self.format,
            "phase_count": self.phase_count,
            "proposal_content_hashes": list(self.proposal_content_hashes),
            "public_identity_content_hash": self.public_identity_content_hash,
            "training_step_count": self.training_step_count,
        }


def audit_no_bayesian_attempt(
    bundle: PublishedSuccessfulAttemptBundle, *, resources: AuditResources
) -> NoBayesianAuditResult:
    exact = PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
    identity = read_published_attempt_identity(exact)
    if identity.builder_kind is not AttemptBuilderKind.NO_BAYESIAN:
        raise TypeError("no-Bayesian audit accepts only its exact builder")
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
    segments = FlowOnlyAuditSourceReducer(identity).consume_published_bundle(exact)
    settlements = event_history.budget_settlements()
    diagnostics: list[BatchDiagnostics] = []
    decisions: list[FlowOnlyDecision] = []
    authoring_reservation_ids: list[str] = []
    for segment in segments:
        pipeline = FlowOnlyProjectionPipeline.fresh(
            identity.application_config.diagnostics, library_version=segment.library_version
        )
        detector = PhaseTransitionDetector.fresh(
            evolution_config=identity.application_config.evolution,
            diagnostics_config=identity.application_config.diagnostics,
            library_version=segment.library_version,
        )
        by_batch: dict[str, FlowOnlyProjectionTransition] = {}
        detected: list[PhaseTransitionDetected] = []
        for commit in segment.training_steps:
            source = FlowOnlyTrainingSource(
                batch_id=commit.batch_id,
                optimizer_step=commit.optimizer_step,
                records=commit.records,
                stats=commit.stats,
                edge_records=commit.edge_records,
            )
            transition = pipeline.preview(source)
            pipeline.commit(transition)
            diagnostics.append(transition.diagnostic)
            by_batch[commit.batch_id] = transition
            if (
                identity.run_plan.slot_kind(commit.run_cursor_after.completed_training_steps)
                is RunSlotKind.PHASE_SEARCH
            ):
                observed = detector.observe(transition.diagnostic)
                if isinstance(observed, PhaseTransitionDetected):
                    detected.append(observed)
        if segment.phase is None:
            if detected:
                raise AuditEvidenceMismatchError("flow-only detected phase has no source")
            continue
        if (
            len(detected) != 1
            or detected[0].event.content_hash != segment.phase.phase_event.content_hash
        ):
            raise AuditEvidenceMismatchError("flow-only phase source does not recompute")
        phase = detected[0].event
        ids = (*phase.previous_window.member_batch_ids, *phase.current_window.member_batch_ids)
        window = tuple(by_batch[item] for item in ids)
        decision = FlowOnlyEvolutionPolicy().decide_from_views(
            window_flow=WindowFlowView(
                phase_event=phase,
                diagnostics=tuple(item.diagnostic for item in window),
                active_skill_ids=segment.library_state.active_skill_ids,
                applicability_by_skill={
                    skill_id: segment.library_state.documents[skill_id].applicability
                    for skill_id in segment.library_state.active_skill_ids
                },
                task_family_universe=identity.authoring_authority.allowed_task_families,
            ),
            trajectories=TrajectoryEvidenceView(
                authoring_by_edge_id={
                    evidence.edge_id: evidence
                    for item in window
                    for evidence in item.next_state.current_segment_batches[-1].authoring_evidence
                }
            ),
            config=identity.application_config.evolution,
        )
        if segment.cycle is None:
            raise AuditEvidenceMismatchError("flow-only phase has no cycle")
        authoring_reservation_ids.extend(
            _verify_flow_only_cycle(
                segment,
                segment.cycle,
                decision,
                phase,
                identity,
                tokenizer,
                settlements,
            )
        )
        decisions.append(decision)
    rollout_reservation_ids = expected_rollout_reservation_ids(
        record
        for segment in segments
        for training_step in segment.training_steps
        for record in training_step.records
    )
    event_history.require_exact_budget_reservations(
        (*rollout_reservation_ids, *authoring_reservation_ids)
    )
    _verify_summary(exact, identity, segments)
    phase_count = sum(item.phase is not None for item in segments)
    cycle_count = sum(item.cycle is not None for item in segments)
    proposal_content_hashes = tuple(
        hash_ for decision in decisions for hash_ in decision.proposal_content_hashes
    )
    evolution_counts = TrainingEvolutionCounts(
        training_step_count=sum(len(item.training_steps) for item in segments),
        phase_count=phase_count,
        cycle_count=cycle_count,
        action_count=len(proposal_content_hashes),
    )
    return NoBayesianAuditResult(
        attempt_id=exact.attempt_id,
        public_identity_content_hash=identity.content_hash,
        training_step_count=evolution_counts.training_step_count,
        phase_count=evolution_counts.phase_count,
        cycle_count=evolution_counts.cycle_count,
        final_library_version=segments[-1].library_version,
        diagnostics_content_hashes=tuple(stable_hash(item.to_value()) for item in diagnostics),
        decision_content_hashes=tuple(item.content_hash for item in decisions),
        proposal_content_hashes=proposal_content_hashes,
        formal_artifacts_verified=formal_artifacts_verified,
        final_training_state=AuditedFinalTrainingState(
            library=segments[-1].library_state,
            calibration_cells=(),
            final_policy_snapshot_id=segments[-1].training_steps[-1].policy_snapshot_after,
            final_optimizer_step=segments[-1].training_steps[-1].optimizer_step,
            evolution_counts=evolution_counts,
        ),
    )


def _verify_flow_only_cycle(
    segment: CommittedFlowOnlySegment,
    cycle: FlowOnlyCycleResult,
    decision: FlowOnlyDecision,
    phase: PhaseTransitionEvent,
    identity: PublishedAttemptIdentity,
    tokenizer: AuthoringTokenizerProtocol,
    settlements: Mapping[str, BudgetVector],
) -> tuple[str, ...]:
    if (
        cycle.decision_content_hash != decision.content_hash
        or cycle.proposal_content_hashes != decision.proposal_content_hashes
    ):
        raise AuditEvidenceMismatchError("flow-only decision does not recompute")
    if len(cycle.actions) != len(decision.proposals):
        raise AuditEvidenceMismatchError("flow-only action count differs")
    created = {
        SkillDocument.from_value(item).manifest.skill_id: SkillDocument.from_value(item)
        for item in cycle.new_documents
    }
    view = _ImmutableLibrary(segment.library_state)
    actual_usage = BudgetVector()
    expected_reservation_ids: list[str] = []
    for index, (action, proposal) in enumerate(zip(cycle.actions, decision.proposals, strict=True)):
        if (
            action.proposal_content_hash != flow_only_proposal_content_hash(proposal)
            or action.evidence != proposal.evidence
            or action.rationale_text != proposal.rationale_text
        ):
            raise AuditEvidenceMismatchError("flow-only action differs from proposal")
        expected_kind = (
            "retain-compress" if isinstance(proposal, FlowOnlyRetainProposal) else "generate"
        )
        expected_targets = (
            (proposal.target_skill_id,) if isinstance(proposal, FlowOnlyRetainProposal) else ()
        )
        if action.action_kind != expected_kind or action.target_skill_ids != expected_targets:
            raise AuditEvidenceMismatchError("flow-only action type or target differs")
        request = flow_only_authoring_request_for_proposal(
            proposal,
            phase_event=phase,
            library=view,
            authority=identity.authoring_authority,
            proposal_index=index,
            base_seed=identity.application_config.trainer.rollout.base_seed,
            cycle_ordinal=segment.cursor_after.committed_cycles,
        )
        expected_id = authoring_reservation_id(request)
        if cycle.authoring_reservation_ids[index] != expected_id:
            raise AuditEvidenceMismatchError("flow-only authoring reservation differs")
        expected_reservation_ids.append(expected_id)
        try:
            actual_usage = actual_usage.add(settlements[expected_id])
            product = created[action.produced_skill_ids[0]]
        except KeyError as error:
            raise AuditEvidenceMismatchError(
                "flow-only source lacks reservation or product"
            ) from error
        result = AuthoringResult((AuthoredSkillDraft.from_document(product),))
        try:
            validate_authoring_result(
                request,
                result,
                tokenizer=tokenizer,
                max_skill_instruction_tokens_per_draft=(
                    identity.application_config.evolution.max_skill_instruction_tokens_per_draft
                ),
            )
        except (TypeError, ValueError) as error:
            raise AuditEvidenceMismatchError(
                "flow-only authored product violates request"
            ) from error
        draft = AuthoredSkillDraft.from_document(product)
        if isinstance(proposal, FlowOnlyRetainProposal):
            if not isinstance(request, RetainAuthoringRequest):
                raise AuditEvidenceMismatchError("flow-only Retain request type differs")
            expected = flow_only_source_document_for_draft(
                draft, source=request.source, phase_event=phase, proposal_index=index
            )
        else:
            if not isinstance(request, GenerateAuthoringRequest):
                raise AuditEvidenceMismatchError("flow-only Generate request type differs")
            expected = flow_only_generated_document_for_draft(
                draft, authority=request.authority, phase_event=phase, proposal_index=index
            )
        if expected.to_value() != product.to_value():
            raise AuditEvidenceMismatchError("flow-only authored manifest differs")
    expected_usage = BudgetVector(
        cycle.authoring_usage.input_tokens,
        cycle.authoring_usage.output_tokens,
        cycle.authoring_usage.model_calls,
    )
    if actual_usage != expected_usage:
        raise AuditEvidenceMismatchError("flow-only authoring usage differs")
    if tuple(expected_reservation_ids) != cycle.authoring_reservation_ids:
        raise AuditEvidenceMismatchError("flow-only authoring reservation sequence differs")
    return tuple(expected_reservation_ids)


@dataclass(frozen=True, slots=True)
class _ImmutableLibrary:
    state: SkillLibraryState

    @property
    def current_version(self) -> str:
        return self.state.current_version

    def document(self, skill_id: str) -> SkillDocument:
        return self.state.documents[skill_id]

    def all_documents(self) -> tuple[SkillDocument, ...]:
        return tuple(self.state.documents[key] for key in sorted(self.state.documents))


def _verify_summary(
    exact: PublishedSuccessfulAttemptBundle,
    identity: PublishedAttemptIdentity,
    segments: tuple[CommittedFlowOnlySegment, ...],
) -> None:
    commits = tuple(commit for segment in segments for commit in segment.training_steps)
    cycles = tuple(segment.cycle for segment in segments if segment.cycle is not None)
    if not commits:
        raise AuditEvidenceMismatchError("flow-only successful attempt has no training commits")
    final = commits[-1]
    expected = FlowOnlyAttemptSummary(
        reports=tuple(item.report for item in commits),
        planned_training_steps_this_attempt=identity.run_plan.total_training_steps
        - identity.initial_run_cursor.completed_training_steps,
        completed_training_steps_this_attempt=len(commits),
        actions_committed_this_attempt=sum(len(item.actions) for item in cycles),
        cycles_committed_this_attempt=len(cycles),
        cycles_committed_in_run=final.run_cursor_after.committed_cycles,
        initial_optimizer_step=identity.initial_optimizer_step,
        final_optimizer_step=final.optimizer_step,
        final_library_version=segments[-1].library_version,
        final_policy_snapshot_id=final.policy_snapshot_after,
    )
    outcome = attempt_outcome_from_value(json.loads(exact.outcome_path.read_text(encoding="utf-8")))
    if not isinstance(outcome, AttemptSucceeded):
        raise AuditEvidenceMismatchError("successful bundle contains a failed outcome")
    if outcome.summary.to_value() != expected.to_value():
        raise AuditEvidenceMismatchError("flow-only summary differs from source truth")


__all__ = ["NO_BAYESIAN_AUDIT_FORMAT", "NoBayesianAuditResult", "audit_no_bayesian_attempt"]
