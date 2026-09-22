"""Exact Phi decision, authored-document, and authoring-budget audit helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from skillev.contracts import (
    EvolutionActionRecord,
    EvolutionCycleCommitted,
    GenerateActionRecord,
    PhaseTransitionEvent,
    PruneActionRecord,
    RefineActionRecord,
    RetainCompressActionRecord,
    SplitActionRecord,
)
from skillev.evolution import (
    AuthoredSkillDraft,
    AuthoringRequest,
    AuthoringResult,
    EvolutionConfig,
    EvolutionDecision,
    FullActionProposal,
    GenerateAuthoringRequest,
    GenerateProposal,
    PruneProposal,
    RefineAuthoringRequest,
    RefineProposal,
    RetainAuthoringRequest,
    RetainProposal,
    SkillAuthoringAuthority,
    SplitAuthoringRequest,
    SplitProposal,
    authoring_request_for_proposal,
    authoring_reservation_id,
    generated_document_for_draft,
    proposal_content_hash,
    source_document_for_draft,
    validate_authoring_result,
)
from skillev.policy import AuthoringTokenizerProtocol
from skillev.runtime import BudgetVector, SkillDocument, SkillLibraryState

from .source_reducer import AuditEvidenceMismatchError


@dataclass(frozen=True, slots=True)
class ImmutableEvolutionLibraryView:
    """Read-only library view used to rebuild authoring requests exactly."""

    state: SkillLibraryState

    @property
    def current_version(self) -> str:
        return self.state.current_version

    @property
    def active_skill_ids(self) -> tuple[str, ...]:
        return self.state.active_skill_ids

    def document(self, skill_id: str) -> SkillDocument:
        return self.state.documents[skill_id]

    def all_documents(self) -> tuple[SkillDocument, ...]:
        return tuple(self.state.documents[key] for key in sorted(self.state.documents))


def verify_action_matches_proposal(
    action: EvolutionActionRecord,
    proposal: FullActionProposal,
) -> None:
    """Reject any source action that does not exactly represent its decision item."""

    if action.proposal_content_hash != proposal_content_hash(proposal):
        raise AuditEvidenceMismatchError("action proposal hash does not recompute")
    if action.evidence != proposal.evidence:
        raise AuditEvidenceMismatchError("action evidence differs from decision")
    if action.rationale_text != proposal.rationale_text:
        raise AuditEvidenceMismatchError("action rationale differs from decision")
    targets: tuple[str, ...]
    match action, proposal:
        case RetainCompressActionRecord(), RetainProposal():
            targets = (proposal.target_skill_id,)
        case RefineActionRecord(), RefineProposal():
            targets = (proposal.target_skill_id,)
        case SplitActionRecord(), SplitProposal():
            targets = (proposal.target_skill_id,)
        case PruneActionRecord(), PruneProposal():
            targets = (proposal.target_skill_id,)
        case GenerateActionRecord(), GenerateProposal():
            targets = ()
        case _:
            raise AuditEvidenceMismatchError("action type differs from exact proposal")
    if action.target_skill_ids != targets:
        raise AuditEvidenceMismatchError("action targets differ from exact proposal")


def verify_authored_documents(
    *,
    library_before: SkillLibraryState,
    cycle: EvolutionCycleCommitted,
    decision: EvolutionDecision,
    phase_event: PhaseTransitionEvent,
    authority: SkillAuthoringAuthority,
    config: EvolutionConfig,
    tokenizer: AuthoringTokenizerProtocol,
    base_seed: int,
    cycle_ordinal: int,
) -> None:
    """Rebuild sealed requests and check the published products byte-for-byte."""

    documents = tuple(SkillDocument.from_value(item) for item in cycle.mutation.new_documents)
    by_id = {item.manifest.skill_id: item for item in documents}
    if len(by_id) != len(documents):
        raise AuditEvidenceMismatchError("cycle repeats authored documents")
    view = ImmutableEvolutionLibraryView(library_before)
    for index, (action, proposal) in enumerate(
        zip(cycle.mutation.actions, decision.proposals, strict=True)
    ):
        request = authoring_request_for_proposal(
            proposal,
            proposal_index=index,
            phase_event=phase_event,
            library=view,
            authority=authority,
            base_seed=base_seed,
            cycle_ordinal=cycle_ordinal,
        )
        if request is None:
            if action.produced_skill_ids:
                raise AuditEvidenceMismatchError("Prune action produced documents")
            continue
        try:
            products = tuple(by_id[skill_id] for skill_id in action.produced_skill_ids)
        except KeyError as error:
            raise AuditEvidenceMismatchError("action product document is absent") from error
        result = AuthoringResult(tuple(AuthoredSkillDraft.from_document(item) for item in products))
        try:
            validate_authoring_result(
                request,
                result,
                tokenizer=tokenizer,
                max_skill_instruction_tokens_per_draft=(
                    config.max_skill_instruction_tokens_per_draft
                ),
            )
        except (TypeError, ValueError) as error:
            raise AuditEvidenceMismatchError("authored document violates sealed request") from error
        expected = _documents_for_request(
            proposal=proposal,
            request=request,
            phase_event=phase_event,
            products=products,
        )
        if tuple(item.to_value() for item in products) != tuple(
            item.to_value() for item in expected
        ):
            raise AuditEvidenceMismatchError("authored document manifest differs from request")


def _documents_for_request(
    *,
    proposal: FullActionProposal,
    request: AuthoringRequest,
    phase_event: PhaseTransitionEvent,
    products: tuple[SkillDocument, ...],
) -> tuple[SkillDocument, ...]:
    """Reconstruct manifest/provenance from already validated drafts."""

    drafts = tuple(AuthoredSkillDraft.from_document(item) for item in products)
    match proposal, request:
        case RetainProposal(), RetainAuthoringRequest(source=source):
            return (
                source_document_for_draft(
                    drafts[0],
                    source=source,
                    action_kind="retain-compress",
                    phase_event=phase_event,
                    output_index=0,
                ),
            )
        case RefineProposal(), RefineAuthoringRequest(source=source):
            return (
                source_document_for_draft(
                    drafts[0],
                    source=source,
                    action_kind="refine",
                    phase_event=phase_event,
                    output_index=0,
                ),
            )
        case SplitProposal(), SplitAuthoringRequest(source=source):
            return tuple(
                source_document_for_draft(
                    draft,
                    source=source,
                    action_kind="split",
                    phase_event=phase_event,
                    output_index=index,
                )
                for index, draft in enumerate(drafts)
            )
        case GenerateProposal(), GenerateAuthoringRequest(authority=authority):
            return (
                generated_document_for_draft(
                    drafts[0],
                    authority=authority,
                    phase_event=phase_event,
                ),
            )
        case _:
            raise AuditEvidenceMismatchError("proposal and authoring request differ")


def verify_cycle_decision_and_documents(
    *,
    library_before: SkillLibraryState,
    cycle: EvolutionCycleCommitted,
    decision: EvolutionDecision,
    phase_event: PhaseTransitionEvent,
    authority: SkillAuthoringAuthority,
    config: EvolutionConfig,
    tokenizer: AuthoringTokenizerProtocol,
    base_seed: int,
    cycle_ordinal: int,
) -> tuple[str, ...]:
    if cycle.decision_content_hash != decision.content_hash:
        raise AuditEvidenceMismatchError("cycle decision hash does not recompute")
    if cycle.proposal_content_hashes != decision.proposal_content_hashes:
        raise AuditEvidenceMismatchError("cycle proposal hashes do not recompute")
    if len(cycle.mutation.actions) != len(decision.proposals):
        raise AuditEvidenceMismatchError("cycle action count differs from decision")
    for action, proposal in zip(cycle.mutation.actions, decision.proposals, strict=True):
        verify_action_matches_proposal(action, proposal)
    expected_reservation_ids = expected_cycle_authoring_reservation_ids(
        library_before=library_before,
        decision=decision,
        phase_event=phase_event,
        authority=authority,
        base_seed=base_seed,
        cycle_ordinal=cycle_ordinal,
    )
    if cycle.authoring_reservation_ids != expected_reservation_ids:
        raise AuditEvidenceMismatchError("cycle authoring reservations do not recompute")
    verify_authored_documents(
        library_before=library_before,
        cycle=cycle,
        decision=decision,
        phase_event=phase_event,
        authority=authority,
        config=config,
        tokenizer=tokenizer,
        base_seed=base_seed,
        cycle_ordinal=cycle_ordinal,
    )
    return expected_reservation_ids


def expected_cycle_authoring_reservation_ids(
    *,
    library_before: SkillLibraryState,
    decision: EvolutionDecision,
    phase_event: PhaseTransitionEvent,
    authority: SkillAuthoringAuthority,
    base_seed: int,
    cycle_ordinal: int,
) -> tuple[str, ...]:
    """Rebuild the exact ordered authoring reservations for one Phi cycle."""

    view = ImmutableEvolutionLibraryView(library_before)
    return tuple(
        authoring_reservation_id(request)
        for index, proposal in enumerate(decision.proposals)
        if (
            request := authoring_request_for_proposal(
                proposal,
                proposal_index=index,
                phase_event=phase_event,
                library=view,
                authority=authority,
                base_seed=base_seed,
                cycle_ordinal=cycle_ordinal,
            )
        )
        is not None
    )


def verify_cycle_authoring_usage(
    cycle: EvolutionCycleCommitted,
    settlements: Mapping[str, BudgetVector],
    *,
    expected_reservation_ids: tuple[str, ...],
) -> None:
    if cycle.authoring_reservation_ids != expected_reservation_ids:
        raise AuditEvidenceMismatchError("cycle authoring reservation sequence differs")
    actual = BudgetVector()
    for reservation_id in cycle.authoring_reservation_ids:
        try:
            actual = actual.add(settlements[reservation_id])
        except KeyError as error:
            raise AuditEvidenceMismatchError(
                "cycle authoring reservation has no settlement"
            ) from error
    expected = BudgetVector(
        input_tokens=cycle.authoring_usage.input_tokens,
        output_tokens=cycle.authoring_usage.output_tokens,
        model_calls=cycle.authoring_usage.model_calls,
    )
    if actual != expected:
        raise AuditEvidenceMismatchError("cycle authoring usage differs from ledger events")


__all__ = [
    "ImmutableEvolutionLibraryView",
    "expected_cycle_authoring_reservation_ids",
    "verify_action_matches_proposal",
    "verify_authored_documents",
    "verify_cycle_authoring_usage",
    "verify_cycle_decision_and_documents",
]
