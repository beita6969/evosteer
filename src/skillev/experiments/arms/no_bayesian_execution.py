"""All-or-nothing executor for the no-Bayesian experiment arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from skillev.contracts import (
    GenerateEvidence,
    JsonValue,
    PhaseTransitionEvent,
    canonical_json,
    stable_hash,
)
from skillev.evolution.authoring import (
    AuthoredSkillDraft,
    AuthoringRequest,
    GenerateAuthoringRequest,
    GenerateAuthoritySelection,
    RetainAuthoringRequest,
    SkillAuthor,
    SkillAuthoringAuthority,
    validate_authoring_result,
)
from skillev.runtime import (
    BudgetVector,
    SkillDocument,
    SkillLibrary,
    SkillManifest,
    skill_library_version,
)

from .no_bayesian_contracts import (
    FlowOnlyActionResult,
    FlowOnlyDecision,
    FlowOnlyGenerateProposal,
    FlowOnlyProposal,
    FlowOnlyRetainProposal,
    flow_only_proposal_content_hash,
)


class FlowOnlyLibraryView(Protocol):
    def document(self, skill_id: str) -> SkillDocument: ...


@dataclass(frozen=True, slots=True)
class FlowOnlyMutation:
    library_version_before: str
    library_version_after: str
    new_documents: tuple[SkillDocument, ...]
    active_skill_ids_after: tuple[str, ...]
    phase_event_id: str
    optimizer_step: int
    actions: tuple[FlowOnlyActionResult, ...]


def flow_only_authoring_request_for_proposal(
    proposal: FlowOnlyProposal,
    *,
    phase_event: PhaseTransitionEvent,
    library: FlowOnlyLibraryView,
    authority: SkillAuthoringAuthority,
    proposal_index: int,
    base_seed: int,
    cycle_ordinal: int,
) -> AuthoringRequest:
    """Reconstruct the one sealed authoring request for an arm proposal.

    The live executor and the offline audit use this exact mapping.  It is
    deliberately explicit rather than deriving a request from a produced
    document, because the request is the provenance of the corresponding
    ledger reservation.
    """

    seed = flow_only_authoring_seed(
        base_seed=base_seed,
        cycle_ordinal=cycle_ordinal,
        proposal_content_hash=flow_only_proposal_content_hash(proposal),
        proposal_index=proposal_index,
    )
    match proposal:
        case FlowOnlyRetainProposal():
            return RetainAuthoringRequest(
                source=library.document(proposal.target_skill_id),
                edge_exemplars=proposal.edge_exemplars,
                evidence_summary=flow_only_evidence_summary(proposal),
                seed=seed,
            )
        case FlowOnlyGenerateProposal():
            return GenerateAuthoringRequest(
                edge_exemplars=proposal.edge_exemplars,
                evidence_summary=flow_only_evidence_summary(proposal),
                authority=authority.select(proposal.edge_exemplars),
                evidence=GenerateEvidence(
                    importance_edge_ids=proposal.evidence.importance_edge_ids,
                    minimum_absolute_log_importance=(
                        proposal.evidence.minimum_absolute_log_importance
                    ),
                    importance_quantile=proposal.evidence.importance_quantile,
                    importance_semantics=proposal.evidence.importance_semantics,
                ),
                seed=seed,
            )
        case _:
            raise TypeError(f"unsupported flow-only proposal: {type(proposal).__name__}")


def required_flow_only_phi_budget(
    decision: FlowOnlyDecision,
    *,
    max_authoring_completion_tokens: int,
    max_authoring_prompt_tokens: int,
) -> BudgetVector:
    """Return the exact authoring envelope for one flow-only decision."""

    if not isinstance(decision, FlowOnlyDecision):
        raise TypeError("decision must be FlowOnlyDecision")
    if type(max_authoring_completion_tokens) is not int or max_authoring_completion_tokens < 1:
        raise ValueError("max_authoring_completion_tokens must be positive")
    model_calls = len(decision.proposals)
    return BudgetVector(
        model_calls=model_calls,
        input_tokens=model_calls * max_authoring_prompt_tokens,
        output_tokens=model_calls * max_authoring_completion_tokens,
    )


def build_flow_only_mutation(
    decision: FlowOnlyDecision,
    *,
    author: SkillAuthor,
    authority: SkillAuthoringAuthority,
    library: SkillLibrary,
    phase_event: PhaseTransitionEvent,
    optimizer_step: int,
    max_skill_instruction_tokens_per_draft: int,
    base_seed: int,
    cycle_ordinal: int,
) -> FlowOnlyMutation:
    if decision.phase_event_id != phase_event.event_id:
        raise ValueError("flow-only decision targets another phase")
    if phase_event.library_version != library.current_version:
        raise ValueError("flow-only phase targets another library version")
    known = {document.manifest.skill_id: document for document in library.all_documents()}
    active = set(library.active_skill_ids)
    prepared: list[tuple[int, FlowOnlyProposal, SkillDocument]] = []

    for index, proposal in enumerate(decision.proposals):
        request = flow_only_authoring_request_for_proposal(
            proposal,
            phase_event=phase_event,
            library=library,
            authority=authority,
            proposal_index=index,
            base_seed=base_seed,
            cycle_ordinal=cycle_ordinal,
        )
        match proposal:
            case FlowOnlyRetainProposal():
                if proposal.target_skill_id not in active:
                    raise ValueError("flow-only Retain targets an inactive skill")
                source = library.document(proposal.target_skill_id)
                if not isinstance(request, RetainAuthoringRequest):
                    raise TypeError("flow-only Retain built another authoring request")
                authoring_result = author.author(request)
                validate_authoring_result(
                    request,
                    authoring_result,
                    tokenizer=author.tokenizer,
                    max_skill_instruction_tokens_per_draft=(max_skill_instruction_tokens_per_draft),
                )
                draft = _single(authoring_result.drafts, action="Retain")
                document = flow_only_source_document_for_draft(
                    draft,
                    source=source,
                    phase_event=phase_event,
                    proposal_index=index,
                )
                active.remove(proposal.target_skill_id)
            case FlowOnlyGenerateProposal():
                if not isinstance(request, GenerateAuthoringRequest):
                    raise TypeError("flow-only Generate built another authoring request")
                authoring_result = author.author(request)
                validate_authoring_result(
                    request,
                    authoring_result,
                    tokenizer=author.tokenizer,
                    max_skill_instruction_tokens_per_draft=(max_skill_instruction_tokens_per_draft),
                )
                draft = _single(authoring_result.drafts, action="Generate")
                document = flow_only_generated_document_for_draft(
                    draft,
                    authority=request.authority,
                    phase_event=phase_event,
                    proposal_index=index,
                )
            case _:
                raise TypeError(f"unsupported flow-only proposal: {type(proposal).__name__}")
        skill_id = document.manifest.skill_id
        if skill_id in known or any(item[2].manifest.skill_id == skill_id for item in prepared):
            raise ValueError("flow-only authoring must produce a new skill ID")
        active.add(skill_id)
        prepared.append((index, proposal, document))

    new_documents = tuple(sorted((item[2] for item in prepared), key=_document_id))
    all_documents = {
        **known,
        **{document.manifest.skill_id: document for document in new_documents},
    }
    active_after = tuple(sorted(active))
    before = library.current_version
    after = skill_library_version(documents=all_documents, active_skill_ids=active_after)
    actions = tuple(
        _action_result(
            index=index,
            proposal=proposal,
            produced_skill_id=document.manifest.skill_id,
            phase_event_id=decision.phase_event_id,
        )
        for index, proposal, document in prepared
    )
    return FlowOnlyMutation(
        library_version_before=before,
        library_version_after=after,
        new_documents=new_documents,
        active_skill_ids_after=active_after,
        phase_event_id=decision.phase_event_id,
        optimizer_step=optimizer_step,
        actions=actions,
    )


def _action_result(
    *,
    index: int,
    proposal: FlowOnlyProposal,
    produced_skill_id: str,
    phase_event_id: str,
) -> FlowOnlyActionResult:
    targets: tuple[str, ...]
    match proposal:
        case FlowOnlyRetainProposal():
            kind = "retain-compress"
            targets = (proposal.target_skill_id,)
        case FlowOnlyGenerateProposal():
            kind = "generate"
            targets = ()
        case _:
            raise TypeError(f"unsupported flow-only proposal: {type(proposal).__name__}")
    return FlowOnlyActionResult(
        action_id=stable_hash(
            {
                "arm": "no-bayesian-calibration",
                "kind": kind,
                "phase_event_id": phase_event_id,
                "proposal_index": index,
                "produced_skill_id": produced_skill_id,
                "targets": list(targets),
            }
        ),
        proposal_content_hash=flow_only_proposal_content_hash(proposal),
        action_kind=kind,
        target_skill_ids=targets,
        produced_skill_ids=(produced_skill_id,),
        evidence=proposal.evidence,
        rationale_text=proposal.rationale_text,
    )


def flow_only_source_document_for_draft(
    draft: AuthoredSkillDraft,
    *,
    source: SkillDocument,
    phase_event: PhaseTransitionEvent,
    proposal_index: int,
) -> SkillDocument:
    return _document(
        draft,
        input_schema_id=source.manifest.input_schema_id,
        output_schema_id=source.manifest.output_schema_id,
        license_id=source.manifest.license_id,
        provenance={
            "action_type": "flow-only-retain-compress",
            "arm": "no-bayesian-calibration",
            "phase_event_id": phase_event.event_id,
            "proposal_index": proposal_index,
            "source_skill_id": source.manifest.skill_id,
        },
    )


def flow_only_generated_document_for_draft(
    draft: AuthoredSkillDraft,
    *,
    authority: GenerateAuthoritySelection,
    phase_event: PhaseTransitionEvent,
    proposal_index: int,
) -> SkillDocument:
    return _document(
        draft,
        input_schema_id=authority.input_schema_id,
        output_schema_id=authority.output_schema_id,
        license_id=authority.license_id,
        provenance={
            "action_type": "flow-only-generate",
            "arm": "no-bayesian-calibration",
            "authority_context": authority.context,
            "authority_task_family": authority.task_family,
            "phase_event_id": phase_event.event_id,
            "proposal_index": proposal_index,
        },
    )


def _document(
    draft: AuthoredSkillDraft,
    *,
    input_schema_id: str,
    output_schema_id: str,
    license_id: str,
    provenance: dict[str, JsonValue],
) -> SkillDocument:
    content_hash = stable_hash(draft.to_value())
    return SkillDocument(
        manifest=SkillManifest(
            skill_id=f"skill-{content_hash.removeprefix('sha256:')[:12]}",
            version="1",
            content_hash=content_hash,
            input_schema_id=input_schema_id,
            output_schema_id=output_schema_id,
            license_id=license_id,
            provenance_hash=stable_hash(provenance),
        ),
        title=draft.title,
        summary=draft.summary,
        instructions=draft.instructions,
        applicability=draft.applicability,
        requirements=draft.requirements,
    )


def flow_only_evidence_summary(proposal: FlowOnlyProposal) -> str:
    targets: tuple[str, ...]
    match proposal:
        case FlowOnlyRetainProposal():
            kind = "retain-compress"
            targets = (proposal.target_skill_id,)
        case FlowOnlyGenerateProposal():
            kind = "generate"
            targets = ()
        case _:
            raise TypeError(f"unsupported flow-only proposal: {type(proposal).__name__}")
    return canonical_json(
        {
            "action_type": kind,
            "arm": "no-bayesian-calibration",
            "evidence": proposal.evidence.to_value(),
            "rationale": proposal.rationale_text,
            "targets": list(targets),
        }
    )


def flow_only_authoring_seed(
    *,
    base_seed: int,
    cycle_ordinal: int,
    proposal_content_hash: str,
    proposal_index: int,
) -> int:
    digest = stable_hash(
        {
            "base_seed": base_seed,
            "cycle_ordinal": cycle_ordinal,
            "operation": "skill-authoring",
            "proposal_content_hash": proposal_content_hash,
            "proposal_index": proposal_index,
        }
    ).removeprefix("sha256:")
    return int(digest[:16], 16)


def _single(values: tuple[AuthoredSkillDraft, ...], *, action: str) -> AuthoredSkillDraft:
    if len(values) != 1:
        raise ValueError(f"{action} requires exactly one draft")
    return values[0]


def _document_id(document: SkillDocument) -> str:
    return document.manifest.skill_id


__all__ = [
    "FlowOnlyMutation",
    "build_flow_only_mutation",
    "flow_only_authoring_request_for_proposal",
    "flow_only_authoring_seed",
    "flow_only_evidence_summary",
    "flow_only_generated_document_for_draft",
    "flow_only_source_document_for_draft",
    "required_flow_only_phi_budget",
]
