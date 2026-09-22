"""Build one complete, immutable Phi mutation with no side effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from skillev.contracts import (
    EvolutionActionRecord,
    GenerateActionRecord,
    JsonValue,
    PhaseTransitionEvent,
    PruneActionRecord,
    RefineActionRecord,
    RetainCompressActionRecord,
    SplitActionRecord,
    canonical_json,
    stable_hash,
)
from skillev.runtime import SkillDocument, SkillManifest, skill_library_version

from .authoring import (
    AuthoredSkillDraft,
    GenerateAuthoringRequest,
    GenerateAuthoritySelection,
    RefineAuthoringRequest,
    RetainAuthoringRequest,
    SkillAuthor,
    SkillAuthoringAuthority,
    SplitAuthoringRequest,
    retain_compression_budget,
    validate_authoring_result,
)
from .config import EvolutionConfig
from .decision import (
    EvolutionDecision,
    FullActionProposal,
    GenerateProposal,
    PruneProposal,
    RefineProposal,
    RetainProposal,
    SplitProposal,
    proposal_content_hash,
)


class EvolutionLibraryView(Protocol):
    @property
    def current_version(self) -> str: ...

    @property
    def active_skill_ids(self) -> tuple[str, ...]: ...

    def document(self, skill_id: str) -> SkillDocument: ...

    def all_documents(self) -> tuple[SkillDocument, ...]: ...


@dataclass(frozen=True, slots=True)
class EvolutionMutation:
    library_version_before: str
    library_version_after: str
    new_documents: tuple[SkillDocument, ...]
    active_skill_ids_after: tuple[str, ...]
    actions: tuple[EvolutionActionRecord, ...]

    def __post_init__(self) -> None:
        if self.library_version_before == self.library_version_after:
            raise ValueError("evolution mutation must change the library version")
        document_ids = tuple(document.manifest.skill_id for document in self.new_documents)
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("evolution mutation repeats a new document")
        if tuple(sorted(set(self.active_skill_ids_after))) != (self.active_skill_ids_after):
            raise ValueError("active_skill_ids_after must be sorted and unique")
        produced_ids = {
            skill_id for action in self.actions for skill_id in action.produced_skill_ids
        }
        if produced_ids != set(document_ids):
            raise ValueError("mutation documents differ from action products")
        if not self.actions:
            raise ValueError("evolution mutation requires actions")


@dataclass(frozen=True, slots=True)
class _AuthoredProposal:
    index: int
    proposal: FullActionProposal
    documents: tuple[SkillDocument, ...]

    @property
    def produced_skill_ids(self) -> tuple[str, ...]:
        return tuple(document.manifest.skill_id for document in self.documents)


def build_evolution_mutation(
    decision: EvolutionDecision,
    *,
    author: SkillAuthor,
    library: EvolutionLibraryView,
    phase_event: PhaseTransitionEvent,
    config: EvolutionConfig,
    authority: SkillAuthoringAuthority,
    base_seed: int,
    cycle_ordinal: int,
) -> EvolutionMutation:
    """Author every proposal once; any failure terminates the child attempt."""

    if decision.phase_event_id != phase_event.event_id:
        raise ValueError("decision targets another phase event")
    if phase_event.library_version != library.current_version:
        raise ValueError("phase event targets another library version")
    active = set(library.active_skill_ids)
    known = {document.manifest.skill_id: document for document in library.all_documents()}
    authored: list[_AuthoredProposal] = []

    # This is a method-level executability check, not a response fallback.  Run
    # it for the complete decision before the first frozen-base call so an
    # irreducible Retain cannot leave a partially authored Phi attempt.
    for proposal in decision.proposals:
        if isinstance(proposal, RetainProposal):
            retain_compression_budget(
                library.document(proposal.target_skill_id),
                tokenizer=author.tokenizer,
            )

    for index, proposal in enumerate(decision.proposals):
        target_skill_id = (
            None if isinstance(proposal, GenerateProposal) else proposal.target_skill_id
        )
        if target_skill_id is not None and target_skill_id not in active:
            raise ValueError("proposal targets an inactive skill")
        if isinstance(proposal, PruneProposal):
            active.remove(proposal.target_skill_id)
            authored.append(_AuthoredProposal(index, proposal, ()))
            continue

        documents = _author_proposal(
            proposal,
            index=index,
            phase_event=phase_event,
            author=author,
            library=library,
            config=config,
            authority=authority,
            base_seed=base_seed,
            cycle_ordinal=cycle_ordinal,
        )
        produced = tuple(document.manifest.skill_id for document in documents)
        if len(set(produced)) != len(produced):
            raise ValueError("one proposal produced duplicate skill IDs")
        if set(produced) & set(known):
            raise ValueError("authoring must produce previously unseen skill IDs")
        if set(produced) & {
            document.manifest.skill_id for item in authored for document in item.documents
        }:
            raise ValueError("two proposals produced the same skill ID")
        if target_skill_id is not None:
            active.remove(target_skill_id)
        active.update(produced)
        authored.append(_AuthoredProposal(index, proposal, documents))

    new_documents = tuple(
        sorted(
            (document for item in authored for document in item.documents),
            key=lambda document: document.manifest.skill_id,
        )
    )
    all_documents = {
        **known,
        **{document.manifest.skill_id: document for document in new_documents},
    }
    active_after = tuple(sorted(active))
    after_version = skill_library_version(
        documents=all_documents,
        active_skill_ids=active_after,
    )
    before_version = library.current_version
    actions = tuple(
        _action_record(
            item,
            phase_event=phase_event,
            before_version=before_version,
            after_version=after_version,
        )
        for item in authored
    )
    if tuple(action.proposal_content_hash for action in actions) != tuple(
        proposal_content_hash(item.proposal) for item in authored
    ):
        raise ValueError("mutation actions differ from authored proposal provenance")
    return EvolutionMutation(
        library_version_before=before_version,
        library_version_after=after_version,
        new_documents=new_documents,
        active_skill_ids_after=active_after,
        actions=actions,
    )


def authoring_seed(
    *,
    base_seed: int,
    cycle_ordinal: int,
    proposal_content_hash: str,
    proposal_index: int,
) -> int:
    """Derive the deterministic base-model sampling seed for one proposal."""

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


def evidence_summary(proposal: FullActionProposal) -> str:
    """Render the exact answer-free evidence summary shared by live and audit."""

    return canonical_json(
        {
            "action_type": _proposal_action_kind(proposal),
            "evidence": proposal.evidence.to_value(),
            "rationale": proposal.rationale_text,
            "target_skill_ids": list(_proposal_target_ids(proposal)),
        }
    )


def _proposal_action_kind(proposal: FullActionProposal) -> str:
    match proposal:
        case RetainProposal():
            return "retain-compress"
        case RefineProposal():
            return "refine"
        case SplitProposal():
            return "split"
        case PruneProposal():
            return "prune"
        case GenerateProposal():
            return "generate"
        case _:
            raise TypeError(f"unsupported full proposal: {type(proposal).__name__}")


def _proposal_target_ids(proposal: FullActionProposal) -> tuple[str, ...]:
    match proposal:
        case RetainProposal() | RefineProposal() | SplitProposal() | PruneProposal():
            return (proposal.target_skill_id,)
        case GenerateProposal():
            return ()
        case _:
            raise TypeError(f"unsupported full proposal: {type(proposal).__name__}")


def authoring_request_for_proposal(
    proposal: FullActionProposal,
    *,
    proposal_index: int,
    phase_event: PhaseTransitionEvent,
    library: EvolutionLibraryView,
    authority: SkillAuthoringAuthority,
    base_seed: int,
    cycle_ordinal: int,
) -> (
    RetainAuthoringRequest
    | RefineAuthoringRequest
    | SplitAuthoringRequest
    | GenerateAuthoringRequest
    | None
):
    """Build the sealed authoring request, or ``None`` for a prune proposal."""

    seed = authoring_seed(
        base_seed=base_seed,
        cycle_ordinal=cycle_ordinal,
        proposal_content_hash=proposal_content_hash(proposal),
        proposal_index=proposal_index,
    )
    summary = evidence_summary(proposal)
    match proposal:
        case RetainProposal():
            return RetainAuthoringRequest(
                source=library.document(proposal.target_skill_id),
                edge_exemplars=proposal.edge_exemplars,
                evidence_summary=summary,
                seed=seed,
            )
        case RefineProposal():
            if not proposal.evidence.target_contexts:
                raise ValueError("Refine authoring requires the selected post-hoc coordinates")
            return RefineAuthoringRequest(
                source=library.document(proposal.target_skill_id),
                edge_exemplars=proposal.edge_exemplars,
                evidence_summary=summary,
                target_context_keys=proposal.evidence.target_context_keys,
                seed=seed,
            )
        case SplitProposal():
            return SplitAuthoringRequest(
                source=library.document(proposal.target_skill_id),
                edge_exemplars=proposal.edge_exemplars,
                evidence_summary=summary,
                modality=proposal.evidence.modality,
                seed=seed,
            )
        case GenerateProposal():
            return GenerateAuthoringRequest(
                edge_exemplars=proposal.edge_exemplars,
                evidence_summary=summary,
                authority=authority.select(proposal.edge_exemplars),
                evidence=proposal.evidence,
                seed=seed,
                related_skills=tuple(
                    library.document(skill_id)
                    for skill_id in library.active_skill_ids
                    if any(
                        edge.task_family in library.document(skill_id).applicability.task_families
                        or library.document(skill_id).applicability.task_families == ("*",)
                        for edge in proposal.edge_exemplars
                    )
                ),
            )
        case PruneProposal():
            return None
        case _:
            raise TypeError(f"unsupported full proposal: {type(proposal).__name__}")


def _author_proposal(
    proposal: FullActionProposal,
    *,
    index: int,
    phase_event: PhaseTransitionEvent,
    author: SkillAuthor,
    library: EvolutionLibraryView,
    config: EvolutionConfig,
    authority: SkillAuthoringAuthority,
    base_seed: int,
    cycle_ordinal: int,
) -> tuple[SkillDocument, ...]:
    request = authoring_request_for_proposal(
        proposal,
        proposal_index=index,
        phase_event=phase_event,
        library=library,
        authority=authority,
        base_seed=base_seed,
        cycle_ordinal=cycle_ordinal,
    )
    if request is None:
        raise TypeError("Prune must not enter SkillAuthor")
    result = author.author(request)
    validate_authoring_result(
        request,
        result,
        tokenizer=author.tokenizer,
        max_skill_instruction_tokens_per_draft=(config.max_skill_instruction_tokens_per_draft),
    )
    match proposal, request:
        case RetainProposal(), RetainAuthoringRequest(source=source):
            return (
                source_document_for_draft(
                    _single(result.drafts, action="Retain"),
                    source=source,
                    action_kind="retain-compress",
                    phase_event=phase_event,
                    output_index=0,
                ),
            )
        case RefineProposal(), RefineAuthoringRequest(source=source):
            return (
                source_document_for_draft(
                    _single(result.drafts, action="Refine"),
                    source=source,
                    action_kind="refine",
                    phase_event=phase_event,
                    output_index=0,
                ),
            )
        case SplitProposal(), SplitAuthoringRequest(source=source):
            if len(result.drafts) != 2:
                raise ValueError("Split authoring must return two drafts")
            return tuple(
                source_document_for_draft(
                    draft,
                    source=source,
                    action_kind="split",
                    phase_event=phase_event,
                    output_index=output_index,
                )
                for output_index, draft in enumerate(result.drafts)
            )
        case GenerateProposal(), GenerateAuthoringRequest(authority=selection):
            return (
                generated_document_for_draft(
                    _single(result.drafts, action="Generate"),
                    authority=selection,
                    phase_event=phase_event,
                ),
            )
        case _:
            raise TypeError("proposal and authoring request differ")


def _action_record(
    prepared: _AuthoredProposal,
    *,
    phase_event: PhaseTransitionEvent,
    before_version: str,
    after_version: str,
) -> EvolutionActionRecord:
    proposal = prepared.proposal
    products = prepared.produced_skill_ids
    targets: tuple[str, ...]
    match proposal:
        case RetainProposal():
            action_kind = "retain-compress"
            targets = (proposal.target_skill_id,)
        case RefineProposal():
            action_kind = "refine"
            targets = (proposal.target_skill_id,)
        case SplitProposal():
            action_kind = "split"
            targets = (proposal.target_skill_id,)
        case PruneProposal():
            action_kind = "prune"
            targets = (proposal.target_skill_id,)
        case GenerateProposal():
            action_kind = "generate"
            targets = ()
        case _:
            raise TypeError(f"unsupported full proposal: {type(proposal).__name__}")
    common = {
        "action_id": stable_hash(
            {
                "action_type": action_kind,
                "phase_event_id": phase_event.event_id,
                "produced_skill_ids": list(products),
                "proposal_index": prepared.index,
                "target_skill_ids": list(targets),
            }
        ),
        "phase_event_id": phase_event.event_id,
        "proposal_content_hash": proposal_content_hash(proposal),
        "library_version_before": before_version,
        "library_version_after": after_version,
        "lineage_ref": phase_event.event_id,
        "rationale_text": proposal.rationale_text,
    }
    match proposal:
        case RetainProposal():
            return RetainCompressActionRecord(
                **common,
                target_skill_id=proposal.target_skill_id,
                produced_skill_id=_single(products, action="Retain"),
                evidence=proposal.evidence,
            )
        case RefineProposal():
            return RefineActionRecord(
                **common,
                target_skill_id=proposal.target_skill_id,
                produced_skill_id=_single(products, action="Refine"),
                evidence=proposal.evidence,
            )
        case SplitProposal():
            if len(products) != 2:
                raise ValueError("Split requires two products")
            return SplitActionRecord(
                **common,
                target_skill_id=proposal.target_skill_id,
                produced_skill_ids=(products[0], products[1]),
                evidence=proposal.evidence,
            )
        case PruneProposal():
            if products:
                raise ValueError("Prune cannot produce skills")
            return PruneActionRecord(
                **common,
                target_skill_id=proposal.target_skill_id,
                evidence=proposal.evidence,
            )
        case GenerateProposal():
            return GenerateActionRecord(
                **common,
                produced_skill_id=_single(products, action="Generate"),
                evidence=proposal.evidence,
            )
        case _:
            raise TypeError(f"unsupported full proposal: {type(proposal).__name__}")


def source_document_for_draft(
    draft: AuthoredSkillDraft,
    *,
    source: SkillDocument,
    action_kind: str,
    phase_event: PhaseTransitionEvent,
    output_index: int,
) -> SkillDocument:
    return _document(
        draft,
        input_schema_id=source.manifest.input_schema_id,
        output_schema_id=source.manifest.output_schema_id,
        license_id=source.manifest.license_id,
        provenance={
            "action_type": action_kind,
            "output_index": output_index,
            "phase_event_id": phase_event.event_id,
            "source_skill_id": source.manifest.skill_id,
        },
    )


def generated_document_for_draft(
    draft: AuthoredSkillDraft,
    *,
    authority: GenerateAuthoritySelection,
    phase_event: PhaseTransitionEvent,
) -> SkillDocument:
    return _document(
        draft,
        input_schema_id=authority.input_schema_id,
        output_schema_id=authority.output_schema_id,
        license_id=authority.license_id,
        provenance={
            "action_type": "generate",
            "authority_context": authority.context,
            "authority_task_family": authority.task_family,
            "phase_event_id": phase_event.event_id,
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
    manifest = SkillManifest(
        skill_id=f"skill-{content_hash.removeprefix('sha256:')[:12]}",
        version="1",
        content_hash=content_hash,
        input_schema_id=input_schema_id,
        output_schema_id=output_schema_id,
        license_id=license_id,
        provenance_hash=stable_hash(provenance),
    )
    return SkillDocument(
        manifest=manifest,
        title=draft.title,
        summary=draft.summary,
        instructions=draft.instructions,
        applicability=draft.applicability,
        requirements=draft.requirements,
    )


def _evidence_summary(proposal: FullActionProposal) -> str:
    return evidence_summary(proposal)


_T = TypeVar("_T")


def _single(values: tuple[_T, ...], *, action: str) -> _T:
    if len(values) != 1:
        raise ValueError(f"{action} requires exactly one value")
    return values[0]


__all__ = [
    "EvolutionLibraryView",
    "EvolutionMutation",
    "authoring_request_for_proposal",
    "authoring_seed",
    "build_evolution_mutation",
    "evidence_summary",
    "generated_document_for_draft",
    "source_document_for_draft",
]
