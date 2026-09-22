"""Action-specific draft acceptance; no model calls, votes or library mutation."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from skillev.contracts import canonical_json, stable_hash
from skillev.policy import AuthoringTokenizerProtocol
from skillev.rollout import (
    AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP,
    MAXIMUM_APPLICABLE_SKILL_POSITION,
    maximum_retrieved_skill_block_token_count,
)
from skillev.runtime import (
    FullRetrievedSkillContext,
    RetrievalInclusionReason,
    SkillApplicability,
    SkillMetadata,
)
from skillev.runtime.attempt_failures import AttemptDomainError
from skillev.runtime.attempt_protocol import AttemptFailureCode, AttemptFailureStage

from .evidence import is_generate_candidate_action
from .requirement_contract import validate_requirement_revisions

if TYPE_CHECKING:
    from .authoring import (
        AuthoredSkillDraft,
        AuthoringRequest,
        AuthoringResult,
        GenerateAuthoringRequest,
        RefineAuthoringRequest,
        RetainAuthoringRequest,
        SplitAuthoringRequest,
    )


class AuthoringFailedError(AttemptDomainError):
    """The sole frozen-base authoring call failed its sealed contract."""

    def __init__(self, private_detail: str) -> None:
        super().__init__(
            code=AttemptFailureCode.AUTHORING_FAILED,
            stage=AttemptFailureStage.EXECUTION,
            private_detail=private_detail,
        )


def validate_authoring_result(
    request: AuthoringRequest,
    result: AuthoringResult,
    *,
    tokenizer: AuthoringTokenizerProtocol,
    max_skill_instruction_tokens_per_draft: int,
) -> None:
    from .authoring import (
        GenerateAuthoringRequest,
        RefineAuthoringRequest,
        RetainAuthoringRequest,
        SplitAuthoringRequest,
    )

    match request:
        case RetainAuthoringRequest():
            _validate_retain(request, result, tokenizer)
        case RefineAuthoringRequest():
            _validate_refine(request, result)
        case SplitAuthoringRequest():
            _validate_split(request, result)
        case GenerateAuthoringRequest():
            _validate_generate(request, result)
        case _:
            raise TypeError(f"unsupported authoring request: {type(request).__name__}")
    if any(
        len(tokenizer.encode(draft.instructions)) > max_skill_instruction_tokens_per_draft
        for draft in result.drafts
    ):
        raise AuthoringFailedError("authored instructions exceed the configured bound")
    if any(
        maximum_retrieved_skill_block_token_count(
            _authored_draft_retrieval_context(request, draft),
            maximum_position=MAXIMUM_APPLICABLE_SKILL_POSITION,
            tokenizer=tokenizer,
        )
        > AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP
        for draft in result.drafts
    ):
        raise AuthoringFailedError("authored complete retrieved-skill block exceeds its bound")


def _authored_draft_retrieval_context(
    request: AuthoringRequest,
    draft: AuthoredSkillDraft,
) -> FullRetrievedSkillContext:
    """Build the exact future ``H_0`` block identity before library mutation."""

    from .authoring import (
        GenerateAuthoringRequest,
        RefineAuthoringRequest,
        RetainAuthoringRequest,
        SplitAuthoringRequest,
    )

    match request:
        case RetainAuthoringRequest() | RefineAuthoringRequest() | SplitAuthoringRequest():
            input_schema_id = request.source.manifest.input_schema_id
            output_schema_id = request.source.manifest.output_schema_id
            license_id = request.source.manifest.license_id
        case GenerateAuthoringRequest():
            input_schema_id = request.authority.input_schema_id
            output_schema_id = request.authority.output_schema_id
            license_id = request.authority.license_id
        case _:
            raise TypeError(f"unsupported authoring request: {type(request).__name__}")
    from .authoring import template_version_for

    content_hash = stable_hash(draft.to_value())
    return FullRetrievedSkillContext(
        metadata=SkillMetadata(
            skill_id=f"skill-{content_hash.removeprefix('sha256:')[:12]}",
            version="1",
            content_hash=content_hash,
            input_schema_id=input_schema_id,
            output_schema_id=output_schema_id,
            license_id=license_id,
            provenance_hash=stable_hash(
                {
                    "authoring_block_admission": template_version_for(request),
                    "content_hash": content_hash,
                }
            ),
        ),
        content=canonical_json(draft.to_value()),
        inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
    )


def _validate_retain(
    request: RetainAuthoringRequest,
    result: AuthoringResult,
    tokenizer: AuthoringTokenizerProtocol,
) -> None:
    draft = _one(result.drafts)
    if draft.applicability != request.source.applicability:
        raise AuthoringFailedError("compress must preserve applicability")
    if draft.requirements != request.source.requirements:
        raise AuthoringFailedError("compress must preserve complete source requirements")
    if len(tokenizer.encode(canonical_json(draft.to_value()))) >= len(
        tokenizer.encode(canonical_json(request.source.content_value()))
    ):
        raise AuthoringFailedError(
            "compress must strictly reduce complete model-visible skill tokens"
        )


def _validate_refine(
    request: RefineAuthoringRequest,
    result: AuthoringResult,
) -> None:
    draft = _one(result.drafts)
    if draft.applicability != request.source.applicability:
        raise AuthoringFailedError("refine must preserve applicability")
    from .authoring import context_patch_requirement_id

    draft_ids = {item.requirement_id for item in draft.requirements}
    try:
        validate_requirement_revisions(request.source, draft.requirements)
    except ValueError as error:
        raise AuthoringFailedError(str(error)) from error
    required = {context_patch_requirement_id(key) for key in request.target_context_keys}
    if not required <= draft_ids or any(
        item.requirement_id in required and item.immutable for item in draft.requirements
    ):
        raise AuthoringFailedError("refine lacks context patches")


def _validate_split(
    request: SplitAuthoringRequest,
    result: AuthoringResult,
) -> None:
    from .authoring import split_applicabilities, split_task_family_requirement_id

    low, high = _two(result.drafts)
    expected_low, expected_high = split_applicabilities(request)
    if low.applicability != expected_low:
        raise AuthoringFailedError("low Split draft has incorrect applicability")
    if high.applicability != expected_high:
        raise AuthoringFailedError("high Split draft has incorrect applicability")

    for draft, families in (
        (low, request.modality.low_task_families),
        (high, request.modality.high_task_families),
    ):
        try:
            validate_requirement_revisions(request.source, draft.requirements)
        except ValueError as error:
            raise AuthoringFailedError(str(error)) from error
        marker = split_task_family_requirement_id(families)
        if not any(
            item.requirement_id == marker and not item.immutable for item in draft.requirements
        ):
            raise AuthoringFailedError("Split draft lacks its evolvable modality strategy")


def _validate_generate(
    request: GenerateAuthoringRequest,
    result: AuthoringResult,
) -> None:
    draft = _one(result.drafts)
    from .authoring import uncovered_edge_requirement_id

    for exemplar in request.edge_exemplars:
        if exemplar.absolute_log_importance < request.evidence.minimum_absolute_log_importance:
            raise AuthoringFailedError("Generate exemplar is below the absolute importance floor")
        if exemplar.log_importance_quantile < request.evidence.importance_quantile:
            raise AuthoringFailedError("Generate exemplar is outside the selected upper tail")
        if exemplar.invoked_skill_ids:
            raise AuthoringFailedError("Generate exemplar already has skill coverage")
        if not is_generate_candidate_action(exemplar.action_kind):
            raise AuthoringFailedError("Generate exemplar action kind is ineligible")
    expected_applicability = SkillApplicability(
        task_families=(request.authority.task_family,),
        contexts=(request.authority.context,),
        required_tools=request.authority.required_tools,
        excluded_contexts=(),
    )
    if draft.applicability != expected_applicability:
        raise AuthoringFailedError("Generate applicability differs from authority")
    expected_requirement_ids = tuple(
        uncovered_edge_requirement_id(item.edge_id) for item in request.edge_exemplars
    )
    if tuple(item.requirement_id for item in draft.requirements) != expected_requirement_ids:
        raise AuthoringFailedError("Generate requirements differ from uncovered-edge contract")
    if any(item.immutable or item.replaces for item in draft.requirements):
        raise AuthoringFailedError("Generate produces new strategies, not immutable authority")


_T = TypeVar("_T")


def _one(values: tuple[_T, ...]) -> _T:
    if len(values) != 1:
        raise AuthoringFailedError("action requires exactly one draft")
    return values[0]


def _two(values: tuple[_T, ...]) -> tuple[_T, _T]:
    if len(values) != 2:
        raise AuthoringFailedError("action requires exactly two drafts")
    return values
