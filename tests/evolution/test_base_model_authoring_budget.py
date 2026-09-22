from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path

import pytest

from skillev.contracts import (
    GenerateEvidence,
    PosteriorTaskFamilyMode,
    SplitBranch,
    SplitModalityEvidence,
    SplitTaskFamilyAssignment,
    canonical_json,
    stable_hash,
)
from skillev.evolution import (
    AuthoredSkillDraft,
    AuthoringCallMaximum,
    AuthoringFailedError,
    AuthoringResult,
    AuthoringSamplingConfig,
    BaseModelSkillAuthor,
    EvolutionConfig,
    GenerateAuthoringRequest,
    GenerateAuthoritySelection,
    RefineAuthoringRequest,
    RetainAuthoringRequest,
    RetainCompressionInfeasibleError,
    SplitAuthoringRequest,
    render_authoring_prompt,
    retain_generation_token_limit,
    validate_authoring_result,
)
from skillev.evolution.authoring import (
    context_patch_requirement_id,
    split_task_family_requirement_id,
    uncovered_edge_requirement_id,
)
from skillev.policy import (
    AUTHORING_JSON_ROOT_BOUNDARY_VERSION,
    AuthoringGenerationRequest,
    GenerationResult,
)
from skillev.rollout import (
    AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP,
    MAXIMUM_APPLICABLE_SKILL_POSITION,
    maximum_retrieved_skill_block_token_count,
)
from skillev.runtime import (
    BudgetLedger,
    BudgetVector,
    FullRetrievedSkillContext,
    LiveAttemptEventLog,
    RetrievalInclusionReason,
    RuntimeEventEmitter,
    SkillApplicability,
    SkillDocument,
    SkillManifest,
    SkillMetadata,
    SkillRequirement,
)
from tests.v3_helpers import CharacterTokenizer, make_authoring_edge, make_skill_document


class _ScriptedBaseBackbone:
    def __init__(
        self,
        tokenizer: CharacterTokenizer,
        outputs: tuple[AuthoringResult | BaseException, ...],
    ) -> None:
        self.tokenizer = tokenizer
        self._outputs = deque(outputs)
        self.requests: list[AuthoringGenerationRequest] = []
        self.results: list[GenerationResult] = []

    def generate_base(self, request: AuthoringGenerationRequest) -> GenerationResult:
        self.requests.append(request)
        outcome = self._outputs.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        content = tuple(self.tokenizer.encode(canonical_json(outcome.to_value())))
        if len(content) > request.max_new_tokens:
            result = GenerationResult(
                content_token_ids=content[: request.max_new_tokens],
                stop_token_ids=(),
                finish_reason="length",
            )
        else:
            result = GenerationResult(
                content_token_ids=content,
                stop_token_ids=(0, 1),
                finish_reason="stop",
            )
        self.results.append(result)
        return result


def _mode(
    context: str,
    *,
    event_id: str,
    mean: float,
    lcb: float,
    ucb: float,
) -> PosteriorTaskFamilyMode:
    return PosteriorTaskFamilyMode(
        task_family=context,
        cell_keys=(f"cell-{context}",),
        posterior_event_ids=(event_id,),
        evidence_mass=8.0,
        mean=mean,
        sigma=0.05,
        lcb=lcb,
        ucb=ucb,
        within_cell_mean_span=0.02,
    )


def _split_modality() -> SplitModalityEvidence:
    low = _mode("debug-low", event_id="posterior-low", mean=0.2, lcb=0.15, ucb=0.25)
    high = _mode("debug-high", event_id="posterior-high", mean=0.8, lcb=0.75, ucb=0.85)
    return SplitModalityEvidence(
        low_mode=low,
        high_mode=high,
        between_mean_gap=0.6,
        intervals_disjoint=True,
        source_task_families=("debug-high", "debug-low"),
        assignments=(
            SplitTaskFamilyAssignment(high, SplitBranch.HIGH),
            SplitTaskFamilyAssignment(low, SplitBranch.LOW),
        ),
        separation_cutpoint=0.5,
    )


def _draft(
    *,
    title: str,
    instructions: str,
    applicability: SkillApplicability,
    requirement_ids: tuple[str, ...],
    source_requirements: tuple[SkillRequirement, ...] = (),
) -> AuthoredSkillDraft:
    preserved = {r.requirement_id: r for r in source_requirements}
    return AuthoredSkillDraft(
        title=title,
        summary=f"Public summary for {title}.",
        instructions=instructions,
        applicability=applicability,
        requirements=tuple(
            preserved.get(
                item,
                SkillRequirement(
                    requirement_id=item, text=f"Satisfy {item}.", kind="evolvable-strategy"
                ),
            )
            for item in requirement_ids
        ),
    )


def _cases() -> tuple[
    tuple[
        RetainAuthoringRequest
        | RefineAuthoringRequest
        | SplitAuthoringRequest
        | GenerateAuthoringRequest,
        AuthoringResult,
        int,
    ],
    ...,
]:
    source = make_skill_document(
        "source",
        instructions="Use this deliberately long source procedure exactly and completely.",
        task_family="*",
    )
    exemplar = make_authoring_edge("trajectory-authoring:1")
    source_ids = tuple(item.requirement_id for item in source.requirements)
    retain = RetainAuthoringRequest(
        source=source,
        edge_exemplars=(exemplar,),
        evidence_summary="high flow",
        seed=11,
    )
    refine_key = "context-cell-a"
    refine = RefineAuthoringRequest(
        source=source,
        edge_exemplars=(exemplar,),
        evidence_summary="low-context posterior",
        target_context_keys=(refine_key,),
        seed=12,
    )
    modality = _split_modality()
    split = SplitAuthoringRequest(
        source=source,
        edge_exemplars=(exemplar,),
        evidence_summary="separated posterior modes",
        modality=modality,
        seed=13,
    )
    authority = GenerateAuthoritySelection(
        task_family=exemplar.task_family,
        context=exemplar.context_id,
        required_tools=(),
        input_schema_id="debug-input@3",
        output_schema_id="debug-output@3",
        license_id="unit-test",
    )
    generate = GenerateAuthoringRequest(
        edge_exemplars=(exemplar,),
        evidence_summary="uncovered high-asymmetry edge",
        authority=authority,
        evidence=GenerateEvidence((exemplar.edge_id,), 0.1, 0.9, "absolute-log-density-ratio@1"),
        seed=14,
    )
    low_applicability = SkillApplicability(
        task_families=modality.low_task_families,
        contexts=source.applicability.contexts,
        required_tools=source.applicability.required_tools,
        excluded_contexts=source.applicability.excluded_contexts,
    )
    high_applicability = SkillApplicability(
        task_families=modality.high_task_families,
        contexts=source.applicability.contexts,
        required_tools=source.applicability.required_tools,
        excluded_contexts=source.applicability.excluded_contexts,
    )
    return (
        (
            retain,
            AuthoringResult(
                (
                    _draft(
                        title="Compressed",
                        instructions="Apply the source procedure.",
                        applicability=source.applicability,
                        requirement_ids=source_ids,
                        source_requirements=source.requirements,
                    ),
                )
            ),
            1,
        ),
        (
            refine,
            AuthoringResult(
                (
                    _draft(
                        title="Refined",
                        instructions="Apply the source procedure with the context patch.",
                        applicability=source.applicability,
                        requirement_ids=(*source_ids, context_patch_requirement_id(refine_key)),
                        source_requirements=source.requirements,
                    ),
                )
            ),
            1,
        ),
        (
            split,
            AuthoringResult(
                (
                    _draft(
                        title="Low mode",
                        instructions="Apply the low-mode procedure.",
                        applicability=low_applicability,
                        source_requirements=source.requirements,
                        requirement_ids=(
                            *source_ids,
                            split_task_family_requirement_id(modality.low_task_families),
                        ),
                    ),
                    _draft(
                        title="High mode",
                        instructions="Apply the high-mode procedure.",
                        applicability=high_applicability,
                        source_requirements=source.requirements,
                        requirement_ids=(
                            *source_ids,
                            split_task_family_requirement_id(modality.high_task_families),
                        ),
                    ),
                )
            ),
            2,
        ),
        (
            generate,
            AuthoringResult(
                (
                    _draft(
                        title="Generated",
                        instructions="Apply the generated uncovered-edge procedure.",
                        applicability=SkillApplicability(
                            task_families=(authority.task_family,),
                            contexts=(authority.context,),
                            required_tools=authority.required_tools,
                            excluded_contexts=(),
                        ),
                        requirement_ids=(uncovered_edge_requirement_id(exemplar.edge_id),),
                    ),
                )
            ),
            1,
        ),
    )


@pytest.mark.parametrize(("authoring_request", "result", "draft_count"), _cases())
def test_base_model_authoring_uses_one_exact_ledger_call(
    tmp_path: Path,
    authoring_request: object,
    result: AuthoringResult,
    draft_count: int,
) -> None:
    tokenizer = CharacterTokenizer()
    backbone = _ScriptedBaseBackbone(tokenizer, (result,))
    maximum = AuthoringCallMaximum(input_tokens=100_000, output_tokens=100_000)
    ledger = BudgetLedger(
        run_id="authoring-budget",
        attempt_id="attempt-1",
        cap=maximum.to_budget_vector(),
    )
    log = LiveAttemptEventLog(
        tmp_path / "events.jsonl",
        run_id=ledger.run_id,
        attempt_id=ledger.attempt_id,
    )
    author = BaseModelSkillAuthor(
        backbone=backbone,  # type: ignore[arg-type]
        config=EvolutionConfig(
            generate_min_absolute_log_importance=0.1,
            max_skill_instruction_tokens_per_draft=4096,
            max_authoring_completion_tokens=100_000,
            max_authoring_prompt_tokens=100_000,
        ),
        sampling=AuthoringSamplingConfig(temperature=1.0, top_p=1.0),
        ledger=ledger,
        emitter=RuntimeEventEmitter(log, producer_id="author"),
        maximum=maximum,
    )

    authored = author.author(authoring_request)  # type: ignore[arg-type]

    assert len(authored.drafts) == draft_count
    assert len(backbone.requests) == 1
    assert len(backbone.results) == 1
    call = backbone.requests[0]
    generated = backbone.results[0]
    expected_output_maximum = maximum.output_tokens
    if isinstance(authoring_request, RetainAuthoringRequest):
        expected_output_maximum = retain_generation_token_limit(
            authoring_request.source,
            tokenizer=tokenizer,
        )
    assert call.max_new_tokens == expected_output_maximum
    assert call.completion_boundary_version == AUTHORING_JSON_ROOT_BOUNDARY_VERSION
    assert ledger.settled == BudgetVector(
        input_tokens=len(call.input_ids),
        output_tokens=len(generated.content_token_ids) + len(generated.stop_token_ids),
        model_calls=1,
    )
    assert ledger.settled.output_tokens < maximum.output_tokens
    assert ledger.entries[0].reservation.maximum == BudgetVector(
        input_tokens=len(call.input_ids),
        output_tokens=expected_output_maximum,
        model_calls=1,
    )
    ledger.assert_fully_settled()


def test_base_model_authoring_generation_failure_leaves_no_successful_settlement(
    tmp_path: Path,
) -> None:
    tokenizer = CharacterTokenizer()
    backbone = _ScriptedBaseBackbone(tokenizer, (RuntimeError("generation failed"),))
    maximum = AuthoringCallMaximum(input_tokens=100_000, output_tokens=100_000)
    ledger = BudgetLedger(
        run_id="authoring-failure",
        attempt_id="attempt-1",
        cap=maximum.to_budget_vector(),
    )
    author = BaseModelSkillAuthor(
        backbone=backbone,  # type: ignore[arg-type]
        config=EvolutionConfig(
            generate_min_absolute_log_importance=0.1,
            max_skill_instruction_tokens_per_draft=4096,
            max_authoring_completion_tokens=100_000,
            max_authoring_prompt_tokens=100_000,
        ),
        sampling=AuthoringSamplingConfig(temperature=1.0, top_p=1.0),
        ledger=ledger,
        emitter=RuntimeEventEmitter(
            LiveAttemptEventLog(
                tmp_path / "events.jsonl",
                run_id=ledger.run_id,
                attempt_id=ledger.attempt_id,
            ),
            producer_id="author",
        ),
        maximum=maximum,
    )

    request = _cases()[0][0]
    with pytest.raises(RuntimeError, match="generation failed"):
        author.author(request)

    assert ledger.settled == BudgetVector()
    assert ledger.reserved.model_calls == 1
    with pytest.raises(RuntimeError):
        ledger.assert_fully_settled()


def test_invalid_authored_payload_is_settled_exactly_but_never_returns_success(
    tmp_path: Path,
) -> None:
    tokenizer = CharacterTokenizer()
    invalid = AuthoringResult(
        (
            _draft(
                title="Not compressed",
                instructions=(
                    "This output is intentionally longer than the original source instructions."
                ),
                applicability=make_skill_document("source").applicability,
                requirement_ids=("requirement-source",),
            ),
        )
    )
    backbone = _ScriptedBaseBackbone(tokenizer, (invalid,))
    maximum = AuthoringCallMaximum(input_tokens=100_000, output_tokens=100_000)
    ledger = BudgetLedger(
        run_id="authoring-invalid",
        attempt_id="attempt-1",
        cap=maximum.to_budget_vector(),
    )
    author = BaseModelSkillAuthor(
        backbone=backbone,  # type: ignore[arg-type]
        config=EvolutionConfig(
            generate_min_absolute_log_importance=0.1,
            max_skill_instruction_tokens_per_draft=4096,
            max_authoring_completion_tokens=100_000,
            max_authoring_prompt_tokens=100_000,
        ),
        sampling=AuthoringSamplingConfig(temperature=1.0, top_p=1.0),
        ledger=ledger,
        emitter=RuntimeEventEmitter(
            LiveAttemptEventLog(
                tmp_path / "events.jsonl",
                run_id=ledger.run_id,
                attempt_id=ledger.attempt_id,
            ),
            producer_id="author",
        ),
        maximum=maximum,
    )

    with pytest.raises(AuthoringFailedError):
        author.author(_cases()[0][0])

    ledger.assert_fully_settled()
    assert ledger.settled.model_calls == 1


def test_retain_counts_complete_model_visible_content_not_only_instructions() -> None:
    request = _cases()[0][0]
    assert isinstance(request, RetainAuthoringRequest)
    draft = _draft(
        title="X" * 500,
        instructions="Short.",
        applicability=request.source.applicability,
        requirement_ids=tuple(item.requirement_id for item in request.source.requirements),
    )
    inflated = AuthoringResult(
        (
            AuthoredSkillDraft(
                title=draft.title,
                summary="Y" * 500,
                instructions=draft.instructions,
                applicability=draft.applicability,
                requirements=tuple(
                    SkillRequirement(
                        requirement_id=item.requirement_id,
                        text="Z" * 500,
                    )
                    for item in draft.requirements
                ),
            ),
        )
    )

    with pytest.raises(AuthoringFailedError):
        validate_authoring_result(
            request,
            inflated,
            tokenizer=CharacterTokenizer(),
            max_skill_instruction_tokens_per_draft=4096,
        )


def _oversized_authoring_case(
    index: int,
) -> tuple[
    RetainAuthoringRequest
    | RefineAuthoringRequest
    | SplitAuthoringRequest
    | GenerateAuthoringRequest,
    AuthoringResult,
]:
    request, result, _ = _cases()[index]
    oversized_instructions = "X" * 4_000
    if isinstance(request, RetainAuthoringRequest):
        source = make_skill_document(
            "oversized-retain-source",
            instructions="S" * 7_000,
            task_family="*",
        )
        request = RetainAuthoringRequest(
            source=source,
            edge_exemplars=request.edge_exemplars,
            evidence_summary=request.evidence_summary,
            seed=request.seed,
        )
        result = AuthoringResult(
            (
                _draft(
                    title="Compressed but too large for a future H0 block",
                    instructions=oversized_instructions,
                    applicability=source.applicability,
                    requirement_ids=tuple(item.requirement_id for item in source.requirements),
                ),
            )
        )
    else:
        result = AuthoringResult(
            (
                replace(result.drafts[0], instructions=oversized_instructions),
                *result.drafts[1:],
            )
        )
    return request, result


@pytest.mark.parametrize("case_index", range(4))
def test_every_authoring_action_rejects_an_oversized_future_h0_block(
    case_index: int,
) -> None:
    request, result = _oversized_authoring_case(case_index)

    with pytest.raises(AuthoringFailedError):
        validate_authoring_result(
            request,
            result,
            tokenizer=CharacterTokenizer(),
            max_skill_instruction_tokens_per_draft=10_000,
        )


def test_complete_block_cap_counts_metadata_labels_and_separators() -> None:
    request = _cases()[3][0]
    result = _cases()[3][1]
    assert isinstance(request, GenerateAuthoringRequest)
    tokenizer = CharacterTokenizer()
    draft = result.drafts[0]
    candidate: AuthoredSkillDraft | None = None
    context: FullRetrievedSkillContext | None = None
    for length in range(2_000, AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP):
        item = replace(draft, instructions="X" * length)
        content_hash = stable_hash(item.to_value())
        rendered = FullRetrievedSkillContext(
            metadata=SkillMetadata(
                skill_id=f"skill-{content_hash.removeprefix('sha256:')[:12]}",
                version="1",
                content_hash=content_hash,
                input_schema_id=request.authority.input_schema_id,
                output_schema_id=request.authority.output_schema_id,
                license_id=request.authority.license_id,
                provenance_hash=stable_hash({"test": "complete-block-cap"}),
            ),
            content=canonical_json(item.to_value()),
            inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
        )
        if (
            len(tokenizer.encode(rendered.content))
            <= AUTHORED_SKILL_FULL_BLOCK_TOKEN_CAP
            < maximum_retrieved_skill_block_token_count(
                rendered,
                maximum_position=MAXIMUM_APPLICABLE_SKILL_POSITION,
                tokenizer=tokenizer,
            )
        ):
            candidate = item
            context = rendered
            break
    assert candidate is not None
    assert context is not None

    with pytest.raises(AuthoringFailedError):
        validate_authoring_result(
            request,
            AuthoringResult((candidate,)),
            tokenizer=tokenizer,
            max_skill_instruction_tokens_per_draft=10_000,
        )


def test_irreducibly_minimal_retain_is_rejected_before_generation() -> None:
    applicability = SkillApplicability(
        task_families=("x",),
        contexts=("x",),
        required_tools=(),
        excluded_contexts=(),
    )
    requirements = (SkillRequirement(requirement_id="r", text="x"),)
    content = {
        "applicability": applicability.to_value(),
        "instructions": "x",
        "requirements": [item.to_value() for item in requirements],
        "summary": "x",
        "title": "x",
    }
    source = SkillDocument(
        manifest=SkillManifest(
            skill_id="skill-minimal",
            version="1",
            content_hash=stable_hash(content),
            input_schema_id="input@3",
            output_schema_id="output@3",
            license_id="unit-test",
            provenance_hash=stable_hash({"source": "minimal-retain-test"}),
        ),
        title="x",
        summary="x",
        instructions="x",
        applicability=applicability,
        requirements=requirements,
    )
    request = RetainAuthoringRequest(
        source=source,
        edge_exemplars=(make_authoring_edge("minimal:1"),),
        evidence_summary="high flow and reliable posterior",
        seed=991,
    )

    with pytest.raises(RetainCompressionInfeasibleError):
        render_authoring_prompt(request, tokenizer=CharacterTokenizer())


def test_authoring_config_persists_separate_instruction_and_completion_limits() -> None:
    config = EvolutionConfig(
        generate_min_absolute_log_importance=0.1,
        max_skill_instruction_tokens_per_draft=31,
        max_authoring_completion_tokens=97,
        max_authoring_prompt_tokens=211,
    )

    restored = EvolutionConfig.from_value(config.to_value())

    assert restored == config
    assert restored.max_skill_instruction_tokens_per_draft == 31
    assert restored.max_authoring_completion_tokens == 97
    assert restored.max_authoring_prompt_tokens == 211


@pytest.mark.parametrize(
    ("applicability", "requirement_ids"),
    [
        (
            SkillApplicability(
                task_families=("debug-family",),
                contexts=("debug-family",),
                required_tools=(),
                excluded_contexts=("other-context",),
            ),
            (uncovered_edge_requirement_id("trajectory-authoring:1"),),
        ),
        (
            SkillApplicability(
                task_families=("debug-family",),
                contexts=("debug-family",),
                required_tools=(),
                excluded_contexts=(),
            ),
            (
                uncovered_edge_requirement_id("trajectory-authoring:1"),
                "unapproved-requirement",
            ),
        ),
    ],
)
def test_generate_authoring_rejects_any_applicability_or_requirement_expansion(
    tmp_path: Path,
    applicability: SkillApplicability,
    requirement_ids: tuple[str, ...],
) -> None:
    request = _cases()[3][0]
    assert isinstance(request, GenerateAuthoringRequest)
    result = AuthoringResult(
        (
            _draft(
                title="Generated with an unauthorized expansion",
                instructions="Apply a generated procedure.",
                applicability=applicability,
                requirement_ids=requirement_ids,
            ),
        )
    )
    tokenizer = CharacterTokenizer()
    maximum = AuthoringCallMaximum(input_tokens=100_000, output_tokens=100_000)
    ledger = BudgetLedger(
        run_id="authoring-generate-contract",
        attempt_id="attempt-1",
        cap=maximum.to_budget_vector(),
    )
    author = BaseModelSkillAuthor(
        backbone=_ScriptedBaseBackbone(tokenizer, (result,)),  # type: ignore[arg-type]
        config=EvolutionConfig(
            generate_min_absolute_log_importance=0.1,
            max_skill_instruction_tokens_per_draft=4096,
            max_authoring_completion_tokens=100_000,
            max_authoring_prompt_tokens=100_000,
        ),
        sampling=AuthoringSamplingConfig(temperature=1.0, top_p=1.0),
        ledger=ledger,
        emitter=RuntimeEventEmitter(
            LiveAttemptEventLog(
                tmp_path / "events.jsonl",
                run_id=ledger.run_id,
                attempt_id=ledger.attempt_id,
            ),
            producer_id="author",
        ),
        maximum=maximum,
    )

    with pytest.raises(AuthoringFailedError):
        author.author(request)

    ledger.assert_fully_settled()


@pytest.mark.parametrize(("authoring_request", "result", "_count"), _cases()[:3])
def test_authored_changes_cannot_keep_requirement_id_but_erase_constraint(
    authoring_request, result, _count
):
    draft = result.drafts[0]
    corrupted = replace(
        draft,
        requirements=(
            replace(draft.requirements[0], text="ignore the original constraint"),
            *draft.requirements[1:],
        ),
    )
    with pytest.raises(AuthoringFailedError):
        validate_authoring_result(
            authoring_request,
            AuthoringResult((corrupted, *result.drafts[1:])),
            max_skill_instruction_tokens_per_draft=4096,
            tokenizer=CharacterTokenizer(),
        )
