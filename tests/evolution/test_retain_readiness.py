from __future__ import annotations

import json

import pytest

from skillev.contracts import GenerateEvidence, RetainEvidence
from skillev.evolution import (
    AUTHORING_TEMPLATE_RETAIN,
    AuthoredSkillDraft,
    AuthoringResult,
    EvolutionConfig,
    EvolutionDecision,
    GenerateProposal,
    RetainCompressionInfeasibleError,
    RetainProposal,
    RetainSuiteKind,
    SkillAuthoringAuthority,
    build_evolution_mutation,
    minimum_legal_retain_draft,
    render_authoring_prompt,
    retain_compression_budget,
    retain_generation_token_limit,
    retain_readiness_suite,
)
from skillev.runtime import SkillLibrary, SkillLibraryState
from tests.fakes.skill_author import ScriptedSkillAuthor
from tests.v3_helpers import CharacterTokenizer, make_authoring_edge, make_phase_event

_REQUIRED_PROFILES = {
    "few-requirements",
    "many-requirements",
    "short-instructions",
    "medium-instructions",
    "long-instructions",
    "single-task-family",
    "multiple-task-families",
    "with-tool",
    "without-tool",
    "highly-repetitive",
    "structured-low-redundancy",
    "near-minimal-legal",
}


@pytest.mark.parametrize("kind", tuple(RetainSuiteKind))
def test_fixed_public_retain_suites_cover_the_owner_matrix(kind: RetainSuiteKind) -> None:
    suite = retain_readiness_suite(kind)
    repeated = retain_readiness_suite(kind)

    assert suite.to_value() == repeated.to_value()
    assert suite.content_hash == repeated.content_hash
    assert _REQUIRED_PROFILES <= {profile for case in suite.cases for profile in case.profiles}
    assert len({case.case_id for case in suite.cases}) == len(suite.cases)
    assert all(case.request.source == case.source for case in suite.cases)


def test_development_and_holdout_suites_are_disjoint() -> None:
    development = retain_readiness_suite(RetainSuiteKind.DEVELOPMENT)
    holdout = retain_readiness_suite(RetainSuiteKind.HOLDOUT)

    assert development.content_hash != holdout.content_hash
    assert not (
        {item.case_id for item in development.cases} & {item.case_id for item in holdout.cases}
    )
    assert not (
        {item.source.manifest.content_hash for item in development.cases}
        & {item.source.manifest.content_hash for item in holdout.cases}
    )


def test_retain_readiness_preserves_constraints_and_uses_current_template() -> None:
    case = retain_readiness_suite(RetainSuiteKind.DEVELOPMENT).cases[0]
    prompt = render_authoring_prompt(case.request, tokenizer=CharacterTokenizer())
    payload = json.loads(prompt.splitlines()[1])
    assert payload["template_version"] == AUTHORING_TEMPLATE_RETAIN
    assert minimum_legal_retain_draft(case.source).requirements == case.source.requirements
    constraints = payload["output_contract"]["action_constraints"]
    assert constraints["exact_requirements"] == [r.to_value() for r in case.source.requirements]


def test_every_suite_case_has_a_pre_generation_budget_and_explicit_prompt_contract() -> None:
    tokenizer = CharacterTokenizer()
    for kind in RetainSuiteKind:
        for case in retain_readiness_suite(kind).cases:
            budget = retain_compression_budget(case.source, tokenizer=tokenizer)
            prompt = render_authoring_prompt(case.request, tokenizer=tokenizer)
            constraints = json.loads(prompt.splitlines()[1])["output_contract"][
                "action_constraints"
            ]

            assert budget.maximum_output_model_visible_token_count == (
                budget.source_model_visible_token_count - 1
            )
            assert budget.minimum_legal_model_visible_token_count < (
                budget.source_model_visible_token_count
            )
            assert constraints["source_model_visible_token_count"] == (
                budget.source_model_visible_token_count
            )
            assert constraints["maximum_output_model_visible_token_count"] == (
                budget.maximum_output_model_visible_token_count
            )
            assert constraints["tokenizer_identity"] == tokenizer.public_identity.to_value()
            assert constraints["maximum_generation_content_token_count"] == (
                retain_generation_token_limit(case.source, tokenizer=tokenizer)
            )


def test_irreducible_retain_preflight_happens_before_any_phi_model_call() -> None:
    tokenizer = CharacterTokenizer()
    near = retain_readiness_suite(RetainSuiteKind.DEVELOPMENT).cases[-1].source
    minimum = minimum_legal_retain_draft(near)
    from skillev.contracts import stable_hash
    from skillev.runtime import SkillDocument, SkillManifest

    minimum_content = minimum.to_value()
    source = SkillDocument(
        manifest=SkillManifest(
            skill_id=near.manifest.skill_id,
            version=near.manifest.version,
            content_hash=stable_hash(minimum_content),
            input_schema_id=near.manifest.input_schema_id,
            output_schema_id=near.manifest.output_schema_id,
            license_id=near.manifest.license_id,
            provenance_hash=near.manifest.provenance_hash,
        ),
        title=minimum.title,
        summary=minimum.summary,
        instructions=minimum.instructions,
        applicability=minimum.applicability,
        requirements=minimum.requirements,
    )
    library = SkillLibrary(SkillLibraryState.from_seed_documents((source,)))
    phase = make_phase_event(library_version=library.current_version)
    gap = make_authoring_edge("preflight-gap:1")
    retain_edge = make_authoring_edge(
        "preflight-retain:1",
        invoked_skill_ids=(source.manifest.skill_id,),
    )
    decision = EvolutionDecision(
        phase_event_id=phase.event_id,
        proposals=(
            GenerateProposal(
                evidence=GenerateEvidence(
                    importance_edge_ids=(gap.edge_id,),
                    minimum_absolute_log_importance=0.1,
                    importance_quantile=0.9,
                    importance_semantics="absolute-log-density-ratio@1",
                ),
                rationale_text="public uncovered edge",
                edge_exemplars=(gap,),
            ),
            RetainProposal(
                target_skill_id=source.manifest.skill_id,
                evidence=RetainEvidence(
                    log_skill_marginal_flow=1.0,
                    flow_quantile=1.0,
                    lcb=0.9,
                    k=1.0,
                ),
                rationale_text="high flow and reliable posterior",
                edge_exemplars=(retain_edge,),
            ),
        ),
    )
    author = ScriptedSkillAuthor(
        tokenizer,
        (AuthoringResult(drafts=(AuthoredSkillDraft.from_document(source),)),),
    )

    with pytest.raises(RetainCompressionInfeasibleError):
        build_evolution_mutation(
            decision,
            author=author,
            library=library,
            phase_event=phase,
            config=EvolutionConfig(generate_min_absolute_log_importance=0.1),
            authority=SkillAuthoringAuthority(
                input_schema_id="input@3",
                output_schema_id="output@3",
                license_id="unit-test",
                allowed_task_families=(gap.task_family,),
                allowed_tools=(),
            ),
            base_seed=7,
            cycle_ordinal=1,
        )

    assert author.requests == ()
    assert library.active_skill_ids == (source.manifest.skill_id,)
