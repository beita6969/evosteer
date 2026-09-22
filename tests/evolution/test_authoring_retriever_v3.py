from __future__ import annotations

import json

from skillev.contracts import (
    FailureMode,
    GenerateEvidence,
    HorizonBucket,
    TokenBucket,
    canonical_json,
)
from skillev.evolution import (
    AUTHORING_TEMPLATE_RETAIN,
    AuthoredSkillDraft,
    AuthoringActionKind,
    AuthoringEdgeEvidence,
    AuthoringResult,
    AuthoringSamplingConfig,
    GenerateAuthoringRequest,
    GenerateAuthoritySelection,
    RetainAuthoringRequest,
    TaskConditionedSkillRetriever,
    render_authoring_prompt,
)
from skillev.evolution.authoring import uncovered_edge_requirement_id
from skillev.rollout import RolloutTask
from skillev.runtime import (
    SkillApplicability,
    SkillLibrary,
    SkillLibraryState,
    SkillRequirement,
)
from tests.v3_helpers import (
    TEST_CONTEXT_ID,
    TEST_TASK_FAMILY,
    CharacterTokenizer,
    make_skill_document,
)


def _exemplar() -> AuthoringEdgeEvidence:
    return AuthoringEdgeEvidence(
        edge_id="trajectory-1:1",
        task_family=TEST_TASK_FAMILY,
        context_id=TEST_CONTEXT_ID,
        action_kind=AuthoringActionKind.TOOL,
        tool_or_skill_name="debug.tool",
        argument_schema_id="schema@1",
        observation_status=FailureMode.SUCCESS,
        token_bucket=TokenBucket.LE_1K,
        horizon_bucket=HorizonBucket.LE_3,
        absolute_log_importance=1.0,
        log_importance_quantile=1.0,
        invoked_skill_ids=(),
        available_tools=(),
    )


def test_action_specific_requests_render_exact_structured_contracts() -> None:
    tokenizer = CharacterTokenizer()
    source = make_skill_document("alpha")
    retain = RetainAuthoringRequest(
        source=source,
        edge_exemplars=(_exemplar(),),
        evidence_summary="high flow and reliable posterior",
        seed=7,
    )
    prompt = render_authoring_prompt(retain, tokenizer=tokenizer)
    payload = json.loads(prompt.splitlines()[1])

    assert payload["template_version"] == AUTHORING_TEMPLATE_RETAIN
    assert payload["material"]["source"] == source.content_value()
    constraints = payload["output_contract"]["action_constraints"]
    source_count = len(tokenizer.encode(canonical_json(source.content_value())))
    assert constraints["source_model_visible_token_count"] == source_count
    assert constraints["maximum_output_model_visible_token_count"] == source_count - 1
    assert constraints["tokenizer_identity"] == tokenizer.public_identity.to_value()
    assert constraints["exact_applicability"] == source.applicability.to_value()
    assert constraints["exact_requirement_ids"] == [
        item.requirement_id for item in source.requirements
    ]
    assert constraints["exact_authored_text_types"]["instructions"] == "JSON string"
    assert "native_payload" not in prompt
    assert AuthoringSamplingConfig(temperature=0.7, top_p=0.9).top_p == 0.9


def test_generate_request_carries_authority_in_its_type() -> None:
    exemplar = _exemplar()
    authority = GenerateAuthoritySelection(
        task_family=exemplar.task_family,
        context=exemplar.context_id,
        required_tools=(),
        input_schema_id="input@3",
        output_schema_id="output@3",
        license_id="unit-test",
    )
    request = GenerateAuthoringRequest(
        edge_exemplars=(exemplar,),
        evidence_summary="uncovered high asymmetry",
        authority=authority,
        evidence=GenerateEvidence((exemplar.edge_id,), 0.1, 0.9, "absolute-log-density-ratio@1"),
        seed=9,
    )
    requirement = SkillRequirement(
        requirement_id=uncovered_edge_requirement_id(exemplar.edge_id),
        text="Cover the uncovered public edge.",
    )
    result = AuthoringResult(
        drafts=(
            AuthoredSkillDraft(
                title="Generated skill",
                summary="Covers a public gap.",
                instructions="Apply the generated procedure.",
                applicability=SkillApplicability(
                    task_families=(TEST_TASK_FAMILY,),
                    contexts=(TEST_CONTEXT_ID,),
                    required_tools=(),
                    excluded_contexts=(),
                ),
                requirements=(requirement,),
            ),
        )
    )

    assert AuthoringResult.from_value(result.to_value()) == result
    assert request.authority == authority


def test_retriever_returns_every_applicable_skill_with_full_content() -> None:
    alpha = make_skill_document("alpha", instructions="FULL ALPHA INSTRUCTIONS")
    beta = make_skill_document("beta", task_family="other-family")
    library = SkillLibrary(SkillLibraryState.from_seed_documents((alpha, beta)))
    retriever = TaskConditionedSkillRetriever(
        library=library,
    )
    task = RolloutTask(
        task_id="task-1",
        environment_id="environment-1",
        task_family=TEST_TASK_FAMILY,
        context_id=TEST_CONTEXT_ID,
        query="public query",
        available_tools=(),
        public_context={},
    )

    contexts = retriever.retrieve(task)

    assert tuple(item.metadata.skill_id for item in contexts) == ("skill-alpha",)
    assert contexts[0].content == canonical_json(alpha.content_value())
    assert "FULL ALPHA INSTRUCTIONS" in contexts[0].content


def test_split_task_family_documents_keep_the_context_axis_independent() -> None:
    low = make_skill_document(
        "low-family",
        task_family="debug/low-family",
        context_id="debug:shared-context",
    )
    high = make_skill_document(
        "high-family",
        task_family="debug/high-family",
        context_id="debug:shared-context",
    )
    library = SkillLibrary(SkillLibraryState.from_seed_documents((low, high)))
    retriever = TaskConditionedSkillRetriever(library=library)

    def task(*, family: str, context: str) -> RolloutTask:
        return RolloutTask(
            task_id=f"task-{family}-{context}",
            environment_id="environment",
            task_family=family,
            context_id=context,
            query="public query",
            available_tools=(),
            public_context={},
        )

    assert tuple(
        item.metadata.skill_id
        for item in retriever.retrieve(
            task(family="debug/low-family", context="debug:shared-context")
        )
    ) == ("skill-low-family",)
    assert tuple(
        item.metadata.skill_id
        for item in retriever.retrieve(
            task(family="debug/high-family", context="debug:shared-context")
        )
    ) == ("skill-high-family",)
    assert retriever.retrieve(task(family="debug/low-family", context="debug:other-context")) == ()
