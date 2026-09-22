import pytest

from skillev.experiments._evolution_preflight_seed import planned_step_zero_seed_documents
from skillev.rollout import CanonicalInitialContextAssembler, RolloutTask
from skillev.rollout.prompt_profiles import InitialContextProfile
from skillev.runtime import FullRetrievedSkillContext, RetrievalInclusionReason, SkillMetadata


class _Tokenizer:
    tokenizer_id = "character@1"

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


def test_step_zero_h0_uses_actual_stable_root_query_and_disables_invocation() -> None:
    document = planned_step_zero_seed_documents()[0]
    retrieved = FullRetrievedSkillContext(
        metadata=SkillMetadata.from_document(document),
        content=document.instructions,
        inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
    )
    task = RolloutTask(
        task_id="public-one",
        environment_id="protocol13:hotpotqa:step-zero",
        task_family="qa",
        context_id="protocol13:hotpotqa:iid",
        query="Which public entity connects the two passages?",
        available_tools=(),
        public_context={
            "benchmark_id": "hotpotqa",
            "skill_exposure_mode": "full-inline-no-invoke",
            "step_zero_terminal_wire": "short-answer",
        },
    )
    assembled = CanonicalInitialContextAssembler(maximum_h0_tokens=8192).assemble(
        task=task,
        retrieved_skills=(retrieved,),
        active_skill_ids=(document.manifest.skill_id,),
        library_version="step0-library@1",
        tokenizer=_Tokenizer(),
        profile=InitialContextProfile.SEEDED_STEP_ZERO,
    )

    assert "### Query\nWhich public entity connects the two passages?" in assembled.text
    assert "skill invocation" not in assembled.text.casefold()
    assert assembled.contract.meta["root_query_source"] == "rollout-task-query"
    assert assembled.contract.meta["root_query_stable_across_episode"] is True


def test_aime_step_zero_controller_only_sets_phase_and_wire() -> None:
    document = next(
        item
        for item in planned_step_zero_seed_documents()
        if item.manifest.skill_id == "skill-aime-integer-verification"
    )
    retrieved = FullRetrievedSkillContext(
        metadata=SkillMetadata.from_document(document),
        content=document.instructions,
        inclusion_reason=RetrievalInclusionReason.APPLICABILITY_MATCH,
    )
    task = RolloutTask(
        task_id="aime-public",
        environment_id="protocol13:aime-2026:step-zero",
        task_family="math",
        context_id="protocol13:aime-2026:iid",
        query="A public competition-math problem",
        available_tools=(),
        public_context={
            "benchmark_id": "aime-2026",
            "skill_exposure_mode": "full-inline-no-invoke",
            "step_zero_terminal_wire": "integer-0-999",
        },
    )
    assembled = CanonicalInitialContextAssembler(maximum_h0_tokens=8192).assemble(
        task=task,
        retrieved_skills=(retrieved,),
        active_skill_ids=(document.manifest.skill_id,),
        library_version="step0-library@1",
        tokenizer=_Tokenizer(),
        profile=InitialContextProfile.SEEDED_STEP_ZERO,
    )

    assert document.instructions in assembled.text
    assert "strategy comes from the retrieved skill" not in assembled.text
    assert "do not emit the terminal value" not in assembled.text
    assert "private working text" not in assembled.text


@pytest.mark.parametrize("forbidden_key", ["answer", "reward", "scorer", "strategy"])
def test_seeded_step_zero_rejects_non_skill_authority_fields(forbidden_key: str) -> None:
    task = RolloutTask(
        task_id="public-task",
        environment_id="protocol13:hotpotqa:step-zero",
        task_family="qa",
        context_id="protocol13:hotpotqa:iid",
        query="Answer-free public task",
        available_tools=(),
        public_context={
            "benchmark_id": "hotpotqa",
            "step_zero_terminal_wire": "short-answer",
            "nested": {forbidden_key: "must not enter H0"},
        },
    )

    with pytest.raises(ValueError):
        CanonicalInitialContextAssembler(maximum_h0_tokens=8192).assemble(
            task=task,
            retrieved_skills=(),
            active_skill_ids=(),
            library_version="step0-library@1",
            tokenizer=_Tokenizer(),
            profile=InitialContextProfile.SEEDED_STEP_ZERO,
        )
