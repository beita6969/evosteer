from __future__ import annotations

from dataclasses import dataclass

import pytest

from skillev.evaluation.step0_conditions import default_step_zero_retrieval_policy
from skillev.evolution import TaskConditionedSkillRetriever
from skillev.experiments._evolution_preflight_seed import planned_step_zero_seed_documents
from skillev.rollout import RolloutTask
from skillev.runtime import SkillLibrary, SkillLibraryState


@dataclass(frozen=True, slots=True)
class _Tokenizer:
    tokenizer_id: str = "fixture"

    def encode(self, text: str) -> list[int]:
        return list(text.encode())


def _task(benchmark: str, tools: tuple[str, ...]) -> RolloutTask:
    return RolloutTask(
        task_id="task",
        environment_id="environment",
        task_family="fixture",
        context_id=f"protocol13:{benchmark}:iid",
        query="answer-free public root query",
        available_tools=tools,
        public_context={"benchmark_id": benchmark},
    )


def test_step0_retrieval_selects_one_benchmark_skill_and_respects_budget() -> None:
    retriever = TaskConditionedSkillRetriever(
        library=SkillLibrary(
            SkillLibraryState.from_seed_documents(planned_step_zero_seed_documents())
        )
    )
    decision = retriever.retrieve_step_zero(
        _task("aime-2026", ()),
        policy=default_step_zero_retrieval_policy(),
        tokenizer=_Tokenizer(),
        instruction_token_budget=4096,
    )

    assert tuple(item.metadata.skill_id for item in decision.selected) == (
        "skill-aime-integer-verification",
    )
    assert not decision.abstained
    assert decision.total_instruction_tokens <= 4096


@pytest.mark.parametrize(
    ("benchmark", "tools", "expected_skill"),
    [
        ("hotpotqa", (), "skill-hotpot-evidence-chain"),
        ("triviaqa", ("search",), "skill-trivia-search-synthesis"),
        ("aime-2026", (), "skill-aime-integer-verification"),
        ("healthbench", (), "skill-health-comprehensive-response"),
        ("webshop", ("click", "search"), "skill-webshop-constraint-ledger"),
        ("alfworld", ("act",), "skill-alfworld-subgoal-machine"),
        ("mbpp-plus", (), "skill-python-function-completion"),
        ("humaneval", (), "skill-python-function-completion"),
    ],
)
def test_exact_eight_routes_behavior_through_one_typed_skill(
    benchmark: str,
    tools: tuple[str, ...],
    expected_skill: str,
) -> None:
    retriever = TaskConditionedSkillRetriever(
        library=SkillLibrary(
            SkillLibraryState.from_seed_documents(planned_step_zero_seed_documents())
        )
    )
    decision = retriever.retrieve_step_zero(
        _task(benchmark, tools),
        policy=default_step_zero_retrieval_policy(),
        tokenizer=_Tokenizer(),
        instruction_token_budget=4096,
    )

    assert tuple(item.metadata.skill_id for item in decision.selected) == (expected_skill,)


def test_step0_retrieval_can_abstain_and_never_injects_all_wildcard_skills() -> None:
    retriever = TaskConditionedSkillRetriever(
        library=SkillLibrary(
            SkillLibraryState.from_seed_documents(planned_step_zero_seed_documents())
        )
    )
    unknown = retriever.retrieve_step_zero(
        _task("unknown", ()),
        policy=default_step_zero_retrieval_policy(),
        tokenizer=_Tokenizer(),
        instruction_token_budget=4096,
    )
    webshop_without_tools = retriever.retrieve_step_zero(
        _task("webshop", ()),
        policy=default_step_zero_retrieval_policy(),
        tokenizer=_Tokenizer(),
        instruction_token_budget=4096,
    )

    assert unknown.abstained
    assert not unknown.selected
    assert webshop_without_tools.abstained
    assert not webshop_without_tools.selected
