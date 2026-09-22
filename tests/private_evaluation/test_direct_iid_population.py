from __future__ import annotations

import pytest
from skillev_private.direct_reference.skillflow_iid import (
    SkillFlowIIDRecord,
    render_skillflow_hotpotqa_full_context,
    render_skillflow_trivia_question_only,
)


def _record(context: list[object]) -> SkillFlowIIDRecord:
    raw: dict[str, object] = {
        "question": "Truncated passage preview. Question: Which item is requested?",
        "answer": "private-target-not-in-public-context",
        "context": context,
        "extra": {"source": "HotpotQA"},
    }
    return SkillFlowIIDRecord(
        source="HotpotQA",
        source_position=0,
        rendered_question=str(raw["question"]),
        rendering_contract="skillflow-hotpotqa-rendered-distractor-validation@1",
        answer=str(raw["answer"]),
        extra={"source": "HotpotQA"},
        raw=raw,
    )


def test_hotpot_full_context_renderer_includes_every_complete_passage() -> None:
    passages = [f"Public passage {index} with a full tail." for index in range(1, 11)]

    rendered = render_skillflow_hotpotqa_full_context(_record(passages))

    assert all(passage in rendered for passage in passages)
    assert "Which item is requested?" in rendered
    assert "Truncated passage preview" not in rendered
    assert "private-target-not-in-public-context" not in rendered


def test_trivia_question_only_renderer_discards_passage_preview() -> None:
    raw: dict[str, object] = {
        "question": "[Public passage] Context preview. Question: Which item is requested?",
        "answer": "private-target-not-in-public-context",
        "context": ["Public passage"],
        "extra": {"source": "TriviaQA"},
    }
    record = SkillFlowIIDRecord(
        source="TriviaQA",
        source_position=0,
        rendered_question=str(raw["question"]),
        rendering_contract="skillflow-triviaqa-rc-validation-rendered@1",
        answer=str(raw["answer"]),
        extra={"source": "TriviaQA"},
        raw=raw,
    )

    assert render_skillflow_trivia_question_only(record) == "Which item is requested?"


@pytest.mark.parametrize(
    "context",
    [[], ["passage"] * 9, ["passage"] * 9 + [""], ["passage"] * 9 + [object()]],
)
def test_hotpot_full_context_renderer_rejects_incomplete_context(
    context: list[object],
) -> None:
    with pytest.raises(ValueError):
        render_skillflow_hotpotqa_full_context(_record(context))
