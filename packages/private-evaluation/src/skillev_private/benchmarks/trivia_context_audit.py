"""Answer-free audit of the prepared TriviaQA wire projection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TriviaContextAudit:
    task_id: str
    released_question_nonempty: bool
    context_marker_present: bool
    question_occurrences: int
    visible_passage_count: int
    retained_context_count: int
    alias_count: int
    alias_source: str
    prompt_tokens: int
    overflow: bool


def audit_trivia_wire(
    *,
    task_id: str,
    rendered_question: str,
    alias_count: int,
    alias_source: str,
    prompt_tokens: int,
    context_length: int,
) -> TriviaContextAudit:
    if alias_count <= 0 or prompt_tokens < 0 or context_length <= 0:
        raise ValueError("TriviaQA audit counts are invalid")
    _prefix, marker, question = rendered_question.rpartition("Question:")
    lowered = rendered_question.casefold()
    passage_count = sum(lowered.count(token) for token in ("passage ", "context:", "document "))
    return TriviaContextAudit(
        task_id=task_id,
        released_question_nonempty=bool(question.strip() if marker else rendered_question.strip()),
        context_marker_present=bool(marker),
        question_occurrences=(
            rendered_question.count(question.strip()) if marker and question.strip() else 0
        ),
        visible_passage_count=passage_count,
        retained_context_count=passage_count,
        alias_count=alias_count,
        alias_source=alias_source,
        prompt_tokens=prompt_tokens,
        overflow=prompt_tokens > context_length,
    )


__all__ = ["TriviaContextAudit", "audit_trivia_wire"]
