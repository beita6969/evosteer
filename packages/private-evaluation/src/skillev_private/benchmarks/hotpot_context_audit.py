"""Answer-free integrity audit for released HotpotQA model-visible context."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ContextEvidenceStatus(StrEnum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially-verified"
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True, slots=True)
class HotpotContextAudit:
    task_id: str
    passage_count: int
    question_occurrences: int
    supporting_fact_count: int | None
    supporting_fact_present_count: int | None
    evidence_status: ContextEvidenceStatus

    @property
    def structurally_valid(self) -> bool:
        if self.passage_count != 10 or self.question_occurrences != 1:
            return False
        if self.supporting_fact_count is not None:
            return self.supporting_fact_present_count == self.supporting_fact_count
        return True

    @property
    def passed(self) -> bool:
        return self.structurally_valid


@dataclass(frozen=True, slots=True)
class HotpotSupportingFactRef:
    title: str
    sentence_index: int


@dataclass(frozen=True, slots=True)
class HotpotPassage:
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HotpotSupportingFact:
    title: str
    sentence_index: int
    sentence_text: str


@dataclass(frozen=True, slots=True)
class HotpotContextAuditV2:
    task_id: str
    render_mode: str
    passage_count: int
    title_count: int
    supporting_fact_count: int
    supporting_titles_present: int
    supporting_sentences_present: int
    question_occurrences: int
    forbidden_private_field_count: int
    prompt_token_count: int
    context_overflow: bool
    support_verification: str

    @property
    def structurally_valid(self) -> bool:
        return (
            self.passage_count == 10
            and self.title_count == 10
            and self.question_occurrences == 1
            and self.forbidden_private_field_count == 0
            and not self.context_overflow
            and (
                self.support_verification != "verified"
                or (
                    self.supporting_titles_present == self.supporting_fact_count
                    and self.supporting_sentences_present == self.supporting_fact_count
                )
            )
        )


def audit_structured_hotpot_context(
    *,
    task_id: str,
    render_mode: str,
    rendered_question: str,
    passages: tuple[HotpotPassage, ...],
    supporting_facts: tuple[HotpotSupportingFact, ...] | None,
    prompt_token_count: int,
    context_length: int,
    forbidden_private_field_count: int = 0,
) -> HotpotContextAuditV2:
    if prompt_token_count < 0 or context_length <= 0 or forbidden_private_field_count < 0:
        raise ValueError("Hotpot structured audit counts are invalid")
    by_title = {passage.title: passage for passage in passages}
    if len(by_title) != len(passages):
        raise ValueError("Hotpot passage titles must be unique")
    title_hits = sentence_hits = 0
    facts = supporting_facts or ()
    for fact in facts:
        passage = by_title.get(fact.title)
        if passage is None:
            continue
        title_hits += 1
        if (
            0 <= fact.sentence_index < len(passage.sentences)
            and passage.sentences[fact.sentence_index].strip() == fact.sentence_text.strip()
        ):
            sentence_hits += 1
    question = rendered_question.rpartition("Question:")[2].strip()
    return HotpotContextAuditV2(
        task_id=task_id,
        render_mode=render_mode,
        passage_count=len(passages),
        title_count=len(by_title),
        supporting_fact_count=len(facts),
        supporting_titles_present=title_hits,
        supporting_sentences_present=sentence_hits,
        question_occurrences=rendered_question.count(question) if question else 0,
        forbidden_private_field_count=forbidden_private_field_count,
        prompt_token_count=prompt_token_count,
        context_overflow=prompt_token_count > context_length,
        support_verification="verified" if supporting_facts is not None else "unavailable",
    )


def audit_hotpot_context(
    *,
    task_id: str,
    rendered_question: str,
    public_context: tuple[str, ...],
    supporting_facts: object,
) -> HotpotContextAudit:
    if not task_id.strip() or not rendered_question.strip() or not public_context:
        raise ValueError("Hotpot context audit identity is incomplete")
    question = rendered_question.rpartition("Question:")[2].strip()
    if not question:
        raise ValueError("Hotpot rendered question lacks a final question")
    snippets = _public_supporting_snippets(supporting_facts)
    if snippets is None:
        refs = parse_supporting_fact_refs(supporting_facts)
        if refs is not None:
            joined = "\n".join(public_context)
            present = sum(ref.title in joined for ref in refs)
            return HotpotContextAudit(
                task_id,
                len(public_context),
                rendered_question.count(question),
                len(refs),
                present,
                ContextEvidenceStatus.PARTIALLY_VERIFIED,
            )
        return HotpotContextAudit(
            task_id,
            len(public_context),
            rendered_question.count(question),
            None,
            None,
            ContextEvidenceStatus.UNVERIFIABLE,
        )
    joined = "\n".join(public_context)
    present = sum(snippet in joined for snippet in snippets)
    return HotpotContextAudit(
        task_id,
        len(public_context),
        rendered_question.count(question),
        len(snippets),
        present,
        ContextEvidenceStatus.VERIFIED,
    )


def parse_supporting_fact_refs(value: object) -> tuple[HotpotSupportingFactRef, ...] | None:
    if not isinstance(value, list):
        return None
    refs: list[HotpotSupportingFactRef] = []
    for item in value:
        if (
            isinstance(item, list | tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and type(item[1]) is int
        ):
            refs.append(HotpotSupportingFactRef(item[0], item[1]))
            continue
        if isinstance(item, dict):
            title = item.get("title")
            sentence_index = item.get("sent_id", item.get("sentence_index"))
            if isinstance(title, str) and type(sentence_index) is int:
                refs.append(HotpotSupportingFactRef(title, sentence_index))
                continue
        return None
    return tuple(refs) if refs else None


def _public_supporting_snippets(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    snippets: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            snippets.append(item.strip())
        elif isinstance(item, dict):
            candidate = item.get("sentence", item.get("text"))
            if not isinstance(candidate, str) or not candidate.strip():
                return None
            snippets.append(candidate.strip())
        else:
            return None
    return tuple(snippets) if snippets else None


__all__ = [
    "ContextEvidenceStatus",
    "HotpotContextAudit",
    "HotpotContextAuditV2",
    "HotpotPassage",
    "HotpotSupportingFact",
    "HotpotSupportingFactRef",
    "audit_hotpot_context",
    "audit_structured_hotpot_context",
    "parse_supporting_fact_refs",
]
