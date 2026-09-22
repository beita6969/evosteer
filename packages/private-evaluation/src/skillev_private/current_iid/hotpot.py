"""Private HotpotQA full-context integrity checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HotpotContextAudit:
    task_id: str
    passage_count: int
    question_occurrences: int
    support_titles_expected: int | None
    support_titles_present: int | None
    support_sentences_expected: int | None
    support_sentences_present: int | None
    forbidden_public_fields: tuple[str, ...]

    @property
    def passed(self) -> bool:
        title_ok = (
            True
            if self.support_titles_expected is None
            else self.support_titles_present == self.support_titles_expected
        )
        sentence_ok = (
            True
            if self.support_sentences_expected is None
            else self.support_sentences_present == self.support_sentences_expected
        )
        return (
            bool(self.task_id.strip())
            and self.passage_count == 10
            and self.question_occurrences == 1
            and title_ok
            and sentence_ok
            and not self.forbidden_public_fields
        )


__all__ = ["HotpotContextAudit"]
