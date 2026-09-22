"""Answer-free action dispatcher for the TriviaQA public retrieval lane."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.evaluation.trivia_retrieval.session import (
    RetrievalObservation,
    TriviaQAPublicRetrievalSession,
)


@dataclass(frozen=True, slots=True)
class TriviaRetrievalAction:
    name: str
    value: str


@dataclass(frozen=True, slots=True)
class TriviaRetrievalStep:
    observation: RetrievalObservation | None
    answer: str | None
    terminal: bool


def dispatch_retrieval_action(
    session: TriviaQAPublicRetrievalSession, action: TriviaRetrievalAction
) -> TriviaRetrievalStep:
    if action.name == "search":
        return TriviaRetrievalStep(session.search(action.value), None, False)
    if action.name == "read":
        return TriviaRetrievalStep(session.read(action.value), None, False)
    if action.name == "complete":
        answer = action.value.strip()
        if not answer:
            raise ValueError("TriviaQA completion must be non-empty")
        return TriviaRetrievalStep(None, answer, True)
    raise ValueError(f"unsupported TriviaQA retrieval action: {action.name}")
