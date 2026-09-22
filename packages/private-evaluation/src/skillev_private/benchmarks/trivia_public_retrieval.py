"""Build final TriviaQA sessions without exposing aliases or scorer feedback."""

from __future__ import annotations

from dataclasses import dataclass

from skillev.evaluation.trivia_retrieval.config import RetrievalLimits
from skillev.evaluation.trivia_retrieval.session import TriviaQAPublicRetrievalSession

from .trivia_public_corpus import PrivatePublicCorpusBinding


@dataclass(frozen=True, slots=True)
class ProtocolV11TriviaSessionBuilder:
    corpus: PrivatePublicCorpusBinding
    limits: RetrievalLimits

    def build(self) -> TriviaQAPublicRetrievalSession:
        return TriviaQAPublicRetrievalSession(self.corpus.database_path, limits=self.limits)


__all__ = ["ProtocolV11TriviaSessionBuilder"]
