"""Deterministic SQLite FTS5 search/read session for public passages."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from skillev.evaluation.trivia_retrieval.config import RetrievalLimits

_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "with",
    }
)
_MAX_QUERY_TOKENS = 12


@dataclass(frozen=True, slots=True)
class SearchHit:
    passage_id: str
    document_id: str
    title: str
    snippet: str
    ranking_score: float


@dataclass(frozen=True, slots=True)
class PublicPassage:
    passage_id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class RetrievalObservation:
    operation: str
    query_or_passage_id: str
    results: tuple[SearchHit, ...] | PublicPassage
    remaining_searches: int
    remaining_reads: int


class TriviaQAPublicRetrievalSession:
    def __init__(self, database_path: Path, *, limits: RetrievalLimits | None = None) -> None:
        if not database_path.is_file():
            raise ValueError("public retrieval database does not exist")
        self._connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        self._limits = limits or RetrievalLimits()
        self._searches = 0
        self._reads = 0

    @staticmethod
    def _fts_query(query: str) -> str:
        raw_tokens = _TOKEN.findall(query.casefold())
        tokens = tuple(
            dict.fromkeys(
                token for token in raw_tokens if len(token) > 2 and token not in _STOPWORDS
            )
        )[:_MAX_QUERY_TOKENS]
        if not tokens:
            tokens = tuple(dict.fromkeys(raw_tokens))[:_MAX_QUERY_TOKENS]
        if not tokens:
            raise ValueError("search query must contain searchable text")
        return " OR ".join(f'"{token}"' for token in tokens)

    def search(self, query: str) -> RetrievalObservation:
        if self._searches >= self._limits.maximum_searches:
            raise RuntimeError("TriviaQA search budget exhausted")
        query = query.strip()
        if not query or len(query) > self._limits.maximum_query_characters:
            raise ValueError("TriviaQA search query is empty or too long")
        self._searches += 1
        rows = self._connection.execute(
            """
            SELECT passage_id, document_id, title,
                   snippet(passages, 3, '', '', ' … ', 24), rank
            FROM passages
            WHERE passages MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (self._fts_query(query), self._limits.hits_per_search),
        ).fetchall()
        hits = tuple(
            SearchHit(str(passage), str(document), str(title), str(snippet), float(score))
            for passage, document, title, snippet, score in rows
        )
        return RetrievalObservation(
            "search",
            query,
            hits,
            self._limits.maximum_searches - self._searches,
            self._limits.maximum_reads - self._reads,
        )

    def read(self, passage_id: str) -> RetrievalObservation:
        if self._reads >= self._limits.maximum_reads:
            raise RuntimeError("TriviaQA read budget exhausted")
        self._reads += 1
        row = self._connection.execute(
            "SELECT title, text FROM passages WHERE rowid = ("
            "SELECT passage_rowid FROM passage_rowids WHERE passage_id = ?)",
            (passage_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown public passage: {passage_id}")
        passage = PublicPassage(passage_id, str(row[0]), str(row[1]))
        return RetrievalObservation(
            "read",
            passage_id,
            passage,
            self._limits.maximum_searches - self._searches,
            self._limits.maximum_reads - self._reads,
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> TriviaQAPublicRetrievalSession:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
