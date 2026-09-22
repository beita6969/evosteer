import sqlite3
from pathlib import Path

import pytest

from skillev.evaluation.trivia_retrieval.config import RetrievalLimits
from skillev.evaluation.trivia_retrieval.session import TriviaQAPublicRetrievalSession


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE VIRTUAL TABLE passages USING fts5("
        "passage_id UNINDEXED, document_id UNINDEXED, title, text)"
    )
    cursors: list[tuple[str, int]] = []
    for row in (
        ("p1", "d1", "Paris", "Paris is the capital city of France."),
        ("p2", "d2", "France", "France is a country in Western Europe."),
    ):
        cursor = connection.execute("INSERT INTO passages VALUES (?, ?, ?, ?)", row)
        assert cursor.lastrowid is not None
        cursors.append((row[0], cursor.lastrowid))
    connection.execute(
        "CREATE TABLE passage_rowids("
        "passage_id TEXT PRIMARY KEY, passage_rowid INTEGER NOT NULL UNIQUE) WITHOUT ROWID"
    )
    connection.executemany(
        "INSERT INTO passage_rowids VALUES (?, ?)",
        cursors,
    )
    connection.commit()
    connection.close()


def test_search_and_read_are_deterministic_and_budgeted(tmp_path: Path) -> None:
    path = tmp_path / "public.sqlite"
    _database(path)
    with TriviaQAPublicRetrievalSession(
        path, limits=RetrievalLimits(maximum_searches=1, maximum_reads=1)
    ) as session:
        search = session.search("What is the capital of France?")
        assert search.operation == "search"
        assert search.results
        passage_id = search.results[0].passage_id  # type: ignore[index]
        read = session.read(passage_id)
        assert read.operation == "read"
        with pytest.raises(RuntimeError):
            session.search("again")
        with pytest.raises(RuntimeError):
            session.read(passage_id)


def test_empty_search_is_a_valid_empty_observation(tmp_path: Path) -> None:
    path = tmp_path / "public.sqlite"
    _database(path)
    with TriviaQAPublicRetrievalSession(path) as session:
        assert session.search("nonexistenttoken").results == ()
