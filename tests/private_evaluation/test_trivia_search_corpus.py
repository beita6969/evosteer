from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import skillev_private.direct_reference.trivia_search as corpus
from skillev_private.direct_reference.trivia_search import (
    DATABASE_FORMAT_V2,
    CodexDetailedKnowledgePlan,
    CodexKnowledgePlan,
    CorpusPassage,
    TriviaSearchDatabase,
    WikipediaClient,
    WikipediaPage,
    build_trivia_search_database,
    codex_detailed_output_schema,
    detailed_passages_for_task,
    generate_codex_knowledge_plan,
    load_frozen_trivia_questions,
    load_official_trivia_aliases_for_questions,
    remove_evaluator_labels,
    select_wikipedia_pages,
    select_wikipedia_pages_for_queries,
)


def _note() -> str:
    return (
        "This background surveys related people, places, institutions, dates, and works without "
        "identifying a requested final value. It records broad historical relationships that can "
        "guide later encyclopedia research, including neighboring events, alternate terminology, "
        "and the kinds of sources in which the relevant fact is normally documented. The material "
        "is deliberately indirect and intended only to help formulate useful retrieval queries."
    )


def test_codex_plan_rejects_direct_answer_wording() -> None:
    with pytest.raises(ValueError):
        CodexKnowledgePlan("task", "The answer is a private value. " + _note(), ("a", "b", "c"))


def test_codex_invocation_uses_ephemeral_stdin_and_validates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["command"] = command
        seen["input"] = kwargs["input"]
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(
            json.dumps({"background_note": _note(), "queries": ["one", "two", "three"]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(corpus.subprocess, "run", fake_run)
    plan = generate_codex_knowledge_plan(
        task_id="task-1",
        question="Licensed public question?",
        work_directory=tmp_path,
    )

    command = seen["command"]
    assert isinstance(command, list)
    assert "--ephemeral" in command
    assert "read-only" in command
    assert "Licensed public question?" not in " ".join(command)
    assert "Licensed public question?" in str(seen["input"])
    assert Path(command[command.index("--output-schema") + 1]).is_absolute()
    assert Path(command[command.index("--output-last-message") + 1]).is_absolute()
    assert plan.queries == ("one", "two", "three")
    assert (tmp_path / "codex-plan-stderr.log").is_file()


def test_codex_schema_uses_supported_structured_output_keywords() -> None:
    schema = corpus.codex_output_schema()
    queries = schema["properties"]["queries"]

    assert "uniqueItems" not in queries
    assert "uniqueItems" not in codex_detailed_output_schema()["properties"]["queries"]


def test_detailed_plan_and_passages_cover_dossier_entities_and_fuller_page() -> None:
    dossier = ("Related historical context and entity relationships. " * 30)[:1500]
    plan = CodexDetailedKnowledgePlan(
        "task",
        dossier,
        tuple(f"query {index}" for index in range(8)),
        tuple(f"entity {index}" for index in range(6)),
    )
    page = WikipediaPage(
        1,
        2,
        "2026-01-01T00:00:00Z",
        "Public page",
        "https://en.wikipedia.org/?curid=1",
        "encyclopedia content " * 2_000,
    )

    passages = detailed_passages_for_task(plan, (page,))

    assert any(row.source_type == "codex-dossier" for row in passages)
    assert any(row.source_type == "codex-entities" for row in passages)
    assert sum(row.source_type == "wikipedia" for row in passages) == 16


def test_frozen_question_loader_returns_public_question_without_answer(tmp_path: Path) -> None:
    population = [
        {
            "question": (
                f"[Public passage] Context that must be removed. Question: Public question {index}?"
            ),
            "answer": f"Private target {index}",
            "extra": {"source": "TriviaQA"},
        }
        for index in range(128)
    ]
    path = tmp_path / "iid.json"
    path.write_text(json.dumps(population), encoding="utf-8")

    questions = load_frozen_trivia_questions(path)

    assert questions[0] == ("skillflow-iid-v3:triviaqa:000", "Public question 0?")
    assert questions[-1] == ("skillflow-iid-v3:triviaqa:127", "Public question 127?")
    assert all("Public passage" not in question for _task_id, question in questions)
    assert all("Private target" not in question for _task_id, question in questions)


def test_official_alias_loader_exactly_joins_full_alias_arrays_by_question(
    tmp_path: Path,
) -> None:
    path = tmp_path / "official-trivia.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "question": "Public question one?",
                    "answer": {"aliases": ["Canonical One", "Alias One"]},
                },
                {
                    "question": "Public question two?",
                    "answer": {"aliases": ["Canonical Two"]},
                },
            ]
        ),
        path,
    )

    aliases = load_official_trivia_aliases_for_questions(
        path,
        {"task-1": "Public question one?", "task-2": "Public question two?"},
    )

    assert aliases == {
        "task-1": ("Canonical One", "Alias One"),
        "task-2": ("Canonical Two",),
    }
    with pytest.raises(ValueError):
        load_official_trivia_aliases_for_questions(
            path,
            {"missing": "Question absent from the official source?"},
        )


def test_database_search_is_task_partitioned_and_document_diverse(tmp_path: Path) -> None:
    path = tmp_path / "corpus.sqlite3"
    rows = (
        CorpusPassage("task-a", "a1", "doc-a", "wikipedia", "Alpha", "orbit public one"),
        CorpusPassage("task-a", "a2", "doc-a", "wikipedia", "Alpha", "orbit public two"),
        CorpusPassage("task-a", "a3", "doc-a", "wikipedia", "Alpha", "orbit public three"),
        CorpusPassage("task-a", "b1", "doc-b", "codex-knowledge", "Beta", "orbit beta"),
        CorpusPassage("task-b", "secret", "doc-c", "wikipedia", "Other", "orbit hidden"),
    )
    build_trivia_search_database(path, rows)

    with TriviaSearchDatabase(path) as database:
        hits = database.search("task-a", "orbit", limit=5, max_per_document=2)

    assert len(hits) == 3
    assert sum(hit.document_id == "doc-a" for hit in hits) == 2
    assert all(hit.passage_id != "secret" for hit in hits)


def test_evaluator_labels_are_removed_from_passage_text_and_title() -> None:
    passages = (
        CorpusPassage(
            "task-a",
            "a1",
            "doc-a",
            "wikipedia",
            "Alpha Target",
            "The alpha target appears in supporting context.",
        ),
    )

    sanitized = remove_evaluator_labels(passages, {"task-a": ("Alpha Target",)})

    assert sanitized[0].title == "[label removed]"
    assert "alpha target" not in sanitized[0].text.casefold()
    assert sanitized[0].source_type == "wikipedia"

    short = remove_evaluator_labels(
        (CorpusPassage("task-b", "b1", "doc-b", "wikipedia", "Data", "A data point"),),
        {"task-b": ("A",)},
    )
    assert short[0].title == "Data"
    assert short[0].text == "[label removed] data point"


def test_wikipedia_selection_deduplicates_and_caps_four_pages() -> None:
    class FakeWikipedia(WikipediaClient):
        def __init__(self) -> None:
            pass

        def search_page_ids(self, query: str, *, limit: int = 5) -> tuple[int, ...]:
            del query, limit
            return (1, 2, 3, 4, 5)

        def fetch_page(self, page_id: int) -> WikipediaPage:
            return WikipediaPage(
                page_id,
                page_id + 100,
                "2026-01-01T00:00:00Z",
                f"Page {page_id}",
                f"https://en.wikipedia.org/?curid={page_id}",
                "Public encyclopedia content.",
            )

    pages = select_wikipedia_pages(
        FakeWikipedia(),
        CodexKnowledgePlan("task", _note(), ("one", "two", "three")),
        maximum_pages=4,
    )

    assert [page.page_id for page in pages] == [1, 2, 3, 4]


def test_wikipedia_detailed_selection_round_robins_across_queries() -> None:
    class DiverseWikipedia(WikipediaClient):
        def __init__(self) -> None:
            pass

        def search_page_ids(self, query: str, *, limit: int = 5) -> tuple[int, ...]:
            assert limit == 5
            return {
                "a": (1, 2, 3),
                "b": (10, 11, 12),
                "c": (20, 21, 22),
            }[query]

        def fetch_page(self, page_id: int) -> WikipediaPage:
            return WikipediaPage(
                page_id,
                page_id + 100,
                "2026-01-01T00:00:00Z",
                f"Page {page_id}",
                f"https://en.wikipedia.org/?curid={page_id}",
                "Public encyclopedia content.",
            )

    pages = select_wikipedia_pages_for_queries(
        DiverseWikipedia(),
        ("a", "b", "c"),
        maximum_pages=5,
    )

    assert [page.page_id for page in pages] == [1, 10, 20, 2, 11]


def test_v2_database_returns_match_centered_snippet(tmp_path: Path) -> None:
    path = tmp_path / "detailed.sqlite3"
    prefix = "irrelevant " * 200
    build_trivia_search_database(
        path,
        (CorpusPassage("task", "p", "d", "wikipedia", "Title", prefix + "needle fact"),),
        database_format=DATABASE_FORMAT_V2,
    )

    with TriviaSearchDatabase(path, expected_format=DATABASE_FORMAT_V2) as database:
        hits = database.search("task", "needle", limit=1, max_per_document=1)

    assert len(hits) == 1
    assert "needle" in hits[0].snippet


def test_v2_database_supports_a_larger_bounded_evidence_window(tmp_path: Path) -> None:
    path = tmp_path / "detailed.sqlite3"
    evidence = "needle " + ("contextual public relation " * 100)
    build_trivia_search_database(
        path,
        (CorpusPassage("task", "p", "d", "wikipedia", "Title", evidence),),
        database_format=DATABASE_FORMAT_V2,
    )

    with TriviaSearchDatabase(
        path,
        expected_format=DATABASE_FORMAT_V2,
        snippet_token_window=96,
        maximum_snippet_characters=1_200,
    ) as database:
        hits = database.search("task", "needle", limit=1, max_per_document=1)

    assert len(hits) == 1
    assert 700 < len(hits[0].snippet) <= 1_200


@pytest.mark.parametrize(
    ("field", "value"),
    [("snippet_token_window", 7), ("maximum_snippet_characters", 4_097)],
)
def test_database_rejects_unbounded_presentation_limits(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    path = tmp_path / "detailed.sqlite3"
    build_trivia_search_database(
        path,
        (CorpusPassage("task", "p", "d", "wikipedia", "Title", "needle fact"),),
        database_format=DATABASE_FORMAT_V2,
    )

    with pytest.raises(ValueError):
        TriviaSearchDatabase(path, expected_format=DATABASE_FORMAT_V2, **{field: value})


def test_wikipedia_client_retries_api_error_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        {"error": {"code": "maxlag"}},
        {"query": {"search": []}},
    ]

    class Response:
        def __init__(self, value: object) -> None:
            self.value = value

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self.value).encode()

    def fake_urlopen(_request: object, *, timeout: float) -> Response:
        assert timeout == 1.0
        return Response(responses.pop(0))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(corpus.time, "sleep", lambda _seconds: None)
    client = WikipediaClient(
        user_agent="test@example.invalid",
        timeout_seconds=1.0,
        maximum_attempts=2,
    )

    assert client.search_page_ids("public query") == ()
    assert responses == []
