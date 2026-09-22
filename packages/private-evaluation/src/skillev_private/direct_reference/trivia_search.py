"""Private, answer-free corpus preparation and search for TriviaQA."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from skillev.evaluation.direct_baseline.search_augmented import TriviaSearchHit

from .skillflow_iid import (
    parse_skillflow_iid_record,
    parse_skillflow_trivia_answers,
    render_skillflow_trivia_question_only,
)

DATABASE_FORMAT = "skillev-triviaqa-search-corpus@1"
DATABASE_FORMAT_V2 = "skillev-triviaqa-search-corpus-detailed@2"
CODEX_PLAN_PROFILE = "triviaqa-codex-knowledge-plan@1"
CODEX_DETAILED_PLAN_PROFILE = "triviaqa-codex-detailed-research-plan@2"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
_DIRECT_ANSWER = re.compile(r"(?i)\b(?:the\s+)?(?:correct\s+|final\s+)?answer\s+(?:is|:)\b")
_QUERY_TERM = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class CodexKnowledgePlan:
    task_id: str
    background_note: str
    queries: tuple[str, str, str]

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not 300 <= len(self.background_note) <= 800:
            raise ValueError("Codex plan identity or background-note length is invalid")
        if _DIRECT_ANSWER.search(self.background_note):
            raise ValueError("Codex background note uses direct-answer wording")
        if len(set(self.queries)) != 3 or any(
            not query.strip() or len(query) > 160 for query in self.queries
        ):
            raise ValueError("Codex plan requires three distinct bounded queries")


@dataclass(frozen=True, slots=True)
class CodexDetailedKnowledgePlan:
    task_id: str
    research_dossier: str
    queries: tuple[str, ...]
    key_entities: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not 1_200 <= len(self.research_dossier) <= 3_000:
            raise ValueError("detailed Codex plan identity or dossier length is invalid")
        if _DIRECT_ANSWER.search(self.research_dossier):
            raise ValueError("detailed Codex dossier uses direct-answer wording")
        if len(self.queries) != 8 or len(set(self.queries)) != 8:
            raise ValueError("detailed Codex plan requires eight distinct queries")
        if any(not query.strip() or len(query) > 180 for query in self.queries):
            raise ValueError("detailed Codex queries must be non-empty and bounded")
        if not 6 <= len(self.key_entities) <= 12 or len(set(self.key_entities)) != len(
            self.key_entities
        ):
            raise ValueError("detailed Codex plan requires six to twelve distinct entities")
        if any(not entity.strip() or len(entity) > 120 for entity in self.key_entities):
            raise ValueError("detailed Codex entities must be non-empty and bounded")


@dataclass(frozen=True, slots=True)
class WikipediaPage:
    page_id: int
    revision_id: int
    revision_timestamp: str
    title: str
    canonical_url: str
    text: str

    def __post_init__(self) -> None:
        if min(self.page_id, self.revision_id) < 1:
            raise ValueError("Wikipedia page and revision IDs must be positive")
        if any(
            not value.strip()
            for value in (
                self.revision_timestamp,
                self.title,
                self.canonical_url,
                self.text,
            )
        ):
            raise ValueError("Wikipedia page fields must be non-empty")


@dataclass(frozen=True, slots=True)
class CorpusPassage:
    task_id: str
    passage_id: str
    document_id: str
    source_type: str
    title: str
    text: str
    source_url: str | None = None
    page_id: int | None = None
    revision_id: int | None = None
    revision_timestamp: str | None = None


def load_frozen_trivia_questions(path: Path) -> tuple[tuple[str, str], ...]:
    """Load only the public TriviaQA question field from the frozen IID panel."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not list:
        raise ValueError("SkillFlow IID population must be an array")
    result: list[tuple[str, str]] = []
    for position, raw in enumerate(value):
        record = parse_skillflow_iid_record(raw, source_position=position)
        if record.source != "TriviaQA":
            continue
        task_id = f"skillflow-iid-v3:triviaqa:{len(result):03d}"
        result.append((task_id, render_skillflow_trivia_question_only(record)))
    if len(result) != 128 or len({task_id for task_id, _ in result}) != 128:
        raise ValueError("TriviaQA frozen population must contain 128 unique questions")
    return tuple(result)


def load_frozen_trivia_labels(path: Path) -> dict[str, tuple[str, ...]]:
    """Load private aliases only for the final corpus-leakage removal pass."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not list:
        raise ValueError("SkillFlow IID population must be an array")
    result: dict[str, tuple[str, ...]] = {}
    trivia_position = 0
    for source_position, raw in enumerate(value):
        record = parse_skillflow_iid_record(raw, source_position=source_position)
        if record.source != "TriviaQA":
            continue
        task_id = f"skillflow-iid-v3:triviaqa:{trivia_position:03d}"
        result[task_id] = parse_skillflow_trivia_answers(record.answer)
        trivia_position += 1
    if len(result) != 128:
        raise ValueError("TriviaQA frozen population must contain 128 label sets")
    return result


def load_official_trivia_aliases_for_questions(
    path: Path,
    questions_by_task: dict[str, str],
) -> dict[str, tuple[str, ...]]:
    """Join complete official aliases to frozen public questions for scoring only.

    The released SkillFlow answer wire stores at most five aliases and often only
    the canonical value. Official TriviaQA evaluation takes the maximum over the
    complete alias array. This loader reads that array from the pinned official
    no-context validation parquet, performs an exact question-only join, and
    returns verifier material that must never enter candidate context.
    """

    if not questions_by_task or any(
        type(task_id) is not str
        or not task_id.strip()
        or type(question) is not str
        or not question.strip()
        for task_id, question in questions_by_task.items()
    ):
        raise ValueError("official TriviaQA alias join requires public task questions")
    rows = pq.read_table(path, columns=["question", "answer"]).to_pylist()
    by_question: dict[str, list[tuple[str, ...]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("official TriviaQA row must be an object")
        question = row.get("question")
        answer = row.get("answer")
        if type(question) is not str or not question.strip() or not isinstance(answer, dict):
            raise ValueError("official TriviaQA row lacks question or answer")
        raw_aliases = answer.get("aliases")
        if not isinstance(raw_aliases, list):
            raise ValueError("official TriviaQA answer lacks aliases")
        aliases = tuple(
            dict.fromkeys(
                alias.strip() for alias in raw_aliases if isinstance(alias, str) and alias.strip()
            )
        )
        if not aliases:
            raise ValueError("official TriviaQA answer aliases are empty")
        by_question.setdefault(question.strip(), []).append(aliases)
    result: dict[str, tuple[str, ...]] = {}
    for task_id, question in questions_by_task.items():
        matches = by_question.get(question.strip(), [])
        if len(matches) != 1:
            raise ValueError("frozen TriviaQA question has no unique official alias row")
        result[task_id] = matches[0]
    if set(result) != set(questions_by_task):
        raise RuntimeError("official TriviaQA alias join changed task coverage")
    return result


def remove_evaluator_labels(
    passages: tuple[CorpusPassage, ...], labels_by_task: dict[str, tuple[str, ...]]
) -> tuple[CorpusPassage, ...]:
    """Remove literal evaluator aliases before any passage can reach the model."""

    result: list[CorpusPassage] = []
    for passage in passages:
        labels = labels_by_task.get(passage.task_id)
        if labels is None:
            raise ValueError("corpus passage has no task-local evaluator labels")
        text = passage.text
        title = passage.title
        patterns: list[re.Pattern[str]] = []
        for label in sorted(labels, key=len, reverse=True):
            pattern = re.compile(
                rf"(?<!\w){re.escape(label)}(?!\w)",
                flags=re.IGNORECASE,
            )
            patterns.append(pattern)
            text = pattern.sub("[label removed]", text)
            title = pattern.sub("[label removed]", title)
        if any(pattern.search(title + "\n" + text) for pattern in patterns):
            raise RuntimeError("evaluator-label removal did not isolate the corpus")
        result.append(
            CorpusPassage(
                passage.task_id,
                passage.passage_id,
                passage.document_id,
                passage.source_type,
                title,
                text,
                passage.source_url,
                passage.page_id,
                passage.revision_id,
                passage.revision_timestamp,
            )
        )
    return tuple(result)


def codex_output_schema() -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["background_note", "queries"],
        "properties": {
            "background_note": {"type": "string", "minLength": 300, "maxLength": 800},
            "queries": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "string", "minLength": 1, "maxLength": 160},
            },
        },
    }


def codex_detailed_output_schema() -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["research_dossier", "queries", "key_entities"],
        "properties": {
            "research_dossier": {"type": "string", "minLength": 1200, "maxLength": 3000},
            "queries": {
                "type": "array",
                "minItems": 8,
                "maxItems": 8,
                "items": {"type": "string", "minLength": 1, "maxLength": 180},
            },
            "key_entities": {
                "type": "array",
                "minItems": 6,
                "maxItems": 12,
                "items": {"type": "string", "minLength": 1, "maxLength": 120},
            },
        },
    }


def generate_codex_knowledge_plan(
    *,
    task_id: str,
    question: str,
    work_directory: Path,
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "xhigh",
    timeout_seconds: float = 900.0,
) -> CodexKnowledgePlan:
    """Invoke an ephemeral Codex process with one public question through stdin."""

    if not task_id.strip() or not question.strip():
        raise ValueError("Codex plan requires a task ID and public question")
    work_directory.mkdir(parents=True, exist_ok=True)
    work_directory = work_directory.resolve()
    schema_path = work_directory / "codex-plan-schema.json"
    output_path = work_directory / "codex-plan-output.json"
    schema_path.write_text(json.dumps(codex_output_schema(), indent=2) + "\n", encoding="utf-8")
    prompt = (
        "You are preparing an answer-free private retrieval corpus for one TriviaQA question. "
        "Use your own general knowledge, but do not state or guess a final answer and do not use "
        "phrases such as 'the answer is', 'correct answer', or 'final answer'. Write a 300-800 "
        "character background note containing related entities, dates, places, works, or relations "
        "that could support later research indirectly. Also provide exactly three concise English "
        "Wikipedia search queries. Treat the question independently. Return only the requested "
        "JSON. The evaluator target and accepted aliases are unavailable.\n\n"
        f"Public question:\n{question}"
    )
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--sandbox",
        "read-only",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "-",
    ]
    try:
        completed = subprocess.run(  # noqa: S603 -- fixed Codex executable and literal flags
            command,
            input=prompt,
            text=True,
            cwd=work_directory,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Codex knowledge-plan invocation failed") from exc
    (work_directory / "codex-plan-stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError("Codex knowledge-plan invocation returned no valid output")
    try:
        value = json.loads(output_path.read_text(encoding="utf-8"))
        if type(value) is not dict or set(value) != {"background_note", "queries"}:
            raise ValueError
        raw_queries = value["queries"]
        if type(raw_queries) is not list or len(raw_queries) != 3:
            raise ValueError
        queries = tuple(str(item).strip() for item in raw_queries)
        if len(queries) != 3:
            raise ValueError
        return CodexKnowledgePlan(
            task_id,
            str(value["background_note"]).strip(),
            queries,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError("Codex knowledge plan has an incompatible shape") from exc


def generate_codex_detailed_knowledge_plan(
    *,
    task_id: str,
    question: str,
    work_directory: Path,
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "xhigh",
    timeout_seconds: float = 900.0,
) -> CodexDetailedKnowledgePlan:
    """Build a richer answer-blind research dossier from one public question."""

    if not task_id.strip() or not question.strip():
        raise ValueError("detailed Codex plan requires a task ID and public question")
    work_directory.mkdir(parents=True, exist_ok=True)
    work_directory = work_directory.resolve()
    schema_path = work_directory / "codex-detailed-plan-schema.json"
    output_path = work_directory / "codex-detailed-plan-output.json"
    schema_path.write_text(
        json.dumps(codex_detailed_output_schema(), indent=2) + "\n",
        encoding="utf-8",
    )
    prompt = (
        "Prepare a detailed research dossier for one public TriviaQA query. Only the public query "
        "is supplied; evaluation labels and aliases are unavailable. Do not resolve the requested "
        "value or make a concluding claim. Write 1200-3000 characters of dense background "
        "covering entity disambiguation, chronology, geography, organizations, people, works, "
        "titles, alternate terminology, and relationships useful for later encyclopedia research. "
        "Include relevant contextual facts and plausible interpretations without concluding the "
        "requested value. Provide exactly eight distinct, concise English Wikipedia search queries "
        "that approach the question from different entity and relation angles, plus six to twelve "
        "key entity strings. Return only the requested JSON.\n\n"
        f"Public question:\n{question}"
    )
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--sandbox",
        "read-only",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "-",
    ]
    try:
        completed = subprocess.run(  # noqa: S603 -- fixed Codex executable and literal flags
            command,
            input=prompt,
            text=True,
            cwd=work_directory,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("detailed Codex knowledge-plan invocation failed") from exc
    (work_directory / "codex-detailed-plan-stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError("detailed Codex knowledge-plan invocation returned no valid output")
    try:
        value = json.loads(output_path.read_text(encoding="utf-8"))
        if type(value) is not dict or set(value) != {
            "research_dossier",
            "queries",
            "key_entities",
        }:
            raise ValueError
        raw_queries = value["queries"]
        raw_entities = value["key_entities"]
        if type(raw_queries) is not list or type(raw_entities) is not list:
            raise ValueError
        return CodexDetailedKnowledgePlan(
            task_id,
            str(value["research_dossier"]).strip(),
            tuple(str(item).strip() for item in raw_queries),
            tuple(str(item).strip() for item in raw_entities),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError("detailed Codex knowledge plan has an incompatible shape") from exc


class WikipediaClient:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float = 60.0,
        maximum_attempts: int = 4,
    ) -> None:
        if not user_agent.strip() or timeout_seconds <= 0 or maximum_attempts < 1:
            raise ValueError("Wikipedia client configuration is invalid")
        self._user_agent = user_agent
        self._timeout_seconds = timeout_seconds
        self._maximum_attempts = maximum_attempts
        self._next_request_at = 0.0

    def search_page_ids(self, query: str, *, limit: int = 5) -> tuple[int, ...]:
        value = self._request(
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "list": "search",
                "srnamespace": "0",
                "srlimit": str(limit),
                "srsearch": query,
            }
        )
        rows = cast(dict[str, object], value).get("query")
        if type(rows) is not dict or type(rows.get("search")) is not list:
            raise RuntimeError("Wikipedia search response is incompatible")
        result: list[int] = []
        for row in cast(list[object], rows["search"]):
            if type(row) is dict and type(row.get("pageid")) is int:
                result.append(cast(int, row["pageid"]))
        return tuple(result)

    def fetch_page(self, page_id: int) -> WikipediaPage:
        value = self._request(
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "pageids": str(page_id),
                "prop": "extracts|info|revisions",
                "explaintext": "1",
                "inprop": "url",
                "rvlimit": "1",
                "rvprop": "ids|timestamp",
            }
        )
        query = cast(dict[str, object], value).get("query")
        if type(query) is not dict or type(query.get("pages")) is not list:
            raise RuntimeError("Wikipedia page response is incompatible")
        pages = cast(list[object], query["pages"])
        if len(pages) != 1 or type(pages[0]) is not dict:
            raise RuntimeError("Wikipedia page response has no unique page")
        page = cast(dict[str, object], pages[0])
        revisions = page.get("revisions")
        if type(revisions) is not list or len(revisions) != 1 or type(revisions[0]) is not dict:
            raise RuntimeError("Wikipedia page response lacks a revision")
        revision = cast(dict[str, object], revisions[0])
        response_page_id = page.get("pageid")
        revision_id = revision.get("revid")
        text_fields = (
            revision.get("timestamp"),
            page.get("title"),
            page.get("fullurl"),
            page.get("extract"),
        )
        if (
            type(response_page_id) is not int
            or type(revision_id) is not int
            or any(type(value) is not str for value in text_fields)
        ):
            raise RuntimeError("Wikipedia page fields are incompatible")
        timestamp, title, full_url, extract = cast(tuple[str, str, str, str], text_fields)
        return WikipediaPage(response_page_id, revision_id, timestamp, title, full_url, extract)

    def _request(self, parameters: dict[str, str]) -> object:
        url = WIKIPEDIA_API + "?" + urllib.parse.urlencode(parameters)
        for attempt in range(1, self._maximum_attempts + 1):
            delay = self._next_request_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_request_at = time.monotonic() + 0.2
            request = urllib.request.Request(  # noqa: S310 -- fixed Wikimedia endpoint
                url,
                headers={"User-Agent": self._user_agent},
            )
            try:
                with urllib.request.urlopen(  # noqa: S310 -- fixed Wikimedia endpoint
                    request, timeout=self._timeout_seconds
                ) as response:
                    value = json.loads(response.read())
            except (OSError, TimeoutError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
                if attempt == self._maximum_attempts:
                    raise RuntimeError("Wikipedia API request failed") from exc
                retry_after = 0.5 * (2 ** (attempt - 1))
                if isinstance(exc, urllib.error.HTTPError):
                    header = exc.headers.get("Retry-After")
                    if header is not None and header.isdecimal():
                        retry_after = max(retry_after, float(header))
                time.sleep(retry_after)
                continue
            if type(value) is dict and value.get("error") is not None:
                if attempt == self._maximum_attempts:
                    raise RuntimeError("Wikipedia API returned an error response")
                time.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            return value
        raise AssertionError("unreachable")


def select_wikipedia_pages(
    client: WikipediaClient,
    plan: CodexKnowledgePlan,
    *,
    maximum_pages: int = 4,
) -> tuple[WikipediaPage, ...]:
    return select_wikipedia_pages_for_queries(
        client,
        plan.queries,
        maximum_pages=maximum_pages,
    )


def select_wikipedia_pages_for_queries(
    client: WikipediaClient,
    queries: tuple[str, ...],
    *,
    maximum_pages: int,
    page_cache: dict[int, WikipediaPage] | None = None,
) -> tuple[WikipediaPage, ...]:
    """Select diverse pages round-robin across answer-blind planner queries."""

    if not queries or maximum_pages < 1:
        raise ValueError("Wikipedia selection requires queries and a positive page budget")
    result_lists: list[tuple[int, ...]] = []
    for query in queries:
        try:
            matches = client.search_page_ids(query, limit=5)
        except RuntimeError:
            continue
        result_lists.append(matches)
    page_ids: list[int] = []
    for rank in range(5):
        for matches in result_lists:
            if rank < len(matches) and matches[rank] not in page_ids:
                page_ids.append(matches[rank])
    pages: list[WikipediaPage] = []
    for page_id in page_ids:
        page = page_cache.get(page_id) if page_cache is not None else None
        if page is None:
            try:
                page = client.fetch_page(page_id)
            except RuntimeError:
                continue
            if page_cache is not None:
                page_cache[page_id] = page
        if page.text.strip():
            pages.append(page)
        if len(pages) == maximum_pages:
            break
    if not pages:
        raise RuntimeError("TriviaQA corpus task has no usable Wikipedia page")
    return tuple(pages)


def passages_for_task(
    plan: CodexKnowledgePlan,
    pages: tuple[WikipediaPage, ...],
    *,
    chunk_characters: int = 1_200,
    overlap_characters: int = 150,
    maximum_chunks_per_page: int = 6,
) -> tuple[CorpusPassage, ...]:
    passages = [
        CorpusPassage(
            plan.task_id,
            f"{plan.task_id}:codex:0",
            f"{plan.task_id}:codex",
            "codex-knowledge",
            "Codex background knowledge",
            plan.background_note,
        )
    ]
    for page in pages:
        chunks = _chunks(
            page.text,
            size=chunk_characters,
            overlap=overlap_characters,
            maximum=maximum_chunks_per_page,
        )
        for index, chunk in enumerate(chunks):
            document_id = f"{plan.task_id}:wiki:{page.page_id}:{page.revision_id}"
            passages.append(
                CorpusPassage(
                    plan.task_id,
                    f"{document_id}:chunk:{index}",
                    document_id,
                    "wikipedia",
                    page.title,
                    chunk,
                    page.canonical_url,
                    page.page_id,
                    page.revision_id,
                    page.revision_timestamp,
                )
            )
    return tuple(passages)


def detailed_passages_for_task(
    plan: CodexDetailedKnowledgePlan,
    pages: tuple[WikipediaPage, ...],
    *,
    chunk_characters: int = 1_600,
    overlap_characters: int = 200,
    maximum_chunks_per_page: int = 16,
) -> tuple[CorpusPassage, ...]:
    passages: list[CorpusPassage] = []
    for index, chunk in enumerate(
        _chunks(plan.research_dossier, size=1_200, overlap=150, maximum=4)
    ):
        passages.append(
            CorpusPassage(
                plan.task_id,
                f"{plan.task_id}:codex-dossier:{index}",
                f"{plan.task_id}:codex-dossier",
                "codex-dossier",
                "Codex answer-blind research dossier",
                chunk,
            )
        )
    passages.append(
        CorpusPassage(
            plan.task_id,
            f"{plan.task_id}:codex-entities:0",
            f"{plan.task_id}:codex-entities",
            "codex-entities",
            "Codex research entities",
            "; ".join(plan.key_entities),
        )
    )
    for page in pages:
        chunks = _chunks(
            page.text,
            size=chunk_characters,
            overlap=overlap_characters,
            maximum=maximum_chunks_per_page,
        )
        for index, chunk in enumerate(chunks):
            document_id = f"{plan.task_id}:wiki:{page.page_id}:{page.revision_id}"
            passages.append(
                CorpusPassage(
                    plan.task_id,
                    f"{document_id}:chunk:{index}",
                    document_id,
                    "wikipedia",
                    page.title,
                    chunk,
                    page.canonical_url,
                    page.page_id,
                    page.revision_id,
                    page.revision_timestamp,
                )
            )
    return tuple(passages)


def _chunks(text: str, *, size: int, overlap: int, maximum: int) -> tuple[str, ...]:
    if not 0 <= overlap < size or min(size, maximum) < 1:
        raise ValueError("chunk configuration is invalid")
    normalized = " ".join(text.split())
    result: list[str] = []
    start = 0
    while start < len(normalized) and len(result) < maximum:
        end = min(len(normalized), start + size)
        chunk = normalized[start:end].strip()
        if chunk:
            result.append(chunk)
        if end == len(normalized):
            break
        start = end - overlap
    return tuple(result)


def build_trivia_search_database(
    path: Path,
    passages: tuple[CorpusPassage, ...],
    *,
    database_format: str = DATABASE_FORMAT,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    if not passages:
        raise ValueError("Trivia search database requires passages")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            "PRAGMA journal_mode=DELETE;"
            "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE passages("
            "task_id TEXT NOT NULL, passage_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, "
            "source_type TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL, "
            "source_url TEXT, page_id INTEGER, revision_id INTEGER, revision_timestamp TEXT);"
            "CREATE INDEX passages_task_id ON passages(task_id);"
            "CREATE VIRTUAL TABLE passage_fts USING fts5("
            "task_id UNINDEXED, title, text, content='passages', content_rowid='rowid', "
            "tokenize='unicode61 remove_diacritics 2');"
        )
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("format", database_format))
        connection.executemany(
            "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row.task_id,
                    row.passage_id,
                    row.document_id,
                    row.source_type,
                    row.title,
                    row.text,
                    row.source_url,
                    row.page_id,
                    row.revision_id,
                    row.revision_timestamp,
                )
                for row in passages
            ],
        )
        connection.execute(
            "INSERT INTO passage_fts(rowid, task_id, title, text) "
            "SELECT rowid, task_id, title, text FROM passages"
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError("Trivia search database integrity check failed")
    finally:
        connection.close()


class TriviaSearchDatabase:
    def __init__(
        self,
        path: Path,
        *,
        expected_format: str = DATABASE_FORMAT,
        snippet_token_window: int = 48,
        maximum_snippet_characters: int = 700,
    ) -> None:
        if type(snippet_token_window) is not int or not 8 <= snippet_token_window <= 256:
            raise ValueError("snippet_token_window must be an integer from 8 through 256")
        if (
            type(maximum_snippet_characters) is not int
            or not 128 <= maximum_snippet_characters <= 4_096
        ):
            raise ValueError("maximum_snippet_characters must be an integer from 128 through 4096")
        resolved = path.resolve(strict=True)
        self._connection = sqlite3.connect(f"file:{resolved}?mode=ro&immutable=1", uri=True)
        self._connection.execute("PRAGMA query_only=ON")
        row = self._connection.execute("SELECT value FROM metadata WHERE key='format'").fetchone()
        if row != (expected_format,):
            self._connection.close()
            raise ValueError("Trivia search database format is incompatible")
        self._format = expected_format
        self._snippet_token_window = snippet_token_window
        self._maximum_snippet_characters = maximum_snippet_characters

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> TriviaSearchDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def task_ids(self) -> tuple[str, ...]:
        rows = self._connection.execute(
            "SELECT DISTINCT task_id FROM passages ORDER BY task_id"
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def search(
        self,
        task_id: str,
        query: str,
        *,
        limit: int,
        max_per_document: int,
    ) -> tuple[TriviaSearchHit, ...]:
        terms = tuple(dict.fromkeys(_QUERY_TERM.findall(query)))
        if not terms:
            return ()
        match = " OR ".join(f'"{term}"' for term in terms)
        try:
            text_expression = (
                f"snippet(passage_fts, 2, '', '', ' … ', {self._snippet_token_window})"
                if self._format == DATABASE_FORMAT_V2
                else "p.text"
            )
            rows = self._connection.execute(
                f"SELECT p.passage_id, p.document_id, p.title, {text_expression}, "  # noqa: S608
                "bm25(passage_fts, 5.0, 1.0) AS score "
                "FROM passage_fts JOIN passages AS p ON p.rowid=passage_fts.rowid "
                "WHERE passage_fts MATCH ? AND p.task_id=? "
                "ORDER BY score ASC, p.passage_id ASC LIMIT ?",
                (match, task_id, max(limit * 8, limit)),
            ).fetchall()
        except sqlite3.Error as exc:
            raise RuntimeError("Trivia search query failed") from exc
        counts: dict[str, int] = {}
        selected: list[TriviaSearchHit] = []
        for passage_id, document_id, title, text, _score in rows:
            document = str(document_id)
            if counts.get(document, 0) >= max_per_document:
                continue
            counts[document] = counts.get(document, 0) + 1
            selected.append(
                TriviaSearchHit(
                    str(passage_id),
                    document,
                    str(title),
                    _snippet(str(text), maximum=self._maximum_snippet_characters),
                    len(selected) + 1,
                )
            )
            if len(selected) == limit:
                break
        return tuple(selected)


def _snippet(text: str, maximum: int = 700) -> str:
    return text if len(text) <= maximum else text[:maximum].rstrip() + "…"
