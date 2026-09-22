"""Trusted, scoped corpus broker. It never receives evaluator references."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache, partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from skillev.evaluation.corpus_search import (
    CORPUS_ID,
    DENSE_SEARCH_PROFILES,
    HYBRID_SEARCH_PROFILE,
    INPUT_PROFILE,
    PUBLIC_QUESTION_QUERY,
    RERANK_CANDIDATE_LIMITS,
    SEARCH_PROFILE,
    SNAPSHOT_DATE,
    CorpusSearchProfile,
)
from skillev.evaluation.input_metric_contracts import PublicTaskView
from skillev.evaluation.native_tool_calls import native_corpus_queries
from skillev.evaluation.sealed_candidates import CandidateJournal, EventOrigin

from .wikipedia_corpus import corpus_metadata, search_corpus

SEARCH_QUEUE_TIMEOUT_SECONDS = 120
DENSE_WORKER_TIMEOUT_SECONDS = 30
HYBRID_WORKER_TIMEOUT_SECONDS = 60
_search_slots_by_loop: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    WeakKeyDictionary()
)


def _search_slots() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    return _search_slots_by_loop.setdefault(loop, asyncio.Semaphore(4))


@lru_cache(maxsize=1)
def _search_pool() -> ProcessPoolExecutor:
    # The first full-corpus run overloaded SQLite with 32 concurrent threads.
    # Four CPU processes keep search bounded without reducing inference batch.
    # Only public search/ranking inputs enter these workers, never references.
    return ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"))


class CorpusSearchSession:
    def __init__(
        self,
        settings: dict[str, Any],
        journal: CandidateJournal,
        scope: tuple[str, str, str],
        *,
        public_question: str | None = None,
    ) -> None:
        self.profile = CorpusSearchProfile(**settings["public_profile"])
        if self.profile.reranker_query_source == PUBLIC_QUESTION_QUERY:
            if not public_question or not public_question.strip():
                raise ValueError("question-aware reranking requires the public question")
            self.ranking_query: str | None = public_question
        else:
            self.ranking_query = None
        self.path = Path(settings["index_path"])
        self.reranker_path = (
            Path(settings["reranker_path"]).resolve(strict=True)
            if self.profile.passage_ranking in RERANK_CANDIDATE_LIMITS
            else None
        )
        self.dense_paths = (
            (
                Path(settings["encoder_path"]).resolve(strict=True),
                Path(settings["dense_index_path"]).resolve(strict=True),
            )
            if self.profile.profile_id in DENSE_SEARCH_PROFILES
            else None
        )
        metadata = corpus_metadata(self.path)
        if (
            metadata.get("completed") != "true"
            or metadata.get("corpus_id") != self.profile.corpus_id
            or metadata.get("profile") != SEARCH_PROFILE
            or (
                self.profile.corpus_id == CORPUS_ID
                and metadata.get("snapshot_date") != SNAPSHOT_DATE
            )
        ):
            raise ValueError("the corpus index does not match the declared retrieval condition")
        self.journal, self.scope = journal, scope
        self.calls: dict[str, int] = {}
        self.query_count = 0
        self.journal.record(
            scope,
            "corpus-profile",
            {
                **asdict(self.profile),
                "index_metadata": metadata,
                "search_workers": 4,
                "lexical_sql_timeout_seconds": 30,
                "dense_queue_timeout_seconds": SEARCH_QUEUE_TIMEOUT_SECONDS,
                "dense_worker_timeout_seconds": self._worker_timeout,
            },
            origin=EventOrigin.ENVIRONMENT,
        )

    @property
    def _worker_timeout(self) -> float:
        return (
            HYBRID_WORKER_TIMEOUT_SECONDS
            if self.profile.profile_id == HYBRID_SEARCH_PROFILE
            else DENSE_WORKER_TIMEOUT_SECONDS
        )

    async def _dense_search(
        self, query: str, request: dict[str, Any]
    ) -> tuple[list[dict[str, object]], dict[str, Any]]:
        from .dpr_retrieval import search_dense_corpus

        assert self.dense_paths is not None
        search = search_dense_corpus
        if self.profile.profile_id == HYBRID_SEARCH_PROFILE:
            from .hybrid_retrieval import search_hybrid_corpus

            search = search_hybrid_corpus
        queued_at = time.monotonic()
        slots = _search_slots()
        # Batch32 previously spent the worker's entire 30s deadline waiting
        # behind four CPU processes. Bound admission separately from execution.
        await asyncio.wait_for(slots.acquire(), timeout=SEARCH_QUEUE_TIMEOUT_SECONDS)
        try:
            self.journal.record(
                self.scope,
                "corpus-search-worker-start",
                {
                    **request,
                    "queue_seconds": time.monotonic() - queued_at,
                    "worker_timeout_seconds": self._worker_timeout,
                },
                origin=EventOrigin.ENVIRONMENT,
            )
            return await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    _search_pool(),
                    partial(
                        search,
                        self.path,
                        query,
                        limit=self.profile.passages_per_query,
                        model_path=self.dense_paths[0],
                        index_path=self.dense_paths[1],
                    ),
                ),
                timeout=self._worker_timeout,
            )
        finally:
            slots.release()

    async def execute(
        self, query: str, *, response: str, call_id: str, participant: str, final_ready: bool
    ) -> dict[str, Any]:
        queries = native_corpus_queries(response) or ()
        query_index = self.calls.get(call_id, 0)
        if (
            participant != "owner"
            or final_ready
            or query_index >= len(queries)
            or queries[query_index] != query
        ):
            raise ValueError("corpus request is not the current unique owner's literal command")
        self.calls[call_id] = query_index + 1
        self.query_count += 1
        started = time.monotonic()
        request = {
            "call_id": call_id,
            "query": query,
            "query_index": query_index,
            "query_number": self.query_count,
        }
        self.journal.record(
            self.scope, "corpus-search-start", request, origin=EventOrigin.ENVIRONMENT
        )
        if self.query_count > self.profile.maximum_queries:
            result: dict[str, Any] = {
                "error": "corpus search budget exhausted",
                "remaining_queries": 0,
            }
        else:
            search = partial(
                search_corpus,
                self.path,
                query,
                limit=self.profile.passages_per_query,
                query_policy=self.profile.query_policy,
            )
            loop = asyncio.get_running_loop()
            if self.dense_paths is not None:
                passages, diagnostics = await self._dense_search(query, request)
                if self.profile.profile_id == HYBRID_SEARCH_PROFILE:
                    diagnostics = dict(diagnostics)
                    dense_cost = diagnostics.pop("dense_retrieval")
                    self.journal.record(
                        self.scope,
                        "corpus-hybrid-retrieval",
                        {**request, **diagnostics},
                        origin=EventOrigin.ENVIRONMENT,
                    )
                    diagnostics = dense_cost
                self.journal.record(
                    self.scope,
                    "corpus-dense-retrieval",
                    {**request, **diagnostics},
                    origin=EventOrigin.ENVIRONMENT,
                )
            elif self.reranker_path is None:
                passages = await loop.run_in_executor(_search_pool(), search)
            else:
                from .passage_reranker import search_ranked_corpus

                passages, diagnostics = await loop.run_in_executor(
                    _search_pool(),
                    partial(
                        search_ranked_corpus,
                        self.path,
                        query,
                        limit=self.profile.passages_per_query,
                        query_policy=self.profile.query_policy,
                        model_path=self.reranker_path,
                        ranking_query=self.ranking_query,
                        ranking_policy=self.profile.passage_ranking,
                    ),
                )
                self.journal.record(
                    self.scope,
                    "corpus-rerank",
                    {**request, **diagnostics},
                    origin=EventOrigin.ENVIRONMENT,
                )
            result = {
                "corpus_id": self.profile.corpus_id,
                "query": query,
                "passages": passages,
                "remaining_queries": self.profile.maximum_queries - self.query_count,
            }
        self.journal.record(
            self.scope,
            "corpus-search-result",
            {**request, "result": result, "elapsed_seconds": time.monotonic() - started},
            origin=EventOrigin.ENVIRONMENT,
        )
        return result


def corpus_session(
    config: dict[str, Any],
    entry: PublicTaskView,
    journal: CandidateJournal,
    scope: tuple[str, str, str],
) -> CorpusSearchSession | None:
    settings = config.get("corpus_retrieval", {}).get(entry.benchmark)
    if settings is None:
        if entry.input_profile == INPUT_PROFILE:
            raise ValueError("retrieval input has no configured corpus")
        return None
    if entry.benchmark != "nq-open" or entry.input_profile != INPUT_PROFILE:
        raise ValueError("corpus retrieval requires its separate NQ input condition")
    return CorpusSearchSession(
        settings, journal, scope, public_question=dict(entry.fields)["question"]
    )
