from __future__ import annotations

import asyncio
import http.client
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import urlsplit

from skillev.contracts.canonical import stable_hash

if TYPE_CHECKING:
    from .text_tools import TextToolsConfig

RETRIEVAL_URL_ENV = "EVOSTEER_RETRIEVAL_URL"
RETRIEVAL_IDENTITY_ENV = "EVOSTEER_RETRIEVAL_IDENTITY"
SERVICE_KIND = "wiki18-e5-base-v2-exact-fp16"
SERVICE_FORMAT = "evosteer-wiki18-e5-retrieval@1"
WIKI18_PASSAGES = 21_015_324
OUTAGE_SECONDS = 900.0
_OUTAGE_PAUSE_SECONDS = (2.0, 30.0)
T = TypeVar("T")


class RetrievalServiceError(RuntimeError):
    pass


class _ServiceUnreachable(RetrievalServiceError):
    pass


class RetrievalIdentityError(RetrievalServiceError):
    pass


class RetrievalQueryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SearchHit:
    rank: int
    doc_id: str
    title: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    top_k: int
    hits: tuple[SearchHit, ...]
    service_identity: str
    elapsed_seconds: float

    def to_value(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "query": self.query,
            "top_k": self.top_k,
            "ids": [hit.doc_id for hit in self.hits],
            "scores": [hit.score for hit in self.hits],
            "service_identity": self.service_identity,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }
        return value


def format_search_r1(hits: tuple[SearchHit, ...], passage_chars: int | None = None) -> str:
    lines = []
    for index, hit in enumerate(hits, 1):
        text = hit.text
        if passage_chars is not None and len(text) > passage_chars:
            text = text[:passage_chars].rstrip() + " ..."
        lines.append(f'Doc {index}(Title: "{hit.title}") {text}\n')
    return "".join(lines)


def service_identity(passages: int, top_k: int) -> str:
    value: dict[str, Any] = {
        "service": SERVICE_KIND,
        "format": SERVICE_FORMAT,
        "passages": passages,
        "top_k": top_k,
    }
    return str(stable_hash(value))


def expected_identity_for(
    text_tools: Mapping[str, Any] | TextToolsConfig,
    *,
    passages: int = WIKI18_PASSAGES,
) -> str:
    from .text_tools import TextToolsConfig as Config

    settings = text_tools if isinstance(text_tools, Config) else Config.from_value(text_tools)
    return service_identity(passages, settings.search_top_k)


class RetrievalClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        top_k: int = 3,
        timeout_seconds: float = 30.0,
        retries: int = 3,
        expected_identity: str | None = None,
        max_workers: int = 64,
        outage_seconds: float = OUTAGE_SECONDS,
    ) -> None:
        raw = base_url or os.environ.get(RETRIEVAL_URL_ENV)
        if not raw:
            raise ValueError(f"set {RETRIEVAL_URL_ENV} to the retrieval service (http://host:port)")
        url = urlsplit(raw)
        if url.scheme != "http" or not url.hostname or url.path not in {"", "/"}:
            raise ValueError(f"retrieval URL must be http://host:port, got {raw!r}")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if isinstance(timeout_seconds, bool) or not timeout_seconds > 0:
            raise ValueError("timeout must be positive")
        if type(retries) is not int or retries < 0:
            raise ValueError("retries must be a nonnegative integer")
        if (
            isinstance(outage_seconds, bool)
            or not isinstance(outage_seconds, int | float)
            or not 0 <= outage_seconds < math.inf
        ):
            raise ValueError("outage_seconds must be a nonnegative number")
        if expected_identity is not None and (
            type(expected_identity) is not str or not expected_identity.strip()
        ):
            raise ValueError("an expected retrieval identity must be nonempty text")
        self._host, self._port = url.hostname, url.port or 80
        self.base_url = f"http://{self._host}:{self._port}"
        self.top_k = top_k
        self.timeout_seconds = float(timeout_seconds)
        self.retries = retries
        self.outage_seconds = float(outage_seconds)
        self._expected = expected_identity
        self._pinned: str | None = None
        self._passages: int | None = None
        self._reverify = False
        self._lock = threading.Lock()
        self._local = threading.local()
        self._max_workers = max_workers
        self._pool: ThreadPoolExecutor | None = None

    @property
    def identity(self) -> str | None:
        return self._pinned

    @property
    def passages(self) -> int | None:
        return self._passages

    def _drop(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
        self._local.conn = None

    def _request(
        self, method: str, path: str, payload: Any = None, *, deadline: float | None = None
    ) -> tuple[int, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"} if body is not None else {}
        if deadline is None:
            deadline = time.monotonic() + self.timeout_seconds
        last: BaseException | None = None
        attempts = 0
        for attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            attempts = attempt + 1
            try:
                conn = getattr(self._local, "conn", None)
                if conn is None:
                    conn = http.client.HTTPConnection(self._host, self._port, timeout=remaining)
                    self._local.conn = conn
                else:
                    conn.timeout = remaining
                    if conn.sock is not None:
                        conn.sock.settimeout(remaining)
                conn.request(method, path, body=body, headers=headers)
                response = conn.getresponse()
                status, raw = response.status, response.read()
            except (OSError, http.client.HTTPException) as exc:
                self._drop()
                self._reverify = True
                last = exc
            else:
                if status >= 500:
                    last = RetrievalServiceError(f"HTTP {status}: {raw[:200]!r}")
                else:
                    try:
                        return status, json.loads(raw)
                    except ValueError as exc:
                        self._drop()
                        raise RetrievalIdentityError(
                            f"non-JSON reply from {self.base_url}{path}"
                        ) from exc
            pause = min(0.25 * 2**attempt, max(0.0, deadline - time.monotonic()))
            if attempt < self.retries and pause > 0:
                time.sleep(pause)
        error = _ServiceUnreachable if isinstance(last, ConnectionError) else RetrievalServiceError
        raise error(f"{method} {self.base_url}{path} failed after {attempts} attempt(s): {last!r}")

    def health(self, *, deadline: float | None = None) -> dict[str, Any]:
        status, value = self._request("GET", "/health", deadline=deadline)
        if status != 200 or not isinstance(value, dict):
            raise RetrievalIdentityError(f"{self.base_url}/health answered HTTP {status}")
        return value

    def _pin(self, value: dict[str, Any]) -> str:
        passages = value.get("passages")
        if value.get("ok") is not True or type(passages) is not int or passages < 1:
            raise RetrievalIdentityError("health reply lacks the service's readiness fields")
        identity = service_identity(passages, self.top_k)
        with self._lock:
            pinned = self._pinned or self._expected
            if pinned is not None and identity != pinned:
                raise RetrievalIdentityError(
                    f"retrieval service identity {identity} differs from pinned {pinned}"
                    f" ({passages} passages, top_k {self.top_k})"
                )
            self._pinned, self._passages = identity, passages
            self._reverify = False
        return identity

    def wait_ready(
        self, timeout_seconds: float = 1800.0, poll_seconds: float = 5.0
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                value = self.health()
            except RetrievalIdentityError:
                raise
            except RetrievalServiceError:
                if time.monotonic() + poll_seconds >= deadline:
                    raise RetrievalServiceError(
                        f"retrieval service at {self.base_url} not ready after {timeout_seconds} s"
                    ) from None
                time.sleep(poll_seconds)
                continue
            self._pin(value)
            return value

    def search(self, query: str) -> SearchResult:
        if self._pinned is None:
            raise RuntimeError("call wait_ready() before searching; the identity is not pinned")
        if not isinstance(query, str) or not query.strip():
            raise RetrievalQueryError("search query is empty")
        began = time.monotonic()
        outage_end: float | None = None
        pause = _OUTAGE_PAUSE_SECONDS[0]
        while True:
            try:
                return self._search(query, began)
            except _ServiceUnreachable:
                now = time.monotonic()
                outage_end = now + self.outage_seconds if outage_end is None else outage_end
                if now >= outage_end:
                    raise
                time.sleep(min(pause, outage_end - now))
                pause = min(2 * pause, _OUTAGE_PAUSE_SECONDS[1])

    def _search(self, query: str, began: float) -> SearchResult:
        deadline = time.monotonic() + self.timeout_seconds
        if self._reverify:
            self._pin(self.health(deadline=deadline))
        body: dict[str, Any] = {"queries": [query], "topk": self.top_k}
        status, value = self._request("POST", "/retrieve", body, deadline=deadline)
        if status == 400:
            error = value.get("error") if isinstance(value, dict) else None
            raise RetrievalQueryError(str(error or "query rejected"))
        if status != 200:
            raise RetrievalIdentityError(f"{self.base_url}/retrieve answered HTTP {status}")
        hits = self._hits(value)
        if self._reverify:
            self._pin(self.health(deadline=deadline))
        return SearchResult(query, self.top_k, hits, self._pinned, time.monotonic() - began)

    def _hits(self, value: Any) -> tuple[SearchHit, ...]:
        try:
            rows = value["result"]
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
                raise TypeError("one result list per query")
            hits = []
            for rank, item in enumerate(rows[0], 1):
                if not isinstance(item, dict):
                    raise TypeError("hit must be an object")
                doc_id, title, text, score = item["id"], item["title"], item["text"], item["score"]
                if type(doc_id) is int:
                    doc_id = str(doc_id)
                if not all(type(field) is str for field in (doc_id, title, text)):
                    raise TypeError("hit id, title and text must be text")
                if isinstance(score, bool) or not isinstance(score, int | float):
                    raise TypeError("hit score must be a number")
                if not math.isfinite(score):
                    raise ValueError("hit score must be finite")
                hits.append(SearchHit(rank, doc_id, title, text, float(score)))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RetrievalIdentityError(f"malformed retrieval reply: {exc}") from exc
        expected = min(self.top_k, self._passages or self.top_k)
        if len(hits) != expected or any(a.score < b.score for a, b in zip(hits, hits[1:])):
            raise RetrievalIdentityError("retrieval reply is incomplete or not ranked")
        return tuple(hits)

    async def in_pool(self, function: Callable[[], T]) -> T:
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self._max_workers, thread_name_prefix="retrieval"
                )
            pool = self._pool
        return await asyncio.get_running_loop().run_in_executor(pool, function)

    async def asearch(self, query: str) -> SearchResult:
        return await self.in_pool(lambda: self.search(query))

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False)


__all__ = [
    "OUTAGE_SECONDS",
    "RETRIEVAL_IDENTITY_ENV",
    "RETRIEVAL_URL_ENV",
    "SERVICE_FORMAT",
    "SERVICE_KIND",
    "WIKI18_PASSAGES",
    "RetrievalClient",
    "RetrievalIdentityError",
    "RetrievalQueryError",
    "RetrievalServiceError",
    "SearchHit",
    "SearchResult",
    "expected_identity_for",
    "format_search_r1",
    "service_identity",
]
