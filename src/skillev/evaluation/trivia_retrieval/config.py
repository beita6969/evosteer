"""Typed identities for the frozen public TriviaQA retrieval corpus."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PublicRetrievalDocument:
    document_id: str
    title: str
    passage_id: str
    text: str
    source_url: str | None

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.document_id, self.title, self.passage_id, self.text)
        ):
            raise ValueError("public retrieval document fields must be non-empty")


@dataclass(frozen=True, slots=True)
class RetrievalCorpusReceipt:
    corpus_id: str
    source_revision: str
    document_count: int
    passage_count: int
    tokenizer: str
    chunk_characters: int
    overlap_characters: int
    retrieval_backend: str

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.corpus_id,
                self.source_revision,
                self.tokenizer,
                self.retrieval_backend,
            )
        ):
            raise ValueError("retrieval receipt text fields must be non-empty")
        if self.document_count <= 0 or self.passage_count <= 0:
            raise ValueError("retrieval corpus must contain documents and passages")
        if self.chunk_characters <= 0 or not 0 <= self.overlap_characters < self.chunk_characters:
            raise ValueError("retrieval chunk geometry is invalid")


@dataclass(frozen=True, slots=True)
class RetrievalLimits:
    maximum_searches: int = 2
    maximum_reads: int = 2
    hits_per_search: int = 8
    maximum_query_characters: int = 512

    def __post_init__(self) -> None:
        if (
            min(
                self.maximum_searches,
                self.maximum_reads,
                self.hits_per_search,
                self.maximum_query_characters,
            )
            <= 0
        ):
            raise ValueError("retrieval limits must be positive")
