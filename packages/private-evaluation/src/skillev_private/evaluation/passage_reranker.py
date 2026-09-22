"""Optional frozen CPU passage ranking, not an additional answering agent.

Adapted from the model publisher's Transformers inference example:
https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
Only the owner's literal query (or the declared original public question) and
public corpus title/text enter the encoder. No reference answers, private task
metadata, generated candidates or scorer are accepted.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from skillev.evaluation.corpus_search import (
    MINILM_RANKING,
    OWNER_QUERY,
    PUBLIC_QUESTION_QUERY,
    RERANK_CANDIDATE_LIMITS,
    RERANKER_MODEL,
)

from .wikipedia_corpus import search_corpus


@lru_cache(maxsize=1)
def _cpu_encoder(model_path: Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Each of the existing four corpus worker processes owns one small encoder.
    # Never create a CUDA context, download weights at evaluation time or run
    # repository-supplied Python. The frozen local snapshot is in runtime config.
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True
    ).to("cpu")
    model.eval()
    return tokenizer, model


def order_passages(
    passages: list[dict[str, object]], scores: list[float], *, limit: int
) -> list[dict[str, object]]:
    """Descending relevance; stable BM25 order for ties, original text unchanged."""
    if len(scores) != len(passages) or not all(math.isfinite(value) for value in scores):
        raise ValueError("incomplete or non-finite passage relevance scores")
    order = sorted(range(len(passages)), key=lambda index: -scores[index])[:limit]
    return [
        {**passages[index], "bm25_rank": passages[index]["rank"], "rank": rank}
        for rank, index in enumerate(order, 1)
    ]


def search_ranked_corpus(
    path: Path,
    query: str,
    *,
    limit: int,
    query_policy: str,
    model_path: Path,
    ranking_query: str | None = None,
    ranking_policy: str = MINILM_RANKING,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    import torch

    started = time.monotonic()
    candidate_limit = RERANK_CANDIDATE_LIMITS[ranking_policy]
    passages = search_corpus(path, query, limit=candidate_limit, query_policy=query_policy)
    relevance_query = query if ranking_query is None else ranking_query
    searched = time.monotonic()
    scores: list[float] = []
    input_tokens = batches = 0
    if passages:
        tokenizer, model = _cpu_encoder(model_path)
        with torch.inference_mode():
            for start in range(0, len(passages), 32):
                batch = passages[start : start + 32]
                encoded = tokenizer(
                    [relevance_query] * len(batch),
                    [f"{item['title']}\n{item['text']}" for item in batch],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                scores.extend(model(**encoded).logits.reshape(-1).tolist())
                input_tokens += int(encoded["attention_mask"].sum().item())
                batches += 1
    ranked = order_passages(passages, scores, limit=limit)
    # Private diagnostic and cost ledger, not truth labels for the owner.
    diagnostics = {
        "ranking_policy": ranking_policy,
        "candidate_limit": candidate_limit,
        "ranking_query_source": OWNER_QUERY if ranking_query is None else PUBLIC_QUESTION_QUERY,
        "ranking_query": relevance_query,
        "model_id": RERANKER_MODEL,
        "device": "cpu",
        "candidate_pairs": len(passages),
        "input_tokens": input_tokens,
        "forward_batches": batches,
        "maximum_pair_tokens": 512,
        "search_seconds": searched - started,
        "rerank_seconds": time.monotonic() - searched,
        "candidates": [
            {"passage_id": passage["passage_id"], "bm25_rank": passage["rank"], "logit": score}
            for passage, score in zip(passages, scores, strict=True)
        ],
        "returned_ids": [row["passage_id"] for row in ranked],
    }
    return ranked, diagnostics
