"""Fixed sparse/dense rank fusion of public passages, without an answer reader.

RRF follows the published rank-sum method used by Pyserini (constant 60):
https://github.com/castorini/pyserini/blob/master/pyserini/fusion/_base.py
This is a thin implementation over existing passage adapters, not Pyserini's
model-loading stack. Only a literal public query and frozen corpus enter here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from skillev.evaluation.corpus_search import HYBRID_RANKING, SOFT_CONTEXT_QUERY_POLICY

from .dpr_retrieval import search_dense_corpus
from .wikipedia_corpus import search_corpus

BRANCH_DEPTH = 100
RRF_CONSTANT = 60


def fuse_passages(
    dense: list[dict[str, object]], lexical: list[dict[str, object]], *, limit: int
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    """Sum reciprocal ranks by source ID; stable dense-first encounter tie order.

    No text concatenation, query expansion or answer-dependent selection. The
    public text is the unchanged canonical passage, not a synthesized snippet.
    """
    passages: dict[str, dict[str, object]] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, branch in (("dense", dense), ("lexical", lexical)):
        for rank, passage in enumerate(branch, 1):
            pid = str(passage["passage_id"])
            passages.setdefault(pid, passage)
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_CONSTANT + rank)
            ranks.setdefault(pid, {})[name] = rank
    order = sorted(passages, key=lambda pid: -scores[pid])
    return (
        [{**passages[pid], "rank": rank} for rank, pid in enumerate(order[:limit], 1)],
        {
            "rrf_constant": RRF_CONSTANT,
            "dense_candidates": len(dense),
            "lexical_candidates": len(lexical),
            "union_candidates": len(passages),
            "candidates": [
                {"passage_id": pid, "branch_ranks": ranks[pid], "rrf_score": scores[pid]}
                for pid in order
            ],
        },
    )


def search_hybrid_corpus(
    path: Path, query: str, *, limit: int, model_path: Path, index_path: Path
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    dense, dense_cost = search_dense_corpus(
        path, query, limit=BRANCH_DEPTH, model_path=model_path, index_path=index_path
    )
    started = time.monotonic()
    # A genuine empty lexical match contributes no ranks. An exception instead
    # propagates: an unavailable branch must not silently become dense-only.
    lexical = search_corpus(path, query, limit=BRANCH_DEPTH, query_policy=SOFT_CONTEXT_QUERY_POLICY)
    fused_at = time.monotonic()
    passages, fusion = fuse_passages(dense, lexical, limit=limit)
    return passages, {
        "ranking_policy": HYBRID_RANKING,
        "branch_depth": BRANCH_DEPTH,
        "lexical_query_policy": SOFT_CONTEXT_QUERY_POLICY,
        "dense_retrieval": dense_cost,
        "lexical_seconds": fused_at - started,
        "fusion_seconds": time.monotonic() - fused_at,
        **fusion,
        "returned_ids": [row["passage_id"] for row in passages],
    }
