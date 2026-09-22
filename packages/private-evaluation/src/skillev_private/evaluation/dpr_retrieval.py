"""Optional CPU DPR retrieval: a supervised retriever, never another answerer.

Uses the publisher's DPRQuestionEncoder.pooler_output and the published HF
wiki_dpr compressed inner-product index (nprobe=64). No query expansion,
reranking, target labels, reference answers or generated candidates enter here.

Index row i maps to original TSV passage id i+1. The upstream builder explicitly
retains source order and omits the last 24 passages without embeddings:
https://github.com/huggingface/datasets/blob/1.18.4/datasets/wiki_dpr/wiki_dpr.py
The parquet successor preserves this ordering; see facebook/wiki_dpr/wiki_dpr.py.
"""

from __future__ import annotations

import math
import sqlite3
import time
from contextlib import closing
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

from skillev.evaluation.corpus_search import DPR_MODEL, DPR_QUERY_POLICY, DPR_RANKING

INDEXED_PASSAGES = 21_015_300
MAX_QUERY_TOKENS = 256


@lru_cache(maxsize=1)
def _cpu_retriever(model_path: Path, index_path: Path) -> tuple[Any, Any, Any, str]:
    import torch
    from transformers import AutoTokenizer, DPRQuestionEncoder

    faiss = import_module("faiss")
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model: Any = DPRQuestionEncoder.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model.to("cpu")
    model.eval()
    index = faiss.read_index(str(index_path))
    # The observed 24-record corpus/index difference must not silently shift
    # passage identities or let a different index pretend to be this condition.
    if (
        not isinstance(index, faiss.IndexIVFPQ)
        or index.ntotal != INDEXED_PASSAGES
        or index.d != 768
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
        or index.nlist != 4096
        or index.pq.M != 128
        or index.pq.nbits != 8
    ):
        raise ValueError("expected the published NQ wiki_dpr compressed index")
    index.nprobe = 64
    faiss.downcast_index(index.quantizer).hnsw.efSearch = 128
    return tokenizer, model, index, faiss.__version__


def indexed_passages(
    path: Path, indices: list[int], scores: list[float]
) -> list[dict[str, object]]:
    """Literal source lookup in index rank order, without reading any task."""
    if len(indices) != len(scores) or any(
        index < 0 or index >= INDEXED_PASSAGES or not math.isfinite(score)
        for index, score in zip(indices, scores, strict=True)
    ):
        raise ValueError("incomplete dense retrieval result")
    rows: list[dict[str, object]] = []
    with closing(
        sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro&immutable=1", uri=True)
    ) as db:
        for rank, index in enumerate(indices, 1):
            pid = index + 1
            row = db.execute("SELECT title,text FROM passages WHERE rowid=?", (pid,)).fetchone()
            if row is None:
                raise ValueError("indexed passage is absent from the frozen public corpus")
            rows.append({"passage_id": str(pid), "title": row[0], "text": row[1], "rank": rank})
    return rows


def search_dense_corpus(
    path: Path,
    query: str,
    *,
    limit: int,
    model_path: Path,
    index_path: Path,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    import torch

    started = time.monotonic()
    tokenizer, model, index, version = _cpu_retriever(model_path, index_path)
    encoded = tokenizer(
        query.strip(),
        padding="max_length",
        truncation=True,
        max_length=MAX_QUERY_TOKENS,
        return_tensors="pt",
    )
    with torch.inference_mode():
        vectors = model(**encoded).pooler_output.float().cpu().numpy()
    # DPR uses unnormalised CLS embeddings and inner product, not cosine.
    scores, indices = index.search(vectors, limit)
    rows = indexed_passages(path, indices[0].tolist(), scores[0].tolist())
    return rows, {
        "ranking_policy": DPR_RANKING,
        "query_policy": DPR_QUERY_POLICY,
        "model_id": DPR_MODEL,
        "training_supervision": "NQ train",
        "device": "cpu",
        "faiss_version": version,
        "indexed_passages": index.ntotal,
        "nprobe": 64,
        "maximum_query_tokens": MAX_QUERY_TOKENS,
        "input_tokens": int(encoded["attention_mask"].sum().item()),
        "forward_batches": 1,
        "search_seconds": time.monotonic() - started,
        "returned_ids": [row["passage_id"] for row in rows],
        "inner_product_scores": scores[0].tolist(),
    }
