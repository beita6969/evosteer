#!/usr/bin/env python3
"""Build a fixed SQLite FTS corpus from public, task-independent JSONL pages."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from skillev.evaluation.trivia_retrieval.config import RetrievalCorpusReceipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to mutate an existing frozen corpus")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.output)
    document_ids: set[str] = set()
    passage_count = 0
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE passages USING fts5("
            "passage_id UNINDEXED, document_id UNINDEXED, title, text, "
            "tokenize='unicode61')"
        )
        connection.execute(
            "CREATE TABLE passage_rowids("
            "passage_id TEXT PRIMARY KEY, passage_rowid INTEGER NOT NULL UNIQUE) WITHOUT ROWID"
        )
        with args.documents.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"document line {line_number} is not an object")
                required = ("passage_id", "document_id", "title", "text")
                if any(
                    not isinstance(raw.get(key), str) or not raw[key].strip() for key in required
                ):
                    raise ValueError(f"document line {line_number} is incomplete")
                cursor = connection.execute(
                    "INSERT INTO passages(passage_id, document_id, title, text) "
                    "VALUES (?, ?, ?, ?)",
                    tuple(raw[key] for key in required),
                )
                connection.execute(
                    "INSERT INTO passage_rowids(passage_id, passage_rowid) VALUES (?, ?)",
                    (raw["passage_id"], cursor.lastrowid),
                )
                document_ids.add(str(raw["document_id"]))
                passage_count += 1
        connection.commit()
    finally:
        connection.close()
    receipt = RetrievalCorpusReceipt(
        corpus_id=args.corpus_id,
        source_revision=args.source_revision,
        document_count=len(document_ids),
        passage_count=passage_count,
        tokenizer="sqlite-fts5-unicode61",
        chunk_characters=2048,
        overlap_characters=256,
        retrieval_backend="sqlite-fts5-rank-with-indexed-read",
    )
    args.receipt.write_text(
        json.dumps(
            {name: getattr(receipt, name) for name in receipt.__dataclass_fields__}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
