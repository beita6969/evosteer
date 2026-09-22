#!/usr/bin/env python3
"""Build the private BM25 runtime assets from the official MedRAG textbook chunks."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_SNIPPETS = 125_847
SOURCE_REVISION = "9c72838920a1323ffa867467d3f7aa7b36b0f994"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    source_files = sorted(arguments.source.glob("*.jsonl"))
    if len(source_files) != 18:
        raise RuntimeError("official MedRAG textbook source must contain 18 JSONL files")
    arguments.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    corpus_temporary = arguments.output / "all_chunks.jsonl.tmp"
    index_temporary = arguments.output / "bm25_index.pkl.tmp"
    inverted: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    document_frequency: Counter[str] = Counter()
    document_lengths: list[int] = []
    count = 0
    with corpus_temporary.open("w", encoding="utf-8") as output:
        for source in source_files:
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    contents = record.get("contents")
                    if not isinstance(contents, str) or not contents.strip():
                        raise RuntimeError("official MedRAG snippet has no contents")
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    term_frequency = Counter(re.findall(r"\b\w+\b", contents.lower()))
                    document_lengths.append(sum(term_frequency.values()))
                    for term, frequency in term_frequency.items():
                        inverted[term].append((count, frequency))
                    document_frequency.update(term_frequency)
                    count += 1
        output.flush()
        os.fsync(output.fileno())
    if count != EXPECTED_SNIPPETS:
        raise RuntimeError("official MedRAG textbook snippet count differs")
    index = {
        "avg_dl": sum(document_lengths) / count,
        "doc_lens": document_lengths,
        "idf": {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        },
        "inverted_index": dict(inverted),
    }
    with index_temporary.open("wb") as output:
        pickle.dump(index, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.flush()
        os.fsync(output.fileno())
    os.replace(corpus_temporary, arguments.output / "all_chunks.jsonl")
    os.replace(index_temporary, arguments.output / "bm25_index.pkl")
    for output in (arguments.output / "all_chunks.jsonl", arguments.output / "bm25_index.pkl"):
        output.chmod(0o600)
    revision = arguments.output / ".source_revision"
    revision.write_text(SOURCE_REVISION + "\n", encoding="utf-8")
    revision.chmod(0o600)
    print(json.dumps({"snippets": count, "status": "ready"}, sort_keys=True))


if __name__ == "__main__":
    main()
