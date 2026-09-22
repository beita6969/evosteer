"""Private path binding for the public, task-independent TriviaQA corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from skillev.evaluation.trivia_retrieval.config import RetrievalCorpusReceipt


@dataclass(frozen=True, slots=True)
class PrivatePublicCorpusBinding:
    database_path: Path
    receipt: RetrievalCorpusReceipt

    def __post_init__(self) -> None:
        if not self.database_path.is_absolute() or not self.database_path.is_file():
            raise ValueError("Trivia public corpus database must be an absolute file")


def load_private_public_corpus(
    database_path: Path, receipt_path: Path
) -> PrivatePublicCorpusBinding:
    raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Trivia corpus receipt must be an object")
    return PrivatePublicCorpusBinding(database_path, RetrievalCorpusReceipt(**raw))


__all__ = ["PrivatePublicCorpusBinding", "load_private_public_corpus"]
