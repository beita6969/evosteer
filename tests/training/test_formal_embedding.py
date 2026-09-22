from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.skills import workspace


def test_formal_embedding_load_failure_is_not_silently_downgraded(monkeypatch) -> None:
    class BrokenSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            raise OSError("private path omitted")

    workspace._EMBEDDING_MODEL_CACHE.clear()
    monkeypatch.setenv("SKILLEV_FORMAL_RUNTIME", "1")
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=BrokenSentenceTransformer),
    )

    with pytest.raises(RuntimeError):
        workspace._get_embedding_model("configured-private-snapshot")
