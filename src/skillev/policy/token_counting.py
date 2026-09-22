"""Policy-owned tokenizer construction for exact model chat-token counts."""

from __future__ import annotations

from typing import Any


def load_qwen_chat_tokenizer(path: str) -> Any:
    if not path.strip():
        raise ValueError("tokenizer path must be non-empty")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


__all__ = ["load_qwen_chat_tokenizer"]
