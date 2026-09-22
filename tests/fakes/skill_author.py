"""Test-only scripted implementation of the sealed skill-author protocol."""

from __future__ import annotations

from collections.abc import Iterable

from skillev.evolution.authoring import (
    AuthoringFailedError,
    AuthoringRequest,
    AuthoringResult,
)
from skillev.policy import AuthoringTokenizerProtocol


class ScriptedSkillAuthor:
    def __init__(
        self,
        tokenizer: AuthoringTokenizerProtocol,
        script: Iterable[AuthoringResult | Exception],
    ) -> None:
        self._tokenizer = tokenizer
        self._script = list(script)
        self._requests: list[AuthoringRequest] = []

    @property
    def tokenizer(self) -> AuthoringTokenizerProtocol:
        return self._tokenizer

    @property
    def requests(self) -> tuple[AuthoringRequest, ...]:
        return tuple(self._requests)

    def author(self, request: AuthoringRequest) -> AuthoringResult:
        self._requests.append(request)
        if not self._script:
            raise AuthoringFailedError("scripted author is exhausted")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


__all__ = ["ScriptedSkillAuthor"]
