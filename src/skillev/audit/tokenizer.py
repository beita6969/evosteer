"""Exact tokenizer hydration boundary for offline authoring validation."""

from __future__ import annotations

from typing import Protocol

from skillev.policy import AuthoringTokenizerProtocol, PublicTokenizerIdentity


class ResolvedAuditTokenizer(AuthoringTokenizerProtocol, Protocol):
    @property
    def tokenizer_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def content_hash(self) -> str: ...


class TokenizerArtifactResolver(Protocol):
    """Resolve only the exact public tokenizer artifact named by identity."""

    def resolve(self, identity: PublicTokenizerIdentity) -> ResolvedAuditTokenizer: ...


def require_exact_tokenizer(
    resolver: TokenizerArtifactResolver,
    identity: PublicTokenizerIdentity,
) -> ResolvedAuditTokenizer:
    tokenizer = resolver.resolve(identity)
    if (
        tokenizer.tokenizer_id != identity.tokenizer_id
        or tokenizer.revision != identity.revision
        or tokenizer.content_hash != identity.content_hash
    ):
        raise ValueError("resolved tokenizer differs from published identity")
    return tokenizer


__all__ = [
    "ResolvedAuditTokenizer",
    "TokenizerArtifactResolver",
    "require_exact_tokenizer",
]
