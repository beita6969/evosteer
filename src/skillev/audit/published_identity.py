"""Read the sole answer-free algorithm identity admitted by a publication."""

from __future__ import annotations

import json

from skillev.experiments.attempt_identity import PublishedAttemptIdentity
from skillev.runtime import PublishedSuccessfulAttemptBundle


def read_published_attempt_identity(
    bundle: PublishedSuccessfulAttemptBundle,
) -> PublishedAttemptIdentity:
    exact = PublishedSuccessfulAttemptBundle.open_exact(bundle.directory)
    identity = PublishedAttemptIdentity.from_value(
        json.loads(exact.public_identity_path.read_text(encoding="utf-8"))
    )
    if identity.content_hash != exact.public_identity_content_hash:
        raise ValueError("published identity semantic hash differs from manifest")
    if identity.builder_kind is not exact.builder_kind:
        raise ValueError("published identity builder differs from manifest")
    if identity.exact_input_sha256 != exact.exact_input_sha256:
        raise ValueError("published identity exact input differs from manifest")
    return identity


__all__ = ["read_published_attempt_identity"]
