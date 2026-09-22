"""Optional live-artifact verification for formal published-attempt audits.

Formal publication records path-free artifact manifests.  An offline audit can
recompute its scientific timeline from those records alone, but an operator
who has the pinned local artifacts may additionally measure them and require
an exact match.  This module deliberately accepts measured identities rather
than paths so the public audit surface never serializes local deployment
locations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from skillev.experiments import ImplementationBuildIdentity
from skillev.policy import BaseModelArtifactIdentity, TokenizerArtifactIdentity


@runtime_checkable
class FormalArtifactResolver(Protocol):
    """Measure the three executable artifacts named by a formal identity."""

    def resolve_base_model(
        self, expected: BaseModelArtifactIdentity
    ) -> BaseModelArtifactIdentity: ...

    def resolve_tokenizer(
        self, expected: TokenizerArtifactIdentity
    ) -> TokenizerArtifactIdentity: ...

    def resolve_implementation_build(
        self, expected: ImplementationBuildIdentity
    ) -> ImplementationBuildIdentity: ...


def require_exact_formal_artifacts(
    resolver: FormalArtifactResolver,
    *,
    base_model: BaseModelArtifactIdentity,
    tokenizer: TokenizerArtifactIdentity,
    implementation_build: ImplementationBuildIdentity,
) -> None:
    """Reject a live artifact set that differs from a formal freeze.

    A resolver is intentionally responsible for measuring the local files and
    runtime.  Returning the expected identity without measuring it would make
    the resolver itself unsound; this boundary keeps filesystem paths and
    heavyweight loaders out of the generic audit package.
    """

    actual_base_model = resolver.resolve_base_model(base_model)
    if not isinstance(actual_base_model, BaseModelArtifactIdentity):
        raise TypeError("formal artifact resolver returned an invalid base model identity")
    if actual_base_model != base_model:
        raise ValueError("resolved base model differs from the formal artifact identity")

    actual_tokenizer = resolver.resolve_tokenizer(tokenizer)
    if not isinstance(actual_tokenizer, TokenizerArtifactIdentity):
        raise TypeError("formal artifact resolver returned an invalid tokenizer identity")
    if actual_tokenizer != tokenizer:
        raise ValueError("resolved tokenizer differs from the formal artifact identity")

    actual_build = resolver.resolve_implementation_build(implementation_build)
    if not isinstance(actual_build, ImplementationBuildIdentity):
        raise TypeError("formal artifact resolver returned an invalid implementation identity")
    if actual_build != implementation_build:
        raise ValueError("resolved implementation build differs from formal identity")


__all__ = ["FormalArtifactResolver", "require_exact_formal_artifacts"]
