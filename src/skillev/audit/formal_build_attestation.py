"""Verify the pre-model implementation-build event in formal source logs."""

from __future__ import annotations

from collections.abc import Sequence

from skillev.experiments import FormalImplementationBuildAttestation, PublishedAttemptIdentity
from skillev.runtime import EventEnvelope, EventType


def require_formal_implementation_build_attestation(
    envelopes: Sequence[EventEnvelope],
    *,
    identity: PublishedAttemptIdentity,
    attempt_id: str,
) -> FormalImplementationBuildAttestation | None:
    """Require one first source event that proves the child measured its build.

    Formal inputs have previously named an implementation build that the child
    did not itself measure.  The explicit first event closes that concrete
    provenance gap; non-formal correctness fixtures intentionally have none.
    """

    if identity.formal_execution is None:
        return None
    matches = tuple(
        (index, event)
        for index, event in enumerate(envelopes)
        if event.event_type is EventType.FORMAL_IMPLEMENTATION_BUILD_ATTESTED
    )
    if len(matches) != 1:
        raise ValueError("formal source requires exactly one implementation build attestation")
    index, event = matches[0]
    if index != 0 or event.producer_id != "skillev-formal-build" or event.producer_seq != 1:
        raise ValueError("formal implementation build attestation must precede all other events")
    attestation = FormalImplementationBuildAttestation.from_value(event.payload)
    if (
        attestation.attempt_id != attempt_id
        or attestation.builder_kind is not identity.builder_kind
        or attestation.exact_input_sha256 != identity.exact_input_sha256
        or attestation.public_identity_content_hash != identity.content_hash
        or attestation.formal_execution_content_hash != identity.formal_execution.content_hash
        or attestation.expected_implementation_build
        != identity.formal_execution.implementation_build
        or attestation.measured_implementation_build
        != identity.formal_execution.implementation_build
    ):
        raise ValueError("formal implementation build attestation differs from public identity")
    return attestation


__all__ = ["require_formal_implementation_build_attestation"]
