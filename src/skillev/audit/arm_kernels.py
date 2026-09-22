"""Frozen arm-kernel selection from the published identity only."""

from __future__ import annotations

from skillev.runtime import PublishedSuccessfulAttemptBundle

from .complete_method import AuditResources, CompleteMethodAuditResult, audit_full_shaped_attempt


def audit_full_shaped_arm(
    bundle: PublishedSuccessfulAttemptBundle,
    *,
    resources: AuditResources,
) -> CompleteMethodAuditResult:
    return audit_full_shaped_attempt(bundle, resources=resources)


__all__ = ["audit_full_shaped_arm"]
